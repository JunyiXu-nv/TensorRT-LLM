# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Post-mortem of a request that stopped on a degenerate token (env-gated diagnostic).

A model whose logits row turns all-NaN samples the last vocab id (for Kimi-K3, its pad
token 163839), and the serving layer stops such a request. Just before its KV is freed,
this scans what the request leaves behind on every rank:

* per attention layer, which of the request's own KV slots hold non-finite values:
  the slots it has written (with the first such global positions), and the slots of its
  pages past that, i.e. its own rejected drafts and the previous owner's stale bytes;
  for an fp8 cache, also the written slots holding the largest finite magnitude, where a
  saturating cast clamps an overflow;
* physical page 0, which every page-table padding entry names;
* the request's recurrent state (KDA/Mamba ssm and conv), when the manager has one.

How to read a report: non-finite values only in a page tail were read, not made. Non-finite
values in written slots from attention layer L on, with clean tails, were made at the
earliest such position by layer L's KV write or upstream of it. The earliest row tells which:
a non-finite hidden state reaches every element of the row it projects to, while a cast
that overflows (fp8 has no infinity) leaves only the overflowing elements non-finite.

Everything is off unless ``TRTLLM_PAD_POSTMORTEM=1``:

* ``TRTLLM_PAD_POSTMORTEM_TOKENS``: comma-separated trigger token ids (default 163839);
* ``TRTLLM_PAD_POSTMORTEM_DIR``: also write one JSON report per rank and request there;
* ``TRTLLM_PAD_POSTMORTEM_MAX``: scans per process (default 20).
"""

import json
import os
import time
from typing import Iterable, List, Optional

import torch

from tensorrt_llm.logger import logger

from ...utils import helix_local_len

ENV_ENABLE = "TRTLLM_PAD_POSTMORTEM"
ENV_TOKENS = "TRTLLM_PAD_POSTMORTEM_TOKENS"
ENV_DIR = "TRTLLM_PAD_POSTMORTEM_DIR"
ENV_MAX = "TRTLLM_PAD_POSTMORTEM_MAX"
DEFAULT_TRIGGER_TOKENS = (163839,)
# How far back from the end a trigger token may sit: a speculative step accepts up to
# 1 + max_draft_len tokens, and the stop token can be any of them.
TRIGGER_LOOKBACK = 16
# MLA latent cache rows are [kv_lora_rank | rope]; only the latent part is ever a V row.
MLA_ROW = 576
MLA_LATENT = 512
# Pages gathered at a time, which bounds the scan's scratch memory on long requests.
SCAN_CHUNK_PAGES = 256
# Non-finite positions listed per layer in a report.
REPORT_POSITIONS = 8


_FP8_NAN_BITS = {torch.float8_e4m3fn: 0x7F, torch.float8_e5m2: 0x7C}
# Largest finite magnitude, where a saturating cast clamps an overflow.
_FP8_MAX_BITS = {torch.float8_e4m3fn: 0x7E, torch.float8_e5m2: 0x7B}


def _byte_view(values: torch.Tensor) -> torch.Tensor:
    """fp8 data as raw bytes (fp8 has no isnan kernel and few gather kernels); else as is."""
    return values.view(torch.uint8) if values.dtype in _FP8_NAN_BITS else values


def non_finite_mask(values: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Elementwise non-finite mask of ``values`` holding ``dtype`` data (default: its own).

    fp8 is read by bit pattern, so ``values`` may be the raw bytes of an fp8 tensor.
    """
    dtype = values.dtype if dtype is None else dtype
    if dtype in _FP8_NAN_BITS:
        bits = _FP8_NAN_BITS[dtype]
        return (_byte_view(values) & bits) == bits
    if dtype.is_floating_point:
        return ~torch.isfinite(values)
    # Packed sub-byte formats carry no NaN encoding this check could test.
    return torch.zeros(values.shape, dtype=torch.bool, device=values.device)


def local_slot_to_global(local_slot: int, tokens_per_block: int, cp_size: int, cp_rank: int) -> int:
    """Global position of a rank-local slot under Helix round-robin pages (page b on rank
    b % cp_size); the identity without Helix (cp_size 1)."""
    page, slot = divmod(local_slot, tokens_per_block)
    return (page * cp_size + cp_rank) * tokens_per_block + slot


def _layer_ranges(layers: Iterable[int]) -> str:
    """Compact "3-7,9" form of a sorted layer list."""
    layers = sorted(layers)
    if not layers:
        return "-"
    out = []
    start = prev = layers[0]
    for layer in layers[1:]:
        if layer == prev + 1:
            prev = layer
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = layer
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(out)


def _first_indices(mask: torch.Tensor, limit: int) -> List[int]:
    return torch.nonzero(mask)[:limit, 0].tolist()


def _first_index(mask: torch.Tensor) -> Optional[int]:
    first = _first_indices(mask, 1)
    return first[0] if first else None


def _slot_masks(source: torch.Tensor, pages: List[int], dtype: torch.dtype, tokens_per_block: int):
    """Per-slot masks over ``pages`` in order: any non-finite value, one in the MLA latent,
    and (fp8 only, else None) a value at the largest finite magnitude."""
    any_bad, latent_bad, saturated = [], [], []
    for start in range(0, len(pages), SCAN_CHUNK_PAGES):
        chunk = pages[start : start + SCAN_CHUNK_PAGES]
        index = torch.tensor(chunk, dtype=torch.long, device=source.device)
        # NHD layout: [pages, kv_factor, tokens_per_block, heads, dim] -> one row per slot.
        view = source.index_select(0, index).transpose(1, 2).contiguous()
        num_rows = len(chunk) * tokens_per_block
        rows = non_finite_mask(view, dtype).reshape(num_rows, -1)
        bad = rows.any(dim=1)
        any_bad.append(bad)
        latent_bad.append(rows[:, :MLA_LATENT].any(dim=1) if rows.shape[1] == MLA_ROW else bad)
        if dtype in _FP8_MAX_BITS:
            at_max = (view & 0x7F) == _FP8_MAX_BITS[dtype]
            saturated.append(at_max.reshape(num_rows, -1).any(dim=1))
    return torch.cat(any_bad), torch.cat(latent_bad), torch.cat(saturated) if saturated else None


def _trigger_index(request, trigger_ids) -> Optional[int]:
    """Index in the request's token list of the first trigger among its last generated tokens."""
    num_tokens = int(request.get_num_tokens(0))
    first = max(int(request.py_prompt_len), num_tokens - TRIGGER_LOOKBACK)
    return next(
        (pos for pos in range(first, num_tokens) if request.get_token(0, pos) in trigger_ids), None
    )


def scan_request(kv_cache_manager, request, trigger_ids: Iterable[int]) -> dict:
    """Scan one request's KV pages (every attention layer) and recurrent state on this rank."""
    mgr = kv_cache_manager
    request_id = request.py_request_id
    helix = bool(getattr(mgr, "_has_cp_helix", False))
    cp_size = int(getattr(mgr, "_helix_cp_size", 1)) if helix else 1
    cp_rank = int(getattr(mgr, "_helix_cp_rank", 0)) if helix else 0
    tokens_per_block = int(mgr.tokens_per_block)
    # A Helix gen request holds only this rank's share of the prompt, followed by every
    # generated token; global positions are the global prompt length plus the generated count.
    local_num_tokens = int(request.get_num_tokens(0))
    local_prompt_len = int(request.py_prompt_len)
    generated = local_num_tokens - local_prompt_len
    prompt_len = int(request.total_input_len_cp) if helix else local_prompt_len
    num_tokens = prompt_len + generated
    trigger_index = _trigger_index(request, set(trigger_ids))
    trigger_pos = None if trigger_index is None else prompt_len + trigger_index - local_prompt_len
    # The last sampled token has not been through a forward yet, so it has no KV.
    written = num_tokens - 1
    local_written = (
        helix_local_len(written, tokens_per_block, cp_size, cp_rank) if helix else written
    )
    is_state_layer = getattr(mgr, "_is_local_mamba_layer", None)

    layers: List[dict] = []
    for layer_id, local_layer in sorted(mgr.layer_offsets.items(), key=lambda kv: kv[1]):
        if is_state_layer is not None and is_state_layer(local_layer):
            continue
        buffer = mgr.get_buffers(layer_id)
        if buffer is None:
            continue
        pages = [
            int(p)
            for p in mgr.get_batch_cache_indices([request_id], layer_id)[0]
            if 0 <= int(p) < buffer.shape[0]
        ]
        entry = {"layer": int(layer_id), "pages": len(pages)}
        source = _byte_view(buffer)
        page0, _, _ = _slot_masks(source, [0], buffer.dtype, tokens_per_block)
        entry["page0_nonfinite_slots"] = int(page0.sum())
        if pages:
            slot_bad, latent_bad, saturated = _slot_masks(
                source, pages, buffer.dtype, tokens_per_block
            )
            slots = torch.arange(slot_bad.numel(), device=slot_bad.device)
            valid = slots < local_written
            valid_bad = slot_bad & valid
            tail_bad = slot_bad & ~valid
            counts = torch.stack(
                [valid_bad.sum(), tail_bad.sum(), (latent_bad & ~valid).sum()]
            ).tolist()
            valid_slots = _first_indices(valid_bad, REPORT_POSITIONS)
            valid_positions = [
                local_slot_to_global(slot, tokens_per_block, cp_size, cp_rank)
                for slot in valid_slots
            ]
            if valid_slots:
                page, slot_in_page = divmod(valid_slots[0], tokens_per_block)
                row = non_finite_mask(
                    source[pages[page]].transpose(0, 1)[slot_in_page], buffer.dtype
                ).reshape(-1)
                entry["first_row_nonfinite"] = int(row.sum())
                entry["row_width"] = row.numel()
                if row.numel() == MLA_ROW:
                    entry["first_row_latent_nonfinite"] = int(row[:MLA_LATENT].sum())
            entry.update(
                {
                    "last_page": pages[-1],
                    "valid_nonfinite_slots": int(counts[0]),
                    "first_valid_nonfinite_pos": valid_positions[0] if valid_positions else None,
                    "valid_nonfinite_positions": valid_positions,
                    "tail_slots": max(0, slot_bad.numel() - local_written),
                    "tail_nonfinite_slots": int(counts[1]),
                    "tail_latent_nonfinite_slots": int(counts[2]),
                    "first_tail_nonfinite_local_slot": _first_index(tail_bad),
                }
            )
            if saturated is not None:
                valid_saturated = saturated & valid
                first_saturated = _first_index(valid_saturated)
                entry["valid_saturated_slots"] = int(valid_saturated.sum())
                entry["first_valid_saturated_pos"] = (
                    None
                    if first_saturated is None
                    else local_slot_to_global(first_saturated, tokens_per_block, cp_size, cp_rank)
                )
        layers.append(entry)

    state: List[dict] = []
    slot = getattr(mgr, "_request_id_to_state_index", {}).get(request_id)
    if slot is not None:
        # Read the map directly: get_state_indices() also rewrites the dummy-request mask.
        ssm_states = getattr(mgr, "all_ssm_states", None) or []
        conv_states = getattr(mgr, "all_conv_states", None) or []
        for i, ssm in enumerate(ssm_states):
            conv = conv_states[i] if i < len(conv_states) else None
            state.append(
                {
                    "state_layer": i,
                    "ssm_nonfinite": int(non_finite_mask(ssm[slot]).sum()),
                    "conv_nonfinite": (
                        None if conv is None else int(non_finite_mask(conv[slot]).sum())
                    ),
                }
            )

    return {
        "request_id": int(request_id),
        "cp_rank": cp_rank,
        "cp_size": cp_size,
        "helix": helix,
        "num_tokens": num_tokens,
        "prompt_len": prompt_len,
        "trigger_pos": trigger_pos,
        "written": written,
        "local_num_tokens": local_num_tokens,
        "local_prompt_len": local_prompt_len,
        "local_written": local_written,
        "tokens_per_block": tokens_per_block,
        "state_slot": slot,
        "layers": layers,
        "state": state,
    }


def summarize(report: dict) -> str:
    """One log line: where non-finite values sit, by layer."""
    layers = report["layers"]
    valid = [e for e in layers if e.get("valid_nonfinite_slots")]
    tail = [e["layer"] for e in layers if e.get("tail_latent_nonfinite_slots")]
    tail_rope = [
        e["layer"]
        for e in layers
        if e.get("tail_nonfinite_slots") and not e.get("tail_latent_nonfinite_slots")
    ]
    page0 = [e["layer"] for e in layers if e.get("page0_nonfinite_slots")]
    state = [s["state_layer"] for s in report["state"] if s["ssm_nonfinite"] or s["conv_nonfinite"]]
    first = None
    if valid:
        first = min(valid, key=lambda e: (e["first_valid_nonfinite_pos"], e["layer"]))
    first_text = (
        "none"
        if first is None
        else f"pos {first['first_valid_nonfinite_pos']} (first at layer {first['layer']}, "
        f"{first['first_row_nonfinite']}/{first['row_width']} of its row)"
    )
    firsts = " ".join(f"{e['layer']}:{e['first_valid_nonfinite_pos']}" for e in valid[:12])
    saturated = [e["layer"] for e in layers if e.get("valid_saturated_slots")]
    pages = max((e["pages"] for e in layers), default=0)
    return (
        f"req={report['request_id']} rank={report['cp_rank']}/{report['cp_size']} "
        f"tokens={report['num_tokens']} prompt={report['prompt_len']} "
        f"trigger_pos={report['trigger_pos']} local_written={report['local_written']} "
        f"pages={pages} attn_layers={len(layers)} | written-slot NaN: layers "
        f"{_layer_ranges(e['layer'] for e in valid)}, earliest {first_text}, "
        f"first pos by layer [{firsts}] | "
        f"tail latent NaN: layers {_layer_ranges(tail)} (rope only: {_layer_ranges(tail_rope)}) | "
        f"page0 NaN: layers {_layer_ranges(page0)} | fp8 at max magnitude: layers "
        f"{_layer_ranges(saturated)} | state NaN (slot {report['state_slot']}): "
        f"{_layer_ranges(state)}"
    )


class PadPostmortem:
    """Decides whether a finishing request gets scanned, and reports the scan."""

    def __init__(
        self, enabled: bool, trigger_ids: Iterable[int], out_dir: Optional[str], max_scans: int
    ):
        self.enabled = enabled
        self.trigger_ids = frozenset(int(t) for t in trigger_ids)
        self.out_dir = out_dir
        self.max_scans = max_scans
        self.num_scans = 0

    @classmethod
    def from_env(cls) -> "PadPostmortem":
        enabled = os.environ.get(ENV_ENABLE, "").strip().lower() in ("1", "true", "yes", "on")
        tokens = os.environ.get(ENV_TOKENS, "").strip()
        trigger_ids = (
            [int(t) for t in tokens.split(",") if t.strip()]
            if tokens
            else list(DEFAULT_TRIGGER_TOKENS)
        )
        out_dir = os.environ.get(ENV_DIR, "").strip() or None
        max_scans = int(os.environ.get(ENV_MAX, "20"))
        if enabled:
            logger.warning(
                f"{ENV_ENABLE}: scanning the KV of requests that stop on "
                f"{sorted(trigger_ids)} (at most {max_scans} per process)"
                f"{', reports in ' + out_dir if out_dir else ''}"
            )
        return cls(enabled, trigger_ids, out_dir, max_scans)

    def triggered(self, request) -> bool:
        return _trigger_index(request, self.trigger_ids) is not None

    def maybe_scan(self, kv_cache_manager, request, *, rank: int) -> Optional[dict]:
        if (
            not self.enabled
            or kv_cache_manager is None
            or self.num_scans >= self.max_scans
            or getattr(request, "is_dummy", False)
        ):
            return None
        # A diagnostic must never take the server down: report a failed scan and move on.
        try:
            if not self.triggered(request):
                return None
            self.num_scans += 1
            report = scan_request(kv_cache_manager, request, self.trigger_ids)
            report["rank"] = rank
            report["time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            logger.warning(f"PADNAN {summarize(report)}")
            if self.out_dir:
                os.makedirs(self.out_dir, exist_ok=True)
                path = os.path.join(
                    self.out_dir, f"padnan-req{report['request_id']}-rank{rank}.json"
                )
                with open(path, "w") as stream:
                    json.dump(report, stream, indent=1)
            return report
        except (
            AssertionError,
            RuntimeError,
            ValueError,
            TypeError,
            IndexError,
            KeyError,
            AttributeError,
            OSError,
        ) as error:
            logger.warning(f"PADNAN scan of request {request.py_request_id} failed: {error!r}")
            return None
