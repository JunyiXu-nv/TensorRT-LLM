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

Two optional probes look for non-finite bytes that no stop has revealed yet:

* ``PADNAN-TAIL``: the last pages of every request that finishes without a trigger. Its
  written slots were all read by its own rows, so only its tail can hold a value nobody saw:
  rows computed and discarded (rejected drafts, the forward after it finished), which the
  next owner of the page inherits.
* ``PADNAN-POOL``: a census of every page of the last attention layer, listing pages whose
  MLA latent holds a non-finite value and who holds them (a request, a reserved sequence such
  as a CUDA-graph dummy, or nobody). It runs on a timer and at every stop.

Everything is off unless ``TRTLLM_PAD_POSTMORTEM=1``:

* ``TRTLLM_PAD_POSTMORTEM_TOKENS``: comma-separated trigger token ids (default 163839);
* ``TRTLLM_PAD_POSTMORTEM_DIR``: also write JSON reports there (one per rank and stop, and
  at most 200 tail and census reports per process); each names its Slurm job, host and pid;
* ``TRTLLM_PAD_POSTMORTEM_MAX``: stop scans per process (default 20);
* ``TRTLLM_PAD_POSTMORTEM_ALL_TAILS=1``: the tail probe;
* ``TRTLLM_PAD_POSTMORTEM_POOL_SCAN_S``: seconds between pool censuses (default 0, off).
"""

import json
import os
import socket
import time
from typing import Iterable, List, Optional

import torch

from tensorrt_llm.logger import logger

from ...nan_tripwire import NanTripwire, summarize_report
from ...utils import helix_local_len

ENV_ENABLE = "TRTLLM_PAD_POSTMORTEM"
ENV_TOKENS = "TRTLLM_PAD_POSTMORTEM_TOKENS"
ENV_DIR = "TRTLLM_PAD_POSTMORTEM_DIR"
ENV_MAX = "TRTLLM_PAD_POSTMORTEM_MAX"
ENV_ALL_TAILS = "TRTLLM_PAD_POSTMORTEM_ALL_TAILS"
ENV_POOL_SCAN_S = "TRTLLM_PAD_POSTMORTEM_POOL_SCAN_S"
DEFAULT_TRIGGER_TOKENS = (163839,)
DEFAULT_MAX_SCANS = 20
# JSON files one process writes, over all report kinds.
MAX_REPORT_FILES = 200
_clock = time.monotonic
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


def _process_origin() -> dict:
    """Which job, host and process wrote a report: several jobs can share one report dir."""
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }


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
    (fp8 only, else None) a value at the largest finite magnitude; and per slot the number of
    non-finite elements in its row.

    The count tells a non-finite hidden state, which reaches every element of the row it
    projects to, from foreign bytes: any data that is not fp8 KV, read as e4m3, holds a NaN
    pattern in about 2 bytes of 256, scattered over the row.
    """
    any_bad, latent_bad, saturated, counts = [], [], [], []
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
        counts.append(rows.sum(dim=1))
        if dtype in _FP8_MAX_BITS:
            at_max = (view & 0x7F) == _FP8_MAX_BITS[dtype]
            saturated.append(at_max.reshape(num_rows, -1).any(dim=1))
    return (
        torch.cat(any_bad),
        torch.cat(latent_bad),
        torch.cat(saturated) if saturated else None,
        torch.cat(counts),
    )


def _trigger_index(request, trigger_ids) -> Optional[int]:
    """Index in the request's token list of the first trigger among its last generated tokens."""
    num_tokens = int(request.get_num_tokens(0))
    first = max(int(request.py_prompt_len), num_tokens - TRIGGER_LOOKBACK)
    return next(
        (pos for pos in range(first, num_tokens) if request.get_token(0, pos) in trigger_ids), None
    )


def _page_rows(pages: torch.Tensor, tokens_per_block: int) -> torch.Tensor:
    """``[n, tokens_per_block, row]`` view of NHD pages."""
    if pages.shape[1] * pages.shape[3] == 1:
        return pages.reshape(pages.shape[0], tokens_per_block, -1)
    return pages.transpose(1, 2).reshape(pages.shape[0], tokens_per_block, -1)


def _latent_rows(pages: torch.Tensor, tokens_per_block: int) -> torch.Tensor:
    """Rows of NHD pages, cut to the MLA latent for MLA rows."""
    rows = _page_rows(pages, tokens_per_block)
    return rows[..., :MLA_LATENT] if rows.shape[-1] == MLA_ROW else rows


def _attention_layers(mgr):
    """``(layer_id, buffer)`` of every local attention layer, in local layer order."""
    is_state_layer = getattr(mgr, "_is_local_mamba_layer", None)
    for layer_id, local_layer in sorted(mgr.layer_offsets.items(), key=lambda kv: kv[1]):
        if is_state_layer is not None and is_state_layer(local_layer):
            continue
        buffer = mgr.get_buffers(layer_id)
        if buffer is not None:
            yield layer_id, buffer


def _request_pages(mgr, request_id, layer_id, num_pages: int) -> List[int]:
    return [
        int(p)
        for p in mgr.get_batch_cache_indices([request_id], layer_id)[0]
        if 0 <= int(p) < num_pages
    ]


def _own_page_stride(mgr, layer_id) -> int:
    """Distance between consecutive pages of ``layer_id`` in its buffer view.

    A KVCacheManagerV2 slot holds one page of every same-size buffer of a life cycle (for an
    MLA model, every attention layer), and a layer's buffer view runs over the whole pool from
    that layer's page in slot 0. Its own pages are every ``stride``-th view index, the
    indices in between are other layers' pages, and the page indices the manager hands out
    are view indices.
    """
    get_scale = getattr(mgr, "get_layer_page_index_scale", None)
    if get_scale is None:
        return 1
    return max(1, int(get_scale(layer_id)) // int(getattr(mgr, "kv_factor", 1)))


def pool_census(kv_cache_manager, max_listed: int = 16, all_layers: bool = False) -> dict:
    """Which pages of the attention KV pool hold a non-finite MLA latent value, and whose they are.

    Every page of the last attention layer is read, or of every attention layer with
    ``all_layers``. A non-finite hidden state reaches the KV of every later layer, so the last
    layer sees every page such a state wrote, at a fraction of the cost. A page is listed by
    its view index (the unit of the manager's page indices and of a stop report's
    ``last_page``) with its holder: a request (with the page's ordinal in that request's page
    list and the request's page count), a reserved sequence (CUDA-graph dummies, the guard
    page: ids >= 2**63), or nobody. With block reuse, nobody's pages include cached blocks
    that a later request with the same prefix can be handed.
    """
    mgr = kv_cache_manager
    tokens_per_block = int(mgr.tokens_per_block)
    holders = list(getattr(mgr, "kv_cache_map", {}).keys())
    attention = list(_attention_layers(mgr))
    layers: List[dict] = []
    total_pages = 0
    for layer_id, buffer in attention if all_layers else attention[-1:]:
        source = _byte_view(buffer)
        stride = _own_page_stride(mgr, layer_id)
        own = source[::stride]
        num_pages = own.shape[0]
        total_pages = max(total_pages, num_pages)
        bad_pages = []
        for start in range(0, num_pages, SCAN_CHUNK_PAGES * 2):
            rows = _latent_rows(own[start : start + SCAN_CHUNK_PAGES * 2], tokens_per_block)
            bad_pages.append(non_finite_mask(rows, buffer.dtype).flatten(1).any(dim=1))
        bad = torch.cat(bad_pages)
        count = int(bad.sum())
        if not count:
            continue
        listed = [index * stride for index in _first_indices(bad, max_listed)]
        owner_of = {}
        wanted = set(listed)
        for holder in holders:
            pages = _request_pages(mgr, holder, layer_id, buffer.shape[0])
            for ordinal, page in enumerate(pages):
                if page in wanted:
                    owner_of[page] = (int(holder), ordinal, len(pages))
        entries = []
        for page in listed:
            rows = _latent_rows(source[page : page + 1], tokens_per_block)[0]
            slots = _first_indices(non_finite_mask(rows, buffer.dtype).any(dim=1), tokens_per_block)
            row_counts = (
                non_finite_mask(
                    _page_rows(source[page : page + 1], tokens_per_block)[0], buffer.dtype
                )
                .sum(dim=1)
                .tolist()
            )
            holder = owner_of.get(page)
            entries.append(
                {
                    "page": page,
                    "slots": len(slots),
                    "first_slot": slots[0] if slots else None,
                    "first_row_nonfinite": row_counts[slots[0]] if slots else 0,
                    "max_row_nonfinite": max(row_counts),
                    "row_width": int(
                        _page_rows(source[page : page + 1], tokens_per_block).shape[-1]
                    ),
                    "holder": None if holder is None else holder[0],
                    "reserved": holder is not None and holder[0] >= 1 << 63,
                    "ordinal": None if holder is None else holder[1],
                    "holder_pages": None if holder is None else holder[2],
                }
            )
        layers.append(
            {
                "layer": int(layer_id),
                "page_stride": stride,
                "nonfinite_pages": count,
                "listed": entries,
            }
        )
    return {
        "scanned_layers": len(attention) if all_layers else min(1, len(attention)),
        "pages_per_layer": total_pages,
        "holders": len(holders),
        "layers": layers,
    }


def summarize_census(census: dict) -> str:
    """One log line: how many pages hold a NaN latent, and whose the listed ones are."""
    layers = census["layers"]
    if not layers:
        return (
            f"clean ({census['scanned_layers']} layer(s) x {census['pages_per_layer']} pages, "
            f"{census['holders']} holders)"
        )
    first = layers[0]
    held = [e for e in first["listed"] if e["holder"] is not None and not e["reserved"]]
    reserved = [e for e in first["listed"] if e["reserved"]]
    free = [e for e in first["listed"] if e["holder"] is None]
    examples = " ".join(
        f"p{e['page']}:{'free' if e['holder'] is None else e['holder']}"
        f"{'' if e['ordinal'] is None else '#%d/%d' % (e['ordinal'], e['holder_pages'])}"
        f"@{e['first_slot']}x{e['slots']}r{e['first_row_nonfinite']}/{e['row_width']}"
        for e in first["listed"][:8]
    )
    return (
        f"NaN-latent pages by layer "
        f"[{' '.join(str(e['layer']) + ':' + str(e['nonfinite_pages']) for e in layers[:12])}] | "
        f"first layer {first['layer']}: listed {len(first['listed'])} = held {len(held)}, "
        f"reserved {len(reserved)}, free {len(free)} | {examples}"
    )


def tail_scan(kv_cache_manager, request, last_pages: int = 2, all_layers: bool = False) -> dict:
    """Non-finite values in one request's last pages on this rank, with a single host sync.

    For a request that finished normally its written slots were all read by its own rows, so a
    non-finite value can only be in its tail: rows that were computed but whose outputs were
    discarded (rejected drafts, the forward after it finished), left for the next owner. Only
    the last attention layer is read unless ``all_layers``: a non-finite hidden state reaches
    every later layer's KV.
    """
    mgr = kv_cache_manager
    helix = bool(getattr(mgr, "_has_cp_helix", False))
    cp_size = int(getattr(mgr, "_helix_cp_size", 1)) if helix else 1
    cp_rank = int(getattr(mgr, "_helix_cp_rank", 0)) if helix else 0
    tokens_per_block = int(mgr.tokens_per_block)
    local_prompt_len = int(request.py_prompt_len)
    generated = int(request.get_num_tokens(0)) - local_prompt_len
    prompt_len = int(request.total_input_len_cp) if helix else local_prompt_len
    written = prompt_len + generated - 1
    local_written = (
        helix_local_len(written, tokens_per_block, cp_size, cp_rank) if helix else written
    )
    layer_ids, rows = [], []
    attention = list(_attention_layers(mgr))
    for layer_id, buffer in attention if all_layers else attention[-1:]:
        pages = _request_pages(mgr, request.py_request_id, layer_id, buffer.shape[0])
        first_page = max(0, len(pages) - last_pages)
        if not pages[first_page:]:
            continue
        slot_bad, latent_bad, _, row_counts = _slot_masks(
            _byte_view(buffer), pages[first_page:], buffer.dtype, tokens_per_block
        )
        slots = torch.arange(slot_bad.numel(), device=slot_bad.device)
        slots += first_page * tokens_per_block
        tail = slots >= local_written
        bad_tail = slot_bad & tail
        rows.append(
            torch.stack(
                [
                    (slot_bad & ~tail).sum(),
                    bad_tail.sum(),
                    (latent_bad & tail).sum(),
                    torch.where(bad_tail, slots, torch.full_like(slots, -1)).max(),
                    torch.where(bad_tail, slots, torch.full_like(slots, 1 << 40)).min(),
                    torch.where(bad_tail, row_counts, torch.zeros_like(row_counts)).max(),
                ]
            )
        )
        layer_ids.append(int(layer_id))
    values = torch.stack(rows).tolist() if rows else []
    layers = [
        {
            "layer": layer_id,
            "written_nonfinite_slots": int(v[0]),
            "tail_nonfinite_slots": int(v[1]),
            "tail_latent_nonfinite_slots": int(v[2]),
            "last_tail_nonfinite_local_slot": int(v[3]) if v[1] else None,
            "first_tail_nonfinite_local_slot": int(v[4]) if v[1] else None,
            "max_tail_row_nonfinite": int(v[5]),
        }
        for layer_id, v in zip(layer_ids, values)
        if v[0] or v[1]
    ]
    return {
        "request_id": int(request.py_request_id),
        "cp_rank": cp_rank,
        "cp_size": cp_size,
        "num_tokens": prompt_len + generated,
        "prompt_len": prompt_len,
        "local_written": local_written,
        "tokens_per_block": tokens_per_block,
        "layers": layers,
    }


def summarize_tail(report: dict) -> str:
    layers = report["layers"]
    latent = [e["layer"] for e in layers if e["tail_latent_nonfinite_slots"]]
    first = min(
        (e["first_tail_nonfinite_local_slot"] for e in layers if e["tail_nonfinite_slots"]),
        default=None,
    )
    written = [e["layer"] for e in layers if e["written_nonfinite_slots"]]
    worst = max((e["max_tail_row_nonfinite"] for e in layers), default=0)
    return (
        f"req={report['request_id']} rank={report['cp_rank']}/{report['cp_size']} "
        f"tokens={report['num_tokens']} prompt={report['prompt_len']} "
        f"local_written={report['local_written']} | tail NaN: layers "
        f"{_layer_ranges(e['layer'] for e in layers if e['tail_nonfinite_slots'])} (latent: "
        f"{_layer_ranges(latent)}), first local slot {first}, worst row {worst} non-finite | "
        f"written-slot NaN: layers "
        f"{_layer_ranges(written)}"
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

    def to_global(local_slot: int) -> int:
        return local_slot_to_global(local_slot, tokens_per_block, cp_size, cp_rank)

    layers: List[dict] = []
    for layer_id, buffer in _attention_layers(mgr):
        pages = _request_pages(mgr, request_id, layer_id, buffer.shape[0])
        entry = {"layer": int(layer_id), "pages": len(pages)}
        source = _byte_view(buffer)
        page0, _, _, page0_counts = _slot_masks(source, [0], buffer.dtype, tokens_per_block)
        entry["page0_nonfinite_slots"] = int(page0.sum())
        entry["page0_max_row_nonfinite"] = int(page0_counts.max())
        if pages:
            slot_bad, latent_bad, saturated, row_counts = _slot_masks(
                source, pages, buffer.dtype, tokens_per_block
            )
            slots = torch.arange(slot_bad.numel(), device=slot_bad.device)
            valid = slots < local_written
            valid_bad = slot_bad & valid
            tail_bad = slot_bad & ~valid
            counts = torch.stack(
                [
                    valid_bad.sum(),
                    tail_bad.sum(),
                    (latent_bad & ~valid).sum(),
                    torch.where(tail_bad, row_counts, torch.zeros_like(row_counts)).max(),
                ]
            ).tolist()
            valid_slots = _first_indices(valid_bad, REPORT_POSITIONS)
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
                    "first_valid_nonfinite_pos": to_global(valid_slots[0]) if valid_slots else None,
                    "valid_nonfinite_positions": [to_global(s) for s in valid_slots],
                    "tail_slots": max(0, slot_bad.numel() - local_written),
                    "tail_nonfinite_slots": int(counts[1]),
                    "tail_latent_nonfinite_slots": int(counts[2]),
                    "max_tail_row_nonfinite": int(counts[3]),
                    "first_tail_nonfinite_local_slot": _first_index(tail_bad),
                }
            )
            if saturated is not None:
                valid_saturated = saturated & valid
                first_saturated = _first_index(valid_saturated)
                entry["valid_saturated_slots"] = int(valid_saturated.sum())
                entry["first_valid_saturated_pos"] = (
                    None if first_saturated is None else to_global(first_saturated)
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
        + (f" | tripwire: {summarize_report(report['tripwire'])}" if "tripwire" in report else "")
    )


class PadPostmortem:
    """Decides whether a finishing request gets scanned, and reports the scan."""

    def __init__(
        self,
        enabled: bool,
        trigger_ids: Iterable[int],
        out_dir: Optional[str],
        max_scans: int,
        all_tails: bool = False,
        pool_scan_s: float = 0.0,
    ):
        self.enabled = enabled
        self.trigger_ids = frozenset(int(t) for t in trigger_ids)
        self.out_dir = out_dir
        self.max_scans = max_scans
        self.num_scans = 0
        self.all_tails = all_tails
        self.pool_scan_s = pool_scan_s
        self.num_censuses = 0
        self.num_reports = 0
        self.last_census_at: Optional[float] = None
        self.last_census: Optional[dict] = None
        self.last_tail: Optional[dict] = None

    @classmethod
    def from_env(cls) -> "PadPostmortem":
        enabled = os.environ.get(ENV_ENABLE, "").strip().lower() in ("1", "true", "yes", "on")
        if not enabled:
            return cls(False, DEFAULT_TRIGGER_TOKENS, None, DEFAULT_MAX_SCANS)
        out_dir = os.environ.get(ENV_DIR, "").strip() or None
        tokens = os.environ.get(ENV_TOKENS, "").strip()
        all_tails = os.environ.get(ENV_ALL_TAILS, "").strip().lower() in ("1", "true", "yes", "on")
        # A malformed setting must not stop the executor from starting.
        try:
            trigger_ids = (
                [int(t) for t in tokens.split(",") if t.strip()]
                if tokens
                else list(DEFAULT_TRIGGER_TOKENS)
            )
            max_scans = int(os.environ.get(ENV_MAX, str(DEFAULT_MAX_SCANS)))
            pool_scan_s = max(0.0, float(os.environ.get(ENV_POOL_SCAN_S, "0") or 0))
        except ValueError as error:
            logger.warning(f"{ENV_ENABLE}: ignoring malformed settings ({error}); using defaults")
            trigger_ids, max_scans = list(DEFAULT_TRIGGER_TOKENS), DEFAULT_MAX_SCANS
            pool_scan_s = 0.0
        logger.warning(
            f"{ENV_ENABLE}: scanning the KV of requests that stop on "
            f"{sorted(trigger_ids)} (at most {max_scans} per process)"
            f"{', reports in ' + out_dir if out_dir else ''}"
            f"{', tails of every finished request' if all_tails else ''}"
            f"{f', pool census every {pool_scan_s:g} s' if pool_scan_s else ''}"
        )
        return cls(True, trigger_ids, out_dir, max_scans, all_tails, pool_scan_s)

    def triggered(self, request) -> bool:
        return _trigger_index(request, self.trigger_ids) is not None

    def _write(self, name: str, report: dict) -> None:
        if self.out_dir and self.num_reports < MAX_REPORT_FILES:
            self.num_reports += 1
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, name), "w") as stream:
                json.dump(report, stream, indent=1)

    def _census(self, kv_cache_manager, rank: int, reason: str) -> None:
        start = _clock()
        census = pool_census(kv_cache_manager)
        self.last_census_at = _clock()
        self.num_censuses += 1
        census.update(
            rank=rank,
            reason=reason,
            seconds=round(self.last_census_at - start, 4),
            time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **_process_origin(),
        )
        self.last_census = census
        # Log every census that finds something, and a heartbeat on the first and every 30th.
        if census["layers"] or reason != "periodic" or self.num_censuses % 30 == 1:
            logger.warning(
                f"PADNAN-POOL rank={rank} n={self.num_censuses} {reason} "
                f"{census['seconds'] * 1000:.0f} ms | {summarize_census(census)}"
            )
        if census["layers"]:
            self._write(f"padnan-pool-rank{rank}-n{self.num_censuses}.json", census)

    def _tail(self, kv_cache_manager, request, rank: int) -> None:
        report = tail_scan(kv_cache_manager, request)
        if report["layers"]:
            report = tail_scan(kv_cache_manager, request, all_layers=True)
        self.last_tail = report
        if report["layers"]:
            report.update(
                rank=rank,
                time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **_process_origin(),
            )
            logger.warning(f"PADNAN-TAIL {summarize_tail(report)}")
            self._write(f"padnan-tail-req{report['request_id']}-rank{rank}.json", report)

    def maybe_scan(self, kv_cache_manager, request, *, rank: int) -> Optional[dict]:
        if not self.enabled or kv_cache_manager is None or getattr(request, "is_dummy", False):
            return None
        # A diagnostic must never take the server down: report a failed scan and move on.
        try:
            triggered = self.triggered(request)
            if self.pool_scan_s and (
                triggered
                or self.last_census_at is None
                or _clock() - self.last_census_at >= self.pool_scan_s
            ):
                self._census(kv_cache_manager, rank, "stop" if triggered else "periodic")
            if not triggered:
                if self.all_tails:
                    self._tail(kv_cache_manager, request, rank)
                return None
            if self.num_scans >= self.max_scans:
                return None
            self.num_scans += 1
            report = scan_request(kv_cache_manager, request, self.trigger_ids)
            tripwire = NanTripwire.get()
            if tripwire is not None:
                report["tripwire"] = tripwire.report(int(request.py_request_id))
            report["rank"] = rank
            report["time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            report.update(_process_origin())
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
