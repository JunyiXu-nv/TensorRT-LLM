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
"""Where each token of a forward first turns non-finite (env-gated diagnostic).

Model code calls :meth:`NanTripwire.mark` at fixed points of every decoder layer. Each mark is
one small kernel that records, for every row (token) of the batch, the first mark at which the
row holds a non-finite value. It runs inside CUDA graphs: the record is a static device buffer.
The model engine calls :meth:`begin_step` with the batch layout (request ids and rows per
request) and :meth:`end_step` after launching the forward, which moves the step's record into a
ring of the last ``RING`` steps. A post-mortem then calls :meth:`report` for one request: the
first non-finite mark of each of its rows in each recent step, and which other rows of those
steps went non-finite.

Off unless ``TRTLLM_NAN_TRIPWIRE`` is set: ``all``/``1`` for every layer, or layer ranges such
as ``36-44,92``. With ``TRTLLM_NAN_TRIPWIRE_MAG=1`` the same kernels also record each row's
largest finite magnitude at every mark (first ``MAG_ROWS`` rows of a step), so a report can tell a
value that grew until it overflowed from a non-finite value that appeared in an ordinary row.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

ENV_TRIPWIRE = "TRTLLM_NAN_TRIPWIRE"
ENV_MAG = "TRTLLM_NAN_TRIPWIRE_MAG"
NONE = 2**31 - 1
STAGES = ("attn_in", "attn", "moe_in", "moe", "out")
STAGES_PER_LAYER = 8
RING = 16
# Larger batches (context chunks) are not marked; the diagnostic is for decode steps.
MAX_ROWS = 4096
# Magnitude profile: rows and layers it covers (a decode step has at most 256 rows on K3 gen).
MAG_ROWS = 256
MAX_LAYERS = 128
MAG_MARKS = MAX_LAYERS * len(STAGES)


def stage_id(layer: int, kind: int) -> int:
    return layer * STAGES_PER_LAYER + kind


def mag_index(sid: int) -> int:
    """Dense index of mark ``sid`` in the magnitude buffer, or -1 when it has no slot."""
    layer, kind = divmod(int(sid), STAGES_PER_LAYER)
    return layer * len(STAGES) + kind if layer < MAX_LAYERS and kind < len(STAGES) else -1


def mag_name(index: int) -> str:
    layer, kind = divmod(int(index), len(STAGES))
    return f"L{layer}.{STAGES[kind]}"


def stage_name(sid: int) -> str:
    if sid == NONE:
        return "-"
    layer, kind = divmod(int(sid), STAGES_PER_LAYER)
    return f"L{layer}.{STAGES[kind] if kind < len(STAGES) else kind}"


def parse_layers(spec: str) -> Optional[frozenset]:
    """``all``/``1`` -> None (every layer); ``36-44,92`` -> {36, ..., 44, 92}."""
    spec = spec.strip().lower()
    if spec in ("1", "all", "true", "on", "yes"):
        return None
    layers = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        layers.update(range(int(lo), int(hi or lo) + 1))
    return frozenset(layers)


@triton.jit(do_not_specialize=["row_stride", "n_cols", "stage", "mag_row"])
def _first_nonfinite_kernel(
    x_ptr,
    row_stride,
    n_cols,
    first_ptr,
    stage,
    mag_ptr,
    mag_row,
    NONE: tl.constexpr,
    BLOCK: tl.constexpr,
    WITH_MAG: tl.constexpr,
    MAG_ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    base = x_ptr + row.to(tl.int64) * row_stride
    count = tl.zeros((BLOCK,), dtype=tl.int32)
    peak = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        value = tl.load(base + offs, mask=offs < n_cols, other=0.0).to(tl.float32)
        magnitude = tl.abs(value)
        nonfinite = (value != value) | (magnitude == float("inf"))
        count += nonfinite.to(tl.int32)
        if WITH_MAG:
            peak = tl.maximum(peak, tl.where(nonfinite, 0.0, magnitude))
    bad = tl.sum(count, axis=0) > 0
    current = tl.load(first_ptr + row)
    tl.store(first_ptr + row, tl.where(bad & (current == NONE), stage, current))
    if WITH_MAG:
        if (mag_row >= 0) & (row < MAG_ROWS):
            tl.store(mag_ptr + mag_row * MAG_ROWS + row, tl.max(peak, axis=0))


class NanTripwire:
    """Per-row first non-finite mark of the last ``RING`` forward steps of this process."""

    _instance: Optional["NanTripwire"] = None
    _resolved = False

    def __init__(self, layers: Optional[frozenset], device: torch.device, with_mag: bool = False):
        self.layers = layers
        self.with_mag = with_mag
        self.cur = torch.full((MAX_ROWS,), NONE, dtype=torch.int32, device=device)
        self.ring = torch.full((RING, MAX_ROWS), NONE, dtype=torch.int32, device=device)
        # -1 means "no mark recorded"; a real magnitude is >= 0.
        mag_shape = (MAG_MARKS, MAG_ROWS) if with_mag else (1, 1)
        self.mag = torch.full(mag_shape, -1.0, dtype=torch.float32, device=device)
        self.ring_mag = torch.full((RING,) + mag_shape, -1.0, dtype=torch.float32, device=device)
        self.layouts: List[Optional[Tuple[int, List[int], List[int], bool]]] = [None] * RING
        self.step = 0
        self._pending: Tuple[List[int], List[int], bool] = ([], [], False)
        # Compile every variant before any CUDA graph capture can launch one.
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            probe = torch.zeros(2, 32, dtype=dtype, device=device)
            self._launch(probe, 0)
        self.cur.fill_(NONE)
        self.mag.fill_(-1.0)

    @classmethod
    def get(cls) -> Optional["NanTripwire"]:
        """The process-wide tripwire, or None when ``TRTLLM_NAN_TRIPWIRE`` is unset."""
        if cls._resolved:
            return cls._instance
        cls._resolved = True
        spec = os.environ.get(ENV_TRIPWIRE, "").strip()
        if not spec or spec.lower() in ("0", "off", "false", "no"):
            return None
        # A diagnostic must never stop the executor from starting: disable it instead.
        try:
            layers = parse_layers(spec)
            with_mag = os.environ.get(ENV_MAG, "").strip().lower() in ("1", "true", "yes", "on")
            device = torch.device("cuda", torch.cuda.current_device())
            cls._instance = cls(layers, device, with_mag)
        except (RuntimeError, ValueError, TypeError, OSError) as error:
            logger.warning(f"{ENV_TRIPWIRE}: disabled, could not set up ({error!r})")
            cls._instance = None
            return None
        logger.warning(
            f"{ENV_TRIPWIRE}: recording the first non-finite mark per token on "
            f"layers {'all' if layers is None else sorted(layers)}"
            f"{', with magnitudes' if cls._instance.with_mag else ''}"
        )
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        cls._instance = None
        cls._resolved = False

    def covers(self, layer: int) -> bool:
        return self.layers is None or layer in self.layers

    def _launch(self, x: torch.Tensor, sid: int) -> None:
        _first_nonfinite_kernel[(x.shape[0],)](
            x,
            x.stride(0),
            x.shape[1],
            self.cur,
            sid,
            self.mag,
            mag_index(sid),
            NONE=NONE,
            BLOCK=1024,
            WITH_MAG=self.with_mag,
            MAG_ROWS=MAG_ROWS,
        )

    def mark(self, x: torch.Tensor, sid: int) -> None:
        """Record rows of ``x`` ([tokens, hidden]) that hold a non-finite value at mark ``sid``."""
        if (
            x.dim() != 2
            or x.shape[0] == 0
            or x.shape[0] > MAX_ROWS
            or x.stride(1) != 1
            or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            return
        self._launch(x, sid)

    def begin_step(self, request_ids: Optional[Sequence[int]], seq_lens, is_dummy: bool) -> None:
        """The batch layout of the forward about to run: request ids and rows per request."""
        if seq_lens is None:
            lens = []
        elif isinstance(seq_lens, torch.Tensor):
            lens = [int(n) for n in seq_lens.tolist()]
        else:
            lens = [int(n) for n in seq_lens]
        rids = [int(r) for r in (request_ids or [])]
        self._pending = (rids, lens[: len(rids)], is_dummy)

    def end_step(self) -> None:
        """Move this step's record into the ring (stream-ordered after the forward)."""
        slot = self.step % RING
        self.ring[slot].copy_(self.cur)
        self.cur.fill_(NONE)
        if self.with_mag:
            self.ring_mag[slot].copy_(self.mag)
            self.mag.fill_(-1.0)
        rids, lens, is_dummy = self._pending
        self.layouts[slot] = (self.step, rids, lens, is_dummy)
        self._pending = ([], [], False)
        self.step += 1

    def report(self, request_id: int, max_others: int = 8) -> Dict:
        """The recent steps holding ``request_id``, oldest first: each of its rows' first
        non-finite mark, and the other rows of each step that went non-finite. Synchronizes
        the device."""
        ring = self.ring.cpu()
        steps = []
        for back in reversed(range(min(self.step, RING))):
            step_no = self.step - 1 - back
            layout = self.layouts[step_no % RING]
            if layout is None or layout[0] != step_no:
                continue
            _, rids, lens, is_dummy = layout
            firsts = ring[step_no % RING]
            start, mine, owner = 0, None, []
            for rid, n in zip(rids, lens):
                if rid == request_id:
                    mine = (start, n)
                owner.extend([rid] * n)
                start += n
            if mine is None:
                continue
            bad = torch.nonzero(firsts != NONE).flatten().tolist()
            others = [
                (owner[r] if r < start else None, r, stage_name(firsts[r]))
                for r in bad
                if not mine[0] <= r < mine[0] + mine[1]
            ]
            steps.append(
                {
                    "back": back,
                    "step": step_no,
                    "dummy_step": is_dummy,
                    "rows": start,
                    "request_rows": [mine[0], mine[1]],
                    "request_first": [
                        stage_name(firsts[r]) for r in range(mine[0], mine[0] + mine[1])
                    ],
                    "other_nonfinite_rows": len(others),
                    "others": others[:max_others],
                }
            )
        result = {"steps_recorded": self.step, "steps": steps}
        if self.with_mag:
            self._attach_magnitudes(result)
        return result

    def _attach_magnitudes(self, result: Dict) -> None:
        """Magnitude profile of the request's rows in its first step with a non-finite row and
        in the step before it: per row, its largest finite magnitude at every recorded mark."""
        steps = result["steps"]
        bad = [i for i, e in enumerate(steps) if any(n != "-" for n in e["request_first"])]
        if not bad:
            return
        mags = self.ring_mag.cpu()
        for i in (bad[0] - 1, bad[0]):
            if i < 0:
                continue
            entry = steps[i]
            slot = entry["step"] % RING
            start, count = entry["request_rows"]
            rows = []
            for k in range(count):
                row = start + k
                if row >= MAG_ROWS:
                    break
                values = mags[slot, :, row].tolist()
                recorded = [(j, v) for j, v in enumerate(values) if v >= 0]
                first = entry["request_first"][k]
                cut = len(recorded)
                if first != "-":
                    cut = next(
                        (n for n, (j, _) in enumerate(recorded) if mag_name(j) == first), cut
                    )
                before = recorded[:cut]
                peak = max(before, key=lambda jv: jv[1], default=None)
                rows.append(
                    {
                        "row": k,
                        "first": first,
                        "peak": None if peak is None else float(f"{peak[1]:.4g}"),
                        "peak_at": None if peak is None else mag_name(peak[0]),
                        "last_before_first": [
                            [mag_name(j), float(f"{v:.4g}")] for j, v in before[-6:]
                        ],
                        # [dense mark index (layer * 5 + stage), magnitude] for every recorded mark.
                        "all": [[j, float(f"{v:.4g}")] for j, v in recorded],
                    }
                )
            entry["magnitudes"] = rows


def summarize_report(report: Dict, max_steps: int = 3) -> str:
    """One log fragment: the request's rows' first marks in its first steps with a non-finite
    row (oldest first), after how many clean steps."""
    steps = report["steps"]
    bad = [
        i
        for i, e in enumerate(steps)
        if any(name != "-" for name in e["request_first"]) or e["other_nonfinite_rows"]
    ]
    if not bad:
        return f"clean in the last {len(steps)} steps holding the request"
    parts = [f"{bad[0]} clean steps before"]
    for entry in steps[bad[0] : bad[0] + max_steps]:
        rows = " ".join(f"r{i}:{name}" for i, name in enumerate(entry["request_first"]))
        others = (
            f" +{entry['other_nonfinite_rows']} other rows "
            f"({', '.join(f'{rid}/{row}:{name}' for rid, row, name in entry['others'][:3])})"
            if entry["other_nonfinite_rows"]
            else ""
        )
        parts.append(f"step -{entry['back']}: {rows}{others}")
        origin = _origin_row(entry)
        if origin is not None:
            before = " ".join(f"{name} {value:.3g}" for name, value in origin["last_before_first"])
            parts.append(
                f"mag r{origin['row']} before {origin['first']}: {before} "
                f"(peak {origin['peak']:.3g} at {origin['peak_at']})"
                if origin["peak"] is not None
                else f"mag r{origin['row']}: non-finite at its first recorded mark"
            )
    return " | ".join(parts)


def _origin_row(entry: Dict) -> Optional[Dict]:
    """The magnitude profile of the row whose first non-finite mark comes first in this step."""
    rows = [r for r in entry.get("magnitudes", []) if r["first"] != "-"]
    return min(rows, key=lambda r: _mark_order(r["first"])) if rows else None


def _mark_order(name: str) -> int:
    layer, _, kind = name[1:].partition(".")
    return int(layer) * len(STAGES) + STAGES.index(kind)
