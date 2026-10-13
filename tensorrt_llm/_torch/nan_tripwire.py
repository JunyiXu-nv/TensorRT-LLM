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

The five layer marks (``STAGES``) sit where every rank holds the same values. The marks in
``RANK_STAGES`` sit inside the attention and MoE blocks, before their all-reduce, where each rank
holds its own part. A rank that makes a non-finite value records the earliest of them; a rank
that only receives it records a later one. They are read per rank, in ``MARK_ORDER``.
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
# Layer marks: the same values on every rank.
STAGES = ("attn_in", "attn", "moe_in", "moe", "out")
# Per-rank marks, before the block's all-reduce:
#   mla_q      MLA query rows fed to the decode (after q_b and the k_b bmm)
#   mla_part   this rank's decode output and softmax stats, rows without local KV skipped
#   mla_recv   the Helix all-to-all entries this rank received
#   mla_comb   the combine of the received entries
#   mla_out    after the v_b bmm
#   attn_part  attention o_proj output (MLA after the gate, KDA)
#   moe_part   routed experts output
#   moe_shared shared experts output (already reduced inside the shared MLP)
RANK_STAGES = (
    "mla_q",
    "mla_part",
    "mla_recv",
    "mla_comb",
    "mla_out",
    "attn_part",
    "moe_part",
    "moe_shared",
)
ALL_STAGES = STAGES + RANK_STAGES
MARK_ORDER = (
    "attn_in",
    "mla_q",
    "mla_part",
    "mla_recv",
    "mla_comb",
    "mla_out",
    "attn_part",
    "attn",
    "moe_in",
    "moe_part",
    "moe_shared",
    "moe",
    "out",
)
STAGES_PER_LAYER = 16
RING = 16
# Larger batches (context chunks) are not marked; the diagnostic is for decode steps.
MAX_ROWS = 4096
# Magnitude profile: rows and layers it covers (a decode step has at most 256 rows on K3 gen).
# Only the layer marks have magnitudes.
MAG_ROWS = 256
MAX_LAYERS = 128
MAG_MARKS = MAX_LAYERS * len(STAGES)
# Helix snapshot: per MLA layer of a step, each row's global position, rank-local KV write slot
# (-1 = another rank owns it) and rank-local KV bound, plus each sequence's rank-local KV length
# (the decode's per-sequence K). Rows and sequences beyond these sizes are not recorded.
HELIX_FIELDS = ("pos", "slot", "bound")
HELIX_ROWS = 256
HELIX_SEQS = 64
HELIX_LAYERS = 32
# Received all-to-all entries holding a non-finite value, captured raw (process-wide, first ones
# only): which layer, step, batch row and sending rank, where the first bad element sits, how
# many there are, and the whole entry: partial_o (at most RECV_O_WORDS bf16; K3 sends 3072) and
# the softmax stats (at most RECV_STATS_WORDS fp32).
RECV_CAPTURES = 16
RECV_META = (
    "layer",
    "step",
    "row",
    "src",
    "o_first",
    "o_count",
    "s_first",
    "s_count",
    "o_words",
    "s_words",
)
RECV_O_WORDS = 4096
RECV_STATS_WORDS = 64


def kind_of(name: str) -> int:
    """Index of a mark kind (``"attn"``, ``"mla_part"``, ...) for :func:`stage_id`."""
    return ALL_STAGES.index(name)


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
    return f"L{layer}.{ALL_STAGES[kind] if kind < len(ALL_STAGES) else kind}"


def mark_order(name: str) -> int:
    """Execution order of a mark name such as ``"L15.mla_part"`` within a forward."""
    layer, _, kind = name[1:].partition(".")
    return int(layer) * len(MARK_ORDER) + MARK_ORDER.index(kind)


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


@triton.jit(
    do_not_specialize=[
        "row_stride",
        "n_cols",
        "stage",
        "mag_row",
        "row_offset",
        "has_skip",
        "allow_neg_inf",
    ]
)
def _first_nonfinite_kernel(
    x_ptr,
    row_stride,
    n_cols,
    first_ptr,
    stage,
    mag_ptr,
    mag_row,
    skip_ptr,
    row_offset,
    has_skip,
    allow_neg_inf,
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
        infinite = magnitude == float("inf")
        # Softmax maxima of rows with no key are -inf by design; such inputs pass allow_neg_inf.
        infinite = infinite & ((allow_neg_inf == 0) | (value > 0))
        nonfinite = (value != value) | infinite
        count += nonfinite.to(tl.int32)
        if WITH_MAG:
            peak = tl.maximum(peak, tl.where(nonfinite, 0.0, magnitude))
    bad = tl.sum(count, axis=0) > 0
    # skip_ptr always points at a bool buffer of at least the grid size; has_skip says to read it.
    skipped = tl.load(skip_ptr + row).to(tl.int32) != 0
    bad = bad & ((has_skip == 0) | (skipped == 0))
    out_row = row + row_offset
    current = tl.load(first_ptr + out_row)
    tl.store(first_ptr + out_row, tl.where(bad & (current == NONE), stage, current))
    if WITH_MAG:
        if (mag_row >= 0) & (out_row < MAG_ROWS):
            tl.store(mag_ptr + mag_row * MAG_ROWS + out_row, tl.max(peak, axis=0))


@triton.jit(do_not_specialize=["row_stride", "n_cols", "stage", "row_offset"])
def _first_nonfinite_fp8_kernel(
    x_ptr,
    row_stride,
    n_cols,
    first_ptr,
    stage,
    row_offset,
    NONE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The fp8 e4m3 twin of ``_first_nonfinite_kernel`` over raw bytes: e4m3 has no infinity,
    and its only non-finite codes are the NaNs 0x7F and 0xFF."""
    row = tl.program_id(0)
    base = x_ptr + row.to(tl.int64) * row_stride
    count = tl.zeros((BLOCK,), dtype=tl.int32)
    for start in range(0, n_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        byte = tl.load(base + offs, mask=offs < n_cols, other=0).to(tl.int32)
        count += ((byte & 0x7F) == 0x7F).to(tl.int32)
    bad = tl.sum(count, axis=0) > 0
    out_row = row + row_offset
    current = tl.load(first_ptr + out_row)
    tl.store(first_ptr + out_row, tl.where(bad & (current == NONE), stage, current))


@triton.jit(do_not_specialize=["o_cols", "s_cols", "cp", "layer", "row_offset"])
def _capture_recv_kernel(
    o_ptr,
    s_ptr,
    o_cols,
    s_cols,
    cp,
    layer,
    row_offset,
    step_ptr,
    count_ptr,
    meta_ptr,
    o_cap_ptr,
    s_cap_ptr,
    BLOCK: tl.constexpr,
    CAPTURES: tl.constexpr,
    META: tl.constexpr,
    O_WORDS: tl.constexpr,
    S_WORDS: tl.constexpr,
):
    """One program per received entry (row, src) of a Helix all-to-all: raw bits of partial_o
    (bf16) and softmax stats (fp32, -inf allowed in the max slots, as a sanitized entry has)."""
    row = tl.program_id(0)
    src = tl.program_id(1)
    entry = row.to(tl.int64) * cp + src
    o_base = o_ptr + entry * o_cols
    big = 2**30
    o_count = tl.zeros((BLOCK,), dtype=tl.int32)
    o_first = tl.full((BLOCK,), big, dtype=tl.int32)
    for start in range(0, o_cols, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        bits = tl.load(o_base + offs, mask=offs < o_cols, other=0).to(tl.int32) & 0xFFFF
        bad = (bits & 0x7F80) == 0x7F80
        o_count += bad.to(tl.int32)
        o_first = tl.minimum(o_first, tl.where(bad, offs, big))
    o_total = tl.sum(o_count, axis=0)
    o_min = tl.min(o_first, axis=0)
    s_offs = tl.arange(0, S_WORDS)
    s_bits = tl.load(s_ptr + entry * s_cols + s_offs, mask=s_offs < s_cols, other=0)
    neg_inf_max = (s_bits == -8388608) & ((s_offs % 2) == 0)  # 0xFF800000 in a max slot
    s_bad = (((s_bits & 0x7F800000) == 0x7F800000) & ~neg_inf_max) & (s_offs < s_cols)
    s_total = tl.sum(s_bad.to(tl.int32), axis=0)
    s_min = tl.min(tl.where(s_bad, s_offs, big), axis=0)
    if o_total + s_total > 0:
        slot = tl.atomic_add(count_ptr, 1)
        if slot < CAPTURES:
            meta = meta_ptr + slot * META
            tl.store(meta + 0, layer)
            tl.store(meta + 1, tl.load(step_ptr))
            tl.store(meta + 2, row + row_offset)
            tl.store(meta + 3, src)
            tl.store(meta + 4, tl.where(o_total > 0, o_min, -1))
            tl.store(meta + 5, o_total)
            tl.store(meta + 6, tl.where(s_total > 0, s_min, -1))
            tl.store(meta + 7, s_total)
            tl.store(meta + 8, tl.minimum(o_cols, O_WORDS))
            tl.store(meta + 9, s_cols)
            for start in range(0, O_WORDS, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                keep = (offs < o_cols) & (offs < O_WORDS)
                w_bits = tl.load(o_base + offs, mask=keep, other=0)
                tl.store(o_cap_ptr + slot * O_WORDS + offs, w_bits, mask=offs < O_WORDS)
            tl.store(s_cap_ptr + slot * S_WORDS + s_offs, s_bits)


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
        # Stands in for a skip mask when a mark has none (the kernel always takes a bool pointer).
        self.no_skip = torch.zeros((MAX_ROWS,), dtype=torch.bool, device=device)
        # Helix snapshot of the current step and its ring; -1 = not recorded.
        self.helix = torch.full(
            (HELIX_LAYERS, len(HELIX_FIELDS), HELIX_ROWS), -1, dtype=torch.int32, device=device
        )
        self.helix_kv = torch.full((HELIX_LAYERS, HELIX_SEQS), -1, dtype=torch.int32, device=device)
        self.ring_helix = torch.full(
            (RING,) + tuple(self.helix.shape), -1, dtype=torch.int32, device=device
        )
        self.ring_helix_kv = torch.full(
            (RING,) + tuple(self.helix_kv.shape), -1, dtype=torch.int32, device=device
        )
        # MLA layer index -> its row in the snapshot, in the order a forward first records them.
        self.helix_layers: Dict[int, int] = {}
        # Raw captures of non-finite received all-to-all entries (see capture_recv).
        self.step_dev = torch.zeros((1,), dtype=torch.int32, device=device)
        self.recv_count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.recv_meta = torch.full(
            (RECV_CAPTURES, len(RECV_META)), -1, dtype=torch.int32, device=device
        )
        self.recv_o = torch.zeros((RECV_CAPTURES, RECV_O_WORDS), dtype=torch.int16, device=device)
        # The all-to-all's channel count, and so each entry's FIFO line offset, follows the SM count.
        self.sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self.recv_s = torch.zeros(
            (RECV_CAPTURES, RECV_STATS_WORDS), dtype=torch.int32, device=device
        )
        self.layouts: List[Optional[Tuple[int, List[int], List[int], bool]]] = [None] * RING
        self.step = 0
        self._pending: Tuple[List[int], List[int], bool] = ([], [], False)
        # Compile every variant before any CUDA graph capture can launch one. The skip mask,
        # row offset and -inf switch are runtime arguments, so the dtype is the only variant.
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            probe = torch.zeros(2, 32, dtype=dtype, device=device)
            self._launch(probe, 0, self.no_skip, False, 0)
        self._launch_fp8(torch.zeros(2, 64, dtype=torch.uint8, device=device), 0, 0)
        self.capture_recv(
            torch.zeros(2, 2, 64, dtype=torch.bfloat16, device=device),
            torch.zeros(2, 2, 4, dtype=torch.float32, device=device),
            0,
        )
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

    def _launch(
        self,
        x: torch.Tensor,
        sid: int,
        skip: torch.Tensor,
        allow_neg_inf: bool,
        row_offset: int,
    ) -> None:
        _first_nonfinite_kernel[(x.shape[0],)](
            x,
            x.stride(0),
            x.shape[1],
            self.cur,
            sid,
            self.mag,
            mag_index(sid),
            skip,
            row_offset,
            0 if skip is self.no_skip else 1,
            1 if allow_neg_inf else 0,
            NONE=NONE,
            BLOCK=1024,
            WITH_MAG=self.with_mag,
            MAG_ROWS=MAG_ROWS,
        )

    def _launch_fp8(self, x: torch.Tensor, sid: int, row_offset: int) -> None:
        _first_nonfinite_fp8_kernel[(x.shape[0],)](
            x,
            x.stride(0),
            x.shape[1],
            self.cur,
            sid,
            row_offset,
            NONE=NONE,
            BLOCK=1024,
        )

    def mark(
        self,
        x: torch.Tensor,
        sid: int,
        skip: Optional[torch.Tensor] = None,
        allow_neg_inf: bool = False,
        row_offset: int = 0,
        fp8: bool = False,
    ) -> None:
        """Record rows of ``x`` that hold a non-finite value at mark ``sid``.

        ``x`` is ``[tokens, ...]``; a contiguous tensor of more dimensions is read as
        ``[tokens, -1]``. Its row ``i`` is batch row ``row_offset + i``. Rows where ``skip``
        (bool, ``[>= tokens]``) is True are not recorded; ``allow_neg_inf`` accepts -inf.
        ``fp8`` reads a ``uint8`` tensor as fp8 e4m3 bytes (``float8_e4m3fn`` is read so anyway).
        Anything the kernel cannot read in place (strided rows, unaligned data, other dtypes,
        more than ``MAX_ROWS`` rows) is not marked: a mark never copies or raises.
        """
        if not isinstance(x, torch.Tensor):
            return
        if x.dim() > 2 and x.is_contiguous():
            x = x.view(x.shape[0], -1)
        if (
            x.dim() != 2
            or x.shape[0] == 0
            or row_offset < 0
            or row_offset + x.shape[0] > MAX_ROWS
            or x.stride(1) != 1
            or x.data_ptr() % 16 != 0
        ):
            return
        if x.dtype == torch.float8_e4m3fn or (fp8 and x.dtype == torch.uint8):
            # Rows of fp8 bytes have no skip mask or magnitude: they are query rows.
            if skip is None:
                self._launch_fp8(x.view(torch.uint8), sid, row_offset)
            return
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return
        if skip is None:
            skip = self.no_skip
        elif (
            skip.dtype != torch.bool
            or skip.dim() != 1
            or skip.numel() < x.shape[0]
            or skip.stride(0) != 1
            or skip.data_ptr() % 16 != 0
            or skip.device != x.device
        ):
            return
        self._launch(x, sid, skip, allow_neg_inf, row_offset)

    def record_helix(
        self,
        layer: int,
        positions: Optional[torch.Tensor],
        local_slots: Optional[torch.Tensor],
        bounds: Optional[torch.Tensor],
        kv_lens: Optional[torch.Tensor],
        row_offset: int = 0,
    ) -> None:
        """Copy, for MLA ``layer``, the per-row Helix buffers its decode reads (global position,
        rank-local write slot, rank-local KV bound; generation rows from batch row
        ``row_offset`` on) and the per-sequence rank-local KV lengths into this step's snapshot.
        Device-to-device copies only, so it runs inside CUDA graphs; anything malformed is
        skipped."""
        index = self.helix_layers.get(layer)
        if index is None:
            if len(self.helix_layers) >= HELIX_LAYERS:
                return
            index = self.helix_layers[layer] = len(self.helix_layers)
        for field, values in enumerate((positions, local_slots, bounds)):
            if not _int32_vector(values):
                continue
            count = min(values.shape[0], HELIX_ROWS - row_offset)
            if count > 0:
                self.helix[index, field, row_offset : row_offset + count].copy_(values[:count])
        if _int32_vector(kv_lens):
            count = min(kv_lens.shape[0], HELIX_SEQS)
            if count > 0:
                self.helix_kv[index, :count].copy_(kv_lens[:count])

    def capture_recv(
        self,
        partial_o: torch.Tensor,
        softmax_stats: torch.Tensor,
        layer: int,
        row_offset: int = 0,
    ) -> None:
        """Capture, raw, every entry of a Helix FIFO v2 all-to-all result that holds a non-finite
        value: ``partial_o`` ``[tokens, cp, heads * rank]`` bf16 and ``softmax_stats``
        ``[tokens, cp, heads * 2]`` fp32, as ``alltoall_native`` returns them (row ``t``, source
        rank ``src``). The first ``RECV_CAPTURES`` per process are kept. Nothing else is read or
        written, so it runs inside CUDA graphs; other shapes are skipped."""
        if (
            not isinstance(partial_o, torch.Tensor)
            or not isinstance(softmax_stats, torch.Tensor)
            or partial_o.dim() != 3
            or softmax_stats.dim() != 3
            or partial_o.dtype != torch.bfloat16
            or softmax_stats.dtype != torch.float32
            or not partial_o.is_contiguous()
            or not softmax_stats.is_contiguous()
            or partial_o.shape[:2] != softmax_stats.shape[:2]
            or softmax_stats.shape[2] > RECV_STATS_WORDS
            or partial_o.shape[0] == 0
            or partial_o.data_ptr() % 16 != 0
            or softmax_stats.data_ptr() % 16 != 0
        ):
            return
        tokens, cp = partial_o.shape[0], partial_o.shape[1]
        _capture_recv_kernel[(tokens, cp)](
            partial_o.view(torch.int16),
            softmax_stats.view(torch.int32),
            partial_o.shape[2],
            softmax_stats.shape[2],
            cp,
            layer,
            row_offset,
            self.step_dev,
            self.recv_count,
            self.recv_meta,
            self.recv_o,
            self.recv_s,
            BLOCK=1024,
            CAPTURES=RECV_CAPTURES,
            META=len(RECV_META),
            O_WORDS=RECV_O_WORDS,
            S_WORDS=RECV_STATS_WORDS,
        )

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
        if self.helix_layers:
            self.ring_helix[slot].copy_(self.helix)
            self.ring_helix_kv[slot].copy_(self.helix_kv)
            self.helix.fill_(-1)
            self.helix_kv.fill_(-1)
        rids, lens, is_dummy = self._pending
        self.layouts[slot] = (self.step, rids, lens, is_dummy)
        self._pending = ([], [], False)
        self.step += 1
        self.step_dev += 1

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
            start, mine, owner, sequence = 0, None, [], None
            for index, (rid, n) in enumerate(zip(rids, lens)):
                if rid == request_id:
                    mine, sequence = (start, n), index
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
                    "_sequence": sequence,
                }
            )
        result = {"steps_recorded": self.step, "steps": steps}
        if self.with_mag:
            self._attach_magnitudes(result)
        if self.helix_layers:
            self._attach_helix(result)
        captured = int(self.recv_count.item())
        if captured:
            result["recv_captures_total"] = captured
            result["recv_captures"] = self._recv_captures(min(captured, RECV_CAPTURES))
        for entry in steps:
            entry.pop("_sequence")
        return result

    def _recv_captures(self, count: int) -> List[Dict]:
        """The first ``count`` raw captures: metadata, the runs of non-finite bf16 words in the
        received partial_o entry (start, length, the words' hex), the whole entry as hex words,
        and the stats entry as hex words."""
        meta = self.recv_meta[:count].cpu().tolist()
        o_bits = self.recv_o[:count].cpu()
        s_bits = self.recv_s[:count].cpu()
        out = []
        for slot, values in enumerate(meta):
            entry = dict(zip(RECV_META, values))
            entry["sm_count"] = self.sm_count
            words = [int(v) & 0xFFFF for v in o_bits[slot, : max(0, entry["o_words"])].tolist()]
            runs, start = [], None
            for index, word in enumerate(words + [0]):
                bad = (word & 0x7F80) == 0x7F80
                if bad and start is None:
                    start = index
                elif not bad and start is not None:
                    runs.append(
                        [
                            start,
                            index - start,
                            " ".join(f"{w:04x}" for w in words[start : min(index, start + 8)]),
                        ]
                    )
                    start = None
            entry["o_bad_runs"] = runs[:32]
            entry["o_hex"] = " ".join(f"{w:04x}" for w in words)
            stats_words = max(0, min(entry["s_words"], RECV_STATS_WORDS))
            entry["stats_hex"] = " ".join(
                f"{int(v) & 0xFFFFFFFF:08x}" for v in s_bits[slot, :stats_words].tolist()
            )
            out.append(entry)
        return out

    def _attach_helix(self, result: Dict, max_changes: int = 16) -> None:
        """Each step's Helix snapshot for the request: its rows' position, write slot and bound
        at the first recorded MLA layer, its sequence's KV length there, and every later MLA
        layer whose values differ from the first (none, unless something rewrote them
        mid-step)."""
        helix = self.ring_helix.cpu()
        helix_kv = self.ring_helix_kv.cpu()
        layers = sorted(self.helix_layers, key=self.helix_layers.get)
        for entry in result["steps"]:
            snapshot = helix[entry["step"] % RING]
            start, count = entry["request_rows"]
            rows = slice(start, min(start + count, HELIX_ROWS))
            if rows.start >= rows.stop:
                continue
            values = snapshot[:, :, rows]
            sequence = entry["_sequence"]
            kv = helix_kv[entry["step"] % RING, :, sequence] if sequence < HELIX_SEQS else None
            first = values[0]
            changes = []
            for index in range(1, len(layers)):
                for field, row in torch.nonzero(values[index] != first).tolist():
                    changes.append(
                        [layers[index], HELIX_FIELDS[field], row, int(values[index, field, row])]
                    )
            if kv is not None:
                for index in range(1, len(layers)):
                    if int(kv[index]) != int(kv[0]):
                        changes.append([layers[index], "kv_len", None, int(kv[index])])
            entry["helix"] = {
                "first_layer": layers[0],
                **{name: first[field].tolist() for field, name in enumerate(HELIX_FIELDS)},
                "kv_len": None if kv is None else int(kv[0]),
                "changes": changes[:max_changes],
                "num_changes": len(changes),
            }

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
                    # The first layer mark at or after the first non-finite one (which may be a
                    # per-rank mark without a magnitude of its own).
                    order = mark_order(first)
                    cut = next(
                        (
                            n
                            for n, (j, _) in enumerate(recorded)
                            if mark_order(mag_name(j)) >= order
                        ),
                        cut,
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


def _int32_vector(values) -> bool:
    return (
        isinstance(values, torch.Tensor)
        and values.dtype == torch.int32
        and values.dim() == 1
        and values.is_cuda
    )


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
    return min(rows, key=lambda r: mark_order(r["first"])) if rows else None
