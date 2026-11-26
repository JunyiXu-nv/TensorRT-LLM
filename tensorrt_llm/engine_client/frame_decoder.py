# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Columnar little-endian ``Frame`` encoder and zero-copy decoder for ``trtllm.serve_engine.v1``."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from tensorrt_llm.contracts.serve_engine.v1 import common_pb2, output_pb2

NO_REPLACE_SENTINEL = 0xFFFFFFFF
NO_EXTRA_SENTINEL = 0xFFFFFFFF

_DTYPE_TO_NP: dict[int, Any] = {
    common_pb2.DTYPE_F32: np.float32,
    common_pb2.DTYPE_F16: np.float16,
    common_pb2.DTYPE_I64: np.int64,
    common_pb2.DTYPE_I32: np.int32,
    common_pb2.DTYPE_I16: np.int16,
    common_pb2.DTYPE_I8: np.int8,
    common_pb2.DTYPE_U8: np.uint8,
    common_pb2.DTYPE_BOOL: np.bool_,
    common_pb2.DTYPE_F64: np.float64,
}


@dataclass(slots=True)
class DecodedTokenLogprob:
    """Decoded per-token logprob entry unpacked from columnar ``Frame.lp_*``."""

    sampled: Optional[float]
    rank: Optional[int]
    topk: list[tuple[int, float]] = field(default_factory=list)


@dataclass(slots=True)
class DecodedRow:
    """One row unpacked from a columnar ``output_pb2.Frame``."""

    handle: int
    seq: int
    tokens: list[int]
    finish: int
    row_flags: int
    stop_index: int = -1
    replace_from: int = NO_REPLACE_SENTINEL
    cum_logprob: Optional[float] = None
    decoding_iter: int = 0
    cached_tokens: int = 0
    logprobs: Optional[list[DecodedTokenLogprob]] = None
    extra: Optional[output_pb2.RowExtra] = None

    @property
    def is_first_row(self) -> bool:
        return bool(self.row_flags & common_pb2.ROW_FLAG_FIRST_ROW)

    @property
    def is_seq_final(self) -> bool:
        return bool(self.row_flags & common_pb2.ROW_FLAG_SEQ_FINAL)

    @property
    def is_request_done(self) -> bool:
        return bool(self.row_flags & common_pb2.ROW_FLAG_REQUEST_DONE)


@dataclass(slots=True)
class DecodedFrame:
    """Decoded batch frame with all rows copied into lightweight Python lists."""

    frame_seq: int
    iter_first: int
    iter_last: int
    emit_time_ns: int
    rows: list[DecodedRow]
    acks: list[common_pb2.SubmitAck]
    iteration_stats: list[Any]
    lora_set_changed: list[common_pb2.LoraSetChanged]
    aux_buffers: list[bytes] = field(default_factory=list)


def _unpack_col(
    raw: bytes,
    dtype: str,
    expected_len: int,
    name: str,
    *,
    optional_default: Any = None,
) -> list[Any]:
    if not raw:
        if optional_default is not None:
            return [optional_default] * expected_len
        if expected_len == 0:
            return []
        raise ValueError(f"Frame column {name!r} is required for n_rows={expected_len}")
    dt = np.dtype(dtype)
    if len(raw) != expected_len * dt.itemsize:
        raise ValueError(
            f"Frame column {name!r} byte length {len(raw)} != expected "
            f"{expected_len} * {dt.itemsize}"
        )
    # Convert whole column once into Python scalars so no long-lived slice pins frame bytes.
    return np.frombuffer(raw, dtype=dt).tolist()


def decode_frame(
    frame: output_pb2.Frame,
    aux_buffers: Optional[Sequence[bytes]] = None,
) -> DecodedFrame:
    """Decode a columnar ``output_pb2.Frame`` into per-row structures."""
    n_rows = int(frame.n_rows)
    if n_rows == 0:
        return DecodedFrame(
            frame_seq=frame.frame_seq,
            iter_first=frame.iter_first,
            iter_last=frame.iter_last,
            emit_time_ns=frame.emit_time_ns,
            rows=[],
            acks=list(frame.acks),
            iteration_stats=list(frame.iteration_stats),
            lora_set_changed=list(frame.lora_set_changed),
            aux_buffers=list(aux_buffers or ()),
        )

    handles = _unpack_col(frame.handle, "<u8", n_rows, "handle")
    seqs = _unpack_col(frame.seq, "<u4", n_rows, "seq", optional_default=0)
    token_offsets = _unpack_col(frame.token_offsets, "<u4", n_rows + 1, "token_offsets")
    if token_offsets[0] != 0:
        raise ValueError(f"Frame token_offsets[0] must be 0, got {token_offsets[0]}")
    n_tokens = int(token_offsets[-1])
    tokens = _unpack_col(frame.tokens, "<i4", n_tokens, "tokens", optional_default=0)
    finishes = _unpack_col(frame.finish, "<u1", n_rows, "finish")
    row_flags = _unpack_col(frame.row_flags, "<u2", n_rows, "row_flags")
    stop_indices = _unpack_col(frame.stop_index, "<i4", n_rows, "stop_index", optional_default=-1)
    replace_froms = _unpack_col(
        frame.replace_from, "<u4", n_rows, "replace_from", optional_default=NO_REPLACE_SENTINEL
    )
    cum_logprobs = _unpack_col(
        frame.cum_logprob, "<f8", n_rows, "cum_logprob", optional_default=math.nan
    )
    decoding_iters = _unpack_col(
        frame.decoding_iter, "<u4", n_rows, "decoding_iter", optional_default=0
    )
    cached_tokens = _unpack_col(
        frame.cached_tokens, "<u4", n_rows, "cached_tokens", optional_default=0
    )
    extra_indices = _unpack_col(
        frame.extra_index, "<u4", n_rows, "extra_index", optional_default=NO_EXTRA_SENTINEL
    )

    has_logprobs = bool(frame.lp_sampled)
    lp_sampled: list[float] = []
    lp_rank: list[int] = []
    lp_topk_offsets: list[int] = []
    lp_topk_ids: list[int] = []
    lp_topk_logprobs: list[float] = []
    if has_logprobs and n_tokens > 0:
        lp_sampled = _unpack_col(frame.lp_sampled, "<f4", n_tokens, "lp_sampled")
        lp_rank = _unpack_col(frame.lp_rank, "<u4", n_tokens, "lp_rank", optional_default=0)
        if frame.lp_topk_offsets:
            lp_topk_offsets = _unpack_col(
                frame.lp_topk_offsets, "<u4", n_tokens + 1, "lp_topk_offsets"
            )
            n_cand = int(lp_topk_offsets[-1])
            lp_topk_ids = _unpack_col(frame.lp_topk_ids, "<i4", n_cand, "lp_topk_ids")
            lp_topk_logprobs = _unpack_col(
                frame.lp_topk_logprobs, "<f4", n_cand, "lp_topk_logprobs"
            )
        else:
            lp_topk_offsets = [0] * (n_tokens + 1)

    extras = list(frame.extras)
    rows: list[DecodedRow] = []
    for i in range(n_rows):
        t0, t1 = int(token_offsets[i]), int(token_offsets[i + 1])
        if t1 < t0 or t1 > n_tokens:
            raise ValueError(f"Invalid token_offsets[{i}]={t0}..{t1} for n_tokens={n_tokens}")
        row_tokens = tokens[t0:t1]
        row_lps: Optional[list[DecodedTokenLogprob]] = None
        if has_logprobs and t1 > t0:
            row_lps = []
            for j in range(t0, t1):
                s_val = lp_sampled[j]
                s_opt = None if math.isnan(s_val) else float(s_val)
                r_opt = int(lp_rank[j]) if lp_rank[j] > 0 else None
                c0, c1 = int(lp_topk_offsets[j]), int(lp_topk_offsets[j + 1])
                topk = [(int(lp_topk_ids[k]), float(lp_topk_logprobs[k])) for k in range(c0, c1)]
                row_lps.append(DecodedTokenLogprob(sampled=s_opt, rank=r_opt, topk=topk))
            if all(lp.sampled is None and not lp.topk for lp in row_lps):
                row_lps = None

        extra: Optional[output_pb2.RowExtra] = None
        ex_idx = int(extra_indices[i])
        if row_flags[i] & common_pb2.ROW_FLAG_HAS_EXTRA:
            if ex_idx >= len(extras):
                raise ValueError(f"extra_index[{i}]={ex_idx} out of range ({len(extras)})")
            extra = extras[ex_idx]

        cum_lp = cum_logprobs[i]
        rows.append(
            DecodedRow(
                handle=int(handles[i]),
                seq=int(seqs[i]),
                tokens=row_tokens,
                finish=int(finishes[i]),
                row_flags=int(row_flags[i]),
                stop_index=int(stop_indices[i]),
                replace_from=int(replace_froms[i]),
                cum_logprob=None if math.isnan(cum_lp) else float(cum_lp),
                decoding_iter=int(decoding_iters[i]),
                cached_tokens=int(cached_tokens[i]),
                logprobs=row_lps,
                extra=extra,
            )
        )

    return DecodedFrame(
        frame_seq=frame.frame_seq,
        iter_first=frame.iter_first,
        iter_last=frame.iter_last,
        emit_time_ns=frame.emit_time_ns,
        rows=rows,
        acks=list(frame.acks),
        iteration_stats=list(frame.iteration_stats),
        lora_set_changed=list(frame.lora_set_changed),
        aux_buffers=list(aux_buffers or ()),
    )


class FrameBuilder:
    """Packs per-step engine rows into a columnar ``output_pb2.Frame``."""

    def __init__(
        self,
        *,
        frame_seq: int,
        iter_first: int = 0,
        iter_last: int = 0,
        emit_time_ns: int = 0,
    ) -> None:
        self._frame_seq = frame_seq
        self._iter_first = iter_first
        self._iter_last = iter_last
        self._emit_time_ns = emit_time_ns
        self._handles: list[int] = []
        self._seqs: list[int] = []
        self._token_offsets: list[int] = [0]
        self._tokens: list[int] = []
        self._finishes: list[int] = []
        self._row_flags: list[int] = []
        self._stop_indices: list[int] = []
        self._replace_froms: list[int] = []
        self._cum_logprobs: list[float] = []
        self._decoding_iters: list[int] = []
        self._cached_tokens: list[int] = []
        self._extra_indices: list[int] = []
        self._extras: list[output_pb2.RowExtra] = []
        self._has_any_logprobs = False
        self._lp_sampled: list[float] = []
        self._lp_rank: list[int] = []
        self._lp_topk_offsets: list[int] = [0]
        self._lp_topk_ids: list[int] = []
        self._lp_topk_logprobs: list[float] = []
        self._acks: list[common_pb2.SubmitAck] = []
        self._iteration_stats: list[Any] = []
        self._lora_set_changed: list[common_pb2.LoraSetChanged] = []

    def add_ack(self, ack: common_pb2.SubmitAck) -> None:
        self._acks.append(ack)

    def add_iteration_stat(self, stat: Any) -> None:
        self._iteration_stats.append(stat)

    def add_lora_delta(self, delta: common_pb2.LoraSetChanged) -> None:
        self._lora_set_changed.append(delta)

    def add_row(
        self,
        *,
        handle: int,
        seq: int = 0,
        tokens: Sequence[int] = (),
        finish: int = common_pb2.FINISH_REASON_NOT_FINISHED,
        row_flags: int = 0,
        stop_index: int = -1,
        replace_from: int = NO_REPLACE_SENTINEL,
        cum_logprob: Optional[float] = None,
        decoding_iter: int = 0,
        cached_tokens: int = 0,
        logprobs: Optional[Sequence[DecodedTokenLogprob]] = None,
        extra: Optional[output_pb2.RowExtra] = None,
    ) -> None:
        self._handles.append(handle)
        self._seqs.append(seq)
        tok_list = [int(t) for t in tokens]
        self._tokens.extend(tok_list)
        self._token_offsets.append(len(self._tokens))
        self._finishes.append(finish)

        flags = int(row_flags)
        if extra is not None:
            flags |= common_pb2.ROW_FLAG_HAS_EXTRA
            self._extra_indices.append(len(self._extras))
            self._extras.append(extra)
        else:
            self._extra_indices.append(NO_EXTRA_SENTINEL)
        self._row_flags.append(flags)

        self._stop_indices.append(stop_index)
        self._replace_froms.append(replace_from)
        self._cum_logprobs.append(math.nan if cum_logprob is None else float(cum_logprob))
        self._decoding_iters.append(decoding_iter)
        self._cached_tokens.append(cached_tokens)

        if logprobs is not None:
            self._has_any_logprobs = True
            for lp in logprobs:
                self._lp_sampled.append(math.nan if lp.sampled is None else float(lp.sampled))
                self._lp_rank.append(0 if lp.rank is None else int(lp.rank))
                for cand_id, cand_lp in lp.topk:
                    self._lp_topk_ids.append(int(cand_id))
                    self._lp_topk_logprobs.append(float(cand_lp))
                self._lp_topk_offsets.append(len(self._lp_topk_ids))
        else:
            for _ in tok_list:
                self._lp_sampled.append(math.nan)
                self._lp_rank.append(0)
                self._lp_topk_offsets.append(len(self._lp_topk_ids))

    def build(self) -> output_pb2.Frame:
        n_rows = len(self._handles)
        frame = output_pb2.Frame(
            frame_seq=self._frame_seq,
            iter_first=self._iter_first,
            iter_last=self._iter_last,
            emit_time_ns=self._emit_time_ns,
            n_rows=n_rows,
        )
        if n_rows > 0:
            frame.handle = np.asarray(self._handles, dtype="<u8").tobytes()
            if any(s != 0 for s in self._seqs):
                frame.seq = np.asarray(self._seqs, dtype="<u4").tobytes()
            frame.token_offsets = np.asarray(self._token_offsets, dtype="<u4").tobytes()
            if self._tokens:
                frame.tokens = np.asarray(self._tokens, dtype="<i4").tobytes()
            frame.finish = np.asarray(self._finishes, dtype="<u1").tobytes()
            frame.row_flags = np.asarray(self._row_flags, dtype="<u2").tobytes()
            if any(s != -1 for s in self._stop_indices):
                frame.stop_index = np.asarray(self._stop_indices, dtype="<i4").tobytes()
            if any(r != NO_REPLACE_SENTINEL for r in self._replace_froms):
                frame.replace_from = np.asarray(self._replace_froms, dtype="<u4").tobytes()
            if any(not math.isnan(c) for c in self._cum_logprobs):
                frame.cum_logprob = np.asarray(self._cum_logprobs, dtype="<f8").tobytes()
            if any(d != 0 for d in self._decoding_iters):
                frame.decoding_iter = np.asarray(self._decoding_iters, dtype="<u4").tobytes()
            if any(c != 0 for c in self._cached_tokens):
                frame.cached_tokens = np.asarray(self._cached_tokens, dtype="<u4").tobytes()
            if self._extras:
                frame.extra_index = np.asarray(self._extra_indices, dtype="<u4").tobytes()
                frame.extras.extend(self._extras)
            if self._has_any_logprobs and self._tokens:
                frame.lp_sampled = np.asarray(self._lp_sampled, dtype="<f4").tobytes()
                if any(r != 0 for r in self._lp_rank):
                    frame.lp_rank = np.asarray(self._lp_rank, dtype="<u4").tobytes()
                if self._lp_topk_ids:
                    frame.lp_topk_offsets = np.asarray(self._lp_topk_offsets, dtype="<u4").tobytes()
                    frame.lp_topk_ids = np.asarray(self._lp_topk_ids, dtype="<i4").tobytes()
                    frame.lp_topk_logprobs = np.asarray(
                        self._lp_topk_logprobs, dtype="<f4"
                    ).tobytes()
        if self._acks:
            frame.acks.extend(self._acks)
        if self._iteration_stats:
            frame.iteration_stats.extend(self._iteration_stats)
        if self._lora_set_changed:
            frame.lora_set_changed.extend(self._lora_set_changed)
        return frame


def encode_tensor_ref(
    array: np.ndarray,
    *,
    aux_buffers: list[bytes],
    max_inline_bytes: int = 65536,
) -> common_pb2.TensorRef:
    """Encode a numpy array into a ``TensorRef`` (inline or zero-copy aux ZMQ frame)."""
    contiguous = np.ascontiguousarray(array)
    dtype_map = {v: k for k, v in _DTYPE_TO_NP.items()}
    pb_dtype = dtype_map.get(contiguous.dtype.type, common_pb2.DTYPE_U8)
    raw = contiguous.tobytes()
    ref = common_pb2.TensorRef(dtype=pb_dtype, shape=list(contiguous.shape))
    if len(raw) <= max_inline_bytes:
        ref.inline_bytes = raw
    else:
        idx = len(aux_buffers)
        aux_buffers.append(raw)
        # 1-based ZMQ auxiliary frame index carried in uint64 blob_id in Phase 1.
        ref.blob.blob_id = idx + 1
        ref.blob.size = len(raw)
    return ref


def decode_tensor_ref(
    ref: common_pb2.TensorRef,
    aux_buffers: Optional[Sequence[bytes]] = None,
) -> np.ndarray:
    """Decode a ``TensorRef`` back into a numpy array."""
    which = ref.WhichOneof("location")
    if which == "inline_bytes":
        raw = ref.inline_bytes
    elif which == "blob" and ref.blob.blob_id > 0:
        idx = int(ref.blob.blob_id) - 1
        if aux_buffers is None or idx >= len(aux_buffers):
            raise ValueError(f"Missing ZMQ aux buffer index {idx}")
        raw = aux_buffers[idx]
    else:
        raise ValueError(f"Unsupported TensorRef delivery kind: {which!r}")
    np_dtype = _DTYPE_TO_NP.get(ref.dtype, np.uint8)
    arr = np.frombuffer(raw, dtype=np_dtype).copy()
    if ref.shape:
        arr = arr.reshape(tuple(ref.shape))
    return arr


__all__ = [
    "DecodedFrame",
    "DecodedRow",
    "DecodedTokenLogprob",
    "FrameBuilder",
    "NO_EXTRA_SENTINEL",
    "NO_REPLACE_SENTINEL",
    "decode_frame",
    "decode_tensor_ref",
    "encode_tensor_ref",
]
