# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin, opt-in rank-0 ZMQ bridge speaking ``trtllm.serve_engine.v1`` over ``BaseWorker``.

Activated only when ``TRTLLM_ENGINE_PROTOCOL=v1_zmq`` is set. Keeps the default pickled
executor path untouched and is designed to be replaced cleanly once the native Proto
engine lands.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import queue
import threading
import time
from typing import Any, Optional

import numpy as np
import zmq

from tensorrt_llm.contracts.serve_engine.v1 import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    capabilities_pb2,
    common_pb2,
    control_pb2,
    disagg_pb2,
    output_pb2,
    request_pb2,
    session_pb2,
    stats_pb2,
)
from tensorrt_llm.contracts.serve_engine.v1.zmq_wire import (
    CONTROL_REQUEST_TYPES,
    ControlMethod,
    WireFramingError,
    pack_control_frames,
    pack_session_frames,
    unpack_control_frames,
    unpack_session_frames,
)
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams

from .request import GenerationRequest, LoRARequest

PROTO_ZMQ_ENV = "TRTLLM_ENGINE_PROTOCOL"
PROTO_ZMQ_VALUE = "v1_zmq"
_NO_REPLACE_SENTINEL = 0xFFFFFFFF
_NO_EXTRA_SENTINEL = 0xFFFFFFFF
_BRIDGE_CLIENT_ID_OFFSET = 1 << 40


def is_proto_zmq_enabled() -> bool:
    return os.getenv(PROTO_ZMQ_ENV, "").strip().lower() == PROTO_ZMQ_VALUE


def _pack_columnar_frame(
    *,
    frame_seq: int,
    iter_num: int,
    rows: list[dict[str, Any]],
    acks: list[common_pb2.SubmitAck],
) -> output_pb2.Frame:
    n_rows = len(rows)
    frame = output_pb2.Frame(
        frame_seq=frame_seq,
        iter_first=iter_num,
        iter_last=iter_num,
        emit_time_ns=time.monotonic_ns(),
        n_rows=n_rows,
    )
    if acks:
        frame.acks.extend(acks)
    if n_rows == 0:
        return frame

    handles: list[int] = []
    seqs: list[int] = []
    offsets: list[int] = [0]
    tokens: list[int] = []
    finishes: list[int] = []
    flags_col: list[int] = []
    stop_indices: list[int] = []
    replace_froms: list[int] = []
    cum_lps: list[float] = []
    cached_col: list[int] = []
    extra_indices: list[int] = []
    extras: list[output_pb2.RowExtra] = []

    for r in rows:
        handles.append(int(r["handle"]))
        seqs.append(int(r.get("seq", 0)))
        toks = [int(t) for t in r.get("tokens", ())]
        tokens.extend(toks)
        offsets.append(len(tokens))
        finishes.append(int(r.get("finish", common_pb2.FINISH_REASON_NOT_FINISHED)))
        rf = int(r.get("row_flags", 0))
        extra = r.get("extra")
        if extra is not None:
            rf |= common_pb2.ROW_FLAG_HAS_EXTRA
            extra_indices.append(len(extras))
            extras.append(extra)
        else:
            extra_indices.append(_NO_EXTRA_SENTINEL)
        flags_col.append(rf)
        stop_indices.append(int(r.get("stop_index", -1)))
        replace_froms.append(int(r.get("replace_from", _NO_REPLACE_SENTINEL)))
        clp = r.get("cum_logprob")
        cum_lps.append(math.nan if clp is None else float(clp))
        cached_col.append(int(r.get("cached_tokens", 0)))

    frame.handle = np.asarray(handles, dtype="<u8").tobytes()
    if any(s != 0 for s in seqs):
        frame.seq = np.asarray(seqs, dtype="<u4").tobytes()
    frame.token_offsets = np.asarray(offsets, dtype="<u4").tobytes()
    if tokens:
        frame.tokens = np.asarray(tokens, dtype="<i4").tobytes()
    frame.finish = np.asarray(finishes, dtype="<u1").tobytes()
    frame.row_flags = np.asarray(flags_col, dtype="<u2").tobytes()
    if any(s != -1 for s in stop_indices):
        frame.stop_index = np.asarray(stop_indices, dtype="<i4").tobytes()
    if any(rf != _NO_REPLACE_SENTINEL for rf in replace_froms):
        frame.replace_from = np.asarray(replace_froms, dtype="<u4").tobytes()
    if any(not math.isnan(c) for c in cum_lps):
        frame.cum_logprob = np.asarray(cum_lps, dtype="<f8").tobytes()
    if any(c != 0 for c in cached_col):
        frame.cached_tokens = np.asarray(cached_col, dtype="<u4").tobytes()
    if extras:
        frame.extra_index = np.asarray(extra_indices, dtype="<u4").tobytes()
        frame.extras.extend(extras)
    return frame


def build_capabilities_for_worker(worker: Any) -> capabilities_pb2.Capabilities:
    """Synthesize a ``Capabilities`` snapshot from an active ``BaseWorker``."""
    caps = capabilities_pb2.Capabilities()
    llm_args = getattr(worker, "llm_args", None)
    caps.model.engine_name = "trtllm-pyexecutor"
    if llm_args is not None:
        caps.model.name = str(getattr(llm_args, "model", "") or "")
        caps.model.max_seq_len = int(getattr(llm_args, "max_seq_len", 0) or 0)
        caps.model.max_batch_size = int(getattr(llm_args, "max_batch_size", 0) or 0)
        caps.model.max_num_tokens = int(getattr(llm_args, "max_num_tokens", 0) or 0)
        caps.model.reasoning_parser = str(getattr(llm_args, "reasoning_parser", "") or "")
        caps.beam.max_beam_width = int(getattr(llm_args, "max_beam_width", 1) or 1)
    caps.sampling.max_num_sequences = 128
    caps.sampling.temperature.kind = capabilities_pb2.PARAM_DOMAIN_KIND_RANGE
    caps.sampling.temperature.min_value = 0.0
    caps.sampling.top_p.kind = capabilities_pb2.PARAM_DOMAIN_KIND_RANGE
    caps.sampling.top_p.min_value = 0.0
    caps.sampling.top_p.max_value = 1.0
    caps.sampling.logit_bias.supported = True
    caps.logprobs.max_top_k = 20
    caps.logprobs.rank = True
    caps.logprobs.prompt.supported = True
    caps.proc.thinking_budget = True
    caps.lora.supported = True
    caps.lora.path_loading = True
    caps.disagg.phases.extend(
        [
            common_pb2.PHASE_AGGREGATED,
            common_pb2.PHASE_CONTEXT_ONLY,
            common_pb2.PHASE_GENERATION_ONLY,
        ]
    )
    caps.disagg.schedule_styles.extend(
        [
            common_pb2.SCHEDULE_STYLE_CONTEXT_FIRST,
            common_pb2.SCHEDULE_STYLE_GENERATION_FIRST,
        ]
    )
    caps.disagg.supports_release = True
    caps.disagg.ctx_forces_single_token = True
    caps.admin.profile = True
    caps.admin.sleep_wakeup = True
    caps.admin.update_weights = True
    return caps


def proto_request_to_generation_request(
    req: request_pb2.Request,
    *,
    vocab_size: int = 0,
    client_id: Optional[int] = None,
) -> GenerationRequest:
    """Translate ``request_pb2.Request`` into a ``GenerationRequest`` for ``BaseWorker``."""
    sp = SamplingParams()
    if req.HasField("sampling"):
        s = req.sampling
        if s.num_sequences:
            sp.n = int(s.num_sequences)
        if s.beam_width > 1:
            sp.use_beam_search = True
            sp.best_of = int(s.beam_width)
        if s.HasField("temperature"):
            sp.temperature = float(s.temperature)
        if s.HasField("top_k"):
            sp.top_k = int(s.top_k)
        if s.HasField("top_p"):
            sp.top_p = float(s.top_p)
        if s.HasField("min_p"):
            sp.min_p = float(s.min_p)
        if s.HasField("seed"):
            sp.seed = int(s.seed)
        if s.HasField("repetition_penalty"):
            sp.repetition_penalty = float(s.repetition_penalty)
        if s.HasField("presence_penalty"):
            sp.presence_penalty = float(s.presence_penalty)
        if s.HasField("frequency_penalty"):
            sp.frequency_penalty = float(s.frequency_penalty)
        if s.logit_bias and vocab_size > 0:
            import torch

            bias = torch.zeros(vocab_size, dtype=torch.float32)
            for entry in s.logit_bias:
                if 0 <= entry.token_id < vocab_size:
                    bias[entry.token_id] = float(entry.bias)
            sp.embedding_bias = bias

    if req.HasField("stop"):
        st = req.stop
        if st.HasField("max_tokens"):
            sp.max_tokens = int(st.max_tokens)
        if st.HasField("min_tokens"):
            sp.min_tokens = int(st.min_tokens)
        sp.end_id = int(st.end_id)
        if st.HasField("pad_id"):
            sp.pad_id = int(st.pad_id)
        sp.ignore_eos = bool(st.ignore_eos)
        if st.sequences:
            sp._stop_word_ids = [list(seq.tokens) for seq in st.sequences]
        if st.bad_sequences:
            sp._bad_word_ids = [list(seq.tokens) for seq in st.bad_sequences]

    if req.HasField("output"):
        o = req.output
        if o.HasField("logprobs"):
            sp.logprobs = int(o.logprobs)
        if o.HasField("prompt_logprobs"):
            sp.prompt_logprobs = int(o.prompt_logprobs)
        sp.return_context_logits = bool(o.tensors.context_logits)
        sp.return_generation_logits = bool(o.tensors.generation_logits)
        sp.return_perf_metrics = bool(o.metrics.perf)

    if req.HasField("guided"):
        g = req.guided
        which = g.WhichOneof("constraint")
        if which == "json_schema":
            sp.guided_decoding = GuidedDecodingParams(json=g.json_schema)
        elif which == "json_object":
            sp.guided_decoding = GuidedDecodingParams(json_object=True)
        elif which == "regex":
            sp.guided_decoding = GuidedDecodingParams(regex=g.regex)
        elif which == "ebnf_grammar":
            sp.guided_decoding = GuidedDecodingParams(grammar=g.ebnf_grammar)
        elif which == "structural_tag":
            sp.guided_decoding = GuidedDecodingParams(structural_tag=g.structural_tag)
        elif which == "choice":
            sp.guided_decoding = GuidedDecodingParams(choice=list(g.choice.choices))

    if req.HasField("thinking_budget"):
        sp.thinking_token_budget = int(req.thinking_budget.budget)

    lora_req = None
    if req.HasField("lora"):
        ckpt_src = "nemo" if req.lora.format == common_pb2.LORA_FORMAT_NEMO else "hf"
        lora_req = LoRARequest(
            lora_name=req.lora.name or f"lora-{req.lora.adapter_id}",
            lora_int_id=int(req.lora.adapter_id),
            lora_path=req.lora.path,
            lora_ckpt_source=ckpt_src,
        )

    disagg_params = None
    if req.phase != common_pb2.PHASE_AGGREGATED or req.HasField("disagg"):
        phase_to_type = {
            common_pb2.PHASE_CONTEXT_ONLY: "context_only",
            common_pb2.PHASE_GENERATION_ONLY: "generation_only",
            common_pb2.PHASE_AGGREGATED: "aggregated",
        }
        handoff = disagg_pb2.KvHandoff()
        if req.HasField("disagg") and req.disagg.handoff.handoff_bytes:
            handoff.ParseFromString(req.disagg.handoff.handoff_bytes)
        disagg_params = DisaggregatedParams(
            request_type=phase_to_type.get(req.phase, "aggregated"),
            disagg_request_id=int(req.disagg.request_id) if req.disagg.request_id else None,
            ctx_request_id=int(handoff.ctx_engine_request_id) or None,
            first_gen_tokens=list(handoff.first_gen_tokens) or None,
            opaque_state=bytes(handoff.opaque_state) or None,
            draft_tokens=list(handoff.draft_tokens) or None,
            ctx_dp_rank=handoff.ctx_dp_rank if handoff.HasField("ctx_dp_rank") else None,
            ctx_info_endpoint=handoff.ctx_endpoint or None,
        )

    trace_headers = None
    if req.HasField("trace") and req.trace.traceparent:
        trace_headers = {"traceparent": req.trace.traceparent}
        if req.trace.tracestate:
            trace_headers["tracestate"] = req.trace.tracestate

    gen_req = GenerationRequest(
        prompt_token_ids=list(req.prompt_tokens),
        sampling_params=sp,
        lora_request=lora_req,
        streaming=bool(req.streaming),
        disaggregated_params=disagg_params,
        trace_headers=trace_headers,
        cache_salt=req.kv.cache_salt or None if req.HasField("kv") else None,
        encoder_input_token_ids=list(req.encoder_input_tokens) or None,
        priority=float(req.sched.priority)
        if req.HasField("sched") and req.sched.HasField("priority")
        else 0.5,
    )
    gen_req.set_id(int(req.handle) if client_id is None else int(client_id))
    return gen_req


class Rank0ProtoBridge:
    """Thin opt-in ZMQ-Proto bridge attached to rank-0 ``BaseWorker``."""

    def __init__(
        self,
        worker: Any,
        *,
        session_addr: str = "tcp://127.0.0.1:0",
        control_addr: str = "tcp://127.0.0.1:0",
        hmac_key: Optional[bytes] = None,
        handoff_hmac_key: Optional[bytes] = None,
    ) -> None:
        self._worker = worker
        self._hmac_key = bytes(hmac_key or os.urandom(32))
        self._handoff_hmac_key = bytes(handoff_hmac_key or self._hmac_key)
        self._ctx = zmq.Context()

        self._session_sock = self._ctx.socket(zmq.ROUTER)
        self._session_sock.set_hwm(0)
        self._session_sock.bind(session_addr)
        self.session_addr: str = self._session_sock.getsockopt(zmq.LAST_ENDPOINT).decode()

        self._control_sock = self._ctx.socket(zmq.ROUTER)
        self._control_sock.set_hwm(0)
        self._control_sock.bind(control_addr)
        self.control_addr: str = self._control_sock.getsockopt(zmq.LAST_ENDPOINT).decode()

        self._caps = build_capabilities_for_worker(worker)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._next_session_id = 1
        self._sessions: dict[bytes, int] = {}
        self._frame_seqs: dict[bytes, int] = {}
        self._batch_seqs: dict[bytes, int] = {}
        self._handle_owner: dict[int, bytes] = {}
        self._internal_to_handle: dict[int, int] = {}
        self._handle_to_internal: dict[tuple[bytes, int], int] = {}
        self._next_internal_id = _BRIDGE_CLIENT_ID_OFFSET + 1
        self._incoming_responses: queue.Queue[Any] = queue.Queue()
        self._active_handoffs: dict[int, disagg_pb2.KvHandoff] = {}
        self._iter_counter = 0

    @property
    def hmac_key(self) -> bytes:
        return self._hmac_key

    def endpoint_spec(self) -> dict[str, str]:
        return {
            "session_addr": self.session_addr,
            "control_addr": self.control_addr,
            "hmac_key_hex": self._hmac_key.hex(),
        }

    def sign_handoff(self, handoff: disagg_pb2.KvHandoff) -> disagg_pb2.SignedHandoff:
        raw = handoff.SerializeToString()
        mac = hmac.new(self._handoff_hmac_key, raw, hashlib.sha256).digest()
        return disagg_pb2.SignedHandoff(handoff_bytes=raw, mac=mac, key_id="bridge-v1")

    def verify_handoff(self, signed: disagg_pb2.SignedHandoff) -> disagg_pb2.KvHandoff:
        if signed.mac:
            expected = hmac.new(
                self._handoff_hmac_key, signed.handoff_bytes, hashlib.sha256
            ).digest()
            if not hmac.compare_digest(signed.mac, expected):
                raise ValueError("SignedHandoff HMAC verification failed")
        handoff = disagg_pb2.KvHandoff()
        handoff.ParseFromString(signed.handoff_bytes)
        return handoff

    def intercept_responses(self, responses: list[Any]) -> list[Any]:
        """Claim responses belonging to ProtoBridge requests before IPC pickling."""
        remaining: list[Any] = []
        for rsp in responses:
            cid = int(getattr(rsp, "client_id", 0))
            if cid in self._internal_to_handle:
                self._incoming_responses.put(rsp)
            else:
                remaining.append(rsp)
        return remaining

    def start(self) -> None:
        self._worker._proto_bridge = self
        self._thread = threading.Thread(
            target=self._serve_loop, name="rank0_proto_bridge", daemon=True
        )
        self._thread.start()
        logger.info(
            "Rank0ProtoBridge listening on session=%s control=%s",
            self.session_addr,
            self.control_addr,
        )

    def stop(self) -> None:
        if getattr(self._worker, "_proto_bridge", None) is self:
            self._worker._proto_bridge = None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._session_sock.close(linger=500)
        self._control_sock.close(linger=500)
        self._ctx.term()

    def _serve_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._session_sock, zmq.POLLIN)
        poller.register(self._control_sock, zmq.POLLIN)
        while not self._stop_event.is_set():
            events = dict(poller.poll(timeout=10))
            if self._session_sock in events:
                frames = self._session_sock.recv_multipart()
                self._handle_session_frames(frames[0], frames[1:])
            if self._control_sock in events:
                frames = self._control_sock.recv_multipart()
                self._handle_control_frames(frames[0], frames[1:])
            self._drain_worker_responses()

    def _handle_session_frames(self, identity: bytes, frames: list[bytes]) -> None:
        try:
            proto_bytes, _aux = unpack_session_frames(self._hmac_key, frames)
        except WireFramingError as exc:
            logger.warning("Rank0ProtoBridge dropped invalid session frame: %s", exc)
            return
        msg = session_pb2.ClientMessage()
        msg.ParseFromString(proto_bytes)
        payload = msg.WhichOneof("payload")
        if payload == "open":
            sid = self._next_session_id
            self._next_session_id += 1
            self._sessions[identity] = sid
            self._frame_seqs[identity] = 0
            self._batch_seqs[identity] = 0
            accepted = session_pb2.SessionAccepted(
                version=common_pb2.ProtocolVersion(major=PROTOCOL_MAJOR, minor=PROTOCOL_MINOR),
                session_id=sid,
                capabilities=self._caps,
                limits=session_pb2.SessionLimits(
                    max_rows_per_frame=256,
                    max_frame_bytes=16 * 1024 * 1024,
                    max_inflight_requests=4096,
                    max_submit_batch=64,
                    max_submit_bytes=16 * 1024 * 1024,
                    max_request_bytes=8 * 1024 * 1024,
                    max_inline_tensor_bytes=65536,
                    progress_interval_frames=8,
                    progress_interval_ms=250,
                ),
                host_clock_ns=time.monotonic_ns(),
            )
            self._send_server_msg(identity, session_pb2.ServerMessage(accepted=accepted))
        elif payload == "submit_batch":
            self._handle_submit_batch(identity, msg.submit_batch)
        elif payload == "cancel_batch":
            for item in msg.cancel_batch.cancels:
                internal_id = self._handle_to_internal.get((identity, int(item.handle)))
                if internal_id is not None and hasattr(self._worker, "abort_request"):
                    self._worker.abort_request(internal_id)

    def _handle_submit_batch(self, identity: bytes, batch: session_pb2.SubmitBatch) -> None:
        self._batch_seqs[identity] = self._batch_seqs.get(identity, 0) + 1
        ack = common_pb2.SubmitAck(batch_seq=self._batch_seqs[identity])
        vocab_size = int(self._caps.model.vocab_size or 0)
        for idx, req in enumerate(batch.requests):
            try:
                if req.phase == common_pb2.PHASE_GENERATION_ONLY and req.disagg.handoff.mac:
                    self.verify_handoff(req.disagg.handoff)
                use_real_worker = hasattr(self._worker, "submit") and hasattr(
                    self._worker, "_await_response_helper"
                )
                if use_real_worker:
                    internal_id = self._next_internal_id
                    self._next_internal_id += 1
                else:
                    internal_id = int(req.handle)
                gen_req = proto_request_to_generation_request(
                    req, vocab_size=vocab_size, client_id=internal_id
                )
                self._internal_to_handle[internal_id] = int(req.handle)
                self._handle_to_internal[(identity, int(req.handle))] = internal_id
                self._handle_owner[internal_id] = identity
                if use_real_worker:
                    self._worker.submit(gen_req)
                else:
                    self._worker._enqueue_request(gen_req)
            except Exception as exc:
                ack.rejected.add(
                    index=idx,
                    handle=req.handle,
                    error=common_pb2.ErrorInfo(
                        code=common_pb2.ERROR_CODE_INVALID_ARGUMENT,
                        message=str(exc),
                    ),
                )
        self._send_server_msg(identity, session_pb2.ServerMessage(submit_ack=ack))

    def _drain_worker_responses(self) -> None:
        responses: list[Any] = []
        while True:
            try:
                responses.append(self._incoming_responses.get_nowait())
            except queue.Empty:
                break
        if (
            not responses
            and not hasattr(self._worker, "_await_response_helper")
            and hasattr(self._worker, "await_responses")
        ):
            try:
                responses = self._worker.await_responses(timeout=0.0) or []
            except Exception:
                return
        if not responses:
            return
        self._iter_counter += 1
        by_identity: dict[bytes, list[dict[str, Any]]] = {}
        for rsp in responses:
            internal_id = int(getattr(rsp, "client_id", 0))
            identity = self._handle_owner.get(internal_id)
            if identity is None:
                continue
            handle = self._internal_to_handle.get(internal_id, internal_id)
            row = self._response_to_row(handle, rsp)
            by_identity.setdefault(identity, []).append(row)
            if row["row_flags"] & common_pb2.ROW_FLAG_REQUEST_DONE:
                self._handle_owner.pop(internal_id, None)
                self._internal_to_handle.pop(internal_id, None)
                self._handle_to_internal.pop((identity, handle), None)
                if hasattr(self._worker, "_pop_result"):
                    self._worker._pop_result(internal_id)

        for identity, rows in by_identity.items():
            self._frame_seqs[identity] = self._frame_seqs.get(identity, 0) + 1
            frame = _pack_columnar_frame(
                frame_seq=self._frame_seqs[identity],
                iter_num=self._iter_counter,
                rows=rows,
                acks=[],
            )
            self._send_server_msg(identity, session_pb2.ServerMessage(frame=frame))

    def _response_to_row(self, handle: int, rsp: Any) -> dict[str, Any]:
        if rsp.has_error():
            extra = output_pb2.RowExtra(
                error=common_pb2.ErrorInfo(
                    code=common_pb2.ERROR_CODE_INTERNAL,
                    message=str(rsp.error_msg),
                )
            )
            return {
                "handle": handle,
                "tokens": [],
                "finish": common_pb2.FINISH_REASON_ERROR,
                "row_flags": common_pb2.ROW_FLAG_SEQ_FINAL | common_pb2.ROW_FLAG_REQUEST_DONE,
                "extra": extra,
            }
        res = rsp.result
        if (
            hasattr(res, "_result")
            and isinstance(res._result, bytes)
            and hasattr(res, "deserialize")
        ):
            res.deserialize()
        raw_ids = getattr(res, "output_token_ids", None)
        if not raw_ids and hasattr(res, "get_result"):
            inner = res.get_result()
            raw_ids = getattr(inner, "output_token_ids", None)
        if raw_ids and isinstance(raw_ids[0], (list, tuple)):
            toks = [int(t) for t in raw_ids[0]]
        elif raw_ids:
            toks = [int(t) for t in raw_ids]
        else:
            toks = []
        is_final = bool(getattr(res, "is_final", False))
        flags = common_pb2.ROW_FLAG_SEQ_FINAL | common_pb2.ROW_FLAG_REQUEST_DONE if is_final else 0
        finish = (
            common_pb2.FINISH_REASON_END_ID if is_final else common_pb2.FINISH_REASON_NOT_FINISHED
        )
        extra = None
        ctx_params = getattr(res, "context_phase_params", None)
        if ctx_params is not None:
            finish = common_pb2.FINISH_REASON_CONTEXT_DONE
            handoff = disagg_pb2.KvHandoff(
                disagg_request_id=int(getattr(ctx_params, "req_id", handle)),
                ctx_engine_request_id=int(getattr(ctx_params, "req_id", handle)),
                first_gen_tokens=list(getattr(ctx_params, "first_gen_tokens", ()) or ()),
                opaque_state=bytes(getattr(ctx_params, "opaque_state", b"") or b""),
            )
            self._active_handoffs[handoff.disagg_request_id] = handoff
            extra = output_pb2.RowExtra(handoff=self.sign_handoff(handoff))
        return {
            "handle": handle,
            "seq": int(getattr(res, "sequence_index", 0) or 0),
            "tokens": toks,
            "finish": finish,
            "row_flags": flags,
            "extra": extra,
        }

    def _handle_control_frames(self, identity: bytes, frames: list[bytes]) -> None:
        try:
            method, call_id, _, body = unpack_control_frames(self._hmac_key, frames)
        except WireFramingError as exc:
            logger.warning("Rank0ProtoBridge dropped invalid control frame: %s", exc)
            return
        req_cls = CONTROL_REQUEST_TYPES[method]
        req = req_cls()
        req.ParseFromString(body)
        try:
            resp = self._dispatch_control(method, req)
            reply = pack_control_frames(
                self._hmac_key, method, call_id, resp.SerializeToString(), is_error=False
            )
        except Exception as exc:
            err = common_pb2.ErrorInfo(code=common_pb2.ERROR_CODE_INTERNAL, message=str(exc))
            reply = pack_control_frames(
                self._hmac_key, method, call_id, err.SerializeToString(), is_error=True
            )
        self._control_sock.send_multipart([identity, *reply])

    def _dispatch_control(self, method: ControlMethod, req: Any) -> Any:
        if method == ControlMethod.GET_INFO:
            return control_pb2.GetInfoResponse(
                capabilities=self._caps,
                server=control_pb2.ServerRuntimeInfo(
                    version=common_pb2.ProtocolVersion(major=PROTOCOL_MAJOR, minor=PROTOCOL_MINOR),
                    host_build="rank0-proto-bridge",
                    host_clock_ns=time.monotonic_ns(),
                ),
            )
        if method == ControlMethod.HEALTH:
            return control_pb2.HealthResponse(
                state=control_pb2.HEALTH_STATE_SERVING,
                active_handoffs=len(self._active_handoffs),
            )
        if method == ControlMethod.GET_LOAD:
            return control_pb2.LoadInfo(
                inflight_handles=len(self._handle_owner),
                sampled_at_unix_nanos=time.time_ns(),
            )
        if method == ControlMethod.GET_KV_CAPACITY:
            cap = (
                self._worker.fetch_kv_cache_capacity()
                if hasattr(self._worker, "fetch_kv_cache_capacity")
                else {}
            )
            return control_pb2.KvCapacityResponse(
                max_blocks=int(cap.get("max_blocks", 0) or 0),
                tokens_per_block=int(cap.get("tokens_per_block", 0) or 0),
                max_tokens=int(cap.get("max_tokens", 0) or 0),
            )
        if method == ControlMethod.GET_ITERATION_STATS:
            return control_pb2.GetIterationStatsResponse()
        if method == ControlMethod.GET_KV_EVENTS:
            return stats_pb2.KvEventBatch()
        if method == ControlMethod.RELEASE_HANDOFFS:
            resp = control_pb2.ReleaseHandoffsResponse()
            for key in req.items:
                removed = self._active_handoffs.pop(int(key.disagg_request_id), None)
                status = (
                    control_pb2.RELEASE_HANDOFF_STATUS_RELEASED
                    if removed is not None
                    else control_pb2.RELEASE_HANDOFF_STATUS_NOT_FOUND
                )
                resp.results.add(key=key, status=status)
            return resp
        if method == ControlMethod.GET_TRANSCEIVER_STATE:
            state = b""
            if hasattr(self._worker, "get_data_transceiver_state"):
                raw = self._worker.get_data_transceiver_state()
                if raw:
                    state = bytes(raw)
            return control_pb2.GetTransceiverStateResponse(state=state)
        if method == ControlMethod.LORA_LIST:
            return control_pb2.LoraListResponse()
        if method == ControlMethod.ADMIN:
            which = req.WhichOneof("command")
            if which == "start_profile" and hasattr(self._worker, "start_profile"):
                cmd = req.start_profile
                self._worker.start_profile(
                    output_dir=cmd.output_dir or None,
                    num_steps=cmd.num_steps or None,
                    start_step=cmd.start_step,
                    activities=list(cmd.activities) or None,
                )
            elif which == "stop_profile" and hasattr(self._worker, "stop_profile"):
                self._worker.stop_profile()
            elif which == "sleep" and hasattr(self._worker, "sleep"):
                self._worker.sleep(list(req.sleep.tags) or None)
            elif which == "wakeup" and hasattr(self._worker, "wakeup"):
                self._worker.wakeup(list(req.wakeup.tags) or None)
            return control_pb2.AdminResponse(message="ok")
        raise NotImplementedError(f"Unsupported control method {method.name}")

    def _send_server_msg(self, identity: bytes, msg: session_pb2.ServerMessage) -> None:
        frames = pack_session_frames(self._hmac_key, msg.SerializeToString())
        self._session_sock.send_multipart([identity, *frames])


__all__ = [
    "PROTO_ZMQ_ENV",
    "PROTO_ZMQ_VALUE",
    "Rank0ProtoBridge",
    "build_capabilities_for_worker",
    "is_proto_zmq_enabled",
    "proto_request_to_generation_request",
]
