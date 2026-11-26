# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real ZMQ-speaking Mock Proto Engine for CPU contract, E2E HTTP, and fault tests."""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
from typing import Any, Callable, Optional

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
    pack_control_frames,
    pack_session_frames,
    unpack_control_frames,
    unpack_session_frames,
)
from tensorrt_llm.engine_client.frame_decoder import (
    DecodedTokenLogprob,
    FrameBuilder,
    encode_tensor_ref,
)


def default_mock_capabilities() -> capabilities_pb2.Capabilities:
    caps = capabilities_pb2.Capabilities()
    caps.model.name = "mock-model"
    caps.model.engine_name = "mock-proto-engine"
    caps.model.vocab_size = 1000
    caps.model.max_seq_len = 2048
    caps.model.eos_token_ids.append(2)
    caps.model.pad_token_id = 0
    caps.tokenizer.vocab_size = 1000
    caps.tokenizer.eos_token_ids.append(2)
    caps.sampling.max_num_sequences = 8
    caps.sampling.temperature.kind = capabilities_pb2.PARAM_DOMAIN_KIND_RANGE
    caps.sampling.temperature.min_value = 0.0
    caps.sampling.temperature.max_value = 2.0
    caps.sampling.top_p.kind = capabilities_pb2.PARAM_DOMAIN_KIND_RANGE
    caps.sampling.top_p.min_value = 0.0
    caps.sampling.top_p.max_value = 1.0
    caps.sampling.logit_bias.supported = True
    caps.sampling.logit_bias.max_entries = 64
    caps.beam.max_beam_width = 4
    caps.logprobs.max_top_k = 10
    caps.logprobs.rank = True
    caps.logprobs.prompt.supported = True
    caps.logprobs.prompt.max_top_k = 10
    caps.proc.thinking_budget = True
    caps.proc.registered_names.append("registered_proc")
    caps.guided.kinds.extend(
        [
            capabilities_pb2.GUIDED_DECODING_KIND_JSON_SCHEMA,
            capabilities_pb2.GUIDED_DECODING_KIND_JSON_OBJECT,
            capabilities_pb2.GUIDED_DECODING_KIND_REGEX,
        ]
    )
    caps.lora.supported = True
    caps.lora.path_loading = True
    caps.mm.modes.extend(
        [
            request_pb2.MULTIMODAL_MODE_MEDIA,
            request_pb2.MULTIMODAL_MODE_PREPROCESSED,
        ]
    )
    caps.mm.image_routing_version = 1
    caps.kv.tokens_per_block = 32
    caps.kv.total_blocks = 128
    caps.kv.events.mode = capabilities_pb2.KV_EVENTS_MODE_BOTH
    caps.disagg.instance_id = "mock-ctx-0"
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


class MockProtoEngine:
    """In-process background-thread ZMQ engine speaking ``trtllm.serve_engine.v1``."""

    def __init__(
        self,
        *,
        capabilities: Optional[capabilities_pb2.Capabilities] = None,
        tokens_per_request: Optional[list[int]] = None,
        hmac_key: Optional[bytes] = None,
        reject_predicate: Optional[Callable[[request_pb2.Request], Optional[str]]] = None,
        step_delay_s: float = 0.0,
    ) -> None:
        self.hmac_key = bytes(hmac_key or os.urandom(32))
        self.capabilities = capabilities or default_mock_capabilities()
        self._tokens = list(tokens_per_request or [10, 11, 12])
        self._reject_predicate = reject_predicate
        self._step_delay_s = step_delay_s

        self._ctx = zmq.Context()
        self._session_sock = self._ctx.socket(zmq.ROUTER)
        self._session_sock.bind("tcp://127.0.0.1:0")
        self.session_addr: str = self._session_sock.getsockopt(zmq.LAST_ENDPOINT).decode()

        self._control_sock = self._ctx.socket(zmq.ROUTER)
        self._control_sock.bind("tcp://127.0.0.1:0")
        self.control_addr: str = self._control_sock.getsockopt(zmq.LAST_ENDPOINT).decode()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_seq: dict[bytes, int] = {}
        self._batch_seq: dict[bytes, int] = {}
        self.received_requests: list[request_pb2.Request] = []
        self.received_aux_counts: list[int] = []
        self.received_cancels: list[tuple[int, int]] = []
        self.received_progress: list[int] = []
        self.admin_commands: list[str] = []
        self.released_handoffs: list[tuple[str, int]] = []
        self._active_handoffs: dict[int, disagg_pb2.KvHandoff] = {}
        self._loaded_loras: dict[int, common_pb2.LoraAdapterInfo] = {}
        self._identities: list[bytes] = []
        self._pending_notices: list[session_pb2.SessionNotice] = []
        self._lock = threading.Lock()

    def start(self) -> "MockProtoEngine":
        self._thread = threading.Thread(target=self._loop, name="mock_proto_engine", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._session_sock.close(linger=200)
        self._control_sock.close(linger=200)
        self._ctx.term()

    def __enter__(self) -> "MockProtoEngine":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def inject_notice(self, notice: session_pb2.SessionNotice) -> None:
        with self._lock:
            self._pending_notices.append(notice)

    def _next_frame_seq(self, ident: bytes) -> int:
        seq = self._frame_seq.get(ident, 0) + 1
        self._frame_seq[ident] = seq
        return seq

    def _send_session(
        self,
        ident: bytes,
        msg: session_pb2.ServerMessage,
        aux_buffers: list[bytes] | None = None,
    ) -> None:
        frames = pack_session_frames(self.hmac_key, msg.SerializeToString(), aux_buffers or [])
        self._session_sock.send_multipart([ident, *frames])

    def _loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._session_sock, zmq.POLLIN)
        poller.register(self._control_sock, zmq.POLLIN)
        while not self._stop.is_set():
            with self._lock:
                notices = self._pending_notices[:]
                self._pending_notices.clear()
                idents = list(self._identities)
            for notice in notices:
                for ident in idents:
                    self._send_session(ident, session_pb2.ServerMessage(notice=notice))

            events = dict(poller.poll(timeout=10))
            if self._session_sock in events:
                raw = self._session_sock.recv_multipart()
                self._on_session(raw[0], raw[1:])
            if self._control_sock in events:
                raw = self._control_sock.recv_multipart()
                self._on_control(raw[0], raw[1:])

    def _on_session(self, ident: bytes, frames: list[bytes]) -> None:
        proto_bytes, aux = unpack_session_frames(self.hmac_key, frames)
        msg = session_pb2.ClientMessage()
        msg.ParseFromString(proto_bytes)
        which = msg.WhichOneof("payload")
        if which == "open":
            with self._lock:
                if ident not in self._identities:
                    self._identities.append(ident)
            accepted = session_pb2.SessionAccepted(
                version=common_pb2.ProtocolVersion(major=PROTOCOL_MAJOR, minor=PROTOCOL_MINOR),
                session_id=len(self._identities),
                capabilities=self.capabilities,
                limits=session_pb2.SessionLimits(
                    max_rows_per_frame=128,
                    max_frame_bytes=8 * 1024 * 1024,
                    max_inflight_requests=1024,
                    max_submit_batch=32,
                    max_submit_bytes=8 * 1024 * 1024,
                    max_request_bytes=4 * 1024 * 1024,
                    max_inline_tensor_bytes=65536,
                    progress_interval_frames=2,
                    progress_interval_ms=100,
                ),
                host_clock_ns=time.monotonic_ns(),
            )
            self._send_session(ident, session_pb2.ServerMessage(accepted=accepted))
        elif which == "submit_batch":
            if msg.submit_batch.consumed_frame_seq:
                self.received_progress.append(int(msg.submit_batch.consumed_frame_seq))
            self.received_aux_counts.append(len(aux))
            self._serve_submit_batch(ident, msg.submit_batch)
        elif which == "cancel_batch":
            if msg.cancel_batch.consumed_frame_seq:
                self.received_progress.append(int(msg.cancel_batch.consumed_frame_seq))
            for item in msg.cancel_batch.cancels:
                self.received_cancels.append((int(item.handle), int(item.reason)))
        elif which == "progress":
            self.received_progress.append(int(msg.progress.consumed_frame_seq))

    def _serve_submit_batch(self, ident: bytes, batch: session_pb2.SubmitBatch) -> None:
        bseq = self._batch_seq.get(ident, 0) + 1
        self._batch_seq[ident] = bseq
        ack = common_pb2.SubmitAck(batch_seq=bseq)
        accepted_reqs: list[request_pb2.Request] = []
        for idx, req in enumerate(batch.requests):
            self.received_requests.append(req)
            err_msg = self._reject_predicate(req) if self._reject_predicate else None
            if err_msg:
                ack.rejected.add(
                    index=idx,
                    handle=req.handle,
                    error=common_pb2.ErrorInfo(
                        code=common_pb2.ERROR_CODE_OVERLOADED,
                        message=err_msg,
                    ),
                )
            else:
                accepted_reqs.append(req)

        if not accepted_reqs:
            self._send_session(ident, session_pb2.ServerMessage(submit_ack=ack))
            return

        # Emit tokens step-by-step so streaming + progress + cancel paths are genuinely exercised.
        first_frame = True
        for req in accepted_reqs:
            if req.phase == common_pb2.PHASE_CONTEXT_ONLY:
                self._emit_context_done(ident, req, ack if first_frame else None)
                first_frame = False
                continue

            max_toks = int(req.stop.max_tokens) if req.stop.max_tokens else len(self._tokens)
            emit_toks = self._tokens[:max_toks]
            n_seqs = max(1, int(req.sampling.num_sequences or 1))
            want_lp = req.HasField("output") and req.output.HasField("logprobs")
            want_gen_logits = req.HasField("output") and req.output.tensors.generation_logits

            if req.streaming:
                for step_idx, tok in enumerate(emit_toks):
                    if self._step_delay_s:
                        time.sleep(self._step_delay_s)
                    is_last = step_idx == len(emit_toks) - 1
                    fb = FrameBuilder(
                        frame_seq=self._next_frame_seq(ident),
                        iter_first=step_idx + 1,
                        iter_last=step_idx + 1,
                    )
                    if first_frame:
                        fb.add_ack(ack)
                        first_frame = False
                    aux_out: list[bytes] = []
                    for seq_i in range(n_seqs):
                        flags = common_pb2.ROW_FLAG_FIRST_ROW if step_idx == 0 else 0
                        finish = common_pb2.FINISH_REASON_NOT_FINISHED
                        extra = None
                        if is_last:
                            flags |= common_pb2.ROW_FLAG_SEQ_FINAL
                            if seq_i == n_seqs - 1:
                                flags |= common_pb2.ROW_FLAG_REQUEST_DONE
                            finish = common_pb2.FINISH_REASON_END_ID
                            fi = output_pb2.FinalInfo(
                                usage=output_pb2.Usage(
                                    prompt_tokens=len(req.prompt_tokens),
                                    completion_tokens=len(emit_toks),
                                    cached_tokens=1,
                                )
                            )
                            if want_gen_logits:
                                arr = np.arange(16, dtype=np.float32).reshape(4, 4)
                                fi.tensors.add(
                                    kind=output_pb2.OUTPUT_TENSOR_KIND_GENERATION_LOGITS,
                                    ref=encode_tensor_ref(
                                        arr, aux_buffers=aux_out, max_inline_bytes=8
                                    ),
                                )
                            extra = output_pb2.RowExtra(final_info=fi)
                        lps = (
                            [DecodedTokenLogprob(sampled=-0.1, rank=1, topk=[(tok, -0.1)])]
                            if want_lp
                            else None
                        )
                        fb.add_row(
                            handle=req.handle,
                            seq=seq_i,
                            tokens=[tok],
                            finish=finish,
                            row_flags=flags,
                            cum_logprob=-0.1 * (step_idx + 1) - 0.05 * seq_i,
                            cached_tokens=1,
                            logprobs=lps,
                            extra=extra,
                        )
                    self._send_session(ident, session_pb2.ServerMessage(frame=fb.build()), aux_out)
            else:
                fb = FrameBuilder(frame_seq=self._next_frame_seq(ident), iter_first=1, iter_last=1)
                if first_frame:
                    fb.add_ack(ack)
                    first_frame = False
                aux_out = []
                for seq_i in range(n_seqs):
                    flags = common_pb2.ROW_FLAG_FIRST_ROW | common_pb2.ROW_FLAG_SEQ_FINAL
                    if seq_i == n_seqs - 1:
                        flags |= common_pb2.ROW_FLAG_REQUEST_DONE
                    fi = output_pb2.FinalInfo(
                        usage=output_pb2.Usage(
                            prompt_tokens=len(req.prompt_tokens),
                            completion_tokens=len(emit_toks),
                            cached_tokens=2,
                        )
                    )
                    if want_gen_logits:
                        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
                        fi.tensors.add(
                            kind=output_pb2.OUTPUT_TENSOR_KIND_GENERATION_LOGITS,
                            ref=encode_tensor_ref(arr, aux_buffers=aux_out, max_inline_bytes=8),
                        )
                    lps = (
                        [
                            DecodedTokenLogprob(sampled=-0.2, rank=1, topk=[(t, -0.2)])
                            for t in emit_toks
                        ]
                        if want_lp
                        else None
                    )
                    fb.add_row(
                        handle=req.handle,
                        seq=seq_i,
                        tokens=emit_toks,
                        finish=common_pb2.FINISH_REASON_END_ID,
                        row_flags=flags,
                        cum_logprob=-0.2 * len(emit_toks) - 0.1 * seq_i,
                        cached_tokens=2,
                        logprobs=lps,
                        extra=output_pb2.RowExtra(final_info=fi),
                    )
                self._send_session(ident, session_pb2.ServerMessage(frame=fb.build()), aux_out)

    def _emit_context_done(
        self,
        ident: bytes,
        req: request_pb2.Request,
        ack: Optional[common_pb2.SubmitAck],
    ) -> None:
        disagg_id = int(req.disagg.request_id or req.handle)
        handoff = disagg_pb2.KvHandoff(
            disagg_request_id=disagg_id,
            ctx_instance_id=self.capabilities.disagg.instance_id or "mock-ctx-0",
            ctx_engine_request_id=req.handle,
            ctx_endpoint="tcp://127.0.0.1:9100",
            first_gen_tokens=[self._tokens[0]],
            opaque_state=b"mock-opaque-kv",
        )
        self._active_handoffs[disagg_id] = handoff
        raw = handoff.SerializeToString()
        mac = hmac.new(self.hmac_key, raw, hashlib.sha256).digest()
        signed = disagg_pb2.SignedHandoff(handoff_bytes=raw, mac=mac, key_id="mock-k1")
        fb = FrameBuilder(frame_seq=self._next_frame_seq(ident), iter_first=1, iter_last=1)
        if ack is not None:
            fb.add_ack(ack)
        fb.add_row(
            handle=req.handle,
            seq=0,
            tokens=[],
            finish=common_pb2.FINISH_REASON_CONTEXT_DONE,
            row_flags=(
                common_pb2.ROW_FLAG_FIRST_ROW
                | common_pb2.ROW_FLAG_SEQ_FINAL
                | common_pb2.ROW_FLAG_REQUEST_DONE
            ),
            extra=output_pb2.RowExtra(
                handoff=signed,
                final_info=output_pb2.FinalInfo(
                    usage=output_pb2.Usage(
                        prompt_tokens=len(req.prompt_tokens), completion_tokens=1
                    )
                ),
            ),
        )
        self._send_session(ident, session_pb2.ServerMessage(frame=fb.build()))

    def _on_control(self, ident: bytes, frames: list[bytes]) -> None:
        method, call_id, _, body = unpack_control_frames(self.hmac_key, frames)
        req = CONTROL_REQUEST_TYPES[method]()
        req.ParseFromString(body)
        resp: Any
        if method == ControlMethod.GET_INFO:
            resp = control_pb2.GetInfoResponse(
                capabilities=self.capabilities,
                server=control_pb2.ServerRuntimeInfo(
                    version=common_pb2.ProtocolVersion(major=PROTOCOL_MAJOR, minor=PROTOCOL_MINOR),
                    host_build="mock-proto-engine",
                    host_clock_ns=time.monotonic_ns(),
                ),
            )
        elif method == ControlMethod.HEALTH:
            resp = control_pb2.HealthResponse(
                state=control_pb2.HEALTH_STATE_SERVING,
                active_handoffs=len(self._active_handoffs),
            )
        elif method == ControlMethod.GET_LOAD:
            resp = control_pb2.LoadInfo(
                running_requests=0,
                kv_blocks_total=128,
                sampled_at_unix_nanos=time.time_ns(),
            )
        elif method == ControlMethod.GET_KV_CAPACITY:
            resp = control_pb2.KvCapacityResponse(
                max_blocks=128, tokens_per_block=32, max_tokens=4096
            )
        elif method == ControlMethod.GET_ITERATION_STATS:
            resp = control_pb2.GetIterationStatsResponse(
                stats=[stats_pb2.IterationStats(iter=1, num_active_requests=1)]
            )
        elif method == ControlMethod.GET_KV_EVENTS:
            resp = stats_pb2.KvEventBatch(
                events=[stats_pb2.KvEvent(event_id=1, kind=stats_pb2.KV_EVENT_KIND_ALL_CLEARED)]
            )
        elif method == ControlMethod.RELEASE_HANDOFFS:
            resp = control_pb2.ReleaseHandoffsResponse()
            for key in req.items:
                self.released_handoffs.append((key.ctx_instance_id, int(key.disagg_request_id)))
                found = self._active_handoffs.pop(int(key.disagg_request_id), None) is not None
                resp.results.add(
                    key=key,
                    status=(
                        control_pb2.RELEASE_HANDOFF_STATUS_RELEASED
                        if found
                        else control_pb2.RELEASE_HANDOFF_STATUS_NOT_FOUND
                    ),
                )
        elif method == ControlMethod.GET_TRANSCEIVER_STATE:
            resp = control_pb2.GetTransceiverStateResponse(state=b"mock-transceiver")
        elif method == ControlMethod.LORA_LOAD:
            info = common_pb2.LoraAdapterInfo(
                adapter_id=req.adapter.adapter_id,
                name=req.adapter.name,
                path=req.adapter.path,
            )
            self._loaded_loras[info.adapter_id] = info
            resp = control_pb2.LoraLoadResponse(adapter=info)
        elif method == ControlMethod.LORA_UNLOAD:
            self._loaded_loras.pop(req.adapter_id, None)
            resp = control_pb2.LoraUnloadResponse()
        elif method == ControlMethod.LORA_LIST:
            resp = control_pb2.LoraListResponse(adapters=list(self._loaded_loras.values()))
        elif method == ControlMethod.ADMIN:
            self.admin_commands.append(req.WhichOneof("command") or "")
            resp = control_pb2.AdminResponse(message="ok")
        else:
            err = common_pb2.ErrorInfo(
                code=common_pb2.ERROR_CODE_UNSUPPORTED, message=f"Unsupported {method.name}"
            )
            reply = pack_control_frames(
                self.hmac_key, method, call_id, err.SerializeToString(), is_error=True
            )
            self._control_sock.send_multipart([ident, *reply])
            return

        reply = pack_control_frames(
            self.hmac_key, method, call_id, resp.SerializeToString(), is_error=False
        )
        self._control_sock.send_multipart([ident, *reply])


__all__ = ["MockProtoEngine", "default_mock_capabilities"]
