# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level async ``ProtoEngineFacade`` replacing serving-side ``LLM`` and ``Proxy``."""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from google.protobuf.json_format import MessageToDict

from tensorrt_llm.contracts.serve_engine.v1 import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    common_pb2,
    control_pb2,
    disagg_pb2,
    output_pb2,
    session_pb2,
    stats_pb2,
)
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.logprobs import Logprob, TokenLogprobs

from .capabilities import CapabilityGate
from .frame_decoder import (
    NO_REPLACE_SENTINEL,
    DecodedFrame,
    DecodedRow,
    decode_frame,
    decode_tensor_ref,
)
from .interface import EngineProtocolError, EngineTransportError
from .postproc_pool import FrontendPostprocPool, extract_final_metrics, select_best_of_outputs
from .prompt_tail import PromptTailPolicy
from .request_mapper import RequestMapper
from .transport import ControlMethod, EngineTransport

_FINISH_REASON_TO_STR: dict[int, Optional[str]] = {
    common_pb2.FINISH_REASON_NOT_FINISHED: None,
    common_pb2.FINISH_REASON_END_ID: "stop",
    common_pb2.FINISH_REASON_STOP_SEQUENCE: "stop",
    common_pb2.FINISH_REASON_LENGTH: "length",
    common_pb2.FINISH_REASON_TIMED_OUT: "timeout",
    common_pb2.FINISH_REASON_CANCELLED: "cancelled",
    common_pb2.FINISH_REASON_ERROR: "error",
    common_pb2.FINISH_REASON_CONTEXT_DONE: "stop",
}


@dataclass
class ProtoCompletionOutput:
    """``CompletionOutput``-compatible per-sequence view driven by ``Frame`` rows."""

    index: int
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: TokenLogprobs = field(default_factory=list)
    prompt_logprobs: Optional[TokenLogprobs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[int | str] = None
    generation_logits: Any = None
    additional_context_outputs: Optional[dict[str, Any]] = None
    additional_generation_outputs: Optional[dict[str, Any]] = None
    disaggregated_params: Optional[DisaggregatedParams] = None
    signed_handoff_b64: Optional[str] = None
    request_perf_metrics: Any = None
    routed_experts: Any = None
    _last_text_len: int = 0
    _last_token_ids_len: int = 0
    _last_logprobs_len: int = 0
    _incremental_states: Optional[dict[str, Any]] = None
    _postprocess_result: Any = None

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def token_ids_diff(self) -> list[int]:
        return self.token_ids[self._last_token_ids_len :]

    @property
    def text_diff(self) -> str:
        return self.text[self._last_text_len :]

    @property
    def logprobs_diff(self) -> TokenLogprobs:
        return self.logprobs[self._last_logprobs_len :]


class ProtoGenerationStream:
    """``RequestOutput``-compatible async stream for a single ``trtllm.serve_engine.v1`` handle."""

    def __init__(
        self,
        *,
        handle: int,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: Any,
        streaming: bool,
        tokenizer: Any,
        abort_callback: Callable[[int, int], None],
        post_processor_hook: Any = None,
        postproc_pool: Optional[FrontendPostprocPool] = None,
    ) -> None:
        self.id = handle
        self.request_id = request_id
        self.prompt: Optional[str] = None
        self.prompt_token_ids = list(prompt_token_ids)
        self.sampling_params = sampling_params
        self._streaming = streaming
        self._tokenizer = tokenizer
        self._abort_callback = abort_callback
        self._post_processor_hook = post_processor_hook
        self._postproc_pool = postproc_pool or FrontendPostprocPool(num_workers=0)

        n_seqs = int(getattr(sampling_params, "n", 1) or 1)
        self.outputs: list[ProtoCompletionOutput] = [
            ProtoCompletionOutput(index=i) for i in range(n_seqs)
        ]
        self.context_logits: Any = None
        self.encoder_output: Any = None
        self.metrics_dict: dict[str, Any] = {}
        self.cached_tokens: int = 0
        self.prompt_tokens_count: int = len(prompt_token_ids)
        self.completion_tokens_count: int = 0
        self.finished: bool = False
        self._done: bool = False
        self._aborted: bool = False
        self._cancel_sent: bool = False
        self._default_abort_reason: int = common_pb2.CANCEL_REASON_CLIENT_DISCONNECTED
        self._error: Optional[BaseException] = None
        self._queue: asyncio.Queue[Optional[ProtoGenerationStream]] = asyncio.Queue()

    @property
    def error(self) -> Optional[str]:
        return str(self._error) if self._error is not None else None

    def abort(self, reason: Optional[int] = None) -> None:
        self._aborted = True
        if not self._cancel_sent:
            self._cancel_sent = True
            actual_reason = self._default_abort_reason if reason is None else reason
            self._abort_callback(self.id, actual_reason)

    def _ensure_seq(self, seq: int) -> ProtoCompletionOutput:
        while len(self.outputs) <= seq:
            self.outputs.append(ProtoCompletionOutput(index=len(self.outputs)))
        return self.outputs[seq]

    def fail(self, exc: BaseException) -> None:
        if self._done:
            return
        self._error = exc
        self._done = True
        self.finished = True
        self._queue.put_nowait(None)

    def apply_row(self, row: DecodedRow, aux_buffers: list[bytes]) -> None:
        if self._done:
            return
        out = self._ensure_seq(row.seq)
        out._last_token_ids_len = len(out.token_ids)
        out._last_logprobs_len = len(out.logprobs)
        out._last_text_len = len(out.text)

        if row.replace_from != NO_REPLACE_SENTINEL:
            out.token_ids = out.token_ids[: row.replace_from] + list(row.tokens)
        else:
            out.token_ids.extend(row.tokens)

        if row.cum_logprob is not None:
            out.cumulative_logprob = row.cum_logprob
        if row.cached_tokens:
            self.cached_tokens = row.cached_tokens

        if row.logprobs:
            for tok, lp in zip(row.tokens, row.logprobs):
                entry: dict[int, Logprob] = {}
                if lp.sampled is not None:
                    entry[tok] = Logprob(logprob=lp.sampled, rank=lp.rank)
                for rank_idx, (cand_id, cand_lp) in enumerate(lp.topk, start=1):
                    entry.setdefault(cand_id, Logprob(logprob=cand_lp, rank=rank_idx))
                out.logprobs.append(entry)

        if row.finish != common_pb2.FINISH_REASON_NOT_FINISHED:
            out.finish_reason = _FINISH_REASON_TO_STR.get(row.finish, "stop")
            if row.finish == common_pb2.FINISH_REASON_STOP_SEQUENCE and row.stop_index >= 0:
                out.stop_reason = row.stop_index

        if row.extra is not None:
            self._apply_extra(out, row.extra, aux_buffers)

        verdict = self._postproc_pool.process_sequence(
            self,
            out,
            is_final=row.is_seq_final or row.is_request_done,
            stop_index=row.stop_index,
        )
        if verdict.should_cancel:
            self.abort(verdict.cancel_reason)

        is_done_now = (
            row.is_request_done
            or row.finish == common_pb2.FINISH_REASON_ERROR
            or verdict.force_done
        )
        if is_done_now:
            self.outputs = select_best_of_outputs(self.outputs, self.sampling_params)
            self._done = True
            self.finished = True
            if self._error is not None:
                self._queue.put_nowait(None)
                return
            self._queue.put_nowait(self)
            self._queue.put_nowait(None)
        elif self._streaming:
            self._queue.put_nowait(self)

    def _apply_extra(
        self,
        out: ProtoCompletionOutput,
        extra: output_pb2.RowExtra,
        aux_buffers: list[bytes],
    ) -> None:
        if extra.HasField("error"):
            self._error = EngineProtocolError.from_proto(extra.error)
            return
        if extra.HasField("prompt_info"):
            pi = extra.prompt_info
            if pi.prompt_len:
                self.prompt_tokens_count = int(pi.prompt_len)
            if pi.expanded_prompt_tokens:
                self.prompt_token_ids = list(pi.expanded_prompt_tokens)
            for tensor in pi.tensors:
                self._attach_tensor(out, tensor, aux_buffers)
        if extra.HasField("handoff"):
            signed = extra.handoff
            handoff = disagg_pb2.KvHandoff()
            if signed.handoff_bytes:
                handoff.ParseFromString(signed.handoff_bytes)
            dp = DisaggregatedParams(
                request_type="context_only",
                first_gen_tokens=list(handoff.first_gen_tokens) or None,
                ctx_request_id=handoff.ctx_engine_request_id or None,
                opaque_state=bytes(handoff.opaque_state) or None,
                draft_tokens=list(handoff.draft_tokens) or None,
                disagg_request_id=handoff.disagg_request_id or None,
                ctx_dp_rank=handoff.ctx_dp_rank if handoff.HasField("ctx_dp_rank") else None,
                ctx_info_endpoint=handoff.ctx_endpoint or None,
            )
            out.signed_handoff_b64 = base64.b64encode(signed.SerializeToString()).decode("ascii")
            out.disaggregated_params = dp
        if extra.HasField("final_info"):
            fi = extra.final_info
            extracted = extract_final_metrics(fi)
            self.metrics_dict.update(extracted)
            if fi.HasField("usage"):
                self.prompt_tokens_count = int(fi.usage.prompt_tokens)
                self.completion_tokens_count = int(fi.usage.completion_tokens)
                self.cached_tokens = int(fi.usage.cached_tokens)
            for tensor in fi.tensors:
                self._attach_tensor(out, tensor, aux_buffers)
        for tensor in extra.tensors:
            self._attach_tensor(out, tensor, aux_buffers)

    def _attach_tensor(
        self,
        out: ProtoCompletionOutput,
        tensor: output_pb2.OutputTensor,
        aux_buffers: list[bytes],
    ) -> None:
        if not tensor.HasField("ref"):
            return
        arr = decode_tensor_ref(tensor.ref, aux_buffers)
        kind = tensor.kind
        if kind == output_pb2.OUTPUT_TENSOR_KIND_CONTEXT_LOGITS:
            self.context_logits = arr
        elif kind == output_pb2.OUTPUT_TENSOR_KIND_GENERATION_LOGITS:
            out.generation_logits = arr
        elif kind == output_pb2.OUTPUT_TENSOR_KIND_ENCODER_OUTPUT:
            self.encoder_output = arr
        elif kind == output_pb2.OUTPUT_TENSOR_KIND_ROUTED_EXPERTS:
            out.routed_experts = arr
        elif kind == output_pb2.OUTPUT_TENSOR_KIND_ADDITIONAL_CONTEXT:
            out.additional_context_outputs = out.additional_context_outputs or {}
            out.additional_context_outputs[tensor.name] = arr
        elif kind == output_pb2.OUTPUT_TENSOR_KIND_ADDITIONAL_GENERATION:
            out.additional_generation_outputs = out.additional_generation_outputs or {}
            out.additional_generation_outputs[tensor.name] = arr

    def __aiter__(self) -> AsyncIterator[ProtoGenerationStream]:
        return self

    async def __anext__(self) -> ProtoGenerationStream:
        item = await self._queue.get()
        if item is None:
            if self._error is not None:
                raise self._error
            raise StopAsyncIteration
        return item

    def __await__(self):
        return self.aresult().__await__()

    async def aresult(self) -> ProtoGenerationStream:
        last: ProtoGenerationStream = self
        async for step in self:
            last = step
        return last


class ProtoEngineFacade:
    """Async frontend facade over ``trtllm.serve_engine.v1`` (ZMQ in Phase 1, gRPC-ready)."""

    def __init__(
        self,
        transport: EngineTransport,
        *,
        tokenizer: Any = None,
        args: Any = None,
        frontend_label: str = "trtllm-serve",
        post_processor_hook: Any = None,
        model_dir: Optional[str] = None,
        num_postproc_workers: int = 0,
        prompt_tail_tokens: int = 0,
    ) -> None:
        self._transport = transport
        self.tokenizer = tokenizer
        self.args = args
        self._hf_model_dir = model_dir
        self._frontend_label = frontend_label
        self._post_processor_hook = post_processor_hook
        self._postproc_pool = FrontendPostprocPool(num_workers=num_postproc_workers)
        self._prompt_tail = PromptTailPolicy(prompt_tail_tokens) if prompt_tail_tokens else None
        self.llm_id = f"proto-facade-{id(self):x}"
        self.input_processor = self._default_input_processor

        self._cap_gate = CapabilityGate()
        self._mapper = RequestMapper(tokenizer=tokenizer)
        self._session_id: int = 0
        self._limits = session_pb2.SessionLimits()
        self._connected = False
        self._healthy = True
        self._engine_dead_error: Optional[BaseException] = None

        self._handle_lock = threading.Lock()
        self._next_handle = 1
        self._inflight: dict[int, ProtoGenerationStream] = {}
        self._pending_submits: list[tuple[Any, list[bytes]]] = []
        self._pending_cancels: list[tuple[int, int]] = []
        self._flush_scheduled = False

        self._consumed_frame_seq = 0
        self._last_reported_frame_seq = 0
        self._last_progress_time = time.monotonic()

        self._recv_task: Optional[asyncio.Task[None]] = None
        self._progress_task: Optional[asyncio.Task[None]] = None
        self._stats_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._kv_events_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def capability_gate(self) -> CapabilityGate:
        return self._cap_gate

    @property
    def capabilities(self):
        return self._cap_gate.capabilities

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def limits(self) -> session_pb2.SessionLimits:
        return self._limits

    async def start(self) -> session_pb2.SessionAccepted:
        """Open ``SessionService.Stream`` handshake and start background frame dispatch."""
        if self._connected:
            return session_pb2.SessionAccepted(
                session_id=self._session_id,
                capabilities=self._cap_gate.capabilities,
                limits=self._limits,
            )
        self._loop = asyncio.get_running_loop()
        open_msg = session_pb2.SessionOpen(
            version=common_pb2.ProtocolVersion(major=PROTOCOL_MAJOR, minor=PROTOCOL_MINOR),
            frontend_label=self._frontend_label,
            want=session_pb2.SessionPreferences(
                tensor_delivery=[
                    common_pb2.TENSOR_DELIVERY_INLINE,
                    common_pb2.TENSOR_DELIVERY_BLOB_FETCH,
                ],
                want_iteration_stats=True,
            ),
        )
        accepted = await self._transport.open_session(open_msg)
        self._session_id = accepted.session_id
        self._limits.CopyFrom(accepted.limits)
        self._cap_gate.update(accepted.capabilities)
        if self.tokenizer is not None:
            self._cap_gate.verify_tokenizer(self.tokenizer)
        if accepted.capabilities.model.eos_token_ids:
            self._mapper = RequestMapper(
                tokenizer=self.tokenizer,
                default_end_id=int(accepted.capabilities.model.eos_token_ids[0]),
                default_pad_id=(
                    int(accepted.capabilities.model.pad_token_id)
                    if accepted.capabilities.model.HasField("pad_token_id")
                    else None
                ),
                max_inline_tensor_bytes=int(accepted.limits.max_inline_tensor_bytes or 65536),
            )
        self._connected = True
        self._healthy = True
        self._recv_task = asyncio.create_task(self._recv_loop(), name="proto_facade_recv")
        self._progress_task = asyncio.create_task(
            self._progress_loop(), name="proto_facade_progress"
        )
        return accepted

    def _alloc_handle(self) -> int:
        with self._handle_lock:
            h = self._next_handle
            self._next_handle += 1
            return h

    def _default_input_processor(
        self, inputs: Any, sampling_params: Any = None
    ) -> tuple[list[int], dict[str, Any]]:
        toks, mm_params, _ = self._tokenize_inputs(inputs)
        extra: dict[str, Any] = {}
        if mm_params is not None:
            extra["multimodal_params"] = mm_params
        return toks, extra

    def _tokenize_inputs(self, inputs: Any) -> tuple[list[int], Any, Any]:
        mm_params = None
        enc_tokens = None
        if isinstance(inputs, Mapping):
            if "prompt_token_ids" in inputs:
                toks = list(inputs["prompt_token_ids"])
            elif "prompt" in inputs and self.tokenizer is not None:
                toks = list(self.tokenizer.encode(inputs["prompt"]))
            else:
                toks = []
            mm_params = inputs.get("multi_modal_data") or inputs.get("multimodal_params")
            enc_tokens = inputs.get("encoder_input_token_ids")
            return toks, mm_params, enc_tokens
        if hasattr(inputs, "prompt_token_ids"):
            return (
                list(inputs.prompt_token_ids),
                getattr(inputs, "multimodal_params", None),
                getattr(inputs, "encoder_input_token_ids", None),
            )
        if isinstance(inputs, str):
            if self.tokenizer is None:
                raise EngineProtocolError(
                    "Tokenizer is required to encode string prompts",
                    code=common_pb2.ERROR_CODE_INVALID_ARGUMENT,
                )
            return list(self.tokenizer.encode(inputs)), None, None
        return [int(t) for t in inputs], None, None

    def generate_async(
        self,
        inputs: Any,
        sampling_params: Any = None,
        lora_request: Any = None,
        prompt_adapter_request: Any = None,
        streaming: bool = False,
        kv_cache_retention_config: Any = None,
        disaggregated_params: Any = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        _postproc_params: Any = None,
        scheduling_params: Any = None,
        conversation_params: Any = None,
        cache_salt: Optional[str] = None,
        priority: float = 0.5,
        kv_hints: Any = None,
    ) -> ProtoGenerationStream:
        if self._engine_dead_error is not None:
            raise EngineTransportError(str(self._engine_dead_error))
        if not self._connected:
            raise EngineTransportError("ProtoEngineFacade.start() has not been called")

        prompt_tokens, mm_params, enc_tokens = self._tokenize_inputs(inputs)
        handle = self._alloc_handle()
        req, aux_buffers = self._mapper.build_request(
            handle=handle,
            request_id=f"req-{handle}",
            prompt_token_ids=prompt_tokens,
            sampling_params=sampling_params,
            streaming=streaming,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            kv_cache_retention_config=kv_cache_retention_config,
            disaggregated_params=disaggregated_params,
            trace_headers=trace_headers,
            multimodal_params=mm_params,
            scheduling_params=scheduling_params,
            conversation_params=conversation_params,
            cache_salt=cache_salt,
            encoder_input_token_ids=enc_tokens,
            priority=priority,
            kv_hints=kv_hints,
        )
        self._cap_gate.validate_request(req)

        stream = ProtoGenerationStream(
            handle=handle,
            request_id=req.request_id,
            prompt_token_ids=prompt_tokens,
            sampling_params=sampling_params,
            streaming=streaming,
            tokenizer=self.tokenizer,
            abort_callback=self.abort,
            post_processor_hook=self._post_processor_hook,
            postproc_pool=self._postproc_pool,
        )
        self._inflight[handle] = stream
        self._pending_submits.append((req, aux_buffers))
        self._schedule_flush()
        return stream

    def abort(
        self, handle: int, reason: int = common_pb2.CANCEL_REASON_CLIENT_DISCONNECTED
    ) -> None:
        if not self._connected:
            return
        self._pending_cancels.append((handle, reason))
        self._schedule_flush()

    def _schedule_flush(self) -> None:
        if self._flush_scheduled or self._loop is None:
            return
        self._flush_scheduled = True
        self._loop.call_soon(lambda: asyncio.create_task(self._flush_outbound()))

    async def _flush_outbound(self) -> None:
        self._flush_scheduled = False
        if not self._connected:
            return
        if self._pending_submits:
            batch_items = self._pending_submits
            self._pending_submits = []
            max_batch = int(self._limits.max_submit_batch or 64)
            for idx in range(0, len(batch_items), max_batch):
                chunk = batch_items[idx : idx + max_batch]
                batch = session_pb2.SubmitBatch(
                    requests=[req for req, _ in chunk],
                    consumed_frame_seq=self._consumed_frame_seq,
                )
                self._last_reported_frame_seq = self._consumed_frame_seq
                self._last_progress_time = time.monotonic()
                aux: list[bytes] = []
                for _, item_aux in chunk:
                    aux.extend(item_aux)
                await self._transport.send_client_message(
                    session_pb2.ClientMessage(submit_batch=batch), aux
                )
        if self._pending_cancels:
            cancels = self._pending_cancels
            self._pending_cancels = []
            cb = session_pb2.CancelBatch(
                cancels=[session_pb2.CancelItem(handle=h, reason=r) for h, r in cancels],
                consumed_frame_seq=self._consumed_frame_seq,
            )
            self._last_reported_frame_seq = self._consumed_frame_seq
            self._last_progress_time = time.monotonic()
            await self._transport.send_client_message(session_pb2.ClientMessage(cancel_batch=cb))

    async def _recv_loop(self) -> None:
        try:
            while self._connected:
                server_msg, aux_buffers = await self._transport.recv_server_message()
                payload = server_msg.WhichOneof("payload")
                if payload == "frame":
                    decoded = decode_frame(server_msg.frame, aux_buffers)
                    self._apply_decoded_frame(decoded)
                elif payload == "submit_ack":
                    self._apply_submit_ack(server_msg.submit_ack)
                elif payload == "notice":
                    self._apply_notice(server_msg.notice)
                elif payload == "iteration_stats":
                    self._enqueue_stat(server_msg.iteration_stats)
                elif payload == "lora_set_changed":
                    self._cap_gate.apply_lora_delta(server_msg.lora_set_changed)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._mark_engine_dead(exc)

    def _apply_decoded_frame(self, decoded: DecodedFrame) -> None:
        for ack in decoded.acks:
            self._apply_submit_ack(ack)
        for row in decoded.rows:
            stream = self._inflight.get(row.handle)
            if stream is None:
                continue
            stream.apply_row(row, decoded.aux_buffers)
            if stream.finished:
                self._inflight.pop(row.handle, None)
        for delta in decoded.lora_set_changed:
            self._cap_gate.apply_lora_delta(delta)
        for stat in decoded.iteration_stats:
            self._enqueue_stat(stat)

        if decoded.frame_seq > self._consumed_frame_seq:
            self._consumed_frame_seq = decoded.frame_seq
            interval = int(self._limits.progress_interval_frames or 8)
            if (self._consumed_frame_seq - self._last_reported_frame_seq) >= interval:
                asyncio.create_task(self._send_progress())

    def _apply_submit_ack(self, ack: common_pb2.SubmitAck) -> None:
        for rej in ack.rejected:
            stream = self._inflight.pop(rej.handle, None)
            if stream is not None:
                stream.fail(EngineProtocolError.from_proto(rej.error))

    def _apply_notice(self, notice: session_pb2.SessionNotice) -> None:
        if notice.kind == session_pb2.SESSION_NOTICE_KIND_CAPABILITIES_CHANGED:
            self._cap_gate.update(notice.capabilities)
        elif notice.kind == session_pb2.SESSION_NOTICE_KIND_DRAINING:
            self._healthy = False
        elif notice.kind == session_pb2.SESSION_NOTICE_KIND_ENGINE_DEAD:
            self._mark_engine_dead(
                EngineTransportError(notice.message or "Engine reported ENGINE_DEAD")
            )

    def _mark_engine_dead(self, exc: BaseException) -> None:
        self._healthy = False
        self._engine_dead_error = exc
        for stream in list(self._inflight.values()):
            stream.fail(exc)
        self._inflight.clear()

    def _enqueue_stat(self, stat: stats_pb2.IterationStats) -> None:
        as_dict = MessageToDict(stat, preserving_proto_field_name=True)
        if self._stats_queue.full():
            try:
                self._stats_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._stats_queue.put_nowait(as_dict)

    async def _send_progress(self) -> None:
        if not self._connected or self._consumed_frame_seq <= self._last_reported_frame_seq:
            return
        seq = self._consumed_frame_seq
        self._last_reported_frame_seq = seq
        self._last_progress_time = time.monotonic()
        await self._transport.send_client_message(
            session_pb2.ClientMessage(progress=session_pb2.Progress(consumed_frame_seq=seq))
        )

    async def _progress_loop(self) -> None:
        interval_ms = int(self._limits.progress_interval_ms or 250)
        sleep_s = max(0.05, interval_ms / 1000.0)
        try:
            while self._connected:
                await asyncio.sleep(sleep_s)
                if (
                    self._consumed_frame_seq > self._last_reported_frame_seq
                    and (time.monotonic() - self._last_progress_time) >= sleep_s
                ):
                    await self._send_progress()
        except asyncio.CancelledError:
            raise

    # ------------------------------------------------------------------
    # ControlService RPC wrappers (13 RPCs)
    # ------------------------------------------------------------------

    async def get_info(self) -> control_pb2.GetInfoResponse:
        return await self._transport.call_control(
            ControlMethod.GET_INFO, control_pb2.GetInfoRequest()
        )

    async def acheck_health(self, *, probe: bool = False) -> control_pb2.HealthResponse:
        return await self._transport.call_control(
            ControlMethod.HEALTH, control_pb2.HealthRequest(probe=probe)
        )

    def check_health(self) -> bool:
        return self._connected and self._healthy and self._engine_dead_error is None

    def _check_health(self) -> bool:
        return self.check_health()

    async def get_load(self) -> control_pb2.LoadInfo:
        return await self._transport.call_control(
            ControlMethod.GET_LOAD, control_pb2.GetLoadRequest()
        )

    async def get_kv_capacity(self) -> control_pb2.KvCapacityResponse:
        return await self._transport.call_control(
            ControlMethod.GET_KV_CAPACITY, control_pb2.GetKvCapacityRequest()
        )

    async def get_iteration_stats(
        self, *, since_iter: Optional[int] = None, max_count: int = 0
    ) -> list[dict[str, Any]]:
        req = control_pb2.GetIterationStatsRequest(max_count=max_count)
        if since_iter is not None:
            req.since_iter = since_iter
        resp: control_pb2.GetIterationStatsResponse = await self._transport.call_control(
            ControlMethod.GET_ITERATION_STATS, req
        )
        return [MessageToDict(s, preserving_proto_field_name=True) for s in resp.stats]

    def get_stats(self, timeout: Optional[float] = 2) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        while not self._stats_queue.empty():
            items.append(self._stats_queue.get_nowait())
        return items

    async def get_stats_async(
        self, timeout: Optional[float] = None
    ) -> AsyncIterator[dict[str, Any]]:
        while self._connected:
            try:
                coro = self._stats_queue.get()
                stat = await (asyncio.wait_for(coro, timeout=timeout) if timeout else coro)
                yield stat
            except asyncio.TimeoutError:
                return

    async def get_kv_events(self, *, max_count: int = 0) -> list[dict[str, Any]]:
        resp: stats_pb2.KvEventBatch = await self._transport.call_control(
            ControlMethod.GET_KV_EVENTS,
            control_pb2.GetKvEventsRequest(max_count=max_count),
        )
        return [MessageToDict(ev, preserving_proto_field_name=True) for ev in resp.events]

    async def get_kv_cache_events_async(
        self, timeout: Optional[float] = 2
    ) -> AsyncIterator[dict[str, Any]]:
        events = await self.get_kv_events()
        for ev in events:
            yield ev

    async def release_handoffs(
        self, keys: list[tuple[str, int]]
    ) -> control_pb2.ReleaseHandoffsResponse:
        req = control_pb2.ReleaseHandoffsRequest(
            items=[
                disagg_pb2.HandoffKey(ctx_instance_id=inst, disagg_request_id=rid)
                for inst, rid in keys
            ]
        )
        return await self._transport.call_control(ControlMethod.RELEASE_HANDOFFS, req)

    async def aget_data_transceiver_state(self) -> bytes:
        resp: control_pb2.GetTransceiverStateResponse = await self._transport.call_control(
            ControlMethod.GET_TRANSCEIVER_STATE, control_pb2.GetTransceiverStateRequest()
        )
        return bytes(resp.state)

    def get_data_transceiver_state(self) -> Optional[bytes]:
        return None

    async def lora_load(
        self, adapter_id: int, name: str, path: str, fmt: str = "hf"
    ) -> control_pb2.LoraLoadResponse:
        lora_fmt = (
            common_pb2.LORA_FORMAT_NEMO if fmt.lower() == "nemo" else common_pb2.LORA_FORMAT_HF
        )
        return await self._transport.call_control(
            ControlMethod.LORA_LOAD,
            control_pb2.LoraLoadRequest(
                adapter=common_pb2.LoraAdapterInfo(
                    adapter_id=adapter_id, name=name, path=path, format=lora_fmt
                )
            ),
        )

    async def lora_unload(self, adapter_id: int) -> control_pb2.LoraUnloadResponse:
        return await self._transport.call_control(
            ControlMethod.LORA_UNLOAD,
            control_pb2.LoraUnloadRequest(adapter_id=adapter_id),
        )

    async def lora_list(self) -> control_pb2.LoraListResponse:
        return await self._transport.call_control(
            ControlMethod.LORA_LIST, control_pb2.LoraListRequest()
        )

    async def astart_profile(
        self,
        output_dir: Optional[str] = None,
        num_steps: Optional[int] = None,
        start_step: int = 0,
        activities: Optional[list[str]] = None,
    ) -> control_pb2.AdminResponse:
        cmd = control_pb2.StartProfileCommand(start_step=start_step)
        if output_dir is not None:
            cmd.output_dir = output_dir
        if num_steps is not None:
            cmd.num_steps = num_steps
        if activities:
            cmd.activities.extend(activities)
        return await self._transport.call_control(
            ControlMethod.ADMIN, control_pb2.AdminRequest(start_profile=cmd)
        )

    async def astop_profile(self) -> control_pb2.AdminResponse:
        return await self._transport.call_control(
            ControlMethod.ADMIN,
            control_pb2.AdminRequest(stop_profile=control_pb2.StopProfileCommand()),
        )

    def start_profile(
        self,
        output_dir: Optional[str] = None,
        num_steps: Optional[int] = None,
        start_step: int = 0,
        activities: Optional[list[str]] = None,
    ) -> None:
        if self._loop is None:
            raise EngineTransportError("ProtoEngineFacade is not started")
        asyncio.run_coroutine_threadsafe(
            self.astart_profile(
                output_dir=output_dir,
                num_steps=num_steps,
                start_step=start_step,
                activities=activities,
            ),
            self._loop,
        ).result(timeout=30.0)

    def stop_profile(self) -> None:
        if self._loop is None:
            raise EngineTransportError("ProtoEngineFacade is not started")
        asyncio.run_coroutine_threadsafe(self.astop_profile(), self._loop).result(timeout=30.0)

    async def collective_rpc(
        self,
        method: str,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> control_pb2.AdminResponse:
        if method == "sleep":
            tags = list(args[0]) if args and args[0] else []
            return await self._transport.call_control(
                ControlMethod.ADMIN,
                control_pb2.AdminRequest(sleep=control_pb2.SleepCommand(tags=tags)),
            )
        if method == "wakeup":
            tags = list(args[0]) if args and args[0] else []
            return await self._transport.call_control(
                ControlMethod.ADMIN,
                control_pb2.AdminRequest(wakeup=control_pb2.WakeupCommand(tags=tags)),
            )
        if method == "update_weights":
            finalize = bool((kwargs or {}).get("finalize", True))
            return await self._transport.call_control(
                ControlMethod.ADMIN,
                control_pb2.AdminRequest(
                    update_weights=control_pb2.UpdateWeightsCommand(finalize=finalize)
                ),
            )
        raise EngineProtocolError(
            f"Unsupported admin collective_rpc method: {method!r}",
            code=common_pb2.ERROR_CODE_UNSUPPORTED,
        )

    async def aclose(self) -> None:
        if not self._connected:
            await self._transport.aclose()
            return
        self._connected = False
        self._healthy = False
        for task in (self._recv_task, self._progress_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._recv_task = None
        self._progress_task = None
        self._postproc_pool.close()
        await self._transport.aclose()

    def shutdown(self) -> None:
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.aclose(), self._loop)


__all__ = [
    "ProtoCompletionOutput",
    "ProtoEngineFacade",
    "ProtoGenerationStream",
]
