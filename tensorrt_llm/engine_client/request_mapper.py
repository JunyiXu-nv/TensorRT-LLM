# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map serving / LLM API requests into ``trtllm.serve_engine.v1.Request`` messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from tensorrt_llm.contracts.serve_engine.v1 import common_pb2, disagg_pb2, request_pb2

from .frame_decoder import encode_tensor_ref
from .interface import EngineCapabilityError

_PHASE_MAP: dict[str, int] = {
    "aggregated": common_pb2.PHASE_AGGREGATED,
    "context_only": common_pb2.PHASE_CONTEXT_ONLY,
    "generation_only": common_pb2.PHASE_GENERATION_ONLY,
}

_SCHEDULE_STYLE_MAP: dict[str, int] = {
    "context_first": common_pb2.SCHEDULE_STYLE_CONTEXT_FIRST,
    "generation_first": common_pb2.SCHEDULE_STYLE_GENERATION_FIRST,
}


def _to_token_list(tokens: Any) -> list[int]:
    if tokens is None:
        return []
    if isinstance(tokens, np.ndarray):
        return tokens.astype(np.int32).ravel().tolist()
    if hasattr(tokens, "detach") and hasattr(tokens, "cpu"):
        return tokens.detach().cpu().numpy().astype(np.int32).ravel().tolist()
    return [int(t) for t in tokens]


def _extract_stop_token_sequences(sampling_params: Any) -> list[list[int]]:
    seqs = getattr(sampling_params, "_stop_word_ids", None)
    if seqs:
        return [[int(t) for t in s] for s in seqs if s]
    raw_ids = getattr(sampling_params, "stop_token_ids", None)
    if raw_ids:
        return [[int(t)] for t in raw_ids]
    return []


def _extract_bad_token_sequences(sampling_params: Any) -> list[list[int]]:
    seqs = getattr(sampling_params, "_bad_word_ids", None)
    if seqs:
        return [[int(t) for t in s] for s in seqs if s]
    return []


def _extract_sparse_logit_bias(sampling_params: Any) -> list[tuple[int, float]]:
    """Extract sparse ``(token_id, bias)`` pairs without constructing a dense vocab tensor."""
    sparse = getattr(sampling_params, "logit_bias", None)
    if isinstance(sparse, Mapping):
        return [(int(k), float(v)) for k, v in sparse.items() if float(v) != 0.0]
    eb = getattr(sampling_params, "embedding_bias", None)
    if eb is None:
        return []
    if hasattr(eb, "detach") and hasattr(eb, "cpu"):
        arr = eb.detach().cpu().numpy().ravel()
    else:
        arr = np.asarray(eb).ravel()
    nz = np.nonzero(arr)[0]
    return [(int(idx), float(arr[idx])) for idx in nz]


class RequestMapper:
    """Translates high-level serving / LLM inputs into ``request_pb2.Request``."""

    def __init__(
        self,
        *,
        tokenizer: Any = None,
        default_end_id: Optional[int] = None,
        default_pad_id: Optional[int] = None,
        max_inline_tensor_bytes: int = 65536,
    ) -> None:
        self._tokenizer = tokenizer
        self._default_end_id = default_end_id
        self._default_pad_id = default_pad_id
        self._max_inline_tensor_bytes = max_inline_tensor_bytes

    def build_request(
        self,
        *,
        handle: int,
        request_id: str,
        prompt_token_ids: Any,
        sampling_params: Any = None,
        streaming: bool = False,
        lora_request: Any = None,
        prompt_adapter_request: Any = None,
        kv_cache_retention_config: Any = None,
        disaggregated_params: Any = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        multimodal_params: Any = None,
        scheduling_params: Any = None,
        conversation_params: Any = None,
        cache_salt: Optional[str] = None,
        encoder_input_token_ids: Any = None,
        priority: Optional[float] = None,
        kv_hints: Any = None,
        queue_age_us: Optional[int] = None,
    ) -> tuple[request_pb2.Request, list[bytes]]:
        if prompt_adapter_request is not None:
            raise EngineCapabilityError(
                "prompt_adapter_request is deprecated and not carried by trtllm.serve_engine.v1",
                code=common_pb2.ERROR_CODE_UNSUPPORTED,
                field="prompt_adapter_request",
            )

        aux_buffers: list[bytes] = []
        req = request_pb2.Request(
            handle=handle,
            request_id=request_id,
            prompt_tokens=_to_token_list(prompt_token_ids),
            encoder_input_tokens=_to_token_list(encoder_input_token_ids),
            phase=common_pb2.PHASE_AGGREGATED,
            streaming=bool(streaming),
        )
        if queue_age_us is not None:
            req.queue_age_us = int(queue_age_us)

        if sampling_params is not None:
            self._populate_sampling_and_stop(req, sampling_params)
            self._populate_outputs(req, sampling_params)
            self._populate_guided(req, sampling_params)
            self._populate_processors(req, sampling_params)
        else:
            req.stop.end_id = self._resolve_end_id(None)

        if lora_request is not None:
            req.lora.adapter_id = int(getattr(lora_request, "lora_int_id", 0))
            req.lora.name = str(getattr(lora_request, "lora_name", ""))
            req.lora.path = str(getattr(lora_request, "lora_path", ""))
            req.lora.format = str(getattr(lora_request, "lora_ckpt_source", "hf"))

        self._populate_scheduling(req, scheduling_params, conversation_params, priority)
        self._populate_kv(req, cache_salt, kv_cache_retention_config, kv_hints)
        self._populate_trace(req, trace_headers)
        self._populate_disagg(req, disaggregated_params)
        self._populate_multimodal(req, multimodal_params, aux_buffers)

        return req, aux_buffers

    def _resolve_end_id(self, sampling_params: Any) -> int:
        if sampling_params is not None and getattr(sampling_params, "end_id", None) is not None:
            return int(sampling_params.end_id)
        if self._default_end_id is not None:
            return int(self._default_end_id)
        if (
            self._tokenizer is not None
            and getattr(self._tokenizer, "eos_token_id", None) is not None
        ):
            return int(self._tokenizer.eos_token_id)
        return 0

    def _populate_sampling_and_stop(self, req: request_pb2.Request, sp: Any) -> None:
        pb = req.sampling
        if getattr(sp, "n", None) is not None:
            pb.num_sequences = int(sp.n)
        if getattr(sp, "best_of", None) is not None and getattr(sp, "use_beam_search", False):
            pb.beam_width = int(sp.best_of)
            pb.num_return_beams = int(getattr(sp, "n", sp.best_of))
        if getattr(sp, "temperature", None) is not None:
            pb.temperature = float(sp.temperature)
        if getattr(sp, "top_k", None) is not None:
            pb.top_k = int(sp.top_k)
        if getattr(sp, "top_p", None) is not None:
            pb.top_p = float(sp.top_p)
        if getattr(sp, "top_p_decay", None) is not None:
            pb.top_p_decay = float(sp.top_p_decay)
        if getattr(sp, "top_p_min", None) is not None:
            pb.top_p_min = float(sp.top_p_min)
        if getattr(sp, "top_p_reset_ids", None) is not None:
            pb.top_p_reset_ids = int(sp.top_p_reset_ids)
        if getattr(sp, "min_p", None) is not None:
            pb.min_p = float(sp.min_p)
        if getattr(sp, "seed", None) is not None:
            pb.seed = int(sp.seed)
        if getattr(sp, "repetition_penalty", None) is not None:
            pb.repetition_penalty = float(sp.repetition_penalty)
        if getattr(sp, "presence_penalty", None) is not None:
            pb.presence_penalty = float(sp.presence_penalty)
        if getattr(sp, "frequency_penalty", None) is not None:
            pb.frequency_penalty = float(sp.frequency_penalty)
        if getattr(sp, "prompt_ignore_length", None) is not None:
            pb.prompt_ignore_length = int(sp.prompt_ignore_length)
        if getattr(sp, "no_repeat_ngram_size", None) is not None:
            pb.no_repeat_ngram_size = int(sp.no_repeat_ngram_size)
        if getattr(sp, "length_penalty", None) is not None:
            pb.length_penalty = float(sp.length_penalty)
        if getattr(sp, "early_stopping", None) is not None:
            pb.early_stopping = int(sp.early_stopping)
        if getattr(sp, "beam_search_diversity_rate", None) is not None:
            pb.beam_search_diversity_rate = float(sp.beam_search_diversity_rate)

        for token_id, bias in _extract_sparse_logit_bias(sp):
            pb.logit_bias.add(token_id=token_id, bias=bias)

        stop = req.stop
        if getattr(sp, "max_tokens", None) is not None:
            stop.max_tokens = int(sp.max_tokens)
        if getattr(sp, "min_tokens", None) is not None:
            stop.min_tokens = int(sp.min_tokens)
        stop.end_id = self._resolve_end_id(sp)
        pad_id = getattr(sp, "pad_id", None)
        if pad_id is None:
            pad_id = self._default_pad_id
        if pad_id is not None:
            stop.pad_id = int(pad_id)
        stop.ignore_eos = bool(getattr(sp, "ignore_eos", False))
        for seq in _extract_stop_token_sequences(sp):
            stop.sequences.add(tokens=seq)
        for seq in _extract_bad_token_sequences(sp):
            stop.bad_sequences.add(tokens=seq)

    def _populate_outputs(self, req: request_pb2.Request, sp: Any) -> None:
        out = req.output
        if getattr(sp, "logprobs", None) is not None:
            out.logprobs = int(sp.logprobs)
            out.logprob_rank = True
        if getattr(sp, "prompt_logprobs", None) is not None:
            out.prompt_logprobs = int(sp.prompt_logprobs)
        if getattr(sp, "return_context_logits", False):
            out.tensors.context_logits = True
        if getattr(sp, "return_generation_logits", False):
            out.tensors.generation_logits = True
        if getattr(sp, "return_encoder_output", False):
            out.tensors.encoder_output = True
        if getattr(sp, "return_perf_metrics", False):
            out.metrics.perf = True
            out.metrics.decoding_stats = True

    def _populate_guided(self, req: request_pb2.Request, sp: Any) -> None:
        gp = getattr(sp, "guided_decoding", None)
        if gp is None:
            return
        if getattr(gp, "json", None) is not None:
            req.guided.json_schema = str(gp.json)
        elif getattr(gp, "json_object", False):
            req.guided.json_object = True
        elif getattr(gp, "regex", None) is not None:
            req.guided.regex = str(gp.regex)
        elif getattr(gp, "grammar", None) is not None:
            req.guided.ebnf_grammar = str(gp.grammar)
        elif getattr(gp, "structural_tag", None) is not None:
            req.guided.structural_tag = str(gp.structural_tag)
        elif getattr(gp, "choice", None) is not None:
            req.guided.choice.choices.extend(str(c) for c in gp.choice)

    def _populate_processors(self, req: request_pb2.Request, sp: Any) -> None:
        tb = getattr(sp, "thinking_budget", None)
        if tb is not None:
            req.thinking_budget.budget = int(tb)
            for attr, field_name in (
                ("_thinking_start_tokens", "start_tokens"),
                ("_thinking_end_tokens", "end_tokens"),
            ):
                toks = getattr(sp, attr, None)
                if toks:
                    getattr(req.thinking_budget, field_name).extend(int(t) for t in toks)

        lps = getattr(sp, "logits_processor", None)
        if lps is not None:
            items = lps if isinstance(lps, (list, tuple)) else [lps]
            for lp in items:
                if isinstance(lp, str):
                    req.logits_processors.append(lp)
                elif getattr(lp, "__class__", None).__name__ == "ThinkingBudgetLogitsProcessor":
                    if not req.HasField("thinking_budget"):
                        req.thinking_budget.budget = int(getattr(lp, "thinking_budget", 0))
                else:
                    raise EngineCapabilityError(
                        "Arbitrary Python logits_processor callables are not supported over "
                        "trtllm.serve_engine.v1; register a named processor on the engine instead.",
                        code=common_pb2.ERROR_CODE_UNSUPPORTED,
                        field="logits_processors",
                    )

    def _populate_scheduling(
        self,
        req: request_pb2.Request,
        scheduling_params: Any,
        conversation_params: Any,
        priority: Optional[float],
    ) -> None:
        if priority is not None:
            req.sched.priority = float(priority)
        if scheduling_params is not None:
            dp_rank = getattr(scheduling_params, "attention_dp_rank", None)
            if dp_rank is not None:
                req.sched.attention_dp_rank = int(dp_rank)
            dp_relax = getattr(scheduling_params, "attention_dp_relax", None)
            if dp_relax is not None:
                req.sched.attention_dp_relax = bool(dp_relax)
            subagent_id = getattr(scheduling_params, "subagent_affinity_id", None)
            if subagent_id:
                req.sched.subagent_affinity_id = str(subagent_id)
        if conversation_params is not None:
            conv_id = getattr(conversation_params, "conversation_id", None)
            if conv_id:
                req.sched.conversation_id = str(conv_id)

    def _populate_kv(
        self,
        req: request_pb2.Request,
        cache_salt: Optional[str],
        retention: Any,
        kv_hints: Any,
    ) -> None:
        if cache_salt:
            req.kv.cache_salt = str(cache_salt)
        if retention is not None:
            for r in getattr(retention, "token_range_retention_configs", None) or ():
                entry = req.kv.retention.ranges.add(
                    token_start=int(r.token_start),
                    priority=int(r.priority),
                )
                if getattr(r, "token_end", None) is not None:
                    entry.token_end = int(r.token_end)
                if getattr(r, "duration_ms", None) is not None:
                    entry.duration_ms = int(r.duration_ms)
        if kv_hints is not None and isinstance(kv_hints, Mapping):
            req.kv.hints.protocol_version = str(kv_hints.get("protocol_version", "0.1"))
            if "message_id" in kv_hints:
                req.kv.hints.message_id = str(kv_hints["message_id"])

    def _populate_trace(
        self, req: request_pb2.Request, trace_headers: Optional[Mapping[str, str]]
    ) -> None:
        if not trace_headers:
            return
        tp = trace_headers.get("traceparent")
        if tp:
            req.trace.traceparent = str(tp)
        ts = trace_headers.get("tracestate")
        if ts:
            req.trace.tracestate = str(ts)

    def _populate_disagg(self, req: request_pb2.Request, dp: Any) -> None:
        if dp is None:
            return
        req_type = getattr(dp, "request_type", None)
        if req_type in _PHASE_MAP:
            req.phase = _PHASE_MAP[req_type]
        if getattr(dp, "disagg_request_id", None) is not None:
            req.disagg.request_id = int(dp.disagg_request_id)
        style = getattr(dp, "schedule_style", None)
        if style is not None:
            style_str = str(getattr(style, "value", style)).lower()
            if style_str in _SCHEDULE_STYLE_MAP:
                req.disagg.schedule_style = _SCHEDULE_STYLE_MAP[style_str]
        signed_b64 = getattr(dp, "signed_handoff_b64", None)
        if signed_b64:
            import base64

            req.disagg.handoff.ParseFromString(base64.b64decode(signed_b64))
        elif req.phase == common_pb2.PHASE_GENERATION_ONLY:
            # Build an unsigned KvHandoff envelope when migrating from legacy DisaggregatedParams fields.
            handoff = disagg_pb2.KvHandoff(
                disagg_request_id=int(getattr(dp, "disagg_request_id", 0) or 0),
                ctx_engine_request_id=int(getattr(dp, "ctx_request_id", 0) or 0),
                ctx_endpoint=str(getattr(dp, "ctx_info_endpoint", "") or ""),
                first_gen_tokens=[int(t) for t in (getattr(dp, "first_gen_tokens", None) or ())],
                draft_tokens=[int(t) for t in (getattr(dp, "draft_tokens", None) or ())],
            )
            if getattr(dp, "ctx_dp_rank", None) is not None:
                handoff.ctx_dp_rank = int(dp.ctx_dp_rank)
            if getattr(dp, "opaque_state", None):
                handoff.opaque_state = bytes(dp.opaque_state)
            req.disagg.handoff.handoff_bytes = handoff.SerializeToString()

    def _populate_multimodal(
        self,
        req: request_pb2.Request,
        mm_params: Any,
        aux_buffers: list[bytes],
    ) -> None:
        if mm_params is None:
            return
        mm_data = getattr(mm_params, "multimodal_data", None)
        if not isinstance(mm_data, Mapping) or not mm_data:
            return
        req.mm.mode = request_pb2.MULTIMODAL_MODE_PREPROCESSED
        for key, value in mm_data.items():
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                arr = value.detach().cpu().numpy()
                mi = req.mm.model_inputs.add(path=str(key))
                mi.tensor.CopyFrom(
                    encode_tensor_ref(
                        arr,
                        aux_buffers=aux_buffers,
                        max_inline_bytes=self._max_inline_tensor_bytes,
                    )
                )
            elif isinstance(value, np.ndarray):
                mi = req.mm.model_inputs.add(path=str(key))
                mi.tensor.CopyFrom(
                    encode_tensor_ref(
                        value,
                        aux_buffers=aux_buffers,
                        max_inline_bytes=self._max_inline_tensor_bytes,
                    )
                )
            elif isinstance(value, list) and value and all(isinstance(v, int) for v in value):
                mi = req.mm.model_inputs.add(path=str(key))
                mi.i64_list.values.extend(value)
            elif (
                isinstance(value, list)
                and value
                and all(isinstance(v, (int, float)) for v in value)
            ):
                mi = req.mm.model_inputs.add(path=str(key))
                mi.f64_list.values.extend(float(v) for v in value)


__all__ = ["RequestMapper"]
