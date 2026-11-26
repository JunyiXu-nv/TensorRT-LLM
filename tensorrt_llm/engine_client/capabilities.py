# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend capability cache, ``ParamDomain`` admission gate, and LoRA tracking."""

from __future__ import annotations

import math
from typing import Any, Optional

from tensorrt_llm.contracts.serve_engine.v1 import capabilities_pb2, common_pb2, request_pb2

from .interface import EngineCapabilityError


def _check_float_domain(
    value: float,
    domain: capabilities_pb2.FloatParamDomain,
    field_name: str,
) -> None:
    if not math.isfinite(value):
        raise EngineCapabilityError(
            f"{field_name} must be finite, got {value!r}",
            field=field_name,
        )
    kind = domain.kind
    if kind == capabilities_pb2.PARAM_DOMAIN_KIND_ANY:
        return
    if kind == capabilities_pb2.PARAM_DOMAIN_KIND_UNSUPPORTED:
        raise EngineCapabilityError(
            f"{field_name} is not supported by this engine",
            code=common_pb2.ERROR_CODE_UNSUPPORTED,
            field=field_name,
        )
    if kind == capabilities_pb2.PARAM_DOMAIN_KIND_NO_EFFECT_ONLY:
        if not math.isclose(value, domain.default_value, rel_tol=1e-6, abs_tol=1e-9):
            raise EngineCapabilityError(
                f"{field_name}={value} is unsupported; only default {domain.default_value} is allowed",
                code=common_pb2.ERROR_CODE_UNSUPPORTED,
                field=field_name,
            )
        return
    if kind == capabilities_pb2.PARAM_DOMAIN_KIND_RANGE:
        if domain.HasField("min_value"):
            if domain.min_exclusive and value <= domain.min_value:
                raise EngineCapabilityError(
                    f"{field_name}={value} must be > {domain.min_value}",
                    field=field_name,
                )
            if not domain.min_exclusive and value < domain.min_value:
                raise EngineCapabilityError(
                    f"{field_name}={value} must be >= {domain.min_value}",
                    field=field_name,
                )
        if domain.HasField("max_value"):
            if domain.max_exclusive and value >= domain.max_value:
                raise EngineCapabilityError(
                    f"{field_name}={value} must be < {domain.max_value}",
                    field=field_name,
                )
            if not domain.max_exclusive and value > domain.max_value:
                raise EngineCapabilityError(
                    f"{field_name}={value} must be <= {domain.max_value}",
                    field=field_name,
                )
        return
    if kind == capabilities_pb2.PARAM_DOMAIN_KIND_VALUES:
        if not any(math.isclose(value, v, rel_tol=1e-6, abs_tol=1e-9) for v in domain.values):
            raise EngineCapabilityError(
                f"{field_name}={value} not in allowed values {list(domain.values)}",
                field=field_name,
            )


def _check_int_domain(
    value: int,
    domain: Any,
    field_name: str,
) -> None:
    kind = domain.kind
    if kind == capabilities_pb2.PARAM_DOMAIN_KIND_ANY:
        return
    if kind == capabilities_pb2.PARAM_DOMAIN_KIND_UNSUPPORTED:
        raise EngineCapabilityError(
            f"{field_name} is not supported by this engine",
            code=common_pb2.ERROR_CODE_UNSUPPORTED,
            field=field_name,
        )
    if kind == capabilities_pb2.PARAM_DOMAIN_KIND_NO_EFFECT_ONLY:
        if value != domain.default_value:
            raise EngineCapabilityError(
                f"{field_name}={value} is unsupported; only default {domain.default_value} is allowed",
                code=common_pb2.ERROR_CODE_UNSUPPORTED,
                field=field_name,
            )
        return
    if kind == capabilities_pb2.PARAM_DOMAIN_KIND_RANGE:
        if domain.HasField("min_value") and value < domain.min_value:
            raise EngineCapabilityError(
                f"{field_name}={value} must be >= {domain.min_value}",
                field=field_name,
            )
        if domain.HasField("max_value") and value > domain.max_value:
            raise EngineCapabilityError(
                f"{field_name}={value} must be <= {domain.max_value}",
                field=field_name,
            )
        return
    if kind == capabilities_pb2.PARAM_DOMAIN_KIND_VALUES:
        if value not in domain.values:
            raise EngineCapabilityError(
                f"{field_name}={value} not in allowed values {list(domain.values)}",
                field=field_name,
            )


class CapabilityGate:
    """Caches engine ``Capabilities`` and performs fast pre-submit admission checks."""

    def __init__(self, capabilities: Optional[capabilities_pb2.Capabilities] = None) -> None:
        self._caps = capabilities_pb2.Capabilities()
        self._loaded_loras: dict[int, common_pb2.LoraAdapterInfo] = {}
        if capabilities is not None:
            self.update(capabilities)

    @property
    def capabilities(self) -> capabilities_pb2.Capabilities:
        return self._caps

    @property
    def loaded_loras(self) -> dict[int, common_pb2.LoraAdapterInfo]:
        return dict(self._loaded_loras)

    def update(self, capabilities: capabilities_pb2.Capabilities) -> None:
        self._caps = capabilities_pb2.Capabilities()
        self._caps.CopyFrom(capabilities)
        self._loaded_loras = {item.adapter_id: item for item in self._caps.lora.loaded}

    def apply_lora_delta(self, delta: common_pb2.LoraSetChanged) -> None:
        for adapter in delta.added:
            self._loaded_loras[adapter.adapter_id] = adapter
        for adapter_id in delta.removed:
            self._loaded_loras.pop(adapter_id, None)

    def verify_tokenizer(self, tokenizer: Any) -> None:
        """Check local frontend tokenizer against ``Capabilities.tokenizer`` when populated."""
        fp = self._caps.tokenizer
        if not fp.vocab_size:
            return
        local_vocab = getattr(tokenizer, "vocab_size", None)
        if local_vocab is None and hasattr(tokenizer, "tokenizer"):
            local_vocab = getattr(tokenizer.tokenizer, "vocab_size", None)
        if local_vocab is not None and int(local_vocab) != int(fp.vocab_size):
            raise EngineCapabilityError(
                f"Frontend tokenizer vocab_size ({local_vocab}) does not match engine "
                f"TokenizerFingerprint ({fp.vocab_size})",
                code=common_pb2.ERROR_CODE_INVALID_ARGUMENT,
                field="tokenizer",
            )

    def validate_request(self, req: request_pb2.Request) -> None:
        """Validate ``Request`` against cached ``Capabilities`` (CAP-01..CAP-21)."""
        caps = self._caps
        if not req.prompt_tokens and not req.encoder_input_tokens:
            raise EngineCapabilityError(
                "prompt_tokens must be non-empty",
                field="prompt_tokens",
            )

        if caps.model.max_seq_len and len(req.prompt_tokens) > caps.model.max_seq_len:
            raise EngineCapabilityError(
                f"Prompt length {len(req.prompt_tokens)} exceeds engine max_seq_len "
                f"{caps.model.max_seq_len}",
                code=common_pb2.ERROR_CODE_CONTEXT_TOO_LONG,
                field="prompt_tokens",
            )

        self._validate_phase_and_disagg(req)
        self._validate_sampling(req)
        self._validate_stop(req)
        self._validate_logprobs_and_outputs(req)
        self._validate_processors_guided_lora(req)

    def _validate_phase_and_disagg(self, req: request_pb2.Request) -> None:
        disagg_caps = self._caps.disagg
        if disagg_caps.phases and req.phase not in disagg_caps.phases:
            raise EngineCapabilityError(
                f"Requested phase {common_pb2.Phase.Name(req.phase)} is not supported by engine",
                code=common_pb2.ERROR_CODE_PHASE_NOT_SUPPORTED,
                field="phase",
            )
        if req.phase == common_pb2.PHASE_CONTEXT_ONLY:
            if not req.HasField("disagg") or req.disagg.schedule_style == 0:
                raise EngineCapabilityError(
                    "CONTEXT_ONLY requests must specify disagg.schedule_style",
                    code=common_pb2.ERROR_CODE_INVALID_ARGUMENT,
                    field="disagg.schedule_style",
                )
            if (
                disagg_caps.schedule_styles
                and req.disagg.schedule_style not in disagg_caps.schedule_styles
            ):
                raise EngineCapabilityError(
                    f"Unsupported disagg schedule_style={req.disagg.schedule_style}",
                    code=common_pb2.ERROR_CODE_UNSUPPORTED_IN_DISAGG,
                    field="disagg.schedule_style",
                )

    def _validate_sampling(self, req: request_pb2.Request) -> None:
        if not req.HasField("sampling"):
            return
        sp = req.sampling
        sc = self._caps.sampling
        if sp.num_sequences and sc.max_num_sequences:
            if sp.num_sequences > sc.max_num_sequences:
                raise EngineCapabilityError(
                    f"sampling.num_sequences={sp.num_sequences} exceeds max {sc.max_num_sequences}",
                    field="sampling.num_sequences",
                )
        if sp.beam_width and self._caps.beam.max_beam_width:
            if sp.beam_width > self._caps.beam.max_beam_width:
                raise EngineCapabilityError(
                    f"sampling.beam_width={sp.beam_width} exceeds max {self._caps.beam.max_beam_width}",
                    field="sampling.beam_width",
                )
        if sp.HasField("temperature") and sc.HasField("temperature"):
            _check_float_domain(sp.temperature, sc.temperature, "sampling.temperature")
        if sp.HasField("top_p") and sc.HasField("top_p"):
            _check_float_domain(sp.top_p, sc.top_p, "sampling.top_p")
        if sp.HasField("min_p") and sc.HasField("min_p"):
            _check_float_domain(sp.min_p, sc.min_p, "sampling.min_p")
        if sp.HasField("top_k") and sc.HasField("top_k"):
            _check_int_domain(sp.top_k, sc.top_k, "sampling.top_k")
        if sp.HasField("repetition_penalty") and sc.HasField("repetition_penalty"):
            _check_float_domain(
                sp.repetition_penalty, sc.repetition_penalty, "sampling.repetition_penalty"
            )
        if sp.HasField("presence_penalty") and sc.HasField("presence_penalty"):
            _check_float_domain(
                sp.presence_penalty, sc.presence_penalty, "sampling.presence_penalty"
            )
        if sp.HasField("frequency_penalty") and sc.HasField("frequency_penalty"):
            _check_float_domain(
                sp.frequency_penalty, sc.frequency_penalty, "sampling.frequency_penalty"
            )
        if sp.logit_bias and sc.HasField("logit_bias"):
            if not sc.logit_bias.supported:
                raise EngineCapabilityError(
                    "sampling.logit_bias is not supported by this engine",
                    code=common_pb2.ERROR_CODE_UNSUPPORTED,
                    field="sampling.logit_bias",
                )
            if sc.logit_bias.max_entries and len(sp.logit_bias) > sc.logit_bias.max_entries:
                raise EngineCapabilityError(
                    f"sampling.logit_bias has {len(sp.logit_bias)} entries; max is {sc.logit_bias.max_entries}",
                    field="sampling.logit_bias",
                )

    def _validate_stop(self, req: request_pb2.Request) -> None:
        if not req.HasField("stop"):
            return
        st = req.stop
        sc = self._caps.stop
        if sc.max_sequences and len(st.sequences) > sc.max_sequences:
            raise EngineCapabilityError(
                f"stop.sequences count {len(st.sequences)} exceeds max {sc.max_sequences}",
                field="stop.sequences",
            )
        if sc.max_bad_sequences and len(st.bad_sequences) > sc.max_bad_sequences:
            raise EngineCapabilityError(
                f"stop.bad_sequences count {len(st.bad_sequences)} exceeds max {sc.max_bad_sequences}",
                field="stop.bad_sequences",
            )

    def _validate_logprobs_and_outputs(self, req: request_pb2.Request) -> None:
        if not req.HasField("output"):
            return
        out = req.output
        lc = self._caps.logprobs
        if out.HasField("logprobs") and lc.max_top_k and out.logprobs > lc.max_top_k:
            raise EngineCapabilityError(
                f"output.logprobs={out.logprobs} exceeds max_top_k={lc.max_top_k}",
                field="output.logprobs",
            )
        if out.HasField("prompt_logprobs") and lc.HasField("prompt"):
            if not lc.prompt.supported:
                raise EngineCapabilityError(
                    "output.prompt_logprobs is not supported by this engine",
                    code=common_pb2.ERROR_CODE_UNSUPPORTED,
                    field="output.prompt_logprobs",
                )
            if lc.prompt.max_top_k and out.prompt_logprobs > lc.prompt.max_top_k:
                raise EngineCapabilityError(
                    f"output.prompt_logprobs={out.prompt_logprobs} exceeds max_top_k={lc.prompt.max_top_k}",
                    field="output.prompt_logprobs",
                )

    def _validate_processors_guided_lora(self, req: request_pb2.Request) -> None:
        if req.HasField("thinking_budget") and self._caps.HasField("proc"):
            if not self._caps.proc.thinking_budget:
                raise EngineCapabilityError(
                    "thinking_budget is not supported by this engine",
                    code=common_pb2.ERROR_CODE_UNSUPPORTED,
                    field="thinking_budget",
                )
        if req.logits_processors and self._caps.HasField("proc"):
            registered = set(self._caps.proc.registered_names)
            for name in req.logits_processors:
                if name not in registered:
                    raise EngineCapabilityError(
                        f"logits_processor {name!r} is not registered on this engine",
                        code=common_pb2.ERROR_CODE_UNSUPPORTED,
                        field="logits_processors",
                    )
        if req.HasField("guided") and self._caps.HasField("guided"):
            allowed = set(self._caps.guided.kinds)
            if allowed:
                which = req.guided.WhichOneof("constraint")
                kind_map = {
                    "json_schema": capabilities_pb2.GUIDED_DECODING_KIND_JSON_SCHEMA,
                    "json_object": capabilities_pb2.GUIDED_DECODING_KIND_JSON_OBJECT,
                    "regex": capabilities_pb2.GUIDED_DECODING_KIND_REGEX,
                    "ebnf_grammar": capabilities_pb2.GUIDED_DECODING_KIND_EBNF_GRAMMAR,
                    "structural_tag": capabilities_pb2.GUIDED_DECODING_KIND_STRUCTURAL_TAG,
                    "choice": capabilities_pb2.GUIDED_DECODING_KIND_CHOICE,
                }
                kind = kind_map.get(which or "")
                if kind is not None and kind not in allowed:
                    raise EngineCapabilityError(
                        f"guided.{which} is not supported by this engine",
                        code=common_pb2.ERROR_CODE_UNSUPPORTED,
                        field=f"guided.{which}",
                    )
        if req.HasField("lora") and self._caps.HasField("lora"):
            if not self._caps.lora.supported:
                raise EngineCapabilityError(
                    "LoRA requests are not supported by this engine",
                    code=common_pb2.ERROR_CODE_UNSUPPORTED,
                    field="lora",
                )
            if not req.lora.path and req.lora.adapter_id not in self._loaded_loras:
                raise EngineCapabilityError(
                    f"LoRA adapter_id={req.lora.adapter_id} is not loaded and no path was provided",
                    code=common_pb2.ERROR_CODE_LORA_NOT_LOADED,
                    field="lora",
                )


__all__ = ["CapabilityGate"]
