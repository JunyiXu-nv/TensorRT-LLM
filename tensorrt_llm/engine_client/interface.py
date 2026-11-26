# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol-facing client interface and typed errors for ``tensorrt_llm.engine_client``."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Optional, Protocol, runtime_checkable

from tensorrt_llm.contracts.serve_engine.v1 import common_pb2, control_pb2

_ERROR_CODE_TO_HTTP: dict[int, int] = {
    common_pb2.ERROR_CODE_INVALID_ARGUMENT: 400,
    common_pb2.ERROR_CODE_UNSUPPORTED: 400,
    common_pb2.ERROR_CODE_CONTEXT_TOO_LONG: 400,
    common_pb2.ERROR_CODE_DUPLICATE_HANDLE: 409,
    common_pb2.ERROR_CODE_OVERLOADED: 503,
    common_pb2.ERROR_CODE_DRAINING: 503,
    common_pb2.ERROR_CODE_LORA_NOT_LOADED: 404,
    common_pb2.ERROR_CODE_LORA_CONFLICT: 409,
    common_pb2.ERROR_CODE_LORA_LOAD_FAILED: 400,
    common_pb2.ERROR_CODE_ENGINE_DEAD: 503,
    common_pb2.ERROR_CODE_SLOW_CONSUMER: 503,
    common_pb2.ERROR_CODE_BLOB_UNAVAILABLE: 410,
    common_pb2.ERROR_CODE_INTERNAL: 500,
    common_pb2.ERROR_CODE_DISAGG_UNAVAILABLE: 503,
    common_pb2.ERROR_CODE_PHASE_NOT_SUPPORTED: 400,
    common_pb2.ERROR_CODE_HANDOFF_INVALID: 400,
    common_pb2.ERROR_CODE_UNSUPPORTED_IN_DISAGG: 400,
    common_pb2.ERROR_CODE_ALREADY_EXISTS: 409,
    common_pb2.ERROR_CODE_PEER_INCOMPATIBLE: 400,
    common_pb2.ERROR_CODE_HANDOFF_NOT_FOUND: 404,
    common_pb2.ERROR_CODE_PEER_UNREACHABLE: 502,
    common_pb2.ERROR_CODE_KV_TRANSFER_FAILED: 502,
    common_pb2.ERROR_CODE_KV_TRANSFER_TIMEOUT: 504,
}


def http_status_for_error_code(code: int) -> int:
    """Map a ``trtllm.serve_engine.v1.ErrorCode`` enum value to an HTTP status code."""
    return _ERROR_CODE_TO_HTTP.get(code, 500)


class EngineProtocolError(RuntimeError):
    """Typed error surfaced by the serve-engine protocol boundary."""

    def __init__(
        self,
        message: str,
        *,
        code: int = common_pb2.ERROR_CODE_INTERNAL,
        field: str = "",
        engine_fatal: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.field = field
        self.engine_fatal = engine_fatal
        self.http_status = http_status_for_error_code(code)

    @classmethod
    def from_proto(cls, info: common_pb2.ErrorInfo) -> "EngineProtocolError":
        return cls(
            info.message or common_pb2.ErrorCode.Name(info.code),
            code=info.code,
            field=info.field,
            engine_fatal=info.engine_fatal,
        )


class EngineCapabilityError(EngineProtocolError):
    """Raised synchronously before submission when a request violates ``Capabilities``."""

    def __init__(
        self,
        message: str,
        *,
        code: int = common_pb2.ERROR_CODE_INVALID_ARGUMENT,
        field: str = "",
    ) -> None:
        super().__init__(message, code=code, field=field, engine_fatal=False)


class EngineTransportError(EngineProtocolError):
    """Raised when the underlying ZMQ or gRPC transport fails or disconnects."""

    def __init__(self, message: str, *, engine_fatal: bool = True) -> None:
        super().__init__(
            message,
            code=common_pb2.ERROR_CODE_ENGINE_DEAD,
            engine_fatal=engine_fatal,
        )


@runtime_checkable
class EngineClientInterface(Protocol):
    """Narrow async interface consumed by ``tensorrt_llm.serve`` frontends.

    Satisfied by ``ProtoEngineFacade`` (speaking ``trtllm.serve_engine.v1`` over ZMQ
    or gRPC) and kept duck-type compatible with the legacy ``LLM`` serving surface so
    ``OpenAIServer`` can switch between them via an opt-in flag.
    """

    tokenizer: Any
    args: Any

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
    ) -> Any: ...

    def check_health(self) -> bool: ...

    async def acheck_health(self, *, probe: bool = False) -> control_pb2.HealthResponse: ...

    def get_stats(self, timeout: Optional[float] = 2) -> list[dict[str, Any]]: ...

    def get_stats_async(self, timeout: Optional[float] = None) -> AsyncIterator[dict[str, Any]]: ...

    def get_kv_cache_events_async(
        self, timeout: Optional[float] = 2
    ) -> AsyncIterator[dict[str, Any]]: ...

    def get_data_transceiver_state(self) -> Optional[bytes]: ...

    def start_profile(
        self,
        output_dir: Optional[str] = None,
        num_steps: Optional[int] = None,
        start_step: int = 0,
        activities: Optional[list[str]] = None,
    ) -> None: ...

    def stop_profile(self) -> None: ...

    async def collective_rpc(
        self,
        method: str,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> Any: ...

    def shutdown(self) -> None: ...


__all__ = [
    "EngineCapabilityError",
    "EngineClientInterface",
    "EngineProtocolError",
    "EngineTransportError",
    "http_status_for_error_code",
]
