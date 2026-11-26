# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable transport interface for ``trtllm.serve_engine.v1``."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from tensorrt_llm.contracts.serve_engine.v1 import session_pb2
from tensorrt_llm.contracts.serve_engine.v1.zmq_wire import (
    CONTROL_REQUEST_TYPES,
    CONTROL_RESPONSE_TYPES,
    ControlMethod,
)


@runtime_checkable
class EngineTransport(Protocol):
    """Transport abstraction separating protocol logic from ZMQ (Phase 1) or gRPC (Phase 2)."""

    async def open_session(
        self, open_msg: session_pb2.SessionOpen
    ) -> session_pb2.SessionAccepted: ...

    async def send_client_message(
        self,
        msg: session_pb2.ClientMessage,
        aux_buffers: Sequence[bytes] = (),
    ) -> None: ...

    async def recv_server_message(
        self,
        timeout: float | None = None,
    ) -> tuple[session_pb2.ServerMessage, list[bytes]]: ...

    async def call_control(
        self,
        method: ControlMethod,
        request: Any,
        *,
        timeout: float = 10.0,
    ) -> Any: ...

    async def aclose(self) -> None: ...


__all__ = [
    "CONTROL_REQUEST_TYPES",
    "CONTROL_RESPONSE_TYPES",
    "ControlMethod",
    "EngineTransport",
]
