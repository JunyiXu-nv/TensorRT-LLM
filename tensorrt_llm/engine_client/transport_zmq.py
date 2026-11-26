# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Language-neutral, pickle-free ZMQ transport for ``trtllm.serve_engine.v1``.

Wire format (all multi-part ZMQ frames, zero pickle):
  Session channel (DEALER with frontend_id identity <-> rank0 ROUTER):
    [hmac_sha256_32B, msg_kind_1B, proto_bytes, aux_tensor_0, aux_tensor_1, ...]
      msg_kind = 0x01: ClientMessage (frontend -> engine) / ServerMessage (engine -> frontend)

  Control channel (DEALER <-> rank0 ROUTER):
    [hmac_sha256_32B, method_id_1B, call_id_8B_le, status_1B, proto_bytes]
      request  status_1B = 0x00
      response status_1B = 0x00 (OK) | 0x01 (ErrorInfo proto in proto_bytes)
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Sequence

import zmq
import zmq.asyncio

from tensorrt_llm.contracts.serve_engine.v1 import common_pb2, session_pb2
from tensorrt_llm.contracts.serve_engine.v1.zmq_wire import (
    CONTROL_RESPONSE_TYPES,
    CTRL_STATUS_ERR,
    CTRL_STATUS_OK,
    MSG_KIND_PROTO,
    ControlMethod,
    WireFramingError,
    compute_wire_hmac,
    pack_control_frames,
    pack_session_frames,
    unpack_control_frames,
    unpack_session_frames,
    verify_wire_hmac,
)

from .interface import EngineProtocolError, EngineTransportError

_CLOSE_LINGER_MS = 1000


class ZmqEngineTransport:
    """Async ZMQ ``DEALER`` transport speaking ``trtllm.serve_engine.v1`` over two channels."""

    def __init__(
        self,
        *,
        session_addr: str,
        control_addr: str,
        hmac_key: bytes,
        frontend_id: int = 0,
        ctx: Optional[zmq.asyncio.Context] = None,
    ) -> None:
        if not hmac_key:
            raise ValueError("hmac_key is required for ZmqEngineTransport")
        self._session_addr = session_addr
        self._control_addr = control_addr
        self._hmac_key = bytes(hmac_key)
        self._frontend_id = frontend_id
        self._owns_ctx = ctx is None
        self._ctx = ctx or zmq.asyncio.Context()

        identity = f"frontend-{frontend_id}".encode("ascii")
        self._session_sock = self._ctx.socket(zmq.DEALER)
        self._session_sock.setsockopt(zmq.IDENTITY, identity)
        self._session_sock.setsockopt(zmq.LINGER, _CLOSE_LINGER_MS)
        self._session_sock.set_hwm(0)
        self._session_sock.connect(session_addr)

        self._control_sock = self._ctx.socket(zmq.DEALER)
        self._control_sock.setsockopt(zmq.IDENTITY, identity)
        self._control_sock.setsockopt(zmq.LINGER, _CLOSE_LINGER_MS)
        self._control_sock.set_hwm(0)
        self._control_sock.connect(control_addr)

        self._send_lock = asyncio.Lock()
        self._control_lock = asyncio.Lock()
        self._next_call_id = 1
        self._closed = False

    async def open_session(
        self,
        open_msg: session_pb2.SessionOpen,
        *,
        timeout: float = 30.0,
    ) -> session_pb2.SessionAccepted:
        client_msg = session_pb2.ClientMessage(open=open_msg)
        await self.send_client_message(client_msg)
        server_msg, _ = await self.recv_server_message(timeout=timeout)
        if server_msg.WhichOneof("payload") != "accepted":
            raise EngineTransportError(
                f"Expected SessionAccepted as first message, got {server_msg.WhichOneof('payload')!r}"
            )
        return server_msg.accepted

    async def send_client_message(
        self,
        msg: session_pb2.ClientMessage,
        aux_buffers: Sequence[bytes] = (),
    ) -> None:
        if self._closed:
            raise EngineTransportError("ZmqEngineTransport is closed")
        frames = pack_session_frames(self._hmac_key, msg.SerializeToString(), aux_buffers)
        async with self._send_lock:
            await self._session_sock.send_multipart(frames, copy=False)

    async def recv_server_message(
        self,
        timeout: Optional[float] = None,
    ) -> tuple[session_pb2.ServerMessage, list[bytes]]:
        if self._closed:
            raise EngineTransportError("ZmqEngineTransport is closed")
        coro = self._session_sock.recv_multipart()
        try:
            raw_frames = await (asyncio.wait_for(coro, timeout=timeout) if timeout else coro)
        except asyncio.TimeoutError as exc:
            raise EngineTransportError("Timed out waiting for engine ServerMessage") from exc
        except zmq.ZMQError as exc:
            raise EngineTransportError(f"ZMQ session recv failed: {exc}") from exc
        try:
            proto_bytes, aux_buffers = unpack_session_frames(self._hmac_key, raw_frames)
        except WireFramingError as exc:
            raise EngineTransportError(str(exc), engine_fatal=False) from exc
        msg = session_pb2.ServerMessage()
        msg.ParseFromString(proto_bytes)
        return msg, aux_buffers

    async def call_control(
        self,
        method: ControlMethod,
        request: Any,
        *,
        timeout: float = 10.0,
    ) -> Any:
        if self._closed:
            raise EngineTransportError("ZmqEngineTransport is closed")
        async with self._control_lock:
            call_id = self._next_call_id
            self._next_call_id += 1
            frames = pack_control_frames(
                self._hmac_key, method, call_id, request.SerializeToString()
            )
            await self._control_sock.send_multipart(frames)
            try:
                raw_reply = await asyncio.wait_for(
                    self._control_sock.recv_multipart(), timeout=timeout
                )
            except asyncio.TimeoutError as exc:
                raise EngineTransportError(
                    f"Control RPC {method.name} timed out after {timeout}s", engine_fatal=False
                ) from exc
            except zmq.ZMQError as exc:
                raise EngineTransportError(f"Control RPC {method.name} failed: {exc}") from exc

        try:
            resp_method, resp_call_id, is_error, body = unpack_control_frames(
                self._hmac_key, raw_reply
            )
        except WireFramingError as exc:
            raise EngineTransportError(str(exc), engine_fatal=False) from exc
        if resp_method != method or resp_call_id != call_id:
            raise EngineTransportError(
                f"Mismatched control reply: expected ({method.name}, {call_id}), "
                f"got ({resp_method.name}, {resp_call_id})",
                engine_fatal=False,
            )
        if is_error:
            err = common_pb2.ErrorInfo()
            err.ParseFromString(body)
            raise EngineProtocolError.from_proto(err)

        resp_cls = CONTROL_RESPONSE_TYPES[method]
        resp = resp_cls()
        resp.ParseFromString(body)
        return resp

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session_sock.close(linger=_CLOSE_LINGER_MS)
        self._control_sock.close(linger=_CLOSE_LINGER_MS)
        if self._owns_ctx:
            self._ctx.term()


__all__ = [
    "CTRL_STATUS_ERR",
    "CTRL_STATUS_OK",
    "MSG_KIND_PROTO",
    "ZmqEngineTransport",
    "compute_wire_hmac",
    "pack_control_frames",
    "pack_session_frames",
    "unpack_control_frames",
    "unpack_session_frames",
    "verify_wire_hmac",
]
