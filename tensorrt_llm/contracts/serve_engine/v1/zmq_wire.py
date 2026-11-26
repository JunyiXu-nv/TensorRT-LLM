# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared language-neutral, pickle-free ZMQ wire framing for ``trtllm.serve_engine.v1``."""

from __future__ import annotations

import hashlib
import hmac
import struct
from enum import IntEnum
from typing import Any, Sequence

from . import control_pb2, stats_pb2

MSG_KIND_PROTO = b"\x01"
CTRL_STATUS_OK = b"\x00"
CTRL_STATUS_ERR = b"\x01"
_CALL_ID_STRUCT = struct.Struct("<Q")


class ControlMethod(IntEnum):
    """Wire method discriminator for the 13 ``ControlService`` RPCs over ZMQ."""

    GET_INFO = 1
    HEALTH = 2
    GET_LOAD = 3
    GET_KV_CAPACITY = 4
    GET_ITERATION_STATS = 5
    SUBSCRIBE_KV_EVENTS = 6
    GET_KV_EVENTS = 7
    RELEASE_HANDOFFS = 8
    GET_TRANSCEIVER_STATE = 9
    LORA_LOAD = 10
    LORA_UNLOAD = 11
    LORA_LIST = 12
    ADMIN = 13


CONTROL_REQUEST_TYPES: dict[ControlMethod, type[Any]] = {
    ControlMethod.GET_INFO: control_pb2.GetInfoRequest,
    ControlMethod.HEALTH: control_pb2.HealthRequest,
    ControlMethod.GET_LOAD: control_pb2.GetLoadRequest,
    ControlMethod.GET_KV_CAPACITY: control_pb2.GetKvCapacityRequest,
    ControlMethod.GET_ITERATION_STATS: control_pb2.GetIterationStatsRequest,
    ControlMethod.SUBSCRIBE_KV_EVENTS: control_pb2.SubscribeKvEventsRequest,
    ControlMethod.GET_KV_EVENTS: control_pb2.GetKvEventsRequest,
    ControlMethod.RELEASE_HANDOFFS: control_pb2.ReleaseHandoffsRequest,
    ControlMethod.GET_TRANSCEIVER_STATE: control_pb2.GetTransceiverStateRequest,
    ControlMethod.LORA_LOAD: control_pb2.LoraLoadRequest,
    ControlMethod.LORA_UNLOAD: control_pb2.LoraUnloadRequest,
    ControlMethod.LORA_LIST: control_pb2.LoraListRequest,
    ControlMethod.ADMIN: control_pb2.AdminRequest,
}

CONTROL_RESPONSE_TYPES: dict[ControlMethod, type[Any]] = {
    ControlMethod.GET_INFO: control_pb2.GetInfoResponse,
    ControlMethod.HEALTH: control_pb2.HealthResponse,
    ControlMethod.GET_LOAD: control_pb2.LoadInfo,
    ControlMethod.GET_KV_CAPACITY: control_pb2.KvCapacityResponse,
    ControlMethod.GET_ITERATION_STATS: control_pb2.GetIterationStatsResponse,
    ControlMethod.SUBSCRIBE_KV_EVENTS: stats_pb2.KvEventBatch,
    ControlMethod.GET_KV_EVENTS: stats_pb2.KvEventBatch,
    ControlMethod.RELEASE_HANDOFFS: control_pb2.ReleaseHandoffsResponse,
    ControlMethod.GET_TRANSCEIVER_STATE: control_pb2.GetTransceiverStateResponse,
    ControlMethod.LORA_LOAD: control_pb2.LoraLoadResponse,
    ControlMethod.LORA_UNLOAD: control_pb2.LoraUnloadResponse,
    ControlMethod.LORA_LIST: control_pb2.LoraListResponse,
    ControlMethod.ADMIN: control_pb2.AdminResponse,
}


class WireFramingError(ValueError):
    """Raised when a ZMQ frame fails structural or HMAC validation."""


def compute_wire_hmac(hmac_key: bytes, parts: Sequence[bytes]) -> bytes:
    """Compute a 32-byte HMAC-SHA256 over length-prefixed multipart frame payloads."""
    mac = hmac.new(hmac_key, digestmod=hashlib.sha256)
    for part in parts:
        mac.update(struct.pack("<Q", len(part)))
        mac.update(part)
    return mac.digest()


def verify_wire_hmac(hmac_key: bytes, tag: bytes, parts: Sequence[bytes]) -> None:
    expected = compute_wire_hmac(hmac_key, parts)
    if not hmac.compare_digest(tag, expected):
        raise WireFramingError("ZMQ wire HMAC verification failed")


def pack_session_frames(
    hmac_key: bytes,
    proto_bytes: bytes,
    aux_buffers: Sequence[bytes] = (),
) -> list[bytes]:
    payload = [MSG_KIND_PROTO, proto_bytes, *aux_buffers]
    tag = compute_wire_hmac(hmac_key, payload)
    return [tag, *payload]


def unpack_session_frames(hmac_key: bytes, frames: Sequence[bytes]) -> tuple[bytes, list[bytes]]:
    if len(frames) < 3:
        raise WireFramingError(
            f"Malformed ZMQ session frame: expected >=3 parts, got {len(frames)}"
        )
    tag = frames[0]
    payload = list(frames[1:])
    verify_wire_hmac(hmac_key, tag, payload)
    if payload[0] != MSG_KIND_PROTO:
        raise WireFramingError(f"Unsupported ZMQ session msg_kind: {payload[0]!r}")
    return payload[1], payload[2:]


def pack_control_frames(
    hmac_key: bytes,
    method: ControlMethod,
    call_id: int,
    proto_bytes: bytes,
    *,
    is_error: bool = False,
) -> list[bytes]:
    status = CTRL_STATUS_ERR if is_error else CTRL_STATUS_OK
    payload = [bytes([int(method)]), _CALL_ID_STRUCT.pack(call_id), status, proto_bytes]
    tag = compute_wire_hmac(hmac_key, payload)
    return [tag, *payload]


def unpack_control_frames(
    hmac_key: bytes, frames: Sequence[bytes]
) -> tuple[ControlMethod, int, bool, bytes]:
    if len(frames) != 5:
        raise WireFramingError(f"Malformed ZMQ control frame: expected 5 parts, got {len(frames)}")
    tag = frames[0]
    payload = list(frames[1:])
    verify_wire_hmac(hmac_key, tag, payload)
    method = ControlMethod(payload[0][0])
    (call_id,) = _CALL_ID_STRUCT.unpack(payload[1])
    is_error = payload[2] != CTRL_STATUS_OK
    return method, call_id, is_error, payload[3]


__all__ = [
    "CONTROL_REQUEST_TYPES",
    "CONTROL_RESPONSE_TYPES",
    "CTRL_STATUS_ERR",
    "CTRL_STATUS_OK",
    "ControlMethod",
    "MSG_KIND_PROTO",
    "WireFramingError",
    "compute_wire_hmac",
    "pack_control_frames",
    "pack_session_frames",
    "unpack_control_frames",
    "unpack_session_frames",
    "verify_wire_hmac",
]
