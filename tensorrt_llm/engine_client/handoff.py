# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend disagg handoff helpers implementing ``DG-04`` / ``DG-60`` separation."""

from __future__ import annotations

import base64
from typing import Any, Optional

from tensorrt_llm.contracts.serve_engine.v1 import common_pb2, disagg_pb2, request_pb2


def encode_signed_handoff_b64(signed: disagg_pb2.SignedHandoff) -> str:
    """Serialize a ``SignedHandoff`` envelope to opaque URL-safe/standard base64."""
    return base64.b64encode(signed.SerializeToString()).decode("ascii")


def decode_signed_handoff_b64(encoded: str) -> disagg_pb2.SignedHandoff:
    """Parse a base64-encoded ``SignedHandoff`` without modifying its bytes or MAC."""
    signed = disagg_pb2.SignedHandoff()
    signed.ParseFromString(base64.b64decode(encoded))
    return signed


def encode_rendezvous_ticket_b64(ticket: disagg_pb2.RendezvousTicket) -> str:
    return base64.b64encode(ticket.SerializeToString()).decode("ascii")


def decode_rendezvous_ticket_b64(encoded: str) -> disagg_pb2.RendezvousTicket:
    ticket = disagg_pb2.RendezvousTicket()
    ticket.ParseFromString(base64.b64decode(encoded))
    return ticket


def extract_handoff_key(signed_b64: str) -> tuple[str, int]:
    """Read only frontend-readable identity fields ``(ctx_instance_id, disagg_request_id)``."""
    signed = decode_signed_handoff_b64(signed_b64)
    handoff = disagg_pb2.KvHandoff()
    handoff.ParseFromString(signed.handoff_bytes)
    return handoff.ctx_instance_id, int(handoff.disagg_request_id)


def build_gen_disagg_params(
    signed_b64: str,
    *,
    schedule_style: int = common_pb2.SCHEDULE_STYLE_CONTEXT_FIRST,
) -> request_pb2.DisaggParams:
    """Build generation-leg ``DisaggParams`` carrying the opaque ``SignedHandoff`` untouched."""
    signed = decode_signed_handoff_b64(signed_b64)
    _, disagg_req_id = extract_handoff_key(signed_b64)
    return request_pb2.DisaggParams(
        request_id=disagg_req_id,
        schedule_style=schedule_style,
        handoff=signed,
    )


class HandoffReleaseTracker:
    """Tracks in-flight ctx handoffs and releases them idempotently on disconnect or retry."""

    def __init__(self, facade: Any) -> None:
        self._facade = facade
        self._pending: dict[int, str] = {}

    def register(self, signed_b64: str) -> tuple[str, int]:
        ctx_instance_id, disagg_id = extract_handoff_key(signed_b64)
        self._pending[disagg_id] = ctx_instance_id
        return ctx_instance_id, disagg_id

    def mark_consumed(self, disagg_request_id: int) -> None:
        self._pending.pop(int(disagg_request_id), None)

    async def release(
        self, disagg_request_id: int, *, ctx_instance_id: Optional[str] = None
    ) -> Any:
        inst = ctx_instance_id or self._pending.pop(int(disagg_request_id), "")
        self._pending.pop(int(disagg_request_id), None)
        return await self._facade.release_handoffs([(inst, int(disagg_request_id))])

    async def release_all(self) -> Any:
        if not self._pending:
            return None
        items = [(inst, rid) for rid, inst in self._pending.items()]
        self._pending.clear()
        return await self._facade.release_handoffs(items)


__all__ = [
    "HandoffReleaseTracker",
    "build_gen_disagg_params",
    "decode_rendezvous_ticket_b64",
    "decode_signed_handoff_b64",
    "encode_rendezvous_ticket_b64",
    "encode_signed_handoff_b64",
    "extract_handoff_key",
]
