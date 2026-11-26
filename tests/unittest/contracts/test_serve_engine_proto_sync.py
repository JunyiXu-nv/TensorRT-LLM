# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify checked-in ``trtllm.serve_engine.v1`` protobuf bindings stay in sync with .proto files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tensorrt_llm.contracts.serve_engine.v1 import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    capabilities_pb2,
    control_pb2,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.cpu_only
def test_serve_engine_proto_bindings_in_sync() -> None:
    script = REPO_ROOT / "scripts" / "generate_serve_engine_proto.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Proto bindings out of sync:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.cpu_only
def test_v1_1_dynamo_additions_present() -> None:
    assert (PROTOCOL_MAJOR, PROTOCOL_MINOR) == (1, 1)
    load = control_pb2.LoadInfo(sampled_at_unix_nanos=123456789)
    assert load.sampled_at_unix_nanos == 123456789
    dp_load = control_pb2.DpRankLoad(dp_rank=0, sampled_at_unix_nanos=987654321)
    assert dp_load.sampled_at_unix_nanos == 987654321
    mm = capabilities_pb2.MultimodalCapabilities(image_routing_version=2)
    assert mm.image_routing_version == 2
    kv_ev = capabilities_pb2.KvEventsCapability(
        sources=[
            capabilities_pb2.KvEventSourceInfo(
                transport="zmq",
                encoding="msgpack",
                schema_version=1,
                endpoint="tcp://127.0.0.1:5557",
                topic="kv-events",
                dp_rank=0,
            )
        ]
    )
    assert len(kv_ev.sources) == 1
    assert kv_ev.sources[0].dp_rank == 0
