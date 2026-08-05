# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import collections
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).parents[4]
    / "examples"
    / "serve"
    / "anthropic_compatibility"
    / "gateway.py"
)
SPEC = importlib.util.spec_from_file_location("anthropic_gateway", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)


class FragmentedReader:
    def __init__(self, fragments):
        self.fragments = collections.deque(bytearray(f) for f in fragments)

    async def read(self, size):
        if not self.fragments:
            return b""
        fragment = self.fragments[0]
        data = bytes(fragment[:size])
        del fragment[:size]
        if not fragment:
            self.fragments.popleft()
        return data


class RecordingWriter:
    def __init__(self):
        self.data = bytearray()

    def write(self, data):
        self.data.extend(data)

    async def drain(self):
        pass


def upstream_chunk(payload):
    return b"%x\r\n%s\r\n" % (len(payload), payload)


def decode_chunked_response(response):
    head, marker, body = bytes(response).partition(b"\r\n\r\n")
    assert marker
    position = 0
    payload = bytearray()
    while True:
        line_end = body.find(b"\r\n", position)
        assert line_end >= 0
        size = int(body[position:line_end], 16)
        position = line_end + 2
        if size == 0:
            assert body[position:] == b"\r\n"
            break
        payload.extend(body[position:position + size])
        position += size
        assert body[position:position + 2] == b"\r\n"
        position += 2
    return head, bytes(payload)


def relay_response(*fragments):
    reader = FragmentedReader(fragments)
    writer = RecordingWriter()
    status = asyncio.run(gateway.Gateway(None).relay_response(reader, writer))
    return status, bytes(writer.data)


def make_args():
    return SimpleNamespace(
        lead_time=2700,
        no_relay=False,
        promote_after=180,
        serve_sh="/unused/serve.sh",
        yaml="/unused/deployment.yaml",
    )


def make_backend(job_id, end_time, healthy=True, state="running attempt 1"):
    backend = gateway.Backend(
        {
            "job_id": job_id,
            "url": "http://localhost:8333",
            "run_dir": "/runs/%s" % job_id,
            "state": state,
            "end_time": end_time,
            "heartbeat": time.time(),
        }
    )
    backend.healthy = healthy
    backend.healthy_since = time.time() - 1000 if healthy else 0
    return backend


def test_sse_message_stop_survives_transport_split():
    payload = (
        b"event: message_stop\n" b'data: {"type":"message_stop"}\n\n'
    )
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        + upstream_chunk(payload)
        + b"0\r\n\r\n"
    )
    first = response.index(b"message_stop") + len(b"message_st")
    second = response.index(b"message_stop", first) + len(b"message_st")

    status, relayed = relay_response(
        response[:first], response[first:second], response[second:]
    )
    head, body = decode_chunked_response(relayed)

    assert status == "200"
    assert b"Transfer-Encoding: chunked" in head
    assert body == payload
    assert gateway.SSE_ROTATED not in body


def test_sse_missing_message_stop_injects_error_before_chunk_end():
    payload = (
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta"}\n\n'
    )
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        + upstream_chunk(payload)
        + b"0\r\n\r\n"
    )

    status, relayed = relay_response(response)
    _, body = decode_chunked_response(relayed)

    assert status == "200!"
    assert body == payload + gateway.SSE_ROTATED


def test_upstream_sse_error_is_not_duplicated():
    payload = (
        b"event: error\n"
        b'data: {"type":"error","error":{"type":"api_error"}}\n\n'
    )
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        + upstream_chunk(payload)
        + b"0\r\n\r\n"
    )

    status, relayed = relay_response(response)
    _, body = decode_chunked_response(relayed)

    assert status == "200"
    assert body == payload


def test_sse_truncated_upstream_chunk_still_produces_valid_downstream():
    payload = b"event: content_block_delta\ndata: partial"
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Transfer-Encoding: chunked\r\n\r\n"
        b"100\r\n"
        + payload
    )

    status, relayed = relay_response(response)
    _, body = decode_chunked_response(relayed)

    assert status == "200!"
    assert body == payload + gateway.SSE_ROTATED


def test_content_length_sse_is_normalized_to_chunked():
    payload = b"event: content_block_delta\ndata: partial\n\n"
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Content-Length: %d\r\n\r\n" % len(payload)
        + payload
    )

    status, relayed = relay_response(response)
    head, body = decode_chunked_response(relayed)

    assert status == "200!"
    assert b"Content-Length" not in head
    assert b"Transfer-Encoding: chunked" in head
    assert body == payload + gateway.SSE_ROTATED


def test_non_sse_response_is_relayed_verbatim():
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: 2\r\n\r\n{}"
    )

    status, relayed = relay_response(response[:20], response[20:])

    assert status == "200"
    assert relayed == response


def test_forward_handover_supersedes_older_backend():
    fleet = gateway.Fleet(make_args())
    fleet.backends = {
        "old": make_backend("old", 100),
        "new": make_backend("new", 200),
    }
    fleet.active = "old"

    fleet.elect()

    assert fleet.active == "new"
    assert fleet.superseded == {"old"}


def test_failback_keeps_newer_backend_available_for_recovery():
    fleet = gateway.Fleet(make_args())
    fleet.backends = {
        "old": make_backend("old", 100),
        "new": make_backend("new", 200, healthy=False),
    }
    fleet.active = "new"

    fleet.elect()

    assert fleet.active == "old"
    assert "new" not in fleet.superseded


def test_failed_newer_backend_does_not_block_another_successor(monkeypatch):
    fleet = gateway.Fleet(make_args())
    fleet.backends = {
        "old": make_backend("old", time.time() + 60),
        "new": make_backend("new", time.time() + 3600, healthy=False),
    }
    fleet.active = "old"
    fleet.ever_active = True
    calls = []

    async def fake_run_serve_sh(_fleet, *args):
        calls.append(args)
        return 0, "Submitted batch job 300"

    monkeypatch.setattr(gateway, "run_serve_sh", fake_run_serve_sh)
    asyncio.run(gateway.supervise(fleet))

    assert calls == [("submit", "--yaml", fleet.args.yaml, "--label", "relay")]
    assert fleet.pending[0] == "300"


def test_re_elected_backend_leaves_draining():
    fleet = gateway.Fleet(make_args())
    fleet.backends = {
        "old": make_backend("old", 100),
        "new": make_backend("new", 200, healthy=False),
    }
    fleet.active = "new"
    fleet.draining["old"] = time.time() + 60

    fleet.elect()

    assert fleet.active == "old"
    assert "old" not in fleet.draining


def test_supervisor_never_reclaims_active_backend(monkeypatch):
    fleet = gateway.Fleet(make_args())
    fleet.backends = {
        "active": make_backend("active", time.time() + 10_000),
    }
    fleet.active = "active"
    fleet.ever_active = True
    fleet.draining["active"] = time.time() - 1
    calls = []

    async def fake_run_serve_sh(_fleet, *args):
        calls.append(args)
        return 0, ""

    monkeypatch.setattr(gateway, "run_serve_sh", fake_run_serve_sh)
    asyncio.run(gateway.supervise(fleet))

    assert not calls
    assert "active" not in fleet.draining


def test_terminal_pending_job_is_replaced_without_an_active_backend(monkeypatch):
    fleet = gateway.Fleet(make_args())
    fleet.ever_active = True
    fleet.pending = ("failed", time.time() - 120)
    calls = []

    async def fake_slurm_job_status(_job_id):
        return "GONE", ""

    async def fake_run_serve_sh(_fleet, *args):
        calls.append(args)
        return 0, "Submitted batch job 300"

    monkeypatch.setattr(gateway, "slurm_job_status", fake_slurm_job_status)
    monkeypatch.setattr(gateway, "run_serve_sh", fake_run_serve_sh)
    asyncio.run(gateway.supervise(fleet))

    assert calls == [("submit", "--yaml", fleet.args.yaml, "--label", "recovery")]
    assert fleet.pending[0] == "300"


def test_invalid_slurm_job_id_is_terminal(monkeypatch):
    async def fake_run_slurm_command(*_args):
        return 1, "slurm_load_jobs error: Invalid job id specified"

    monkeypatch.setattr(gateway, "run_slurm_command", fake_run_slurm_command)

    status = asyncio.run(gateway.slurm_job_status("missing"))

    assert status == ("GONE", "")


def test_held_pending_job_is_cancelled_before_retry(monkeypatch):
    fleet = gateway.Fleet(make_args())
    fleet.pending = ("held", time.time() - 120)
    calls = []

    async def fake_slurm_job_status(_job_id):
        return "PENDING", "JobHeldUser"

    async def fake_run_slurm_command(*args):
        calls.append(args)
        return 0, ""

    monkeypatch.setattr(gateway, "slurm_job_status", fake_slurm_job_status)
    monkeypatch.setattr(gateway, "run_slurm_command", fake_run_slurm_command)
    asyncio.run(gateway.supervise_pending(fleet, time.time()))

    assert calls == [("scancel", "held")]
    assert fleet.pending is None


def test_failed_registered_successor_restarts_retained_allocation(monkeypatch):
    fleet = gateway.Fleet(make_args())
    fleet.pending = ("new", time.time() - 120)
    fleet.backends = {
        "new": make_backend(
            "new",
            200,
            healthy=False,
            state="attempt 1 exited with status 1; allocation retained",
        ),
    }
    calls = []

    async def fake_run_serve_sh(_fleet, *args):
        calls.append(args)
        return 0, "requested restart"

    monkeypatch.setattr(gateway, "run_serve_sh", fake_run_serve_sh)
    asyncio.run(gateway.supervise_pending(fleet, time.time()))

    assert calls == [("restart", "/runs/new")]
    assert fleet.pending[0] == "new"
