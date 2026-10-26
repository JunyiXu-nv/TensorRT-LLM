# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The recorder keeps a record of every stream, however it ended (CPU only)."""

import asyncio
import json
import os
import sys

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)

import record_requests  # noqa: E402

CHUNKS = [b'event: ping\ndata: {"type": "ping"}\n\n'] * 3


async def _upstream(request):
    resp = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
    await resp.prepare(request)
    for chunk in CHUNKS:
        await resp.write(chunk)
        await asyncio.sleep(0.01)
    if request.query.get("break"):
        # The upstream dies mid-stream: no terminating chunk, connection gone.
        request.transport.close()
        return resp
    await resp.write_eof()
    return resp


async def _run(out_dir, path="/v1/messages", build_id=None, which="responses"):
    upstream_app = web.Application()
    upstream_app.router.add_route("POST", "/v1/messages", _upstream)
    async with TestServer(upstream_app) as upstream:
        recorder = record_requests.build_app(
            str(out_dir), forward=str(upstream.make_url("")), build_id=build_id
        )
        async with TestServer(recorder) as server:
            async with aiohttp.ClientSession() as client:
                try:
                    async with client.post(server.make_url(path), json={"stream": True}) as reply:
                        await reply.read()
                except aiohttp.ClientError:
                    pass
    with open(os.path.join(out_dir, f"{which}-capture.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def test_a_completed_stream_is_recorded(tmp_path):
    (record,) = asyncio.run(_run(tmp_path))

    assert record["status"] == "completed"
    assert record["response"]["body"] == b"".join(CHUNKS).decode()
    assert "first_byte_s" in record["timing"]


@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("Connection lost"),
        ConnectionResetError(104, "Connection reset by peer"),
        TimeoutError(110, "Connection timed out"),
        OSError(113, "No route to host"),
    ],
    ids=["lost", "reset", "timed_out", "no_route"],
)
def test_a_client_that_went_away_is_recorded(tmp_path, monkeypatch, error):
    """Each way a write to a departed client failed on 2026-10-23 still leaves a record."""
    real_write = web.StreamResponse.write
    calls = {"n": 0}

    async def write(self, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise error
        return await real_write(self, data)

    monkeypatch.setattr(web.StreamResponse, "write", write)

    (record,) = asyncio.run(_run(tmp_path))

    assert record["status"] == "client_disconnected"
    assert record["timing"]["client_error"] == type(error).__name__
    assert record["response"]["body"].startswith(CHUNKS[0].decode())


async def _slow_upstream(request):
    await asyncio.sleep(1.0)  # no headers for a second, like a server under a prefill stall
    resp = web.StreamResponse(status=200, headers={"content-type": "text/event-stream"})
    await resp.prepare(request)
    await resp.write(CHUNKS[0])
    await resp.write_eof()
    return resp


async def _run_slow(out_dir, client_timeout, stop_recorder_after=None):
    """A client that waits ``client_timeout`` s for a reply that takes a second to start.

    With ``stop_recorder_after``, the recorder is stopped that many seconds in, the request still
    in flight; its graceful-shutdown wait is shorter than the reply's delay, so the handler is
    cancelled.
    """
    upstream_app = web.Application()
    upstream_app.router.add_route("POST", "/v1/messages", _slow_upstream)
    async with TestServer(upstream_app) as upstream:
        recorder = record_requests.build_app(str(out_dir), forward=str(upstream.make_url("")))
        server = TestServer(recorder)
        await server.start_server(shutdown_timeout=0.1)
        timeout = aiohttp.ClientTimeout(total=client_timeout)

        async def call():
            try:
                async with aiohttp.ClientSession(timeout=timeout) as client:
                    async with client.post(
                        server.make_url("/v1/messages"), json={"stream": True}
                    ) as r:
                        await r.read()
            except (asyncio.TimeoutError, aiohttp.ClientError):
                pass

        task = asyncio.ensure_future(call())
        if stop_recorder_after is not None:
            await asyncio.sleep(stop_recorder_after)
            await server.close()
        await task
        await asyncio.sleep(1.5)  # long enough for the upstream to have answered
        if stop_recorder_after is None:
            await server.close()
    with open(os.path.join(out_dir, "responses-capture.jsonl")) as fh:
        return [json.loads(line) for line in fh]


def test_a_client_that_left_before_the_reply_is_recorded(tmp_path):
    """On 2026-10-23 six requests whose client gave up before the first byte left no record."""
    records = asyncio.run(_run_slow(tmp_path, client_timeout=0.3))

    assert len(records) == 1, "the request has no response record"
    (record,) = records
    assert record["status"] == "client_disconnected"
    assert record["timing"]["before_reply"] is True
    assert record["response"] == {"kind": "none", "body": None}


def test_a_request_in_flight_when_the_recorder_stops_is_recorded(tmp_path):
    records = asyncio.run(_run_slow(tmp_path, client_timeout=5.0, stop_recorder_after=0.3))

    assert len(records) == 1, "the request has no response record"
    (record,) = records
    assert record["status"] == "recorder_shutdown"
    assert record["timing"]["before_reply"] is True


def test_an_upstream_that_breaks_is_not_the_client(tmp_path):
    (record,) = asyncio.run(_run(tmp_path, "/v1/messages?break=1"))

    assert record["status"] in ("upstream_error", "completed")
    if record["status"] == "upstream_error":
        assert "upstream_error" in record["timing"]
    assert "client_error" not in record["timing"]


def test_every_record_carries_the_build_id(tmp_path):
    """The validator proves which build served from the records (I-05, R-05)."""
    (response,) = asyncio.run(_run(tmp_path / "r", build_id="1.4.0rc2+abc"))
    (request,) = asyncio.run(_run(tmp_path / "q", build_id="1.4.0rc2+abc", which="requests"))

    assert response["build_id"] == request["build_id"] == "1.4.0rc2+abc"
