#!/usr/bin/env python3
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
"""What the gateway does after the client's socket reaches EOF.

These come from the jhb gateway log of 10-15 and 10-16. It held 30 "Task was
destroyed but it is pending!" lines for Gateway.handle, each followed by
"coroutine ignored GeneratorExit", and 25 "Task exception was never retrieved"
lines for a StreamReader.read() task. Three things lay behind them:

- A client that closed partway through its upload had the part that arrived
  forwarded to a backend. The backend waited for the rest, and the handler
  waited for the backend. Every one of the 30 was such a request: a body cut
  off in a client-abandon burst.
- A handler waiting on its backend after the client's EOF was referenced from
  nowhere, so garbage collection destroyed it mid-await. The pumped-body path
  is the one that has to keep waiting there: it serves clients that half-close
  after the upload.
- The buffered path's client watcher sometimes failed in the same tick in
  which the response finished. Its exception was never taken.
"""

import asyncio
import gc
import importlib.util
import socket
import time
import weakref

import pytest
from handover_harness import API_USER, CONVERSATION_HEADER, GATEWAY_PY, Scenario, control, wait_for

SPEC = importlib.util.spec_from_file_location("gw_client_eof_under_test", GATEWAY_PY)
gw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gw)


@pytest.fixture
def make_scenario():
    created = []

    def factory(*args, **kwargs):
        scenario = Scenario(*args, **kwargs)
        created.append(scenario)
        return scenario

    yield factory
    for scenario in reversed(created):
        scenario.close()


def stream_reader(data, eof=True):
    reader = asyncio.StreamReader()
    if data:
        reader.feed_data(data)
    if eof:
        reader.feed_eof()
    return reader


def content_length(n):
    return [("Content-Length", str(n))]


def run_collecting_loop_errors(main):
    """Run `main()` on a fresh loop; return what asyncio reported to the loop."""
    reported = []

    async def wrapper():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: reported.append(context["message"]))
        await main()
        # A collection inside the loop, so a report about a task of this loop
        # reaches the handler installed above.
        gc.collect()
        await asyncio.sleep(0)

    asyncio.run(wrapper())
    return reported


# ---------------------------------------------------------------------------
# read_body
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_body_cut_short_by_eof_is_an_error_not_a_request():
    reader = stream_reader(b"0123456789")

    with pytest.raises(gw.ClientClosedMidBody) as caught:
        await gw.read_body(reader, b"", content_length(1000), 4096)

    assert "after 10 of 1000 body byte(s)" in str(caught.value)
    # Every caller already catches the body-read failures as OSError.
    assert isinstance(caught.value, OSError)


@pytest.mark.asyncio
async def test_a_complete_body_still_reads_whole():
    reader = stream_reader(b"6789tail", eof=False)

    body, leftover = await gw.read_body(reader, b"012345", content_length(10), 4096)

    assert body == b"0123456789"
    assert leftover == b""


@pytest.mark.asyncio
async def test_a_reset_while_reading_the_body_is_tagged_as_the_clients():
    reader = asyncio.StreamReader()
    reader.set_exception(ConnectionResetError(104, "Connection reset by peer"))

    with pytest.raises(gw.SideError) as caught:
        await gw.read_body(reader, b"", content_length(1000), 4096)

    assert gw.error_side(caught.value) == "client_read"


# ---------------------------------------------------------------------------
# settle
# ---------------------------------------------------------------------------
def test_settle_takes_the_failure_of_a_task_that_already_finished():
    async def main():
        async def fail():
            raise ConnectionResetError(104, "Connection reset by peer")

        task = asyncio.create_task(fail())
        await asyncio.sleep(0)
        assert task.done()
        await gw.settle(task)

    assert not [m for m in run_collecting_loop_errors(main) if "never retrieved" in m]


def test_without_settle_the_same_failure_is_reported():
    """The control: what `settle` prevents is real, and this test can see it."""

    async def main():
        async def fail():
            raise ConnectionResetError(104, "Connection reset by peer")

        task = asyncio.create_task(fail())
        await asyncio.sleep(0)
        assert task.done()

    assert [m for m in run_collecting_loop_errors(main) if "never retrieved" in m]


@pytest.mark.asyncio
async def test_settle_cancels_a_task_that_still_runs():
    started = asyncio.Event()

    async def forever():
        started.set()
        await asyncio.sleep(3600)

    task = asyncio.create_task(forever())
    await started.wait()

    await gw.settle(task)

    assert task.cancelled()


# ---------------------------------------------------------------------------
# Gateway.handle keeps its task referenced
# ---------------------------------------------------------------------------
def test_a_handler_waiting_on_an_unreferenced_future_survives_collection(monkeypatch):
    """The mechanism, without sockets: what an EOF leaves a pumped request with.

    The handler awaits a future that only the handler itself references -- the
    shape a backend read takes once the client's socket has left the event
    loop. Without a reference from outside, a collection destroys the task.
    """
    state = {}

    async def waits_on_an_orphan(self, reader, writer):
        future = asyncio.get_running_loop().create_future()
        state["future"] = weakref.ref(future)
        await future
        state["finished"] = True

    monkeypatch.setattr(gw.Gateway, "_handle", waits_on_an_orphan)

    async def main():
        gateway = gw.Gateway.__new__(gw.Gateway)
        task = asyncio.ensure_future(gateway.handle(None, None))
        task_ref = weakref.ref(task)
        del task
        await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
        assert task_ref() is not None, "the handler was collected mid-await"
        state["future"]().set_result(None)
        await task_ref()
        assert state["finished"]
        assert not gw.HANDLERS

    reported = run_collecting_loop_errors(main)
    assert not [m for m in reported if "destroyed but it is pending" in m]


# ---------------------------------------------------------------------------
# Against a running gateway
# ---------------------------------------------------------------------------
def fleet_inflight(port):
    status, data, _ = control(port, "GET", "/_gateway/fleet")
    assert status == 200, data
    return {job_id: entry.get("inflight", 0) for job_id, entry in data["backends"].items()}


def raw_head(port, length, convo, extra=()):
    lines = [
        "POST /v1/messages HTTP/1.1",
        "Host: 127.0.0.1:%d" % port,
        "x-api-key: %s" % API_USER,
        "%s: %s" % (CONVERSATION_HEADER, convo),
        "Content-Length: %d" % length,
        "Content-Type: application/json",
    ]
    lines.extend("%s: %s" % header for header in extra)
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def read_all(sock, timeout):
    sock.settimeout(timeout)
    received = b""
    while True:
        try:
            chunk = sock.recv(65536)
        except ConnectionResetError:
            break
        except TimeoutError:
            pytest.fail("no end of response within %.0fs; received %r" % (timeout, received))
        if not chunk:
            return received
        received += chunk
    return received


def test_an_upload_cut_short_never_reaches_a_backend(make_scenario):
    scenario = make_scenario("client-eof-midbody", backends=2).start()

    with socket.create_connection(("127.0.0.1", scenario.port), timeout=5.0) as sock:
        sock.sendall(raw_head(scenario.port, 1000, "midbody") + b'{"model":"m"}')
        sock.shutdown(socket.SHUT_WR)
        received = read_all(sock, 10.0)

    assert received == b"", "nothing can be said to a request with half a body"
    wait_for(
        "the dropped upload in the gateway log",
        lambda: "reading body: client closed after 13 of 1000 body byte(s)"
        in scenario.gateway.log_text(),
        timeout=5.0,
    )
    # Never placed: no backend counts it in flight, then or later.
    time.sleep(0.5)
    assert set(fleet_inflight(scenario.port).values()) == {0}


def churn(port, seconds):
    """Keep the gateway allocating, so that its garbage collector runs.

    An idle test gateway can go seconds without a collection, and a handler
    that nothing references survives until one runs. Health checks are the
    cheapest requests that make it allocate.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        control(port, "GET", "/_gateway/health", authenticated=False, timeout=5.0)


def test_a_pumped_request_whose_client_half_closes_still_gets_its_answer(make_scenario):
    """The path `Gateway.handle` holds its task for.

    A body over --max-body-buffer is pumped, not buffered, and on that path the
    client's EOF is not taken for abandonment: a client may half-close after
    its upload and still read the answer. The backend takes three seconds, the
    gateway collects garbage meanwhile, and the handler has to be alive when
    the answer comes.
    """
    scenario = make_scenario(
        "client-eof-pumped", backends=1, extra_args=["--max-body-buffer", "16"]
    ).start()
    body = b'{"model":"m","messages":[],"pad":"' + b"x" * 200 + b'"}'

    with socket.create_connection(("127.0.0.1", scenario.port), timeout=5.0) as sock:
        sock.sendall(raw_head(scenario.port, len(body), "pumped", [("x-test-delay", "3")]) + body)
        sock.shutdown(socket.SHUT_WR)
        churn(scenario.port, 2.5)
        received = read_all(sock, 15.0)

    assert received.startswith(b"HTTP/1.1 200"), received[:200]
    assert b'"request_bytes": %d' % len(body) in received
    log = scenario.gateway.log_text()
    assert "destroyed but it is pending" not in log
    assert "coroutine ignored GeneratorExit" not in log
