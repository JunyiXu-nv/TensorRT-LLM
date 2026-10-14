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
"""What the gateway does in the seconds after a backend stops answering.

The window these cover is small and expensive. A backend dies; the health
probe samples every few seconds and has not run yet; every request routed in
the meantime fails at connect. Measured on this fleet over 21 hours: 2,670
502s, 2,669 of them backend-side, arriving in per-minute bursts -- 551 in one
minute when a backend hung, 192 in another when one hit its walltime.

What made those bursts cost whole campaigns rather than single requests is
that the clients retried and the retries failed too. A retry carries the same
conversation id, the pin still named the dead backend, and the gateway sent it
straight back: conversations took 2 to 8 consecutive 502s and then gave up.
The gateway is the only party in that exchange that knows which backend just
refused, so it is the only one that can route around it.
"""

import pytest
from handover_harness import (
    API_USER,
    CONVERSATION_HEADER,
    RequestStream,
    Scenario,
    control,
    json_body,
    request,
    stream_report,
    wait_for,
)


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


# The router keys conversations by where the id was found, not by the id, so a
# header-borne "x" is "hdr:x" in the pin table. Pinning the bare value silently
# pins nothing: the pin is stored, the request derives a different key, and the
# placement falls through to the load-based choice. That failure looks exactly
# like a working test whose assertions happen to pass.
PIN_PREFIX = "hdr:"


def pin(port, convo, job_id):
    status, data, _ = control(
        port,
        "POST",
        "/_gateway/pin",
        {"conversation": PIN_PREFIX + convo, "backend": job_id},
    )
    assert status == 200, data
    return data


def ask(port, convo):
    return request(
        port,
        "POST",
        "/v1/messages",
        headers=[("x-api-key", API_USER), (CONVERSATION_HEADER, convo)],
        body=b'{"model":"m","messages":[]}',
    )


def backend_health(port):
    _, data, _ = control(port, "GET", "/_gateway/fleet")
    return {j: b["healthy"] for j, b in data["backends"].items()}


def test_a_request_to_a_dead_backend_is_retried_elsewhere(make_scenario):
    """The client gets an answer instead of the 502 the old path returned.

    Staged the way production produces it: the backend process is stopped
    while its registration keeps being written, so the gateway still lists it
    as healthy and still routes to it. That is the whole window -- the probe
    has not run, the table is stale, and the request arrives anyway.

    Pinned by hand rather than left to the router so the request is guaranteed
    to pick the dead backend. Waiting for a natural placement would make the
    test depend on the probe not having fired yet, which is a race.
    """
    scenario = make_scenario("retry-dead", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    dead, alive = scenario.fleet.backends[0], scenario.fleet.backends[1]

    convo = "retry-me"
    pin(scenario.port, convo, dead.job_id)
    first = ask(scenario.port, convo)
    assert first.status == 200, first
    assert first.backend == dead.job_id, (
        "the pin did not take, so this test would pass without exercising a retry"
    )

    dead.stop()  # heartbeats continue: the gateway still believes in it

    out = ask(scenario.port, convo)
    assert out.status == 200, (
        "a request to a backend that stopped answering came back %r; the retry "
        "did not happen" % (out,)
    )
    assert out.backend == alive.job_id, (
        "the retry went back to %r, which is the loop the retry exists to break" % (out.backend,)
    )


def test_refusals_drop_a_backend_without_waiting_for_the_probe(make_scenario):
    """Traffic is a denser health sampler than the probe, so it is believed.

    --unhealthy-after is left at its default here on purpose: if the probe
    were what dropped the backend, this test would pass for the wrong reason.
    Twenty consecutive probe timeouts cannot happen in the seconds this takes.
    """
    scenario = make_scenario(
        "retry-refusals",
        backends=2,
        extra_args=("--refusals-before-drop", "3", "--refusal-window", "30"),
    )
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    dead, alive = scenario.fleet.backends[0], scenario.fleet.backends[1]

    assert backend_health(scenario.port)[dead.job_id] is True
    dead.stop()

    # Each pinned request refuses once, then retries onto the live backend.
    for i in range(3):
        convo = "refusal-%d" % i
        pin(scenario.port, convo, dead.job_id)
        out = ask(scenario.port, convo)
        assert out.status == 200, out

    wait_for(
        "the refused backend to be dropped",
        lambda: backend_health(scenario.port)[dead.job_id] is False,
        timeout=15.0,
    )
    assert backend_health(scenario.port)[alive.job_id] is True, (
        "dropping the dead backend also took out the healthy one"
    )


def test_one_dead_backend_does_not_cost_the_stream_anything(make_scenario):
    """The aggregate claim, which is the one that mattered in production.

    A single retried request proves the mechanism; this proves it holds while
    real traffic is flowing, which is when the 551-in-a-minute burst happened.
    """
    scenario = make_scenario("retry-stream", backends=3)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)

    stream = RequestStream(scenario.port, workers=6).start()
    try:
        wait_for("40 served", lambda: len(stream.served()) >= 40, timeout=30.0)
        scenario.fleet.backends[0].stop()
        before = len(stream.served())
        wait_for("60 more served", lambda: len(stream.served()) > before + 60, timeout=45.0)
    finally:
        stream.stop()

    report = stream_report(stream)
    assert not stream.refused(), report
    assert not stream.lost(), report
    assert not stream.errored(), report


def test_retry_can_be_turned_off(make_scenario):
    """--retry-upstream 0 restores the old behaviour exactly.

    Worth a test because the flag is the rollback: if the retry ever turns out
    to mask something worth seeing, this is how it gets switched off, and a
    rollback that does not work is not one.
    """
    scenario = make_scenario("retry-off", backends=2, extra_args=("--retry-upstream", "0"))
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    dead = scenario.fleet.backends[0]

    convo = "no-retry"
    pin(scenario.port, convo, dead.job_id)
    assert ask(scenario.port, convo).status == 200
    dead.stop()

    out = ask(scenario.port, convo)
    assert out.status == 502, (
        "with retries disabled the client should see the backend's failure, got %r" % (out,)
    )


# A backend in its last seconds still answers, with 500s: lhr is1 (10-14) gave
# 190 of them, about a second each, in the five seconds before it went away at
# its wall clock and the probe noticed. Until then the gateway routed to it,
# and a client's retry, carrying the same conversation id, was pinned straight
# back to it.


def test_a_server_error_is_retried_elsewhere(make_scenario):
    scenario = make_scenario("retry-5xx", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    failing, alive = scenario.fleet.backends[0], scenario.fleet.backends[1]

    convo = "answered-500"
    pin(scenario.port, convo, failing.job_id)
    failing.set_mode("failing")

    out = ask(scenario.port, convo)
    assert out.status == 200, (
        "a backend's quick 500 reached the client as %r instead of being retried" % (out,)
    )
    assert out.backend == alive.job_id, out
    assert backend_health(scenario.port)[failing.job_id] is True, (
        "one server error must not drop a backend and every pin on it"
    )


def test_with_nowhere_else_the_backends_own_error_reaches_the_client(make_scenario):
    """Not a 502 made up by the gateway: the client is owed the backend's message."""
    scenario = make_scenario("retry-5xx-alone", backends=1)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    only = scenario.fleet.backends[0]
    only.set_mode("failing")

    out = ask(scenario.port, "nowhere-else")
    assert out.status == 500, out
    assert out.backend == only.job_id, out
    assert json_body(out) == {"error": "workers gone", "backend": only.job_id}, out.body


def test_a_slow_server_error_is_relayed_not_retried(make_scenario):
    """Past --retry-server-errors-within, a 500 is more about the request than the backend."""
    scenario = make_scenario(
        "retry-5xx-slow", backends=2, extra_args=("--retry-server-errors-within", "0.5")
    )
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    failing = scenario.fleet.backends[0]

    convo = "slow-500"
    pin(scenario.port, convo, failing.job_id)
    failing.set_mode("failing")
    out = request(
        scenario.port,
        "POST",
        "/v1/messages",
        headers=[("x-api-key", API_USER), (CONVERSATION_HEADER, convo), ("X-Test-Delay", "1.5")],
        body=b'{"model":"m","messages":[]}',
    )
    assert out.status == 500, out
    assert out.backend == failing.job_id, out


def test_a_server_error_off_the_inference_paths_is_relayed(make_scenario):
    """A batch create that failed may still have created something; never send it twice."""
    scenario = make_scenario("retry-5xx-batches", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    failing = scenario.fleet.backends[0]

    convo = "batch-500"
    pin(scenario.port, convo, failing.job_id)
    failing.set_mode("failing")
    out = request(
        scenario.port,
        "POST",
        "/v1/messages/batches",
        headers=[("x-api-key", API_USER), (CONVERSATION_HEADER, convo)],
        body=b'{"requests":[]}',
    )
    assert out.status == 500, out
    assert out.backend == failing.job_id, out


def test_server_error_retry_can_be_turned_off(make_scenario):
    scenario = make_scenario(
        "retry-5xx-off", backends=2, extra_args=("--retry-server-errors-within", "0")
    )
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    failing = scenario.fleet.backends[0]

    convo = "no-5xx-retry"
    pin(scenario.port, convo, failing.job_id)
    failing.set_mode("failing")
    out = ask(scenario.port, convo)
    assert out.status == 500, out
    assert out.backend == failing.job_id, out
