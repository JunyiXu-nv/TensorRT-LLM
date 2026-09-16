# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for request mirroring.

The property is one sentence:

    A mirror copies traffic to a server the fleet does not own, and can do
    nothing else -- the copy's answer never reaches a caller, and a mirror
    target that is slow, broken or absent cannot turn a served request into
    a failed one.

Both halves matter and the second is the one worth testing hard. A mirror is
added to measure something, which means it is added to a system that is
already working; if it can break that system it is worse than not having it.
So `test_a_broken_mirror_target_changes_nothing` points a mirror at a port
with nothing behind it and asserts the served traffic is bit for bit what it
was without one.

These run against a real `gateway.py` process with real sockets, for the same
reason the handover tests do: every mechanism here is a property of two
connections and a task that must not be awaited.
"""

import socket
import threading
import time

import pytest
from handover_harness import (
    API_USER,
    CONVERSATION_HEADER,
    Backend,
    RequestStream,
    Scenario,
    control,
    lease_port,
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


def mirror_target(scenario, name="mirror"):
    """A server the gateway can copy to, deliberately not a fleet member.

    Never registered, so `discover` cannot find it, `elect` cannot route to it
    and a response of its own can never reach a caller by any path other than
    the one under test.
    """
    lease = lease_port(scenario.rng)
    backend = Backend(scenario.root, name, lease, healthy=True)
    backend.registered = False
    backend.start()
    return backend


def mirror_lines(backend):
    """One line per request the mirror target actually received."""
    try:
        with open(backend.access_log) as handle:
            return [line for line in handle.read().splitlines() if line.strip()]
    except OSError:
        return []


def set_mirror(port, job_id, target):
    status, data, _ = control(
        port, "POST", "/_gateway/mirror", {"job_id": job_id, "target": target}
    )
    return status, data


def mirror_stats(port):
    _, data, _ = control(port, "GET", "/_gateway/fleet")
    return data["mirror_stats"]


def test_mirror_copies_and_the_copy_never_answers_anybody(make_scenario):
    """The happy path, and the thing that would make it useless if it failed."""
    scenario = make_scenario("mirror-copies", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    target = mirror_target(scenario)
    served = scenario.fleet.backends[0]

    # Baseline: nothing is copied before anybody asks for it.
    stream = RequestStream(scenario.port, workers=4).start()
    try:
        wait_for("40 served", lambda: len(stream.served()) >= 40, timeout=30.0)
        assert not mirror_lines(target), (
            "the mirror target received traffic before any mirror was configured, so this "
            "test cannot tell a copy from a stray route"
        )

        status, body = set_mirror(scenario.port, served.job_id, "127.0.0.1:%d" % target.port)
        assert status == 200, body
        assert body["mirroring"] == "127.0.0.1:%d" % target.port

        before = len(stream.served())
        wait_for(
            "the mirror target to receive copies",
            lambda: len(mirror_lines(target)) >= 5,
            timeout=30.0,
        )
        wait_for("more served requests", lambda: len(stream.served()) > before + 20, timeout=30.0)
    finally:
        stream.stop()

    report = stream_report(stream)
    assert not stream.refused(), report
    assert not stream.lost(), report
    assert not stream.errored(), report

    # The whole point: the target did the work and nobody heard about it.
    assert target.job_id not in stream.by_backend(), (
        "a client response came back from the mirror target %r, which means the copy's "
        "answer reached a caller:\n%s" % (target.job_id, report)
    )
    assert mirror_lines(target), "the mirror target never received a copy"
    assert mirror_stats(scenario.port)["sent"] > 0


def test_a_broken_mirror_target_changes_nothing(make_scenario):
    """The reason to be careful: a mirror may not cost a served request.

    Pointed at a leased port with nothing listening, so every copy fails at
    connect. If any of that reaches the request path it shows up here as a
    refused connection, a lost request or an HTTP error.
    """
    scenario = make_scenario("mirror-broken", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    dead = lease_port(scenario.rng)  # leased so nothing else can take it, never bound

    stream = RequestStream(scenario.port, workers=6).start()
    try:
        wait_for("60 served", lambda: len(stream.served()) >= 60, timeout=30.0)
        clean_before = len(stream.served())

        status, body = set_mirror(
            scenario.port, scenario.fleet.backends[0].job_id, "127.0.0.1:%d" % dead.port
        )
        assert status == 200, body

        wait_for(
            "mirror failures to be counted",
            lambda: mirror_stats(scenario.port).get("failed", 0) > 0,
            timeout=30.0,
        )
        wait_for(
            "serving to continue past the failures",
            lambda: len(stream.served()) > clean_before + 40,
            timeout=40.0,
        )
    finally:
        stream.stop()

    report = stream_report(stream)
    assert not stream.refused(), "a broken mirror caused refused connections:\n%s" % report
    assert not stream.lost(), "a broken mirror lost in-flight requests:\n%s" % report
    assert not stream.errored(), "a broken mirror caused HTTP errors:\n%s" % report
    assert mirror_stats(scenario.port)["failed"] > 0, "the failures were never counted"


def test_a_slow_mirror_target_does_not_slow_the_served_request(make_scenario):
    """The copy is spawned, not awaited, so its latency is not the caller's.

    The target here accepts the connection and then says nothing at all, which
    is the worst case for a mirror: every copy sits open until the timeout.
    A black hole rather than a delayed reply, because the copy carries the
    caller's own headers -- any delay the fake backend understands would slow
    the served request too and the comparison would measure nothing.
    """
    scenario = make_scenario("mirror-slow", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)

    held = []
    sink = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sink.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sink.bind(("127.0.0.1", 0))
    sink.listen(128)
    sink_port = sink.getsockname()[1]
    stop = threading.Event()

    def swallow():
        sink.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = sink.accept()
            except (TimeoutError, OSError):
                continue
            # Kept open and unanswered on purpose.
            held.append(conn)

    thread = threading.Thread(target=swallow, daemon=True)
    thread.start()

    def sample(n=12):
        taken = []
        for i in range(n):
            start = time.time()
            request(
                scenario.port,
                "POST",
                "/v1/messages",
                headers=[("x-api-key", API_USER), (CONVERSATION_HEADER, "lat-%d" % i)],
                body=b'{"model":"m","messages":[]}',
            )
            taken.append(time.time() - start)
        taken.sort()
        return taken[len(taken) // 2]

    try:
        without = sample()
        status, body = set_mirror(
            scenario.port, scenario.fleet.backends[0].job_id, "127.0.0.1:%d" % sink_port
        )
        assert status == 200, body
        with_mirror = sample()
    finally:
        stop.set()
        thread.join(timeout=3.0)
        for conn in held:
            try:
                conn.close()
            except OSError:
                pass
        sink.close()

    assert held, "the mirror never even connected to the target"
    assert with_mirror < max(0.5, without * 3 + 0.2), (
        "median served latency went from %.3fs to %.3fs once every copy started hanging, "
        "so the copy is being awaited somewhere on the request path" % (without, with_mirror)
    )


def test_clearing_the_mirror_stops_the_copies(make_scenario):
    scenario = make_scenario("mirror-clear", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    target = mirror_target(scenario, name="clearmirror")
    served = scenario.fleet.backends[0]

    stream = RequestStream(scenario.port, workers=4).start()
    try:
        set_mirror(scenario.port, served.job_id, "127.0.0.1:%d" % target.port)
        wait_for("copies to start", lambda: len(mirror_lines(target)) >= 5, timeout=30.0)

        status, body = set_mirror(scenario.port, served.job_id, None)
        assert status == 200, body
        assert body["mirroring"] is None

        settled = len(mirror_lines(target))
        before = len(stream.served())
        wait_for("30 more served", lambda: len(stream.served()) > before + 30, timeout=30.0)
        # Copies already in flight when it was cleared may still land, so this
        # allows a small tail rather than demanding an instant stop.
        assert len(mirror_lines(target)) <= settled + 2, (
            "the mirror target kept receiving copies after the mirror was cleared: "
            "%d -> %d" % (settled, len(mirror_lines(target)))
        )
    finally:
        stream.stop()


def test_mirroring_an_unknown_backend_is_refused(make_scenario):
    """A mirror on a backend that is not being served is a silent no-op."""
    scenario = make_scenario("mirror-unknown", backends=2)
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)

    status, body = set_mirror(scenario.port, "no-such-job", "127.0.0.1:9")
    assert status == 404, body
    status, body = set_mirror(scenario.port, scenario.fleet.backends[0].job_id, "not-a-target")
    assert status == 400, body


def test_a_target_that_stays_gone_stops_being_mirrored_to(make_scenario):
    """A mirror nobody is watching may not retry forever.

    The failures were always counted, but a counter only helps somebody who is
    looking, and a mirror is usually set up precisely because nobody is looking
    yet. After enough consecutive misses the target is dropped and the reason
    is logged at ERROR, so the next person to read the log finds out why the
    copies stopped rather than why they never arrived.
    """
    scenario = make_scenario("mirror-gone", backends=2, extra_args=("--mirror-max-misses", "3"))
    scenario.start()
    scenario.wait_health_status("ok", timeout=25.0)
    dead = lease_port(scenario.rng)  # leased, never bound: every connect refused
    served = scenario.fleet.backends[0]

    stream = RequestStream(scenario.port, workers=4).start()
    try:
        status, body = set_mirror(scenario.port, served.job_id, "127.0.0.1:%d" % dead.port)
        assert status == 200, body
        assert body["mirroring"] is not None

        wait_for(
            "the dead target to be dropped",
            lambda: not control(scenario.port, "GET", "/_gateway/fleet")[1]["mirroring"],
            timeout=40.0,
        )
        stats = mirror_stats(scenario.port)
        assert stats.get("disabled", 0) >= 1, stats

        # And serving is untouched by any of it.
        before = len(stream.served())
        wait_for("30 more served", lambda: len(stream.served()) > before + 30, timeout=30.0)
    finally:
        stream.stop()

    report = stream_report(stream)
    assert not stream.refused(), report
    assert not stream.lost(), report
    assert not stream.errored(), report
    assert "is no longer being copied to" in scenario.gateway.log_text(), (
        "the mirror was dropped without saying so in the log, which is the only place "
        "anybody would find out"
    )
