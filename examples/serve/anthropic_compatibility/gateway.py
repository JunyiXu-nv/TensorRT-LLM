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
"""Stable front door for the Anthropic-compatibility server.

A serving job lives for hours and lands on whatever node the scheduler gives
it, so its URL changes every time it is rescheduled. Agent CLIs need one
address that never changes. This gateway holds that address and forwards to
whichever backend is currently healthy, so `ANTHROPIC_BASE_URL` is written once
and never edited again.

    python3 gateway.py --fleet-dir /shared/fleet --users gateway_users.txt --no-relay

Backends announce themselves by dropping one JSON file per job into
`--fleet-dir` (see `discover` for the fields). Nothing else couples the gateway
to the jobs: it never talks to the launcher on the request path, so an
unhealthy or missing registry only costs routing, never correctness.

Relay mode (`--yaml` without `--no-relay`) additionally submits the successor
job before the current one hits its wall clock and reclaims the predecessor
once the successor has proven itself. That half drives the serving jobs through
a launcher script exposing `submit`, `restart` and `quit` subcommands, passed
with `--serve-sh`. That script is not part of this example yet, so relay mode
refuses to start without one rather than failing at the first handover.

Standard library only, on purpose: the gateway has to outlive every serving
job, so it runs outside the TRT-LLM container on whatever long-lived host is
available. Requiring httpx or uvicorn there would mean a venv, which means
outbound network -- one more thing that host has to provide.

Reading the access log
----------------------
Every line carries `rid=rNNNNNN`; grepping one id gives that request end to
end, which is the only way to read the log at all once requests overlap. The
status field is the backend's own status plus at most two suffixes:

    200     relayed intact
    200!    the stream had no terminal event, so the gateway appended one
    200?    the ending could not be delivered; the client had already gone
    502     the BACKEND failed -- it never answered, or answered unusably

The distinction between `200?` and `502` is the point. Both used to be logged
as 502, so a client that hung up mid-stream was indistinguishable from a
serving job that had fallen over, and the counts blamed the backend for both.
`side=` on each warning names which end of the proxy actually broke.
"""

import argparse
import asyncio
import collections
import glob
import itertools
import json
import logging
import math
import os
import re
import sys
import time

LOG = logging.getLogger("gateway")

# Hop-by-hop request headers plus the ones this gateway owns. Request framing
# headers (content-length, transfer-encoding, te, trailer) deliberately survive:
# request bodies are relayed byte for byte. SSE response framing is normalized
# separately so the gateway can append a valid terminal error event.
STRIP_REQUEST_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "upgrade",
    "host",
    "x-api-key",
    "authorization",
    "accept-encoding",
    # Owned by the gateway: it is the only identity the backend sees, so a
    # client copy must not ride along beside the one written below.
    "x-gateway-user",
}

MAX_HEAD_BYTES = 64 * 1024
RELAY_CHUNK = 64 * 1024
PENDING_VISIBILITY_GRACE = 60

# Identity used when a client sends no key at all. It is an ordinary allowlist
# entry, not a bypass: keyless requests are refused unless this name is listed
# in the users file, and the access log then attributes them all to one shared
# identity.
ANONYMOUS_USER = "anonymous"

# Anthropic-shaped so a client's error handling reports something meaningful
# instead of a bare transport failure.
ERROR_BODIES = {
    401: (
        "authentication_error",
        "unknown api key; ask the gateway owner to add your username to the users file",
    ),
    404: ("not_found_error", "no such gateway endpoint"),
    502: ("api_error", "backend refused the connection"),
    503: (
        "overloaded_error",
        "no healthy backend right now; the serving job is rotating, retry shortly",
    ),
}

ERROR_REASONS = {
    401: "Unauthorized",
    404: "Not Found",
    502: "Bad Gateway",
    503: "Service Unavailable",
}

SSE_ROTATED = (
    b"event: error\n"
    b'data: {"type":"error","error":{"type":"overloaded_error",'
    b'"message":"backend rotated mid-stream; the response is '
    b'incomplete, please resend"}}\n\n'
)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
# Monotonic rather than random: request ids exist to be read in a log file, and
# increasing ones also show how many requests the gateway has taken since it
# started. Wrapped so the field stays a fixed width in the access log.
_REQUEST_IDS = itertools.count(1)


def next_request_id():
    return "r%06d" % (next(_REQUEST_IDS) % 1000000)


class SideError(OSError):
    """An I/O failure tagged with the side of the proxy that produced it.

    The relay reads from the backend and writes to the client inside a single
    try block. An untagged OSError from one side is indistinguishable from the
    other, and the handlers used to report every one of them as "upstream
    failed" -- which was simply wrong whenever the client was the end that went
    away. A post-mortem then blames the serving job for a client disconnect.

    Deliberately an OSError subclass rather than a new branch of the exception
    hierarchy: every handler on the request path already catches OSError, so
    adding the tag can never turn a handled failure into an escaped one. The
    tag is purely additive, and code that does not care reads it as before.
    """

    def __init__(self, side, exc):
        super().__init__("%s %s: %s" % (side, type(exc).__name__, exc))
        self.side = side
        self.cause = exc


def error_side(exc, default="unknown"):
    """Name the side an exception came from, for exceptions that carry no tag.

    The default is "unknown" rather than a plausible guess on purpose. Guessing
    is what produced the misleading logs this machinery replaces: an untagged
    failure reported as the likelier side reads exactly like a measured one,
    and there is no way to tell them apart afterwards. A caller that can prove
    the side from its position in the code passes it explicitly.
    """
    return getattr(exc, "side", default)


async def read_upstream(reader, size):
    """Read from the backend, tagging failures with the side."""
    try:
        return await reader.read(size)
    except OSError as exc:
        raise SideError("upstream_read", exc) from exc


async def write_client(writer, data):
    """Write to the downstream client, tagging failures with the side."""
    try:
        writer.write(data)
        await writer.drain()
    except OSError as exc:
        raise SideError("client_write", exc) from exc


class RequestTrace:
    """Per-request diagnostic state, populated as the request progresses.

    Failure logs used to name only the exception. That is not enough to analyse
    anything after the fact: warnings from concurrent requests interleave with
    nothing to correlate them, the side that failed is unrecorded, and both the
    backend's own status code and the stream's progress are discarded at the
    moment they become interesting. Every failure path emits `detail()`, so a
    single grep on the id reconstructs one request end to end.
    """

    __slots__ = ("rid", "upstream_status", "tracker", "request_body_error")

    def __init__(self):
        self.rid = next_request_id()
        self.upstream_status = None
        self.tracker = None
        self.request_body_error = None

    def detail(self):
        parts = ["rid=%s" % self.rid]
        if self.upstream_status is not None:
            parts.append("upstream_status=%s" % self.upstream_status)
        if self.tracker is not None:
            parts.append(self.tracker.progress())
        if self.request_body_error is not None:
            # A client that died mid-upload leaves the backend holding a
            # truncated request, and the backend's reaction to that reads like
            # a server bug unless this says otherwise.
            parts.append("request_body=%s" % self.request_body_error)
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Fleet state
# ---------------------------------------------------------------------------
def finite_float(value, field, source=""):
    """Coerce to float, rejecting NaN and infinity.

    float() happily accepts "nan" and "inf", and NaN then poisons every
    comparison it touches: `now - nan > stale_after` is False, so a dead
    backend with a NaN heartbeat never goes stale and is never retired, while
    a NaN end_time makes the max() in the election return an arbitrary winner.
    Both fail silently, which is why they are rejected at the boundary.
    """
    number = float(value or 0)
    if not math.isfinite(number):
        raise ValueError("%s is not finite (%r)%s" % (field, value, source))
    return number


class Backend:
    """One serving job, as seen through its registration file."""

    def __init__(self, record):
        self.job_id = str(record["job_id"])
        self.url = record["url"].rstrip("/")
        self.run_dir = record.get("run_dir", "")
        self.state = record.get("state", "")
        self.end_time = finite_float(record.get("end_time"), "end_time")
        self.heartbeat = finite_float(record.get("heartbeat"), "heartbeat")
        self.healthy = False
        self.timeouts = 0  # consecutive probe timeouts
        self.healthy_since = 0.0
        # Probing resolves the URL once; every request reuses host/port.
        match = re.match(r"^http://([^:/]+):(\d+)$", self.url)
        if not match:
            raise ValueError("unusable url %r" % self.url)
        self.host = match.group(1)
        self.port = int(match.group(2))

    def refresh(self, record):
        """Take every mutable field, not just state and heartbeat.

        A registration file is rewritten as the job learns about itself. Two
        fields in particular arrive late, so latching the first value read is
        not enough: `end_time`, absent until the job resolves its wall clock -
        and `end_time == 0` makes the supervisor skip relay entirely, so the
        fleet would expire with no successor queued, which is the one failure
        the relay exists to prevent. And `url`, which changes when a failed
        successor is restarted in place and binds a different port, leaving the
        gateway proxying a dead address while a fresh heartbeat keeps the entry
        from ever being retired.
        """
        self.state = record.get("state", self.state)
        self.heartbeat = finite_float(record.get("heartbeat"), "heartbeat")
        self.end_time = finite_float(record.get("end_time") or self.end_time, "end_time")
        self.run_dir = record.get("run_dir", self.run_dir)
        # Normalised the same way as in __init__, so a rewrite that differs
        # only by a trailing slash does not read as a move.
        url = (record.get("url") or "").rstrip("/")
        if url and url != self.url:
            match = re.match(r"^http://([^:/]+):(\d+)$", url)
            if not match:
                LOG.warning("ignoring unusable url %r for %s", url, self.job_id)
            else:
                LOG.info("%s moved from %s to %s", self.job_id, self.url, url)
                self.url = url
                self.host = match.group(1)
                self.port = int(match.group(2))


class Fleet:
    """Everything the request path and the supervisor share."""

    def __init__(self, args):
        self.args = args
        self.backends = {}  # job_id -> Backend
        self.active = None  # job_id currently taking new requests
        self.draining = {}  # job_id -> reclaim deadline (unix ts)
        # A default dict on purpose. Requests outlive their backend's entry --
        # discovery can retire a job while its streams are still draining -- and
        # the release below runs in a finally that also closes the client
        # socket. A KeyError there would leak the connection, so counting must
        # not be able to raise.
        self.inflight = collections.defaultdict(int)
        self.users = set()
        self.users_mtime = 0.0
        # (job_id, submitted_at), retained through registration until healthy.
        self.pending = None
        self.ever_active = False  # distinguish recovery from initial startup
        # Replaced, but not yet cleared for reclaim: draining ends in a `quit`,
        # so it waits until the successor has proven itself.
        self.superseded = set()
        self.started = time.time()

    # -- users ------------------------------------------------------------
    def reload_users(self):
        path = self.args.users
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            if self.users:
                LOG.warning(
                    "users file disappeared: %s (keeping %d entries)", path, len(self.users)
                )
            return
        if mtime == self.users_mtime:
            return
        names = set()
        with open(path) as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if line:
                    names.add(line)
        self.users_mtime = mtime
        if names != self.users:
            LOG.info("users reloaded: %d entries", len(names))
        self.users = names

    # -- discovery --------------------------------------------------------
    def discover(self):
        """Rebuild the backend table from the registration directory.

        Each serving job owns exactly one file named after its scheduler job id,
        so there is never more than one writer per file and the gateway never
        has to coordinate with anybody. The union is just the directory listing.
        """
        now = time.time()
        seen = set()
        for path in glob.glob(os.path.join(self.args.fleet_dir, "*.json")):
            try:
                with open(path) as handle:
                    record = json.load(handle)
            except (OSError, ValueError):
                # Mid-rename or truncated. The writer replaces the file
                # atomically, so the next sweep gets a whole one. Stay quiet:
                # this is expected and would otherwise log on every sweep.
                continue
            # A registration has to be a JSON object. Valid JSON that is not one
            # (a bare array, string or number) would otherwise reach .get() and
            # raise AttributeError, which is not a coercion error and escapes
            # the guard below -- aborting the sweep before the retirement pass.
            if not isinstance(record, dict):
                LOG.warning("ignoring %s: registration is not a JSON object", path)
                continue
            try:
                job_id = str(record.get("job_id", ""))
                heartbeat = finite_float(record.get("heartbeat"), "heartbeat")
            except (ValueError, TypeError) as exc:
                # One unusable field must not cost every other backend its
                # sweep, so the failure stays local to this file.
                LOG.warning("ignoring %s: %s", path, exc)
                continue
            if not job_id:
                continue
            if now - heartbeat > self.args.stale_after:
                continue
            seen.add(job_id)
            if job_id in self.backends:
                try:
                    self.backends[job_id].refresh(record)
                except (ValueError, TypeError, AttributeError) as exc:
                    # Same exposure as construction: keep a bad rewrite local to
                    # its own file and hold the last good view of the backend.
                    LOG.warning("ignoring refresh from %s: %s", path, exc)
            else:
                try:
                    self.backends[job_id] = Backend(record)
                # AttributeError covers a non-string url, which reaches
                # .rstrip() in __init__. One unusable record must cost only
                # itself, never the rest of the sweep.
                except (KeyError, ValueError, TypeError, AttributeError) as exc:
                    LOG.warning("ignoring %s: %s", path, exc)
                    continue
                self.inflight.setdefault(job_id, 0)
                LOG.info(
                    "backend appeared: %s at %s (ends %s)",
                    job_id,
                    self.backends[job_id].url,
                    fmt_time(self.backends[job_id].end_time),
                )

        for job_id in [j for j in self.backends if j not in seen]:
            LOG.info("backend gone: %s (no heartbeat for %ds)", job_id, self.args.stale_after)
            self.backends.pop(job_id, None)
            self.draining.pop(job_id, None)
            self.superseded.discard(job_id)
            # Keep the counter while anything is still streaming off this
            # backend; a later sweep collects it once the count reaches zero.
            if not self.inflight.get(job_id):
                self.inflight.pop(job_id, None)

        # Discovery, probing and supervision run on independent timers, so
        # retiring a backend has to clear the pointer to it here rather than
        # waiting for the next election. Otherwise `active` names a job that is
        # no longer in the table, and everything that dereferences it raises.
        if self.active is not None and self.active not in self.backends:
            LOG.warning("active backend %s retired; serving 503", self.active)
            self.active = None

    # -- election ---------------------------------------------------------
    def elect(self):
        """Pick the healthy backend that will live the longest.

        Choosing by end time is what makes relay work without anybody
        orchestrating it: a freshly started job outlives the one it replaces,
        so the moment it passes /health it wins the election on its own.
        """
        candidates = [j for j, b in self.backends.items() if b.healthy]
        if self.pending and self.pending[0] in candidates:
            LOG.info("successor %s is healthy", self.pending[0])
            self.pending = None
        winner = max(candidates, key=lambda j: self.backends[j].end_time) if candidates else None
        if winner == self.active:
            return
        previous = self.active
        self.active = winner
        if winner is None:
            LOG.warning("no healthy backend; serving 503")
        else:
            self.ever_active = True
            LOG.info("active backend -> %s (%s)", winner, self.backends[winner].url)
            # Won the election back: whatever replaced it is gone or sicker, so
            # it is no longer a candidate for reclaim.
            self.superseded.discard(winner)
            if winner in self.draining:
                self.draining.pop(winner, None)
                LOG.info("cancelled drain of re-elected backend %s", winner)

        # Only a forward handover marks the predecessor. Falling back to an
        # older job after the active backend fails is reversible: the newer job
        # may merely be restarting and must remain eligible to win back routing.
        if winner is not None and previous and previous in self.backends:
            if self.backends[winner].end_time > self.backends[previous].end_time:
                self.superseded.add(previous)
                LOG.info("superseded %s; reclaim held until %s is stable", previous, winner)
            else:
                LOG.warning(
                    "failed back from %s to older backend %s; keeping %s available for recovery",
                    previous,
                    winner,
                    previous,
                )


def fmt_time(ts):
    if not ts:
        return "unknown"
    return time.strftime("%H:%M:%S", time.localtime(ts))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
async def read_head(reader):
    """Read up to and including the blank line ending an HTTP head."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = await reader.read(8192)
        if not chunk:
            return None, b""
        buf += chunk
        if len(buf) > MAX_HEAD_BYTES:
            raise ValueError("head exceeds %d bytes" % MAX_HEAD_BYTES)
    head, _, rest = buf.partition(b"\r\n\r\n")
    return head, rest


def parse_request_head(head):
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3:
        raise ValueError("malformed request line: %r" % lines[0])
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep:
            raise ValueError("malformed header: %r" % line)
        headers.append((name.strip(), value.strip()))
    return parts[0], parts[1], headers


def header_value(headers, name):
    name = name.lower()
    for key, value in headers:
        if key.lower() == name:
            return value
    return None


def extract_key(headers):
    key = header_value(headers, "x-api-key")
    if key:
        return key.strip()
    auth = header_value(headers, "authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def build_response(status, reason, body, extra_headers=()):
    head = [
        "HTTP/1.1 %d %s" % (status, reason),
        "Content-Type: application/json",
        "Content-Length: %d" % len(body),
        "Connection: close",
    ]
    head.extend(extra_headers)
    return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body


def error_response(status, retry_after=None):
    kind, message = ERROR_BODIES[status]
    body = json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode()
    extra = ["Retry-After: %d" % retry_after] if retry_after else []
    return build_response(status, ERROR_REASONS[status], body, extra)


def json_response(payload):
    body = json.dumps(payload, indent=2).encode()
    return build_response(200, "OK", body)


def parse_response_head(head):
    lines = head.decode("latin-1").split("\r\n")
    if not lines or not lines[0].startswith("HTTP/"):
        raise ValueError("malformed upstream status line")
    headers = []
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep:
            raise ValueError("malformed upstream header: %r" % line)
        headers.append((name.strip(), value.strip()))
    return lines[0], headers


def rewrite_sse_head(status_line, headers):
    """Make downstream SSE framing independent from the upstream framing."""
    owned = {"connection", "content-length", "keep-alive", "trailer", "transfer-encoding"}
    lines = [status_line]
    lines.extend("%s: %s" % (name, value) for name, value in headers if name.lower() not in owned)
    lines.extend(("Transfer-Encoding: chunked", "Connection: close"))
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def chunk_frame(payload):
    return b"%x\r\n%s\r\n" % (len(payload), payload)


class SseTracker:
    """Recognize a terminal event across arbitrary transport reads.

    Two stream dialects reach this gateway. Anthropic ends a stream with a
    named `event: message_stop`; OpenAI's chat-completions stream has no event
    names at all and ends with a bare `data: [DONE]` sentinel. Recognising only
    the Anthropic form makes every OpenAI stream look truncated, and the caller
    then appends an Anthropic-shaped error event to a stream that had in fact
    completed normally -- which is worse than the missing terminal it was meant
    to repair, because it corrupts a good response.
    """

    def __init__(self):
        self.buffer = bytearray()
        self.current_event = None
        self.saw_stop = False
        self.saw_error = False
        # Progress counters. A truncated stream is only diagnosable if the log
        # says how far it got: "died before the first event" and "died after
        # 400 events" have entirely different causes, and the exception text
        # alone cannot tell them apart. These are counted on the way past, so
        # they cost one add per line already being parsed.
        self.bytes = 0
        self.events = 0
        self.last_event = None
        self.last_data = b""

    @property
    def terminal(self):
        return self.saw_stop or self.saw_error

    def progress(self):
        return "bytes=%d events=%d last_event=%s" % (
            self.bytes,
            self.events,
            self.last_event.decode("latin-1", "replace") if self.last_event else "-",
        )

    def preview(self):
        """Last data line, truncated. Response content, so debug level only."""
        return self.last_data.decode("latin-1", "replace")

    def feed(self, payload):
        self.bytes += len(payload)
        self.buffer.extend(payload)
        while True:
            newline = self.buffer.find(b"\n")
            if newline < 0:
                return
            line = bytes(self.buffer[:newline]).rstrip(b"\r")
            del self.buffer[: newline + 1]
            if not line:
                if self.current_event == b"message_stop":
                    self.saw_stop = True
                elif self.current_event == b"error":
                    self.saw_error = True
                self.current_event = None
                continue
            name, sep, value = line.partition(b":")
            if not sep:
                continue
            if name == b"event":
                self.current_event = value.lstrip(b" ")
                self.last_event = self.current_event
            elif name == b"data":
                # Both dialects carry exactly one data line per event, so this
                # counts events without needing a separate state machine.
                self.events += 1
                self.last_data = value.strip()[:120]
                if value.strip() == b"[DONE]":
                    # OpenAI's terminal sentinel. It carries no event name, so
                    # it has to be matched on the data line itself.
                    self.saw_stop = True


class BufferedUpstream:
    """Expose bytes already read with the response head before the socket."""

    def __init__(self, reader, initial):
        self.reader = reader
        self.buffer = bytearray(initial)

    async def read(self, size):
        if self.buffer:
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data
        return await read_upstream(self.reader, size)

    async def read_exact(self, size):
        parts = []
        remaining = size
        while remaining:
            data = await self.read(remaining)
            if not data:
                return b"".join(parts), False
            parts.append(data)
            remaining -= len(data)
        return b"".join(parts), True

    async def read_line(self):
        while True:
            newline = self.buffer.find(b"\r\n")
            if newline >= 0:
                line = bytes(self.buffer[:newline])
                del self.buffer[: newline + 2]
                return line
            if len(self.buffer) > MAX_HEAD_BYTES:
                raise ValueError("upstream framing line is too long")
            data = await read_upstream(self.reader, 8192)
            if not data:
                return None
            self.buffer.extend(data)


async def emit_sse_payload(writer, tracker, payload):
    if not payload:
        return
    tracker.feed(payload)
    await write_client(writer, chunk_frame(payload))


async def relay_chunked_sse(source, writer, tracker):
    while True:
        line = await source.read_line()
        if line is None:
            return False
        try:
            size = int(line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            LOG.warning("invalid upstream chunk size: %r", line)
            return False
        if size < 0:
            return False
        if size == 0:
            # Consume trailers, but do not forward them: rewrite_sse_head removes
            # Trailer and the gateway owns the downstream terminal chunk.
            while True:
                trailer = await source.read_line()
                if trailer is None:
                    return False
                if not trailer:
                    return True

        remaining = size
        while remaining:
            payload = await source.read(min(RELAY_CHUNK, remaining))
            if not payload:
                return False
            remaining -= len(payload)
            await emit_sse_payload(writer, tracker, payload)
        ending, complete = await source.read_exact(2)
        if not complete or ending != b"\r\n":
            return False


async def relay_sized_sse(source, writer, tracker, size):
    remaining = size
    while remaining:
        payload = await source.read(min(RELAY_CHUNK, remaining))
        if not payload:
            return False
        remaining -= len(payload)
        await emit_sse_payload(writer, tracker, payload)
    return True


async def relay_close_delimited_sse(source, writer, tracker):
    while True:
        payload = await source.read(RELAY_CHUNK)
        if not payload:
            return True
        await emit_sse_payload(writer, tracker, payload)


# ---------------------------------------------------------------------------
# Request path
# ---------------------------------------------------------------------------
class Gateway:
    """Terminates client connections and forwards them to the active backend."""

    def __init__(self, fleet):
        self.fleet = fleet

    async def handle(self, reader, writer):
        peer = writer.get_extra_info("peername")
        started = time.time()
        trace = RequestTrace()
        try:
            head, rest = await read_head(reader)
            if head is None:
                return
            method, path, headers = parse_request_head(head)
        # OSError, not just ConnectionError: reader.read raises other OSError
        # subclasses (a socket timeout, for one). ConnectionError is itself an
        # OSError subclass, so this only widens. Letting one escape means the
        # task dies with an unretrieved exception and close(writer) never runs,
        # leaking the connection.
        except (ValueError, OSError) as exc:
            # INFO, not DEBUG: the gateway runs at INFO, so an unparseable
            # client request used to leave no trace whatsoever -- and that is
            # the single failure mode a client-integration bug presents as.
            LOG.info("dropped rid=%s from %s: %s", trace.rid, peer, exc)
            await close(writer)
            return

        if path.startswith("/_gateway/"):
            await self.serve_introspection(path, headers, writer)
            return

        key = extract_key(headers) or ANONYMOUS_USER
        if key not in self.fleet.users:
            LOG.info("401 %s %s user=%r [rid=%s]", method, path, key, trace.rid)
            await respond(writer, error_response(401))
            return

        # Resolved once. The election loop may move `active` while this request
        # is in flight; everything below must keep talking about the same
        # backend, or the inflight count is incremented on one and decremented
        # on another. Read through .get(): this runs outside the try below, so
        # a lookup that raises here would leak the client socket.
        job_id = self.fleet.active
        backend = self.fleet.backends.get(job_id) if job_id else None
        if backend is None:
            LOG.info(
                "503 %s %s user=%s (no backend) [rid=%s]", method, path, key, trace.rid
            )
            await respond(writer, error_response(503, retry_after=20))
            return
        self.fleet.inflight[job_id] += 1
        status = "-"
        # Closing the client socket sits in its own finally so that no amount of
        # bookkeeping trouble above can leak the connection.
        try:
            try:
                status = await self.proxy(
                    backend, method, path, headers, rest, reader, writer, key, trace
                )
            # ValueError covers an unusable upstream head: read_head raises it
            # past MAX_HEAD_BYTES and parse_response_head on a malformed one.
            # Both happen before anything is written downstream, so 502 is
            # safe here -- without it the client just sees the socket close
            # with no status at all. Framing failures after the head is out are
            # caught in relay_response, which cannot use this path.
            except (ConnectionError, OSError, ValueError) as exc:
                # `side` distinguishes a backend that never answered from one
                # whose response could not be handed to the client. Both used
                # to be logged as "upstream failed" and counted as 502, which
                # blamed the serving job for client-side disconnects.
                # "upstream" is the justified default HERE and only here: every
                # untagged failure that can reach this handler is backend-side
                # (open_connection, the request head write, or read_head on the
                # upstream reader). All client writes happen inside
                # relay_response, which handles them itself.
                LOG.warning(
                    "upstream %s failed side=%s: %s [%s]",
                    backend.url,
                    error_side(exc, "upstream"),
                    exc,
                    trace.detail(),
                )
                status = "502"
                await respond(writer, error_response(502))
            finally:
                self.fleet.inflight[job_id] -= 1
                LOG.info(
                    "%s %s %s user=%s backend=%s %.1fs [%s]",
                    status,
                    method,
                    path,
                    key,
                    job_id,
                    time.time() - started,
                    trace.detail(),
                )
        finally:
            await close(writer)

    async def serve_introspection(self, path, headers, writer):
        if path == "/_gateway/health":
            # Deliberately unauthenticated: whatever watches the gateway from
            # outside has no reason to hold an allowlist entry, and the answer
            # names no user and no request.
            healthy = self.fleet.active is not None
            payload = {
                "status": "ok" if healthy else "no_backend",
                "active": self.fleet.active,
                "uptime_s": round(time.time() - self.fleet.started),
            }
            await respond(writer, json_response(payload))
            return
        if path == "/_gateway/fleet":
            if extract_key(headers) not in self.fleet.users:
                await respond(writer, error_response(401))
                return
            now = time.time()
            payload = {
                "active": self.fleet.active,
                "pending_successor": self.fleet.pending[0] if self.fleet.pending else None,
                "backends": {
                    job_id: {
                        "url": b.url,
                        "healthy": b.healthy,
                        "healthy_for_s": round(now - b.healthy_since) if b.healthy_since else None,
                        "probe_timeouts": b.timeouts,
                        "state": b.state,
                        "ends_at": fmt_time(b.end_time),
                        "ends_in_s": round(b.end_time - now),
                        "last_beat_s": round(now - b.heartbeat, 1),
                        "inflight": self.fleet.inflight.get(job_id, 0),
                        "superseded": job_id in self.fleet.superseded,
                        "draining": job_id in self.fleet.draining,
                    }
                    for job_id, b in sorted(self.fleet.backends.items())
                },
            }
            await respond(writer, json_response(payload))
            return
        await respond(writer, error_response(404))

    async def proxy(self, backend, method, path, headers, rest, reader, writer, user, trace):
        up_reader, up_writer = await asyncio.open_connection(backend.host, backend.port)
        try:
            up_writer.write(self.upstream_head(backend, method, path, headers, user))
            if rest:
                up_writer.write(rest)
            await up_writer.drain()

            # Nothing here parses the request body. The pump runs until the
            # client stops sending or the response finishes, so content-length
            # and chunked bodies both work without being understood.
            pump = asyncio.create_task(relay(reader, up_writer, trace))
            try:
                return await self.relay_response(up_reader, writer, trace)
            finally:
                pump.cancel()
        finally:
            await close(up_writer)

    def upstream_head(self, backend, method, path, headers, user):
        # The request line is forwarded verbatim, which is what keeps the
        # gateway independent of the Anthropic surface the backend happens to
        # register. /v1/messages, /v1/messages/count_tokens and every Message
        # Batches route reach the backend without this file naming them, and a
        # route added later needs no change here. The only paths the gateway
        # claims for itself are under /_gateway/.
        lines = [
            "%s %s HTTP/1.1" % (method, path),
            "Host: %s:%d" % (backend.host, backend.port),
            # Close framing gives non-SSE responses an unambiguous EOF.
            # SSE responses are decoded and reframed below.
            "Connection: close",
            "Accept-Encoding: identity",
            "X-Gateway-User: %s" % user,
        ]
        for name, value in headers:
            if name.lower() in STRIP_REQUEST_HEADERS:
                continue
            lines.append("%s: %s" % (name, value))
        return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")

    async def relay_response(self, up_reader, writer, trace):
        head, rest = await read_head(up_reader)
        if head is None:
            raise ConnectionError("upstream closed before sending a response")
        status_line, headers = parse_response_head(head)
        parts = status_line.split(" ")
        if len(parts) < 2:
            raise ConnectionError("malformed upstream status line")
        status = parts[1]
        # Recorded before anything can fail downstream. The backend's own view
        # of the request is the first thing a post-mortem wants, and it used to
        # be discarded the moment the relay threw.
        trace.upstream_status = status
        content_type = header_value(headers, "content-type") or ""
        content_encoding = header_value(headers, "content-encoding") or ""
        transfer_encoding = header_value(headers, "transfer-encoding") or ""
        encodings = [
            value.strip().lower() for value in transfer_encoding.split(",") if value.strip()
        ]
        # Only SSE is reframed, and only SSE gets a terminal event appended.
        # Batch results come back as application/x-ndjson and count_tokens as
        # application/json, so both take the verbatim path below untouched.
        is_sse = "text/event-stream" in content_type.lower()
        supported_transfer = not encodings or encodings == ["chunked"]

        # Reframing an encoding the gateway does not decode would mix an
        # unencoded injected event into that stream. Accept-Encoding: identity
        # prevents content encoding for the normal server; retain raw relay as a
        # safe fallback for encoded bodies or unknown transfer codings.
        if not is_sse or content_encoding.lower() not in ("", "identity") or not supported_transfer:
            await write_client(writer, head + b"\r\n\r\n" + rest)
            while True:
                chunk = await read_upstream(up_reader, RELAY_CHUNK)
                if not chunk:
                    return status
                await write_client(writer, chunk)

        await write_client(writer, rewrite_sse_head(status_line, headers))
        source = BufferedUpstream(up_reader, rest)
        tracker = SseTracker()
        trace.tracker = tracker
        # The head is already on the wire, so a framing failure from here on
        # cannot become a 502: writing error_response would put a second HTTP
        # head inside a response the client is already reading. Degrade to the
        # truncated-stream path instead and let the terminal-event injection
        # below give the client a valid end to the stream.
        clean_end = False
        try:
            if "chunked" in encodings:
                clean_end = await relay_chunked_sse(source, writer, tracker)
            else:
                content_length = header_value(headers, "content-length")
                if content_length is None:
                    clean_end = await relay_close_delimited_sse(source, writer, tracker)
                else:
                    try:
                        length = int(content_length)
                    except ValueError:
                        length = -1
                    if length < 0:
                        clean_end = False
                    else:
                        clean_end = await relay_sized_sse(source, writer, tracker, length)
        # The response head is already on the wire, so a failure from here on
        # cannot become a 502 -- that would put a second HTTP head inside a
        # body the client is reading. OSError covers an upstream that dies
        # mid-stream, which is the common case during a handover. Both degrade
        # to the truncated path so the terminal event below closes the stream.
        #
        # side= is the whole point of the tag: this block reads the backend AND
        # writes the client, so the old unconditional "upstream framing failed"
        # was a guess that happened to be wrong every time the client was the
        # end that left.
        except (ValueError, OSError) as exc:
            LOG.warning(
                "sse relay stopped side=%s: %s [%s]",
                error_side(exc),
                exc,
                trace.detail(),
            )
            LOG.debug("last data line before failure: %s", tracker.preview())
            clean_end = False

        # Deliberately its own try. Everything below writes to the client, so
        # it can raise -- and it used to raise straight past this function into
        # the caller's handler, which appended exactly the second HTTP head the
        # block above exists to prevent. That went unnoticed only because the
        # client socket was already broken in every observed case, so the stray
        # 502 head was discarded by the kernel instead of corrupting a stream
        # the client was still reading. It also mislabelled the request 502
        # when the backend had in fact answered it, which made the access log
        # blame the serving job for client-side disconnects.
        try:
            if not tracker.terminal:
                ending = "clean end" if clean_end else "truncated upstream framing"
                LOG.warning(
                    "stream reached %s without message_stop or [DONE]; injecting error [%s]",
                    ending,
                    trace.detail(),
                )
                await emit_sse_payload(writer, tracker, SSE_ROTATED)
                status += "!"
            await write_client(writer, b"0\r\n\r\n")
        except (ValueError, OSError) as exc:
            # The client is gone; there is nobody left to tell. Record that the
            # ending never landed and let the caller log a delivered-partially
            # status rather than a backend failure.
            LOG.warning(
                "could not deliver stream ending side=%s: %s [%s]",
                error_side(exc),
                exc,
                trace.detail(),
            )
            status += "?"
        return status


async def relay(reader, writer, trace=None):
    """Pump the client's request body into the backend."""
    try:
        while True:
            chunk = await reader.read(RELAY_CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except asyncio.CancelledError:
        # Routine: the response finished first and proxy() cancels the pump.
        pass
    except (ConnectionError, OSError) as exc:
        # Not routine, and previously invisible -- this handler was a bare
        # `pass`. A client that dies mid-upload leaves the backend holding a
        # truncated request body, and the backend's reaction to that reads like
        # a server-side bug with nothing in the log to contradict it.
        if trace is not None:
            trace.request_body_error = "%s: %s" % (type(exc).__name__, exc)
        LOG.debug("request body pump failed: %s", exc)


async def respond(writer, payload):
    try:
        writer.write(payload)
        await writer.drain()
    except (ConnectionError, OSError):
        pass
    await close(writer)


async def close(writer):
    try:
        writer.close()
        await writer.wait_closed()
    except (ConnectionError, OSError):
        pass


# ---------------------------------------------------------------------------
# Background loops
# ---------------------------------------------------------------------------
async def probe(backend, timeout):
    """GET /health, classified into three outcomes rather than a boolean.

    "dead" and "timeout" look the same to a boolean probe but mean opposite
    things. A refused connection or an unresolvable host says the process is
    gone -- unambiguous, act at once. A timeout usually says the server is too
    busy to answer a health check, and a server that busy is normally still
    generating tokens fine; taking it out of rotation would turn "slow" into
    "503" with nowhere better to send the traffic.

    A non-200 is not ambiguous either: /health only fails when the engine
    reports itself broken, which trtllm-serve follows with a shutdown.
    """
    writer = None
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(backend.host, backend.port), timeout
        )
        writer.write(
            b"GET /health HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n"
            % backend.host.encode("latin-1")
        )
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout)
        return "ok" if b" 200 " in line else "dead"
    except asyncio.TimeoutError:
        return "timeout"
    except (ConnectionError, OSError):
        return "dead"
    finally:
        if writer is not None:
            await close(writer)


def apply_probe(backend, result, unhealthy_after):
    if result == "ok":
        backend.timeouts = 0
        if not backend.healthy:
            backend.healthy = True
            backend.healthy_since = time.time()
            LOG.info("backend %s healthy", backend.job_id)
        return
    if result == "dead":
        if backend.healthy:
            LOG.warning("backend %s unreachable; dropping it now", backend.job_id)
        backend.timeouts = 0
        backend.healthy = False
        backend.healthy_since = 0.0
        return
    backend.timeouts += 1
    if backend.healthy and backend.timeouts >= unhealthy_after:
        LOG.warning(
            "backend %s timed out %d times in a row; marking unhealthy",
            backend.job_id,
            backend.timeouts,
        )
        backend.healthy = False
        backend.healthy_since = 0.0
    elif backend.healthy:
        LOG.info(
            "backend %s health probe timed out (%d/%d)",
            backend.job_id,
            backend.timeouts,
            unhealthy_after,
        )


async def discovery_loop(fleet):
    while True:
        try:
            fleet.reload_users()
            fleet.discover()
        except Exception:
            LOG.exception("discovery failed")
        await asyncio.sleep(fleet.args.discover_interval)


async def health_loop(fleet):
    while True:
        try:
            backends = list(fleet.backends.values())
            if backends:
                results = await asyncio.gather(
                    *[probe(b, fleet.args.probe_timeout) for b in backends],
                    return_exceptions=True,
                )
                for backend, result in zip(backends, results):
                    if not isinstance(result, str):
                        result = "timeout"
                    apply_probe(backend, result, fleet.args.unhealthy_after)
            fleet.elect()
        except Exception:
            LOG.exception("health loop failed")
        await asyncio.sleep(fleet.args.health_interval)


async def run_serve_sh(fleet, *serve_args):
    proc = await asyncio.create_subprocess_exec(
        fleet.args.serve_sh,
        *serve_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace").strip()


async def run_slurm_command(*command):
    try:
        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )
    except OSError as exc:
        LOG.error("cannot run %s: %s", command[0], exc)
        return None, ""
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace").strip()


async def slurm_job_status(job_id):
    """Return (state, reason), ("GONE", ""), or None on query failure."""
    code, out = await run_slurm_command("squeue", "-h", "-j", job_id, "-o", "%T|%r")
    if code is not None and code != 0 and "invalid job id" in out.lower():
        return "GONE", ""
    if code is None or code != 0:
        LOG.warning("cannot query successor %s: %s", job_id, out)
        return None
    if not out:
        return "GONE", ""
    state, _, reason = out.splitlines()[0].partition("|")
    return state.strip().upper(), reason.strip()


async def submit_successor(fleet, now, label):
    code, out = await run_serve_sh(fleet, "submit", "--yaml", fleet.args.yaml, "--label", label)
    match = re.search(r"Submitted batch job (\d+)", out)
    if code == 0 and match:
        fleet.pending = (match.group(1), now)
        LOG.info("successor submitted: job %s", match.group(1))
        return True
    LOG.error("submit failed (rc=%d): %s", code, out)
    return False


async def supervise_pending(fleet, now):
    """Keep a submitted successor tracked until it is actually healthy."""
    if not fleet.pending:
        return
    job_id, submitted_at = fleet.pending
    backend = fleet.backends.get(job_id)
    if backend is not None:
        failed_attempt = " exited with status " in backend.state or backend.state.startswith(
            "stopped;"
        )
        if not backend.healthy and failed_attempt:
            LOG.warning("successor %s failed to start; restarting its retained allocation", job_id)
            code, out = await run_serve_sh(fleet, "restart", backend.run_dir)
            if code == 0:
                fleet.pending = (job_id, now)
            else:
                LOG.error("restart %s failed (rc=%d): %s", job_id, code, out)
                cancel_code, cancel_out = await run_slurm_command("scancel", job_id)
                if cancel_code == 0:
                    fleet.pending = None
                else:
                    LOG.error("scancel %s failed (rc=%s): %s", job_id, cancel_code, cancel_out)
        return

    # sbatch can take a moment to publish a new job into squeue. Do not mistake
    # that visibility gap for an immediate terminal failure.
    if now - submitted_at < PENDING_VISIBILITY_GRACE:
        return
    status = await slurm_job_status(job_id)
    if status is None:
        return
    state, reason = status
    terminal = {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "TIMEOUT",
    }
    if state == "GONE" or state in terminal:
        LOG.warning("successor %s is no longer runnable (%s); retrying", job_id, state)
        fleet.pending = None
        return

    # Slurm reports RUNNING as soon as it hands the node over, which is before
    # the prolog finishes and therefore before the batch script exists to
    # register anything. Restart the grace clock instead of reading that as a
    # launcher failure: node setup here regularly outlasts
    # PENDING_VISIBILITY_GRACE, and cancelling mid-prolog throws away a
    # successor that was about to come up -- repeatedly, since every retry
    # meets the same prolog. A prolog cannot stall forever, because Slurm caps
    # it with PrologEpilogTimeout and fails the job into a terminal state the
    # branch above already handles.
    if state == "RUNNING" and "prolog" in reason.lower():
        LOG.info("successor %s is still in prolog; deferring its failure check", job_id)
        fleet.pending = (job_id, now)
        return

    # A RUNNING job reaches its serving command and registers before it starts
    # loading the model. If it remains invisible here, the launcher failed
    # before that point. Held jobs cannot make progress either. Cancel before
    # retrying so a delayed job cannot later appear as a duplicate successor.
    should_cancel = state == "RUNNING" or "held" in reason.lower()
    if should_cancel:
        LOG.warning(
            "successor %s is %s without registering (%s); cancelling and retrying",
            job_id,
            state,
            reason or "no reason",
        )
        code, out = await run_slurm_command("scancel", job_id)
        if code == 0:
            fleet.pending = None
        else:
            LOG.error("scancel %s failed (rc=%s): %s", job_id, code, out)


async def supervisor_loop(fleet):
    while True:
        try:
            await supervise(fleet)
        except Exception:
            LOG.exception("supervisor failed")
        await asyncio.sleep(fleet.args.supervisor_interval)


async def supervise(fleet):
    now = time.time()
    await supervise_pending(fleet, now)

    # Relay: submit the next job early enough that it finishes loading weights
    # before this one hits the wall clock.
    backend = fleet.backends.get(fleet.active) if fleet.active else None
    if backend is not None and not fleet.args.no_relay:
        remaining = backend.end_time - now
        # A submitted/loading successor is represented by `pending`. An older
        # job that took traffic and then failed remains discoverable so it can
        # recover, but must not block a replacement successor forever.
        successors = [
            j for j, candidate in fleet.backends.items() if j != fleet.active and candidate.healthy
        ]
        if backend.end_time <= 0:
            # Registration could not determine the wall clock. Routing still
            # works; relaying on `0 - now` would read as "already expired" and
            # submit a job every single sweep.
            LOG.warning("backend %s has no end time; relay disabled for it", fleet.active)
        elif remaining < fleet.args.lead_time and not successors and not fleet.pending:
            LOG.info("%s ends in %ds; submitting successor", fleet.active, int(remaining))
            await submit_successor(fleet, now, "relay")
    elif not fleet.args.no_relay and fleet.ever_active and not fleet.backends and not fleet.pending:
        # Recovery cannot depend on a live active backend: a cancelled pending
        # job may disappear just as its predecessor reaches the wall clock.
        LOG.warning("fleet lost every backend; submitting recovery successor")
        await submit_successor(fleet, now, "recovery")

    # Promote superseded backends to draining, but only once the successor has
    # held up. Handing over routing is reversible and happens the instant the
    # successor is healthy; releasing the predecessor's allocation is not, so it
    # waits. Without this, a successor that passes one probe and then dies takes
    # the predecessor down with it and leaves nothing serving until the next job
    # finishes loading.
    winner = fleet.backends.get(fleet.active) if fleet.active else None
    if winner is not None and fleet.superseded:
        stable_for = now - winner.healthy_since if winner.healthy_since else 0
        if winner.healthy and stable_for >= fleet.args.promote_after:
            for job_id in sorted(fleet.superseded):
                fleet.superseded.discard(job_id)
                if job_id == fleet.active:
                    LOG.warning("refusing to drain active backend %s", job_id)
                    continue
                backend = fleet.backends.get(job_id)
                if backend is None:
                    continue
                deadline = backend.end_time - 60
                fleet.draining[job_id] = deadline
                LOG.info(
                    "draining %s (%s stable %ds, inflight=%d, reclaim by %s)",
                    job_id,
                    fleet.active,
                    int(stable_for),
                    fleet.inflight.get(job_id, 0),
                    fmt_time(deadline),
                )

    # Reclaim: the drained job is already past being useful, and its allocation
    # is worth releasing a little early. Skipped under --no-relay, which
    # promises not to touch job lifecycles at all -- submitting and reclaiming
    # are two halves of the same authority.
    if fleet.args.no_relay:
        return
    for job_id, deadline in list(fleet.draining.items()):
        if job_id == fleet.active:
            LOG.warning("cancelled stale drain of active backend %s", job_id)
            fleet.draining.pop(job_id, None)
            continue
        backend = fleet.backends.get(job_id)
        if backend is None:
            fleet.draining.pop(job_id, None)
            continue
        inflight = fleet.inflight.get(job_id, 0)
        # A backend that never published an end_time yields deadline < 0, so a
        # bare `now <= deadline` is false immediately and the quit below kills
        # whatever is still streaming. Without a real deadline there is nothing
        # to time out against, so wait for the drain instead.
        if inflight and (deadline <= 0 or now <= deadline):
            continue
        why = "drained" if not inflight else "deadline"
        LOG.info("reclaiming %s (%s, inflight=%d)", job_id, why, inflight)
        code, out = await run_serve_sh(fleet, "quit", backend.run_dir)
        if code != 0:
            LOG.error("quit %s failed (rc=%d): %s", job_id, code, out)
        fleet.draining.pop(job_id, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="stable front door for the Anthropic-compatibility server"
    )
    parser.add_argument(
        "--fleet-dir", required=True, help="directory the serving jobs register into"
    )
    parser.add_argument("--users", required=True, help="allowlist, one username per line")
    parser.add_argument("--yaml", default="", help="deployment YAML the supervisor resubmits")
    parser.add_argument(
        "--serve-sh",
        default="",
        help="launcher script exposing submit/restart/quit (defaults to serve.sh next to this file)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8333)
    parser.add_argument(
        "--lead-time",
        type=int,
        default=2700,
        help="seconds before the wall clock to submit the successor (default 45min)",
    )
    parser.add_argument(
        "--stale-after",
        type=int,
        default=30,
        help="drop a backend after this long without a heartbeat",
    )
    parser.add_argument("--discover-interval", type=float, default=5.0)
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--supervisor-interval", type=float, default=30.0)
    parser.add_argument("--probe-timeout", type=float, default=3.0)
    parser.add_argument(
        "--unhealthy-after",
        type=int,
        default=20,
        help="consecutive probe timeouts before a backend is taken out of rotation; a refused "
        "connection is acted on immediately regardless",
    )
    parser.add_argument(
        "--promote-after",
        type=float,
        default=180.0,
        help="seconds a successor must stay healthy before its predecessor may be reclaimed",
    )
    parser.add_argument(
        "--no-relay",
        action="store_true",
        help="proxy only; never submit a successor and never reclaim a drained job",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
        # DEBUG additionally prints the last SSE data line before a failure,
        # which is response content -- useful when reproducing a truncation,
        # inappropriate as a standing default.
        help="DEBUG also logs response previews on failure (contains model output)",
    )
    args = parser.parse_args(argv)
    if not args.serve_sh:
        args.serve_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.sh")
    if not args.no_relay:
        if not args.yaml:
            parser.error("--yaml is required unless --no-relay is given")
        # Relay drives job lifecycles through the launcher script. That script
        # does not ship with this example, so refuse at startup rather than
        # discovering it hours later at the first handover, which is exactly
        # when there is no time to recover.
        if not os.path.isfile(args.serve_sh):
            parser.error(
                "launcher script %s does not exist; pass --serve-sh or run with --no-relay"
                % args.serve_sh
            )
    return args


async def main_async(args):
    fleet = Fleet(args)
    os.makedirs(args.fleet_dir, exist_ok=True)
    fleet.reload_users()
    if not fleet.users:
        LOG.warning("users file %s is empty; every request will get 401", args.users)
    fleet.discover()

    gateway = Gateway(fleet)
    server = await asyncio.start_server(gateway.handle, args.host, args.port)
    LOG.info("listening on %s:%d", args.host, args.port)
    LOG.info("fleet dir: %s", args.fleet_dir)
    LOG.info(
        "relay: %s",
        "off" if args.no_relay else "lead time %ds from %s" % (args.lead_time, args.yaml),
    )

    async with server:
        await asyncio.gather(discovery_loop(fleet), health_loop(fleet), supervisor_loop(fleet))


def main():
    args = parse_args(sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        LOG.info("interrupted")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
