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
import hashlib
import importlib.util
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
    405: ("invalid_request_error", "this endpoint only accepts POST"),
    502: ("api_error", "backend refused the connection"),
    503: (
        "overloaded_error",
        "no healthy backend right now; the serving job is rotating, retry shortly",
    ),
}

ERROR_REASONS = {
    401: "Unauthorized",
    404: "Not Found",
    405: "Method Not Allowed",
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


async def read_body(reader, rest, headers, limit):
    """Read the whole request body, when it is bounded and worth buffering.

    Returns (body, remainder). A body of None means it was not buffered -- no
    content-length, chunked, or over the limit -- and the caller must fall back
    to pumping `remainder` and whatever follows it.

    Buffering exists only so the conversation can be identified before a
    backend is chosen. It is capped because the alternative, holding an
    unbounded client upload in memory, trades a routing optimisation for an
    availability risk.
    """
    if header_value(headers, "transfer-encoding"):
        return None, rest
    raw = header_value(headers, "content-length")
    if raw is None:
        return None, rest
    try:
        length = int(raw)
    except ValueError:
        return None, rest
    if length <= 0:
        return b"", rest
    if length > limit:
        return None, rest
    chunks = [rest[:length]]
    have = len(chunks[0])
    leftover = rest[length:]
    while have < length:
        chunk = await read_upstream(reader, min(RELAY_CHUNK, length - have))
        if not chunk:
            # The client stopped mid-body. Give back what arrived and let the
            # backend reject the short request on its own terms, rather than
            # inventing an error for a conversation we cannot even name.
            return None, b"".join(chunks)
        chunks.append(chunk)
        have += len(chunk)
    return b"".join(chunks), leftover


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

    __slots__ = ("rid", "upstream_status", "tracker", "request_body_error", "conversation")

    def __init__(self):
        self.rid = next_request_id()
        self.upstream_status = None
        self.tracker = None
        self.request_body_error = None
        self.conversation = None

    def detail(self):
        parts = ["rid=%s" % self.rid]
        if self.conversation is not None:
            # Which conversation a failure belongs to is the first thing wanted
            # when one agent session misbehaves and five others are fine.
            parts.append("convo=%s" % short_convo(self.conversation))
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
        # Set by stop_server, cleared by start_server. Separate from
        # ever_active because elect() overwrites that one.
        self.stopped = False
        self.last_submit = 0.0
        self.seen_running = set()
        # Replaced, but not yet cleared for reclaim: draining ends in a `quit`,
        # so it waits until the successor has proven itself.
        self.superseded = set()
        # job_id -> (restarts attempted, last attempt). Kept on the fleet
        # rather than the Backend so a job that re-registers cannot reset its
        # own budget by being rediscovered.
        self.revived = {}
        # job_id -> (run_dir, retired_at) for backends that left the table.
        # revive_dead_backends cannot see these: a preempted deployment takes
        # its controller with it, and the controller is what deletes the
        # registration. Recovery therefore needs its own record of what used
        # to be here, kept until the job is confirmed gone from the scheduler
        # or the instance comes back.
        self.lost = {}
        # job_id -> (recoveries attempted, last attempt), same reasoning as
        # `revived`: a job that reappears and vanishes again must not get a
        # fresh budget just for having been rediscovered.
        self.recovered = {}
        self.last_recovery = 0.0
        self.started = time.time()
        self.router = Router(
            args.sticky_ttl,
            args.sticky_capacity,
            policy=args.route_policy,
            state_path=args.router_state,
            key_sources=args.key_sources or DEFAULT_KEY_SOURCES,
        )
        # Loaded before the saved state is restored, so a policy that came from
        # a file is a legitimate thing to have been using before the restart.
        # Otherwise every rotation would silently demote a custom policy back
        # to the default, which is the sort of thing nobody notices until the
        # placement they tuned for stops happening.
        policy_dir = getattr(args, "policy_dir", None)
        self.router.policies = PolicyDir(policy_dir) if policy_dir else None
        if self.router.policies is not None:
            self.router.policies.reload()
        self.router.load()
        if getattr(args, "route_policy", "least_conversations") not in known_policies(self.router):
            LOG.warning(
                "--route-policy %r is not a built-in and no such file is in "
                "--policy-dir; placement will use least_conversations until "
                "it appears",
                getattr(args, "route_policy", None),
            )

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
    def read_registrations(self):
        """Read the registration directory. Blocking, and nothing else.

        Split out of discover() so the caller can put it on a thread: the
        directory lives on the shared filesystem every serving job writes to,
        and at 46 backends one sweep costs ~45 ms there against ~1 ms locally.
        Spent on the event loop that is 45 ms of added tail latency on every
        stream in flight. Only the reads move -- applying the result stays on
        the loop, because the backend table is read by the request path and a
        dict that changes size mid-iteration raises.
        """
        records = []
        for path in glob.glob(os.path.join(self.args.fleet_dir, "*.json")):
            try:
                with open(path) as handle:
                    records.append((path, json.load(handle)))
            except (OSError, ValueError):
                # Mid-rename or truncated. The writer replaces the file
                # atomically, so the next sweep gets a whole one. Stay quiet:
                # this is expected and would otherwise log on every sweep.
                continue
        return records

    def discover(self, records=None):
        """Rebuild the backend table from the registration directory.

        Each serving job owns exactly one file named after its scheduler job id,
        so there is never more than one writer per file and the gateway never
        has to coordinate with anybody. The union is just the directory listing.

        `records` is what read_registrations() returned, for callers that did
        the reads elsewhere; omitting it reads them here, which is what the
        synchronous start-up path and the tests want.
        """
        now = time.time()
        seen = set()
        if records is None:
            records = self.read_registrations()
        for path, record in records:
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
            if record.get("manual") and job_id in self.backends:
                # Nothing writes heartbeats for a backend registered by hand,
                # so the staleness rule below would retire it the moment it
                # first went unhealthy -- and then never re-add it, because
                # the re-add path also refuses a stale record. The probe is
                # already the authority on whether a backend can serve; for
                # these it is the only authority.
                pass
            elif now - heartbeat > self.args.stale_after:
                # A stale heartbeat is a hint, not a verdict. The file lives on
                # a shared filesystem that every serving job writes to, so one
                # stall there makes every backend look stale in the same sweep
                # -- and that is exactly what happened under load: all four went
                # `gone: no heartbeat for 30s` together and all four came back
                # five seconds later, having answered /health throughout. The
                # gateway served 503 to 43 requests for a fleet that was fine.
                #
                # The probe is the authority on whether a backend can serve, so
                # a backend that is still passing it keeps its place. One that
                # is not gets retired as before, and a backend whose file is
                # gone entirely never reaches this branch.
                existing = self.backends.get(job_id)
                if existing is None or not existing.healthy:
                    continue
                LOG.debug(
                    "%s heartbeat is %ds stale but it is still passing probes; keeping",
                    job_id,
                    int(now - heartbeat),
                )
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
                # It came back on its own -- SLURM requeued it, or an operator
                # brought it up. Either way nothing needs recovering.
                self.lost.pop(job_id, None)
                LOG.info(
                    "backend appeared: %s at %s (ends %s)",
                    job_id,
                    self.backends[job_id].url,
                    fmt_time(self.backends[job_id].end_time),
                )

        for job_id in [j for j in self.backends if j not in seen]:
            LOG.info("backend gone: %s (no heartbeat for %ds)", job_id, self.args.stale_after)
            # Remember it before dropping it. Every way a node is preempted
            # ends here -- SIGTERM and "allocation gone" both delete the
            # registration through clear_fleet, and SIGKILL leaves one that
            # goes stale -- so this is the single place that sees all three,
            # and the only place still holding the run_dir.
            self.lost.setdefault(job_id, (self.backends[job_id].run_dir, now))
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

    # -- routable sets ----------------------------------------------------
    def serving(self):
        """Every backend that can still answer, including ones being drained.

        A conversation pinned to a draining backend keeps going there. Moving
        it would throw away the KV cache the pin exists to preserve, and the
        drain already guarantees the backend outlives the requests on it.
        """
        return {j for j, b in self.backends.items() if b.healthy}

    def accepting(self):
        """Backends eligible for a NEW conversation, with their load.

        Returns job_id -> (conversations, inflight) so the router can order
        them without reaching back into the fleet.

        Excluded: anything draining, superseded, or too close to its own end
        time to see a conversation through. That last one is what replaces the
        single-active election -- instead of one backend holding all traffic
        until it is replaced, an ageing backend simply stops being offered new
        work and empties out on its own.
        """
        now = time.time()
        horizon = self.args.new_conversation_margin
        eligible = {}
        for job_id, backend in self.backends.items():
            if not backend.healthy:
                continue
            # `superseded` is deliberately not consulted. It means "a
            # longer-lived backend exists", which is a statement about which
            # allocation to reclaim first -- not about whether this one can
            # take work. Excluding it here collapsed a fleet of N healthy
            # instances down to one accepting instance the moment a successor
            # was elected, because every other instance is superseded by
            # definition. Only `draining`, which means "this one is going
            # away", keeps new conversations out.
            if job_id in self.draining:
                continue
            if job_id in self.router.paused:
                continue
            remaining = backend.end_time - now if backend.end_time else 0.0
            if backend.end_time and remaining < horizon:
                continue
            eligible[job_id] = (0, self.inflight.get(job_id, 0), remaining)
        # Every backend is ageing out at once: rather than serve 503, offer the
        # longest-lived one. A conversation started here may be cut short, which
        # is strictly better than refusing to start it.
        if not eligible:
            # A hand-set pause is an instruction, so it survives this fallback;
            # ageing out is not, so it does not.
            healthy = [
                j for j, b in self.backends.items() if b.healthy and j not in self.router.paused
            ]
            if healthy:
                best = max(healthy, key=lambda j: self.backends[j].end_time)
                eligible[best] = (0, self.inflight.get(best, 0), self.backends[best].end_time - now)
        return eligible

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


def json_response(payload, status=200, reason="OK"):
    body = json.dumps(payload, indent=2).encode()
    return build_response(status, reason, body)


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


# How each dialect says "that is the whole response".
#
# Anthropic closes with `message_stop`. Chat Completions closes with a `[DONE]`
# data line and no event name, handled separately below. The Responses API
# closes with `response.completed` and never sends `[DONE]`, so a Responses
# stream that finished cleanly used to fall through every check here and be
# reported as truncated -- the gateway then injected an error into a response
# that was complete, and a client reading that error gives up on the endpoint.
_TERMINAL_EVENTS = frozenset(
    (
        b"message_stop",
        b"response.completed",
    )
)

# Terminal too, but terminal as a failure: the response is over and did not
# succeed. Distinguished from the set above so the gateway does not report a
# server-side refusal as a healthy completion.
_FAILURE_EVENTS = frozenset(
    (
        b"error",
        b"response.failed",
        b"response.incomplete",
    )
)


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
                if self.current_event in _TERMINAL_EVENTS:
                    self.saw_stop = True
                elif self.current_event in _FAILURE_EVENTS:
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

# --------------------------------------------------------------------------
# conversation routing
# --------------------------------------------------------------------------

# Checked in order. A header wins because it is the only source a client can
# set deliberately; everything below is inferred from what clients already send.
_CONVERSATION_HEADERS = ("x-conversation-id", "x-session-id", "session-id", "thread-id")

# Where each API surface keeps its conversation identity. Measured against
# 2501 captured requests rather than guessed:
#   /v1/responses         prompt_cache_key, on 58% of them, one value per
#                         session and about 120 requests per value.
#   /v1/chat/completions  client_metadata.session_id -- nested, which is why a
#                         scan of top-level body keys finds nothing and makes
#                         this surface look unroutable.
# The order requests are asked for their identity. Each entry is one of
#   header:<name>       a request header
#   body:<a.b.c>        a body field, dotted for nesting
#   prefix              hash the opening of the conversation
# Configurable because a client that appears later should not need a code
# change to be routed, and because the right order is a property of the fleet's
# traffic rather than of the gateway.
DEFAULT_KEY_SOURCES = (
    "header:x-conversation-id",
    "header:x-session-id",
    "header:session-id",
    "header:thread-id",
    "body:prompt_cache_key",
    "body:previous_response_id",
    "body:conversation_id",
    "body:client_metadata.session_id",
    "body:client_metadata.thread_id",
    "body:metadata.session_id",
    "body:metadata.conversation_id",
    "prefix",
)

POLICIES = ("least_conversations", "least_inflight", "round_robin", "longest_lived")


class PolicyDir:
    """Custom routing policies, loaded from .py files in a directory.

    A directory rather than an HTTP endpoint, deliberately. A policy is code,
    and this gateway authenticates with a username that its own users file
    describes as guessable -- so taking code over HTTP would make "knows a
    colleague's name" enough to run anything on this node. Writing a file to
    the shared filesystem is a far higher bar, and it is the same bar the
    fleet directory already sets for registering a backend.

    A file defines `select(accepting)` and is named by the policy it provides:
    `prefill_aware.py` supplies `prefill_aware`. `accepting` maps job id to
    `(conversations, inflight, remaining_seconds)`; return one of its keys.

    Custom policies run on the request path, so failure is contained rather
    than trusted away: anything that raises, returns a job that is not on
    offer, or takes the process down a path the built-ins would not is counted
    against the policy, and after `strikes` it is dropped and placement falls
    back to least_conversations. Editing the file clears the count -- the
    intended way to fix a policy is to fix it, not to restart the gateway.
    """

    strikes = 3

    def __init__(self, path, strikes=None):
        self.path = path
        self.policies = {}
        self.mtimes = {}
        self.failures = collections.Counter()
        self.disabled = set()
        if strikes is not None:
            self.strikes = strikes

    def names(self):
        return tuple(sorted(n for n in self.policies if n not in self.disabled))

    def reload(self):
        """Pick up new, changed and deleted policy files. Never raises."""
        if not self.path:
            return
        try:
            paths = sorted(glob.glob(os.path.join(self.path, "*.py")))
        except OSError as exc:
            LOG.warning("policy dir unreadable: %s", exc)
            return
        seen = set()
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if name.startswith("_"):
                continue
            seen.add(name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if self.mtimes.get(name) == mtime:
                continue
            self.mtimes[name] = mtime
            # An edit is a retraction of the last verdict on this policy.
            self.failures.pop(name, None)
            self.disabled.discard(name)
            fn = self._load(path, name)
            if fn is None:
                self.policies.pop(name, None)
                continue
            self.policies[name] = fn
            LOG.info("routing policy loaded: %s (%s)", name, path)
        for name in [n for n in self.policies if n not in seen]:
            self.policies.pop(name, None)
            self.mtimes.pop(name, None)
            self.failures.pop(name, None)
            self.disabled.discard(name)
            LOG.info("routing policy withdrawn: %s", name)

    def _load(self, path, name):
        """Import one file. A broken policy costs itself and nothing else."""
        if name in POLICIES:
            LOG.warning("policy %s shadows a built-in; ignoring %s", name, path)
            return None
        try:
            spec = importlib.util.spec_from_file_location("kfpolicy_" + name, path)
            if spec is None or spec.loader is None:
                raise ImportError("no loader for %s" % path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - arbitrary user code
            LOG.warning("policy %s failed to load: %s: %s", name, type(exc).__name__, exc)
            return None
        fn = getattr(module, "select", None)
        if not callable(fn):
            LOG.warning("policy %s defines no select(accepting); ignoring", name)
            return None
        return fn

    def run(self, name, accepting):
        """Ask a custom policy to place a conversation, or return None.

        None means "the caller should use the built-in default" and is the
        answer for every way this can go wrong, so the request path has one
        branch instead of a taxonomy of failures.
        """
        fn = self.policies.get(name)
        if fn is None or name in self.disabled:
            return None
        try:
            picked = fn(dict(accepting))
        except Exception as exc:  # noqa: BLE001 - arbitrary user code
            return self._strike(name, "%s: %s" % (type(exc).__name__, exc))
        if picked not in accepting:
            return self._strike(name, "returned %r, which is not on offer" % (picked,))
        return picked

    def _strike(self, name, why):
        self.failures[name] += 1
        count = self.failures[name]
        if count >= self.strikes:
            self.disabled.add(name)
            LOG.error(
                "routing policy %s disabled after %d failures (%s); "
                "placement falls back to least_conversations until the "
                "file is edited",
                name,
                count,
                why,
            )
        else:
            LOG.warning("routing policy %s failed (%d/%d): %s", name, count, self.strikes, why)
        return None


SAFE_JOB_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def registration_path(fleet_dir, job_id):
    """Where a backend's registration file goes, or None if the id is unsafe.

    The id becomes a filename, so it decides where this writes. `../../x` or an
    absolute path would put a file wherever the gateway can reach, and this is
    reachable by anyone the users file admits -- which that file says is anyone
    who can guess a colleague's name. The check is a whitelist because a
    blacklist of traversal spellings is a game nobody wins.
    """
    if not isinstance(job_id, str) or not SAFE_JOB_ID.match(job_id):
        return None
    path = os.path.join(fleet_dir, job_id + ".json")
    # Belt and braces: even a whitelist-passing id must not escape the dir.
    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(fleet_dir):
        return None
    return path


def known_policies(router):
    """Built-ins plus whatever the policy directory currently offers."""
    custom = getattr(router, "policies", None)
    return tuple(POLICIES) + (custom.names() if custom is not None else ())


def conversation_prefix(payload):
    """Hashable bytes for the part of a conversation that does not change.

    A last resort, used only when a client sends no identity of its own. Agent
    turns replay the whole history, so the opening of that history is stable
    for the life of the session -- and it is also exactly the prefix whose KV
    cache the routing exists to reuse.

    Only the first two entries are hashed. Including more would fold in text
    that grows every turn, and the key would then change under the session it
    is supposed to pin.
    """
    parts = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions:
        parts.append(instructions)
    for field in ("input", "messages"):
        value = payload.get(field)
        if isinstance(value, str) and value:
            parts.append(value)
            break
        if isinstance(value, list) and value:
            try:
                parts.append(json.dumps(value[:2], sort_keys=True, separators=(",", ":")))
            except (TypeError, ValueError):
                pass
            break
    if not parts:
        return None
    return "\0".join(parts).encode("utf-8", "replace")


def dig(payload, path):
    """Follow a dotted path into a decoded body, returning a string or None."""
    node = payload
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node if isinstance(node, str) and node else None


def conversation_key(headers, body, sources=DEFAULT_KEY_SOURCES):
    """Identify the conversation a request belongs to, or None.

    Sources are tried in order and the first that answers wins, so the caller
    controls precedence without this function knowing which client is which.
    """
    payload = None
    decoded = False
    for source in sources:
        if source.startswith("header:"):
            value = header_value(headers, source[7:])
            if value and value.strip():
                return "hdr:" + value.strip()
            continue
        if not body:
            continue
        if not decoded:
            decoded = True
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                payload = None
            if not isinstance(payload, dict):
                payload = None
        if payload is None:
            continue
        if source.startswith("body:"):
            path = source[5:]
            value = dig(payload, path)
            if value:
                return "%s:%s" % (path, value)
        elif source == "prefix":
            prefix = conversation_prefix(payload)
            if prefix:
                return "prefix:" + hashlib.sha256(prefix).hexdigest()[:32]
    return None


def short_convo(key):
    """Shorten a conversation key for a log line without losing what it names.

    Truncating the head is worse than useless here: the source prefix is the
    fixed part and the identifier is the tail, so `key[:28]` renders every
    conversation from one client identically -- which is exactly the case the
    log line exists to tell apart. Keep both ends.
    """
    if key is None:
        return "-"
    if len(key) <= 44:
        return key
    return key[:20] + ".." + key[-20:]


class Router:
    """Pins a conversation to a backend and keeps it there.

    Affinity is not about correctness -- any healthy backend can answer any
    request. It is about the prefix KV cache: an agent turn resends the whole
    conversation, so a turn that lands where the previous one landed reuses
    that cache and one that does not pays to rebuild it.

    So a broken pin is a slowdown, never a wrong answer. That is what lets
    every failure here resolve by re-pinning rather than by refusing to serve.
    """

    def __init__(
        self,
        ttl,
        capacity,
        policy="least_conversations",
        state_path=None,
        key_sources=DEFAULT_KEY_SOURCES,
    ):
        self.ttl = ttl
        self.capacity = capacity
        self.policy = policy
        self.key_sources = list(key_sources)
        self.state_path = state_path
        self.pins = collections.OrderedDict()  # key -> (job_id, last_seen)
        # Set by hand and authoritative: never expired, never displaced by a
        # placement decision. Only a backend that has actually gone away can
        # override one, because the alternative is refusing to serve.
        self.manual = {}  # key -> job_id
        self.paused = set()  # job ids held out of new placement by hand
        self.hits = 0
        self.misses = 0
        self.rehomed = 0
        self.dirty = False
        self._rr = 0
        # Conversations per backend, maintained incrementally. Rebuilding it by
        # walking `pins` on every request made placement cost O(pins) rather
        # than O(backends), and `pins` is bounded by --sticky-capacity, not by
        # the fleet size: at the 20000 default that was 2.2 ms of the 3.7 ms a
        # request spent routing. Every mutation of `pins` must go through
        # _set_pin/_drop_pin so this stays in step.
        self._tally = collections.Counter()
        self._last_expire = 0.0

    # -- persistence ------------------------------------------------------
    def load(self):
        """Restore pins written by a previous process.

        Without this, restarting the gateway silently discards every pin, and
        each conversation in flight rebuilds a prefix cache it already had. On
        a fleet whose gateway lives inside a four-hour scheduler job, that is
        not a rare event.
        """
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as handle:
                state = json.load(handle)
        except (OSError, ValueError) as exc:
            LOG.warning("ignoring unreadable router state %s: %s", self.state_path, exc)
            return
        if not isinstance(state, dict):
            LOG.warning("ignoring router state %s: not an object", self.state_path)
            return
        now = time.time()
        restored = 0
        for key, value in (state.get("pins") or {}).items():
            try:
                job_id, last_seen = str(value[0]), float(value[1])
            except (TypeError, ValueError, IndexError):
                continue
            if now - last_seen <= self.ttl:
                self.pins[key] = (job_id, last_seen)
                restored += 1
        self.manual = {
            str(k): str(v) for k, v in (state.get("manual") or {}).items() if isinstance(v, str)
        }
        self.paused = {str(j) for j in (state.get("paused") or [])}
        if state.get("policy") in known_policies(self):
            self.policy = state["policy"]
        sources = state.get("key_sources")
        if isinstance(sources, list) and sources:
            self.key_sources = [str(x) for x in sources]
        # load() is the one path that fills `pins` without going through
        # _set_pin, and it runs after __init__, so the tally has to be rebuilt
        # here. Skipping this leaves every count at zero while the table is
        # full, and least_conversations then reads the whole fleet as idle and
        # piles every new conversation onto whichever job id sorts first.
        self._rebuild_tally()
        LOG.info(
            "router state restored: %d pins, %d manual, %d paused, policy=%s",
            restored,
            len(self.manual),
            len(self.paused),
            self.policy,
        )

    def snapshot(self):
        """Copy out the state to persist, or None if there is nothing to write.

        Every container here is copied rather than referenced, because the
        write runs on a thread and json.dump would otherwise iterate a table
        the request path is still inserting into. Cheap enough to stay on the
        event loop: at the 20000-pin default it is one dict comprehension.
        """
        if not self.state_path or not self.dirty:
            return None
        # Cleared at snapshot time, not after the write: this snapshot is what
        # the write will persist, so a mutation arriving while it is in flight
        # belongs to the next flush and has to re-dirty the table itself.
        self.dirty = False
        return {
            "version": 1,
            "saved_at": time.time(),
            "policy": self.policy,
            "key_sources": list(self.key_sources),
            "manual": dict(self.manual),
            "paused": sorted(self.paused),
            "pins": {k: [j, t] for k, (j, t) in self.pins.items()},
        }

    def write_snapshot(self, state):
        """Write a snapshot out. Blocking, and safe to run on a thread."""
        if not state:
            return
        tmp = "%s.tmp.%d" % (self.state_path, os.getpid())
        try:
            with open(tmp, "w") as handle:
                json.dump(state, handle)
            os.replace(tmp, self.state_path)
        except OSError as exc:
            # Losing the snapshot costs cache warmth on the next restart and
            # nothing else, so it must never take the gateway down with it.
            # Put the flag back, or an unlucky write leaves the on-disk copy
            # behind for as long as nothing else happens to dirty the table.
            LOG.warning("could not save router state: %s", exc)
            self.dirty = True
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def save(self):
        self.write_snapshot(self.snapshot())

    # -- pin table bookkeeping --------------------------------------------
    # `pins` and `_tally` are one structure in two parts. These three are the
    # only places that write to it.
    def _rebuild_tally(self):
        self._tally = collections.Counter(job_id for job_id, _ in self.pins.values())

    def _set_pin(self, key, job_id, now):
        previous = self.pins.get(key)
        if previous is not None:
            self._tally[previous[0]] -= 1
        self.pins[key] = (job_id, now)
        self.pins.move_to_end(key)
        self._tally[job_id] += 1

    def _drop_pin(self, key):
        dropped = self.pins.pop(key, None)
        if dropped is not None:
            self._tally[dropped[0]] -= 1
        return dropped

    def _expire(self, now):
        """Reclaim pins the TTL has passed, at most once a second.

        This used to run per request, which made every request pay for the
        whole table. It can be amortised because it is only reclaiming memory:
        route() expires the one key it was asked about exactly, so a caller
        never observes a stale pin no matter how long this waits.
        """
        if now - self._last_expire < 1.0:
            return
        self._last_expire = now
        for key in [k for k, (_, seen) in self.pins.items() if now - seen > self.ttl]:
            self._drop_pin(key)

    def _trim(self):
        # Runs after the insert, not before it. Trimming first leaves room for
        # exactly one more and the table settles one over capacity forever.
        while len(self.pins) > self.capacity:
            key, (job_id, _) = self.pins.popitem(last=False)
            self._tally[job_id] -= 1

    def route(self, key, accepting, serving, now=None):
        """Return the job id to use, pinning the conversation on first sight.

        `accepting` are the backends taking new conversations; `serving` also
        includes those that are draining, because a conversation already pinned
        to one should stay there while it lasts rather than lose its cache to a
        handover it did not need.
        """
        now = time.time() if now is None else now
        self._expire(now)
        if key is None:
            return self.select(accepting) if accepting else None
        # A hand-placed pin outranks everything, including the load picture --
        # that is the point of setting one. It still yields to a backend that
        # has actually gone away, because refusing to serve would be worse.
        manual = self.manual.get(key)
        if manual is not None:
            if manual in serving:
                self.hits += 1
                return manual
            LOG.warning("manual pin %s -> %s is not serving; falling back", key[:40], manual)
        pinned = self.pins.get(key)
        if pinned is not None and now - pinned[1] > self.ttl:
            # The amortised sweep in _expire may not have reached this key yet.
            # Expiring the one key we were asked about is O(1) and keeps the
            # observable behaviour identical to the old full scan per request.
            self._drop_pin(key)
            pinned = None
        if pinned is not None:
            job_id = pinned[0]
            if job_id in serving:
                self._set_pin(key, job_id, now)
                self.hits += 1
                return job_id
            # The backend it was pinned to is gone. Re-pin rather than fail:
            # the conversation loses its cache, which is the whole cost.
            self.rehomed += 1
            LOG.info("conversation %s lost backend %s; re-homing", key[:40], job_id)
        else:
            self.misses += 1
        if not accepting:
            return None
        job_id = self.select(accepting)
        self._set_pin(key, job_id, now)
        self._trim()
        self.dirty = True
        return job_id

    @staticmethod
    def _stat(stats, index):
        return stats[index] if len(stats) > index else 0

    def select(self, accepting):
        """Place a new conversation according to the current policy.

        `accepting` maps job id to (conversations, inflight, remaining_seconds).

        least_conversations is the default because conversation count is what
        predicts future load: in-flight counts only what is happening this
        instant, and an agent spends most of a turn thinking rather than
        streaming, so a backend holding five idle sessions reads as empty.
        """
        jobs = sorted(accepting)
        custom = getattr(self, "policies", None)
        if custom is not None and self.policy not in POLICIES:
            picked = custom.run(self.policy, accepting)
            if picked is not None:
                return picked
            # run() has already logged why. Fall through rather than fail the
            # placement: a bad policy should cost its own behaviour, not the
            # conversation that happened to arrive while it was loaded.
        if self.policy == "least_inflight":
            return min(
                jobs, key=lambda j: (self._stat(accepting[j], 1), self._stat(accepting[j], 0), j)
            )
        if self.policy == "round_robin":
            job = jobs[self._rr % len(jobs)]
            self._rr += 1
            return job
        if self.policy == "longest_lived":
            # Ties broken by load, so a fleet of equally long-lived backends
            # still balances instead of piling onto whichever sorts first.
            return max(
                jobs, key=lambda j: (self._stat(accepting[j], 2), -self._stat(accepting[j], 0))
            )
        return min(
            jobs, key=lambda j: (self._stat(accepting[j], 0), self._stat(accepting[j], 1), j)
        )

    # -- manual control ---------------------------------------------------
    def pin(self, key, job_id):
        self.manual[key] = job_id
        self._set_pin(key, job_id, time.time())
        self.dirty = True

    def unpin(self, key):
        removed = self.manual.pop(key, None)
        self._drop_pin(key)
        self.dirty = True
        return removed

    def counts(self, job_ids):
        # O(backends), not O(pins) -- see _tally in __init__.
        return {job_id: self._tally.get(job_id, 0) for job_id in job_ids}


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
            # INFO, not DEBUG: the gateway runs at INFO, so an unparsable
            # client request used to leave no trace whatsoever -- and that is
            # the single failure mode a client-integration bug presents as.
            LOG.info("dropped rid=%s from %s: %s", trace.rid, peer, exc)
            await close(writer)
            return

        if path.startswith("/_gateway/"):
            # `rest` has to travel with the request: read_head returns whatever
            # arrived past the head, and for a small control POST that is the
            # entire body. Dropping it leaves the control handler waiting for
            # bytes that were already consumed, which the client sees as a hang.
            await self.serve_introspection(method, path, headers, rest, reader, writer)
            return

        key = extract_key(headers) or ANONYMOUS_USER
        if key not in self.fleet.users:
            LOG.info("401 %s %s user=%r [rid=%s]", method, path, key, trace.rid)
            await respond(writer, error_response(401))
            return

        try:
            body, rest = await read_body(reader, rest, headers, self.fleet.args.max_body_buffer)
        except (ValueError, OSError) as exc:
            LOG.info("dropped rid=%s from %s reading body: %s", trace.rid, peer, exc)
            await close(writer)
            return

        convo = (
            conversation_key(headers, body, self.fleet.router.key_sources)
            if self.fleet.args.route_by_conversation
            else None
        )
        trace.conversation = convo
        # Resolved once. Discovery and election run on their own timers and may
        # move things while this request is in flight; everything below must
        # keep talking about the same backend, or the inflight count is
        # incremented on one and decremented on another.
        job_id = self.route(convo)
        backend = self.fleet.backends.get(job_id) if job_id else None
        if backend is None:
            LOG.info("503 %s %s user=%s (no backend) [rid=%s]", method, path, key, trace.rid)
            await respond(writer, error_response(503, retry_after=20))
            return
        self.fleet.inflight[job_id] += 1
        status = "-"
        # Closing the client socket sits in its own finally so that no amount of
        # bookkeeping trouble above can leak the connection.
        try:
            try:
                status = await self.proxy(
                    backend, method, path, headers, rest, body, reader, writer, key, trace
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
                    "%s %s %s user=%s backend=%s convo=%s %.1fs [%s]",
                    status,
                    method,
                    path,
                    key,
                    job_id,
                    short_convo(convo),
                    time.time() - started,
                    trace.detail(),
                )
        finally:
            await close(writer)

    async def serve_introspection(self, method, path, headers, rest, reader, writer):
        if path == "/_gateway/health":
            # Deliberately unauthenticated: whatever watches the gateway from
            # outside has no reason to hold an allowlist entry, and the answer
            # names no user and no request.
            healthy = self.fleet.active is not None
            payload = {
                "status": "ok" if healthy else "no_backend",
                # Which deployment is answering, so a caller about to stop a
                # serving job can check it has reached the right gateway.
                "deployment": os.path.basename(os.path.normpath(self.fleet.args.fleet_dir)),
                "active": self.fleet.active,
                "pending": self.fleet.pending[0] if self.fleet.pending else None,
                "uptime_s": round(time.time() - self.fleet.started),
            }
            await respond(writer, json_response(payload))
            return
        if path in ("/_gateway/start_server", "/_gateway/stop_server"):
            if method != "POST":
                await respond(writer, error_response(405))
                return
            if path.endswith("/start_server"):
                status, payload = await start_server(self.fleet)
            else:
                status, payload = await stop_server(self.fleet)
            await respond(writer, json_response(payload, status, ERROR_REASONS.get(status, "OK")))
            return
        if path in (
            "/_gateway/route",
            "/_gateway/pin",
            "/_gateway/drain",
            "/_gateway/backend",
            "/_gateway/backend/remove",
        ):
            # Authenticated: these change how every other user's traffic is
            # placed, which is not something an unlisted caller should reach.
            if extract_key(headers) not in self.fleet.users:
                await respond(writer, error_response(401))
                return
            if method != "POST":
                await respond(writer, error_response(405))
                return
            status, payload = await self.control(path, rest, reader, headers)
            await respond(writer, json_response(payload, status, ERROR_REASONS.get(status, "OK")))
            return
        if path == "/_gateway/fleet":
            if extract_key(headers) not in self.fleet.users:
                await respond(writer, error_response(401))
                return
            now = time.time()
            router = self.fleet.router
            accepting = self.fleet.accepting()
            conversations = router.counts(self.fleet.backends)
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
                        "conversations": conversations.get(job_id, 0),
                        "accepting": job_id in accepting,
                        "superseded": job_id in self.fleet.superseded,
                        "revived": self.fleet.revived.get(job_id, (0, 0.0))[0],
                        "draining": job_id in self.fleet.draining,
                    }
                    for job_id, b in sorted(self.fleet.backends.items())
                },
                "routing": {
                    "enabled": self.fleet.args.route_by_conversation,
                    "policy": router.policy,
                    "key_sources": router.key_sources,
                    "manual_pins": router.manual,
                    "paused": sorted(router.paused),
                    "state_file": router.state_path,
                    "pinned": len(router.pins),
                    "hits": router.hits,
                    "misses": router.misses,
                    "rehomed": router.rehomed,
                    "accepting": sorted(accepting),
                    "serving": sorted(self.fleet.serving()),
                },
            }
            await respond(writer, json_response(payload))
            return
        await respond(writer, error_response(404))

    def route(self, convo):
        """Choose the backend for this request."""
        accepting = self.fleet.accepting()
        conversations = self.fleet.router.counts(accepting)
        loads = {
            job_id: (conversations.get(job_id, 0), inflight, remaining)
            for job_id, (_, inflight, remaining) in accepting.items()
        }
        return self.fleet.router.route(convo, loads, self.fleet.serving())

    async def control(self, path, rest, reader, headers):
        """Change routing while the gateway keeps running.

        Restarting to change a policy would drop the pin table, so every
        conversation in flight would rebuild the prefix cache it already had.
        Making these changes hot is what keeps "adjust the policy" from meaning
        "throw away everyone's warm cache".
        """
        body, _ = await read_body(reader, rest, headers, 64 * 1024)
        try:
            request = json.loads(body) if body else {}
        except ValueError:
            return 400, {"error": "body is not JSON"}
        if not isinstance(request, dict):
            return 400, {"error": "body must be a JSON object"}
        router = self.fleet.router

        if path.endswith("/route"):
            changed = {}
            if "policy" in request:
                policy = request["policy"]
                custom = getattr(router, "policies", None)
                if custom is not None:
                    custom.reload()
                if policy not in known_policies(router):
                    return 400, {
                        "error": "unknown policy %r" % policy,
                        "known": list(known_policies(router)),
                    }
                router.policy = policy
                changed["policy"] = policy
            if "enabled" in request:
                self.fleet.args.route_by_conversation = bool(request["enabled"])
                changed["enabled"] = self.fleet.args.route_by_conversation
            if "sticky_ttl" in request:
                try:
                    router.ttl = float(request["sticky_ttl"])
                except (TypeError, ValueError):
                    return 400, {"error": "sticky_ttl must be a number"}
                changed["sticky_ttl"] = router.ttl
            if "key_sources" in request:
                sources = request["key_sources"]
                if not isinstance(sources, list) or not sources:
                    return 400, {"error": "key_sources must be a non-empty list"}
                bad = [
                    x
                    for x in sources
                    if not isinstance(x, str)
                    or not (x.startswith(("header:", "body:")) or x == "prefix")
                ]
                if bad:
                    return 400, {
                        "error": "unusable key sources",
                        "sources": bad,
                        "expected": "header:<name>, body:<a.b.c>, or prefix",
                    }
                router.key_sources = sources
                changed["key_sources"] = sources
            if not changed:
                return 400, {
                    "error": "nothing to change",
                    "accepts": ["policy", "enabled", "sticky_ttl", "key_sources"],
                }
            router.dirty = True
            router.save()
            LOG.info("routing changed: %s", changed)
            return 200, {
                "changed": changed,
                "policy": router.policy,
                "enabled": self.fleet.args.route_by_conversation,
            }

        if path.endswith("/pin"):
            key = request.get("conversation")
            if not isinstance(key, str) or not key:
                return 400, {"error": "conversation is required"}
            job_id = request.get("backend")
            if job_id in (None, "", False):
                removed = router.unpin(key)
                router.save()
                LOG.info("unpinned %s (was %s)", key[:40], removed)
                return 200, {"conversation": key, "unpinned": removed}
            job_id = str(job_id)
            if job_id not in self.fleet.backends:
                return 404, {
                    "error": "no such backend",
                    "backend": job_id,
                    "known": sorted(self.fleet.backends),
                }
            router.pin(key, job_id)
            router.save()
            LOG.info("pinned %s -> %s", key[:40], job_id)
            return 200, {"conversation": key, "backend": job_id}

        if path.endswith("/backend"):
            # Registration is a file in the fleet directory, so this endpoint
            # writes one rather than keeping a second, HTTP-shaped list of
            # backends beside it. Two sources of truth for "what is in the
            # fleet" would disagree within one discovery sweep, and the sweep
            # would win: anything registered only in memory is removed five
            # seconds later for not being in the directory.
            job_id = request.get("job_id")
            url = request.get("url")
            reg = registration_path(self.fleet.args.fleet_dir, job_id)
            if reg is None:
                return 400, {
                    "error": "job_id must be 1-64 chars of [A-Za-z0-9._-] and start alphanumeric",
                    "job_id": job_id,
                }
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                return 400, {"error": "url is required and must be http(s)://", "url": url}
            record = {
                "job_id": job_id,
                "url": url.rstrip("/"),
                "run_dir": str(request.get("run_dir") or ""),
                "state": str(request.get("state") or "registered by hand"),
                "end_time": 0,
                "heartbeat": time.time(),
                # Nothing will write heartbeats for this one; see discover().
                "manual": True,
            }
            try:
                candidate = Backend(record)
            except (KeyError, ValueError, TypeError, AttributeError) as exc:
                return 400, {"error": "unusable registration: %s" % exc}
            # Probe before writing. fleetctl once decided a deployment was ready
            # by reading a state string, killed the instance it was replacing,
            # and left a serving gap -- the lesson being that only an answer
            # from the address counts as evidence about the address. Pass
            # "probe": false to register something that is still loading.
            verdict = "skipped"
            if request.get("probe", True):
                verdict = await probe(candidate, self.fleet.args.probe_timeout)
                if verdict != "ok":
                    return 503, {
                        "error": "backend did not answer /health",
                        "verdict": verdict,
                        "url": record["url"],
                        "hint": 'pass "probe": false to register it anyway',
                    }
            existing = job_id in self.fleet.backends
            try:
                tmp = reg + ".tmp"
                with open(tmp, "w") as handle:
                    json.dump(record, handle)
                os.replace(tmp, reg)
            except OSError as exc:
                return 500, {"error": "could not write registration: %s" % exc}
            # Take effect now rather than at the next sweep, so the caller can
            # send the next request to a fleet that already contains this.
            self.fleet.discover()
            LOG.info(
                "backend registered by hand: %s at %s (probe %s)", job_id, record["url"], verdict
            )
            return 200, {
                "registered": job_id,
                "url": record["url"],
                "probe": verdict,
                "replaced": existing,
                "backends": sorted(self.fleet.backends),
            }

        if path.endswith("/backend/remove"):
            job_id = request.get("job_id")
            reg = registration_path(self.fleet.args.fleet_dir, job_id)
            if reg is None:
                return 400, {"error": "job_id is not a usable name", "job_id": job_id}
            if not os.path.exists(reg) and job_id not in self.fleet.backends:
                return 404, {
                    "error": "no such backend",
                    "backend": job_id,
                    "known": sorted(self.fleet.backends),
                }
            # Held out of new placement first. Deleting the file alone works,
            # but every conversation pinned here migrates at once and rebuilds
            # a prefix cache it already had; pausing first lets the ones in
            # flight finish where their cache is. Pass "drain": false to take
            # it out immediately.
            drained = False
            if request.get("drain", True):
                router.paused.add(job_id)
                router.dirty = True
                router.save()
                drained = True
            still = router.counts([job_id]).get(job_id, 0)
            if drained and still and not request.get("force"):
                return 202, {
                    "draining": job_id,
                    "conversations_still_here": still,
                    "message": "held out of new placement; call again with "
                    '"force": true to remove it now, or wait for '
                    "these to finish",
                }
            try:
                if os.path.exists(reg):
                    os.remove(reg)
            except OSError as exc:
                return 500, {"error": "could not remove registration: %s" % exc}
            self.fleet.discover()
            router.paused.discard(job_id)
            router.dirty = True
            router.save()
            LOG.info(
                "backend deregistered by hand: %s (%d conversations were pinned)", job_id, still
            )
            return 200, {
                "removed": job_id,
                "conversations_moved": still,
                "backends": sorted(self.fleet.backends),
            }

        # /drain
        job_id = request.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            return 400, {"error": "job_id is required"}
        if job_id not in self.fleet.backends:
            return 404, {
                "error": "no such backend",
                "backend": job_id,
                "known": sorted(self.fleet.backends),
            }
        accepting = request.get("accepting")
        if not isinstance(accepting, bool):
            return 400, {"error": "accepting must be true or false"}
        if accepting:
            router.paused.discard(job_id)
        else:
            router.paused.add(job_id)
        router.dirty = True
        router.save()
        LOG.info(
            "backend %s %s new conversations",
            job_id,
            "accepts" if accepting else "no longer accepts",
        )
        # Conversations already pinned here keep running: this holds a backend
        # out of new placement, it does not evict anybody.
        return 200, {
            "job_id": job_id,
            "accepting": accepting,
            "paused": sorted(router.paused),
            "conversations_still_here": router.counts([job_id])[job_id],
        }

    async def proxy(self, backend, method, path, headers, rest, body, reader, writer, user, trace):
        up_reader, up_writer = await asyncio.open_connection(backend.host, backend.port)
        try:
            up_writer.write(self.upstream_head(backend, method, path, headers, user))
            pump = None
            if body is not None:
                # Already read in full, to identify the conversation. Forward it
                # in one piece; there is nothing left for a pump to carry.
                if body:
                    up_writer.write(body)
                await up_writer.drain()
            else:
                # Not buffered, so nothing here understands the body. The pump
                # runs until the client stops sending or the response finishes,
                # which serves content-length and chunked alike.
                if rest:
                    up_writer.write(rest)
                await up_writer.drain()
                pump = asyncio.create_task(relay(reader, up_writer, trace))
            try:
                return await self.relay_response(up_reader, writer, trace)
            finally:
                if pump is not None:
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


async def in_thread(fn, *args):
    """Run blocking work on the default executor.

    asyncio.to_thread reads better but is 3.9+, and this file runs under
    whatever python3 the CPU node happens to have -- deliberately, so the
    gateway does not depend on the serving container it outlives.
    """
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def read_fleet_state(fleet):
    """The blocking half of one discovery sweep. Runs on a thread.

    reload_users() belongs here because it ends in a single rebind of
    fleet.users, which a reader either sees or does not; the registration
    records are returned rather than applied for the opposite reason.
    """
    fleet.reload_users()
    return fleet.read_registrations()


async def discovery_loop(fleet):
    while True:
        try:
            # Reads on a thread, mutation on the loop. The fleet directory is
            # on the shared filesystem every serving job writes to, and a stall
            # there used to land directly on the request path: 46 backends cost
            # ~45 ms a sweep there against ~1 ms locally.
            records = await in_thread(read_fleet_state, fleet)
            fleet.discover(records)
            if fleet.router.policies is not None:
                fleet.router.policies.reload()
        except Exception:
            LOG.exception("discovery failed")
        await asyncio.sleep(fleet.args.discover_interval)


async def router_state_loop(fleet):
    """Snapshot the pin table periodically rather than on every request.

    Saving inline would put a filesystem write on the request path for a
    benefit that is entirely about surviving a restart. A periodic flush costs
    at most one interval's worth of pins, and losing those only means a few
    conversations re-home.
    """
    while True:
        await asyncio.sleep(30)
        try:
            # Snapshot on the loop, write on a thread: at 20000 pins the write
            # is a 43 ms json.dump onto the shared filesystem, and the snapshot
            # is what makes it safe to leave the loop.
            state = fleet.router.snapshot()
            if state is not None:
                await in_thread(fleet.router.write_snapshot, state)
        except Exception:
            LOG.exception("router state save failed")


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
    try:
        proc = await asyncio.create_subprocess_exec(
            fleet.args.serve_sh,
            *serve_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as exc:
        # --no-relay skips the start-up check that this path exists, so a
        # missing or unexecutable serve.sh first shows up here. Returning like
        # any other failure keeps it local: raising would abandon the rest of
        # the sweep, and every backend after this one would go unexamined
        # while having already spent its retry budget.
        LOG.error("cannot run %s: %s", fleet.args.serve_sh, exc)
        return None, ""
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace").strip()


async def run_fleetctl(fleet, *fleetctl_args):
    """Invoke fleetctl, the only thing that knows the configured fleet."""
    try:
        proc = await asyncio.create_subprocess_exec(
            fleet.args.fleetctl,
            "--config",
            fleet.args.fleet_config,
            *fleetctl_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as exc:
        LOG.error("cannot run %s: %s", fleet.args.fleetctl, exc)
        return None, ""
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


async def find_untracked_job(fleet):
    """Newest queued job for this deployment that the fleet has no handle on."""
    name = os.path.basename(os.path.normpath(fleet.args.fleet_dir))
    code, out = await run_slurm_command(
        "squeue", "--me", "--noheader", "--name", name, "--format", "%i", "--sort", "-V"
    )
    if code != 0:
        LOG.error("cannot reconcile %s by job name (rc=%s): %s", name, code, out)
        return None
    known = set(fleet.backends) | ({fleet.pending[0]} if fleet.pending else set())
    return next((job_id for job_id in out.split() if job_id not in known), None)


async def submit_successor(fleet, now, label):
    fleet.last_submit = now
    code, out = await run_serve_sh(fleet, "submit", "--yaml", fleet.args.yaml, "--label", label)
    match = re.search(r"Submitted batch job (\d+)", out)
    if code == 0 and match:
        fleet.pending = (match.group(1), now)
        LOG.info("successor submitted: job %s", match.group(1))
        return True
    LOG.error("submit failed (rc=%d): %s", code, out)
    job_id = await find_untracked_job(fleet)
    if job_id is None:
        return False
    fleet.pending = (job_id, now)
    LOG.warning("adopted %s: submit reported failure but the job exists", job_id)
    return True


async def release_job(fleet, job_id, run_dir):
    """Stop one serving job. `quit` needs a live control dir; else scancel."""
    if run_dir:
        code, out = await run_serve_sh(fleet, "quit", run_dir)
        if code == 0:
            LOG.info("released %s via quit (%s)", job_id, run_dir)
            return True
        LOG.warning("quit %s failed (rc=%s): %s; falling back to scancel", job_id, code, out)
    code, out = await run_slurm_command("scancel", job_id)
    if code == 0:
        LOG.info("released %s via scancel", job_id)
        return True
    LOG.error("scancel %s failed (rc=%s): %s", job_id, code, out)
    return False


async def start_server(fleet):
    """POST /_gateway/start_server: get a serving job going if there is none."""
    fleet.stopped = False
    if fleet.backends or fleet.pending:
        # Name the job being adopted: the one still loading if there is one,
        # else the one taking traffic, else whatever registered.
        if fleet.pending:
            job_id = fleet.pending[0]
        else:
            job_id = fleet.active or sorted(fleet.backends)[0]
        LOG.info("start_server: a serving job is already present (%s)", job_id)
        return 200, {"action": "adopted", "job_id": job_id, "active": fleet.active}
    if await submit_successor(fleet, time.time(), "start"):
        return 200, {
            "action": "submitted",
            "job_id": fleet.pending[0],
            "active": None,
            "message": "queued; poll /_gateway/health until status is ok",
        }
    return 503, {
        "action": "submit_failed",
        "job_id": None,
        "active": None,
        "message": "sbatch refused the job; the gateway log has its output",
    }


async def stop_server(fleet):
    """POST /_gateway/stop_server: release every serving job, keep the gateway."""
    targets = []
    if fleet.pending and fleet.pending[0] not in fleet.backends:
        # Submitted but never registered, so scancel is the only handle on it.
        targets.append((fleet.pending[0], ""))
    targets.extend((job_id, b.run_dir) for job_id, b in sorted(fleet.backends.items()))
    fleet.pending = None
    # The supervisor resubmits whenever a fleet that once had a backend goes
    # empty, which is exactly what is about to happen. Recorded in its own flag
    # rather than by clearing ever_active: the job being released stays
    # discoverable until its heartbeat goes stale, so it can win one more
    # election on the way down -- and elect() sets ever_active back to True.
    fleet.stopped = True

    inflight = sum(fleet.inflight.get(job_id, 0) for job_id, _ in targets)
    fleet.draining.clear()
    fleet.superseded.clear()
    # Stop routing before the servers go down: a clean 503 beats a truncated
    # response mid-stream.
    fleet.active = None

    outcomes = await asyncio.gather(
        *[release_job(fleet, job_id, run_dir) for job_id, run_dir in targets],
        return_exceptions=True,
    )
    released, failed = [], []
    for (job_id, _), outcome in zip(targets, outcomes):
        if isinstance(outcome, BaseException):
            LOG.error("releasing %s raised: %r", job_id, outcome)
        (released if outcome is True else failed).append(job_id)

    LOG.info(
        "stop_server: released %s (interrupted %d request(s))%s",
        ", ".join(released) or "nothing",
        inflight,
        "; FAILED on %s" % ", ".join(failed) if failed else "",
    )
    message = (
        "could not stop %s; check it with squeue and scancel it by hand" % ", ".join(failed)
        if failed
        else "POST /_gateway/start_server to bring a serving job back"
    )
    return (502 if failed else 200), {
        "action": "stopped",
        "released": released,
        "failed": failed,
        "interrupted_requests": inflight,
        "message": message,
    }


def attempt_failed(backend):
    """True when a deployment has declared its own server dead.

    serve.sh writes this state itself and keeps the allocation, waiting for a
    `restart` control file that nothing writes on its own. Read from the state
    rather than inferred from probes: a failed probe can be a network blip,
    while this string is the deployment saying so.
    """
    return " exited with status " in backend.state or backend.state.startswith("stopped;")


async def supervise_pending(fleet, now):
    """Keep a submitted successor tracked until it is actually healthy."""
    if not fleet.pending:
        return
    job_id, submitted_at = fleet.pending
    backend = fleet.backends.get(job_id)
    if backend is not None:
        if not backend.healthy and attempt_failed(backend):
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

    if state == "RUNNING" and job_id not in fleet.seen_running:
        fleet.seen_running.add(job_id)
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


async def revive_dead_backends(fleet, now):
    """Restart an in-service deployment that exited but kept its allocation.

    `supervise_pending` already does exactly this, but only for a successor the
    gateway itself submitted -- so a backend that served for hours and then had
    its server killed sat dead indefinitely, holding eight idle nodes, while
    the controller waited for a `restart` nobody was going to write. Twice in
    one night that cost tens of minutes of a quarter of the fleet.

    Restarting in place reuses the existing allocation and takes about ten
    seconds, against roughly fifteen minutes and eight more nodes to roll.

    Deliberately conservative, because this is the routing process taking a
    lifecycle action:
      - only on the deployment's own "exited" state, never on probe failure;
      - never while a roll is in flight (`draining`/`superseded`), or the
        gateway would fight fleetctl over the same job;
      - only while the heartbeat is fresh, since a controller that is gone
        cannot see the file;
      - capped per job, so a deployment that cannot start is left alone and
        reported instead of being restarted forever.
    """
    if fleet.args.revive_limit <= 0:
        return
    pending_job = fleet.pending[0] if fleet.pending else None
    for job_id, backend in sorted(fleet.backends.items()):
        if job_id == pending_job:
            continue  # supervise_pending owns this one
        if job_id in fleet.draining or job_id in fleet.superseded:
            continue
        if backend.healthy or not attempt_failed(backend):
            continue
        if now - backend.heartbeat > fleet.args.stale_after:
            continue  # controller is not there to act
        tries, last = fleet.revived.get(job_id, (0, 0.0))
        if now - last < fleet.args.revive_cooldown:
            continue
        if tries >= fleet.args.revive_limit:
            if tries == fleet.args.revive_limit:
                LOG.error(
                    "%s exited %d times; leaving it alone -- roll or investigate it by hand",
                    job_id,
                    tries,
                )
                fleet.revived[job_id] = (tries + 1, now)
            continue
        LOG.warning(
            "%s is not serving (%s); restarting its retained allocation (attempt %d of %d)",
            job_id,
            backend.state,
            tries + 1,
            fleet.args.revive_limit,
        )
        fleet.revived[job_id] = (tries + 1, now)
        code, out = await run_serve_sh(fleet, "restart", backend.run_dir)
        if code != 0:
            LOG.error("restart %s failed (rc=%d): %s", job_id, code, out)


def instance_label(run_dir):
    """serve.sh names a run <user>_<date>_<jobid>_<cfgname>_<label>."""
    base = os.path.basename(os.path.normpath(run_dir or ""))
    return base.rsplit("_", 1)[-1] if "_" in base else ""


async def recover_lost_backends(fleet, now):
    """Bring back instances the scheduler is not going to bring back itself.

    revive_dead_backends cannot cover preemption. It acts on the deployment's
    own "exited" state, written by a controller that is still running -- and
    preemption takes the controller with it. All three ways a node is lost end
    with the backend simply absent: SIGTERM and "allocation gone" delete the
    registration through clear_fleet, and SIGKILL leaves one that goes stale.
    Nothing in the table is left to iterate, so recovery works off `fleet.lost`
    instead.

    The hard part is not noticing, it is not fighting SLURM. `PreemptMode` here
    is REQUEUE, so a preempted job usually comes back on its own, keeping its
    job id -- and resubmitting on top of that both duplicates the instance and
    throws away the PreemptExemptTime already earned. So nothing is submitted
    until the scheduler says it has no record of the job at all, which is the
    one state that means nobody else is going to bring it back.

    The action is `fleetctl up`, not a direct submit, because fleetctl is what
    knows the configured fleet -- and it reconciles against squeue itself, so
    an instance the scheduler is still holding is skipped there too. That makes
    this idempotent: the worst a spurious call does is print what is already
    running.
    """
    if not fleet.args.fleet_config or fleet.args.recover_limit <= 0:
        return
    if now - fleet.last_recovery < fleet.args.recover_cooldown:
        return
    due = [
        (j, rec)
        for j, rec in sorted(fleet.lost.items())
        if now - rec[1] >= fleet.args.recover_grace
    ]
    if not due:
        return

    orphaned = []
    for job_id, (run_dir, _) in due:
        status = await slurm_job_status(job_id)
        if status is None:
            continue  # cannot tell; ask again next sweep
        if status[0] != "GONE":
            continue  # requeued or still queued: not ours
        tries, _ = fleet.recovered.get(job_id, (0, 0.0))
        if tries >= fleet.args.recover_limit:
            if tries == fleet.args.recover_limit:
                LOG.error(
                    "%s never came back after %d recovery attempts; leaving it to an operator",
                    job_id,
                    tries,
                )
                fleet.recovered[job_id] = (tries + 1, now)
            continue
        orphaned.append((job_id, run_dir, tries))
    if not orphaned:
        return

    labels = sorted(
        {label for label in (instance_label(run_dir) for _, run_dir, _ in orphaned) if label}
    )
    LOG.warning(
        "scheduler has no record of %s (was %s); reconciling the fleet",
        ", ".join(job_id for job_id, _, _ in orphaned),
        ", ".join(labels) if labels else "unlabelled",
    )
    # Counted before the call, so a fleetctl that cannot run still spends the
    # budget rather than being retried every cooldown forever.
    for job_id, _, tries in orphaned:
        fleet.recovered[job_id] = (tries + 1, now)
    fleet.last_recovery = now
    code, out = await run_fleetctl(fleet, "up")
    if code == 0:
        for job_id, _, _ in orphaned:
            fleet.lost.pop(job_id, None)
        LOG.info("fleetctl up: %s", out or "(no output)")
    else:
        # Left in `lost` on purpose: a scheduler that was busy this minute may
        # not be the next, and the attempt counter bounds the retries.
        LOG.error("fleetctl up failed (rc=%s): %s", code, out)


async def check_fleetctl(fleet):
    """Prove at startup that recovery could actually run.

    The failure this catches is quiet and slow: fleetctl has to reach the
    cluster's scheduler, and a gateway running anywhere else cannot. Pointed at
    a local fleetctl from off-cluster it execs fine and then fails on squeue --
    which nobody sees until the first preemption, hours later, by which time
    the fleet has been shrinking unattended. One `status` call here turns that
    into a line in the startup log.

    A warning rather than a refusal: the scheduler being briefly unreachable is
    not a reason to decline to route traffic, and recovery retries anyway.
    """
    if not fleet.args.fleet_config:
        return
    code, out = await run_fleetctl(fleet, "status")
    if code == 0:
        LOG.info("preemption recovery ready: %s %s", fleet.args.fleetctl, fleet.args.fleet_config)
        return
    LOG.warning(
        "preemption recovery is configured but `%s --config %s status` failed (rc=%s): %s",
        fleet.args.fleetctl,
        fleet.args.fleet_config,
        code,
        (out or "")[-300:],
    )
    LOG.warning(
        "nothing will bring a preempted instance back until this works. "
        "fleetctl runs squeue and sbatch, so it has to run on the cluster's "
        "login node -- from elsewhere, point --fleetctl at a wrapper that "
        "ssh-es there and give --fleet-config the path as that host sees it."
    )


async def supervise(fleet):
    now = time.time()
    await supervise_pending(fleet, now)
    await revive_dead_backends(fleet, now)
    # Ahead of the --no-relay return below, deliberately. Relay is about a job
    # reaching its own wall clock, which a proxy-only gateway has no business
    # preempting; losing a node to the scheduler is the opposite situation --
    # nothing else is watching for it, and the fleet only shrinks until
    # somebody notices.
    await recover_lost_backends(fleet, now)

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
    elif (
        not fleet.args.no_relay
        and fleet.ever_active
        and not fleet.stopped
        and not fleet.backends
        and not fleet.pending
        and now - fleet.last_submit >= fleet.args.min_submit_interval
    ):
        # Recovery cannot depend on a live active backend: a cancelled pending
        # job may disappear just as its predecessor reaches the wall clock.
        LOG.warning("fleet lost every backend; submitting recovery successor")
        await submit_successor(fleet, now, "recovery")

    # Reclaim, and the drain that prepares for it, are two halves of the same
    # authority, and --no-relay withholds it. Draining a backend that will
    # never be reclaimed only takes it out of rotation for nothing -- which is
    # exactly what it did to every instance but the longest-lived one once
    # routing began serving them all at once.
    if fleet.args.no_relay:
        return

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
    # is worth releasing a little early.
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
    parser.add_argument(
        "--min-submit-interval",
        type=int,
        default=300,
        help="minimum seconds between recovery submits (default 5min); start_server ignores it",
    )
    parser.add_argument("--discover-interval", type=float, default=5.0)
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--supervisor-interval", type=float, default=30.0)
    parser.add_argument("--probe-timeout", type=float, default=3.0)
    parser.add_argument(
        "--revive-limit",
        type=int,
        default=3,
        help="restart a deployment that exited but kept its allocation, at most "
        "this many times (0 disables)",
    )
    parser.add_argument(
        "--revive-cooldown",
        type=int,
        default=180,
        help="seconds to wait between restart attempts on the same job",
    )
    parser.add_argument(
        "--fleet-config",
        default="",
        help="fleet.yaml to reconcile against when an instance is lost to the "
        "scheduler; unset disables preemption recovery entirely",
    )
    parser.add_argument(
        "--fleetctl",
        default="",
        help="fleet launcher used for that reconciliation (defaults to fleetctl next to this file)",
    )
    parser.add_argument(
        "--recover-grace",
        type=int,
        default=120,
        help="seconds a backend must stay absent before the scheduler is asked "
        "whether anything still owns its job",
    )
    parser.add_argument(
        "--recover-cooldown",
        type=int,
        default=600,
        help="seconds between fleet reconciliations, whatever went missing",
    )
    parser.add_argument(
        "--recover-limit",
        type=int,
        default=3,
        help="reconcile on behalf of the same lost job at most this many times (0 disables)",
    )
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
        "--sticky-ttl",
        type=float,
        default=1800.0,
        help="seconds of silence after which a conversation loses its pin (default 1800)",
    )
    parser.add_argument(
        "--sticky-capacity",
        type=int,
        default=20000,
        help="most conversations pinned at once; the oldest are dropped (default 20000)",
    )
    parser.add_argument(
        "--max-body-buffer",
        type=int,
        default=4 * 1024 * 1024,
        # Agent turns resend the whole history, so the request grows with the
        # conversation. 4 MiB covers a long one; past that the request still
        # goes through, it just loses its affinity.
        help="largest request body read whole to identify a conversation (default 4 MiB)",
    )
    parser.add_argument(
        "--new-conversation-margin",
        type=float,
        default=1800.0,
        # A conversation started on a backend with less time left than this
        # would be cut off partway. Better to start it somewhere it can finish.
        help="stop giving a backend new conversations this many seconds before it ends "
        "(default 1800)",
    )
    parser.add_argument(
        "--route-policy",
        default="least_conversations",
        help="how a new conversation picks a backend (default least_conversations). "
        "One of %s, or the name of a file in --policy-dir." % ", ".join(POLICIES),
    )
    parser.add_argument(
        "--policy-dir",
        default=None,
        help="directory of custom routing policies. Each <name>.py defines "
        "select(accepting) and supplies the policy <name>; accepting maps job "
        "id to (conversations, inflight, remaining_seconds). Rescanned when a "
        "file changes, so a policy can be added or fixed without a restart.",
    )
    parser.add_argument(
        "--router-state",
        default=None,
        # Defaults next to the fleet directory below, so the pins live with the
        # deployment they describe rather than in whatever the cwd happens to be.
        help="file the pin table is saved to and restored from "
        "(default <fleet-dir>/../router_state.json; empty string disables)",
    )
    parser.add_argument(
        "--key-source",
        dest="key_sources",
        action="append",
        default=None,
        metavar="SOURCE",
        help="where to look for a conversation id, in order; repeatable. "
        "header:<name>, body:<a.b.c>, or prefix. Replaces the built-in chain.",
    )
    parser.add_argument(
        "--no-conversation-routing",
        dest="route_by_conversation",
        action="store_false",
        help="ignore conversation identity and balance every request independently",
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
    if args.router_state is None:
        args.router_state = os.path.join(
            os.path.dirname(os.path.normpath(args.fleet_dir)), "router_state.json"
        )
    elif not args.router_state:
        args.router_state = None
    if args.key_sources:
        bad = [
            x for x in args.key_sources if not (x.startswith(("header:", "body:")) or x == "prefix")
        ]
        if bad:
            parser.error(
                "unusable --key-source %s; expected header:<name>, "
                "body:<a.b.c> or prefix" % ", ".join(bad)
            )
    if not args.serve_sh:
        args.serve_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.sh")
    if not args.fleetctl:
        args.fleetctl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleetctl")
    if args.fleet_config:
        # --fleetctl is something this host executes, so it is checked here.
        # --fleet-config deliberately is not: it is passed *to* fleetctl, and
        # fleetctl has to run where the cluster's scheduler is. A gateway that
        # outlives the scheduler is by definition not there, so it points
        # --fleetctl at a wrapper that runs the real one over ssh -- and then
        # --fleet-config names a path on that host, which does not exist on
        # this one. Whether the pair actually works is a question only running
        # them can answer; check_fleetctl does that at startup.
        if not os.path.isfile(args.fleetctl):
            parser.error("--fleetctl %s does not exist" % args.fleetctl)
        if not os.access(args.fleetctl, os.X_OK):
            parser.error("--fleetctl %s is not executable" % args.fleetctl)
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
    # After the listener is up, so a slow or unreachable scheduler delays the
    # answer about recovery rather than the serving of traffic.
    await check_fleetctl(fleet)

    async with server:
        await asyncio.gather(
            discovery_loop(fleet),
            health_loop(fleet),
            supervisor_loop(fleet),
            router_state_loop(fleet),
        )


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
