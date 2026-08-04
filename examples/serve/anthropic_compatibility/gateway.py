#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Stable front door for the Anthropic-compatibility servers.
#
# A serving job lives for hours and lands on whatever node Slurm gives it, so
# its URL changes every time it is rescheduled. Users need one address that
# never changes. This gateway holds that address and forwards to whichever
# backend is currently healthy, so `ANTHROPIC_BASE_URL` is written once and
# never edited again.
#
#   ./serve.sh gateway --yaml deployments/computelab_glm5.2.yaml
#
# Standard library only, on purpose: the gateway has to outlive every serving
# job, so it runs outside the TRT-LLM container on whatever long-lived host is
# available. Requiring httpx or uvicorn there would mean a venv, which means
# outbound network -- one more thing that host has to provide.

import argparse
import asyncio
import collections
import glob
import json
import logging
import os
import re
import sys
import time

LOG = logging.getLogger("gateway")

# Hop-by-hop headers plus the ones this gateway owns. Framing headers
# (content-length, transfer-encoding, te, trailer) are deliberately NOT here:
# the body is relayed byte for byte, so its framing must survive untouched.
STRIP_REQUEST_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "upgrade",
    "host",
    "x-api-key",
    "authorization",
}

MAX_HEAD_BYTES = 64 * 1024
RELAY_CHUNK = 64 * 1024

# Anthropic-shaped so a client's error handling reports something meaningful
# instead of a bare transport failure.
ERROR_BODIES = {
    401: ("authentication_error", "unknown api key; ask the gateway owner to "
                                  "add your username to users.txt"),
    502: ("api_error", "backend refused the connection"),
    503: ("overloaded_error", "no healthy backend right now; the serving job "
                              "is rotating, retry shortly"),
}

SSE_ROTATED = (b'event: error\n'
               b'data: {"type":"error","error":{"type":"overloaded_error",'
               b'"message":"backend rotated mid-stream; the response is '
               b'incomplete, please resend"}}\n\n')


# ---------------------------------------------------------------------------
# Fleet state
# ---------------------------------------------------------------------------
class Backend:
    """One serving job, as seen through its registration file."""

    def __init__(self, record):
        self.job_id = str(record["job_id"])
        self.url = record["url"].rstrip("/")
        self.run_dir = record.get("run_dir", "")
        self.state = record.get("state", "")
        self.end_time = float(record.get("end_time") or 0)
        self.heartbeat = float(record.get("heartbeat") or 0)
        self.healthy = False
        self.timeouts = 0        # consecutive probe timeouts
        self.healthy_since = 0.0
        # Probing resolves the URL once; every request reuses host/port.
        match = re.match(r"^http://([^:/]+):(\d+)$", self.url)
        if not match:
            raise ValueError("unusable url %r" % self.url)
        self.host = match.group(1)
        self.port = int(match.group(2))

    def refresh(self, record):
        self.state = record.get("state", self.state)
        self.heartbeat = float(record.get("heartbeat") or 0)


class Fleet:
    """Everything the request path and the supervisor share."""

    def __init__(self, args):
        self.args = args
        self.backends = {}          # job_id -> Backend
        self.active = None          # job_id currently taking new requests
        self.draining = {}          # job_id -> reclaim deadline (unix ts)
        # A default dict on purpose. Requests outlive their backend's entry --
        # discovery can retire a job while its streams are still draining -- and
        # the release below runs in a finally that also closes the client
        # socket. A KeyError there would leak the connection, so counting must
        # not be able to raise.
        self.inflight = collections.defaultdict(int)
        self.users = set()
        self.users_mtime = 0.0
        self.pending = None         # (job_id, submitted_at) of a successor
        # Replaced, but not yet cleared for reclaim: draining ends in
        # `serve.sh quit`, so it waits until the successor has proven itself.
        self.superseded = set()
        self.started = time.time()

    # -- users ------------------------------------------------------------
    def reload_users(self):
        path = self.args.users
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            if self.users:
                LOG.warning("users file disappeared: %s (keeping %d entries)",
                            path, len(self.users))
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

        Each serving job owns exactly one file named after its Slurm job id, so
        there is never more than one writer per file and the gateway never has
        to coordinate with anybody. The union is just the directory listing.
        """
        now = time.time()
        seen = set()
        for path in glob.glob(os.path.join(self.args.fleet_dir, "*.json")):
            try:
                with open(path) as handle:
                    record = json.load(handle)
            except (OSError, ValueError):
                # Mid-rename, or truncated. The writer replaces the file
                # atomically, so the next sweep gets a whole one.
                continue
            job_id = str(record.get("job_id", ""))
            if not job_id:
                continue
            if now - float(record.get("heartbeat") or 0) > self.args.stale_after:
                continue
            seen.add(job_id)
            if job_id in self.backends:
                self.backends[job_id].refresh(record)
            else:
                try:
                    self.backends[job_id] = Backend(record)
                except (KeyError, ValueError) as exc:
                    LOG.warning("ignoring %s: %s", path, exc)
                    continue
                self.inflight.setdefault(job_id, 0)
                LOG.info("backend appeared: %s at %s (ends %s)", job_id,
                         self.backends[job_id].url,
                         fmt_time(self.backends[job_id].end_time))

        for job_id in [j for j in self.backends if j not in seen]:
            LOG.info("backend gone: %s (no heartbeat for %ds)", job_id,
                     self.args.stale_after)
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

        if self.pending and self.pending[0] in self.backends:
            LOG.info("successor %s registered", self.pending[0])
            self.pending = None

    # -- election ---------------------------------------------------------
    def elect(self):
        """Pick the healthy backend that will live the longest.

        Choosing by end time is what makes relay work without anybody
        orchestrating it: a freshly started job outlives the one it replaces,
        so the moment it passes /health it wins the election on its own.
        """
        candidates = [j for j, b in self.backends.items() if b.healthy]
        winner = max(candidates,
                     key=lambda j: self.backends[j].end_time) if candidates else None
        if winner == self.active:
            return
        previous = self.active
        self.active = winner
        if winner is None:
            LOG.warning("no healthy backend; serving 503")
        else:
            LOG.info("active backend -> %s (%s)", winner,
                     self.backends[winner].url)
            # Won the election back: whatever replaced it is gone or sicker, so
            # it is no longer a candidate for reclaim.
            self.superseded.discard(winner)
        # Only a real handover marks the predecessor. A backend that merely
        # failed a probe -- restarting between attempts, or /health timing out
        # under load -- must not be marked, because this path ends in
        # `serve.sh quit` and would release an allocation that was coming back.
        if winner is not None and previous and previous in self.backends:
            self.superseded.add(previous)
            LOG.info("superseded %s; reclaim held until %s is stable",
                     previous, winner)


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
    head = ["HTTP/1.1 %d %s" % (status, reason),
            "Content-Type: application/json",
            "Content-Length: %d" % len(body),
            "Connection: close"]
    head.extend(extra_headers)
    return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body


def error_response(status, retry_after=None):
    kind, message = ERROR_BODIES[status]
    body = json.dumps({"type": "error",
                       "error": {"type": kind, "message": message}}).encode()
    extra = ["Retry-After: %d" % retry_after] if retry_after else []
    reason = {401: "Unauthorized", 502: "Bad Gateway",
              503: "Service Unavailable"}[status]
    return build_response(status, reason, body, extra)


def json_response(payload):
    body = json.dumps(payload, indent=2).encode()
    return build_response(200, "OK", body)


def chunk_frame(payload, chunked):
    """Wrap an injected SSE event so it survives the response's framing.

    Best effort by construction: if the upstream socket died halfway through a
    chunk, the bytes already forwarded end mid-frame and nothing appended can
    repair that. In practice uvicorn writes one SSE event per chunk, so the
    break lands on a boundary and the client sees a real error event.
    """
    if not chunked:
        return payload
    return b"%x\r\n%s\r\n0\r\n\r\n" % (len(payload), payload)


# ---------------------------------------------------------------------------
# Request path
# ---------------------------------------------------------------------------
class Gateway:

    def __init__(self, fleet):
        self.fleet = fleet

    async def handle(self, reader, writer):
        peer = writer.get_extra_info("peername")
        started = time.time()
        try:
            head, rest = await read_head(reader)
            if head is None:
                return
            method, path, headers = parse_request_head(head)
        except (ValueError, ConnectionError) as exc:
            LOG.debug("bad request from %s: %s", peer, exc)
            await close(writer)
            return

        if path.startswith("/_gateway/"):
            await self.serve_introspection(path, headers, writer)
            return

        key = extract_key(headers)
        if key not in self.fleet.users:
            LOG.info("401 %s %s user=%r", method, path, key)
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
            LOG.info("503 %s %s user=%s (no backend)", method, path, key)
            await respond(writer, error_response(503, retry_after=20))
            return
        self.fleet.inflight[job_id] += 1
        status = "-"
        # Closing the client socket sits in its own finally so that no amount of
        # bookkeeping trouble above can leak the connection.
        try:
            try:
                status = await self.proxy(backend, method, path, headers, rest,
                                          reader, writer, key)
            except (ConnectionError, OSError) as exc:
                LOG.warning("upstream %s failed: %s", backend.url, exc)
                status = "502"
                await respond(writer, error_response(502))
            finally:
                self.fleet.inflight[job_id] -= 1
                LOG.info("%s %s %s user=%s backend=%s %.1fs", status, method,
                         path, key, job_id, time.time() - started)
        finally:
            await close(writer)

    async def serve_introspection(self, path, headers, writer):
        if path == "/_gateway/health":
            healthy = self.fleet.active is not None
            payload = {"status": "ok" if healthy else "no_backend",
                       "active": self.fleet.active,
                       "uptime_s": round(time.time() - self.fleet.started)}
            await respond(writer, json_response(payload))
            return
        if path == "/_gateway/fleet":
            if extract_key(headers) not in self.fleet.users:
                await respond(writer, error_response(401))
                return
            now = time.time()
            payload = {
                "active": self.fleet.active,
                "pending_successor": self.fleet.pending[0] if self.fleet.pending
                                     else None,
                "backends": {
                    job_id: {
                        "url": b.url,
                        "healthy": b.healthy,
                        "healthy_for_s": round(now - b.healthy_since)
                                         if b.healthy_since else None,
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
        await respond(writer, error_response(502))

    async def proxy(self, backend, method, path, headers, rest, reader, writer,
                    user):
        up_reader, up_writer = await asyncio.open_connection(backend.host,
                                                             backend.port)
        try:
            up_writer.write(self.upstream_head(backend, method, path, headers,
                                               user))
            if rest:
                up_writer.write(rest)
            await up_writer.drain()

            # Nothing here parses the request body. The pump runs until the
            # client stops sending or the response finishes, so content-length
            # and chunked bodies both work without being understood.
            pump = asyncio.create_task(relay(reader, up_writer))
            try:
                return await self.relay_response(up_reader, writer)
            finally:
                pump.cancel()
        finally:
            await close(up_writer)

    def upstream_head(self, backend, method, path, headers, user):
        lines = ["%s %s HTTP/1.1" % (method, path),
                 "Host: %s:%d" % (backend.host, backend.port),
                 # Close framing is what lets the response end at EOF, so the
                 # relay below never has to decode chunked encoding.
                 "Connection: close",
                 "X-Gateway-User: %s" % user]
        for name, value in headers:
            if name.lower() in STRIP_REQUEST_HEADERS:
                continue
            lines.append("%s: %s" % (name, value))
        return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")

    async def relay_response(self, up_reader, writer):
        head, rest = await read_head(up_reader)
        if head is None:
            raise ConnectionError("upstream closed before sending a response")
        lowered = head.lower()
        is_sse = b"text/event-stream" in lowered
        chunked = b"transfer-encoding: chunked" in lowered
        status = head.split(b"\r\n", 1)[0].split(b" ")[1].decode("latin-1")

        writer.write(head + b"\r\n\r\n" + rest)
        await writer.drain()
        saw_stop = b"message_stop" in rest

        while True:
            chunk = await up_reader.read(RELAY_CHUNK)
            if not chunk:
                break
            if is_sse and not saw_stop and b"message_stop" in chunk:
                saw_stop = True
            writer.write(chunk)
            await writer.drain()

        if is_sse and not saw_stop:
            # The stream ended without the terminal event, which means the
            # backend went away underneath it. Say so in the protocol the
            # client already parses rather than letting it surface as a reset.
            LOG.warning("stream ended without message_stop; injecting error")
            writer.write(chunk_frame(SSE_ROTATED, chunked))
            await writer.drain()
            status += "!"
        return status


async def relay(reader, writer):
    try:
        while True:
            chunk = await reader.read(RELAY_CHUNK)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, OSError, asyncio.CancelledError):
        pass


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
            asyncio.open_connection(backend.host, backend.port), timeout)
        writer.write(b"GET /health HTTP/1.1\r\nHost: %s\r\n"
                     b"Connection: close\r\n\r\n"
                     % backend.host.encode("latin-1"))
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
            LOG.warning("backend %s unreachable; dropping it now",
                        backend.job_id)
        backend.timeouts = 0
        backend.healthy = False
        backend.healthy_since = 0.0
        return
    backend.timeouts += 1
    if backend.healthy and backend.timeouts >= unhealthy_after:
        LOG.warning("backend %s timed out %d times in a row; marking unhealthy",
                    backend.job_id, backend.timeouts)
        backend.healthy = False
        backend.healthy_since = 0.0
    elif backend.healthy:
        LOG.info("backend %s health probe timed out (%d/%d)", backend.job_id,
                 backend.timeouts, unhealthy_after)


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
                    return_exceptions=True)
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
        fleet.args.serve_sh, *serve_args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    out, _ = await proc.communicate()
    return proc.returncode, out.decode(errors="replace").strip()


async def supervisor_loop(fleet):
    while True:
        try:
            await supervise(fleet)
        except Exception:
            LOG.exception("supervisor failed")
        await asyncio.sleep(fleet.args.supervisor_interval)


async def supervise(fleet):
    now = time.time()

    # Relay: submit the next job early enough that it finishes loading weights
    # before this one hits the wall clock.
    backend = fleet.backends.get(fleet.active) if fleet.active else None
    if backend is not None and not fleet.args.no_relay:
        remaining = backend.end_time - now
        successors = [j for j in fleet.backends if j != fleet.active]
        if backend.end_time <= 0:
            # Registration could not determine the wall clock. Routing still
            # works; relaying on `0 - now` would read as "already expired" and
            # submit a job every single sweep.
            LOG.warning("backend %s has no end time; relay disabled for it",
                        fleet.active)
        elif remaining < fleet.args.lead_time and not successors and not fleet.pending:
            LOG.info("%s ends in %ds; submitting successor",
                     fleet.active, int(remaining))
            code, out = await run_serve_sh(fleet, "submit",
                                           "--yaml", fleet.args.yaml,
                                           "--label", "relay")
            match = re.search(r"Submitted batch job (\d+)", out)
            if code == 0 and match:
                fleet.pending = (match.group(1), now)
                LOG.info("successor submitted: job %s", match.group(1))
            else:
                LOG.error("submit failed (rc=%d): %s", code, out)

    # A submitted job that never shows up (held, cancelled, node failure) must
    # not block the next attempt forever.
    if fleet.pending and now - fleet.pending[1] > fleet.args.lead_time:
        LOG.warning("successor %s never registered; clearing", fleet.pending[0])
        fleet.pending = None

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
                backend = fleet.backends.get(job_id)
                if backend is None:
                    continue
                deadline = backend.end_time - 60
                fleet.draining[job_id] = deadline
                LOG.info("draining %s (%s stable %ds, inflight=%d, reclaim "
                         "by %s)", job_id, fleet.active, int(stable_for),
                         fleet.inflight.get(job_id, 0), fmt_time(deadline))

    # Reclaim: the drained job is already past being useful, and its allocation
    # is worth releasing a little early. Skipped under --no-relay, which
    # promises not to touch job lifecycles at all -- submitting and reclaiming
    # are two halves of the same authority.
    if fleet.args.no_relay:
        return
    for job_id, deadline in list(fleet.draining.items()):
        backend = fleet.backends.get(job_id)
        if backend is None:
            fleet.draining.pop(job_id, None)
            continue
        inflight = fleet.inflight.get(job_id, 0)
        if inflight and now <= deadline:
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
        description="stable front door for the Anthropic-compatibility servers")
    parser.add_argument("--fleet-dir", required=True,
                        help="directory the serving jobs register into")
    parser.add_argument("--users", required=True,
                        help="allowlist, one username per line")
    parser.add_argument("--yaml", default="",
                        help="deployment YAML the supervisor resubmits")
    parser.add_argument("--serve-sh", default="",
                        help="path to serve.sh (defaults next to this file)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8333)
    parser.add_argument("--lead-time", type=int, default=2700,
                        help="seconds before the wall clock to submit the "
                             "successor (default 45min)")
    parser.add_argument("--stale-after", type=int, default=30,
                        help="drop a backend after this long without a "
                             "heartbeat")
    parser.add_argument("--discover-interval", type=float, default=5.0)
    parser.add_argument("--health-interval", type=float, default=5.0)
    parser.add_argument("--supervisor-interval", type=float, default=30.0)
    parser.add_argument("--probe-timeout", type=float, default=3.0)
    parser.add_argument("--unhealthy-after", type=int, default=20,
                        help="consecutive probe timeouts before a backend is "
                             "taken out of rotation; a refused connection is "
                             "acted on immediately regardless")
    parser.add_argument("--promote-after", type=float, default=180.0,
                        help="seconds a successor must stay healthy before its "
                             "predecessor may be reclaimed")
    parser.add_argument("--no-relay", action="store_true",
                        help="proxy only; never submit a successor and never "
                             "reclaim a drained job")
    args = parser.parse_args(argv)
    if not args.serve_sh:
        args.serve_sh = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "serve.sh")
    if not args.no_relay and not args.yaml:
        parser.error("--yaml is required unless --no-relay is given")
    return args


async def main_async(args):
    fleet = Fleet(args)
    os.makedirs(args.fleet_dir, exist_ok=True)
    fleet.reload_users()
    if not fleet.users:
        LOG.warning("users file %s is empty; every request will get 401",
                    args.users)
    fleet.discover()

    gateway = Gateway(fleet)
    server = await asyncio.start_server(gateway.handle, args.host, args.port)
    LOG.info("listening on %s:%d", args.host, args.port)
    LOG.info("fleet dir: %s", args.fleet_dir)
    LOG.info("relay: %s", "off" if args.no_relay
             else "lead time %ds from %s" % (args.lead_time, args.yaml))

    async with server:
        await asyncio.gather(discovery_loop(fleet),
                             health_loop(fleet),
                             supervisor_loop(fleet))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S")
    args = parse_args(sys.argv[1:])
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        LOG.info("interrupted")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
