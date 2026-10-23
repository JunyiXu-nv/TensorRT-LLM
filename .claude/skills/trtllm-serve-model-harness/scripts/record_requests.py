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
"""Record what an agent harness sends, before any server supports it (capture only).

Point the harness's base URL at this process. Every request on any path is
recorded in the request-trace shape that summarize_capture.py and the conformance
checks read:

    <out-dir>/requests-capture.jsonl   {"event": "request", "trace_id", "method", "route", "headers", "body"}
    <out-dir>/responses-capture.jsonl  {"event": "response", "trace_id", "status", "http_status", "response",
                                        "timing"}

In forward mode ``timing`` carries ``headers_s``, ``first_byte_s``, ``total_s`` and, for
streams, ``chunks``: ``[seconds since the request arrived, bytes]`` per relayed chunk.

Two modes:
  --forward URL   proxy to an upstream that already serves the harness (another
                  trtllm-serve, or any OpenAI-compatible server) and record its
                  answer too; streamed answers are relayed chunk by chunk and kept
                  byte-exact.
  (default)       answer every request with --status (default 501) and a JSON
                  error. Enough to learn the first request's API and shape; the
                  harness will then report the error.

--session TAG writes TAG into every request record (the validator's last-resort
session and thread identity), for a harness that sends none: run one recorder per
session.

--override KEY=JSON (repeatable, forward mode) sets a top-level field of every JSON
object body before it is forwarded, e.g. ``--override temperature=0`` for greedy
decoding in a protocol differential. The recorded body stays what the harness
sent; the record names the overrides under ``forwarded_overrides``.

Credentials: Authorization, Cookie and any header with key/token/secret in its
name are recorded as "<redacted len=N>" and are NOT forwarded unless
--forward-auth is set. Bodies are recorded as sent; treat the output like
production traces (it can hold user code and secrets the agent printed).

Binds 127.0.0.1 by default.

usage: record_requests.py --out-dir DIR [--port 8398] [--forward http://host:port]
"""

import argparse
import asyncio
import json
import os
import re
import time
import uuid

import aiohttp
from aiohttp import web

SECRET_HEADER = re.compile(r"(authorization|cookie|key|token|secret|password)", re.I)
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def recorded_headers(headers):
    out = []
    for name, value in headers.items():
        out.append([name, f"<redacted len={len(value)}>" if SECRET_HEADER.search(name) else value])
    return out


# A write to a client that went away fails with whatever the socket says: a reset, "Connection
# lost", a timeout, "No route to host" (all OSError), or aiohttp's own wrapper of one of those.
CLIENT_GONE = (OSError, aiohttp.ClientError)


async def _relay_stream(request, resp, up, chunks, timing, start):
    """Relay an event stream chunk by chunk; return how it ended.

    ``completed``; ``client_disconnected`` when a write to the client fails, which can be long
    after the client left (it shows only at the next write); ``upstream_error`` when reading
    the upstream fails. ``chunks`` and ``timing`` keep what arrived until then.
    """
    try:
        await resp.prepare(request)
    except CLIENT_GONE as error:
        timing["client_error"] = type(error).__name__
        return "client_disconnected"
    try:
        async for chunk in up.content.iter_any():
            now = round(time.monotonic() - start, 4)
            timing.setdefault("first_byte_s", now)
            timing.setdefault("chunks", []).append([now, len(chunk)])
            chunks.append(chunk)
            try:
                await resp.write(chunk)
            except CLIENT_GONE as error:
                timing["client_error"] = type(error).__name__
                return "client_disconnected"
    except aiohttp.ClientError as error:
        timing["upstream_error"] = type(error).__name__
        return "upstream_error"
    try:
        await resp.write_eof()
    except CLIENT_GONE as error:
        timing["client_error"] = type(error).__name__
        return "client_disconnected"
    return "completed"


async def handle(request):
    app = request.app
    raw = await request.read()
    try:
        body = json.loads(raw) if raw else None
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = {"_unparsed_bytes": len(raw)}
    trace_id = "cap_" + uuid.uuid4().hex
    start = time.monotonic()
    overrides = app["overrides"] if isinstance(body, dict) and app["forward"] else {}
    record_in = {
        "event": "request",
        "trace_id": trace_id,
        "recorded_at": utc_now(),
        "method": request.method,
        "route": request.path,
        "query": request.query_string or None,
        "status": "accepted",
        "headers": recorded_headers(request.headers),
        "body": body,
    }
    if app["session"]:
        record_in["session"] = app["session"]
    if overrides:
        record_in["forwarded_overrides"] = overrides
        raw = json.dumps(dict(body, **overrides), ensure_ascii=False).encode()
    app["req_out"].write(json.dumps(record_in, ensure_ascii=False) + "\n")
    app["req_out"].flush()

    def record(status, http_status, kind, payload, timing=None):
        app["resp_out"].write(
            json.dumps(
                {
                    "event": "response",
                    "trace_id": trace_id,
                    "finished_at": utc_now(),
                    "status": status,
                    "http_status": http_status,
                    "response": {"kind": kind, "body": payload},
                    "timing": dict(timing or {}, total_s=round(time.monotonic() - start, 4)),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        app["resp_out"].flush()

    upstream = app["forward"]
    if not upstream:
        payload = {
            "error": {
                "message": "recording only: this endpoint records requests and serves nothing",
                "type": "not_implemented",
                "code": "recording_only",
            }
        }
        record("completed", app["status"], "json", payload)
        return web.json_response(payload, status=app["status"])

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP and (app["forward_auth"] or not SECRET_HEADER.search(k))
    }
    url = upstream.rstrip("/") + request.path_qs
    try:
        async with app["client"].request(request.method, url, data=raw, headers=headers) as up:
            ctype = up.headers.get("content-type", "")
            timing = {"headers_s": round(time.monotonic() - start, 4)}
            if "text/event-stream" in ctype:
                resp = web.StreamResponse(status=up.status, headers={"content-type": ctype})
                chunks, status = [], "completed"
                try:
                    status = await _relay_stream(request, resp, up, chunks, timing, start)
                except asyncio.CancelledError:
                    status = "client_disconnected"
                    raise
                finally:
                    record(
                        status,
                        up.status,
                        "sse_text",
                        b"".join(chunks).decode("utf-8", "replace"),
                        timing,
                    )
                return resp
            data = await up.read()
            text = data.decode("utf-8", "replace")
            try:
                record("completed", up.status, "json", json.loads(text), timing)
            except json.JSONDecodeError:
                record("completed", up.status, "text", text, timing)
            return web.Response(
                status=up.status, body=data, headers={"content-type": ctype or "text/plain"}
            )
    except aiohttp.ClientError as error:
        record("upstream_error", 502, "text", type(error).__name__)
        return web.json_response(
            {"error": {"message": "upstream unreachable", "type": "bad_gateway"}}, status=502
        )


async def on_startup(app):
    app["client"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=3600))


async def on_cleanup(app):
    await app["client"].close()
    app["req_out"].close()
    app["resp_out"].close()


def build_app(out_dir, forward=None, forward_auth=False, overrides=None, session=None, status=501):
    """The recorder as an aiohttp application (``main`` serves it)."""
    os.makedirs(out_dir, exist_ok=True)
    app = web.Application(client_max_size=1 << 30)
    app.update(
        forward=forward,
        forward_auth=forward_auth,
        overrides=overrides or {},
        session=session,
        status=status,
        req_out=open(os.path.join(out_dir, "requests-capture.jsonl"), "a"),
        resp_out=open(os.path.join(out_dir, "responses-capture.jsonl"), "a"),
    )
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8398)
    ap.add_argument(
        "--status", type=int, default=501, help="HTTP status to answer with when not forwarding"
    )
    ap.add_argument("--forward", help="upstream base URL to proxy to (record both sides)")
    ap.add_argument(
        "--forward-auth", action="store_true", help="also forward credential headers upstream"
    )
    ap.add_argument(
        "--session",
        help="session tag written into every request record, for a harness that sends no "
        "session identity (one recorder per session)",
    )
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="set a top-level body field before forwarding (recorded as forwarded_overrides)",
    )
    args = ap.parse_args()
    overrides = {}
    for item in args.override:
        key, _, value = item.partition("=")
        overrides[key] = json.loads(value)
    app = build_app(args.out_dir, args.forward, args.forward_auth, overrides, args.session,
                    args.status)  # fmt: skip
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
