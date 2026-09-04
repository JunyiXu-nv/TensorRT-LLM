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
"""Replayable per-request trace dump for the serving frontends.

Enabled by pointing ``TRTLLM_REQUEST_TRACE_DIR`` at a directory; unset leaves the
whole feature inert. Two JSONL files per UTC hour::

    $TRTLLM_REQUEST_TRACE_DIR/
      2026-09-03T14/
        requests-<pid>.jsonl   one line per request, at handler entry
        responses-<pid>.jsonl  one line per request, when the response ends
"""

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.conversation_id import extract_conversation_id_from_headers

REQUEST_TRACE_DIR_ENV = "TRTLLM_REQUEST_TRACE_DIR"

_WRITER_QUEUE_SIZE = 1024
_WRITER_BATCH_SIZE = 32
_WRITER_SHUTDOWN_TIMEOUT_SECONDS = 5

_REQUESTS = "requests"
_RESPONSES = "responses"

_HOUR_BUCKET_LEN = 13

# Requests whose session id cannot be resolved. Structurally common rather than
# exceptional: /v1/responses reads no headers at all, the disaggregated hop
# forwards none, and proxies routinely strip the x-claude-* ones.
_NO_SESSION = "_no_session"

_SESSION_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_SESSION_LEN = 128

# The two request_type values the disaggregated orchestrator stamps on the hops
# it fans a client request out to (openai_disagg_service). A whitelist, not a
# check for the field: "context_and_generation" is the third legal value
# (disaggregated_params validates all three) and it rides on requests a single
# server answers end to end -- the gRPC frontend defaults to it when the proto
# leaves request_type unset, and EPD multimodal sets it on the prefill+decode
# half. Those are client-facing and must stay traced.
_INTERNAL_DISAGG_REQUEST_TYPES = frozenset(("context_only", "generation_only"))


def request_trace_dir_from_env() -> Optional[str]:
    """Read the enabling variable.

    A function rather than a module constant so the value is picked up when the
    server is constructed. Reading it at import time makes the setting invisible
    to anything that imports this module early -- and untestable without
    reloading it.
    """
    value = os.environ.get(REQUEST_TRACE_DIR_ENV)
    return value or None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hour_bucket(recorded_at: str) -> str:
    return recorded_at[:_HOUR_BUCKET_LEN]


def _server_arrival_time(raw_request: Any) -> Optional[float]:
    """The arrival stamp the serving app already takes, in seconds.

    ``ServerArrivalTimeMiddleware`` sets it on every HTTP request before any
    handler runs, off the same steady clock the executor's perf metrics use --
    which is what lets a trace line be lined up with a perf-metrics record.
    Not a wall clock: it counts from an arbitrary origin.
    """
    state = getattr(raw_request, "state", None)
    if state is None:
        return None
    value = getattr(state, "server_arrival_time", None)
    return float(value) if isinstance(value, (int, float)) else None


def sanitize_session_key(value: Optional[str]) -> str:
    """Reduce a client-supplied session id to a stable, bounded key."""
    if not value:
        return _NO_SESSION
    cleaned = _SESSION_UNSAFE.sub("_", str(value).strip())[:_MAX_SESSION_LEN]
    if not cleaned or cleaned.startswith("."):
        return _NO_SESSION
    return cleaned


def resolve_session_key(headers: Optional[Mapping[str, str]], body: Any) -> str:
    """Pick the key a request's trace lines are stamped with.

    Headers first, in the order the sticky-routing path already trusts, then the
    one body field an agent harness is known to carry. This is the single piece
    of normalization the trace does; every other identifier is stored as the
    client sent it and interpreted offline.
    """
    session = extract_conversation_id_from_headers(headers)
    if not session and isinstance(body, dict):
        client_metadata = body.get("client_metadata")
        if isinstance(client_metadata, dict):
            session = client_metadata.get("session_id")
    return sanitize_session_key(session)


def is_internal_disagg_request(body: Any) -> bool:
    """True for a hop the disaggregated orchestrator generated, not client traffic.

    Read off the body because nothing else separates the two. A worker is never
    told which half it is -- the proxy holds a static url list and passes
    neither --server_role nor --disagg_cluster_uri -- and the internal auth
    header is set for only some of these requests, so ``request_type`` is the
    one marker present on every orchestrator hop and on no client request.

    Unknown values read as client traffic. Should a future topology add a fourth
    hop type, its requests get recorded rather than dropped, which is the safe
    direction for a trace to fail in.
    """
    if not isinstance(body, dict):
        return False
    params = body.get("disaggregated_params")
    if not isinstance(params, dict):
        return False
    return params.get("request_type") in _INTERNAL_DISAGG_REQUEST_TYPES


def _dump_headers(headers: Optional[Mapping[str, str]]) -> List[List[str]]:
    """Headers as ordered pairs.

    A dict would lose repeats, and HTTP allows them -- two proxies each append
    their own ``x-forwarded-for``. Starlette's Headers is a multidict whose
    ``items()`` walks the raw list, so both survive here.
    """
    if headers is None:
        return []
    return [[str(name), str(value)] for name, value in headers.items()]


def _route_of(raw_request: Any) -> str:
    """The URL path, e.g. ``/v1/messages``.

    Recorded because the body's schema is per-route and cannot be told apart
    reliably by inspection. Read off the request rather than passed in: the
    Anthropic route forwards into the chat handler with the same Request object,
    so the path stays the one the client called.
    """
    url = getattr(raw_request, "url", None)
    return str(getattr(url, "path", "")) if url is not None else ""


async def read_request_body(raw_request: Any) -> Tuple[Any, Optional[str]]:
    """Return the request body as JSON, falling back to text.

    ``raw_request.json()`` rather than ``json.loads(await body())`` because the
    serving app swaps in a Request subclass that also decodes msgpack bodies, and
    because it memoizes -- the route handler is about to ask for the same body.

    The fallback only fires for a body that never parsed, which by construction
    means the request is on its way to a 400: anything reaching a handler was
    parsed by FastAPI first.
    """
    try:
        return await raw_request.json(), None
    except Exception as error:  # noqa: BLE001 - a trace must not break serving
        try:
            raw = await raw_request.body()
            return raw.decode("utf-8", "replace"), f"{type(error).__name__}: {error}"
        except Exception as inner:  # noqa: BLE001
            return None, f"{type(inner).__name__}: {inner}"


def brief_validation_errors(errors: Any) -> List[Dict[str, Any]]:
    """Reduce ``RequestValidationError.errors()`` to what is safe to store.

    ``input`` echoes the offending value, which for a body-level failure is the
    whole request -- already stored beside this. ``handle`` can hold a live
    exception object, and one unserializable field would drop the entire record,
    losing exactly the payload this exists to keep.
    """
    brief: List[Dict[str, Any]] = []
    try:
        for error in errors:
            if not isinstance(error, Mapping):
                continue
            brief.append(
                {
                    "loc": [str(part) for part in error.get("loc", ())],
                    "type": str(error.get("type", "")),
                    "msg": str(error.get("msg", "")),
                }
            )
    except Exception as error:  # noqa: BLE001
        logger.warning("Failed to summarize validation errors: %s", error)
    return brief


@dataclass
class RequestTraceHandle:
    """What the request hook hands back and the response hook redeems.

    Carries the trace id the two lines are joined on, the session both are
    stamped with, and the engine-side join keys as they become available -- ``client_id``
    only exists once the request has been submitted, and a disaggregated
    frontend never has one at all.

    Held on ``raw_request.state`` as ``request_trace_handle`` so the streaming
    wrapper can reach it without the generator knowing anything about the engine.
    """

    trace_id: str
    session: str
    route: str
    client_id: Optional[int] = None
    disagg_request_id: Optional[int] = None
    ctx_request_id: Optional[int] = None
    response_written: bool = field(default=False, repr=False)

    def set_ids(
        self,
        *,
        client_id: Optional[int] = None,
        disagg_request_id: Optional[int] = None,
        ctx_request_id: Optional[int] = None,
    ) -> None:
        """Record whichever join keys this deployment produces.

        A single-engine frontend has ``client_id`` and no disaggregated ids; a
        disaggregated frontend is an HTTP proxy with no engine and so has the
        reverse. Callers set what they have.
        """
        if client_id is not None:
            self.client_id = client_id
        if disagg_request_id is not None:
            self.disagg_request_id = disagg_request_id
        if ctx_request_id is not None:
            self.ctx_request_id = ctx_request_id


class RequestTraceWriter:
    """Best-effort bounded JSONL writer for request/response traces.

    Mirrors ``PerfMetricsJsonlWriter``: a bounded queue drained by one task that
    batches and hands the blocking write to a thread, started and closed from the
    app lifespan. It differs in fanning out to a file per (hour, kind) instead
    of a single path.

    ``submit`` is deliberately synchronous. The streaming hook calls it from an
    async generator's ``finally``, which on a client disconnect runs while
    ``GeneratorExit`` is propagating; awaiting there risks "async generator
    ignored GeneratorExit" and, during shutdown, may never resume.
    """

    def __init__(self, output_dir: Optional[str], writer_suffix: Optional[str] = None):
        self._output_dir = Path(output_dir) if output_dir else None
        self._writer_suffix = f"-{os.getpid()}" if writer_suffix is None else writer_suffix
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=_WRITER_QUEUE_SIZE)
        self._task: Optional[asyncio.Task] = None
        self._known_dirs: set = set()
        self.dropped_records = 0
        self._write_error_count = 0

    @property
    def enabled(self) -> bool:
        return self._task is not None

    async def start(self) -> None:
        if self._output_dir is None or self._task is not None:
            return
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.error("Disabling request trace output: %s", error)
            self._output_dir = None
            return
        logger.info("Recording request traces to %s", self._output_dir)
        self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._task is None:
            return
        task = self._task
        try:
            await asyncio.wait_for(
                self._queue.put(None),
                timeout=_WRITER_SHUTDOWN_TIMEOUT_SECONDS,
            )
            await asyncio.wait_for(task, timeout=_WRITER_SHUTDOWN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("Timed out flushing request traces; dropping remaining records")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            self._task = None

    # -- hooks ---------------------------------------------------------------

    async def on_request(self, raw_request: Any) -> Optional[RequestTraceHandle]:
        """Record one accepted request and return the handle that owns it.

        The handle is what the response side redeems: it carries the trace id
        the two lines are joined on, plus the join keys as they turn up.

        Returns None in three cases: tracing is off; a handler is re-entering
        on a Request that already has a handle; or the request is an
        orchestrator hop rather than client traffic. ``/v1/messages`` converts
        and then calls the chat handler with the same Request; only the outer
        one may record the response, because only its frames are the ones the
        client receives. Handing the inner one None makes ``wrap_stream`` a
        no-op there without either handler having to know the other exists, and
        the same holds for the hops dropped below.
        """
        if self._task is None:
            return None
        state = getattr(raw_request, "state", None)
        # The Anthropic route forwards into the chat handler with the same
        # Request object, so the hook fires twice for one client request. The
        # first call owns both lines; a second would duplicate the request line
        # and orphan the first handle's trace_id.
        if state is not None and getattr(state, "request_trace_handle", None) is not None:
            return None
        body, parse_error = await read_request_body(raw_request)
        # Dropped before a handle exists, so nothing downstream believes it owns
        # one. The orchestrator re-posts every client request to a context and a
        # generation worker, so a worker that sees the enabling variable records
        # the same conversation twice more. Those lines cost more than they
        # carry: the generation one holds prompt_token_ids, which is tokenizer
        # output and so tied to the model that produced it -- the one thing this
        # trace exists not to record. They are also unjoinable, the two workers
        # numbering client_id from independent counters. What they would have
        # shown is already on the proxy's line, whose usage block carries prompt
        # length, completion length and the cached prefix.
        if is_internal_disagg_request(body):
            return None
        headers = getattr(raw_request, "headers", None)
        route = _route_of(raw_request)
        handle = RequestTraceHandle(
            trace_id=f"tr_{uuid.uuid4().hex}",
            session=resolve_session_key(headers, body),
            route=route,
        )
        if state is not None:
            state.request_trace_handle = handle
        recorded_at = _utc_now()
        record: Dict[str, Any] = {
            "event": "request",
            "trace_id": handle.trace_id,
            "session": handle.session,
            "recorded_at": recorded_at,
            "server_arrival_time": _server_arrival_time(raw_request),
            "route": route,
            "status": "accepted",
            "headers": _dump_headers(headers),
            "body": body,
        }
        if parse_error is not None:
            record["body_parse_error"] = parse_error
        self._submit(_hour_bucket(recorded_at), _REQUESTS, record)
        return handle

    async def on_rejected(self, raw_request: Any, validation_errors: Any) -> None:
        """Record a request that never reached a handler.

        The body of a request rejected by validation exists nowhere else: it
        never reached a worker, and the error response names only the offending
        locations. Written to ``requests.jsonl`` with no response line.

        Orchestrator hops are skipped here as they are in ``on_request``, so a
        worker writes nothing at all. It does cost something: a hop the worker
        422s is a defect in the request the orchestrator built, and that body is
        now recorded nowhere. Taken because the alternative leaves workers
        creating trace files for one rare case, which is the whole arrangement
        this guard exists to prevent -- the client body that provoked it is on
        the proxy's line either way.
        """
        if self._task is None:
            return
        body, parse_error = await read_request_body(raw_request)
        if is_internal_disagg_request(body):
            return
        headers = getattr(raw_request, "headers", None)
        recorded_at = _utc_now()
        record: Dict[str, Any] = {
            "event": "request",
            "trace_id": f"tr_{uuid.uuid4().hex}",
            "session": resolve_session_key(headers, body),
            "recorded_at": recorded_at,
            "server_arrival_time": _server_arrival_time(raw_request),
            "route": _route_of(raw_request),
            "status": "rejected_400",
            "headers": _dump_headers(headers),
            "body": body,
            "validation_errors": brief_validation_errors(validation_errors),
        }
        if parse_error is not None:
            record["body_parse_error"] = parse_error
        self._submit(_hour_bucket(recorded_at), _REQUESTS, record)

    def on_response(
        self,
        handle: Optional[RequestTraceHandle],
        *,
        frames: Optional[List[Any]] = None,
        payload: Any = None,
        status: str = "completed",
    ) -> None:
        """Record the response side. Synchronous: safe to call from ``finally``."""
        if handle is None or self._task is None or handle.response_written:
            return
        handle.response_written = True
        finished_at = _utc_now()
        record: Dict[str, Any] = {
            "event": "response",
            "trace_id": handle.trace_id,
            "session": handle.session,
            "finished_at": finished_at,
            "status": status,
            "client_id": handle.client_id,
            "disagg_request_id": handle.disagg_request_id,
            "ctx_request_id": handle.ctx_request_id,
        }
        if frames is not None:
            record["response"] = {
                "kind": "sse_frames",
                "frames": [_as_text(frame) for frame in frames],
            }
        else:
            record["response"] = {"kind": "json", "body": payload}
        self._submit(_hour_bucket(finished_at), _RESPONSES, record)

    def wrap_stream(
        self, stream: AsyncIterator[Any], handle: Optional[RequestTraceHandle]
    ) -> AsyncIterator[Any]:
        """Tee a streaming response into the trace without touching its producer.

        Wrapping the outermost generator is what makes one implementation cover
        every route: each frame recorded is a frame the client received, in the
        protocol the client speaks, whatever conversions happened upstream.
        """
        if handle is None or self._task is None:
            return stream

        async def _traced() -> AsyncIterator[Any]:
            frames: List[Any] = []
            status = "unknown"
            try:
                async for chunk in stream:
                    frames.append(chunk)
                    yield chunk
                status = "completed"
            except GeneratorExit:
                # Re-raised because swallowing it turns into "async generator
                # ignored GeneratorExit"; recorded because a client that hangs
                # up mid-turn is a sample worth keeping, not an error.
                status = "client_disconnected"
                raise
            except BaseException:
                status = "error"
                raise
            finally:
                self.on_response(handle, frames=frames, status=status)

        return _traced()

    # -- writer --------------------------------------------------------------

    def _submit(self, bucket: str, kind: str, record: Dict[str, Any]) -> None:
        if self._task is None:
            return
        try:
            self._queue.put_nowait((bucket, kind, record))
        except asyncio.QueueFull:
            self.dropped_records += 1
            if self.dropped_records == 1 or self.dropped_records % 1000 == 0:
                logger.warning("Dropped %d request trace records", self.dropped_records)

    async def _run(self) -> None:
        stop = False
        while not stop:
            item = await self._queue.get()
            if item is None:
                return
            batch = [item]
            while len(batch) < _WRITER_BATCH_SIZE:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    stop = True
                    break
                batch.append(item)
            groups: Dict[Tuple[str, str], List[str]] = {}
            for bucket, kind, record in batch:
                try:
                    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                except (TypeError, ValueError) as error:
                    self.dropped_records += 1
                    if self.dropped_records == 1 or self.dropped_records % 1000 == 0:
                        logger.warning("Dropped malformed request trace record: %s", error)
                    continue
                groups.setdefault((bucket, kind), []).append(line)
            if not groups:
                continue
            try:
                await asyncio.to_thread(self._write_groups, groups)
            except OSError as error:
                self.dropped_records += sum(len(lines) for lines in groups.values())
                self._write_error_count += 1
                if self._write_error_count == 1:
                    logger.warning("Failed to write request trace JSONL: %s", error)

    def _write_groups(self, groups: Dict[Tuple[str, str], List[str]]) -> None:
        """Append each group to its file. Runs on a worker thread.

        At most one bucket per kind in practice, so a batch is two opens rather
        than the one-per-session it used to be.
        """
        for (bucket, kind), lines in groups.items():
            directory = self._output_dir / bucket
            if bucket not in self._known_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                self._known_dirs.add(bucket)
            path = directory / f"{kind}{self._writer_suffix}.jsonl"
            with path.open("a", encoding="utf-8") as output:
                output.write("".join(lines))


def _as_text(chunk: Any) -> str:
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", "replace")
    return chunk if isinstance(chunk, str) else str(chunk)
