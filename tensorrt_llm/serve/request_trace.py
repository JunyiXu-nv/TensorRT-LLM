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
import contextlib
import functools
import json
import os
import re
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.conversation_id import find_conversation_id

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

# Request headers whose values are credentials, matched case-insensitively: the
# names below, plus any name containing a fragment, which is what catches
# x-api-key, api-key and the vendor spellings (x-auth-token,
# x-amz-security-token, x-client-secret, ...).
_CREDENTIAL_HEADERS = frozenset(("authorization", "proxy-authorization", "cookie", "set-cookie"))
_CREDENTIAL_HEADER_FRAGMENTS = ("api-key", "api_key", "apikey", "token", "secret", "password")
_REDACTED = "[redacted]"


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


def _running_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


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

    The id comes from ``find_conversation_id``, the function sticky routing
    resolves through, so the precedence is the router's by construction: body
    ``conversation_params`` first, then the headers, then the client-native body
    fields. A trace line's session is therefore the id the router keyed on, put
    through ``sanitize_session_key`` -- the single piece of normalization the
    trace does; every other identifier is stored as the client sent it and
    interpreted offline.
    """
    return sanitize_session_key(find_conversation_id(body, headers))


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


def _is_credential_header(name: str) -> bool:
    lowered = name.lower()
    return lowered in _CREDENTIAL_HEADERS or any(
        fragment in lowered for fragment in _CREDENTIAL_HEADER_FRAGMENTS
    )


def _dump_headers(headers: Optional[Mapping[str, str]]) -> List[List[str]]:
    """Headers as ordered pairs, with credential values redacted.

    A dict would lose repeats, and HTTP allows them -- two proxies each append
    their own ``x-forwarded-for``. Starlette's Headers is a multidict whose
    ``items()`` walks the raw list, so both survive here.

    Every header the trace records passes through here, so this is where
    credentials stop: a credential header keeps its name, which shows the
    client sent one, and its value becomes ``_REDACTED``.
    """
    if headers is None:
        return []
    return [
        [str(name), _REDACTED if _is_credential_header(str(name)) else str(value)]
        for name, value in headers.items()
    ]


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
        logger.warning(f"Failed to summarize validation errors: {error}")
    return brief


@functools.lru_cache(maxsize=None)
def _tool_call_markup_tokens() -> Tuple[str, ...]:
    """The markup every registered tool-call format writes around its calls."""
    from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

    tokens = set()
    for parser_class in ToolParserFactory.parsers.values():
        tokens.update(getattr(parser_class, "markup_tokens", ()))
    return tuple(sorted(tokens))


_TOOL_CALL_ITEM_TYPES = {"function_call": "arguments", "custom_tool_call": "input"}


def _responses_tool_call_items(text: Optional[str], payload: Any) -> List[Dict[str, Any]]:
    """The function and custom tool calls a Responses reply delivered.

    Read from the streamed ``output_item.done`` events and the terminal
    snapshot alike, so a stream cut before its terminal event still counts.
    """
    items: List[Dict[str, Any]] = []
    if text is None:
        output = payload.get("output") if isinstance(payload, dict) else None
        return [item for item in output or [] if isinstance(item, dict)]
    for frame in text.split("\n\n"):
        if "_call" not in frame:
            continue
        for line in frame.split("\n"):
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[len("data: ") :])
            except ValueError:
                continue
            if not isinstance(data, dict):
                continue
            if isinstance(data.get("item"), dict):
                items.append(data["item"])
            response = data.get("response")
            if isinstance(response, dict):
                items.extend(
                    item for item in response.get("output") or [] if isinstance(item, dict)
                )
    return items


def find_tool_call_markup(text: Optional[str], payload: Any = None) -> List[Dict[str, Any]]:
    """Tool calls whose arguments still contain tool-call markup.

    Such a call was not cleanly separated from its format: the model wrote
    its own structure into a value (``"max_output_tokens</arg_key><arg_value>2000"``
    inside an exec header), or the parser split the markup wrongly. The call
    is delivered unchanged; this only lets a consumer of the trace find it.
    ``text`` is a streamed reply, ``payload`` a JSON one.
    """
    tokens = _tool_call_markup_tokens()
    if not tokens:
        return []
    if text is not None and not any(token in text for token in tokens):
        return []
    found: Dict[str, Dict[str, Any]] = {}
    for item in _responses_tool_call_items(text, payload):
        field_name = _TOOL_CALL_ITEM_TYPES.get(item.get("type"))
        value = item.get(field_name) if field_name else None
        if not isinstance(value, str):
            continue
        markers = [token for token in tokens if token in value]
        if markers:
            key = item.get("call_id") or item.get("id") or str(len(found))
            found[key] = {
                "call_id": item.get("call_id"),
                "name": item.get("name"),
                "markers": markers,
            }
    return list(found.values())


_RESPONSES_TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


def responses_outcome(text: Optional[str], payload: Any = None) -> Dict[str, Any]:
    """How a Responses reply ended, as the reply itself states it.

    A response record's ``status`` describes the transport: a stream that ran
    to its last event is ``completed`` whether the generation finished, ran
    out of token budget, or failed. The Responses object says which in its own
    ``status`` (completed / incomplete / failed), ``incomplete_details.reason``
    and ``error.code``; they are copied onto the record as ``response_status``,
    ``incomplete_reason`` and ``error_code`` so a census can tell a truncated
    generation from a finished one without parsing the body. Empty for a reply
    that is not a Responses object and for a stream cut before its terminal
    event. ``text`` is a streamed reply, ``payload`` a JSON one.
    """
    response = None
    if text is not None:
        # The terminal event is the last frame of a finished stream (the relay
        # can follow a cut with ``error`` then ``response.failed``), so only the
        # tail is read rather than the whole, possibly very long, stream.
        for frame in reversed(text.rstrip().rsplit("\n\n", 2)):
            event_type, data = None, None
            for line in frame.split("\n"):
                if line.startswith("event: "):
                    event_type = line[len("event: ") :].strip()
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[len("data: ") :])
                    except ValueError:
                        data = None
            if not isinstance(data, dict):
                continue
            if (event_type or data.get("type")) in _RESPONSES_TERMINAL_EVENTS and isinstance(
                data.get("response"), dict
            ):
                response = data["response"]
                break
    elif isinstance(payload, dict) and payload.get("object") == "response":
        response = payload
    if response is None:
        return {}
    outcome: Dict[str, Any] = {}
    if isinstance(response.get("status"), str):
        outcome["response_status"] = response["status"]
    details = response.get("incomplete_details")
    if isinstance(details, dict) and isinstance(details.get("reason"), str):
        outcome["incomplete_reason"] = details["reason"]
    error = response.get("error")
    if isinstance(error, dict) and error.get("code") is not None:
        outcome["error_code"] = error["code"]
    return outcome


def _join_frames(frames) -> str:
    """Decode the stream once, not once per transport read.

    ``iter_any()`` hands back whatever the socket delivered, and that split can
    fall inside a multi-byte character. Decoding each read on its own puts
    U+FFFD on both sides of the seam, and the result still parses as JSON, so
    the trace records mangled text without anything looking wrong.
    """
    parts: List[str] = []
    pending = bytearray()
    for frame in frames:
        if isinstance(frame, bytes):
            pending += frame
            continue
        if pending:
            parts.append(pending.decode("utf-8", "replace"))
            pending.clear()
        parts.append(_as_text(frame))
    if pending:
        parts.append(pending.decode("utf-8", "replace"))
    return "".join(parts)


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
    # Why a streaming response stopped, as the producer understood it. The
    # wrapper below can only see the exception that reached it, which says
    # "something failed" and not what; the producer knows whether generation
    # itself died, whether an upstream response was cut, or whether the fault
    # was in this process's own code -- the distinction that decides whether
    # the text lost with the stream still existed in memory when it was lost.
    # A body never iterated has no producer to ask; wrap_stream records
    # "not_started" for it instead.
    stream_termination: Optional[Dict[str, Any]] = None
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
        self.sanitized_records = 0
        self._write_error_count = 0
        # Durable progress of the batch currently being written, in lines.
        # _write_groups bumps it per group it finishes appending; _run reads it
        # on a write failure so a batch that lost only its tail groups does not
        # count the groups already on disk as dropped.
        self._lines_written_last_batch = 0
        # Stamped after every successful flush, in the same ISO form the records
        # carry. Nothing surfaces it yet; it exists so a monitor -- or a person
        # with a debugger -- can tell an idle writer from one whose writes have
        # stopped landing, a distinction ``enabled`` cannot make.
        self.last_write_at: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._task is not None

    async def start(self) -> None:
        if self._output_dir is None or self._task is not None:
            return
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.error(f"Disabling request trace output: {error}")
            self._output_dir = None
            return
        logger.info(f"Recording request traces to {self._output_dir}")
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

    def note_stream_termination(
        self,
        handle: Optional[RequestTraceHandle],
        cause: str,
        detail: str = "",
    ) -> None:
        """Record why a streaming producer stopped, for the response line.

        Synchronous and None-tolerant so a producer can call it from an
        exception handler without knowing whether tracing is on.
        """
        if handle is None:
            return
        handle.stream_termination = {"cause": cause, "detail": detail}

    def on_response(
        self,
        handle: Optional[RequestTraceHandle],
        *,
        frames: Optional[List[Any]] = None,
        payload: Any = None,
        status: str = "completed",
        http_status: Optional[int] = None,
    ) -> None:
        """Record the response side. Synchronous: safe to call from ``finally``.

        ``status`` is how the transport ended. For a Responses reply, how the
        generation ended is added from the reply itself (``responses_outcome``).

        ``payload`` is the body the client receives: a JSON value, or a ``str``
        for a plain-text answer (recorded with kind ``text``), such as the
        framework's own 500. ``http_status`` is the answer's HTTP status; the
        error exits pass it, since a 4xx and every 5xx would otherwise be
        indistinguishable in the record.
        """
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
        if http_status is not None:
            record["http_status"] = http_status
        if handle.stream_termination is not None:
            record["termination"] = handle.stream_termination
        text = _join_frames(frames) if frames is not None else None
        if text is not None:
            record["response"] = {"kind": "sse_text", "body": text}
        elif isinstance(payload, str):
            record["response"] = {"kind": "text", "body": payload}
        else:
            record["response"] = {"kind": "json", "body": payload}
        record.update(responses_outcome(text, payload))
        markup = find_tool_call_markup(text, payload)
        if markup:
            record["tool_call_markup"] = markup
            logger.warning(
                f"Trace {handle.trace_id}: {len(markup)} tool call(s) carry "
                f"tool-call markup in their arguments: {markup}"
            )
        self._submit(_hour_bucket(finished_at), _RESPONSES, record)

    def wrap_stream(
        self, stream: AsyncIterator[Any], handle: Optional[RequestTraceHandle]
    ) -> AsyncIterator[Any]:
        """Tee a streaming response into the trace without touching its producer.

        Wrapping the outermost generator is what makes one implementation cover
        every route: what gets recorded is the stream the client received, in
        the protocol the client speaks, whatever conversions happened upstream.

        A body that is never iterated runs none of the generator, its
        ``finally`` included. Starlette drops one that way when the client is
        already gone as the response begins (under ASGI 2.4 ``send`` raises and
        the response raises ClientDisconnect), which left an accepted request
        with no terminal record. A finalizer on the generator covers it: freed
        unstarted, the stream is recorded as ``client_disconnected`` with cause
        ``not_started``. Starlette releases a dropped body by reference
        counting, so this does not wait on the cyclic GC, which both servers
        disable; ``on_response`` keeps it to one record whichever side gets
        there first.
        """
        if handle is None or self._task is None:
            return stream
        started = False
        loop = _running_loop()

        async def _traced() -> AsyncIterator[Any]:
            nonlocal started
            started = True
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
            except asyncio.CancelledError:
                # The other half of the same event. Starlette watches for
                # http.disconnect and cancels the scope the body iterator runs
                # in, so a hangup arrives here as CancelledError whenever it is
                # delivered while this generator is suspended, and as
                # GeneratorExit only when the iterator is abandoned and closed
                # later. Without this branch that first form fell through to
                # the catch-all below and was recorded as "error", which is
                # where a census of truncated streams looks for server faults.
                # Server shutdown also cancels, and is recorded the same way;
                # it is rare and separately visible in the logs.
                status = "client_disconnected"
                raise
            except BaseException:
                status = "error"
                raise
            finally:
                self.on_response(handle, frames=frames, status=status)

        def _freed() -> None:
            if not started:
                self._close_unstarted_stream(handle, loop)

        traced = _traced()
        # Not at exit: a stream still alive then was not dropped unstarted.
        weakref.finalize(traced, _freed).atexit = False
        return traced

    def _close_unstarted_stream(
        self, handle: RequestTraceHandle, loop: Optional[asyncio.AbstractEventLoop]
    ) -> None:
        """Record a stream freed before its first iteration.

        Runs from the finalizer, in whichever thread dropped the last
        reference. That is the event loop's own in practice, and the record is
        made on the spot, ahead of anything queued after it; a cyclic
        collection on another thread hands it to the loop instead, because the
        queue is not thread-safe.
        """
        if loop is not None and _running_loop() is not loop:
            with contextlib.suppress(RuntimeError):  # loop closed: nothing to record into
                loop.call_soon_threadsafe(self._close_unstarted_stream, handle, None)
            return
        if handle.response_written:
            return
        if handle.stream_termination is None:
            handle.stream_termination = {
                "cause": "not_started",
                "detail": "the response body was never iterated",
            }
        self.on_response(handle, frames=[], status="client_disconnected")

    # -- writer --------------------------------------------------------------

    def _submit(self, bucket: str, kind: str, record: Dict[str, Any]) -> None:
        if self._task is None:
            return
        try:
            self._queue.put_nowait((bucket, kind, record))
        except asyncio.QueueFull:
            self.dropped_records += 1
            if self.dropped_records == 1 or self.dropped_records % 1000 == 0:
                logger.warning(f"Dropped {self.dropped_records} request trace records")

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
                        logger.warning(f"Dropped malformed request trace record: {error}")
                    continue
                # json.dumps checks JSON shape, not encodability: with
                # ensure_ascii=False an unpaired surrogate in client text rides
                # through it into a str no UTF-8 file can take, and the
                # explosion happens later, inside file.write() on the worker
                # thread. Caught here, that one record is written with \u
                # escapes instead -- pure ASCII, and JSON reads the escape back
                # as the code point the client sent, so nothing is lost and the
                # record's shape is unchanged. Its batchmates keep raw UTF-8.
                try:
                    line.encode("utf-8")
                except UnicodeEncodeError:
                    line = json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n"
                    self.sanitized_records += 1
                    if self.sanitized_records == 1 or self.sanitized_records % 1000 == 0:
                        logger.warning(
                            f"Escaped {self.sanitized_records} request trace records carrying "
                            f"text UTF-8 cannot encode (unpaired surrogates)"
                        )
                groups.setdefault((bucket, kind), []).append(line)
            if not groups:
                continue
            # _write_groups appends one file per group and can fail partway
            # through -- quota, a bad path -- with earlier groups already on
            # disk. Charge dropped_records only the lines that did not land: the
            # call records its durable progress on _lines_written_last_batch,
            # reset here so a prior batch's count cannot leak in should
            # to_thread never run the call, and the except paths subtract it.
            total_lines = sum(len(lines) for lines in groups.values())
            self._lines_written_last_batch = 0
            try:
                await asyncio.to_thread(self._write_groups, groups)
            except OSError as error:
                self.dropped_records += total_lines - self._lines_written_last_batch
                self._write_error_count += 1
                if self._write_error_count == 1 or self._write_error_count % 1000 == 0:
                    logger.warning(f"Failed to write request trace JSONL: {error}")
            except Exception as error:  # noqa: BLE001 - a dead writer loses every future trace
                # Deliberately broad, and the one place in this file it has to
                # be. An exception escaping here crashes nothing visible: it
                # kills this task while ``enabled`` stays True, so every later
                # record queues toward a drain that no longer runs and the
                # trace silently records nothing from that moment on.
                # UnicodeEncodeError -- a ValueError, not an OSError -- did
                # exactly that by slipping past the clause above, and finding
                # out cost a debugging session. Losing one batch to a surprise
                # is acceptable; losing the writer is not. CancelledError is a
                # BaseException and still passes, so ``close`` can cancel a
                # stuck task.
                self.dropped_records += total_lines - self._lines_written_last_batch
                self._write_error_count += 1
                if self._write_error_count == 1 or self._write_error_count % 1000 == 0:
                    logger.warning(
                        f"Failed to write request trace batch ({type(error).__name__}): {error}"
                    )
            else:
                self.last_write_at = _utc_now()

    def _write_groups(self, groups: Dict[Tuple[str, str], List[str]]) -> None:
        """Append each group to its file. Runs on a worker thread.

        At most one bucket per kind in practice, so a batch is two opens rather
        than the one-per-session it used to be.

        Records durable progress on ``_lines_written_last_batch`` as it goes:
        one group is one file append, so a group whose ``write`` returned has
        its bytes on disk (a torn tail from a crash is a later batch's problem,
        isolated by the half-line repair below) and its lines are counted; a
        failure on a later group therefore leaves the count at exactly what
        survived, and ``_run`` charges only the rest as dropped.
        """
        for (bucket, kind), lines in groups.items():
            directory = self._output_dir / bucket
            if bucket not in self._known_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                self._known_dirs.add(bucket)
            path = directory / f"{kind}{self._writer_suffix}.jsonl"
            # A write cut short (crash, quota) leaves the file ending in half a
            # line, and appending straight onto it welds the next record to the
            # fragment -- two records lost where one already was. One byte read
            # from the tail decides; a lone newline first isolates the fragment
            # as a single bad line, which JSONL readers already skip.
            needs_newline = False
            try:
                if path.stat().st_size > 0:
                    with path.open("rb") as tail:
                        tail.seek(-1, os.SEEK_END)
                        needs_newline = tail.read(1) != b"\n"
            except FileNotFoundError:
                pass  # First write to this file: nothing to repair.
            with path.open("a", encoding="utf-8") as output:
                if needs_newline:
                    output.write("\n")
                output.write("".join(lines))
            # This group's append returned, so its bytes are on disk; count them
            # before moving to the next group so a failure there charges only the
            # genuinely unwritten remainder to dropped_records.
            self._lines_written_last_batch += len(lines)


def _as_text(chunk: Any) -> str:
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", "replace")
    return chunk if isinstance(chunk, str) else str(chunk)
