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
"""Unit tests for the request trace writer."""

import json
from unittest.mock import AsyncMock

import pytest

from tensorrt_llm.serve.request_trace import (
    _WRITER_QUEUE_SIZE,
    REQUEST_TRACE_DIR_ENV,
    RequestTraceWriter,
    brief_validation_errors,
    is_internal_disagg_request,
    request_trace_dir_from_env,
    resolve_session_key,
    sanitize_session_key,
)

pytestmark = pytest.mark.cpu_only


class FakeHeaders(dict):
    """Minimal stand-in for starlette's case-insensitive header mapping."""


class FakeState:
    """Stand-in for starlette's request.state (a plain attribute bag)."""


class FakeURL:
    def __init__(self, path):
        self.path = path


class FakeRequest:
    """Enough of starlette's Request for the trace hooks."""

    def __init__(
        self,
        body=None,
        headers=None,
        raw_body=b"",
        json_error=None,
        path="/v1/messages",
        arrival_time=None,
    ):
        self._body = body
        self._raw_body = raw_body
        self._json_error = json_error
        self.headers = FakeHeaders(headers or {})
        self.url = FakeURL(path)
        self.state = FakeState()
        if arrival_time is not None:
            # What ServerArrivalTimeMiddleware puts there before any handler.
            self.state.server_arrival_time = arrival_time

    async def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._body

    async def body(self):
        return self._raw_body


def read_lines(directory, session, kind):
    path = directory / session / f"{kind}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


async def drain(writer):
    """Flush the queue by closing; the writer is single-use afterwards."""
    await writer.close()


class TestSanitizeSessionKey:
    @pytest.mark.parametrize("value", [None, "", "   "])
    def test_missing_falls_back(self, value):
        assert sanitize_session_key(value) == "_no_session"

    @pytest.mark.parametrize("value", ["..", ".", "./../etc", ".hidden"])
    def test_traversal_and_hidden_rejected(self, value):
        # A leading dot is the only way a sanitized leaf can still escape or
        # hide, so it is rejected outright rather than patched up.
        assert sanitize_session_key(value) == "_no_session"

    def test_separators_replaced(self):
        assert sanitize_session_key("a/b\\c") == "a_b_c"
        assert "/" not in sanitize_session_key("../../root")

    def test_null_byte_replaced(self):
        assert sanitize_session_key("sess\x00id") == "sess_id"

    def test_length_bounded(self):
        assert len(sanitize_session_key("x" * 500)) == 128

    def test_ordinary_value_preserved(self):
        assert sanitize_session_key("sess_ab12-cd.3") == "sess_ab12-cd.3"


class TestResolveSessionKey:
    def test_header_wins(self):
        headers = {"x-claude-code-session-id": "sess_hdr"}
        body = {"client_metadata": {"session_id": "sess_body"}}
        assert resolve_session_key(headers, body) == "sess_hdr"

    def test_header_priority_order(self):
        # x-claude-code-session-id precedes x-session-id in the shared tuple.
        headers = {"x-session-id": "generic", "x-claude-code-session-id": "claude"}
        assert resolve_session_key(headers, None) == "claude"

    def test_body_fallback(self):
        body = {"client_metadata": {"session_id": "sess_body"}}
        assert resolve_session_key({}, body) == "sess_body"

    @pytest.mark.parametrize("body", [None, {}, "not-a-dict", {"client_metadata": 7}])
    def test_no_session(self, body):
        assert resolve_session_key({}, body) == "_no_session"


class TestBriefValidationErrors:
    def test_keeps_only_safe_keys(self):
        errors = [
            {
                "loc": ("body", "client_metadata"),
                "type": "extra_forbidden",
                "msg": "Extra inputs are not permitted",
                "input": {"the": "whole body"},
                "handle": {"error": ValueError("live object")},
            }
        ]
        assert brief_validation_errors(errors) == [
            {
                "loc": ["body", "client_metadata"],
                "type": "extra_forbidden",
                "msg": "Extra inputs are not permitted",
            }
        ]

    def test_result_is_serializable(self):
        errors = [
            {"loc": ("body",), "type": "x", "msg": "y", "handle": {"error": ValueError("live")}}
        ]
        json.dumps(brief_validation_errors(errors))

    @pytest.mark.parametrize("errors", [[], None, ["not-a-mapping"]])
    def test_tolerates_junk(self, errors):
        assert brief_validation_errors(errors) == []


class TestIsInternalDisaggRequest:
    """The predicate that keeps orchestrator hops out of a worker's trace."""

    @pytest.mark.parametrize("request_type", ["context_only", "generation_only"])
    def test_orchestrator_hops(self, request_type):
        body = {"messages": [], "disaggregated_params": {"request_type": request_type}}
        assert is_internal_disagg_request(body) is True

    def test_context_and_generation_is_client_traffic(self):
        """The third legal value rides on requests one server answers end to end.

        The gRPC frontend falls back to it when the proto leaves request_type
        unset, and EPD multimodal sets it on the prefill+decode half. A check
        for the presence of disaggregated_params rather than for these two
        values would drop both.
        """
        body = {"disaggregated_params": {"request_type": "context_and_generation"}}
        assert is_internal_disagg_request(body) is False

    @pytest.mark.parametrize(
        "body",
        [
            {"messages": []},
            {"disaggregated_params": None},
            {"disaggregated_params": {}},
            {"disaggregated_params": {"request_type": None}},
            {"disaggregated_params": {"ctx_request_id": 7}},
            {"disaggregated_params": "context_only"},
            None,
            "not a mapping",
            [],
        ],
    )
    def test_anything_else_reads_as_client_traffic(self, body):
        assert is_internal_disagg_request(body) is False

    def test_unknown_request_type_fails_open(self):
        """A hop type this predicate has not heard of is recorded, not dropped."""
        body = {"disaggregated_params": {"request_type": "encoder_only"}}
        assert is_internal_disagg_request(body) is False


class TestInternalDisaggRequestsAreNotTraced:
    """A worker behind the orchestrator writes nothing, on either hook."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("request_type", ["context_only", "generation_only"])
    async def test_on_request_records_nothing(self, tmp_path, request_type):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(
            body={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "prompt_token_ids": [1, 2, 3],
                "disaggregated_params": {"request_type": request_type},
            },
            headers={"x-session-id": "sess_a"},
            path="/v1/chat/completions",
        )
        handle = await writer.on_request(request)
        await drain(writer)

        assert handle is None
        # No handle was stamped, so a nested handler cannot mistake this for a
        # request someone else already owns.
        assert getattr(request.state, "request_trace_handle", None) is None
        assert read_lines(tmp_path, "sess_a", "requests") == []
        assert read_lines(tmp_path, "_no_session", "requests") == []

    @pytest.mark.asyncio
    async def test_response_hooks_are_inert_for_the_dropped_handle(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(
            body={"disaggregated_params": {"request_type": "generation_only"}},
            path="/v1/chat/completions",
        )
        handle = await writer.on_request(request)

        async def upstream():
            yield "data: {}\n\n"

        stream = writer.wrap_stream(upstream(), handle)
        assert [chunk async for chunk in stream] == ["data: {}\n\n"]
        writer.on_response(handle, payload={"id": "x"}, status="completed")
        await drain(writer)

        assert read_lines(tmp_path, "_no_session", "responses") == []

    @pytest.mark.asyncio
    async def test_on_rejected_records_nothing(self, tmp_path):
        """Deliberate: a worker creates no trace files at all, not even rare ones."""
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        await writer.on_rejected(
            FakeRequest(
                body={"disaggregated_params": {"request_type": "context_only"}},
                path="/v1/chat/completions",
            ),
            [{"loc": ("body", "model"), "type": "missing", "msg": "Field required"}],
        )
        await drain(writer)

        assert read_lines(tmp_path, "_no_session", "requests") == []

    @pytest.mark.asyncio
    async def test_the_proxys_own_client_request_is_still_traced(self, tmp_path):
        """The same writer on the orchestrator keeps recording: no such field."""
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(
            FakeRequest(
                body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-session-id": "sess_a"},
            )
        )
        await drain(writer)

        assert handle is not None
        (record,) = read_lines(tmp_path, "sess_a", "requests")
        assert record["body"]["messages"][0]["content"] == "hi"


class TestDisabledWriter:
    @pytest.mark.asyncio
    async def test_all_hooks_are_inert(self):
        writer = RequestTraceWriter(None)
        await writer.start()
        assert not writer.enabled
        assert await writer.on_request(FakeRequest(body={})) is None
        await writer.on_rejected(FakeRequest(body={}), [])
        writer.on_response(None)
        await writer.close()

    @pytest.mark.asyncio
    async def test_wrap_stream_returns_original(self):
        writer = RequestTraceWriter(None)
        await writer.start()

        async def source():
            yield "a"

        stream = source()
        assert writer.wrap_stream(stream, None) is stream
        await stream.aclose()

    def test_env_helper(self, monkeypatch):
        monkeypatch.delenv(REQUEST_TRACE_DIR_ENV, raising=False)
        assert request_trace_dir_from_env() is None
        monkeypatch.setenv(REQUEST_TRACE_DIR_ENV, "")
        assert request_trace_dir_from_env() is None
        monkeypatch.setenv(REQUEST_TRACE_DIR_ENV, "/tmp/traces")
        assert request_trace_dir_from_env() == "/tmp/traces"


class TestRequestRecords:
    @pytest.mark.asyncio
    async def test_route_derived_from_url(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, path="/v1/responses"))
        await drain(writer)

        assert handle.route == "/v1/responses"
        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["route"] == "/v1/responses"

    @pytest.mark.asyncio
    async def test_accepted_request(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(
            body={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-session-id": "sess_a", "authorization": "Bearer secret"},
        )
        handle = await writer.on_request(request)
        await drain(writer)

        assert handle.session == "sess_a"
        assert handle.trace_id.startswith("tr_")
        assert handle.route == "/v1/messages"
        # Part 3 reaches the context through the request, not through an
        # argument threaded down the handler.
        assert request.state.request_trace_handle is handle
        (record,) = read_lines(tmp_path, "sess_a", "requests")
        assert record["event"] == "request"
        assert record["status"] == "accepted"
        assert record["recorded_at"].endswith("+00:00")
        assert record["route"] == "/v1/messages"
        assert record["trace_id"] == handle.trace_id
        assert record["body"]["messages"][0]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_arrival_time_recorded(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        await writer.on_request(FakeRequest(body={}, arrival_time=12345.5))
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["server_arrival_time"] == 12345.5

    @pytest.mark.asyncio
    async def test_arrival_time_absent_is_null(self, tmp_path):
        # The middleware is installed unconditionally, but the field must not
        # be load-bearing: a request that reaches the hook without one still
        # produces a record.
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        await writer.on_request(FakeRequest(body={}))
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["server_arrival_time"] is None

    @pytest.mark.asyncio
    async def test_repeated_headers_all_kept(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()

        class MultiHeaders(FakeHeaders):
            # HTTP allows repeats and starlette's Headers is a multidict; a
            # dict-shaped dump would silently keep only the last hop.
            def items(self):
                return [
                    ("x-forwarded-for", "10.0.0.1"),
                    ("x-forwarded-for", "10.0.0.2"),
                    ("user-agent", "claude-cli/2.1"),
                ]

        request = FakeRequest(body={})
        request.headers = MultiHeaders()
        await writer.on_request(request)
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["headers"] == [
            ["x-forwarded-for", "10.0.0.1"],
            ["x-forwarded-for", "10.0.0.2"],
            ["user-agent", "claude-cli/2.1"],
        ]

    @pytest.mark.asyncio
    async def test_unparseable_body_kept_as_text(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(
            raw_body=b'{"model": "m",',
            json_error=ValueError("Expecting value"),
        )
        await writer.on_rejected(
            request, [{"loc": ("body",), "type": "json_invalid", "msg": "bad"}]
        )
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["status"] == "rejected_400"
        assert "server_arrival_time" in record
        assert record["body"] == '{"model": "m",'
        assert "ValueError" in record["body_parse_error"]
        assert record["validation_errors"][0]["type"] == "json_invalid"

    @pytest.mark.asyncio
    async def test_rejected_request_has_no_response_line(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        await writer.on_rejected(FakeRequest(body={"a": 1}), [])
        await drain(writer)

        assert len(read_lines(tmp_path, "_no_session", "requests")) == 1
        assert read_lines(tmp_path, "_no_session", "responses") == []


class TestStreamingResponse:
    @pytest.mark.asyncio
    async def test_frames_recorded_verbatim(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s1"}))
        handle.set_ids(client_id=184)

        async def source():
            yield "event: message_start\ndata: {}\n\n"
            yield "event: message_stop\ndata: {}\n\n"

        seen = [chunk async for chunk in writer.wrap_stream(source(), handle)]
        await drain(writer)

        assert seen == [
            "event: message_start\ndata: {}\n\n",
            "event: message_stop\ndata: {}\n\n",
        ]
        (record,) = read_lines(tmp_path, "s1", "responses")
        assert record["status"] == "completed"
        assert record["client_id"] == 184
        assert record["response"]["kind"] == "sse_frames"
        assert record["response"]["frames"] == seen

    @pytest.mark.asyncio
    async def test_client_disconnect_keeps_partial_frames(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s2"}))

        async def source():
            yield "frame-0"
            yield "frame-1"
            yield "frame-2"

        stream = writer.wrap_stream(source(), handle)
        assert await stream.__anext__() == "frame-0"
        assert await stream.__anext__() == "frame-1"
        # What starlette does when the client hangs up mid-stream.
        await stream.aclose()
        await drain(writer)

        (record,) = read_lines(tmp_path, "s2", "responses")
        assert record["status"] == "client_disconnected"
        assert record["response"]["frames"] == ["frame-0", "frame-1"]

    @pytest.mark.asyncio
    async def test_upstream_error_is_recorded_and_reraised(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s3"}))

        async def source():
            yield "frame-0"
            raise RuntimeError("engine died")

        with pytest.raises(RuntimeError, match="engine died"):
            async for _ in writer.wrap_stream(source(), handle):
                pass
        await drain(writer)

        (record,) = read_lines(tmp_path, "s3", "responses")
        assert record["status"] == "error"
        assert record["response"]["frames"] == ["frame-0"]

    @pytest.mark.asyncio
    async def test_bytes_frames_coerced(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}))

        async def source():
            yield b"raw-bytes"

        async for _ in writer.wrap_stream(source(), handle):
            pass
        await drain(writer)

        (record,) = read_lines(tmp_path, "_no_session", "responses")
        assert record["response"]["frames"] == ["raw-bytes"]


class TestNonStreamingResponse:
    @pytest.mark.asyncio
    async def test_json_payload(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s4"}))
        handle.set_ids(disagg_request_id=99, ctx_request_id=7)
        writer.on_response(handle, payload={"id": "msg_1", "content": []})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s4", "responses")
        assert record["response"] == {"kind": "json", "body": {"id": "msg_1", "content": []}}
        assert record["disagg_request_id"] == 99
        assert record["ctx_request_id"] == 7
        assert record["client_id"] is None

    @pytest.mark.asyncio
    async def test_written_at_most_once(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}))
        writer.on_response(handle, payload={"first": True})
        writer.on_response(handle, payload={"second": True})
        await drain(writer)

        records = read_lines(tmp_path, "_no_session", "responses")
        assert len(records) == 1
        assert records[0]["response"]["body"] == {"first": True}


class TestWriterMechanics:
    @pytest.mark.asyncio
    async def test_sessions_land_in_separate_directories(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        for session in ("s_a", "s_b", "s_a"):
            await writer.on_request(FakeRequest(body={}, headers={"x-session-id": session}))
        await drain(writer)

        assert len(read_lines(tmp_path, "s_a", "requests")) == 2
        assert len(read_lines(tmp_path, "s_b", "requests")) == 1

    @pytest.mark.asyncio
    async def test_writer_suffix_applied(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path), writer_suffix="-1234")
        await writer.start()
        await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s"}))
        await drain(writer)

        assert (tmp_path / "s" / "requests-1234.jsonl").exists()

    @pytest.mark.asyncio
    async def test_unserializable_record_dropped_not_raised(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}))
        # object() has no JSON form; the batch must survive it.
        writer.on_response(handle, payload=object())
        await drain(writer)

        assert read_lines(tmp_path, "_no_session", "responses") == []
        assert writer.dropped_records == 1

    @pytest.mark.asyncio
    async def test_unwritable_directory_disables_quietly(self, tmp_path):
        blocker = tmp_path / "traces"
        blocker.write_text("not a directory")
        writer = RequestTraceWriter(str(blocker))
        await writer.start()
        assert not writer.enabled
        assert await writer.on_request(FakeRequest(body={})) is None
        await writer.close()

    @pytest.mark.asyncio
    async def test_full_queue_drops_without_raising(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        # Stall the drain so put_nowait hits the bound, then overfill it.
        # Sized off the constant: a hardcoded count silently stops testing the
        # drop path the moment the queue is resized.
        writer._task.cancel()
        for _ in range(_WRITER_QUEUE_SIZE + 10):
            writer._submit("s", "requests", {"ok": True})
        assert writer.dropped_records == 10
        writer._task = None


class TestNestedHandlerOwnership:
    """Only the handler the client called may record the response.

    /v1/messages converts and then calls the chat handler with the same Request.
    Both would wrap their own generator, and the inner one is drained by the
    outer, so it reaches its ``finally`` first and claims the record -- storing
    the OpenAI frames the Anthropic route exists to convert away. Handing the
    inner handler None is what stops that, and ``response_written`` cannot: it
    enforces exactly-once, not which one wins.
    """

    @pytest.mark.asyncio
    async def test_reentry_gets_no_handle(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"})

        outer = await writer.on_request(request)
        inner = await writer.on_request(request)
        await drain(writer)

        assert outer is not None
        assert inner is None
        assert len(read_lines(tmp_path, "s", "requests")) == 1

    @pytest.mark.asyncio
    async def test_inner_wrap_is_inert_and_outer_keeps_client_frames(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/messages")
        outer = await writer.on_request(request)  # anthropic_messages
        inner = await writer.on_request(request)  # openai_chat, nested

        async def openai_frames():
            yield "openai-frame"

        async def reframed(source):
            async for _ in source:
                yield "anthropic-frame"

        # Both layers wrap, exactly as the two handlers do.
        teed_inner = writer.wrap_stream(openai_frames(), inner)
        teed_outer = writer.wrap_stream(reframed(teed_inner), outer)
        assert [chunk async for chunk in teed_outer] == ["anthropic-frame"]
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["response"]["frames"] == ["anthropic-frame"]

    @pytest.mark.asyncio
    async def test_engine_ids_reach_the_handle_from_the_request(self, tmp_path):
        # The chat handler holds the promise even as the inner half of an
        # Anthropic turn -- precisely when it has no handle of its own -- so the
        # id has to be set through the request.
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"})
        outer = await writer.on_request(request)
        assert await writer.on_request(request) is None

        request.state.request_trace_handle.set_ids(client_id=184)
        writer.on_response(outer, payload={})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["client_id"] == 184


class TestNonStreamingOwnership:
    """The nested handler must stay silent on the non-streaming path too."""

    @pytest.mark.asyncio
    async def test_inner_handler_records_nothing(self, tmp_path):
        # openai_chat reached via /v1/messages returns its own JSON response and
        # calls on_response with the handle it was given -- which is None.
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/messages")
        outer = await writer.on_request(request)
        inner = await writer.on_request(request)

        writer.on_response(inner, payload={"openai": "shape"})
        writer.on_response(outer, payload={"anthropic": "shape"})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["response"]["body"] == {"anthropic": "shape"}

    @pytest.mark.asyncio
    async def test_conversion_failure_is_recorded_with_the_upstream_body(self, tmp_path):
        # convert_chat_response raises on a tool call whose arguments are not a
        # JSON object; the client only sees a 500, so the trace is the only
        # place that sample survives.
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}, headers={"x-session-id": "s"}))

        writer.on_response(
            handle,
            payload={"upstream_body": '{"choices":[{"message":{"tool_calls":[]}}]}'},
            status="conversion_error",
        )
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["status"] == "conversion_error"
        assert "tool_calls" in record["response"]["body"]["upstream_body"]


class TestAnthropicRouteEndToEnd:
    """The /v1/messages chain with tracing actually on.

    ``anthropic_messages`` converts, hands the request to ``openai_chat``, takes
    the ``body_iterator`` off the StreamingResponse that comes back, reframes it
    and returns a *new* StreamingResponse. Two response objects for one client
    request, and only the second one's frames are what the client reads.
    """

    @staticmethod
    def _client(tmp_path, openai_response):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tensorrt_llm.serve.openai_server import OpenAIServer

        app = FastAPI()
        server = object.__new__(OpenAIServer)
        server.model = "test-model"
        server._request_trace = RequestTraceWriter(str(tmp_path))
        server.openai_chat = AsyncMock(return_value=openai_response)
        app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
        return TestClient(app), server

    @staticmethod
    def _openai_stream():
        from starlette.responses import StreamingResponse

        async def source():
            yield (
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
                '"created":0,"model":"test-model","choices":[{"index":0,'
                '"delta":{"role":"assistant"}}]}\n\n'
            )
            yield (
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
                '"created":0,"model":"test-model","choices":[{"index":0,'
                '"delta":{"content":"hello"}}]}\n\n'
            )
            yield (
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
                '"created":0,"model":"test-model","choices":[{"index":0,'
                '"delta":{},"finish_reason":"stop"}]}\n\n'
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(source(), media_type="text/event-stream")

    @pytest.mark.asyncio
    async def test_streaming_client_output_is_unchanged_and_traced(self, tmp_path):
        client, server = self._client(tmp_path, self._openai_stream())
        await server._request_trace.start()

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-session-id": "sess_e2e"},
        )
        await server._request_trace.close()

        # The client still gets a correct Anthropic event stream.
        assert response.status_code == 200
        assert response.text.startswith("event: message_start\n")
        assert '"text":"hello"' in response.text
        assert response.text.rstrip().endswith('event: message_stop\ndata: {"type":"message_stop"}')

        # And the trace holds exactly those frames -- not the OpenAI ones the
        # inner handler produced.
        (record,) = read_lines(tmp_path, "sess_e2e", "responses")
        assert record["status"] == "completed"
        assert "".join(record["response"]["frames"]) == response.text
        assert not any("chatcmpl-" in frame for frame in record["response"]["frames"])

        (request_record,) = read_lines(tmp_path, "sess_e2e", "requests")
        assert request_record["route"] == "/v1/messages"
        assert request_record["body"]["messages"][0]["content"] == "hi"
        assert request_record["trace_id"] == record["trace_id"]

    @pytest.mark.asyncio
    async def test_disabled_writer_leaves_the_stream_alone(self, tmp_path):
        client, server = self._client(tmp_path, self._openai_stream())
        server._request_trace = RequestTraceWriter(None)
        await server._request_trace.start()

        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 16,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert response.status_code == 200
        assert '"text":"hello"' in response.text
        assert not list(tmp_path.iterdir())


class TestRejectedRequestEndToEnd:
    """A body FastAPI refuses never reaches a handler; the trace is its only copy."""

    @staticmethod
    def _client(tmp_path):
        from fastapi import FastAPI
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        writer = RequestTraceWriter(str(tmp_path))
        app = FastAPI()

        @app.exception_handler(RequestValidationError)
        async def handler(request, exc):
            # The same call openai_server makes at the top of its own handler.
            await writer.on_rejected(request, exc.errors())
            return JSONResponse(status_code=400, content={"error": str(exc)})

        @app.post("/v1/chat/completions")
        async def chat(request: ChatCompletionRequest):
            return {"ok": True}

        return TestClient(app), writer

    @pytest.mark.asyncio
    async def test_unknown_field_is_recorded_with_the_body(self, tmp_path):
        # extra="forbid" on ChatCompletionRequest: an agent client that starts
        # sending a new field gets every turn rejected, and this is where the
        # field name and the body it came in are visible.
        client, writer = self._client(tmp_path)
        await writer.start()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "brand_new_field": {"nested": 1},
            },
            headers={"x-session-id": "sess_400"},
        )
        await writer.close()

        assert response.status_code == 400
        (record,) = read_lines(tmp_path, "sess_400", "requests")
        assert record["status"] == "rejected_400"
        assert record["route"] == "/v1/chat/completions"
        assert record["body"]["brand_new_field"] == {"nested": 1}
        assert record["validation_errors"][0]["loc"][-1] == "brand_new_field"
        assert record["validation_errors"][0]["type"] == "extra_forbidden"
        # No response line: the request never produced one.
        assert read_lines(tmp_path, "sess_400", "responses") == []

    @pytest.mark.asyncio
    async def test_malformed_json_body_is_kept_as_text(self, tmp_path):
        client, writer = self._client(tmp_path)
        await writer.start()

        response = client.post(
            "/v1/chat/completions",
            content=b'{"model": "m", "messages": [',
            headers={"content-type": "application/json"},
        )
        await writer.close()

        assert response.status_code == 400
        (record,) = read_lines(tmp_path, "_no_session", "requests")
        assert record["status"] == "rejected_400"
        assert record["body"] == '{"model": "m", "messages": ['
        assert record["body_parse_error"]


class TestDisaggIdHandoff:
    """``disagg_request_id`` reaches the handle across the handler split.

    On /v1/messages the id is minted inside the wrapped chat entry point, which
    by the ownership rule holds no handle, while the handle that gets written
    belongs to the outer Anthropic handler. The two are locals of different
    functions; the Request they share is the only thing both can see.
    """

    @staticmethod
    def _hooks(request, disagg_request_id):
        from types import SimpleNamespace

        return SimpleNamespace(raw_req=request, disagg_request_id=disagg_request_id)

    @pytest.mark.asyncio
    async def test_id_set_from_the_half_without_a_handle(self, tmp_path):
        from tensorrt_llm.serve.openai_disagg_server import _set_disagg_ids

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/messages")

        outer = await writer.on_request(request)  # anthropic_messages
        inner = await writer.on_request(request)  # the wrapped entry point
        assert inner is None  # the half that has hooks

        # What the wrapper does: it holds no handle, only hooks.
        _set_disagg_ids(self._hooks(request, 8812345))

        writer.on_response(outer, payload={"ok": True})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["disagg_request_id"] == 8812345
        assert record["client_id"] is None  # no engine on a proxy

    @pytest.mark.asyncio
    async def test_id_set_when_the_wrapper_owns_the_request(self, tmp_path):
        # /v1/completions and /v1/chat/completions route straight to the
        # wrapper, so it is the creator and the same code path applies.
        from tensorrt_llm.serve.openai_disagg_server import _set_disagg_ids

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/chat/completions")
        handle = await writer.on_request(request)

        _set_disagg_ids(self._hooks(request, 991))

        writer.on_response(handle, payload={"ok": True})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["disagg_request_id"] == 991

    def test_no_handle_is_a_no_op(self):
        # Tracing off: the request carries nothing and this must not raise.
        from tensorrt_llm.serve.openai_disagg_server import _set_disagg_ids

        _set_disagg_ids(self._hooks(FakeRequest(body={}), 7))

    @pytest.mark.asyncio
    async def test_none_id_leaves_the_field_null(self, tmp_path):
        # A deployment that never allocates one (or a failure before it does).
        from tensorrt_llm.serve.openai_disagg_server import _set_disagg_ids

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body={}, headers={"x-session-id": "s"})
        handle = await writer.on_request(request)

        _set_disagg_ids(self._hooks(request, None))

        writer.on_response(handle, payload={})
        await drain(writer)

        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["disagg_request_id"] is None


class TestDisaggNonStreamingShape:
    """Same wrapper, two routes, two recorded shapes.

    On the disaggregated server ``_wrap_entry_point`` serves /v1/completions and
    /v1/chat/completions directly and is *also* called by anthropic_messages.
    It only ever holds the OpenAI-shaped response, so on the Anthropic route its
    own on_response is a no-op (handle is None) and the record has to be written
    after convert_chat_response, from the handler the client actually called.
    """

    OPENAI_BODY = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }

    @staticmethod
    def _server(tmp_path, openai_response):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer

        server = object.__new__(OpenAIDisaggServer)
        server._request_trace = RequestTraceWriter(str(tmp_path))
        server._service = SimpleNamespace(openai_chat_completion=object())
        server._wrap_entry_point = Mock(return_value=AsyncMock(return_value=openai_response))
        return server

    @staticmethod
    def _anthropic_request():
        from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest

        return AnthropicMessagesRequest(
            model="test-model", max_tokens=16, messages=[{"role": "user", "content": "hi"}]
        )

    @pytest.mark.asyncio
    async def test_anthropic_route_records_the_anthropic_shape(self, tmp_path):
        from fastapi.responses import JSONResponse

        server = self._server(tmp_path, JSONResponse(content=self.OPENAI_BODY))
        await server._request_trace.start()
        raw = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/messages")

        response = await server.anthropic_messages(self._anthropic_request(), raw)
        await server._request_trace.close()

        assert response.status_code == 200
        (record,) = read_lines(tmp_path, "s", "responses")
        body = record["response"]["body"]
        # Anthropic shape: content blocks and stop_reason...
        assert body["content"] == [{"type": "text", "text": "hello"}]
        assert body["stop_reason"] == "end_turn"
        # ...not the OpenAI one the wrapper handed back.
        assert "choices" not in body
        assert body["id"] != "chatcmpl-1"

    @pytest.mark.asyncio
    async def test_conversion_failure_keeps_the_upstream_body(self, tmp_path):
        from fastapi.responses import JSONResponse

        # convert_chat_response raises when a tool call's arguments are not a
        # JSON object; the client only gets a 500, so this is the only copy.
        broken = {
            **self.OPENAI_BODY,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "not json at all"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        server = self._server(tmp_path, JSONResponse(content=broken))
        await server._request_trace.start()
        raw = FakeRequest(body={}, headers={"x-session-id": "s"}, path="/v1/messages")

        response = await server.anthropic_messages(self._anthropic_request(), raw)
        await server._request_trace.close()

        assert response.status_code == 500
        (record,) = read_lines(tmp_path, "s", "responses")
        assert record["status"] == "conversion_error"
        assert "not json at all" in record["response"]["body"]["upstream_body"]

    @pytest.mark.asyncio
    async def test_chat_route_records_the_openai_shape(self, tmp_path):
        # The other half of the contrast: the same wrapper, reached directly, is
        # the outermost handler and so records what *its* client receives.
        from types import SimpleNamespace

        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ChatCompletionResponse

        server = self._server(tmp_path, None)
        await server._request_trace.start()
        server._perf_metrics_collector = SimpleNamespace(
            total_requests=SimpleNamespace(inc=lambda: None),
            stream_requests=SimpleNamespace(inc=lambda: None),
            nonstream_requests=SimpleNamespace(inc=lambda: None),
            total_responses=SimpleNamespace(inc=lambda: None),
            queue_latency_seconds=SimpleNamespace(observe=lambda _v: None),
        )
        server._collect_perf_metrics = False
        server._allow_request_chat_template = True
        server._extract_conversation_id = lambda req, raw: None

        chat_response = ChatCompletionResponse(**self.OPENAI_BODY)
        entry_point = AsyncMock(return_value=chat_response)
        # The real factory this time, not the Mock the other tests install.
        from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer

        wrapper = OpenAIDisaggServer._wrap_entry_point(server, entry_point, ChatCompletionRequest)

        raw = FakeRequest(
            body={}, headers={"x-session-id": "s"}, path="/v1/chat/completions", arrival_time=1.0
        )
        req = ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "hi"}]
        )
        await wrapper(req, raw)
        await server._request_trace.close()

        (record,) = read_lines(tmp_path, "s", "responses")
        body = record["response"]["body"]
        assert body["id"] == "chatcmpl-1"
        assert body["choices"][0]["message"]["content"] == "hello"
        assert "content" not in body or not isinstance(body.get("content"), list)
