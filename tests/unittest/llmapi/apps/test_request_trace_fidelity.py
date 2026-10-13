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
"""The trace fields that let a training export verify a record instead of trusting it.

What these pin down: the prompt-token digest is the same whichever form the
ids arrive in and is the digest an export can recompute; the identities a
reader needs (thread, campaign, gateway, build) are on the line, lifted from
what the line already holds; the response summary states what the terminal
object states; and none of it changes a record whose request carried none of
the inputs.
"""

import asyncio
import base64
import hashlib
import json
import struct
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.request_trace import (
    BUILD_ID_ENV,
    PROMPT_TOKEN_DIGEST_ENCODING,
    PROMPT_TOKEN_IDS_ENV,
    PROVENANCE_FILE_ENV,
    RequestTraceHandle,
    RequestTraceWriter,
    body_sha256,
    chat_finish_reasons,
    decode_prompt_token_ids_b64,
    gateway_identity,
    gateway_injected_terminal,
    kf_identity,
    prompt_token_digest,
    read_provenance,
    responses_outcome,
    responses_summary,
    sampling_as_sent,
    thread_identity,
)

pytestmark = pytest.mark.cpu_only


from tensorrt_llm.serve.request_trace import body_sha256, thread_identity  # noqa: E402


class FakeHeaders(dict):
    pass


class FakeState:
    pass


class FakeURL:
    def __init__(self, path):
        self.path = path


class FakeRequest:
    def __init__(self, body=None, headers=None, path="/v1/responses", arrival_time=1.5):
        self._body = body
        self.headers = FakeHeaders(headers or {})
        self.url = FakeURL(path)
        self.state = FakeState()
        self.state.server_arrival_time = arrival_time

    async def json(self):
        return self._body

    async def body(self):
        return b""


def read_lines(directory, kind):
    records = []
    for path in sorted(directory.glob(f"*/{kind}*.jsonl")):
        records += [json.loads(line) for line in path.read_text().splitlines() if line]
    return records


async def drain(writer):
    await writer.close()


def _sse(event_type, data):
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _int32le(ids):
    return struct.pack(f"<{len(ids)}i", *ids)


CODEX_BODY = {
    "model": "gpt-5.6-sol",
    "input": [{"type": "message", "role": "user", "content": "go"}],
    "parallel_tool_calls": False,
    "reasoning": {"context": "all_turns", "effort": "high"},
    "text": {"verbosity": "low"},
    "tool_choice": "auto",
    "stream": True,
    "store": False,
    "prompt_cache_key": "parent-session",
    "client_metadata": {
        "session_id": "parent-session",
        "thread_id": "child-thread",
        "turn_id": "turn-1",
        "root_turn_id": "turn-0",
        "x-codex-turn-metadata": json.dumps(
            {
                "agent_name": "explorer",
                "request_kind": "turn",
                "window_id": "parent-session:0",
                "window_number": 0,
                "parent_thread_id": "parent-session",
                "turn_started_at_unix_ms": 1791816831226,
            }
        ),
    },
}

KF_HEADERS = {
    "x-kf-campaign-id": "camp",
    "x-kf-agent-run-id": "camp-r001-a000",
    "x-kf-round": "1",
    "x-kf-agent-runtime": "codex",
    "x-kf-model": "kimi-k3",
    "x-kf-request-id": "req-9",
    "x-gateway-request-id": "000123",
    "x-gateway-attempt": "2",
    "x-api-key": "secret",
}


class TestPromptTokenDigest:
    def test_ids_and_b64_give_the_same_digest(self):
        ids = [163586, 1, 2, 3, 2**31 - 1, -1]
        from_ids = prompt_token_digest(ids=ids)
        from_b64 = prompt_token_digest(b64=base64.b64encode(_int32le(ids)).decode())
        assert from_ids == from_b64
        assert from_ids["count"] == len(ids)
        assert from_ids["encoding"] == PROMPT_TOKEN_DIGEST_ENCODING

    def test_the_digest_is_what_an_export_recomputes(self):
        ids = [10, 20, 30]
        digest = prompt_token_digest(ids=ids)
        assert digest["sha256"] == hashlib.sha256(_int32le(ids)).hexdigest()

    def test_nothing_in_nothing_out(self):
        assert prompt_token_digest() is None
        assert prompt_token_digest(ids=None, b64=None) is None
        assert prompt_token_digest(b64="not base64!!") is None

    def test_b64_round_trips_through_the_decoder(self):
        ids = [5, 6, 7]
        encoded = base64.b64encode(_int32le(ids)).decode()
        assert decode_prompt_token_ids_b64(encoded) == ids

    def test_a_context_retry_replaces_the_digest_and_an_engine_report_does_not(self):
        handle = RequestTraceHandle(trace_id="tr_1", session="s", route="/v1/responses")
        handle.set_prompt_tokens([1, 2, 3], source="ctx_response")
        handle.set_prompt_tokens([9, 9, 9], source="ctx_response")  # prefill re-run on another worker
        assert handle.prompt_tokens["sha256"] == prompt_token_digest(ids=[9, 9, 9])["sha256"]
        handle.set_prompt_tokens([4, 4], source="engine")
        assert handle.prompt_tokens["sha256"] == prompt_token_digest(ids=[9, 9, 9])["sha256"]
        assert handle.prompt_tokens["source"] == "ctx_response"
        assert "ids" not in handle.prompt_tokens

    def test_the_first_engine_report_is_kept(self):
        handle = RequestTraceHandle(trace_id="tr_1", session="s", route="/v1/chat/completions")
        handle.set_prompt_tokens([1, 2, 3], source="engine")
        handle.set_prompt_tokens([5], source="engine")
        assert handle.prompt_tokens["count"] == 3

    def test_odd_inputs_never_raise(self):
        import numpy as np

        handle = RequestTraceHandle(trace_id="tr_1", session="s", route="/v1/responses")
        handle.set_prompt_tokens(np.asarray([1, 2], dtype=np.int64), source="engine")
        assert handle.prompt_tokens["count"] == 2
        other = RequestTraceHandle(trace_id="tr_2", session="s", route="/v1/responses")
        other.set_prompt_tokens(b64="abc", source="ctx_response")  # not a whole number of int32s
        other.set_prompt_tokens(ids=None, b64=None)
        other.set_prompt_tokens([], source="engine")
        assert other.prompt_tokens is None or other.prompt_tokens["count"] == 0

    def test_ids_are_kept_only_when_asked(self):
        handle = RequestTraceHandle(
            trace_id="tr_1", session="s", route="/v1/responses", keep_prompt_token_ids=True
        )
        handle.set_prompt_tokens(b64=base64.b64encode(_int32le([4, 5])).decode())
        assert handle.prompt_tokens["ids"] == [4, 5]
        assert handle.prompt_tokens["count"] == 2


class TestIdentityLifting:
    def test_thread_from_codex_client_metadata(self):
        thread = thread_identity(CODEX_BODY)
        assert thread["thread_id"] == "child-thread"
        assert thread["session_id"] == "parent-session"
        assert thread["turn_id"] == "turn-1"
        assert thread["root_turn_id"] == "turn-0"
        # From the JSON-encoded turn metadata.
        assert thread["agent_name"] == "explorer"
        assert thread["request_kind"] == "turn"
        assert thread["parent_thread_id"] == "parent-session"
        assert thread["window_number"] == 0

    def test_a_probe_has_no_thread(self):
        assert thread_identity({"model": "m", "messages": []}) is None
        assert thread_identity("not json") is None
        assert thread_identity({"client_metadata": {}}) is None

    def test_kf_headers_lifted_without_their_prefix(self):
        kf = kf_identity(FakeHeaders(KF_HEADERS))
        assert kf == {
            "campaign_id": "camp",
            "agent_run_id": "camp-r001-a000",
            "round": "1",
            "agent_runtime": "codex",
            "model": "kimi-k3",
            "request_id": "req-9",
        }
        assert kf_identity(FakeHeaders({"host": "x"})) is None

    def test_gateway_identity(self):
        assert gateway_identity(FakeHeaders(KF_HEADERS)) == {"request_id": "000123", "attempt": 2}
        assert gateway_identity(FakeHeaders({"x-gateway-attempt": "two"})) == {"attempt": "two"}
        assert gateway_identity(FakeHeaders({})) is None

    def test_sampling_records_what_was_sent_and_nothing_else(self):
        sampling = sampling_as_sent(CODEX_BODY, "/v1/responses")
        # Codex sends no max_output_tokens: the key must be absent, not null.
        assert "max_output_tokens" not in sampling
        assert sampling["parallel_tool_calls"] is False
        assert sampling["reasoning_effort"] == "high"
        assert sampling["reasoning_context"] == "all_turns"
        assert sampling["verbosity"] == "low"
        assert sampling["tool_choice"] == "auto"
        assert sampling["stream"] is True

    def test_chat_sampling_keys(self):
        body = {"model": "m", "messages": [], "max_tokens": 1, "temperature": 0.0}
        assert sampling_as_sent(body, "/v1/chat/completions") == {
            "max_tokens": 1,
            "temperature": 0.0,
        }

    def test_body_sha_and_identity_tolerate_odd_bodies(self):
        assert body_sha256({"a": "\u00e9"}) == body_sha256({"a": "\u00e9"})
        lone = body_sha256({"a": "\ud800"})  # a lone surrogate: a digest or None, never an exception
        assert lone is None or len(lone) == 64
        assert thread_identity({"client_metadata": "a string"}) in (None, {})
        assert thread_identity({"client_metadata": None}) in (None, {})
        assert thread_identity(None) in (None, {})

    def test_body_sha_is_canonical(self):
        a = body_sha256({"b": 1, "a": [1, 2]})
        b = body_sha256({"a": [1, 2], "b": 1})
        assert a == b
        assert body_sha256({"a": object()}) is None


class TestRequestLine:
    @pytest.mark.asyncio
    async def test_identities_on_the_request_line(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        await writer.on_request(FakeRequest(body=CODEX_BODY, headers=KF_HEADERS))
        await drain(writer)

        (record,) = read_lines(tmp_path, "requests")
        # The router key is unchanged: it is the parent session, by design.
        assert record["session"] == "parent-session"
        assert record["thread"]["thread_id"] == "child-thread"
        assert record["kf"]["agent_run_id"] == "camp-r001-a000"
        assert record["gateway"] == {"request_id": "000123", "attempt": 2}
        assert "max_output_tokens" not in record["sampling"]
        assert record["body_sha256"] == body_sha256(CODEX_BODY)
        # The credential header stays redacted; the kf lift never copies it.
        assert ["x-api-key", "[redacted]"] in record["headers"]
        assert "api_key" not in record["kf"]
        assert "build_id" not in record

    @pytest.mark.asyncio
    async def test_a_probe_gets_no_thread_and_the_record_still_writes(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        body = {"model": "glm5.2", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
        await writer.on_request(FakeRequest(body=body, path="/v1/chat/completions"))
        await drain(writer)

        (record,) = read_lines(tmp_path, "requests")
        assert "thread" not in record
        assert "kf" not in record
        assert record["sampling"] == {"max_tokens": 1}

    @pytest.mark.asyncio
    async def test_a_rejected_request_is_lifted_too(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        await writer.on_rejected(FakeRequest(body=CODEX_BODY, headers=KF_HEADERS), [])
        await drain(writer)

        (record,) = read_lines(tmp_path, "requests")
        assert record["status"] == "rejected_400"
        assert record["thread"]["thread_id"] == "child-thread"
        assert record["kf"]["round"] == "1"


class TestBuildId:
    @pytest.mark.asyncio
    async def test_build_id_on_both_lines_and_provenance_in_the_sidecar(self, tmp_path, monkeypatch):
        provenance = tmp_path / "provenance.txt"
        provenance.write_text(
            "commit=abc\ndirty=no\npy_manifest_sha256=7c1e9a0b2f44deadbeef\nchat_template=none\n"
        )
        monkeypatch.setenv(BUILD_ID_ENV, "7c1e9a0b2f44")
        monkeypatch.setenv(PROVENANCE_FILE_ENV, str(provenance))
        writer = RequestTraceWriter(str(tmp_path / "trace"))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body=CODEX_BODY))
        writer.on_response(handle, payload={"object": "response", "id": "resp_1", "status": "completed"})
        await drain(writer)

        (request,) = read_lines(tmp_path / "trace", "requests")
        (response,) = read_lines(tmp_path / "trace", "responses")
        assert request["build_id"] == "7c1e9a0b2f44"
        assert response["build_id"] == "7c1e9a0b2f44"
        (sidecar,) = sorted((tmp_path / "trace" / "_writer").glob("*.json"))
        document = json.loads(sidecar.read_text())
        assert document["build_id"] == "7c1e9a0b2f44"
        assert document["provenance"]["commit"] == "abc"
        assert document["provenance"]["chat_template"] == "none"
        assert document["prompt_token_ids_kept"] is False

    def test_read_provenance_tolerates_junk(self, tmp_path):
        path = tmp_path / "p.txt"
        path.write_text("no equals sign here\n=novalue\nkey=value=with=equals\n")
        assert read_provenance(str(path)) == {"key": "value=with=equals"}
        assert read_provenance(str(tmp_path / "missing.txt")) is None
        assert read_provenance(None) is None

    @pytest.mark.asyncio
    async def test_unset_leaves_the_records_as_before(self, tmp_path, monkeypatch):
        monkeypatch.delenv(BUILD_ID_ENV, raising=False)
        monkeypatch.delenv(PROVENANCE_FILE_ENV, raising=False)
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}))
        writer.on_response(handle, payload={})
        await drain(writer)
        (request,) = read_lines(tmp_path, "requests")
        (response,) = read_lines(tmp_path, "responses")
        assert "build_id" not in request and "build_id" not in response


TERMINAL = {
    "object": "response",
    "id": "resp_abc",
    "status": "completed",
    "max_output_tokens": 32768,
    "usage": {
        "input_tokens": 90824,
        "input_tokens_details": {"cached_tokens": 90620},
        "output_tokens": 2989,
        "output_tokens_details": {"reasoning_tokens": 2581},
        "total_tokens": 93813,
    },
    "output": [
        {"type": "reasoning", "id": "rs_1"},
        {"type": "message", "id": "msg_1"},
        {"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_a", "name": "exec", "input": "x"},
        {"type": "function_call", "id": "fc_1", "call_id": "call_b", "name": "wait", "arguments": "{}"},
    ],
}


class TestResponseLine:
    @pytest.mark.asyncio
    async def test_prompt_tokens_and_replay_ride_the_handle_onto_the_record(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body=CODEX_BODY))
        handle.set_prompt_tokens([1, 2, 3], source="ctx_response")
        handle.set_input_replay({"namespace_restored": ["call_a"], "messages": 7})

        async def source():
            yield _sse("response.created", {"response": {**TERMINAL, "status": "in_progress"}})
            yield _sse("response.completed", {"response": TERMINAL})

        [chunk async for chunk in writer.wrap_stream(source(), handle)]
        await drain(writer)

        (record,) = read_lines(tmp_path, "responses")
        assert record["prompt_tokens"]["count"] == 3
        assert record["prompt_tokens"]["source"] == "ctx_response"
        assert record["input_replay"] == {"namespace_restored": ["call_a"], "messages": 7}
        # Lifted from the terminal event.
        assert record["response_status"] == "completed"
        assert record["response_id"] == "resp_abc"
        assert record["max_output_tokens"] == 32768
        assert record["usage"] == {
            "input_tokens": 90824,
            "output_tokens": 2989,
            "total_tokens": 93813,
            "cached_tokens": 90620,
            "reasoning_tokens": 2581,
        }
        assert record["output_call_ids"] == ["call_a", "call_b"]
        assert record["output_item_types"] == ["reasoning", "message", "custom_tool_call", "function_call"]
        assert "gateway_injected_terminal" not in record
        assert "finish_reasons" not in record

    def test_summary_is_empty_for_non_responses_replies(self):
        assert responses_summary(None, {"id": "chatcmpl-1", "object": "chat.completion", "choices": []}) == {}
        assert responses_summary(_sse("response.output_text.delta", {"delta": "x"})) == {}

    def test_summary_from_a_json_body(self):
        summary = responses_summary(None, TERMINAL)
        assert summary["response_id"] == "resp_abc"
        assert summary["output_call_ids"] == ["call_a", "call_b"]

    def test_outcome_is_unchanged(self):
        # The pre-existing contract: exactly the three outcome fields.
        text = _sse("response.completed", {"response": {"object": "response", "status": "completed"}})
        assert responses_outcome(text) == {"response_status": "completed"}

    def test_chat_finish_reasons(self):
        assert chat_finish_reasons(None, {"choices": [{"finish_reason": "stop"}, {"finish_reason": "length"}]}) == [
            "stop",
            "length",
        ]
        stream = (
            "data: " + json.dumps({"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": None}]}) + "\n\n"
            "data: " + json.dumps({"object": "chat.completion.chunk", "choices": [{"index": 0, "finish_reason": "length"}]}) + "\n\n"
            "data: [DONE]\n\n"
        )
        assert chat_finish_reasons(stream) == ["length"]
        assert chat_finish_reasons(None, {"object": "response"}) is None
        assert chat_finish_reasons(_sse("response.completed", {"response": {}})) is None

    def test_gateway_injected_terminal_is_recognized_by_its_marker(self):
        rotated = (
            "event: error\n"
            'data: {"type":"error","error":{"type":"overloaded_error","code":"backend_rotated",'
            '"source":"gateway","message":"backend rotated mid-stream"}}\n\n'
        )
        worker_error = _sse("error", {"type": "error", "error": {"type": "api_error", "message": "boom"}})
        assert gateway_injected_terminal(_sse("response.created", {"response": {}}) + rotated)
        assert not gateway_injected_terminal(_sse("response.created", {"response": {}}) + worker_error)
        assert not gateway_injected_terminal(None)

    @pytest.mark.asyncio
    async def test_a_record_without_prompt_facts_has_no_such_fields(self, tmp_path):
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body={}))
        writer.on_response(handle, payload={"detail": "nope"}, status="rejected_400", http_status=400)
        await drain(writer)
        (record,) = read_lines(tmp_path, "responses")
        assert "prompt_tokens" not in record
        assert "input_replay" not in record
        assert "response_id" not in record


class TestServerWiring:
    """The two frontends hand the handle what only they can know."""

    @pytest.mark.asyncio
    async def test_disagg_proxy_learns_the_prompt_from_the_context_worker(self, tmp_path):
        from tensorrt_llm.serve.openai_disagg_server import RawRequestResponseHooks

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body=CODEX_BODY)
        handle = await writer.on_request(request)
        hooks = RawRequestResponseHooks(
            request, SimpleNamespace(observe=lambda value: None), collect_perf_metrics=False
        )
        ids = [7, 8, 9, 10]
        ctx_response = SimpleNamespace(
            prompt_token_ids=None,
            prompt_token_ids_b64=base64.b64encode(_int32le(ids)).decode(),
            input_replay={"messages": 4, "namespace_restored": ["call_1"]},
        )
        hooks.on_ctx_resp("ctx-server", ctx_response)

        assert handle.prompt_tokens["sha256"] == prompt_token_digest(ids=ids)["sha256"]
        assert handle.prompt_tokens["source"] == "ctx_response"
        assert handle.input_replay == {"messages": 4, "namespace_restored": ["call_1"]}
        await drain(writer)

    @pytest.mark.asyncio
    async def test_disagg_proxy_without_a_handle_is_a_no_op(self):
        from tensorrt_llm.serve.openai_disagg_server import RawRequestResponseHooks

        request = FakeRequest(body=CODEX_BODY)  # tracing off: no handle on the state
        hooks = RawRequestResponseHooks(
            request, SimpleNamespace(observe=lambda value: None), collect_perf_metrics=False
        )
        hooks.on_ctx_resp("ctx-server", SimpleNamespace(prompt_token_ids=[1], prompt_token_ids_b64=None))

    @pytest.mark.asyncio
    async def test_aggregated_server_learns_the_prompt_from_the_promise(self, tmp_path):
        from tensorrt_llm.serve.openai_server import OpenAIServer

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body=CODEX_BODY)
        handle = await writer.on_request(request)
        promise = SimpleNamespace(request_id=42, prompt_token_ids=[1, 2, 3])
        model_request = SimpleNamespace(_input_replay={"messages": 2, "kimi_k3": True})

        OpenAIServer._trace_engine_ids(request, promise, model_request)

        assert handle.client_id == 42
        assert handle.prompt_tokens["count"] == 3
        assert handle.prompt_tokens["source"] == "engine"
        assert handle.input_replay == {"messages": 2, "kimi_k3": True}
        await drain(writer)

    @pytest.mark.asyncio
    async def test_aggregated_server_closes_the_trace_of_a_rejected_request(self, tmp_path):
        """A 400 from the handler leaves a response line, like the disagg proxy's."""
        from tensorrt_llm.serve.openai_server import OpenAIServer

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        request = FakeRequest(body=CODEX_BODY)
        handle = await writer.on_request(request)
        server = OpenAIServer.__new__(OpenAIServer)
        server._request_trace = writer

        answer = server._traced_error_response(
            handle, "The prompt length (10277.0) should not exceed max_num_tokens (128)")
        await drain(writer)

        assert answer.status_code == 400
        (record,) = read_lines(tmp_path, "responses")
        assert record["status"] == "rejected_400"
        assert record["http_status"] == 400
        assert "max_num_tokens" in record["response"]["body"]["message"]
        assert record["response"]["body"]["type"] == "BadRequestError"

    @pytest.mark.asyncio
    async def test_a_server_fault_answered_with_400_is_traced_as_an_error(self, tmp_path):
        from tensorrt_llm.serve.openai_server import OpenAIServer

        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body=CODEX_BODY))
        server = OpenAIServer.__new__(OpenAIServer)
        server._request_trace = writer

        answer = server._traced_error_response(handle, "KeyError: 'output'", trace_status="error")
        await drain(writer)

        assert answer.status_code == 400  # the client sees what it saw before
        (record,) = read_lines(tmp_path, "responses")
        assert record["status"] == "error"
        assert record["http_status"] == 400

    def test_a_traced_error_without_a_handle_still_answers(self):
        from tensorrt_llm.serve.openai_server import OpenAIServer

        server = OpenAIServer.__new__(OpenAIServer)
        server._request_trace = RequestTraceWriter(None)
        answer = server._traced_error_response(None, "boom")
        assert answer.status_code == 400

    def test_the_context_only_path_strips_the_prompt_ids_with_the_handoff(self):
        from tensorrt_llm.serve.openai_disagg_service import _drop_ctx_handoff
        from tensorrt_llm.serve.openai_protocol import ResponsesResponse

        response = ResponsesResponse(
            model="m", output=[], parallel_tool_calls=False, temperature=1.0, tool_choice="auto",
            tools=[], top_p=1.0, background=False, service_tier="auto", status="completed",
            top_logprobs=0, truncation="disabled",
            prompt_token_ids=[1, 2, 3], prompt_token_ids_b64="AQAAAA==",
            input_replay={"messages": 2}, finish_reason="stop",
        )
        _drop_ctx_handoff(response)
        dumped = json.loads(response.model_dump_json(by_alias=True))
        assert dumped["prompt_token_ids"] is None
        assert dumped["prompt_token_ids_b64"] is None
        assert dumped["input_replay"] is None
        assert response.finish_reason == "stop"

        chat_shaped = SimpleNamespace(
            choices=[SimpleNamespace(disaggregated_params="handoff", finish_reason="stop", prompt_token_ids=[1])],
            prompt_token_ids=[1, 2], prompt_token_ids_b64="AQAAAA==",
        )
        _drop_ctx_handoff(chat_shaped)
        assert chat_shaped.choices[0].disaggregated_params is None
        assert chat_shaped.choices[0].prompt_token_ids is None
        assert chat_shaped.prompt_token_ids is None and chat_shaped.prompt_token_ids_b64 is None

    def test_a_context_only_responses_reply_gets_its_prompt_ids_from_the_promise(self):
        """With postprocessing workers the reply arrives without ids; the promise has them."""
        from tensorrt_llm.serve.openai_server import OpenAIServer

        ctx_request = SimpleNamespace(disaggregated_params=SimpleNamespace(request_type="context_only"))
        promise = SimpleNamespace(prompt_token_ids=[7, 8, 9])
        reply = SimpleNamespace(prompt_token_ids=None)
        OpenAIServer._attach_ctx_prompt_ids(ctx_request, reply, promise)
        assert reply.prompt_token_ids == [7, 8, 9]

        already = SimpleNamespace(prompt_token_ids=[1])  # an in-process postprocessor set them
        OpenAIServer._attach_ctx_prompt_ids(ctx_request, already, promise)
        assert already.prompt_token_ids == [1]

        for plain in (SimpleNamespace(disaggregated_params=None),
                      SimpleNamespace(disaggregated_params=SimpleNamespace(request_type="generation_only"))):
            reply = SimpleNamespace(prompt_token_ids=None)
            OpenAIServer._attach_ctx_prompt_ids(plain, reply, promise)
            assert reply.prompt_token_ids is None  # never added to a client-facing reply

    @pytest.mark.asyncio
    async def test_env_switch_keeps_the_ids(self, tmp_path, monkeypatch):
        monkeypatch.setenv(PROMPT_TOKEN_IDS_ENV, "1")
        writer = RequestTraceWriter(str(tmp_path))
        await writer.start()
        handle = await writer.on_request(FakeRequest(body=CODEX_BODY))
        handle.set_prompt_tokens([3, 1, 4], source="engine")
        writer.on_response(handle, payload={})
        await drain(writer)
        (record,) = read_lines(tmp_path, "responses")
        assert record["prompt_tokens"]["ids"] == [3, 1, 4]
