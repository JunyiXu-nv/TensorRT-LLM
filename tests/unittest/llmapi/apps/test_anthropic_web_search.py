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
"""Offline unit tests for the Anthropic ``web_search`` server tool.

No GPU, engine or network required: the search backend's HTTP layer is
stubbed and the model is replaced by a scripted fake, so these exercise the
result parsing, the domain filters, and the server-side search loop.
"""

import asyncio
import json

import pytest

import tensorrt_llm.serve.anthropic_web_search as web_search_mod
from tensorrt_llm.serve.anthropic_adapter import (
    WEB_SEARCH_TOOL_NAME,
    AnthropicRequestError,
    convert_anthropic_request,
    resolve_web_search,
    run_web_search_turns,
    synthesize_anthropic_stream,
)
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.anthropic_web_search import (
    WebSearchConfig,
    WebSearchError,
    WebSearchResult,
    filter_results,
    results_as_model_text,
    results_as_tool_content,
    run_web_search,
    validate_web_search_config,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)

# A trimmed but structurally faithful Mojeek results page.
MOJEEK_HTML = """
<html><body>
<ul class="results-standard">
<li><h2><a class="title" title="https://github.com/NVIDIA/TensorRT-LLM"
   href="https://github.com/NVIDIA/TensorRT-LLM">GitHub - NVIDIA/<b>TensorRT</b>-LLM</a></h2>
<p class="s">TensorRT LLM optimizes inference &amp; more</p></li>
<li><h2><a class="title" title="https://docs.nvidia.com/tensorrt-llm/"
   href="https://docs.nvidia.com/tensorrt-llm/">NVIDIA TensorRT-LLM Docs</a></h2>
<p class="s">Python API docs</p></li>
</ul></body></html>
"""


def _request(**overrides):
    payload = {
        "model": "m",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "who won?"}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }
    payload.update(overrides)
    return AnthropicMessagesRequest(**payload)


def _chat_response(content=None, tool_calls=None, finish_reason="stop"):
    # tool_calls must go through the constructor so pydantic coerces the dicts
    # into tool-call models; assigning them afterwards leaves plain dicts.
    message = ChatMessage(role="assistant", content=content,
                          tool_calls=tool_calls or [])
    return ChatCompletionResponse(
        model="m",
        choices=[
            ChatCompletionResponseChoice(
                index=0, message=message, finish_reason=finish_reason
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TRTLLM_ANTHROPIC_WEB_SEARCH", raising=False)
    config, error = resolve_web_search(_request())
    assert config is None
    # Same phrasing as the generic server-tool rejection, so clients and the
    # existing adapter tests that match on it keep working.
    assert "is not supported by this server" in error
    assert "TRTLLM_ANTHROPIC_WEB_SEARCH" in error


def test_disabled_web_search_rejected_through_convert(monkeypatch):
    """The disabled path must raise, not silently drop the tool."""
    monkeypatch.delenv("TRTLLM_ANTHROPIC_WEB_SEARCH", raising=False)
    with pytest.raises(AnthropicRequestError, match="server tool.*not supported"):
        convert_anthropic_request(_request())


def test_request_without_web_search_tool_is_untouched(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")
    config, error = resolve_web_search(_request(tools=None))
    assert config is None and error is None


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "brave")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    config, error = resolve_web_search(_request())
    assert config is None
    assert "BRAVE_SEARCH_API_KEY" in error


def test_unknown_provider_is_rejected():
    assert "unknown web search provider" in validate_web_search_config(
        WebSearchConfig(provider="bogus")
    )


def test_client_max_uses_can_lower_but_not_raise(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH_MAX_USES", "5")

    lowered, _ = resolve_web_search(
        _request(tools=[{"type": "web_search_20250305", "name": "web_search",
                         "max_uses": 2}])
    )
    assert lowered.max_uses == 2

    raised, _ = resolve_web_search(
        _request(tools=[{"type": "web_search_20250305", "name": "web_search",
                         "max_uses": 99}])
    )
    assert raised.max_uses == 5, "server cap must win over a larger client cap"


def test_other_server_tools_still_rejected(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")
    request = _request(
        tools=[{"type": "code_execution_20250522", "name": "code_execution"}]
    )
    with pytest.raises(AnthropicRequestError, match="not supported by this server"):
        convert_anthropic_request(request)


def test_web_search_becomes_a_function_tool(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")
    chat_request = convert_anthropic_request(_request())
    names = [tool.function.name for tool in chat_request.tools]
    assert names == [WEB_SEARCH_TOOL_NAME]
    assert "query" in chat_request.tools[0].function.parameters["properties"]


# ---------------------------------------------------------------------------
# Result parsing and filtering
# ---------------------------------------------------------------------------


def test_mojeek_parsing(monkeypatch):
    async def fake_fetch(session, method, url, **kwargs):
        return MOJEEK_HTML

    monkeypatch.setattr(web_search_mod, "_fetch", fake_fetch)
    config = WebSearchConfig(provider="mojeek", max_results=5)
    results = asyncio.run(run_web_search("tensorrt", config))

    assert [r.url for r in results] == [
        "https://github.com/NVIDIA/TensorRT-LLM",
        "https://docs.nvidia.com/tensorrt-llm/",
    ]
    # Tags stripped and entities unescaped.
    assert results[0].title == "GitHub - NVIDIA/TensorRT-LLM"
    assert results[0].snippet == "TensorRT LLM optimizes inference & more"


WIKIPEDIA_JSON = json.dumps(
    {
        "query": {
            "search": [
                {
                    "title": "Blackwell (microarchitecture)",
                    "snippet": 'Blackwell is a <span class="searchmatch">GPU</span> '
                               "microarchitecture by Nvidia",
                    "timestamp": "2026-07-01T00:00:00Z",
                }
            ]
        }
    }
)


def test_wikipedia_parsing(monkeypatch):
    async def fake_fetch(session, method, url, **kwargs):
        return WIKIPEDIA_JSON

    monkeypatch.setattr(web_search_mod, "_fetch", fake_fetch)
    results = asyncio.run(
        run_web_search("blackwell", WebSearchConfig(provider="wikipedia"))
    )
    assert len(results) == 1
    assert results[0].title == "Blackwell (microarchitecture)"
    # Title is turned into a stable article URL and the snippet is de-tagged.
    assert results[0].url == (
        "https://en.wikipedia.org/wiki/Blackwell_%28microarchitecture%29"
    )
    assert "<span" not in results[0].snippet
    assert results[0].page_age == "2026-07-01T00:00:00Z"


def test_wikipedia_needs_no_credentials():
    assert validate_web_search_config(WebSearchConfig(provider="wikipedia")) is None


def test_max_results_truncates(monkeypatch):
    async def fake_fetch(session, method, url, **kwargs):
        return MOJEEK_HTML

    monkeypatch.setattr(web_search_mod, "_fetch", fake_fetch)
    results = asyncio.run(
        run_web_search("tensorrt", WebSearchConfig(provider="mojeek", max_results=1))
    )
    assert len(results) == 1


def test_backend_error_becomes_web_search_error(monkeypatch):
    async def failing_fetch(session, method, url, **kwargs):
        raise WebSearchError("search backend returned HTTP 503")

    monkeypatch.setattr(web_search_mod, "_fetch", failing_fetch)
    with pytest.raises(WebSearchError):
        asyncio.run(run_web_search("x", WebSearchConfig(provider="mojeek")))


def test_empty_query_rejected():
    with pytest.raises(WebSearchError, match="empty"):
        asyncio.run(run_web_search("   ", WebSearchConfig(provider="mojeek")))


@pytest.mark.parametrize(
    "allowed,blocked,expected",
    [
        (("nvidia.com",), (), ["https://docs.nvidia.com/a"]),
        ((), ("github.com",), ["https://docs.nvidia.com/a"]),
        ((), (), ["https://github.com/x", "https://docs.nvidia.com/a"]),
        # allowed_domains wins when a caller sends both.
        (("github.com",), ("github.com",), ["https://github.com/x"]),
    ],
)
def test_domain_filters(allowed, blocked, expected):
    results = [
        WebSearchResult(url="https://github.com/x", title="x"),
        WebSearchResult(url="https://docs.nvidia.com/a", title="a"),
    ]
    config = WebSearchConfig(allowed_domains=allowed, blocked_domains=blocked)
    assert [r.url for r in filter_results(results, config)] == expected


def test_subdomains_match_parent_domain():
    results = [WebSearchResult(url="https://deep.docs.nvidia.com/a", title="a")]
    config = WebSearchConfig(allowed_domains=("nvidia.com",))
    assert len(filter_results(results, config)) == 1


def test_model_text_and_tool_content_shapes():
    results = [WebSearchResult(url="https://x.com/a", title="A", snippet="s")]
    text = results_as_model_text("q", results)
    assert "A" in text and "https://x.com/a" in text

    blocks = results_as_tool_content(results)
    assert blocks[0]["type"] == "web_search_result"
    assert blocks[0]["url"] == "https://x.com/a"

    assert "No results" in results_as_model_text("q", [])


# ---------------------------------------------------------------------------
# The server-side search loop
# ---------------------------------------------------------------------------


def _run_loop(responses, config, search=None):
    """Drive run_web_search_turns against a scripted sequence of responses."""
    calls = iter(responses)
    seen = []

    async def invoke(chat_request):
        seen.append(len(chat_request.messages))
        return next(calls)

    async def fake_search(query, cfg):
        if search is not None:
            return search(query)
        return [WebSearchResult(url="https://x.com/a", title="A", snippet="s")]

    original = web_search_mod.run_web_search
    web_search_mod.run_web_search = fake_search
    try:
        request = AnthropicMessagesRequest(
            model="m",
            max_tokens=64,
            messages=[{"role": "user", "content": "q"}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )
        chat_request = convert_anthropic_request(request)
        return asyncio.run(run_web_search_turns(chat_request, config, invoke)), seen
    finally:
        web_search_mod.run_web_search = original


def test_search_then_answer(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")
    responses = [
        _chat_response(tool_calls=[_tool_call("c1", WEB_SEARCH_TOOL_NAME,
                                              {"query": "who won"})],
                       finish_reason="tool_calls"),
        _chat_response(content="Team A won."),
    ]
    message, seen = _run_loop(responses, WebSearchConfig(provider="mojeek", max_uses=3))

    kinds = [b.type for b in message.content]
    assert kinds == ["server_tool_use", "web_search_tool_result", "text"]
    assert message.content[0].input == {"query": "who won"}
    assert message.content[1].tool_use_id == message.content[0].id
    assert message.content[2].text == "Team A won."
    assert message.stop_reason == "end_turn"
    assert message.usage.server_tool_use.web_search_requests == 1
    # Usage is summed across both model turns.
    assert message.usage.output_tokens == 10
    # The second turn saw the assistant tool call plus the tool result.
    assert seen[1] == seen[0] + 2


def test_no_search_requested_is_passthrough(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")
    message, seen = _run_loop(
        [_chat_response(content="No search needed.")],
        WebSearchConfig(provider="mojeek", max_uses=3),
    )
    assert [b.type for b in message.content] == ["text"]
    assert message.usage.server_tool_use is None
    assert len(seen) == 1


def test_client_tool_call_ends_the_loop(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")
    responses = [
        _chat_response(tool_calls=[_tool_call("c1", "get_weather", {"city": "Paris"})],
                       finish_reason="tool_calls")
    ]
    message, _ = _run_loop(responses, WebSearchConfig(provider="mojeek", max_uses=3))
    assert [b.type for b in message.content] == ["tool_use"]
    assert message.content[0].name == "get_weather"
    assert message.stop_reason == "tool_use"


def test_max_uses_is_enforced(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")
    searching = _chat_response(
        tool_calls=[_tool_call("c1", WEB_SEARCH_TOOL_NAME, {"query": "q"})],
        finish_reason="tool_calls",
    )
    # Model keeps trying to search; budget is 1.
    message, _ = _run_loop(
        [searching, searching, _chat_response(content="done")],
        WebSearchConfig(provider="mojeek", max_uses=1),
    )
    errors = [
        b for b in message.content
        if b.type == "web_search_tool_result" and not isinstance(b.content, list)
    ]
    assert errors and errors[0].content.error_code == "max_uses_exceeded"
    assert message.usage.server_tool_use.web_search_requests == 1


def test_exhausted_budget_still_yields_an_answer(monkeypatch):
    """Regression: a model that keeps searching must not return empty text.

    With max_uses=1 and a turn budget of max_uses+1, the rejected second
    search consumed the answer turn and the request came back with an empty
    text block and stop_reason "max_tokens".
    """
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")
    searching = _chat_response(
        tool_calls=[_tool_call("c1", WEB_SEARCH_TOOL_NAME, {"query": "q"})],
        finish_reason="tool_calls",
    )
    message, _ = _run_loop(
        [searching, searching, _chat_response(content="final answer")],
        WebSearchConfig(provider="mojeek", max_uses=1),
    )
    assert message.content[-1].type == "text"
    assert message.content[-1].text == "final answer"
    assert message.stop_reason == "end_turn"


def test_web_search_tool_stays_registered_after_budget(monkeypatch):
    """Regression: withdrawing the tool made the parser leak raw markup.

    Removing web_search from the request also removes the tool parser from
    it, so a model that emits another tool call has that call passed through
    as literal text and parser markup reaches the user. The tool therefore
    stays registered and over-budget calls are rejected here instead.
    """
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "wikipedia")
    seen_tools, seen_roles = [], []

    async def invoke(chat_request):
        seen_tools.append([t.function.name for t in (chat_request.tools or [])])
        seen_roles.append([m.get("role") for m in chat_request.messages])
        if len(seen_tools) == 1:
            return _chat_response(
                tool_calls=[_tool_call("c1", WEB_SEARCH_TOOL_NAME, {"query": "q"})],
                finish_reason="tool_calls",
            )
        return _chat_response(content="done")

    async def fake_search(query, cfg):
        return [WebSearchResult(url="https://x.com/a", title="A")]

    original = web_search_mod.run_web_search
    web_search_mod.run_web_search = fake_search
    try:
        chat_request = convert_anthropic_request(
            AnthropicMessagesRequest(
                model="m",
                max_tokens=64,
                messages=[{"role": "user", "content": "q"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
            )
        )
        asyncio.run(
            run_web_search_turns(
                chat_request, WebSearchConfig(provider="wikipedia", max_uses=1), invoke
            )
        )
    finally:
        web_search_mod.run_web_search = original

    assert seen_tools[0] == [WEB_SEARCH_TOOL_NAME]
    assert seen_tools[1] == [WEB_SEARCH_TOOL_NAME], "tool must stay registered"
    # A nudge is appended instead, telling the model to stop searching.
    assert seen_roles[1][-1] == "user"


def test_prose_is_salvaged_when_model_never_stops(monkeypatch):
    """A model that keeps searching must not yield an empty message."""
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "wikipedia")
    stubborn = _chat_response(
        content="Partial analysis so far.",
        tool_calls=[_tool_call("c1", WEB_SEARCH_TOOL_NAME, {"query": "q"})],
        finish_reason="tool_calls",
    )
    message, _ = _run_loop(
        [stubborn] * 5, WebSearchConfig(provider="wikipedia", max_uses=1)
    )
    assert message.content[-1].type == "text"
    assert message.content[-1].text == "Partial analysis so far."


def test_transport_failure_is_retried(monkeypatch):
    """A flaky hop must not surface to the model as 'no results'."""
    attempts = {"n": 0}

    async def flaky_fetch(session, method, url, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise WebSearchError("connection reset")
        return MOJEEK_HTML

    monkeypatch.setattr(web_search_mod, "_fetch", flaky_fetch)
    config = WebSearchConfig(provider="mojeek", retries=2, retry_backoff_s=0)
    results = asyncio.run(run_web_search("tensorrt", config))
    assert attempts["n"] == 3
    assert len(results) == 2


def test_retries_are_bounded(monkeypatch):
    attempts = {"n": 0}

    async def always_fails(session, method, url, **kwargs):
        attempts["n"] += 1
        raise WebSearchError("down")

    monkeypatch.setattr(web_search_mod, "_fetch", always_fails)
    config = WebSearchConfig(provider="mojeek", retries=2, retry_backoff_s=0)
    with pytest.raises(WebSearchError, match="after 3 attempts"):
        asyncio.run(run_web_search("q", config))
    assert attempts["n"] == 3


def test_search_failure_becomes_an_error_block(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")

    def boom(query):
        raise WebSearchError("backend down")

    responses = [
        _chat_response(tool_calls=[_tool_call("c1", WEB_SEARCH_TOOL_NAME,
                                              {"query": "q"})],
                       finish_reason="tool_calls"),
        _chat_response(content="I could not search."),
    ]
    message, _ = _run_loop(
        responses, WebSearchConfig(provider="mojeek", max_uses=3), search=boom
    )
    result_block = message.content[1]
    assert result_block.type == "web_search_tool_result"
    assert result_block.content.error_code == "unavailable"
    # A failed search must not fail the request.
    assert message.content[-1].text == "I could not search."


# ---------------------------------------------------------------------------
# Synthetic streaming
# ---------------------------------------------------------------------------


def test_synthetic_stream_event_order(monkeypatch):
    monkeypatch.setenv("TRTLLM_ANTHROPIC_WEB_SEARCH", "mojeek")
    responses = [
        _chat_response(tool_calls=[_tool_call("c1", WEB_SEARCH_TOOL_NAME,
                                              {"query": "q"})],
                       finish_reason="tool_calls"),
        _chat_response(content="answer"),
    ]
    message, _ = _run_loop(responses, WebSearchConfig(provider="mojeek", max_uses=2))

    async def collect():
        return [chunk async for chunk in synthesize_anthropic_stream(message)]

    chunks = asyncio.run(collect())
    events = [
        line.split("event: ", 1)[1].strip()
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("event: ")
    ]
    assert events[0] == "message_start"
    assert events[-1] == "message_stop"
    assert events[-2] == "message_delta"
    assert events.count("content_block_start") == len(message.content)
    assert events.count("content_block_stop") == len(message.content)
    # The terminal delta carries the stop reason and the search count.
    payload = json.loads(chunks[-2].split("data: ", 1)[1])
    assert payload["delta"]["stop_reason"] == "end_turn"
    assert payload["usage"]["server_tool_use"]["web_search_requests"] == 1
