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
"""Offline unit tests for the Anthropic Messages API protocol adapter.

No GPU or engine required: these exercise only the request/response
conversion functions and the streaming reframer state machine.
"""

import json

import pytest

from tensorrt_llm.serve.anthropic_adapter import (
    AnthropicRequestError,
    AnthropicStreamReframer,
    convert_anthropic_request,
    convert_chat_response,
    convert_usage,
    map_stop_reason,
)
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    FunctionCall,
    PromptTokensDetails,
    ToolCall,
    UsageInfo,
)

MODEL = "test-model"


def make_request(**overrides) -> AnthropicMessagesRequest:
    payload = {
        "model": MODEL,
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return AnthropicMessagesRequest(**payload)


# ---------------------------------------------------------------------------
# Request conversion
# ---------------------------------------------------------------------------


def test_simple_text_request():
    chat = convert_anthropic_request(make_request())
    assert chat.model == MODEL
    assert chat.max_completion_tokens == 128
    assert chat.messages == [{"role": "user", "content": "hello"}]
    assert not chat.stream


def test_system_field_becomes_leading_system_message():
    chat = convert_anthropic_request(
        make_request(system="be brief", messages=[{"role": "user", "content": "hi"}])
    )
    assert chat.messages[0] == {"role": "system", "content": "be brief"}
    assert chat.messages[1]["role"] == "user"


def test_system_blocks_and_inline_system_merged():
    chat = convert_anthropic_request(
        make_request(
            system=[{"type": "text", "text": "part-a"}],
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "part-b"},
                {"role": "user", "content": "again"},
            ],
        )
    )
    assert chat.messages[0]["role"] == "system"
    assert "part-a" in chat.messages[0]["content"]
    assert "part-b" in chat.messages[0]["content"]
    # No mid-conversation system message survives.
    assert all(m["role"] != "system" for m in chat.messages[1:])


def test_tool_use_and_tool_result_round_trip():
    chat = convert_anthropic_request(
        make_request(
            messages=[
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "let me check"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "beijing"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "sunny",
                        }
                    ],
                },
            ]
        )
    )
    assistant = chat.messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "let me check"
    # Pydantic validates the assistant typed-dict's tool_calls lazily into a
    # single-pass ValidatorIterator (same shape the FastAPI-parsed OpenAI
    # path produces); materialize it for inspection.
    tool_calls = [dict(tc) for tc in assistant["tool_calls"]]
    assert tool_calls[0]["id"] == "toolu_1"
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "beijing"}
    tool_msg = chat.messages[2]
    assert tool_msg == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "sunny",
    }


def test_tool_result_ordering_preserved():
    """Text before a tool_result must be flushed before the tool message."""
    chat = convert_anthropic_request(
        make_request(
            messages=[
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "f",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "result",
                        },
                        {"type": "text", "text": "after"},
                    ],
                },
            ]
        )
    )
    roles = [m["role"] for m in chat.messages]
    assert roles == ["user", "assistant", "user", "tool", "user"]


def test_tools_converted_and_server_tools_skipped():
    chat = convert_anthropic_request(
        make_request(
            tools=[
                {
                    "name": "get_weather",
                    "description": "d",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
                {"name": "web_search", "type": "web_search_20260209"},
            ]
        )
    )
    assert len(chat.tools) == 1
    assert chat.tools[0].function.name == "get_weather"
    assert chat.tools[0].function.parameters["properties"]["city"] == {"type": "string"}
    # tools present and no explicit tool_choice -> auto
    assert chat.tool_choice == "auto"


def test_tool_choice_mappings():
    tools = [{"name": "f", "input_schema": {"type": "object"}}]
    for anthropic_type, expected in [("auto", "auto"), ("any", "auto"), ("none", "none")]:
        chat = convert_anthropic_request(
            make_request(tools=tools, tool_choice={"type": anthropic_type})
        )
        assert chat.tool_choice == expected

    chat = convert_anthropic_request(
        make_request(tools=tools, tool_choice={"type": "tool", "name": "f"})
    )
    assert chat.tool_choice.function.name == "f"


def test_tool_choice_unknown_tool_rejected():
    with pytest.raises(AnthropicRequestError):
        convert_anthropic_request(
            make_request(
                tools=[{"name": "f", "input_schema": {"type": "object"}}],
                tool_choice={"type": "tool", "name": "missing"},
            )
        )


def test_tool_choice_without_client_tools_rejected():
    with pytest.raises(AnthropicRequestError):
        convert_anthropic_request(
            make_request(
                tools=[{"name": "web_search", "type": "web_search_20260209"}],
                tool_choice={"type": "any"},
            )
        )


def test_base64_image_converted_to_data_uri():
    chat = convert_anthropic_request(
        make_request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "abcd",
                            },
                        },
                        {"type": "text", "text": "what is this"},
                    ],
                }
            ]
        )
    )
    parts = chat.messages[0]["content"]
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"] == "data:image/jpeg;base64,abcd"


def test_stop_sequences_and_sampling_passthrough():
    chat = convert_anthropic_request(
        make_request(stop_sequences=["END"], temperature=0.5, top_p=0.9, top_k=40)
    )
    assert chat.stop == ["END"]
    assert chat.temperature == 0.5
    assert chat.top_p == 0.9
    assert chat.top_k == 40


def test_unknown_extra_fields_tolerated():
    # Claude Code attaches metadata / betas / output_config and other
    # evolving fields; they must not fail validation.
    request = AnthropicMessagesRequest(
        model=MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": "hi"}],
        metadata={"user_id": "u1"},
        betas=["some-beta"],
        output_config={"effort": "high"},
        unknown_future_field={"x": 1},
    )
    chat = convert_anthropic_request(request)
    assert chat.messages[0]["content"] == "hi"


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


def make_chat_response(
    message: ChatMessage, finish_reason: str = "stop", usage: UsageInfo = None
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model=MODEL,
        choices=[
            ChatCompletionResponseChoice(index=0, message=message, finish_reason=finish_reason)
        ],
        usage=usage or UsageInfo(prompt_tokens=10, completion_tokens=5),
    )


def test_text_response():
    resp = convert_chat_response(
        make_chat_response(ChatMessage(role="assistant", content="hi there"))
    )
    assert resp.type == "message"
    assert resp.role == "assistant"
    assert resp.stop_reason == "end_turn"
    assert resp.content[0].type == "text"
    assert resp.content[0].text == "hi there"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5


def test_tool_call_response():
    message = ChatMessage(
        role="assistant",
        content="checking",
        tool_calls=[
            ToolCall(
                id="call_1", function=FunctionCall(name="get_weather", arguments='{"city": "sf"}')
            )
        ],
    )
    resp = convert_chat_response(make_chat_response(message, finish_reason="tool_calls"))
    assert resp.stop_reason == "tool_use"
    types = [block.type for block in resp.content]
    assert types == ["text", "tool_use"]
    tool_block = resp.content[1]
    assert tool_block.id == "call_1"
    assert tool_block.name == "get_weather"
    assert tool_block.input == {"city": "sf"}


def test_malformed_tool_arguments_degrade_to_empty_input():
    message = ChatMessage(
        role="assistant",
        tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="f", arguments="{broken"))],
    )
    resp = convert_chat_response(make_chat_response(message, finish_reason="tool_calls"))
    assert resp.content[-1].input == {}


def test_reasoning_becomes_thinking_block():
    message = ChatMessage(role="assistant", content="answer", reasoning_content="step by step")
    resp = convert_chat_response(make_chat_response(message))
    assert resp.content[0].type == "thinking"
    assert resp.content[0].thinking == "step by step"
    assert resp.content[1].type == "text"


def test_empty_content_gets_placeholder_text_block():
    resp = convert_chat_response(make_chat_response(ChatMessage(role="assistant", content=None)))
    assert len(resp.content) == 1
    assert resp.content[0].type == "text"
    assert resp.content[0].text == ""


def test_stop_reason_mapping():
    assert map_stop_reason("stop") == "end_turn"
    assert map_stop_reason("length") == "max_tokens"
    assert map_stop_reason("tool_calls") == "tool_use"
    assert map_stop_reason(None) == "end_turn"
    assert map_stop_reason("unknown") == "end_turn"


def test_usage_cache_read_split():
    usage = convert_usage(
        UsageInfo(
            prompt_tokens=100,
            completion_tokens=7,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=60),
        )
    )
    assert usage.input_tokens == 40
    assert usage.cache_read_input_tokens == 60
    assert usage.output_tokens == 7


# ---------------------------------------------------------------------------
# Streaming reframer
# ---------------------------------------------------------------------------


def parse_frames(frames):
    """Parse SSE frames into (event_name, payload dict) tuples."""
    parsed = []
    for frame in frames:
        lines = [l for l in frame.strip().splitlines() if l]
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        event = lines[0][len("event: ") :]
        payload = json.loads(lines[1][len("data: ") :])
        assert payload["type"] == event
        parsed.append((event, payload))
    return parsed


def chunk(delta: dict, finish_reason=None, usage: UsageInfo = None) -> ChatCompletionStreamResponse:
    return ChatCompletionStreamResponse(
        model=MODEL,
        choices=[{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        usage=usage,
    )


def run_reframer(chunks):
    reframer = AnthropicStreamReframer(model=MODEL)
    frames = []
    for c in chunks:
        frames.extend(reframer.process_chunk(c))
    frames.extend(reframer.finish())
    return parse_frames(frames)


def assert_event_invariants(events):
    """First-principles invariants of the Anthropic streaming protocol."""
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert names.count("message_start") == 1
    assert names[-1] == "message_stop"
    assert names[-2] == "message_delta"
    open_blocks = {}
    max_index = -1
    for name, payload in events:
        if name == "content_block_start":
            index = payload["index"]
            assert index not in open_blocks
            assert index == max_index + 1, "indices must be monotonic"
            max_index = index
            open_blocks[index] = payload["content_block"]["type"]
        elif name == "content_block_delta":
            index = payload["index"]
            assert index in open_blocks
            delta_type = payload["delta"]["type"]
            block_type = open_blocks[index]
            assert (block_type, delta_type) in {
                ("text", "text_delta"),
                ("tool_use", "input_json_delta"),
                ("thinking", "thinking_delta"),
                ("thinking", "signature_delta"),
            }
        elif name == "content_block_stop":
            assert payload["index"] in open_blocks
            del open_blocks[payload["index"]]
    assert not open_blocks, "all blocks must be closed"


def test_stream_text_only():
    events = run_reframer(
        [
            chunk({"role": "assistant"}, usage=UsageInfo(prompt_tokens=12, completion_tokens=0)),
            chunk({"content": "hel"}),
            chunk({"content": "lo"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    assert_event_invariants(events)
    assert events[0][1]["message"]["usage"]["input_tokens"] == 12
    text_deltas = [p["delta"]["text"] for n, p in events if n == "content_block_delta"]
    assert "".join(text_deltas) == "hello"
    message_delta = [p for n, p in events if n == "message_delta"][0]
    assert message_delta["delta"]["stop_reason"] == "end_turn"


def test_stream_tool_call_arguments_concatenate():
    events = run_reframer(
        [
            chunk({"content": "using tool"}),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                }
            ),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' "sf"}'}}]}),
            chunk({}, finish_reason="tool_calls"),
        ]
    )
    assert_event_invariants(events)
    starts = [p for n, p in events if n == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["text", "tool_use"]
    assert starts[1]["content_block"]["name"] == "get_weather"
    args = "".join(
        p["delta"]["partial_json"]
        for n, p in events
        if n == "content_block_delta" and p["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(args) == {"city": "sf"}
    message_delta = [p for n, p in events if n == "message_delta"][0]
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_stream_parallel_tool_calls_get_separate_blocks():
    events = run_reframer(
        [
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "function": {"name": "f1", "arguments": '{"a": 1}'},
                        }
                    ]
                }
            ),
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "c2",
                            "function": {"name": "f2", "arguments": '{"b": 2}'},
                        }
                    ]
                }
            ),
            chunk({}, finish_reason="tool_calls"),
        ]
    )
    assert_event_invariants(events)
    starts = [p["content_block"] for n, p in events if n == "content_block_start"]
    assert [s["name"] for s in starts] == ["f1", "f2"]
    assert starts[0]["id"] == "c1"
    assert starts[1]["id"] == "c2"


def test_stream_thinking_then_text():
    events = run_reframer(
        [
            chunk({"reasoning_content": "thinking..."}),
            chunk({"content": "answer"}),
            chunk({}, finish_reason="stop"),
        ]
    )
    assert_event_invariants(events)
    starts = [p["content_block"]["type"] for n, p in events if n == "content_block_start"]
    assert starts == ["thinking", "text"]


def test_stream_empty_generation_still_valid():
    events = run_reframer([chunk({}, finish_reason="stop")])
    assert_event_invariants(events)
