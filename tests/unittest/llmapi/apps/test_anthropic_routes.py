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
"""Route-level tests for the Anthropic Messages compatibility handlers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

MODEL = "test-model"


def _request(**overrides):
    payload = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }
    payload.update(overrides)
    return payload


def _chat_response(*, tool_arguments=None):
    tool_calls = []
    if tool_arguments is not None:
        tool_calls = [
            ToolCall(
                id="call_1",
                function=FunctionCall(
                    name="get_weather", arguments=tool_arguments
                ),
            )
        ]
    return ChatCompletionResponse(
        id="chatcmpl-route-test",
        model=MODEL,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant", content="hello", tool_calls=tool_calls
                ),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def _json_chat_response(*, tool_arguments=None, status_code=200):
    return JSONResponse(
        content=_chat_response(tool_arguments=tool_arguments).model_dump(),
        status_code=status_code,
    )


def _streaming_chat_response():
    chunks = [
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[{"index": 0, "delta": {"role": "assistant"}}],
        ),
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[{"index": 0, "delta": {"content": "hello"}}],
        ),
        ChatCompletionStreamResponse(
            model=MODEL,
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            usage=UsageInfo(
                prompt_tokens=3, completion_tokens=2, total_tokens=5
            ),
        ),
    ]

    async def source():
        for chunk in chunks:
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(source(), media_type="text/event-stream")


def _make_route_client(server_kind, openai_response):
    app = FastAPI()
    if server_kind == "standard":
        server = object.__new__(OpenAIServer)
        server.model = MODEL
        backend = AsyncMock(return_value=openai_response)
        server.openai_chat = backend
    else:
        server = object.__new__(OpenAIDisaggServer)
        server._service = SimpleNamespace(openai_chat_completion=object())
        backend = AsyncMock(return_value=openai_response)
        server._wrap_entry_point = Mock(return_value=backend)
    app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
    return TestClient(app), backend


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_converts_nonstream_response(server_kind):
    client, backend = _make_route_client(server_kind, _json_chat_response())

    response = client.post("/v1/messages", json=_request())

    assert response.status_code == 200
    assert response.json() | {"id": "ignored"} == {
        "id": "ignored",
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    chat_request = backend.await_args.args[0]
    assert chat_request.model == MODEL
    assert chat_request.max_completion_tokens == 64
    assert not chat_request.stream


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_rejects_anthropic_server_tools(server_kind):
    client, backend = _make_route_client(server_kind, _json_chat_response())

    response = client.post(
        "/v1/messages",
        json=_request(
            tools=[
                {
                    "name": "web_search",
                    "type": "web_search_20250305",
                }
            ]
        ),
    )

    assert response.status_code == 400
    assert response.json()["type"] == "error"
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "server tool" in response.json()["error"]["message"]
    backend.assert_not_awaited()


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_hides_invalid_generated_tool_arguments(server_kind):
    client, _ = _make_route_client(
        server_kind, _json_chat_response(tool_arguments="{not-json")
    )

    response = client.post("/v1/messages", json=_request())

    assert response.status_code == 500
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "Internal server error"},
    }


@pytest.mark.parametrize("server_kind", ["standard", "disagg"])
def test_messages_route_reframes_streaming_response(server_kind):
    client, backend = _make_route_client(
        server_kind, _streaming_chat_response()
    )

    response = client.post("/v1/messages", json=_request(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.startswith("event: message_start\n")
    assert "event: content_block_delta\n" in response.text
    assert '"text":"hello"' in response.text
    assert response.text.rstrip().endswith(
        'event: message_stop\ndata: {"type":"message_stop"}'
    )
    assert backend.await_args.args[0].stream


def test_standard_and_disagg_register_messages_route():
    standard = object.__new__(OpenAIServer)
    standard.app = FastAPI()
    standard.generator = SimpleNamespace(
        _executor=SimpleNamespace(resource_governor_queue=None),
        args=SimpleNamespace(return_perf_metrics=False),
    )
    standard.use_harmony = False
    standard.register_routes()

    disagg = object.__new__(OpenAIDisaggServer)
    disagg.app = FastAPI()
    disagg._service = SimpleNamespace(
        openai_completion=AsyncMock(), openai_chat_completion=AsyncMock()
    )
    disagg._perf_metrics_collector = SimpleNamespace(
        get_perf_metrics=AsyncMock()
    )
    disagg._disagg_cluster_storage = None
    disagg.register_routes()

    for server in (standard, disagg):
        paths = {route.path for route in server.app.routes}
        assert "/v1/messages" in paths
