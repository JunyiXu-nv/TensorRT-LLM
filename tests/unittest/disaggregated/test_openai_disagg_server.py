# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request
from starlette.datastructures import Headers

from tensorrt_llm.llmapi.disagg_utils import ServerRole, extract_disagg_cfg
from tensorrt_llm.serve import openai_disagg_server
from tensorrt_llm.serve.openai_disagg_server import (
    OpenAIDisaggServer,
    _DownstreamClientDisconnected,
)
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
)

pytestmark = pytest.mark.cpu_only


def _raw_request(headers: dict[str, str]):
    return SimpleNamespace(headers=Headers(headers=headers))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_kwargs", "expected_timeout"),
    [({}, 10), ({"server_keep_alive_timeout": 3600}, 3600)],
)
async def test_server_keep_alive_timeout_is_passed_to_uvicorn(
    monkeypatch, config_kwargs, expected_timeout
):
    config = extract_disagg_cfg(
        context_servers={"num_instances": 0},
        generation_servers={"num_instances": 0},
        **config_kwargs,
    )
    server = object.__new__(OpenAIDisaggServer)
    server._config = config
    server.app = object()

    uvicorn_config = object()
    config_factory = Mock(return_value=uvicorn_config)
    uvicorn_server = SimpleNamespace(serve=AsyncMock())
    server_factory = Mock(return_value=uvicorn_server)
    monkeypatch.setattr(openai_disagg_server.uvicorn, "Config", config_factory)
    monkeypatch.setattr(openai_disagg_server.uvicorn, "Server", server_factory)

    await server(host="localhost", port=8000)

    assert config_factory.call_args.kwargs["timeout_keep_alive"] == expected_timeout
    server_factory.assert_called_once_with(uvicorn_config)
    uvicorn_server.serve.assert_awaited_once_with(sockets=None)


@pytest.mark.asyncio
async def test_http_cluster_storage_request_is_proxied_to_coordinator():
    payload = b'{"key":"worker","value":"ready"}'

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/set",
            "query_string": b"source=worker",
            "headers": [(b"content-type", b"application/json")],
        },
        receive,
    )
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._coordinator = SimpleNamespace(
        proxy_cluster_storage_request=AsyncMock(
            return_value=(b'{"result":true}', 200, "application/json")
        )
    )

    response = await server._proxy_cluster_storage_request(request)

    server._coordinator.proxy_cluster_storage_request.assert_awaited_once_with(
        "POST", "/set", [("source", "worker")], payload, "application/json"
    )
    assert response.status_code == 200
    assert response.body == b'{"result":true}'


def test_create_client_does_not_register_with_server_metrics_collector():
    server = OpenAIDisaggServer.__new__(OpenAIDisaggServer)
    server._coordinator = SimpleNamespace(get_disagg_request_id=AsyncMock(return_value=1))
    server._req_timeout_secs = 30
    server._collect_perf_metrics = True
    server._config = SimpleNamespace(internal_request_auth_key="key")
    server._perf_metrics_collector = SimpleNamespace()

    with patch("tensorrt_llm.serve.openai_disagg_server.OpenAIHttpClient") as mock_client:
        client = server._create_client(SimpleNamespace(), ServerRole.GENERATION, max_retries=2)

    assert client is mock_client.return_value
    mock_client.assert_called_once()


def test_extract_conversation_id_from_headers():
    cases = [
        ({"X-Session-ID": "session-id"}, "session-id"),
        ({"X-Correlation-ID": "correlation-id"}, "correlation-id"),
        ({"x-session-affinity": "session-affinity"}, "session-affinity"),
        ({"x-multi-turn-session-id": "multi-turn-session-id"}, "multi-turn-session-id"),
        (
            {
                "X-Correlation-ID": "correlation-id",
                "X-Session-ID": "session-id",
                "x-session-affinity": "session-affinity",
                "x-multi-turn-session-id": "multi-turn-session-id",
            },
            "session-id",
        ),
        (
            {
                "x-session-affinity": "session-affinity",
                "x-multi-turn-session-id": "multi-turn-session-id",
            },
            "session-affinity",
        ),
        (
            {
                "X-Session-ID": "",
                "X-Correlation-ID": "correlation-id",
            },
            "correlation-id",
        ),
    ]

    for headers, expected_conversation_id in cases:
        request = CompletionRequest(model="test-model", prompt="hello")

        OpenAIDisaggServer._extract_conversation_id(request, _raw_request(headers))

        assert request.disaggregated_params is None
        assert request.conversation_params.conversation_id == expected_conversation_id


def test_extract_conversation_id_ignores_empty_headers():
    request = CompletionRequest(model="test-model", prompt="hello")

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request(
            {
                "X-Session-ID": "",
                "X-Correlation-ID": " ",
                "x-session-affinity": "",
                "x-multi-turn-session-id": " ",
            }
        ),
    )

    assert request.disaggregated_params is None
    assert request.conversation_params is None


def test_extract_conversation_id_preserves_body_conversation_params():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(conversation_id="body-id"),
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"X-Session-ID": "header-id"}),
    )

    assert request.conversation_params.conversation_id == "body-id"


def test_extract_conversation_id_populates_conversation_params_with_existing_disaggregated_params():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )

    OpenAIDisaggServer._extract_conversation_id(
        request,
        _raw_request({"x-multi-turn-session-id": "multi-turn-session-id"}),
    )

    assert request.conversation_params.conversation_id == "multi-turn-session-id"


class TestClientDisconnectWatch:
    """A client that hangs up mid-pipeline must take the pipeline down with it.

    The pre-response window is the proxy's blind spot: the pinned
    fastapi/starlette request_response runs a handler to completion no matter
    what the socket does, so a context-first request whose client timed out
    kept its whole prefill grinding on the context worker -- and the retry
    storm stacked one giant prefill per abandoned attempt. These tests drive
    _serve_until_client_disconnect (and the wrapper around it) at the seam
    below FastAPI, where no engine and no worker are needed.
    """

    @staticmethod
    def _server():
        return OpenAIDisaggServer.__new__(OpenAIDisaggServer)

    @staticmethod
    def _raw(is_disconnected):
        return SimpleNamespace(is_disconnected=is_disconnected)

    @pytest.mark.asyncio
    async def test_disconnect_mid_pipeline_cancels_it_and_raises(self):
        """The disconnect cancels the entry task; its finalizers run first."""
        server = self._server()
        started = asyncio.Event()
        finalized = asyncio.Event()

        async def entry(req, hooks):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Stands in for the pipeline's real cleanup: the client's
                # finalize (router settle + upstream socket close) runs under
                # this very cancellation, so it must be given the chance to.
                finalized.set()
                raise

        with pytest.raises(_DownstreamClientDisconnected):
            await server._serve_until_client_disconnect(
                entry, "req", "hooks", self._raw(AsyncMock(return_value=True))
            )

        assert started.is_set()
        # The settlement was awaited before the disconnect was reported.
        assert finalized.is_set()

    @pytest.mark.asyncio
    async def test_pipeline_completing_first_is_returned_unchanged(self):
        """No disconnect: the watch is invisible and the result flows through."""
        server = self._server()
        polls = AsyncMock(return_value=False)

        async def entry(req, hooks):
            return "response"

        result = await server._serve_until_client_disconnect(
            entry, "req", "hooks", self._raw(polls)
        )

        assert result == "response"

    @pytest.mark.asyncio
    async def test_pipeline_error_propagates_unchanged(self):
        """An entry failure keeps its meaning; the watch adds nothing to it."""
        server = self._server()

        async def entry(req, hooks):
            raise ValueError("upstream rejected")

        with pytest.raises(ValueError, match="upstream rejected"):
            await server._serve_until_client_disconnect(
                entry, "req", "hooks", self._raw(AsyncMock(return_value=False))
            )

    @pytest.mark.asyncio
    async def test_cancel_losing_the_race_returns_the_finished_response(self):
        """A pipeline that completes during the cancel is a response, not an abort.

        Whatever it produced is handed back: uvicorn discards the send and a
        streaming body is torn down by StreamingResponse's own disconnect
        handling, both of which finalize -- reporting a disconnect instead
        would leave a never-consumed generator and its routing entry behind.
        """
        server = self._server()

        async def entry(req, hooks):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # The pipeline finished on its own while the cancel unwound --
                # e.g. the gen generator was already built and returned without
                # another suspension point.
                return "finished-during-unwind"

        result = await server._serve_until_client_disconnect(
            entry, "req", "hooks", self._raw(AsyncMock(return_value=True))
        )

        assert result == "finished-during-unwind"

    @pytest.mark.asyncio
    async def test_watch_failure_degrades_to_serving_not_cancelling(self):
        """A broken watcher says nothing about the client; the request survives."""
        server = self._server()

        async def entry(req, hooks):
            await asyncio.sleep(0)
            return "served"

        result = await server._serve_until_client_disconnect(
            entry, "req", "hooks", self._raw(AsyncMock(side_effect=RuntimeError("watch broke")))
        )

        assert result == "served"

    @pytest.mark.asyncio
    async def test_wrapper_answers_a_disconnect_with_499_and_traces_it(self):
        """End of the proxy's story: a finalized abort, an honest status, a trace terminal."""
        server = self._server()
        counter = lambda: SimpleNamespace(inc=Mock())  # noqa: E731
        server._perf_metrics_collector = SimpleNamespace(
            total_requests=counter(),
            stream_requests=counter(),
            nonstream_requests=counter(),
            total_responses=counter(),
            queue_latency_seconds=SimpleNamespace(observe=Mock()),
        )
        server._request_trace = SimpleNamespace(
            on_request=AsyncMock(return_value=None),
            on_response=Mock(),
            wrap_stream=lambda stream, handle: stream,
        )
        server._allow_request_chat_template = False
        server._collect_perf_metrics = False

        cancelled = asyncio.Event()

        async def entry(req, hooks):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        raw_req = SimpleNamespace(
            state=SimpleNamespace(server_arrival_time=0.0),
            headers=Headers({}),
            is_disconnected=AsyncMock(return_value=True),
            client=("10.0.0.1", 40000),
        )
        request = CompletionRequest(model="m", prompt="hi", stream=False)

        wrapper = server._wrap_entry_point(entry, CompletionRequest)
        response = await wrapper(request, raw_req)

        assert cancelled.is_set()
        assert response.status_code == 499
        server._request_trace.on_response.assert_called_once_with(
            None, payload=None, status="client_disconnected"
        )


def test_disagg_config_allows_request_chat_template_opt_in():
    config = extract_disagg_cfg(
        context_servers={"num_instances": 0},
        generation_servers={"num_instances": 0},
        allow_request_chat_template=True,
    )

    assert config.allow_request_chat_template is True


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_disagg_config_rejects_non_bool_request_chat_template_opt_in(value):
    with pytest.raises(ValueError, match="allow_request_chat_template must be a boolean"):
        extract_disagg_cfg(
            context_servers={"num_instances": 0},
            generation_servers={"num_instances": 0},
            allow_request_chat_template=value,
        )
