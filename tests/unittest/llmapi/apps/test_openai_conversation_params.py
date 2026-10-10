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

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.conversation_id import resolve_request_conversation_id
from tensorrt_llm.serve.openai_protocol import ConversationParams, ResponsesRequest


def test_openai_conversation_params_requires_conversation_id():
    with pytest.raises(ValidationError):
        ConversationParams()


def test_openai_conversation_params_normalizes_conversation_id():
    params = ConversationParams(conversation_id=" body-id ")

    assert params.conversation_id == "body-id"

    params.conversation_id = " next-id "

    assert params.conversation_id == "next-id"


def test_openai_conversation_params_rejects_empty_conversation_id():
    with pytest.raises(ValidationError):
        ConversationParams(conversation_id=" ")


def test_resolve_request_conversation_id_preserves_body_params():
    conversation_params = ConversationParams(conversation_id="body-id")
    request = SimpleNamespace(conversation_params=conversation_params)

    conversation_id = resolve_request_conversation_id(request, {"X-Session-ID": "header-id"})

    assert conversation_id == "body-id"
    assert request.conversation_params is conversation_params


def test_resolve_request_conversation_id_uses_headers_when_body_params_missing():
    request = SimpleNamespace(conversation_params=None)

    conversation_id = resolve_request_conversation_id(request, {"X-Session-ID": " header-id "})

    assert conversation_id == "header-id"
    assert request.conversation_params.conversation_id == "header-id"


def test_resolve_request_conversation_id_uses_prompt_cache_key_when_headers_missing():
    # Codex over the Responses API: the thread id rides in prompt_cache_key.
    request = SimpleNamespace(conversation_params=None, prompt_cache_key=" thread-1 ")

    conversation_id = resolve_request_conversation_id(request, {})

    assert conversation_id == "thread-1"
    assert request.conversation_params.conversation_id == "thread-1"


def test_resolve_request_conversation_id_uses_client_metadata_thread_id():
    # Codex over chat completions: the thread id sits in client_metadata; a run
    # (session_id) can hold several threads, so thread_id is preferred.
    request = SimpleNamespace(conversation_params=None,
                              client_metadata={"session_id": "run-1", "thread_id": "thread-2", "turn_id": "turn-9"})

    assert resolve_request_conversation_id(request, None) == "thread-2"
    assert request.conversation_params.conversation_id == "thread-2"


def test_resolve_request_conversation_id_falls_back_to_client_metadata_session_id():
    request = SimpleNamespace(conversation_params=None, client_metadata={"session_id": "run-1"})

    assert resolve_request_conversation_id(request, None) == "run-1"


def test_resolve_request_conversation_id_prefers_headers_over_body_fields():
    request = SimpleNamespace(conversation_params=None, prompt_cache_key="thread-1")

    assert resolve_request_conversation_id(request, {"X-Session-ID": "header-id"}) == "header-id"


def test_resolve_request_conversation_id_ignores_blank_body_fields():
    request = SimpleNamespace(conversation_params=None, prompt_cache_key="  ",
                              client_metadata={"session_id": ""})

    assert resolve_request_conversation_id(request, {}) is None
    assert request.conversation_params is None


def test_resolve_request_conversation_id_reads_responses_request_extra_field():
    # ResponsesRequest keeps unknown fields as pydantic extras; getattr must see them.
    request = ResponsesRequest(model="test-model", input="hello", prompt_cache_key="thread-3")

    assert resolve_request_conversation_id(request, {}) == "thread-3"
    assert request.conversation_params.conversation_id == "thread-3"
