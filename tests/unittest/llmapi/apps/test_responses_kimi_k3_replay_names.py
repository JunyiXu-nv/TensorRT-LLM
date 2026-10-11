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
"""A replayed kimi_k3 tool call keeps the name the model wrote.

Codex echoes a namespaced call back as `name="exec"` and may leave `namespace` out.
The model wrote `functions.exec` unless the call_id carries the bare marker, so the
qualified name is restored from the request's declared tools -- when, and only when,
exactly one namespaced tool answers to the bare name.
"""

import asyncio
import copy

import pytest

from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    _BARE_TOOL_NAME_CALL_ID_SUFFIX,
    _create_input_messages,
)

pytestmark = pytest.mark.cpu_only


def _namespace(name, *tools):
    return {
        "type": "namespace",
        "name": name,
        "description": "",
        "tools": [{"type": "custom", "name": tool, "description": "Run."} for tool in tools],
    }


def _request(tools, call):
    items = [
        {"type": "additional_tools", "tools": tools},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
        call,
        {"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "ok"},
    ]
    return ResponsesRequest(
        model="kimi-k3",
        input=copy.deepcopy(items),
        instructions="You are Codex.",
        stream=False,
        store=False,
    )


def _call(call_id, namespace=None):
    call = {
        "type": "custom_tool_call",
        "call_id": call_id,
        "name": "exec",
        "input": "ls",
        "status": "completed",
    }
    if namespace is not None:
        call["namespace"] = namespace
    return call


def _replayed_name(tools, call, kimi_k3=True):
    messages = asyncio.run(_create_input_messages(_request(tools, call), [], kimi_k3=kimi_k3))
    (assistant,) = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    return assistant["tool_calls"][0]["function"]["name"]


FUNCTIONS = [_namespace("functions", "exec", "wait")]


def test_a_dropped_namespace_is_restored():
    assert _replayed_name(FUNCTIONS, _call("call_a")) == "functions.exec"


def test_a_call_the_model_wrote_bare_stays_bare():
    call = _call("call_b" + _BARE_TOOL_NAME_CALL_ID_SUFFIX)
    assert _replayed_name(FUNCTIONS, call) == "exec"


def test_a_call_that_kept_its_namespace_is_unchanged():
    assert _replayed_name(FUNCTIONS, _call("call_c", namespace="functions")) == "functions.exec"


def test_an_ambiguous_bare_name_is_replayed_as_received():
    tools = [_namespace("functions", "exec"), _namespace("sandbox", "exec")]
    assert _replayed_name(tools, _call("call_d")) == "exec"


def test_a_top_level_tool_of_that_name_wins():
    tools = FUNCTIONS + [{"type": "custom", "name": "exec", "description": "Top level."}]
    assert _replayed_name(tools, _call("call_e")) == "exec"


def test_glm_replay_is_unchanged():
    assert _replayed_name(FUNCTIONS, _call("call_f"), kimi_k3=False) == "exec"
