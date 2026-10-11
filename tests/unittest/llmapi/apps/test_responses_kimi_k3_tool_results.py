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
"""A kimi_k3 call whose result came back in several items replays as one result.

The K3 renderer sorts a run of results by the call each one names but numbers them
by counting result messages, so a second segment of call 1 took call 2's index and
pushed call 2's result to an index no call has. The conversion joins one call's
segments within a run. The renderer test needs the checkpoint's ``encoding_k3.py``.
"""

import asyncio
import copy
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import _create_input_messages

pytestmark = pytest.mark.cpu_only

TOOLS = {
    "type": "additional_tools",
    "tools": [
        {
            "type": "namespace",
            "name": "functions",
            "description": "",
            "tools": [{"type": "custom", "name": "exec", "description": "Run."}],
        }
    ],
}


def _call(call_id):
    return {
        "type": "custom_tool_call",
        "call_id": call_id,
        "name": "exec",
        "namespace": "functions",
        "input": f"run {call_id}",
        "status": "completed",
    }


def _result(call_id, text):
    return {"type": "custom_tool_call_output", "call_id": call_id, "output": text}


def _messages(items, kimi_k3=True):
    request = ResponsesRequest(
        model="kimi-k3",
        input=copy.deepcopy(
            [
                TOOLS,
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                },
                *items,
            ]
        ),
        instructions="You are Codex.",
        stream=False,
        store=False,
    )
    return asyncio.run(_create_input_messages(request, [], kimi_k3=kimi_k3))


def _tool_messages(messages):
    return [(m["tool_call_id"], m["content"]) for m in messages if m.get("role") == "tool"]


def test_a_call_with_two_result_segments_replays_one_result():
    messages = _messages(
        [_call("a"), _call("b"), _result("a", "A1"), _result("a", "A2"), _result("b", "B")]
    )
    assert _tool_messages(messages) == [("a", "A1\nA2"), ("b", "B")]


def test_interleaved_segments_are_joined_in_order():
    messages = _messages(
        [_call("a"), _call("b"), _result("a", "A1"), _result("b", "B"), _result("a", "A2\n")]
    )
    assert _tool_messages(messages) == [("a", "A1\nA2\n"), ("b", "B")]


def test_one_result_per_call_is_unchanged():
    messages = _messages([_call("a"), _call("b"), _result("b", "B"), _result("a", "A")])
    assert _tool_messages(messages) == [("b", "B"), ("a", "A")]


def test_a_result_in_a_later_run_is_not_moved_into_an_earlier_one():
    later = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "and?"}]}
    messages = _messages([_call("a"), _result("a", "A1"), later, _result("a", "A2")])
    tool_and_notes = [m for m in messages if m.get("role") in ("tool", "user")]
    # The first segment binds to its call; the late one stays after the user turn.
    assert tool_and_notes[1] == {"role": "tool", "content": "A1", "tool_call_id": "a"}
    assert "A2" in str(tool_and_notes[-1]["content"])


def test_glm_keeps_every_segment():
    messages = _messages([_call("a"), _result("a", "A1"), _result("a", "A2")], kimi_k3=False)
    assert _tool_messages(messages) == [("a", "A1"), ("a", "A2")]


def _renderer():
    roots = [os.environ.get("TRTLLM_K3_RENDERER_DIR"), os.environ.get("LLM_MODELS_ROOT")]
    for root in filter(None, roots):
        for candidate in (Path(root), Path(root) / "Kimi-K3-NVFP4", Path(root) / "Kimi-K3"):
            path = candidate / "encoding_k3.py"
            if path.is_file():
                spec = importlib.util.spec_from_file_location("encoding_k3", path)
                module = importlib.util.module_from_spec(spec)
                # Its dataclasses resolve annotations through sys.modules.
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module
    pytest.skip("the K3 checkpoint renderer (encoding_k3.py) is not available")


def test_the_renderer_numbers_each_result_with_its_own_call():
    encoding = _renderer()
    messages = _messages(
        [_call("a"), _call("b"), _result("a", "A1"), _result("a", "A2"), _result("b", "B")]
    )
    prompt = "".join(segment.text for segment in encoding.build_chat_segments(messages))
    indexes = re.findall(
        r'role="tool" tool="functions\.exec" index="(\d+)"', prompt.replace("<|sep|>", "")
    )
    assert indexes == ["1", "2"]
    assert prompt.index("A1") < prompt.index("A2") < prompt.index("B")
