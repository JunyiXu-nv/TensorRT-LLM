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
"""What rendering changed about a request's input is recorded on the request.

The body Codex sends is not the prompt the model sees: a namespace it dropped
is restored, a tool result nothing claims becomes a note, an unknown item is
replayed as text. Each is counted on ``request._input_replay`` so the trace
can carry it; a request rendered as sent gets the counts and nothing else.
Also here: the deployment-wide output cap a request without one inherits, and
the per-call ``name_unresolved`` event.
"""

import asyncio
import copy
import glob
import json
import os
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    _BARE_TOOL_NAME_CALL_ID_SUFFIX,
    RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS_ENV,
    ReplayLedger,
    _create_input_messages,
    _tool_call_output_item,
    _tool_resolution,
    responses_default_max_output_tokens,
)

pytestmark = pytest.mark.cpu_only


def _namespace(name, *tools):
    return {
        "type": "namespace",
        "name": name,
        "description": "",
        "tools": [{"type": "custom", "name": tool, "description": "Run."} for tool in tools],
    }


def _request(items, **overrides):
    fields = dict(model="kimi-k3", input=copy.deepcopy(items), instructions="You are Codex.", stream=False, store=False)
    fields.update(overrides)
    return ResponsesRequest(**fields)


def _render(request, kimi_k3=True):
    return asyncio.run(_create_input_messages(request=request, prev_msgs=[], kimi_k3=kimi_k3))


class TestLedger:
    def test_counts_notes_and_the_cap(self):
        ledger = ReplayLedger()
        ledger.count("a")
        ledger.count("a", 2)
        for i in range(70):
            ledger.note("ids", f"call_{i}")
        ledger.set("n", 5)
        data = ledger.as_dict()
        assert data["a"] == 3
        assert len(data["ids"]) == 64
        assert data["ids_truncated"] == 6
        assert data["n"] == 5


class TestInputReplayOnTheRequest:
    def test_a_request_rendered_as_sent_gets_only_the_counts(self):
        request = _request(
            [
                {"type": "additional_tools", "tools": [_namespace("functions", "exec")]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
            ]
        )
        assert request._input_replay is None
        messages = _render(request)
        replay = request._input_replay
        assert replay["messages"] == len(messages)
        assert replay["input_items"] == 1  # additional_tools was hoisted into tools
        assert replay["kimi_k3"] is True
        assert "namespace_restored" not in replay
        assert "orphan_tool_results_noted" not in replay

    def test_a_restored_namespace_is_recorded_by_call_id(self):
        call = {"type": "custom_tool_call", "call_id": "call_1", "name": "exec", "input": "ls"}
        request = _request(
            [
                {"type": "additional_tools", "tools": [_namespace("functions", "exec")]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
                call,
                {"type": "custom_tool_call_output", "call_id": "call_1", "output": "ok"},
            ]
        )
        messages = _render(request)
        assistant = next(m for m in messages if m.get("tool_calls"))
        assert assistant["tool_calls"][0]["function"]["name"] == "functions.exec"
        replay = request._input_replay
        assert replay["namespace_restored"] == ["call_1"]
        assert replay["custom_tool_calls_rewrapped"] == 1

    def test_a_bare_name_the_model_wrote_is_recorded_too(self):
        call_id = "call_2" + _BARE_TOOL_NAME_CALL_ID_SUFFIX
        call = {"type": "custom_tool_call", "call_id": call_id, "name": "exec", "input": "ls"}
        request = _request(
            [
                {"type": "additional_tools", "tools": [_namespace("functions", "exec")]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
                call,
                {"type": "custom_tool_call_output", "call_id": call_id, "output": "ok"},
            ]
        )
        _render(request)
        assert request._input_replay["bare_name_replayed"] == [call_id]
        assert "namespace_restored" not in request._input_replay

    def test_an_orphan_result_and_an_unknown_item_are_recorded(self):
        request = _request(
            [
                {"type": "additional_tools", "tools": [_namespace("functions", "exec")]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
                {"type": "custom_tool_call_output", "call_id": "call_lost", "output": "late result"},
                {"type": "mystery_item", "content": [{"type": "input_text", "text": "hm"}]},
                {"type": "mystery_empty"},
            ]
        )
        _render(request)
        replay = request._input_replay
        assert replay["orphan_tool_results_noted"] == ["call_lost"]
        assert replay["unknown_items_as_text"] == ["mystery_item"]
        assert replay["unknown_items_dropped"] == ["mystery_empty"]

    def test_non_kimi_rendering_records_without_the_k3_passes(self):
        request = _request(
            [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}],
            model="glm-5.3",
        )
        _render(request, kimi_k3=False)
        replay = request._input_replay
        assert replay["kimi_k3"] is False
        assert replay["input_items"] == 1


class TestDefaultOutputCap:
    def test_a_request_with_no_cap_inherits_the_default(self):
        request = _request([{"type": "message", "role": "user", "content": "hi"}])
        params = request.to_sampling_params(default_sampling_params={"max_tokens": 32768})
        assert params.max_tokens == 32768

    def test_the_request_wins_over_the_default(self):
        request = _request([{"type": "message", "role": "user", "content": "hi"}], max_output_tokens=10)
        params = request.to_sampling_params(default_sampling_params={"max_tokens": 32768})
        assert params.max_tokens == 10

    def test_no_default_leaves_the_request_uncapped(self):
        request = _request([{"type": "message", "role": "user", "content": "hi"}])
        assert request.to_sampling_params(default_sampling_params={}).max_tokens is None
        assert request.to_sampling_params().max_tokens is None

    @pytest.mark.parametrize(
        "raw, expected",
        [(None, None), ("", None), ("32768", 32768), ("0", None), ("-5", None), ("many", None), (" 64 ", 64)],
    )
    def test_env_parsing(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv(RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS_ENV, raising=False)
        else:
            monkeypatch.setenv(RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS_ENV, raw)
        assert responses_default_max_output_tokens() == expected


class TestNameUnresolvedEvent:
    def _normalizations(self, directory):
        events = []
        for path in glob.glob(os.path.join(directory, "*", "normalize-*.jsonl")):
            with open(path, encoding="utf-8") as handle:
                events += [json.loads(line) for line in handle if line.strip()]
        return events

    def test_a_call_to_an_undeclared_tool_is_recorded_per_call(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRTLLM_RAW_OUTPUT_DIR", str(tmp_path))
        request = _request([{"type": "additional_tools", "tools": [_namespace("functions", "exec")]}])
        resolution = _tool_resolution(request.tools)
        call = SimpleNamespace(name="exec_command", parameters='{"cmd": "ls"}', normalizations=None)

        item = _tool_call_output_item(call, resolution, call_id="call_x")

        assert item.type == "function_call"
        assert item.name == "exec_command"
        (record,) = self._normalizations(str(tmp_path))
        assert record["call_id"] == "call_x"
        assert {"kind": "name_unresolved", "name": "exec_command"} in record["normalizations"]

    def test_a_declared_tool_records_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRTLLM_RAW_OUTPUT_DIR", str(tmp_path))
        request = _request([{"type": "additional_tools", "tools": [_namespace("functions", "exec")]}])
        resolution = _tool_resolution(request.tools)
        call = SimpleNamespace(name="functions.exec", parameters='{"input": "ls"}', normalizations=None)

        item = _tool_call_output_item(call, resolution, call_id="call_y")

        assert item.type == "custom_tool_call"
        assert self._normalizations(str(tmp_path)) == []


def test_a_key_counted_before_it_is_noted_keeps_both():
    ledger = ReplayLedger()
    ledger.count("unknown_items_as_text")
    ledger.note("unknown_items_as_text", None)  # an item whose type dumped as null
    assert ledger.as_dict() == {"unknown_items_as_text": [None], "unknown_items_as_text_count": 1}
