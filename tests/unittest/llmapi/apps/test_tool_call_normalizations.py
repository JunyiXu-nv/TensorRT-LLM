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
"""A delivered tool call that serving changed is recorded under its call_id.

The custom tool path delivers a near-miss rather than refusing it -- extra keys dropped,
a sole key taken as `input`, a non-string input serialized -- and the parser's own repairs
ride on the call. Each such call gets one line beside the raw output, so a training export
can tell it from a call the model wrote as declared. The delivered item itself is unchanged.
"""

import glob
import json
import os

import pytest

from tensorrt_llm.serve import tool_call_normalizations
from tensorrt_llm.serve.responses_utils import _tool_call_output_item
from tensorrt_llm.serve.tool_parser.core_types import ToolCallItem

pytestmark = pytest.mark.cpu_only

CUSTOM = {"exec": ("functions", "exec", True), "functions.exec": ("functions", "exec", True)}
FUNCTION = {"wait": ("functions", "wait", False)}


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("TRTLLM_RAW_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(tool_call_normalizations, "_seen", tool_call_normalizations.OrderedDict())

    def lines():
        found = []
        for path in sorted(glob.glob(os.path.join(str(tmp_path), "*", "normalize-*.jsonl"))):
            with open(path, encoding="utf-8") as handle:
                found += [json.loads(line) for line in handle if line.strip()]
        return found

    return lines


def _deliver(arguments, resolution=CUSTOM, name="exec", call_id="call_1", normalizations=None):
    call = ToolCallItem(
        tool_index=0, name=name, parameters=arguments, normalizations=normalizations
    )
    return _tool_call_output_item(call, resolution, call_id=call_id)


def _kinds(lines):
    return [[event["kind"] for event in line["normalizations"]] for line in lines]


def test_a_call_written_as_declared_writes_nothing(ledger):
    item = _deliver(json.dumps({"input": "ls"}))
    assert item.input == "ls"
    assert ledger() == []


def test_extra_keys_are_recorded_with_their_values(ledger):
    item = _deliver(json.dumps({"input": "ls", "workdir": "/tmp", "yield_time_ms": 500}))
    assert item.input == "ls"
    (line,) = ledger()
    assert line["call_id"] == "call_1" and line["item_id"] == item.id and line["name"] == "exec"
    assert line["normalizations"] == [
        {
            "kind": "custom_input_extra_keys_dropped",
            "dropped": {"workdir": "/tmp", "yield_time_ms": 500},
        }
    ]


def test_a_sole_key_taken_as_input_is_recorded(ledger):
    item = _deliver(json.dumps({"apply_patch": "*** Begin Patch"}))
    assert item.input == "*** Begin Patch"
    assert ledger()[0]["normalizations"] == [
        {"kind": "custom_input_from_sole_key", "key": "apply_patch"}
    ]


@pytest.mark.parametrize(
    "arguments, kinds",
    [
        (json.dumps({"a": 1, "b": 2}), ["custom_input_missing"]),
        (json.dumps({"input": 5}), ["custom_input_not_string"]),
        ("not json", ["custom_arguments_not_json"]),
        (json.dumps(["ls"]), ["custom_arguments_not_object"]),
    ],
)
def test_other_departures_are_recorded(ledger, arguments, kinds):
    _deliver(arguments)
    assert _kinds(ledger()) == [kinds]


def test_the_parser_events_ride_along_on_function_calls_too(ledger):
    events = [{"kind": "argument_type_unknown", "key": "ms", "type": "undefined"}]
    item = _deliver(
        json.dumps({"ms": 10108}), resolution=FUNCTION, name="wait", normalizations=events
    )
    assert item.type == "function_call"
    assert ledger()[0]["normalizations"] == events


def test_a_call_reported_twice_is_recorded_once(ledger):
    arguments = json.dumps({"input": "ls", "workdir": "/tmp"})
    _deliver(arguments, call_id="call_9")
    _deliver(arguments, call_id="call_9")
    assert len(ledger()) == 1


def test_nothing_is_written_without_the_raw_output_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("TRTLLM_RAW_OUTPUT_DIR", raising=False)
    item = _deliver(json.dumps({"input": "ls", "workdir": "/tmp"}))
    assert item.input == "ls"
    assert glob.glob(os.path.join(str(tmp_path), "**", "normalize-*.jsonl"), recursive=True) == []
