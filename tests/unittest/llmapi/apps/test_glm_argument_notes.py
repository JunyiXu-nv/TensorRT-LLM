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
"""A GLM tool call says, per argument, when its delivered JSON is not the text the model wrote.

Delivery itself is unchanged: a schema-typed argument is still parsed and
re-serialized, which serving wants. What changes is that the call now carries
an event for each argument whose literal that round trip altered, so a
training export can tell ``1000.0`` delivered from ``1e3`` written.
"""

import json

import pytest

from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam, FunctionDefinition
from tensorrt_llm.serve.tool_parser.glm4_parser import Glm4ToolParser, note_argument_delivery
from tensorrt_llm.serve.tool_parser.glm47_parser import Glm47ToolParser

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def tools():
    return [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="run",
                description="Run.",
                parameters={
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "opts": {"type": "object"},
                        "cell_id": {"type": "string"},
                        "flag": {"type": "boolean"},
                    },
                },
            ),
        )
    ]


def _glm47(args):
    return "<tool_call>run" + "".join(f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in args) + "</tool_call>"


def _glm4(args):
    return "<tool_call>run\n" + "".join(f"<arg_key>{k}</arg_key>\n<arg_value>{v}</arg_value>\n" for k, v in args) + "</tool_call>"


def _kinds(call):
    return [event["kind"] for event in (call.normalizations or [])]


class TestNoteArgumentDelivery:
    def test_identical_delivery_records_nothing(self):
        notes = []
        note_argument_delivery(notes, "count", "integer", "25", 25, True)
        assert notes == []

    def test_a_changed_literal_records_both_spellings(self):
        notes = []
        note_argument_delivery(notes, "ratio", "number", "1e3", 1000.0, True)
        assert notes == [
            {"kind": "argument_coerced", "key": "ratio", "declared_type": "number", "from": "1e3", "to": "1000.0"}
        ]

    def test_an_unparsed_value_is_recorded_as_such(self):
        notes = []
        note_argument_delivery(notes, "opts", "object", "not json at all", "not json at all", False)
        assert notes[0]["kind"] == "argument_unparsed"
        assert notes[0]["from"] == "not json at all"

    def test_excerpts_are_bounded(self):
        notes = []
        note_argument_delivery(notes, "opts", "object", "{'a': 1}" + " " * 500, {"a": 1}, True)
        assert len(notes[0]["from"]) == 200

    def test_whitespace_outside_strings_is_not_a_coercion(self):
        notes = []
        note_argument_delivery(notes, "opts", "object", '{"a":1,\n  "b": [1,2]}', {"a": 1, "b": [1, 2]}, True)
        note_argument_delivery(notes, "count", "integer", "\n25\n", 25, True)
        assert notes == []

    def test_whitespace_inside_a_string_still_counts(self):
        notes = []
        note_argument_delivery(notes, "opts", "object", '{"a": "x  y"}', {"a": "x y"}, True)
        assert [n["kind"] for n in notes] == ["argument_coerced"]

    def test_the_passthrough_shape_is_unparsed_whatever_the_flag_says(self):
        # parse_arguments' last strategy hands the text back as a string and
        # reports parsed=True; under a non-string schema that is no parse.
        notes = []
        note_argument_delivery(notes, "opts", "object", "not json at all", "not json at all", True)
        assert [n["kind"] for n in notes] == ["argument_unparsed"]

    def test_a_string_or_untyped_passthrough_is_nothing(self):
        notes = []
        note_argument_delivery(notes, "cell_id", "string", "25", "25", True)
        note_argument_delivery(notes, "free", None, "x y", "x y", True)
        assert notes == []


@pytest.mark.parametrize("parser_class, render", [(Glm47ToolParser, _glm47), (Glm4ToolParser, _glm4)])
class TestDetectAndParseCarriesNotes:
    def test_clean_arguments_carry_no_notes(self, parser_class, render, tools):
        result = parser_class().detect_and_parse(render([("count", "25"), ("cell_id", "99797")]), tools)
        (call,) = result.calls
        assert json.loads(call.parameters) == {"count": 25, "cell_id": "99797"}
        assert call.normalizations is None

    def test_a_coerced_number_is_noted_and_still_delivered(self, parser_class, render, tools):
        result = parser_class().detect_and_parse(render([("ratio", "1e3")]), tools)
        (call,) = result.calls
        assert json.loads(call.parameters) == {"ratio": 1000.0}
        (event,) = call.normalizations
        assert event["kind"] == "argument_coerced"
        assert event["key"] == "ratio"
        assert (event["from"], event["to"]) == ("1e3", "1000.0")

    def test_a_python_literal_object_is_noted(self, parser_class, render, tools):
        result = parser_class().detect_and_parse(render([("opts", "{'a': 1}")]), tools)
        (call,) = result.calls
        assert json.loads(call.parameters) == {"opts": {"a": 1}}
        assert _kinds(call) == ["argument_coerced"]

    def test_a_quoted_integer_under_an_integer_schema_is_noted(self, parser_class, render, tools):
        result = parser_class().detect_and_parse(render([("count", '"25"')]), tools)
        (call,) = result.calls
        assert json.loads(call.parameters) == {"count": 25}
        assert _kinds(call) == ["argument_coerced"]

    def test_a_string_argument_is_verbatim_and_unnoted(self, parser_class, render, tools):
        result = parser_class().detect_and_parse(render([("cell_id", "true")]), tools)
        (call,) = result.calls
        assert json.loads(call.parameters) == {"cell_id": "true"}
        assert call.normalizations is None

    def test_unparseable_text_under_an_object_schema_is_noted_as_unparsed(self, parser_class, render, tools):
        result = parser_class().detect_and_parse(render([("opts", "not json at all")]), tools)
        (call,) = result.calls
        assert json.loads(call.parameters) == {"opts": "not json at all"}
        assert _kinds(call) == ["argument_unparsed"]

    def test_json_whitespace_is_not_noted(self, parser_class, render, tools):
        result = parser_class().detect_and_parse(render([("opts", '{"a":1}'), ("count", " 25 ")]), tools)
        (call,) = result.calls
        assert json.loads(call.parameters) == {"opts": {"a": 1}, "count": 25}
        assert call.normalizations is None

    def test_a_python_boolean_is_noted(self, parser_class, render, tools):
        result = parser_class().detect_and_parse(render([("flag", "True")]), tools)
        (call,) = result.calls
        assert json.loads(call.parameters) == {"flag": True}
        (event,) = call.normalizations
        assert (event["kind"], event["from"], event["to"]) == ("argument_coerced", "True", "true")

    def test_a_conflicting_repeat_is_noted(self, parser_class, render, tools):
        result = parser_class().detect_and_parse(render([("count", "1"), ("count", "2")]), tools)
        (call,) = result.calls
        assert json.loads(call.parameters) == {"count": 1}
        assert _kinds(call) == ["argument_repeated"]
