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
"""Tests of the validator in scripts/validate (CPU only, no TensorRT-LLM import).

The fast tests cover the parts every verdict depends on; the control matrix
(fake servers for all three protocols) runs with MH_RUN_CONTROLS=1.

    pytest .claude/skills/trtllm-serve-model-harness/tests
    MH_RUN_CONTROLS=1 pytest .claude/skills/trtllm-serve-model-harness/tests
"""

import json
import os
import sys

import pytest

VALIDATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts",
                        "validate")  # fmt: skip
sys.path.insert(0, VALIDATE)

import controls  # noqa: E402
import profiles  # noqa: E402
import proto_anthropic  # noqa: E402
import proto_chat  # noqa: E402
import verdict  # noqa: E402
from tracelib import sse_frames  # noqa: E402
from turnlib import Item, RequestView, Tool, classify_call, payload_digest  # noqa: E402


def test_sse_frames_keep_every_frame():
    frames = sse_frames(
        'event: ping\ndata: {"type":"ping"}\n\n: keepalive\n\ndata: [DONE]\n\ndata: x\n\n'
    )
    assert [(f.name, f.kind) for f in frames] == [
        ("ping", "json"),
        (None, "comment"),
        (None, "done"),
        (None, "unparsable"),
    ]


def test_payload_digest_is_typed_and_order_free():
    assert payload_digest('{"a": 1, "b": [1, 2]}', "function") == payload_digest(
        '{"b":[1,2],"a":1}', "function"
    )
    assert payload_digest('{"a": true}', "function") != payload_digest('{"a": 1}', "function")
    assert payload_digest('{"a": [1, 2]}', "function") != payload_digest(
        '{"a": [2, 1]}', "function"
    )
    assert payload_digest("echo hi", "custom") != payload_digest("echo  hi", "custom")


def test_wilson_bounds_follow_the_rule_of_three():
    lower, upper = verdict.wilson(0, 20)
    assert lower == 0.0 and upper > 0.01  # kit case C25: 0 of 20 proves nothing at 1%
    _, upper = verdict.wilson(0, 3000)
    assert upper < 0.001


def test_classify_call_against_own_declarations():
    request = RequestView(protocol="responses", route="/v1/responses")
    request.tools = {"exec": Tool("exec", "functions", "custom"),
                     "wait": Tool("wait", "functions", "function",
                                  schema={"type": "object", "required": ["ms"]})}  # fmt: skip
    assert classify_call(Item("call", name="exec", call_kind="function", payload="{}"),
                         request)["class"] == "type_mismatch"  # fmt: skip
    assert classify_call(Item("call", name="functions.wait", call_kind="function", payload='{"ms": 1}'),
                         request)["class"] == "prefixed_name_of_declared"  # fmt: skip
    assert classify_call(Item("call", name="wait", call_kind="function", payload="{}"),
                         request)["class"] == "payload_invalid"  # fmt: skip
    assert classify_call(Item("call", name="nope", call_kind="function", payload="{}"),
                         request)["class"] == "undeclared"  # fmt: skip


def test_anthropic_stream_rules():
    good = controls._anthropic_sse([{"type": "text", "text": "hi"},
                                    {"type": "tool_use", "id": "t1", "name": "Read", "json": '{"a": 1}'}])  # fmt: skip
    reply = proto_anthropic.parse_reply({"response": {"kind": "sse_text", "body": good}}, None)
    assert reply.grammar == []
    assert [i.kind for i in reply.stream_items] == ["text", "call"]
    assert reply.terminal["kind"] == "completed"
    truncated = good.split("event: message_delta")[0]
    reply = proto_anthropic.parse_reply({"response": {"kind": "sse_text", "body": truncated}}, None)
    assert {"ANT-08", "ANT-10"} <= {v.rule for v in reply.grammar}


def test_chat_stream_rules():
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                           "function": {"name": "read", "arguments": '{"a"'}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]  # fmt: skip
    body = "".join("data: " + json.dumps(dict(c, id="x", object="chat.completion.chunk", model="m",
                                              created=1)) + "\n\n" for c in chunks) + "data: [DONE]\n\n"  # fmt: skip
    reply = proto_chat.parse_reply({"response": {"kind": "sse_text", "body": body}}, None)
    assert reply.grammar == []
    assert reply.stream_items[-1].payload == '{"a": 1}'
    reply = proto_chat.parse_reply({"response": {"kind": "sse_text",
                                                 "body": body.replace("data: [DONE]\n\n", "")}}, None)  # fmt: skip
    assert "CHT-09" in {v.rule for v in reply.grammar}


def test_todo_profile_fields_are_unknown():
    pi = profiles.load_profile(os.path.join(os.path.dirname(VALIDATE), "..", "profiles", "harness",
                                            "pi-chat.yaml"))  # fmt: skip
    assert profiles.outcome_rules(pi) is None
    assert pi.get_path("timeouts.idle_s") is None


def test_session_rates_need_session_identity():
    report = {"counts": {"den:replies": 50, "den:calls": 40, "M-02": 0}, "sessions": 0,
              "sessions_with": {}, "outcome_classes": {}, "examples": {}, "by_rule": {}}  # fmt: skip
    thresholds = {"unit_for_rates": "session", "model_behavior": {"degenerate_rate": 0.5}}
    records = verdict.check_records(report, thresholds, profiles.Profile(), "window")
    m02 = next(r for r in records if r["id"] == "M-02")
    assert m02["status"] == "BLOCKED" and "session" in m02["reason"]


def test_cross_cases(tmp_path):
    results = controls.cross_cases(str(tmp_path))
    assert results and all(r["ok"] for r in results), results


@pytest.mark.skipif(os.environ.get("MH_RUN_CONTROLS") != "1", reason="set MH_RUN_CONTROLS=1")
def test_control_matrix(tmp_path):
    assert controls.main(["--out-dir", str(tmp_path)]) == 0
