#!/usr/bin/env python3
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
"""The control matrix that qualifies this validator (references/acceptance-cases.md).

For each protocol, a fake server is started and a synthetic corpus is replayed
once per fault mode; ``verdict.py`` must trip exactly the mode's expected set
(rules plus declared consequences) and nothing else, and every normal control
must trip nothing. Then the cross-protocol fixture cases (X-..) run without a
server. No real trace is read.

The receipt (``<out-dir>/controls.json``) binds the result to the validator tree
SHA-256; ``verdict.py --tier full --controls-receipt`` accepts only a passing
receipt for the exact tree that issues the verdict (check V-01).

usage: controls.py --out-dir DIR [--only responses,anthropic,chat,cross] [--keep]
Exit 0 when every case passes, 1 otherwise.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
SKILL = os.path.dirname(SCRIPTS)
sys.path.insert(0, HERE)
import pipeline as pipeline_mod  # noqa: E402
import profiles as profiles_mod  # noqa: E402
import verdict as verdict_mod  # noqa: E402

PY = sys.executable
GLM_PROFILE = os.path.join(SKILL, "profiles", "model", "glm-5.yaml")
CODEX_OUTCOMES = {
    "aborted": {"exact": "aborted"},
    "unsupported": {"prefix": "unsupported call:"},
    "arguments_rejected": {"prefix": "failed to parse function arguments"},
    "unavailable_in_mode": {"regex": r"is unavailable in (\w+) mode"},
}
# Claude Code's tool results for a call it did not run (2.1.289-2.1.309 binaries): rejected by the
# user or a permission rule, or interrupted.
CLAUDE_ABORTED = [
    "The user doesn't want to proceed with this tool use.",
    "Permission for this tool use was denied.",
    "Permission to use ",
    "[Request interrupted by user for tool use]",
    "[Tool call did not complete:",
]
CONTROL_PROFILES = {
    "responses": {
        "id": "control-responses", "status": "measured", "protocol": "responses",
        "identity": {"session": {"body": "prompt_cache_key"},
                     "thread": {"body": "client_metadata.thread_id"}},
        "outcomes": CODEX_OUTCOMES,
        "errors": {"overlength": {"shape": "in_stream", "event": "response.failed",
                                  "code": "context_length_exceeded"}},
        "timeouts": {"first_event_s": 30, "idle_s": 1.0},
    },
    "anthropic": {
        "id": "control-anthropic", "status": "measured", "protocol": "anthropic-messages",
        "identity": {"session": {"header": "x-claude-code-session-id"}},
        "outcomes": {"aborted": {"prefix": CLAUDE_ABORTED},
                     "unsupported": {"contains": "No such tool available"},
                     "arguments_rejected": {"contains": "InputValidationError"},
                     "error": {"is_error": True}},
        "echo": {"text": "exact", "reasoning": "exact"},
        "errors": {"overlength": {"shape": "http_400", "error_type": "invalid_request_error",
                                  "wording": "prompt is too long"}},
        "timeouts": {"first_event_s": 30, "idle_s": 1.0},
    },
    "chat": {
        "id": "control-chat", "status": "measured", "protocol": "chat-completions",
        "identity": {"thread": {"header": "x-thread"}},
        "outcomes": {"aborted": {"none": "control: the chat client never cancels a call"},
                     "arguments_rejected": {"none": "control: the chat client runs every call"},
                     "error": {"prefix": "Error:"}},
        "echo": {"text": "exact", "reasoning": "exact"},
        "errors": {"overlength": {"shape": "http_400", "error_type": "invalid_request_error",
                                  "wording": "maximum context length"}},
        "timeouts": {"first_event_s": 30, "idle_s": 1.0},
    },
}  # fmt: skip

RESPONSES_EXPECT = {
    "ok": set(), "overlength": set(),
    "drop_part_done": {"C-01", "RSP-07"},
    "snapshot_text_differs": {"C-02", "RSP-11"},
    "delta_differs": {"C-01", "RSP-08", "RSP-09"},
    "no_terminal": {"C-04", "RSP-02"},
    "custom_as_function": {"C-06"},
    "two_calls": {"C-07"},
    "prefixed_name": {"C-06"},
    "unknown_event": {"C-11", "WIRE-05"},
    "created_without_output": {"C-01", "RSP-01", "C-12", "SDK-01"},
    "wrong_output_index": {"C-01", "RSP-15", "C-12", "SDK-01"},
    "usage_missing": {"C-14", "USG-01"},
    "usage_total_mismatch": {"C-14", "USG-02"},
    "error_string": {"C-01", "RSP-16", "C-16"},  # the store rejection pi met on main
}  # fmt: skip
ANTHROPIC_EXPECT = {
    "ok": set(), "ok_thinking": set(), "overlength_profile": set(),
    "overlength_400_wrong_wording": {"C-10"},
    "no_message_delta": {"C-01", "ANT-08", "C-04", "ANT-10"},
    "no_message_stop": {"C-04", "ANT-10"},
    "delta_before_start": {"C-01", "ANT-02", "C-12", "SDK-01"},
    "interleaved_blocks": {"C-01", "ANT-03"},
    "delta_type_mismatch": {"C-01", "ANT-04"},
    "bad_partial_json": {"C-01", "ANT-06", "M-01:payload_invalid"},
    "duplicate_tool_use_id": {"C-01", "ANT-06", "C-05"},
    "stop_reason_mismatch": {"C-01", "ANT-09"},
    "two_tools_disable_parallel": {"C-07"},
    "undeclared_tool": {"M-01:undeclared"},
    "invalid_input": {"M-01:payload_invalid"},
    "error_untyped": {"C-01", "ANT-11"},
    "content_after_stop": {"C-04", "ANT-10"},
    "silent_gap": {"R-02"},
    "html_body": {"C-01", "WIRE-01"},
    "json_for_stream": {"C-01", "WIRE-03"},
    "unknown_block": {"C-11", "WIRE-05"},
    "no_event_lines": {"C-01", "ANT-15", "C-12", "SDK-01"},
    "usage_zero_output": {"C-14", "USG-04"},
    "count_tokens_off": {"C-14", "USG-05"},
}  # fmt: skip
CHAT_EXPECT = {
    "ok": set(), "ok_reasoning": set(), "overlength_profile": set(),
    "no_done": {"C-04", "CHT-09"},
    "two_finish_reasons": {"C-01", "CHT-06"},
    "fragment_without_id": {"C-01", "CHT-04", "C-05"},
    "arguments_not_object": {"C-01", "CHT-05", "M-01:payload_invalid"},
    "finish_reason_mismatch": {"C-01", "CHT-07"},
    "duplicate_tool_call_id": {"C-01", "CHT-04", "C-05"},
    "two_calls_ptc_false": {"C-07"},
    "reasoning_in_content": {"C-09", "M-03"},
    "model_changes": {"C-01", "CHT-02"},
    "invalid_input": {"M-01:payload_invalid"},
    "silent_gap": {"R-02"},
    "html_body": {"C-01", "WIRE-01"},
    "json_for_stream": {"C-01", "WIRE-03"},
    "unknown_delta_field": {"C-11", "WIRE-05"},
    "bad_extension_field": {"C-01", "CHT-13"},
    "tool_index_gap": {"C-01", "CHT-04", "C-12", "SDK-01"},
    "repeated_call_id": {"C-01", "CHT-04", "C-12", "SDK-02"},
    "usage_cached_exceeds": {"C-14", "USG-03"},
    "usage_requested_missing": {"C-14", "USG-01"},
    "two_choices": set(),
    "choice_missing": {"C-01", "CHT-14"},
    "choice_without_finish": {"C-01", "CHT-06"},
    "error_string": {"C-01", "CHT-15", "C-16"},  # the store rejection pi met on main
}  # fmt: skip
READ_SCHEMA = {"type": "object", "properties": {"file_path": {"type": "string"}},
               "required": ["file_path"]}  # fmt: skip


# --------------------------------------------------------------------------- corpora


def responses_body(i, mode):
    grammar = "start: CMD\nCMD: /[a-z ]+/\n"
    tools = [{"type": "namespace", "name": "functions", "tools": [
        {"type": "custom", "name": "exec", "description": "run",
         "format": {"type": "grammar", "syntax": "lark", "definition": grammar}},
        {"type": "function", "name": "wait", "description": "wait",
         "parameters": {"type": "object", "properties": {"ms": {"type": "integer"}},
                        "required": ["ms"]}}]}]  # fmt: skip
    return "/v1/responses", {
        "model": "control", "stream": True, "parallel_tool_calls": False,
        "prompt_cache_key": f"s{i}", "client_metadata": {"thread_id": f"t{i}"},
        "input": [{"type": "additional_tools", "tools": tools},
                  {"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": f"task {i}"}]}],
    }  # fmt: skip


def anthropic_body(i, mode):
    body = {"model": "control", "max_tokens": 256, "stream": True,
            "system": [{"type": "text", "text": "control"}],
            "messages": [{"role": "user", "content": f"task {i}"}],
            "tools": [{"name": "Read", "description": "read a file", "input_schema": READ_SCHEMA}]}  # fmt: skip
    if mode == "two_tools_disable_parallel":
        body["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    return "/v1/messages", body


def chat_body(i, mode):
    body = {"model": "control", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"task {i}"}],
            "tools": [{"type": "function", "function": {"name": "read", "description": "read",
                                                        "parameters": READ_SCHEMA}}]}  # fmt: skip
    if mode == "two_calls_ptc_false":
        body["parallel_tool_calls"] = False
    if mode in ("two_choices", "choice_missing", "choice_without_finish"):
        body["n"] = 2
    return "/v1/chat/completions", body


# --------------------------------------------------------------------------- servers


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start(argv, port, log):
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f"server did not start: {' '.join(argv)}")


def write_profile(path, data):
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)
    return path


def run_verdict(out, paths, harness, population="controls", claims=None, model=GLM_PROFILE,
                matrix=None, raw=None, responses=None):  # fmt: skip
    argv = ["--harness", harness, "--model", model, "--population", population]
    if responses:
        argv += ["--responses"] + list(responses)
    if claims:
        argv += ["--claims", claims]
    if matrix:
        argv += ["--matrix", matrix]
    if raw:
        argv += ["--raw"] + list(raw)
    argv += ["--out-dir", out]  # also ends --raw's list before the positional paths
    code = verdict_mod.main(argv + list(paths))
    with open(os.path.join(out, "verdict.json")) as fh:
        return code, json.load(fh)


def count_tokens_body(body):
    """What a client sends to count a Messages request: its prompt-shaping fields."""
    keep = ("model", "messages", "system", "tools", "tool_choice", "thinking", "output_config",
            "context_management")  # fmt: skip
    return {k: v for k, v in body.items() if k in keep}


def protocol_matrix(name, expect, body_fn, server_argv, work, n=3):
    results = []
    port = free_port()
    log = open(os.path.join(work, f"{name}-server.log"), "w")
    proc = start(server_argv + ["--port", str(port)], port, log)
    harness = write_profile(os.path.join(work, f"{name}-profile.yaml"), CONTROL_PROFILES[name])
    try:
        for mode, expected in expect.items():
            corpus = os.path.join(work, f"{name}-{mode}.corpus.jsonl")
            with open(corpus, "w") as fh:
                for i in range(n):
                    route, body = body_fn(i, mode)
                    fh.write(json.dumps({"corpus_id": f"{name}-{mode}-{i}", "route": route,
                                         "thread": f"{name}-{mode}-{i}", "body": body}) + "\n")  # fmt: skip
                    if route == "/v1/messages":  # a count before the turn, as Claude Code may send
                        fh.write(json.dumps({"corpus_id": f"{name}-{mode}-{i}-count",
                                             "route": "/v1/messages/count_tokens",
                                             "body": count_tokens_body(body)}) + "\n")  # fmt: skip
            out = os.path.join(work, f"{name}-{mode}")
            shutil.rmtree(out, ignore_errors=True)
            replay = subprocess.run(
                [PY, os.path.join(SCRIPTS, "replay_requests.py"), "--base-url",
                 f"http://127.0.0.1:{port}", "--out-dir", out, "--concurrency", "4",
                 "--first-event-timeout", "30", "--header", f"x-fake-mode: {mode}",
                 "--header", "x-claude-code-session-id: control", corpus],
                capture_output=True, text=True, timeout=120)  # fmt: skip
            if replay.returncode != 0:
                results.append({"protocol": name, "mode": mode, "ok": False,
                                "error": "replay failed: " + replay.stderr[-300:]})  # fmt: skip
                continue
            _, verdict = run_verdict(os.path.join(out, "verdict"), [out], harness)
            tripped = set(verdict["tripped"])
            ok = tripped == expected
            results.append({"protocol": name, "mode": mode, "ok": ok,
                            "expected": sorted(expected), "tripped": sorted(tripped)})  # fmt: skip
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        log.close()
    return results


# --------------------------------------------------------------------------- cross cases


def _write_trace(directory, pairs):
    """pairs: [(request_record, response_record_or_None)] -> request/response shards."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "requests-x.jsonl"), "w") as req, \
            open(os.path.join(directory, "responses-x.jsonl"), "w") as resp:  # fmt: skip
        for request, response in pairs:
            req.write(json.dumps(request) + "\n")
            if response is not None:
                resp.write(json.dumps(response) + "\n")


def _anthropic_sse(blocks, stop_reason="tool_use"):
    events = [("message_start", {"message": {"id": "m", "type": "message", "role": "assistant",
                                             "model": "x", "content": [],
                                             "usage": {"input_tokens": 1}}})]  # fmt: skip
    for index, block in enumerate(blocks):
        if block["type"] == "thinking":
            events += [("content_block_start", {"index": index, "content_block": {"type": "thinking", "thinking": ""}}),
                       ("content_block_delta", {"index": index, "delta": {"type": "thinking_delta",
                                                                          "thinking": block["text"]}})]  # fmt: skip
            if block.get("signature"):
                events.append(("content_block_delta", {"index": index, "delta": {
                    "type": "signature_delta", "signature": block["signature"]}}))  # fmt: skip
        elif block["type"] == "text":
            events += [("content_block_start", {"index": index, "content_block": {"type": "text", "text": ""}}),
                       ("content_block_delta", {"index": index, "delta": {"type": "text_delta",
                                                                          "text": block["text"]}})]  # fmt: skip
        else:
            events += [("content_block_start", {"index": index, "content_block": {
                "type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}}),
                ("content_block_delta", {"index": index, "delta": {"type": "input_json_delta",
                                                                   "partial_json": block["json"]}})]  # fmt: skip
        events.append(("content_block_stop", {"index": index}))
    events += [("message_delta", {"delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 1}}),
               ("message_stop", {})]  # fmt: skip
    return "".join(f"event: {n}\ndata: {json.dumps(dict(d, type=n))}\n\n" for n, d in events)


def _anth_request(trace_id, messages, at):
    return {"event": "request", "trace_id": trace_id, "route": "/v1/messages", "recorded_at": at,
            "headers": [["x-claude-code-session-id", "s1"]],
            "body": {"model": "x", "max_tokens": 64, "stream": True, "messages": messages,
                     "tools": [{"name": "Read", "input_schema": READ_SCHEMA}]}}  # fmt: skip


def _sse_response(trace_id, body, status="completed", engine_id=None):
    record = {"event": "response", "trace_id": trace_id, "status": status, "http_status": 200,
              "response": {"kind": "sse_text", "body": body}}  # fmt: skip
    if engine_id is not None:
        record["disagg_request_id"] = engine_id
    return record


GLM_RAW = "Plan.</think>Reading.<tool_call>Read<arg_key>file_path</arg_key><arg_value>a</arg_value></tool_call>"
RAW_CASES = {
    # case: (raw text, delivered blocks, disable_parallel_tool_use, expected C-13 rules, M-07 expected)
    "X-13": (GLM_RAW, [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading."},
                       {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}],
             False, set(), False),
    "X-14": (GLM_RAW, [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading."}],
             False, {"RAW-01"}, False),
    "X-15": (GLM_RAW, [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading."},
                       {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "b"}'}],
             False, {"RAW-02"}, False),
    "X-16": (GLM_RAW, [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading"},
                       {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}],
             False, {"RAW-03"}, False),
    "X-17": (GLM_RAW.replace("</arg_value></tool_call>", ""),
             [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading."}],
             False, set(), True),
    "X-18": (GLM_RAW + GLM_RAW.split("Reading.", 1)[1],
             [{"type": "thinking", "text": "Plan."}, {"type": "text", "text": "Reading."},
              {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}],
             True, set(), False),
}  # fmt: skip


def echo_cases(work, anth):
    """C-03 for texts: Messages and Chat echo our text and reasoning at the appended position."""
    results = []
    user = {"role": "user", "content": "go"}
    delivered = [{"type": "thinking", "text": "Plan.", "signature": "c2ln"},
                 {"type": "text", "text": "Reading."},
                 {"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}]  # fmt: skip
    use = {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a"}}
    result = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
    }
    variants = {
        "X-21": ([{"type": "thinking", "thinking": "Plan.", "signature": "c2ln"},
                  {"type": "text", "text": "Reading."}, use], anth, "PASS", None),
        "X-22": ([{"type": "thinking", "thinking": "Plan.", "signature": "c2ln"},
                  {"type": "text", "text": "Reading!"}, use], anth, "FAIL", "text"),
        "X-23": ([{"type": "thinking", "thinking": "Plan."}, {"type": "text", "text": "Reading."}, use],
                 anth, "FAIL", "reasoning"),
        "X-24": ([{"type": "thinking", "thinking": "Plan.", "signature": "c2ln"},
                  {"type": "text", "text": "Reading."}, use],
                 write_profile(os.path.join(work, "x24.yaml"), dict(CONTROL_PROFILES["anthropic"],
                                                                     echo={"text": "TODO"})),
                 "BLOCKED", None),
    }  # fmt: skip
    for case, (echoed, profile, status, field) in variants.items():
        directory = os.path.join(work, case.lower())
        turn2 = [user, {"role": "assistant", "content": echoed}, result]
        _write_trace(directory, [
            (_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
             _sse_response("t1", _anthropic_sse(delivered))),
            (_anth_request("t2", turn2, "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
        ])  # fmt: skip
        _, v = run_verdict(
            os.path.join(directory, "v"), [directory], profile, "window", "conformance"
        )
        c03 = next(r for r in v["checks"] if r["id"] == "C-03")
        ok = c03["status"] == status and (status != "PASS" or c03["text_echo_equal"] == 1)
        if field:
            with open(os.path.join(directory, "v", "report.json")) as fh:
                fields = {e.get("field") for e in json.load(fh)["examples"].get("C-03", [])}
            ok = ok and field in fields
        results.append({"case": case, "ok": ok})
    # X-47: the same request answered twice (a retry after the network dropped the first answer);
    # the client goes on with the second answer, so only the second is held against the echo.
    directory = os.path.join(work, "x-47")
    first = [{"type": "text", "text": "First answer."}]
    second = [{"type": "text", "text": "Second answer."}]
    turn2 = [user, {"role": "assistant", "content": [{"type": "text", "text": "Second answer."}]},
             {"role": "user", "content": "go on"}]  # fmt: skip
    _write_trace(directory, [
        (_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", _anthropic_sse(first, "end_turn"))),
        (_anth_request("t2", [user], "2026-01-01T00:00:02Z"), _sse_response("t2", _anthropic_sse(second, "end_turn"))),
        (_anth_request("t3", turn2, "2026-01-01T00:00:03Z"),
         _sse_response("t3", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
    ])  # fmt: skip
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
    c03 = next(r for r in v["checks"] if r["id"] == "C-03")
    results.append({"case": "X-47", "ok": c03["status"] == "PASS" and c03["superseded_answers"] == 1
                    and c03["text_echo_equal"] == 1})  # fmt: skip
    # X-48: the harness dropped an empty answer and asked again at the same position, with a text
    # block appended to its last user message (Claude Code after an empty turn). The client went
    # on with the second answer, so only the second is held against the echo.
    # X-49: the second request changed the last user message instead of appending to it: not a
    # re-ask, so the first answer is still judged, and its echo differs.
    reminder = {"role": "system", "content": "reminder"}
    asked = {"role": "user", "content": [{"type": "text", "text": "go"}]}
    nudge = {"type": "text", "text": "Your reply was empty. Continue."}
    empty = [{"type": "thinking", "text": "Hmm.", "signature": "c2ln"}]
    answer = [{"type": "thinking", "text": "Plan.", "signature": "c2ln"},
              {"type": "text", "text": "Second answer."}]  # fmt: skip
    for case, again, status in (
        ("X-48", {"role": "user", "content": asked["content"] + [nudge]}, "PASS"),
        ("X-49", {"role": "user", "content": [{"type": "text", "text": "go!"}, nudge]}, "FAIL"),
    ):
        directory = os.path.join(work, case.lower())
        turn3 = [again, reminder, {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Plan.", "signature": "c2ln"},
            {"type": "text", "text": "Second answer."}]}, {"role": "user", "content": "go on"}]  # fmt: skip
        _write_trace(directory, [
            (_anth_request("t1", [asked, reminder], "2026-01-01T00:00:01Z"),
             _sse_response("t1", _anthropic_sse(empty, "end_turn"))),
            (_anth_request("t2", [again, reminder], "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse(answer, "end_turn"))),
            (_anth_request("t3", turn3, "2026-01-01T00:00:03Z"),
             _sse_response("t3", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
        ])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
        c03 = next(r for r in v["checks"] if r["id"] == "C-03")
        results.append({"case": case, "ok": c03["status"] == status and c03["text_echo_equal"] == 1
                        and c03["reasked_answers"] == (1 if status == "PASS" else 0)})  # fmt: skip
    # X-25: Chat Completions; the echo is the assistant message's content at the appended position.
    chat = write_profile(os.path.join(work, "x25.yaml"), CONTROL_PROFILES["chat"])
    directory = os.path.join(work, "x-25")
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Reading."}}]},
              {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}]  # fmt: skip
    body = "".join("data: " + json.dumps(dict(c, id="x", object="chat.completion.chunk", model="m",
                                              created=1)) + "\n\n" for c in chunks) + "data: [DONE]\n\n"  # fmt: skip

    def chat_request(trace_id, messages, at):
        return {"event": "request", "trace_id": trace_id, "route": "/v1/chat/completions",
                "recorded_at": at, "headers": [["x-thread", "th"]],
                "body": {"model": "m", "stream": True, "messages": messages}}  # fmt: skip

    _write_trace(directory, [
        (chat_request("c1", [user], "2026-01-01T00:00:01Z"), _sse_response("c1", body)),
        (chat_request("c2", [user, {"role": "assistant", "content": "Reading?"},
                             {"role": "user", "content": "and?"}], "2026-01-01T00:00:02Z"),
         _sse_response("c2", body)),
    ])  # fmt: skip
    _, v = run_verdict(os.path.join(directory, "v"), [directory], chat, "window", "conformance")
    c03 = next(r for r in v["checks"] if r["id"] == "C-03")
    results.append({"case": "X-25", "ok": c03["status"] == "FAIL"})
    return results


EDIT_SCHEMA = {"type": "object", "required": ["file_path"],
               "properties": {"file_path": {"type": "string"},
                              "replace_all": {"type": "boolean", "default": False}}}  # fmt: skip
BASH_SCHEMA = {"type": "object", "properties": {"command": {"type": "string"}},
               "required": ["command"]}  # fmt: skip


def rewrite_cases(work, anth):
    """C-03 for calls a harness sends back changed as its profile declares (echo.tool_input)."""
    results = []
    user = {"role": "user", "content": "go"}
    tools = [
        {"name": "Edit", "input_schema": EDIT_SCHEMA},
        {"name": "Bash", "input_schema": BASH_SCHEMA},
    ]
    rewriting = write_profile(os.path.join(work, "x-rewrites.yaml"), dict(
        CONTROL_PROFILES["anthropic"], echo={"text": "exact", "reasoning": "exact",
                                             "tool_input": ["schema_defaults", "cd_cwd_prefix"]}))  # fmt: skip

    def request(trace_id, messages, at):
        return {"event": "request", "trace_id": trace_id, "route": "/v1/messages", "recorded_at": at,
                "headers": [["x-claude-code-session-id", "s1"]],
                "body": {"model": "x", "max_tokens": 64, "stream": True, "messages": messages,
                         "tools": tools}}  # fmt: skip

    # (case, tool, delivered arguments, echoed arguments, profile, C-03 status)
    variants = (
        # X-39: an argument filled in at its declared default, declared by the profile: equal.
        ("X-39", "Edit", '{"file_path": "a"}', {"file_path": "a", "replace_all": False}, rewriting, "PASS"),
        # X-40: the same echo where the profile declares no rewrite: it differs.
        ("X-40", "Edit", '{"file_path": "a"}', {"file_path": "a", "replace_all": False}, anth, "FAIL"),
        # X-41: an added argument that is not its declared default is a change.
        ("X-41", "Edit", '{"file_path": "a"}', {"file_path": "a", "replace_all": True}, rewriting, "FAIL"),
        # X-42: a leading `cd <dir> &&` cut from what we delivered, declared: equal.
        ("X-42", "Bash", '{"command": "cd /w && ls"}', {"command": "ls"}, rewriting, "PASS"),
        # X-43: the cut is one-sided: a `cd` the harness added is a change.
        ("X-43", "Bash", '{"command": "ls"}', {"command": "cd /w && ls"}, rewriting, "FAIL"),
        # X-44: and so is a `cd` into another directory.
        ("X-44", "Bash", '{"command": "cd /w && ls"}', {"command": "cd /v && ls"}, rewriting, "FAIL"),
    )  # fmt: skip
    for case, name, delivered, echoed, profile, status in variants:
        directory = os.path.join(work, case.lower())
        reply = _anthropic_sse(
            [{"type": "tool_use", "id": "toolu_1", "name": name, "json": delivered}]
        )
        turn2 = [user, {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": name, "input": echoed}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": "ok"}]}]  # fmt: skip
        _write_trace(directory, [
            (request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", reply)),
            (request("t2", turn2, "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
        ])  # fmt: skip
        _, v = run_verdict(
            os.path.join(directory, "v"), [directory], profile, "window", "conformance"
        )
        c03 = next(r for r in v["checks"] if r["id"] == "C-03")
        ok = c03["status"] == status and c03["echo_rewritten"] == (1 if status == "PASS" else 0)
        results.append({"case": case, "ok": ok})
    # X-45: a rewrite the validator does not know fails the run instead of passing silently.
    unknown = write_profile(os.path.join(work, "x45.yaml"), dict(
        CONTROL_PROFILES["anthropic"], echo={"text": "exact", "tool_input": ["strip_everything"]}))  # fmt: skip
    try:
        pipeline_mod.Pipeline(
            profiles_mod.load_profile(unknown), profiles_mod.load_profile(GLM_PROFILE)
        )
        refused = False
    except ValueError:
        refused = True
    results.append({"case": "X-45", "ok": refused})
    return results


K3_PROFILE = os.path.join(SKILL, "profiles", "model", "kimi-k3.yaml")


def usage_cases(work, anth):
    """C-14 against the server's own counts: its prompt count (trace) and the engine's (raw)."""
    results = []
    user = {"role": "user", "content": "go"}
    reply = _anthropic_sse([{"type": "text", "text": "ok"}], "end_turn")  # bills input 1, output 1
    for case, prompt, model, expected in (
        ("X-26", 99, GLM_PROFILE, True),
        ("X-27", 4, K3_PROFILE, False),
    ):
        directory = os.path.join(work, case.lower())
        record = dict(
            _sse_response("t1", reply), prompt_tokens={"count": prompt, "source": "control"}
        )
        _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), record)])
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                           model=model)  # fmt: skip
        c14 = next(r for r in v["checks"] if r["id"] == "C-14")
        results.append({"case": case, "ok": ("USG-07" in v["tripped"]) == expected
                        and c14["usage_vs_trace_prompt"] == 1})  # fmt: skip
    directory = os.path.join(work, "x-28")
    _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
                              _sse_response("t1", reply, engine_id="e1"))])  # fmt: skip
    with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
        fh.write(
            json.dumps({"disagg_request_id": "e1", "text": "x</think>ok", "num_output_tokens": 7})
            + "\n"
        )
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                       raw=[directory])  # fmt: skip
    results.append({"case": "X-28", "ok": "USG-06" in v["tripped"] and "C-14" in v["tripped"]})
    # X-29 / X-30: the turn after an echoed call billed less input; allowed only when the client
    # asked the server to clear history (context_management).
    for case, clearing in (("X-29", False), ("X-30", True)):
        directory = os.path.join(work, case.lower())
        call = [{"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}]
        first = _anthropic_sse(call).replace('"input_tokens": 1', '"input_tokens": 50')
        turn2 = [
            user,
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "a"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
            },
        ]
        second = _anth_request("t2", turn2, "2026-01-01T00:00:02Z")
        if clearing:
            second["body"]["context_management"] = {"edits": [{"type": "clear_tool_uses_20250919"}]}
        _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", first)),
                                 (second, _sse_response("t2", _anthropic_sse([{"type": "text", "text": "done"}],
                                                                             "end_turn")))])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
        results.append({"case": case, "ok": ("USG-08" in v["tripped"]) != clearing})
    return results


def diff_runs(directory, system="sys", billed=7, text="ok", blocks=None):
    """One turn over Chat and over Messages: the same conversation, configurable server answers."""
    user = {"role": "user", "content": "go"}
    chunks = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}},
    ]
    chat_sse = (
        "".join(
            "data: "
            + json.dumps(dict(c, id="x", object="chat.completion.chunk", model="m", created=1))
            + "\n\n"
            for c in chunks
        )
        + "data: [DONE]\n\n"
    )
    chat = {
        "event": "request",
        "trace_id": "c1",
        "route": "/v1/chat/completions",
        "recorded_at": "2026-01-01T00:00:01Z",
        "body": {
            "model": "m",
            "stream": True,
            "messages": [{"role": "system", "content": "sys"}, user],
            "tools": [
                {"type": "function", "function": {"name": "read", "parameters": READ_SCHEMA}}
            ],
        },
    }
    anth_sse = _anthropic_sse(blocks or [{"type": "text", "text": text}], "end_turn").replace(
        '"input_tokens": 1', f'"input_tokens": {billed}'
    )
    anth = {"event": "request", "trace_id": "a1", "route": "/v1/messages", "recorded_at": "2026-01-01T00:00:01Z",
            "body": {"model": "m", "stream": True, "max_tokens": 64, "system": system, "messages": [user],
                     "tools": [{"name": "read", "input_schema": READ_SCHEMA}]}}  # fmt: skip
    _write_trace(os.path.join(directory, "chat"), [(chat, _sse_response("c1", chat_sse))])
    _write_trace(os.path.join(directory, "anth"), [(anth, _sse_response("a1", anth_sse))])
    return os.path.join(directory, "chat"), os.path.join(directory, "anth")


def diff_runs_two_turns(directory, first=("r", "r"), replay=("r", "r"), billed=(9, 9)):
    """Two turns over Chat and over Messages: turn 1 answers with reasoning, turn 2 replays it.

    first: the reasoning each path generated on turn 1; replay: the reasoning each request of
    turn 2 sends back; billed: each path's input bill on turn 2 (turn 1 bills 7 on both).
    """
    user, more = {"role": "user", "content": "go"}, {"role": "user", "content": "more"}

    def chat_sse(reasoning, prompt):
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "reasoning_content": reasoning}}
                ]
            },
            {"choices": [{"index": 0, "delta": {"content": "ok"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": 2,
                    "total_tokens": prompt + 2,
                },
            },
        ]
        frames = [
            json.dumps(dict(c, id="x", object="chat.completion.chunk", model="m", created=1))
            for c in chunks
        ]
        return "".join(f"data: {frame}\n\n" for frame in frames) + "data: [DONE]\n\n"

    tools = [{"type": "function", "function": {"name": "read", "parameters": READ_SCHEMA}}]

    def chat_request(trace_id, messages, at):
        body = {
            "model": "m",
            "stream": True,
            "messages": [{"role": "system", "content": "sys"}] + messages,
            "tools": tools,
        }
        return {"event": "request", "trace_id": trace_id, "route": "/v1/chat/completions",
                "recorded_at": at, "body": body}  # fmt: skip

    def anth_request(trace_id, messages, at):
        body = {
            "model": "m",
            "stream": True,
            "max_tokens": 64,
            "system": "sys",
            "messages": messages,
            "tools": [{"name": "read", "input_schema": READ_SCHEMA}],
        }
        return {"event": "request", "trace_id": trace_id, "route": "/v1/messages",
                "recorded_at": at, "body": body}  # fmt: skip

    def anth_sse(reasoning, prompt):
        blocks = [
            {"type": "thinking", "text": reasoning, "signature": "c2ln"},
            {"type": "text", "text": "ok"},
        ]
        sse = _anthropic_sse(blocks, "end_turn")
        return sse.replace('"input_tokens": 1', f'"input_tokens": {prompt}')

    chat_history = [
        user,
        {"role": "assistant", "content": "ok", "reasoning_content": replay[0]},
        more,
    ]
    thinking = {"type": "thinking", "thinking": replay[1], "signature": "c2ln"}
    anth_history = [
        user,
        {"role": "assistant", "content": [thinking, {"type": "text", "text": "ok"}]},
        more,
    ]
    t1, t2 = "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z"
    chat_pairs = [
        (chat_request("c1", [user], t1), _sse_response("c1", chat_sse(first[0], 7))),
        (chat_request("c2", chat_history, t2), _sse_response("c2", chat_sse(first[0], billed[0]))),
    ]
    anth_pairs = [
        (anth_request("a1", [user], t1), _sse_response("a1", anth_sse(first[1], 7))),
        (anth_request("a2", anth_history, t2), _sse_response("a2", anth_sse(first[1], billed[1]))),
    ]
    _write_trace(os.path.join(directory, "chat"), chat_pairs)
    _write_trace(os.path.join(directory, "anth"), anth_pairs)
    return os.path.join(directory, "chat"), os.path.join(directory, "anth")


def protocol_diff_cases(work):
    """protocol_diff.py: an equal pair passes; a server-side difference fails; a harness one does not."""
    import protocol_diff

    results = []
    for case, kwargs, code, attribution in (("X-31", {}, 0, "equal"),
                                            ("X-32", {"billed": 9}, 1, "server_render"),
                                            ("X-33", {"system": "other"}, 0, "harness"),
                                            ("X-34", {"text": "no"}, 2, "divergent_generation"),
                                            ("X-35", {"blocks": [{"type": "thinking", "text": "ok"}]},
                                             1, "server_output")):  # fmt: skip
        directory = os.path.join(work, case.lower())
        chat, anth = diff_runs(directory, **kwargs)
        out = os.path.join(directory, "diff.json")
        rc = protocol_diff.main(["--out", out, f"chat={chat}", f"messages={anth}"])
        with open(out) as fh:
            row = json.load(fh)["rows"][0]["messages"]
        results.append({"case": case, "ok": rc == code and row["attribution"] == attribution})
    # X-36: turn 1 diverged in its reasoning; turn 2 replays each path's own reasoning and is
    # billed differently. The bill follows the replayed reasoning: after_divergence, not the server.
    # X-37: turn 1 agreed, but the harness replays other reasoning over one path: harness.
    # X-38: everything replayed agrees and the bills differ: the server rendered it differently.
    for case, kwargs, code, attribution in (
            ("X-36", {"first": ("ra", "rb"), "replay": ("ra", "rb"), "billed": (9, 10)}, 2, "after_divergence"),
            ("X-37", {"replay": ("r", "other"), "billed": (9, 11)}, 0, "harness"),
            ("X-38", {"billed": (9, 12)}, 1, "server_render")):  # fmt: skip
        directory = os.path.join(work, case.lower())
        chat, anth = diff_runs_two_turns(directory, **kwargs)
        out = os.path.join(directory, "diff.json")
        rc = protocol_diff.main(["--out", out, f"chat={chat}", f"messages={anth}"])
        with open(out) as fh:
            rows = json.load(fh)["rows"]
        second = rows[1]["messages"]["attribution"] if len(rows) > 1 else None
        results.append({"case": case, "ok": rc == code and second == attribution})
    return results


def raw_cases(work, anth):
    """C-13: what was delivered against what the model wrote (GLM's native format)."""
    results = []
    user = {"role": "user", "content": "go"}
    for case, (text, blocks, single, rules, anomaly) in RAW_CASES.items():
        directory = os.path.join(work, case.lower())
        request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        if single:
            request["body"]["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        _write_trace(directory, [(request, _sse_response("t1", _anthropic_sse(blocks, stop),
                                                         engine_id="e1"))])  # fmt: skip
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(
                json.dumps({"disagg_request_id": "e1", "text": text, "finish_reason": "stop"})
                + "\n"
            )
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                           raw=[directory])  # fmt: skip
        c13 = next(r for r in v["checks"] if r["id"] == "C-13")
        tripped = set(v["tripped"])
        ok = ({t for t in tripped if t.startswith("RAW-")} == rules
              and ("C-13" in tripped) == bool(rules) and ("M-07" in tripped) == anomaly
              and c13["denominator"] == 1)  # fmt: skip
        results.append({"case": case, "ok": ok})
    # X-20: an engine id that names two raw records joins nothing (fail closed).
    directory = os.path.join(work, "x-20")
    request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
    _write_trace(directory, [(request, _sse_response("t1", _anthropic_sse(RAW_CASES["X-13"][1]),
                                                     engine_id="e1"))])  # fmt: skip
    with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
        for _ in range(2):
            fh.write(json.dumps({"disagg_request_id": "e1", "text": GLM_RAW}) + "\n")
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                       raw=[directory])  # fmt: skip
    c13 = next(r for r in v["checks"] if r["id"] == "C-13")
    results.append({"case": "X-20", "ok": c13["status"] == "BLOCKED" and c13["denominator"] == 0})
    # X-46: on an aggregated server the raw record has no orchestrator id; the reply joins it by
    # the engine request id its Anthropic message id carries (msg_<id>_<hex>).
    directory = os.path.join(work, "x-46")
    reply = _anthropic_sse(RAW_CASES["X-13"][1]).replace(
        '"id": "m"', '"id": "msg_5_0123456789abcdef"', 1
    )
    _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
                              _sse_response("t1", reply))])  # fmt: skip
    with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
        fh.write(
            json.dumps({"request_id": 5, "text": RAW_CASES["X-13"][0], "finish_reason": "stop"})
            + "\n"
        )
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                       raw=[directory])  # fmt: skip
    c13 = next(r for r in v["checks"] if r["id"] == "C-13")
    results.append({"case": "X-46", "ok": c13["status"] == "PASS" and c13["denominator"] == 1})
    # X-19: without raw output the raw view is not applicable, never PASS.
    directory = os.path.join(work, "x-13")
    _, v = run_verdict(os.path.join(directory, "v19"), [directory], anth, "window", "conformance")
    c13 = next(r for r in v["checks"] if r["id"] == "C-13")
    results.append({"case": "X-19", "ok": c13["status"] == "NOT_APPLICABLE"})
    return results


V41_PROFILE = os.path.join(SKILL, "profiles", "model", "deepseek-v41.yaml")


def pairing_cases(work, anth):
    """Pairing by call ids (C-03 texts, USG-08) and the raw view per thinking mode.

    The request a reply is held against carries its call ids, and a raw record is read in the
    mode its request asked for.
    """
    results = []

    def use(call_id, text):
        return {"role": "assistant", "content": [
            {"type": "text", "text": text},
            {"type": "tool_use", "id": call_id, "name": "Read", "input": {"file_path": "a"}}]}  # fmt: skip

    def result(call_id):
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id,
                                             "content": "ok"}]}  # fmt: skip

    def billed(sse, tokens):
        return sse.replace('"input_tokens": 1', f'"input_tokens": {tokens}', 1)

    def reply(call_id, text, tokens=1):
        return billed(_anthropic_sse([{"type": "text", "text": text},
                                      {"type": "tool_use", "id": call_id, "name": "Read",
                                       "json": '{"file_path": "a"}'}]), tokens)  # fmt: skip

    done = _anthropic_sse([{"type": "text", "text": "done"}], "end_turn")
    user = {"role": "user", "content": "go"}
    # X-62: two sub-agents of one session ask with different first messages and the same last
    # one, so both replies sit at the same position after the same message, and the second's
    # echo arrives first. Matched by call ids, both texts echo equal (matched by position, the
    # first reply was held against the second's echo: a false C-03).
    directory = os.path.join(work, "x-62")
    tail = {"role": "user", "content": "continue"}
    task_a, task_b = {"role": "user", "content": "task a"}, {"role": "user", "content": "task b"}
    _write_trace(directory, [
        (_anth_request("a1", [task_a, tail], "2026-01-01T00:00:01Z"), _sse_response("a1", reply("toolu_a", "A."))),
        (_anth_request("b1", [task_b, tail], "2026-01-01T00:00:02Z"), _sse_response("b1", reply("toolu_b", "B."))),
        (_anth_request("b2", [task_b, tail, use("toolu_b", "B."), result("toolu_b")], "2026-01-01T00:00:03Z"),
         _sse_response("b2", done)),
        (_anth_request("a2", [task_a, tail, use("toolu_a", "A."), result("toolu_a")], "2026-01-01T00:00:04Z"),
         _sse_response("a2", done)),
    ])  # fmt: skip
    _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
    c03 = next(r for r in v["checks"] if r["id"] == "C-03")
    results.append({"case": "X-62", "ok": c03["status"] == "PASS" and c03["text_echo_equal"] == 2})

    def usage_case(case, second_messages, second_tokens, context_management=None, side=False):
        directory = os.path.join(work, case.lower())
        pairs = [(_anth_request("t1", second_messages[0], "2026-01-01T00:00:01Z"),
                  _sse_response("t1", reply("toolu_1", "R.", 100)))]  # fmt: skip
        if side:
            request = _anth_request(
                "s", [{"role": "user", "content": "a title?"}], "2026-01-01T00:00:02Z"
            )
            request["body"].pop("tools")
            pairs.append((request, _sse_response("s", billed(_anthropic_sse(
                [{"type": "text", "text": "T"}], "end_turn"), 10))))  # fmt: skip
        second = _anth_request("t2", second_messages[1], "2026-01-01T00:00:03Z")
        if context_management is not None:
            second["body"]["context_management"] = context_management
        pairs.append((second, _sse_response("t2", billed(done, second_tokens))))
        _write_trace(directory, pairs)
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance")
        with open(os.path.join(directory, "v", "report.json")) as fh:
            counts = json.load(fh)["counts"]
        return v, counts

    echo = [user, use("toolu_1", "R."), result("toolu_1")]
    # X-63: a side request of the session (no tools, small) comes between a turn and the request
    # that sends its call back. USG-08 holds the turn against the latter, which grew.
    v, counts = usage_case("X-63", ([user], echo), 120, side=True)
    results.append({"case": "X-63", "ok": "USG-08" not in v["tripped"]
                    and counts.get("den:next_input_pairs") == 1})  # fmt: skip
    # X-64: the request that sends the call back replaced the history with a summary
    # (compaction): its prompt is smaller, which is the harness's doing, not a usage error.
    earlier = [user, use("toolu_0", "Z."), result("toolu_0"), {"role": "user", "content": "more"}]
    summary = {"role": "user", "content": "the conversation so far, summarized"}
    v, counts = usage_case(
        "X-64", (earlier, [summary, use("toolu_1", "R."), result("toolu_1")]), 40
    )
    results.append({"case": "X-64", "ok": "USG-08" not in v["tripped"]
                    and counts.get("C-14:next_input_shrank_by_compaction") == 1})  # fmt: skip
    # X-65: context_management that clears thinking with keep "all" removes nothing: a smaller
    # next prompt is still USG-08. X-66: clearing tool uses may shrink it.
    for case, edit, trips in (
        ("X-65", {"type": "clear_thinking_20251015", "keep": "all"}, True),
        ("X-66", {"type": "clear_tool_uses_20250919"}, False),
    ):
        v, _ = usage_case(case, ([user], echo), 50, context_management={"edits": [edit]})
        results.append({"case": case, "ok": ("USG-08" in v["tripped"]) == trips})
    # X-67: V4.1 renders a Messages request without `thinking` in chat mode, so the raw output has
    # no </think>; read from the content it is judged by C-13 (PASS) and is no M-07. X-68: the
    # same output for a request that asked for thinking is an unclosed think (M-07).
    for case, thinking, unclosed in (("X-67", None, False), ("X-68", {"type": "adaptive"}, True)):
        directory = os.path.join(work, case.lower())
        request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        if thinking is not None:
            request["body"]["thinking"] = thinking
        answer = _anthropic_sse([{"type": "text", "text": "OK"}], "end_turn")
        _write_trace(directory, [(request, _sse_response("t1", answer, engine_id="e1"))])
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(
                json.dumps({"disagg_request_id": "e1", "text": "OK", "finish_reason": "stop"})
                + "\n"
            )
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window", "conformance",
                           model=V41_PROFILE, raw=[directory])  # fmt: skip
        c13 = next(r for r in v["checks"] if r["id"] == "C-13")
        ok = ("M-07" in v["tripped"]) == unclosed and "C-13" not in v["tripped"]
        if not unclosed:
            ok = ok and c13["status"] == "PASS" and c13["denominator"] == 1
        results.append({"case": case, "ok": ok})
    return results


def applicability_cases(work, anth):
    """A check whose situation never arose tested nothing: NOT_APPLICABLE, never PASS."""
    results = []
    user = {"role": "user", "content": "go"}
    call = [{"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}]
    # X-69: no request limits calls or sets tool_choice, and nothing ran over length.
    # X-70: one request sets a named tool_choice and at most one call, and is answered by one call.
    for case, choice, expect in (
        ("X-69", None, {"C-07": "NOT_APPLICABLE", "C-08": "NOT_APPLICABLE", "H-04": "NOT_APPLICABLE"}),
        ("X-70", {"type": "tool", "name": "Read", "disable_parallel_tool_use": True},
         {"C-07": "PASS", "C-08": "PASS", "H-04": "NOT_APPLICABLE"}),
    ):  # fmt: skip
        directory = os.path.join(work, case.lower())
        request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        if choice is not None:
            request["body"]["tool_choice"] = choice
        _write_trace(directory, [(request, _sse_response("t1", _anthropic_sse(call)))])
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window")
        status = {r["id"]: r["status"] for r in v["checks"]}
        results.append({"case": case, "ok": all(status.get(k) == x for k, x in expect.items())})
    return results


def rate_unit_cases(work, anth):
    """The matrix's unit for rates: sessions, or each check's own unit (turns)."""
    results = []
    directory = os.path.join(work, "x-71")
    pairs = []
    for session in ("s1", "s2"):
        for i in range(80):
            trace_id = f"{session}-{i}"
            request = _anth_request(trace_id, [{"role": "user", "content": f"step {i}"}],
                                    f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z")  # fmt: skip
            request["headers"] = [["x-claude-code-session-id", session]]
            pairs.append((request, _sse_response(trace_id, _anthropic_sse(
                [{"type": "text", "text": "Done."}], "end_turn"))))  # fmt: skip
    _write_trace(directory, pairs)
    # X-71: 160 clean turns from 2 sessions, 2% thresholds counted per session: insufficient n.
    # X-72: the same counted per turn: PASS (0 of 160, upper bound 1.7%), sessions still reported.
    # X-73: a unit the validator does not know is not guessed: BLOCKED.
    for case, unit, expect in (("X-71", "session", "BLOCKED"), ("X-72", "turn", "PASS"),
                               ("X-73", "turns", "BLOCKED")):  # fmt: skip
        matrix = os.path.join(work, case.lower() + "-matrix.yaml")
        with open(matrix, "w") as fh:
            json.dump({"thresholds": {"model_behavior": {"markup_leak_rate": 0.02,
                                                         "empty_turn_rate": 0.02},
                                      "unit_for_rates": unit}}, fh)  # fmt: skip
        _, v = run_verdict(os.path.join(directory, case), [directory], anth, "window",
                           "model-behavior", matrix=matrix)  # fmt: skip
        rates = [r for r in v["checks"] if r["id"] in ("M-03", "M-06")]
        ok = len(rates) == 2 and all(r["status"] == expect for r in rates)
        if case == "X-71":
            ok = ok and all(r["denominator"] == 2 for r in rates)
        if case == "X-72":
            ok = ok and all(r["denominator"] == 160 and r["sessions"] == 2 for r in rates)
        if case == "X-73":
            ok = ok and all("unknown unit_for_rates" in (r["reason"] or "") for r in rates)
        results.append({"case": case, "ok": ok})
    return results


def cut_bound_cases(work):
    """R-01/R-02 against a client cut known only from below (`{at_least: s}`) or exactly."""
    results = []
    user = {"role": "user", "content": "go"}

    def profile(name, first, idle):
        data = json.loads(json.dumps(CONTROL_PROFILES["anthropic"]))
        timeouts = dict(data.get("timeouts") or {})
        timeouts.update(first_event_s=first, idle_s=idle)
        data["timeouts"] = timeouts
        return write_profile(os.path.join(work, name), data)

    bound = profile("x-bound.yaml", {"at_least": 10}, {"at_least": 30})
    exact = profile("x-exact.yaml", 10, 30)
    # X-74: first event at 0.5 s, longest silence 5 s, under bounds of 10 s and 30 s: PASS.
    # X-75: a 34.5 s silence, above the known 30 s: whether the client cuts it is unknown, BLOCKED.
    # X-76: the same silence against an exact 30 s cut: FAIL.
    # X-77: a first event at 12 s against a known 10 s: BLOCKED.
    for case, chunks, first, harness, expect in (
        ("X-74", [0.5, 1.0, 6.0], 0.5, bound, {"R-01": "PASS", "R-02": "PASS"}),
        ("X-75", [0.5, 6.0, 40.5], 0.5, bound, {"R-01": "PASS", "R-02": "BLOCKED"}),
        ("X-76", [0.5, 6.0, 40.5], 0.5, exact, {"R-01": "PASS", "R-02": "FAIL"}),
        ("X-77", [12.0, 12.5], 12.0, bound, {"R-01": "BLOCKED", "R-02": "PASS"}),
    ):  # fmt: skip
        directory = os.path.join(work, case.lower())
        response = _sse_response(
            "t1", _anthropic_sse([{"type": "text", "text": "Done."}], "end_turn")
        )
        response["timing"] = {"headers_s": 0.01, "first_byte_s": first,
                              "chunks": [[t, 1] for t in chunks]}  # fmt: skip
        _write_trace(directory, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), response)])
        _, v = run_verdict(os.path.join(directory, "v"), [directory], harness, "window", "runtime")
        status = {r["id"]: r["status"] for r in v["checks"]}
        results.append({"case": case, "ok": all(status.get(k) == x for k, x in expect.items())})
    return results


def outcome_and_signal_cases(work, anth):
    """Outcomes, applicability, signatures and M-02 on raw text (trial 6's full tier).

    Rated outcomes the profile cannot see, checks with nothing to compare, signatures by value,
    and M-02 on the raw text the model wrote.
    """
    results = []
    user = {"role": "user", "content": "go"}
    call = [{"type": "tool_use", "id": "toolu_1", "name": "Read", "json": '{"file_path": "a"}'}]
    rejected = ("The user doesn't want to proceed with this tool use. The tool use was rejected (eg. "
                "if it was a file edit, the new_string was NOT written to the file).")  # fmt: skip
    matrix = os.path.join(work, "x-78-matrix.yaml")
    with open(matrix, "w") as fh:
        json.dump({"thresholds": {"harness_outcome": {"aborted_rate": 0.02,
                                                      "arguments_rejected_rate": 0.02},
                                  "model_behavior": {"degenerate_rate": 0.02},
                                  "unit_for_rates": "turn"}}, fh)  # fmt: skip

    def answered(name, result, is_error=True):
        turn2 = [user, {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": result, "is_error": is_error}]}]  # fmt: skip
        directory = os.path.join(work, name)
        _write_trace(directory, [
            (_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
             _sse_response("t1", _anthropic_sse(call))),
            (_anth_request("t2", turn2, "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse([{"type": "text", "text": "ok"}], "end_turn"))),
        ])  # fmt: skip
        return directory

    def profile(name, outcomes):
        data = json.loads(json.dumps(CONTROL_PROFILES["anthropic"]))
        data["outcomes"] = outcomes
        return write_profile(os.path.join(work, name), data)

    # X-78: Claude Code's rejection of a call is an abort, a rated outcome, not a tool error.
    d = answered("x-78", rejected)
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "harness-outcome", matrix=matrix)
    h = {r["id"]: r for r in v["checks"]}
    results.append({"case": "X-78", "ok": h["H-01:aborted"]["count"] == 1
                    and h["H-01:aborted"]["status"] != "PASS"})  # fmt: skip
    # X-79: a profile that does not name the rated `aborted` outcome cannot count one: BLOCKED.
    blind = profile("x-79.yaml", {"unsupported": {"contains": "No such tool available"},
                                  "arguments_rejected": {"contains": "InputValidationError"},
                                  "error": {"is_error": True}})  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v79"), [d], blind, "window", "harness-outcome",
                       matrix=matrix)  # fmt: skip
    h = {r["id"]: r for r in v["checks"]}
    results.append({"case": "X-79", "ok": h["H-01:aborted"]["status"] == "BLOCKED"
                    and "declares no `aborted`" in (h["H-01:aborted"]["reason"] or "")
                    and v["claims"]["harness-outcome"] == "BLOCKED"})  # fmt: skip
    # X-80: declared absent with a reason, it is NOT_APPLICABLE, and the claim can pass.
    absent = profile("x-80.yaml", {"aborted": {"none": "this client never cancels a call"},
                                   "arguments_rejected": {"contains": "InputValidationError"},
                                   "error": {"is_error": True}})  # fmt: skip
    d80 = answered("x-80", "done", is_error=False)
    _, v = run_verdict(os.path.join(d80, "v"), [d80], absent, "window", "harness-outcome",
                       matrix=matrix)  # fmt: skip
    h = {r["id"]: r for r in v["checks"]}
    results.append({"case": "X-80", "ok": h["H-01:aborted"]["status"] == "NOT_APPLICABLE"
                    and h["H-01:aborted"]["reason"] == "this client never cancels a call"})  # fmt: skip
    # X-81: Anthropic streams carry no final snapshot, so C-02 compared nothing.
    _, v = run_verdict(os.path.join(d80, "v81"), [d80], anth, "window", "conformance")
    c02 = next(r for r in v["checks"] if r["id"] == "C-02")
    results.append({"case": "X-81", "ok": c02["status"] == "NOT_APPLICABLE"})
    # X-82: a thinking signature echoed with another value is not an exact echo; the same value is.
    for case, echoed, expect in (("X-82", "c2lnQg==", "FAIL"), ("X-82b", "c2lnQQ==", "PASS")):
        delivered = [{"type": "thinking", "text": "Plan.", "signature": "c2lnQQ=="}] + call
        turn2 = [user, {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "Plan.", "signature": echoed},
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": "ok"}]}]  # fmt: skip
        directory = os.path.join(work, case.lower())
        _write_trace(directory, [
            (_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
             _sse_response("t1", _anthropic_sse(delivered))),
            (_anth_request("t2", turn2, "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse([{"type": "text", "text": "ok"}], "end_turn"))),
        ])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window",
                           "conformance")  # fmt: skip
        results.append({"case": case, "ok": v["claims"]["conformance"] == expect})
    # X-83: with raw output, M-02 reads what the model wrote: a run of pad tokens the server kept
    # out of the delivered text is found, in a reply whose raw text kept special tokens.
    # X-84: a raw text without special tokens (the server dropped them) cannot show the run: the
    # reply is not counted, and with no reply left M-02 is BLOCKED.
    pad, eos = "<\uff5c\u2581pad\u2581\uff5c>", "<\uff5cend\u2581of\u2581sentence\uff5c>"
    for case, text, count, denominator, status in (
        ("X-83", "</think>Done." + pad * 25 + eos, 1, 1, None),
        ("X-84", "</think>Done.", 0, 0, "BLOCKED"),
    ):  # fmt: skip
        directory = os.path.join(work, case.lower())
        request = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        request["body"]["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        _write_trace(directory, [(request, _sse_response("t1", _anthropic_sse(
            [{"type": "thinking", "text": ""}, {"type": "text", "text": "Done."}], "end_turn"),
            engine_id="e1"))])  # fmt: skip
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(json.dumps({"disagg_request_id": "e1", "text": text}) + "\n")
        _, v = run_verdict(os.path.join(directory, "v"), [directory], anth, "window",
                           "model-behavior", model=V41_PROFILE, matrix=matrix,
                           raw=[directory])  # fmt: skip
        m02 = next(r for r in v["checks"] if r["id"] == "M-02")
        ok = m02["count"] == count and m02["denominator"] == denominator
        if status:
            ok = ok and m02["status"] == status and m02.get("raw_unobservable") == 1
        results.append({"case": case, "ok": ok})
    return results


def responses_reasoning_echo_cases(work):
    """C-03 on Responses reasoning items: echoed by id, the carrier compared by value.

    The Codex CLI asks for ``encrypted_content`` and sends each reasoning item back with its id;
    what it sends must be the carrier we delivered.
    """
    results = []
    carrier = "trtllm-reasoning-v1:UGxhbi4="
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": [], "status": "completed",
                 "content": [{"type": "reasoning_text", "text": "Plan."}],
                 "encrypted_content": carrier}  # fmt: skip
    call = {"id": "fc_1", "type": "function_call", "name": "wait", "call_id": "c1",
            "arguments": '{"ms": 1}', "status": "completed"}  # fmt: skip
    tools = [{"type": "function", "name": "wait", "parameters": {"type": "object"}}]

    def sse(items):
        frames = [("response.created", {"response": {"id": "r", "model": "x", "status": "in_progress",
                                                     "output": []}})]  # fmt: skip
        for index, item in enumerate(items):
            frames += [("response.output_item.added", {"output_index": index, "item": item}),
                       ("response.output_item.done", {"output_index": index, "item": item})]  # fmt: skip
        frames.append(("response.completed", {"response": {
            "id": "r", "model": "x", "status": "completed", "output": items,
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}}))  # fmt: skip
        return "".join(f"event: {n}\ndata: {json.dumps(dict(d, type=n, sequence_number=i))}\n\n"
                       for i, (n, d) in enumerate(frames))  # fmt: skip

    def request(trace_id, items, at):
        return {"event": "request", "trace_id": trace_id, "route": "/v1/responses", "recorded_at": at,
                "body": {"model": "x", "stream": True, "tools": tools, "store": False,
                         "include": ["reasoning.encrypted_content"], "prompt_cache_key": "s1",
                         "client_metadata": {"thread_id": "th"}, "input": items}}  # fmt: skip

    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]}
    output = {"type": "function_call_output", "call_id": "c1", "output": "ok"}
    echoed = {k: v for k, v in reasoning.items() if k != "status"}
    exact = write_profile(os.path.join(work, "x85.yaml"), dict(
        CONTROL_PROFILES["responses"], echo={"text": "exact", "reasoning": "exact"}))  # fmt: skip
    todo = write_profile(os.path.join(work, "x88.yaml"), dict(
        CONTROL_PROFILES["responses"], echo={"text": "exact", "reasoning": "TODO"}))  # fmt: skip
    # X-85: the carrier comes back as delivered. X-86: another value. X-87: without a carrier.
    # X-88: the profile does not say how reasoning is echoed: not compared, so BLOCKED.
    # X-89: the next request leaves the reasoning item out: counted as missing, not a failure.
    cases = (
        ("X-85", [echoed], exact, "PASS", "reasoning_echo_equal"),
        ("X-86", [dict(echoed, encrypted_content=carrier[:-4] + "QQ==")], exact, "FAIL", None),
        ("X-87", [{k: v for k, v in echoed.items() if k != "encrypted_content"}], exact, "FAIL", None),
        ("X-88", [echoed], todo, "BLOCKED", "reasoning_echo_unchecked"),
        ("X-89", [], exact, "PASS", "reasoning_echo_missing"),
    )  # fmt: skip
    for case, replayed, profile, status, counter in cases:
        directory = os.path.join(work, case.lower())
        second = [user] + replayed + [{k: v for k, v in call.items() if k not in ("id", "status")},
                                      output]  # fmt: skip
        _write_trace(directory, [
            (request("r1", [user], "2026-01-01T00:00:01Z"), _sse_response("r1", sse([reasoning, call]))),
            (request("r2", second, "2026-01-01T00:00:02Z"), _sse_response("r2", sse([
                {"id": "msg_2", "type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "done", "annotations": []}]}]))),
        ])  # fmt: skip
        _, v = run_verdict(os.path.join(directory, "v"), [directory], profile, "window",
                           "conformance")  # fmt: skip
        c03 = next(r for r in v["checks"] if r["id"] == "C-03")
        ok = c03["status"] == status
        if counter:
            ok = ok and c03.get(counter) == 1
        if status == "FAIL":
            with open(os.path.join(directory, "v", "report.json")) as fh:
                fields = {e.get("field") for e in json.load(fh)["examples"].get("C-03", [])}
            ok = ok and "reasoning" in fields
        results.append({"case": case, "ok": ok})
    # X-90: a Responses reply names its engine request only in a header (its id is a random resp_
    # id); recorded as engine_request_id, it joins the reply to its raw record, and C-13 reads it.
    # X-90b: without that field nothing joins, and C-13 is BLOCKED.
    eos = "<\uff5cend\u2581of\u2581sentence\uff5c>"
    for case, engine_id, status in (("X-90", "7", "PASS"), ("X-90b", None, "BLOCKED")):
        directory = os.path.join(work, case.lower())
        items = [{"id": "rs_9", "type": "reasoning", "summary": [], "status": "completed",
                  "content": [{"type": "reasoning_text", "text": "Plan."}]},
                 {"id": "msg_9", "type": "message", "role": "assistant", "status": "completed",
                  "content": [{"type": "output_text", "text": "Done.", "annotations": []}]}]  # fmt: skip
        reply = _sse_response("r1", sse(items))
        if engine_id is not None:
            reply["engine_request_id"] = engine_id
        req = request("r1", [user], "2026-01-01T00:00:01Z")
        req["body"]["reasoning"] = {"summary": "auto"}
        _write_trace(directory, [(req, reply)])
        with open(os.path.join(directory, "raw-x.jsonl"), "w") as fh:
            fh.write(json.dumps({"request_id": "7", "text": "Plan.</think>Done." + eos}) + "\n")
        _, v = run_verdict(os.path.join(directory, "v"), [directory], exact, "window",
                           "conformance", model=V41_PROFILE, raw=[directory])  # fmt: skip
        c13 = next(r for r in v["checks"] if r["id"] == "C-13")
        results.append({"case": case, "ok": c13["status"] == status
                        and (status != "PASS" or c13["denominator"] == 1)})  # fmt: skip
    return results


def cross_cases(work):
    results = []
    anth = write_profile(os.path.join(work, "x-anthropic.yaml"), CONTROL_PROFILES["anthropic"])
    codex = write_profile(os.path.join(work, "x-responses.yaml"), CONTROL_PROFILES["responses"])
    user = {"role": "user", "content": "go"}

    def two_turn(name, echoed_input):
        delivered = _anthropic_sse([{"type": "tool_use", "id": "toolu_1", "name": "Read",
                                     "json": '{"file_path": "a", "n": 1}'}])  # fmt: skip
        turn2 = [user, {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": echoed_input}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                          "content": "ok"}]}]  # fmt: skip
        directory = os.path.join(work, name)
        _write_trace(directory, [
            (_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", delivered)),
            (_anth_request("t2", turn2, "2026-01-01T00:00:02Z"),
             _sse_response("t2", _anthropic_sse([{"type": "text", "text": "done"}], "end_turn"))),
        ])  # fmt: skip
        return directory

    # X-01: the harness echoes a different payload under our call id.
    d = two_turn("x01", {"file_path": "b", "n": 1})
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    results.append(
        {"case": "X-01", "ok": "C-03" in v["tripped"] and v["claims"]["conformance"] == "FAIL"}
    )
    # X-02: lossless re-serialization (other key order and spacing) is not a difference.
    d = two_turn("x02", {"n": 1, "file_path": "a"})
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    results.append(
        {"case": "X-02", "ok": "C-03" not in v["tripped"] and v["claims"]["conformance"] == "PASS"}
    )
    # X-03: a call still pending at the cutoff (no next turn) is not a failure.
    d = os.path.join(work, "x03")
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", _anthropic_sse(
        [{"type": "tool_use", "id": "toolu_9", "name": "Read", "json": '{"file_path": "a"}'}])))])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    with open(os.path.join(d, "v", "report.json")) as fh:
        report = json.load(fh)
    results.append({"case": "X-03", "ok": report["outcome_classes"].get("pending") == 1
                    and "I-08:unanswered" not in report["counts"]})  # fmt: skip
    # X-04: Codex outcome strings are classified from the next request.
    d = os.path.join(work, "x04")
    sse = "".join(f"event: {n}\ndata: {json.dumps(dict(o, type=n))}\n\n" for n, o in [
        ("response.created", {"response": {"id": "r", "model": "x", "status": "in_progress", "output": []},
                              "sequence_number": 0}),
        ("response.output_item.added", {"output_index": 0, "item": {"id": "fc1", "type": "function_call",
                                                                    "arguments": ""}, "sequence_number": 1}),
        ("response.output_item.done", {"output_index": 0, "item": {"id": "fc1", "type": "function_call", "name": "wait",
                                                "call_id": "c1", "arguments": '{"ms": 1}'}, "sequence_number": 2}),
        ("response.completed", {"response": {"id": "r", "model": "x", "status": "completed",
                                             "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                                             "output": [
            {"id": "fc1", "type": "function_call", "name": "wait", "call_id": "c1",
             "arguments": '{"ms": 1}', "status": "completed"}]}, "sequence_number": 3})])  # fmt: skip
    tools = [{"type": "function", "name": "wait", "parameters": {"type": "object"}}]
    meta = {"thread_id": "th"}
    first = {"event": "request", "trace_id": "r1", "route": "/v1/responses", "recorded_at": "2026-01-01T00:00:01Z",
             "body": {"model": "x", "stream": True, "tools": tools, "client_metadata": meta, "input": []}}  # fmt: skip
    second = {"event": "request", "trace_id": "r2", "route": "/v1/responses", "recorded_at": "2026-01-01T00:00:02Z",
              "body": {"model": "x", "stream": True, "tools": tools, "client_metadata": meta, "input": [
                  {"type": "function_call", "call_id": "c1", "name": "wait", "arguments": '{"ms": 1}'},
                  {"type": "function_call_output", "call_id": "c1", "output": "aborted"}]}}  # fmt: skip
    _write_trace(d, [(first, _sse_response("r1", sse)), (second, None)])
    _, v = run_verdict(os.path.join(d, "v"), [d], codex, "window", "conformance")
    with open(os.path.join(d, "v", "report.json")) as fh:
        report = json.load(fh)
    results.append({"case": "X-04", "ok": report["outcome_classes"].get("aborted") == 1})
    # X-05: a harness profile without measured outcomes blocks the harness-outcome claim.
    unknown = write_profile(
        os.path.join(work, "x05.yaml"), dict(CONTROL_PROFILES["responses"], outcomes="TODO")
    )
    _, v = run_verdict(os.path.join(d, "v5"), [d], unknown, "window", "harness-outcome")
    results.append({"case": "X-05", "ok": v["claims"]["harness-outcome"] == "BLOCKED"
                    and any("outcomes are TODO" in b for b in v["claim_blockers"]["harness-outcome"])})  # fmt: skip
    # X-06: a completed reply with neither text nor a call is a vacuous turn.
    d = os.path.join(work, "x06")
    blank = _anthropic_sse([{"type": "text", "text": "  \n"}], "end_turn")
    _write_trace(
        d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", blank))]
    )
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    results.append({"case": "X-06", "ok": "M-06" in v["tripped"]})
    # X-08: a client disconnect is not a server wire failure; H-03 attributes it.
    d = os.path.join(work, "x08")
    cut = _anthropic_sse([{"type": "text", "text": "partial"}]).split("event: message_delta")[0]
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"),
                      _sse_response("t1", cut, status="client_disconnected"))])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    with open(os.path.join(d, "v", "report.json")) as fh:
        report = json.load(fh)
    results.append({"case": "X-08", "ok": "C-04" not in v["tripped"]
                    and report["counts"].get("H-03:client_disconnected") == 1})  # fmt: skip
    # X-50: the recorder's own 502 while the server was down, and a stream the recorder lost
    # mid-reply. Neither is the server's reply to judge: R-06 counts both, C-01 and C-04 do not.
    d = os.path.join(work, "x50")
    gone = {"event": "response", "trace_id": "t1", "status": "upstream_error", "http_status": 502,
            "response": {"kind": "text", "body": "ServerDisconnectedError"}}  # fmt: skip
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), gone),
                     (_anth_request("t2", [user], "2026-01-01T00:00:02Z"),
                      _sse_response("t2", cut, status="upstream_error"))])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "smoke", "conformance,runtime")
    checks = {r["id"]: r for r in v["checks"]}
    results.append({"case": "X-50", "ok": checks["C-01"]["status"] == "PASS"
                    and checks["C-04"]["status"] == "PASS" and checks["R-06"]["status"] == "FAIL"
                    and checks["R-06"]["count"] == 2})  # fmt: skip
    # X-51: a request with no response record (the capture lost it): inventory is BLOCKED.
    # X-52: the recorder stopped before the reply: pending at the capture's end, not judged, and
    # inventory is not blocked by it.
    done = _sse_response("t1", _anthropic_sse([{"type": "text", "text": "ok"}], "end_turn"))
    stopped = {"event": "response", "trace_id": "t2", "status": "recorder_shutdown",
               "http_status": None, "response": {"kind": "none", "body": None}}  # fmt: skip
    for case, second in (("X-51", None), ("X-52", stopped)):
        d = os.path.join(work, case.lower())
        _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), done),
                         (_anth_request("t2", [user], "2026-01-01T00:00:02Z"), second)])  # fmt: skip
        _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "inventory,conformance")
        with open(os.path.join(d, "v", "report.json")) as fh:
            counts = json.load(fh)["counts"]
        lost = counts.get("I-01:request_without_response", 0)
        if case == "X-51":
            ok = lost == 1 and v["claims"]["inventory"] == "BLOCKED"
        else:
            ok = (lost == 0 and counts.get("inventory:pending_at_capture_end") == 1
                  and v["claims"]["inventory"] == "PASS" and "C-04" not in v["tripped"])  # fmt: skip
        results.append({"case": case, "ok": ok})
    # X-53: the server rejects a capability with the wording the harness profile declares; the
    # harness retries without it. C-16 is flagged, and conformance still passes.
    # X-54: the same rejection with a wording the profile does not declare: C-16 FAIL.
    retrying = write_profile(os.path.join(work, "x53.yaml"), dict(CONTROL_PROFILES["anthropic"], errors=dict(
        CONTROL_PROFILES["anthropic"].get("errors") or {}, capability_retries=[
            {"field": "thinking.display", "wording": "Unknown parameter: ... display"}])))  # fmt: skip
    for case, message, status in (
        ("X-53", "Unknown parameter: 'thinking.adaptive.display'", "FLAGGED"),
        ("X-54", "thinking.display is not supported", "FAIL"),
    ):
        d = os.path.join(work, case.lower())
        rejected = {"event": "response", "trace_id": "t1", "status": "completed", "http_status": 400,
                    "response": {"kind": "json", "body": {"type": "error", "error": {
                        "type": "invalid_request_error", "message": message}}}}  # fmt: skip
        asked = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
        asked["body"]["thinking"] = {"type": "adaptive", "display": "updates"}
        retried = _anth_request("t2", [user], "2026-01-01T00:00:02Z")  # the same, without the field
        retried["body"]["thinking"] = {"type": "adaptive"}
        _write_trace(d, [(asked, rejected), (retried, done | {"trace_id": "t2"})])
        _, v = run_verdict(os.path.join(d, "v"), [d], retrying, "window", "conformance")
        c16 = next(r for r in v["checks"] if r["id"] == "C-16")
        ok = c16["status"] == status and "H-02" not in v["tripped"]
        if status == "FLAGGED":
            ok = ok and c16["declared_fields"] == {"thinking.display": 1} and "C-16" in v["tripped"] \
                and v["claims"]["conformance"] == "PASS"  # fmt: skip
        results.append({"case": case, "ok": ok})
    # X-55: an operator probe (a self-test through the recorder) is not the cell's traffic.
    d = os.path.join(work, "x55")
    probe = _anth_request("t1", [user], "2026-01-01T00:00:01Z")
    probe["headers"] = [["x-mh-probe", "1"]]
    _write_trace(d, [(probe, done), (_anth_request("t2", [user], "2026-01-01T00:00:02Z"),
                                     done | {"trace_id": "t2"})])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "inventory,conformance")
    with open(os.path.join(d, "v", "report.json")) as fh:
        counts = json.load(fh)["counts"]
    results.append({"case": "X-55", "ok": counts.get("inventory:operator_probes") == 1
                    and counts.get("den:replies") == 1 and v["claims"]["inventory"] == "PASS"})  # fmt: skip
    # X-56: count_tokens rejected with a wording the profile does not declare: C-16 FAIL.
    d = os.path.join(work, "x56")
    counting = {"event": "request", "trace_id": "c1", "route": "/v1/messages/count_tokens",
                "recorded_at": "2026-01-01T00:00:01Z", "headers": [["x-claude-code-session-id", "s1"]],
                "body": {"model": "x", "messages": [user]}}  # fmt: skip
    refused = {"event": "response", "trace_id": "c1", "status": "completed", "http_status": 400,
               "response": {"kind": "json", "body": {"type": "error", "error": {
                   "type": "invalid_request_error", "message": "unknown field"}}}}  # fmt: skip
    _write_trace(d, [(counting, refused), (_anth_request("t2", [user], "2026-01-01T00:00:02Z"),
                                           done | {"trace_id": "t2"})])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    c16 = next(r for r in v["checks"] if r["id"] == "C-16")
    results.append({"case": "X-56", "ok": c16["status"] == "FAIL" and c16["count"] == 1})
    # X-57: two clients gave up before the first byte, after 35 s and after 5 s, against a 30 s
    # first-byte cutoff: the first saw the cutoff pass (R-01), the second left early.
    # X-58: each cut is attributed to the client in an H-03 check record (a flag).
    timed = write_profile(os.path.join(work, "x57.yaml"), dict(
        CONTROL_PROFILES["anthropic"], timeouts={"first_event_s": 30, "idle_s": 30}))  # fmt: skip
    d = os.path.join(work, "x57")
    gave_up = [{"event": "response", "trace_id": t, "status": "client_disconnected", "http_status": None,
                "response": {"kind": "none", "body": None},
                "timing": {"client_error": "CancelledError", "before_reply": True, "total_s": w}}
               for t, w in (("t1", 35.0), ("t2", 5.0))]  # fmt: skip
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), gave_up[0]),
                     (_anth_request("t2", [user], "2026-01-01T00:00:40Z"), gave_up[1])])  # fmt: skip
    _, v = run_verdict(os.path.join(d, "v"), [d], timed, "smoke", "runtime,harness-outcome")
    checks = {r["id"]: r for r in v["checks"]}
    with open(os.path.join(d, "v", "report.json")) as fh:
        counts = json.load(fh)["counts"]
    results.append({"case": "X-57", "ok": checks["R-01"]["status"] == "FAIL" and checks["R-01"]["count"] == 1
                    and counts.get("den:cuts_before_reply") == 2})  # fmt: skip
    results.append(
        {
            "case": "X-58",
            "ok": checks["H-03"]["status"] == "FLAGGED" and checks["H-03"]["count"] == 2,
        }
    )
    # X-59: a window's last request is answered in the next hour. Without that hour's responses
    # inventory is BLOCKED (I-01); with them (--responses) the request is answered, and a
    # response there for a request outside the window is ignored.
    d = os.path.join(work, "x59")
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:59:59Z"), None)])
    later = os.path.join(work, "x59-next")
    os.makedirs(later, exist_ok=True)
    with open(os.path.join(later, "responses-x.jsonl"), "w") as fh:
        fh.write(json.dumps(done) + "\n" + json.dumps(done | {"trace_id": "elsewhere"}) + "\n")
    _, alone = run_verdict(os.path.join(d, "v1"), [d], anth, "window", "inventory")
    _, joined = run_verdict(
        os.path.join(d, "v2"), [d], anth, "window", "inventory", responses=[later]
    )
    with open(os.path.join(d, "v2", "report.json")) as fh:
        counts = json.load(fh)["counts"]
    results.append({"case": "X-59", "ok": alone["claims"]["inventory"] == "BLOCKED"
                    and joined["claims"]["inventory"] == "PASS"
                    and counts.get("inventory:boundary_responses_joined") == 1
                    and counts.get("inventory:boundary_responses_ignored") == 1})  # fmt: skip
    # X-09: a degenerate run that reached the client: M-02 and R-04.
    d = os.path.join(work, "x09")
    runaway = _anthropic_sse([{"type": "text", "text": "[PAD]" * 25}], "end_turn")
    _write_trace(
        d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", runaway))]
    )
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    results.append({"case": "X-09", "ok": {"M-02", "R-04"} <= set(v["tripped"])})
    # X-60: a model profile that names no degenerate tokens leaves M-02 blind to the model's own
    # runaway output: BLOCKED, not PASS, even under a lenient threshold.
    # X-61: the same trace judged with a profile that names them (K3's): PASS.
    d = os.path.join(work, "x60")
    _write_trace(d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", _anthropic_sse(
        [{"type": "text", "text": "Done."}], "end_turn")))])  # fmt: skip
    lenient = os.path.join(work, "x60-matrix.yaml")
    with open(lenient, "w") as fh:
        json.dump(
            {
                "thresholds": {
                    "model_behavior": {"degenerate_rate": 0.9},
                    "unit_for_rates": "session",
                }
            },
            fh,
        )
    for case, model, status in (("X-60", GLM_PROFILE, "BLOCKED"), ("X-61", K3_PROFILE, "PASS")):
        _, v = run_verdict(os.path.join(d, case), [d], anth, "window", "model-behavior", model=model,
                           matrix=lenient)  # fmt: skip
        m02 = next(r for r in v["checks"] if r["id"] == "M-02")
        ok = m02["status"] == status and (
            status == "PASS" or "no degenerate" in (m02["reason"] or "")
        )
        results.append({"case": case, "ok": ok})
    # X-10: zero events in 20 sessions does not establish a rate below 1%.
    bounds = verdict_mod.wilson(0, 20)
    results.append({"case": "X-10", "ok": bounds is not None and bounds[1] > 0.01})
    return (results + raw_cases(work, anth) + echo_cases(work, anth) + rewrite_cases(work, anth)
            + usage_cases(work, anth) + protocol_diff_cases(work)
            + pairing_cases(work, anth) + applicability_cases(work, anth)
            + rate_unit_cases(work, anth) + cut_bound_cases(work)
            + outcome_and_signal_cases(work, anth) + responses_reasoning_echo_cases(work))  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default="responses,anthropic,chat,cross")
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    args = ap.parse_args(argv)
    only = set(args.only.split(","))
    os.makedirs(args.out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="mh-controls-")
    results, cross = [], []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        if "responses" in only:
            results += protocol_matrix("responses", RESPONSES_EXPECT, responses_body,
                                       [PY, os.path.join(HERE, "fake_responses_server.py")], work)  # fmt: skip
        if "anthropic" in only:
            results += protocol_matrix("anthropic", ANTHROPIC_EXPECT, anthropic_body,
                                       [PY, os.path.join(HERE, "fake_agent_server.py"),
                                        "--gap-seconds", "1.5"], work)  # fmt: skip
        if "chat" in only:
            results += protocol_matrix("chat", CHAT_EXPECT, chat_body,
                                       [PY, os.path.join(HERE, "fake_agent_server.py"),
                                        "--gap-seconds", "1.5"], work)  # fmt: skip
        if "cross" in only:
            cross = cross_cases(work)
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    passed = all(r["ok"] for r in results + cross) and bool(results or cross)
    receipt = {
        "schema": "trtllm-serve-model-harness/controls@1",
        "validator_tree_sha256": verdict_mod.tree_sha256(),
        "sdk_versions": verdict_mod.sdk_diff.versions(),
        "started_at": started,
        "scope": sorted(only),
        "complete_scope": only >= {"responses", "anthropic", "chat", "cross"},
        "results": results,
        "cross_cases": cross,
        "passed": passed and only >= {"responses", "anthropic", "chat", "cross"},
    }
    with open(os.path.join(args.out_dir, "controls.json"), "w") as fh:
        json.dump(receipt, fh, indent=1)
    for r in results:
        print(
            f"{'OK ' if r['ok'] else 'BAD'} {r['protocol']:9s} {r['mode']:28s} "
            f"expected={r.get('expected')} tripped={r.get('tripped')} {r.get('error', '')}"
        )
    for r in cross:
        print(f"{'OK ' if r['ok'] else 'BAD'} cross     {r['case']}")
    print(json.dumps({"passed": receipt["passed"], "cases": len(results) + len(cross),
                      "failed": [f"{r.get('protocol', 'cross')}:{r.get('mode', r.get('case'))}"
                                 for r in results + cross if not r["ok"]]}))  # fmt: skip
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
