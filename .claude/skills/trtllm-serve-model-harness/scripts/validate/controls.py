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
import verdict as verdict_mod  # noqa: E402

PY = sys.executable
GLM_PROFILE = os.path.join(SKILL, "profiles", "model", "glm-5.yaml")
CODEX_OUTCOMES = {
    "aborted": {"exact": "aborted"},
    "unsupported": {"prefix": "unsupported call:"},
    "arguments_rejected": {"prefix": "failed to parse function arguments"},
    "unavailable_in_mode": {"regex": r"is unavailable in (\w+) mode"},
}
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
        "outcomes": {"unsupported": {"contains": "No such tool available"},
                     "error": {"is_error": True}},
        "echo": {"text": "exact", "reasoning": "exact"},
        "errors": {"overlength": {"shape": "http_400", "error_type": "invalid_request_error",
                                  "wording": "prompt is too long"}},
        "timeouts": {"first_event_s": 30, "idle_s": 1.0},
    },
    "chat": {
        "id": "control-chat", "status": "measured", "protocol": "chat-completions",
        "identity": {"thread": {"header": "x-thread"}},
        "outcomes": {"error": {"prefix": "Error:"}},
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
                matrix=None, raw=None):  # fmt: skip
    argv = ["--harness", harness, "--model", model, "--population", population]
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
    # X-19: without raw output the raw view is not applicable, never PASS.
    directory = os.path.join(work, "x-13")
    _, v = run_verdict(os.path.join(directory, "v19"), [directory], anth, "window", "conformance")
    c13 = next(r for r in v["checks"] if r["id"] == "C-13")
    results.append({"case": "X-19", "ok": c13["status"] == "NOT_APPLICABLE"})
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
    pi = os.path.join(SKILL, "profiles", "harness", "pi-chat.yaml")
    _, v = run_verdict(os.path.join(d, "v5"), [d], pi, "window", "harness-outcome")
    results.append({"case": "X-05", "ok": v["claims"]["harness-outcome"] == "BLOCKED"})
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
    # X-09: a degenerate run that reached the client: M-02 and R-04.
    d = os.path.join(work, "x09")
    runaway = _anthropic_sse([{"type": "text", "text": "[PAD]" * 25}], "end_turn")
    _write_trace(
        d, [(_anth_request("t1", [user], "2026-01-01T00:00:01Z"), _sse_response("t1", runaway))]
    )
    _, v = run_verdict(os.path.join(d, "v"), [d], anth, "window", "conformance")
    results.append({"case": "X-09", "ok": {"M-02", "R-04"} <= set(v["tripped"])})
    # X-10: zero events in 20 sessions does not establish a rate below 1%.
    bounds = verdict_mod.wilson(0, 20)
    results.append({"case": "X-10", "ok": bounds is not None and bounds[1] > 0.01})
    return results + raw_cases(work, anth) + echo_cases(work, anth) + usage_cases(work, anth)


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
