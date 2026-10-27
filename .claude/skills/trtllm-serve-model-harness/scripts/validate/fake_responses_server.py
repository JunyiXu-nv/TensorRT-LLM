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
"""A fake /v1/responses server that answers with canned streams, for controls (test-only).

It exercises the replay -> validate pipeline without a GPU, and it produces the
POSITIVE controls the validator needs: every mode except ``ok`` breaks exactly
one rule that a conformance check must catch.

  ok                     reasoning + message + one call to the first declared tool
  drop_part_done         one content_part.done missing          -> unpaired_content_part
  snapshot_text_differs  completed text differs from the stream -> item_content_differs
  delta_differs          output_text.done != concatenated deltas -> delta_done_mismatch
  no_terminal            the stream stops before response.completed -> no_terminal_event
  custom_as_function     a custom tool delivered as function_call -> type_mismatch
  two_calls              two call items (fails under parallel_tool_calls=false)
  prefixed_name          the call name carries its namespace prefix -> prefixed_name_of_declared
  overlength             response.failed with context_length_exceeded (a correct answer)
  unknown_event          an event type no rule covers               -> C-11
  created_without_output response.created carries no output list  -> SDK clients cannot accumulate
  wrong_output_index     the message's events carry the reasoning item's output_index
  usage_missing          response.completed without usage           -> USG-01
  usage_total_mismatch   total_tokens is not input + output           -> USG-02
  error_string           HTTP 400 whose `error` is a bare string      -> RSP-16

The mode comes from ``--mode`` or, per request, from the header ``x-fake-mode``.
Tools are read from the request (``tools``, namespace groups, ``additional_tools``).

Function-call arguments satisfy the declared schema's required fields, and with
--max-tool-turns N the server answers without a call once the input holds N tool
outputs, so a real harness (e.g. `codex exec`) finishes its turn: a wiring test
of harness -> recorder/server -> trace -> checks, without a model.

usage: fake_responses_server.py [--host 127.0.0.1] [--port 8399] [--mode ok] [--max-tool-turns N]
"""

import argparse
import json
import uuid

from aiohttp import web


def first_tool(body):
    def walk(tools, namespace):
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "namespace":
                found = walk(tool.get("tools"), tool.get("name"))
                if found:
                    return found
            elif tool.get("type") in ("function", "custom") and tool.get("name"):
                return tool, namespace
        return None

    found = walk(body.get("tools"), None)
    for item in body.get("input") or []:
        if found:
            break
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            found = walk(item.get("tools"), None)
    return found or ({"type": "function", "name": "noop", "parameters": {"type": "object"}}, None)


def schema_arguments(tool):
    """A JSON object that satisfies the tool's required parameters, so a real harness accepts the call."""
    params = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    args = {}
    for key in params.get("required") or []:
        kind = (props.get(key) or {}).get("type")
        if kind in ("integer", "number"):
            args[key] = 1
        elif kind == "boolean":
            args[key] = False
        elif kind == "array":
            args[key] = []
        elif kind == "object":
            args[key] = {}
        else:
            args[key] = "echo hi" if any(w in key for w in ("cmd", "command")) else "x"
    return json.dumps(args)


def tool_outputs(body):
    return sum(
        1
        for i in body.get("input") or []
        if isinstance(i, dict)
        and i.get("type") in ("function_call_output", "custom_tool_call_output")
    )


def frames(body, mode, max_tool_turns=0):
    rid, model = "resp_" + uuid.uuid4().hex, body.get("model", "fake")
    seq = iter(range(0, 10_000))  # from 0, as OpenAI and trtllm-serve number a stream
    out = []

    def emit(kind, **data):
        data = dict(type=kind, sequence_number=next(seq), **data)
        out.append(f"event: {kind}\ndata: {json.dumps(data)}\n\n")

    base = {"id": rid, "object": "response", "model": model, "status": "in_progress", "output": []}
    opening = base
    if mode == "created_without_output":
        opening = {k: v for k, v in base.items() if k != "output"}
    emit("response.created", response=opening)
    emit("response.in_progress", response=opening)
    if mode == "overlength":
        failed = dict(
            base,
            status="failed",
            error={
                "code": "context_length_exceeded",
                "message": "input exceeds the context window",
            },
        )
        emit("response.failed", response=failed)
        return out
    items = []
    rs_id, msg_id = "rs_" + uuid.uuid4().hex, "msg_" + uuid.uuid4().hex
    reasoning = {
        "id": rs_id,
        "type": "reasoning",
        "status": "completed",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "Plan the call."}],
    }
    emit(
        "response.output_item.added",
        output_index=0,
        item=dict(reasoning, content=[], status="in_progress"),
    )
    emit(
        "response.reasoning_text.delta",
        item_id=rs_id,
        output_index=0,
        content_index=0,
        delta="Plan the call.",
    )
    emit(
        "response.reasoning_text.done",
        item_id=rs_id,
        output_index=0,
        content_index=0,
        text="Plan the call.",
    )
    emit("response.output_item.done", output_index=0, item=reasoning)
    items.append(reasoning)
    text = "Running it now."
    # wrong_output_index: the message's part and text events point at the reasoning item (0).
    msg_index = 0 if mode == "wrong_output_index" else 1
    message = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    emit(
        "response.output_item.added",
        output_index=1,
        item=dict(message, content=[], status="in_progress"),
    )
    emit(
        "response.content_part.added",
        item_id=msg_id,
        output_index=msg_index,
        content_index=0,
        part={"type": "output_text", "text": "", "annotations": []},
    )
    for piece in ("Running ", "it now."):
        emit(
            "response.output_text.delta",
            item_id=msg_id,
            output_index=msg_index,
            content_index=0,
            delta=piece,
        )
    emit(
        "response.output_text.done",
        item_id=msg_id,
        output_index=msg_index,
        content_index=0,
        text=text + ("!" if mode == "delta_differs" else ""),
    )
    if mode != "drop_part_done":
        emit(
            "response.content_part.done",
            item_id=msg_id,
            output_index=msg_index,
            content_index=0,
            part={"type": "output_text", "text": text, "annotations": []},
        )
    emit("response.output_item.done", output_index=1, item=message)
    items.append(
        dict(
            message,
            content=[
                {
                    "type": "output_text",
                    "annotations": [],
                    "text": text + (" (rewritten)" if mode == "snapshot_text_differs" else ""),
                }
            ],
        )
    )
    tool, namespace = first_tool(body)
    n_calls = 2 if mode == "two_calls" else 1
    if max_tool_turns and tool_outputs(body) >= max_tool_turns:
        n_calls = 0  # end the conversation: a real harness stops when a turn has no call
    for k in range(n_calls):
        index = 2 + k
        name = (namespace + "." if mode == "prefixed_name" and namespace else "") + tool["name"]
        call_id = "call_" + uuid.uuid4().hex
        as_custom = tool.get("type") == "custom" and mode != "custom_as_function"
        if as_custom:
            item = {
                "id": "ctc_" + uuid.uuid4().hex,
                "type": "custom_tool_call",
                "call_id": call_id,
                "name": name,
                "input": "echo hi",
            }
            if namespace and mode != "prefixed_name":
                item["namespace"] = namespace
            emit("response.output_item.added", output_index=index, item=dict(item, input=""))
            emit(
                "response.custom_tool_call_input.delta",
                item_id=item["id"],
                output_index=index,
                delta="echo hi",
            )
            emit(
                "response.custom_tool_call_input.done",
                item_id=item["id"],
                output_index=index,
                input="echo hi",
            )
        else:
            arguments = schema_arguments(tool) if tool.get("type") == "function" else "{}"
            item = {
                "id": "fc_" + uuid.uuid4().hex,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": "completed",
            }
            if namespace and mode != "prefixed_name":
                item["namespace"] = namespace
            emit(
                "response.output_item.added",
                output_index=index,
                item=dict(item, arguments="", status="in_progress"),
            )
            emit(
                "response.function_call_arguments.delta",
                item_id=item["id"],
                output_index=index,
                delta=arguments,
            )
            emit(
                "response.function_call_arguments.done",
                item_id=item["id"],
                output_index=index,
                arguments=arguments,
            )
        emit("response.output_item.done", output_index=index, item=item)
        items.append(item)
    if mode == "unknown_event":
        # An event type the validator has no rule for (a reasoning summary): must be recorded.
        emit("response.reasoning_summary_text.delta", item_id="rs_x", output_index=0, delta="sum")
    if mode != "no_terminal":
        done = dict(base, status="completed", output=items)
        if mode != "usage_missing":
            output_tokens = 5 + len(items)
            total = 10 + output_tokens + (1 if mode == "usage_total_mismatch" else 0)
            done["usage"] = {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0},
                             "output_tokens": output_tokens,
                             "output_tokens_details": {"reasoning_tokens": 2},
                             "total_tokens": total}  # fmt: skip
        emit("response.completed", response=done)
    return out


async def responses(request):
    body = await request.json()
    mode = request.headers.get("x-fake-mode") or request.app["mode"]
    if mode == "error_string":
        return web.json_response({"error": "1 validation error: store: Extra inputs are not permitted"},
                                 status=400)  # fmt: skip
    turns = request.app["max_tool_turns"]
    if not body.get("stream"):
        chunks = frames(body, mode, turns)
        last = json.loads(chunks[-1].split("data: ", 1)[1])
        return web.json_response(last.get("response") or {})
    resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
    await resp.prepare(request)
    for chunk in frames(body, mode, turns):
        await resp.write(chunk.encode())
    await resp.write_eof()
    return resp


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--mode", default="ok")
    ap.add_argument(
        "--max-tool-turns",
        type=int,
        default=0,
        help="answer without a call once the input holds N tool outputs (lets a real harness finish)",
    )
    args = ap.parse_args()
    app = web.Application(client_max_size=1 << 30)
    app["mode"] = args.mode
    app["max_tool_turns"] = args.max_tool_turns
    app.router.add_post("/v1/responses", responses)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
