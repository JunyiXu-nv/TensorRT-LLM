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
"""A model-free Anthropic Messages + Chat Completions server, correct or broken on purpose.

Two uses:
  * controls: each fault mode breaks exactly one wire rule of
    references/protocols.md, so the validator must flag exactly that rule
    (references/acceptance-cases.md sections 2 and 3);
  * harness wiring: a real Claude Code or pi session against ``ok`` runs end to
    end without a GPU, and its requests show what the harness sends.

Routes: POST /v1/messages (``?beta=true`` accepted), POST /v1/messages/count_tokens,
HEAD /api/hello, GET /v1/models, POST /v1/chat/completions.

The fault mode comes from ``--mode`` or, per request, the ``x-fake-mode`` header.
The server answers with a call to a declared tool (``--prefer-tools`` first, else
the first declared one), with arguments built from its schema, until the history
holds ``--max-tool-turns`` tool results; then it answers with text and ends the turn.

Anthropic modes: ok, ok_thinking, overlength_profile, overlength_400_wrong_wording,
  no_message_delta, no_message_stop, delta_before_start, interleaved_blocks,
  delta_type_mismatch, bad_partial_json, duplicate_tool_use_id, stop_reason_mismatch,
  two_tools_disable_parallel, undeclared_tool, invalid_input, error_untyped, content_after_stop,
  silent_gap, html_body, json_for_stream, unknown_block
Chat modes: ok, ok_reasoning, overlength_profile, no_done, two_finish_reasons,
  fragment_without_id, arguments_not_object, finish_reason_mismatch, duplicate_tool_call_id,
  two_calls_ptc_false, reasoning_in_content, model_changes, invalid_input, silent_gap, html_body,
  json_for_stream, unknown_delta_field, bad_extension_field

usage: fake_agent_server.py [--host 127.0.0.1] [--port 8400] [--mode ok] [--max-tool-turns N]
                            [--gap-seconds S] [--prefer-tools Read,read,...]
"""

import argparse
import asyncio
import json
import time
import uuid

from aiohttp import web

MODEL = "fake-model"


# --------------------------------------------------------------------------- helpers


def schema_arguments(schema):
    """An object satisfying the schema's required properties, so a real harness accepts it."""
    schema = schema if isinstance(schema, dict) else {}
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    args = {}
    for key in schema.get("required") or []:
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
    return args


def pick_tool(tools, prefer):
    """(name, schema) of the preferred declared tool, else the first one; None without tools."""
    by_name = {t["name"]: t["schema"] for t in tools}
    for name in prefer:
        if name in by_name:
            return name, by_name[name]
    return (tools[0]["name"], tools[0]["schema"]) if tools else None


async def send(resp, text, gap=0.0):
    if gap:
        await asyncio.sleep(gap)
    await resp.write(text.encode())


# --------------------------------------------------------------------------- Anthropic


def anthropic_tools(body):
    out = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("name") and tool.get("type") in (None, "custom"):
            out.append({"name": tool["name"], "schema": tool.get("input_schema")})
    return out


def anthropic_tool_results(body):
    count = 0
    for message in body.get("messages") or []:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            count += sum(1 for b in message["content"]
                         if isinstance(b, dict) and b.get("type") == "tool_result")  # fmt: skip
    return count


def sse(event_type, payload):
    payload = dict(payload, type=event_type)
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


async def messages(request):
    app = request.app
    body = await request.json()
    mode = request.headers.get("x-fake-mode") or app["mode"]
    if mode == "overlength_profile":
        return web.json_response({"type": "error", "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 210000 tokens > 200000 maximum"}}, status=400)  # fmt: skip
    if mode == "overlength_400_wrong_wording":
        return web.json_response({"type": "error", "error": {
            "type": "invalid_request_error",
            "message": "This model's maximum context length is 200000 tokens"}}, status=400)  # fmt: skip
    if mode == "html_body":
        return web.Response(text="<html>upstream error</html>", content_type="text/html")
    if not body.get("stream") or mode == "json_for_stream":
        return web.json_response(_anthropic_message(body, app))
    resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
    await resp.prepare(request)
    gap = app["gap"] if mode == "silent_gap" else 0.0
    tool = pick_tool(anthropic_tools(body), app["prefer"])
    call = tool is not None and anthropic_tool_results(body) < app["max_tool_turns"]
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    start = {"message": {"id": msg_id, "type": "message", "role": "assistant", "model": MODEL,
                         "content": [], "stop_reason": None, "stop_sequence": None,
                         "usage": {"input_tokens": 10, "output_tokens": 0}}}  # fmt: skip
    if mode == "error_untyped":
        await send(resp, sse("message_start", start))
        await send(resp, sse("error", {"error": {"message": "boom"}}))
        await resp.write_eof()
        return resp
    await send(resp, sse("message_start", start))
    await send(resp, sse("ping", {}))
    index = 0
    if mode == "ok_thinking":
        await send(resp, sse("content_block_start", {"index": index, "content_block":
                                                     {"type": "thinking", "thinking": ""}}))  # fmt: skip
        await send(resp, sse("content_block_delta", {"index": index, "delta":
                                                     {"type": "thinking_delta", "thinking": "Plan."}}))  # fmt: skip
        await send(resp, sse("content_block_delta", {"index": index, "delta":
                                                     {"type": "signature_delta", "signature": "c2ln"}}))  # fmt: skip
        await send(resp, sse("content_block_stop", {"index": index}))
        index += 1
    if mode == "delta_before_start":
        await send(resp, sse("content_block_delta", {"index": index, "delta":
                                                     {"type": "text_delta", "text": "early"}}))  # fmt: skip
    await send(resp, sse("content_block_start", {"index": index, "content_block":
                                                 {"type": "text", "text": ""}}))  # fmt: skip
    await send(resp, sse("content_block_delta", {"index": index, "delta":
                                                 {"type": "text_delta", "text": "Working"}}), gap)  # fmt: skip
    if mode != "interleaved_blocks":
        await send(resp, sse("content_block_stop", {"index": index}))
    text_index = index
    index += 1
    stop_reason = "end_turn"
    if call:
        name, schema = tool
        if mode == "undeclared_tool":
            name = "definitely_not_declared"
        args = "{}" if mode == "invalid_input" else json.dumps(schema_arguments(schema))
        ids = ["toolu_" + uuid.uuid4().hex[:20]]
        if mode in ("duplicate_tool_use_id", "two_tools_disable_parallel"):
            ids.append(
                ids[0] if mode == "duplicate_tool_use_id" else "toolu_" + uuid.uuid4().hex[:20]
            )
        for tool_id in ids:
            await send(resp, sse("content_block_start", {"index": index, "content_block": {
                "type": "tool_use", "id": tool_id, "name": name, "input": {}}}))  # fmt: skip
            if mode == "delta_type_mismatch":
                await send(resp, sse("content_block_delta", {"index": index, "delta":
                                                             {"type": "text_delta", "text": "x"}}))  # fmt: skip
            payload = "[1, 2]" if mode == "bad_partial_json" else args
            half = len(payload) // 2
            for piece in (payload[:half], payload[half:]):
                await send(resp, sse("content_block_delta", {"index": index, "delta":
                                                             {"type": "input_json_delta",
                                                              "partial_json": piece}}))  # fmt: skip
            await send(resp, sse("content_block_stop", {"index": index}))
            index += 1
        stop_reason = "end_turn" if mode == "stop_reason_mismatch" else "tool_use"
    if mode == "interleaved_blocks":
        await send(resp, sse("content_block_stop", {"index": text_index}))
    if mode == "unknown_block":
        # A server tool block the validator has no rule for: must be recorded, not dropped.
        await send(resp, sse("content_block_start", {"index": index, "content_block": {
            "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}}))  # fmt: skip
        await send(resp, sse("content_block_stop", {"index": index}))
        index += 1
    if mode != "no_message_delta":
        await send(resp, sse("message_delta", {"delta": {"stop_reason": stop_reason,
                                                         "stop_sequence": None},
                                               "usage": {"output_tokens": 5}}))  # fmt: skip
        if mode != "no_message_stop":
            await send(resp, sse("message_stop", {}))
        if mode == "content_after_stop":
            await send(resp, sse("ping", {}))
    await resp.write_eof()
    return resp


def _anthropic_message(body, app):
    tool = pick_tool(anthropic_tools(body), app["prefer"])
    content = [{"type": "text", "text": "Working"}]
    stop = "end_turn"
    if tool is not None and anthropic_tool_results(body) < app["max_tool_turns"]:
        content.append({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20],
                        "name": tool[0], "input": schema_arguments(tool[1])})  # fmt: skip
        stop = "tool_use"
    return {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant",
            "model": MODEL, "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5}}  # fmt: skip


async def count_tokens(request):
    body = await request.json()
    return web.json_response({"input_tokens": max(1, len(json.dumps(body)) // 4)})


async def hello(request):
    return web.Response(status=200)


async def models(request):
    return web.json_response({"data": [{"id": MODEL, "type": "model", "display_name": MODEL,
                                        "created_at": "2026-01-01T00:00:00Z"}],
                              "object": "list", "has_more": False})  # fmt: skip


# --------------------------------------------------------------------------- Chat


def chat_tools(body):
    out = []
    for tool in body.get("tools") or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            out.append({"name": fn["name"], "schema": fn.get("parameters")})
    return out


def chat_tool_results(body):
    return sum(
        1 for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") == "tool"
    )


async def chat(request):
    app = request.app
    body = await request.json()
    mode = request.headers.get("x-fake-mode") or app["mode"]
    if mode == "overlength_profile":
        return web.json_response({"error": {
            "message": "This model's maximum context length is 200000 tokens",
            "type": "invalid_request_error", "code": "context_length_exceeded"}}, status=400)  # fmt: skip
    tool = pick_tool(chat_tools(body), app["prefer"])
    call = tool is not None and chat_tool_results(body) < app["max_tool_turns"]
    chunk_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    def chunk(delta, finish=None, model=MODEL, usage=None, choices=True):
        obj = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
        }
        obj["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}] if choices else []
        if usage is not None:
            obj["usage"] = usage
        return f"data: {json.dumps(obj)}\n\n"

    if mode == "html_body":
        return web.Response(text="<html>upstream error</html>", content_type="text/html")
    if not body.get("stream") or mode == "json_for_stream":
        message = {"role": "assistant", "content": "Working"}
        finish = "stop"
        if call:
            message["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:20], "type": "function",
                                      "function": {"name": tool[0],
                                                   "arguments": json.dumps(schema_arguments(tool[1]))}}]  # fmt: skip
            finish = "tool_calls"
        return web.json_response({"id": chunk_id, "object": "chat.completion", "created": created,
                                  "model": MODEL, "choices": [{"index": 0, "message": message,
                                                               "finish_reason": finish}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                            "total_tokens": 15}})  # fmt: skip
    resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
    await resp.prepare(request)
    gap = app["gap"] if mode == "silent_gap" else 0.0
    await send(resp, chunk({"role": "assistant", "content": ""}))
    if mode == "ok_reasoning":
        await send(resp, chunk({"reasoning_content": "Plan."}))
    if mode == "unknown_delta_field":
        await send(resp, chunk({"refusal": "not checked by the validator"}))
    text = "<think>leak</think> Working" if mode == "reasoning_in_content" else "Working"
    await send(resp, chunk({"content": text}, model="other-model" if mode == "model_changes" else MODEL),
               gap)  # fmt: skip
    finish = "stop"
    if call:
        name, schema = tool
        args = ("[1]" if mode == "arguments_not_object" else
                "{}" if mode == "invalid_input" else json.dumps(schema_arguments(schema)))  # fmt: skip
        ids = ["call_" + uuid.uuid4().hex[:20]]
        if mode in ("duplicate_tool_call_id", "two_calls_ptc_false"):
            ids.append(
                ids[0] if mode == "duplicate_tool_call_id" else "call_" + uuid.uuid4().hex[:20]
            )
        for index, call_id in enumerate(ids):
            first = {
                "index": index,
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }
            if mode != "fragment_without_id":
                first["id"] = call_id
            await send(resp, chunk({"tool_calls": [first]}))
            half = len(args) // 2
            for piece in (args[:half], args[half:]):
                await send(resp, chunk({"tool_calls": [{"index": index,
                                                        "function": {"arguments": piece}}]}))  # fmt: skip
        finish = "stop" if mode == "finish_reason_mismatch" else "tool_calls"
    finishing = chunk({}, finish=finish)
    # trtllm-serve adds its speculative-decoding statistic to the finishing choice (CHT-13).
    stat = json.dumps("fast" if mode == "bad_extension_field" else 1.5)
    finishing = finishing.replace(
        '"finish_reason":', '"avg_decoded_tokens_per_iter": ' + stat + ', "finish_reason":', 1
    )
    await send(resp, finishing)
    if mode == "two_finish_reasons":
        await send(resp, chunk({}, finish=finish))
    options = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
    if options.get("include_usage"):
        await send(resp, chunk(None, usage={"prompt_tokens": 10, "completion_tokens": 5,
                                            "total_tokens": 15}, choices=False))  # fmt: skip
    if mode != "no_done":
        await send(resp, "data: [DONE]\n\n")
    await resp.write_eof()
    return resp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--mode", default="ok")
    ap.add_argument("--max-tool-turns", type=int, default=1)
    ap.add_argument("--gap-seconds", type=float, default=1.5)
    ap.add_argument("--prefer-tools", default="Read,read,LS,ls,Glob,glob",
                    help="tools to call first when declared (read-only ones by default)")  # fmt: skip
    args = ap.parse_args()
    app = web.Application(client_max_size=1 << 30)
    app.update(mode=args.mode, max_tool_turns=args.max_tool_turns, gap=args.gap_seconds,
               prefer=[p for p in args.prefer_tools.split(",") if p])  # fmt: skip
    app.router.add_post("/v1/messages", messages)
    app.router.add_post("/v1/messages/count_tokens", count_tokens)
    app.router.add_route("HEAD", "/api/hello", hello)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
