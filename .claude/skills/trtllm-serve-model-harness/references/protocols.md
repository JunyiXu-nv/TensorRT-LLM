<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wire rules per protocol

Validator-owned. These rules back check C-01 (`validate.md`), and each rule id appears in a
turn record's `reply.grammar` when it is broken. They cover what trtllm-serve sends. What a
particular harness does with it lives in that harness's contract and profile.

Sources:
- the OpenAI Responses and Chat Completions streaming formats;
- the Anthropic Messages streaming format;
- the invariants trtllm-serve itself states. The Anthropic stream reframer docstring is in
  `tensorrt_llm/serve/anthropic_adapter.py`; the Responses rules come from what
  `check_stream.py` enforced over 161,639 production streams.

When a source and this file disagree, raise it as a policy change. Do not patch a check to
match the server.

## 1. Responses (`POST /v1/responses`, `stream: true`)

Every SSE frame is `event: <name>` plus `data: <json>` with the same `type`.

| Rule | Requirement | `check_stream.py` name |
|---|---|---|
| RSP-01 | Opens with `response.created`. `response.in_progress` may follow and may repeat as a keepalive | — (planned) |
| RSP-02 | Exactly one terminal event: `response.completed`, `response.incomplete` or `response.failed` | `no_terminal_event`, `multiple_terminal_events` |
| RSP-03 | Nothing after the terminal event | `events_after_terminal` |
| RSP-04 | `sequence_number` strictly increasing | `sequence_not_increasing` |
| RSP-05 | One response id and one model across the opening and terminal events | `response_id_changes`, `model_changes` |
| RSP-06 | Every `output_item.added` has its `output_item.done`, and the reverse | `unpaired_output_item` |
| RSP-07 | Every `content_part.added` has its `content_part.done`, and the reverse | `unpaired_content_part` |
| RSP-08 | Joined deltas equal the matching `*.done` payload: `output_text`, `reasoning_text`, `function_call_arguments`, `custom_tool_call_input` | `delta_done_mismatch` |
| RSP-09 | Each `*.done` payload equals the text, arguments or input of its `output_item.done` item | `done_item_mismatch` |
| RSP-10 | The `(type, id)` list of `output_item.done` equals the terminal snapshot's `output` | `item_list_differs` |
| RSP-11 | Per item id, the snapshot's text, name, call_id, arguments and input equal the streamed item | `item_content_differs` |
| RSP-12 | A snapshot `message` has `status`. `function_call` and `reasoning` may omit it but never carry an explicit null. `custom_tool_call` has no status field | `item_status_missing` |
| RSP-13 | A custom tool is delivered as `custom_tool_call` with `input`; a function as `function_call` with JSON `arguments` | check C-06 `type_mismatch` |
| RSP-14 | A `response.failed` snapshot carries no generated output: count it, do not compare items | `failed_snapshot_without_streamed_items` (counted) |

**Over-length input.** Fail in-stream: `response.failed` with
`error.code == "context_length_exceeded"`. Codex compacts only on that. An HTTP 400 makes it
resend. This is harness-specific, and the harness profile's `errors.overlength` states it.

## 2. Anthropic Messages (`POST /v1/messages`, `stream: true`)

Framing is `event: <type>` plus `data: <json>` whose `type` equals the event name. The
sequence:

```text
message_start
  (content_block_start  content_block_delta*  content_block_stop)*
message_delta
message_stop
```

`ping` may appear anywhere, and `error` ends the stream.

| Rule | Requirement |
|---|---|
| ANT-01 | The first non-`ping` event is `message_start`, exactly once. `message.role == "assistant"`, `message.content == []`, `message.id` and `message.model` are set, and `usage.input_tokens` is present |
| ANT-02 | `content_block_start` for index `i` comes before any delta or stop for `i`. Indices start at 0 and increase by 1 |
| ANT-03 | Blocks are sequential: block `i+1` starts only after `content_block_stop` for `i`. trtllm-serve states this invariant; the reframer serializes interleaved upstream tool calls |
| ANT-04 | The delta type matches the block type: `text` takes `text_delta`; `tool_use` takes `input_json_delta`; `thinking` takes `thinking_delta`, then optionally one `signature_delta`; `redacted_thinking` takes none |
| ANT-05 | Every started block is stopped. There are no deltas after a block's stop |
| ANT-06 | A `tool_use` start carries an `id` that is unique within the message, a `name`, and `input: {}`. The joined `partial_json` is a JSON object, and an empty string means `{}` |
| ANT-07 | `signature_delta`, if present, comes after the last `thinking_delta` of its block and before its stop. Whether a signature is required is harness policy (`reasoning.carrier` in the profile) |
| ANT-08 | Exactly one `message_delta`, after every block has stopped. `delta.stop_reason` is one of `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `pause_turn` or `refusal`. `usage.output_tokens` is present |
| ANT-09 | `stop_reason == "tool_use"` exactly when there is at least one `tool_use` block, except when `max_tokens` cut the reply |
| ANT-10 | Exactly one `message_stop`, as the last event. A body that ends after a block has started but before `message_delta` reads to Claude Code as a dropped connection |
| ANT-11 | `error` events carry `{"type": "error", "error": {"type": T, "message": M}}`, with `T` one of `invalid_request_error`, `authentication_error`, `permission_error`, `not_found_error`, `request_too_large`, `rate_limit_error`, `api_error` or `overloaded_error`. Nothing follows an error |
| ANT-12 | Under `disable_parallel_tool_use: true`, at most one `tool_use` block (check C-07) |
| ANT-13 | Non-streaming bodies: `type: "message"`, `role: "assistant"`, a content block list, a non-null `stop_reason`, and `usage` |
| ANT-14 | HTTP errors use the same envelope, with matching statuses: 400 `invalid_request_error`, 401, 403, 404, 413 `request_too_large`, 429 `rate_limit_error`, 500 `api_error`, 529 `overloaded_error`. `retry-after` is an integer number of seconds |

**Liveness.** `ping` is the keepalive. Claude Code aborts a stream when no bytes arrive within
its idle watchdog, and long thinking or prefill needs pings (checks R-01 and R-02).

**Gaps on main, measured 2026-10-21 (not rules):**
- no `ping` is emitted;
- `signature_delta` is defined but never emitted;
- errors are always `api_error`;
- over-length input is an HTTP 400 with OpenAI wording;
- `tool_choice` `any`/`tool` and `disable_parallel_tool_use` are rejected with a 400.

Whether each one breaks Claude Code is an S0 measurement.

## 3. Chat Completions (`POST /v1/chat/completions`, `stream: true`)

Each frame is `data: <json>`. The stream ends with `data: [DONE]`.

| Rule | Requirement |
|---|---|
| CHT-01 | Every frame before `[DONE]` is a `chat.completion.chunk` object or an error object |
| CHT-02 | `id`, `model` and `created` are constant across chunks |
| CHT-03 | For each choice, the first delta carries `role: "assistant"` |
| CHT-04 | `tool_calls` fragments are keyed by `index`. The first fragment of an index carries `id`, `type: "function"` and `function.name`; later fragments carry only `function.arguments`. Ids are unique |
| CHT-05 | The joined `function.arguments` of each index is a JSON object |
| CHT-06 | Exactly one non-null `finish_reason` per choice, in that choice's last chunk, from `stop`, `length`, `tool_calls` and `content_filter`. Nothing follows it for that choice |
| CHT-07 | `finish_reason == "tool_calls"` exactly when the choice has at least one tool call, except `length` |
| CHT-08 | A usage chunk (`choices: []`) appears only when `stream_options.include_usage` asked for it, at most once, after the finishing chunks |
| CHT-09 | `[DONE]` appears exactly once, last |
| CHT-10 | Reasoning travels in `delta.reasoning_content`, or in `delta.reasoning` for GPT-OSS. Which one the harness reads is in its profile (`reasoning.field`). Reasoning text never appears in `content` (check C-09) |
| CHT-11 | Under `parallel_tool_calls: false`, at most one tool call index (check C-07) |
| CHT-12 | Non-streaming bodies: `choices[].message` with `role`, `content`, optional `reasoning_content` and `tool_calls`, plus `finish_reason` |

Chat Completions has no standard mid-stream error. What trtllm-serve emits, and what pi does
with it, are S0 measurements recorded in `contracts/pi.md`.

## 4. Rules common to all three

- **The streamed bytes are the truth.** The client already received them, and may have acted
  on them. Every other view must agree with the stream, byte for byte.
- **Over-length, overload and unsupported fields are answered the way the harness can
  recover from.** The harness profile's `errors` section is the requirement, and checks C-10
  and R-03 are the test.
- **Unknown request fields are accepted, not rejected,** unless the harness profile says a
  rejection is the recovery signal. Claude Code retries without some capabilities only when the
  400's wording matches. Main's Anthropic request models use `extra="allow"`.
