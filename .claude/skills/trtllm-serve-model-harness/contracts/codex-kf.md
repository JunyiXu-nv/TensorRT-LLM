<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex over the Responses API (Kernel Factory configuration)

Harness profile: `profiles/harness/codex-kf.yaml`.

This is a worked example of a harness contract, and the reference for any new model served to
Codex.

Sources are marked on each item:
- **[measured]**: `summarize_capture.py` over the Kernel Factory window 2026-10-16 T16, as seen by
  the server. KF runs Codex behind an in-pod LiteLLM gateway.
- **[source]**: read from the codex-rs source during the campaign.
- **[fixed]**: a defect we hit and fixed, with its commit on the campaign branches.

Codex versions move. Re-measure before relying on any line.

**Configuration matters as much as version.** A default-config codex-cli 0.155.1 captured on
2026-10-19 sends a different contract:
- a top-level `tools` list of JSON functions (`exec_command`, `write_stdin`, `multi_agent_v1.*`, ...);
- an `instructions` field;
- `parallel_tool_calls: true`.

See `codex-default.md`. The contract below is the
Kernel Factory one.

## Wire

- **[measured]** `POST /v1/responses`, `stream: true` on every request.
  - The config is `wire_api = "responses"`. Codex is the only KF runtime that speaks Responses;
    claude, n3 and opencode arrive as chat completions or Anthropic.
- **[measured]** Body keys on every agent request:

  | Key | Value |
  |---|---|
  | `model` | The client's model id (e.g. `gpt-5.6-sol`), not the served name. The server or gateway must accept it |
  | `input` | Item list |
  | `include` | `["reasoning.encrypted_content"]` |
  | `parallel_tool_calls` | `false` |
  | `reasoning` | `{"context": "all_turns"}` (no `effort` in the KF config) |
  | `store` | `false` |
  | `text` | `{"verbosity": "low"}` |
  | `tool_choice` | `"auto"` |
  | `prompt_cache_key` | |
  | `client_metadata` | |

- **[measured]** There is no top-level `tools` and no `instructions`:
  - Tools arrive in one `{"type": "additional_tools", "tools": [...]}` input item.
  - Instructions arrive as `developer` messages with `input_text` parts.
- **[fixed]** Fields that once broke serving:

  | Field | What happened | Fix |
  |---|---|---|
  | `web_search` | Declared with external access off; it was a 400 on every turn | `afdb91551a` |
  | `service_tier` | 6,300 of 6,622 rejected | `d66713abe7` |
  | `client_metadata` | — | `04883b0fb5` |
  | Image and reasoning content parts | — | `ecbe431a56` |
  | Tool results as content parts | 483 of 490 rejected | `a9dc20daac` |

## Tools

**[measured]** Every tool is declared under a namespace:

| Namespace | Tool | Type |
|---|---|---|
| `functions` | `exec` | `custom`, with a Lark grammar (freeform input, not JSON) |
| `functions` | `wait`, `request_user_input` | `function` |
| `collaboration` | `followup_task`, `interrupt_agent`, `list_agents`, `send_message`, `spawn_agent`, `wait_agent` | `function` |

- **[measured]** `exec`'s description declares further tools under `` ### `name` `` headers
  (`exec_command`, `apply_patch`, `write_stdin`, ...). They are reached through `exec` and are not
  callable at the top level. Calling one directly is the model's error (`nested_tool_called_directly`).
- **[fixed]** Models write the namespaced tool both ways. One session had 247 bare `exec` against
  34 `functions.exec`.
  - The server must resolve every spelling to the declared tool and deliver a custom tool as
    `custom_tool_call` with `input`.
  - Delivered as `function_call` with JSON `arguments`, Codex answers `aborted`. That was 1,278 of
    3,985 tool results (32.1%) before `b52237f18c`.
- **[fixed]** Codex echoes some calls back without their namespace. When rendering history, the
  server restores the qualified name if it is unique (`91e18070a6`).
- **[source]** How Codex answers a call it could not use:
  - `aborted`: the call was cancelled. A custom tool sent as a function lands here.
  - `unsupported call: <name>`.
  - `... is unavailable in <mode> mode`.

  Count these in later requests' `*_call_output` items. They are the harness-side failure signal
  (`scripts/validate/harness_signals.py`, `references/smoke.md`).

## Input items the server must render

**[measured]** Item types, by frequency:
- `custom_tool_call` / `custom_tool_call_output`;
- `reasoning` (replayed every turn, with an `encrypted_content` carrier);
- `message` (`assistant/output_text`, `developer/input_text`, `user/input_text`);
- `function_call` / `function_call_output`;
- `additional_tools`;
- `agent_message` (sub-agent traffic).

Tool outputs are usually `input_text` parts, sometimes plain strings, and sometimes include
`input_image`.

Rendering requirements:
- **[fixed]** `developer` messages must reach the template. GLM's template dropped Codex's brief
  until `f13c9de3a7`.
- **[fixed]** Reasoning replay:
  - The server must read back its own reasoning carrier. Foreign carriers (another model's
    reasoning after re-routing) must neither crash nor be rendered as this model's thinking.
  - Pool affinity (`1317e90bf4`) keeps a conversation on one model.
- **[source]** Codex sends a reasoning item back with its `id` and its `encrypted_content`
  unchanged (codex-rs `ResponseItem::Reasoning`). C-03 compares that carrier by value
  (`echo.reasoning: exact`).

## Identity and routing

- **[measured]** Thread identity is in `client_metadata`:
  - `session_id`, `thread_id`, `turn_id`, `root_turn_id`, `x-codex-window-id`;
  - `x-codex-turn-metadata`, a JSON string with `agent_name`, `request_kind`, window ids, sandbox
    mode, ...
- **[measured]** `prompt_cache_key` is the *parent* session. Sub-agents share it, so group
  conversations by `thread_id`, but route cache affinity by `prompt_cache_key`.

## Streaming and endings

- **[source]** Codex records items on `response.output_item.done`.
  - The terminal snapshot must equal the stream, item by item: ids, text, arguments, input. The
    server must keep the streamed call ids (`b52237f18c`, `e0a8699934`, `be7746ef53`).
  - Snapshot items keep their `status` (`ec5240533c`). `custom_tool_call` has no status field.
- **[source]** Context window:
  - Codex treats a full context ONLY as SSE `response.failed` with
    `error.code == "context_length_exceeded"`; it then compacts on the next turn.
  - An HTTP 400 is a terminal InvalidRequest. KF's build re-sent one six times.
  - Fix: fail in-stream (`d1b379e870`, `6643a33bf5`).
- **[source]** After `response.incomplete` (max_output_tokens), the next request is the old input
  plus the truncated output.
- **[fixed]** Silence: KF and Codex cut a stream that sends nothing for about 120 s.
  - Long tool calls and cold disaggregated prefills hit this, and about 4,500 requests were
    abandoned in one preemption wave.
  - Fixes: open the stream after 30 s (`78ac9978b2`) and repeat `in_progress` every 15 s
    (`f8efc82403`).
  - The connect timeout must be shorter than 120 s too; SYN retries take 127 s (`c1e5bb8ae5`).
- **[fixed]** `parallel_tool_calls=false` must be enforced, and Codex always sends false.
  - Stop generation at the end of the first call (`single_call_stop`: `a3e176fef0`, `221949e395`).
  - A model that omits the first call's close tag can still produce two calls (B009-01,
    1 in 187,895). That needs a serving-side policy decision.
- **[fixed]** Degenerate generations (runaway `[PAD]`) must fail without output (`a13ecfbb50`).
  Codex records whatever `output_item.done` carried and replays it on every later turn.
- **[source]** KF's runner configures the compaction window: Codex compacts when its context count
  reaches `model_context_window` (codex 0.147.0).

## What a new model must be checked for, given this contract

- Tool-call format parsed in streaming and whole-text paths **identically** (custom vs function
  shape, names in both spellings).
- A `single_call_stop` for `parallel_tool_calls=false`.
- Reasoning carrier round-trip.
- `developer` and tool-output content parts rendered.
- In-stream `context_length_exceeded`.
- First event within 120 s at the model's worst prefill.
- Degenerate token ids stopped.
