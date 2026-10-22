<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harness contract: pi (coding agent) against trtllm-serve

Harness profiles: `profiles/harness/pi-chat.yaml`, `pi-responses.yaml` and `pi-messages.yaml`, one
per API. **Status: partial.** Wiring was measured on 2026-10-22 against the fake servers
(section 9); outcomes, errors and timeouts are still open.

Source marks:
- **[doc]**: pi's README and `docs/models.md` (package `@mariozechner/pi-coding-agent` 0.68.0;
  the project now lives at `earendil-works/pi`), read 2026-10-21;
- **[main]**: trtllm-serve on main;
- **[measure]**: open.

pi is open source (TypeScript). Answer the [measure] items from its source with file:line, as
the Codex contract did with codex-rs.

## 1. Identity

- **Install and pin a version.** The package was renamed; record the exact package and version.
- **[doc] A custom provider lives in `~/.pi/agent/models.json`:**

  ```json
  {"providers": {"trtllm": {
     "baseUrl": "http://<host>:<port>/v1", "api": "openai-completions", "apiKey": "dummy",
     "models": [{"id": "<served name>", "name": "<label>", "reasoning": true,
                 "input": ["text"], "contextWindow": <served max_seq_len>, "maxTokens": <cap>,
                 "compat": { ... }}]}}}
  ```

- **[doc] API choices:** `openai-completions` ("most compatible"), `openai-responses`,
  `anthropic-messages` and `google-generative-ai`.
  - **Use `openai-completions` for the matrix.** Codex already covers Responses and Claude Code
    covers Messages, so pi over Chat Completions covers the third path.
  - pi's other API choices can later serve as a protocol differential: the same agent, the same
    task, a different wire.
- **Isolation.** Use a separate `HOME` (or pi's own directory variable, [measure]) per session,
  so `~/.pi` is never shared.

## 2. Wire

- **[doc]** `POST <baseUrl>/chat/completions`.
- **[measure]**
  - Whether it always streams.
  - Which body keys it sends.
  - Whether it sends `stream_options.include_usage`, `parallel_tool_calls`, `reasoning_effort`
    and `chat_template_kwargs`.
- **[doc] Its `compat` flags are contract items.** Each one changes the wire:

  | Flag | Effect | trtllm-serve [main] |
  |---|---|---|
  | `supportsDeveloperRole` | `developer` vs `system` role | The developer role is passed through and may not render (f13c9de3a7 is not on main) |
  | `supportsReasoningEffort`, `reasoningEffortMap` | Sends `reasoning_effort` | Reaches the chat template for K3 only |
  | `thinkingFormat` | `reasoning_effort`, `zai`, `qwen` or `qwen-chat-template` | [measure] which one turns thinking on through trtllm-serve for each model |
  | `maxTokensField` | `max_tokens` vs `max_completion_tokens` | [measure] |
  | `requiresToolResultName`, `requiresAssistantAfterToolResult` | Shape of tool results | [measure] |
  | `requiresThinkingAsText` | Past thinking is sent back as text | Interacts with the model's reasoning history policy |
  | `supportsUsageInStreaming` | `stream_options.include_usage` | Supported |
  | `supportsStrictMode` | `strict` in tool definitions | [measure] |

## 3. Tools

- **[doc]** The built-in tools are `read`, `write`, `edit` and `bash`; `grep`, `find` and `ls`
  are optional. They are declared as Chat `tools`.
- **[measure]**
  - The error strings pi writes into `role: tool` messages for an unknown tool, invalid
    arguments and a failing command. These become the profile's `outcomes`.
  - Whether it ever disables parallel calls.

## 4. Conversation state

- **History.** The full history is sent every turn [measure].
- **Reasoning.** **[measure]** Which delta field it reads (`reasoning_content` or `reasoning`;
  trtllm-serve emits `reasoning_content`), and how it sends past reasoning back.
- **Session identity.** **[measure]** Whether any header or field identifies a session.
  - If none does, give each smoke session its own recorder port, so the capture tags the
    thread.
  - pi stores sessions as JSONL under `~/.pi/agent/sessions/` [doc]. That is a secondary,
    harness-side record.

## 5. Stream semantics

[measure]:
- the idle and first-byte timeouts;
- the retry policy;
- what it does with a mid-stream error chunk (Chat Completions has no standard one).

## 6. Context window

- **[doc]** The model entry's `contextWindow` (default 128000) and `maxTokens` (default 16384)
  drive pi. Compaction runs "on context overflow (recovers and retries)" and also proactively
  near the limit.
- **[measure]** How it recognizes overflow, i.e. which error status and wording. Then check
  that trtllm-serve's over-length answer triggers it (check H-04).

## 7. Checks to enable

- CHT-01 to CHT-12;
- C-03 echo;
- C-07 if parallel calls are disabled;
- C-09 reasoning separation;
- C-10;
- R-01/R-02;
- H-01 to H-04 with the measured strings.

## 8. Open S0 measurements (in order)

1. ~~Install and pin.~~ `@earendil-works/pi-coding-agent` 1.4.0 (section 9). Record 10 sessions
   over the session catalog. Run `summarize_capture.py`.
2. Read the source for: the request builder (fields, compat flags), tool error strings,
   overflow detection, timeouts and retries.
3. Pick `compat` values per model with trtllm-serve, then re-record.
4. ~~Decide on session identity.~~ Responses carries a `session_id` header; Chat and Messages
   carry nothing, so the recorder tags each session (`record_requests.py --session`).
5. Feed pi a Chat stream that ends with an unrequested usage chunk (`choices: []`). The Kimi-K3
   extension sends one on every stream (`profiles/model/kimi-k3.yaml`).
6. ~~Echo.~~ Verbatim over all three APIs against the fake servers (section 9). Re-check over
   Messages against trtllm-serve, which sends no thinking signature.

## 9. Measured 2026-10-22: wiring runs against the fake servers

- **Setup.** `@earendil-works/pi-coding-agent` 1.4.0 (renamed from `@mariozechner/pi-coding-agent`;
  node >= 22.19) under `scripts/run_pi_sessions.sh`, one session per API, `--tools read,ls`,
  behind `record_requests.py`. Chat and Messages against `fake_agent_server.py`, Responses
  against `fake_responses_server.py`.
- **[measured] Routes.** `POST /v1/chat/completions`, `POST /v1/responses` and
  `POST /v1/messages`, always `stream: true`. For anthropic-messages pi appends `/v1/messages`
  to the base URL, so that base URL has no `/v1` (with one, pi called `/v1/v1/messages`).
- **[measured] Bodies.**

  | API | Keys | Reasoning at the default level | Reasoning with `--thinking off` |
  |---|---|---|---|
  | openai-completions | `model`, `messages`, `tools`, `stream`, `stream_options.include_usage`, `max_completion_tokens`, `store`, `reasoning_effort` | `reasoning_effort: medium` | nothing |
  | openai-responses | `model`, `input`, `tools`, `stream`, `max_output_tokens`, `store`, `include`, `prompt_cache_key`, `reasoning` | `reasoning: {effort: medium, summary: auto}` | `reasoning: {effort: none}` |
  | anthropic-messages | `model`, `messages`, `system` (blocks), `tools`, `stream`, `max_tokens`, `thinking` | `thinking: {type: enabled, budget_tokens: 3072, display: summarized}` | `thinking: {type: disabled}` |

  With the model entry's `reasoning: false`, none of the three sends a reasoning parameter.
- **[measured] System prompt.** Sent as a `developer` message over Chat and Responses, as
  `system` blocks over Messages. It ends with the working directory (`<cwd>`).
- **[measured] Tools.** Function tools in each API's own shape. Over Messages, pi adds
  `eager_input_streaming`, `cache_control` on the last tool, and `"required": []` to a schema with
  no required parameter (`ls`).
- **[measured] Identity.** Responses: a `session_id` header and `prompt_cache_key`. Chat and
  Messages: none. Every API sends the `x-stainless-*` headers of the OpenAI or Anthropic
  TypeScript SDK.
- **[measured] Echo.** The next request carried our reply back verbatim: over Chat the assistant
  `content`, `reasoning_content` and `tool_calls`; over Responses the reasoning, message and
  function-call items with their ids; over Messages a signed thinking block, text and `tool_use`.
  C-03 passed on all three with `echo: exact` for text and reasoning.

## 10. Measured 2026-10-22 against trtllm-serve main (Qwen3-14B, one H200)

pi 1.4.0 ran four read-only tasks over each API against a main build
(`1.4.0rc2`, `--tool_parser qwen3 --reasoning_parser qwen3`), behind `record_requests.py`,
twice: comparable requests for the protocol differential, then pi's defaults.

- **openai-completions fails out of the box.** pi sends `store: false`; main's chat request
  model forbids unknown fields and answers HTTP 400 `extra_forbidden`. The body is
  `{"error": "<string>"}` (CHT-15) and the string carries a server source path and line.
  With the model entry's `compat.supportsStore: false`, every Chat check passed.
- **openai-responses.** Main's Responses stream fails RSP-07 (a `content_part.done` missing),
  RSP-12 (a reasoning item with `status: null` in the snapshot) and RSP-10 / C-02 (item ids
  differ between the stream and the snapshot). The campaign branches carry fixes for these.
- **openai-responses renders an echoed turn as two assistant messages.** pi sends back a
  reasoning item and a function-call item; the Responses path renders them as two assistant
  messages (an extra `<|im_end|>\n<|im_start|>assistant\n`), the Chat path as one. The same
  history bills 5 to 6 more tokens on the Responses path (C-15 `server_render`).
- **anthropic-messages: `count_tokens` drops user text blocks.** pi (like Claude Code) sends user
  content as text blocks. The `count_tokens` renderer does not flatten them, so the user message
  renders empty and the count falls short of the billed input by the user text (10 to 17 tokens
  here, USG-05). The Messages route itself flattens them.
- **anthropic-messages: thinking comes back as text.** main sends no `signature_delta`; pi, with
  `allowEmptySignature: false`, replays the unsigned thinking block as a text block. The model then
  sees its own reasoning as visible text in the next turn (C-03 `text,reasoning`). Either the
  server signs thinking blocks or the model entry sets `allowEmptySignature: true`.
- **Schema key order is pi's.** Over Messages pi writes `type, properties, required`, over the
  OpenAI APIs `type, required, properties`; the server keeps the client's order, so that one-token
  difference is the harness's (`harness_literal`).
- **The official SDKs agree.** C-12 read every successful reply of the three APIs (26 per mode)
  with no disagreement.
- **Tool errors.** A failing read comes back as Node's error message,
  `ENOENT: no such file or directory, access '<path>'` (profile `outcomes.tool_error`).
