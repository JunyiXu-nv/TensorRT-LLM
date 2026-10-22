<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harness contract: pi (coding agent) against trtllm-serve

Harness profile: `profiles/harness/pi-chat.yaml`, one profile per protocol choice. **Status:
doc-only. pi is not installed on the dev host, and no traffic has been recorded.**

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

1. Install and pin. Record 10 sessions over the session catalog. Run `summarize_capture.py`.
2. Read the source for: the request builder (fields, compat flags), tool error strings,
   overflow detection, timeouts and retries.
3. Pick `compat` values per model with trtllm-serve, then re-record.
4. Decide on session identity: an observed key, or a recorder port per session.
5. Feed pi a Chat stream that ends with an unrequested usage chunk (`choices: []`). The Kimi-K3
   extension sends one on every stream (`profiles/model/kimi-k3.yaml`).
