<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harness contract: Claude Code 2.1.289 against trtllm-serve

Harness profile: `profiles/harness/claude-code.yaml`. **Status: doc-only. No traffic has been
recorded yet.**

Source marks:
- **[doc]**: Claude Code's gateway documentation, read 2026-10-21
  (`code.claude.com/docs/en/llm-gateway-protocol` and `.../llm-gateway`);
- **[main]**: trtllm-serve on main, read 2026-10-21;
- **[measure]**: an open S0 item.

Claude Code is closed source. Its semantics come from the docs plus black-box measurement:
record real sessions, then inject faults with the fake Messages server
(`references/acceptance-cases.md` section 2) and watch what it does.

**Scope warning [doc].** "Anthropic doesn't support routing Claude Code to non-Claude models
through any gateway." Capabilities arrive as new beta values, body fields and headers in new
releases. Pin the version and re-run this contract on every upgrade.

## 1. Identity

- **Version 2.1.289** is installed at `/opt/claude/claude` on the dev host.
- **Point it at trtllm-serve with:**
  - `ANTHROPIC_BASE_URL=http://<host>:<port>` without `/v1`, because Claude Code appends
    `/v1/messages`;
  - `ANTHROPIC_AUTH_TOKEN=<dummy>` (sent as `Authorization: Bearer`);
  - `ANTHROPIC_MODEL=<served name>` [doc].
- **Per-session isolation:** set `CLAUDE_CONFIG_DIR=<dir>`. The model-discovery cache and
  settings live there [doc].
- **Recommended for smoke runs** [doc], each to be confirmed [measure]:

  | Variable | Why |
  |---|---|
  | `ANTHROPIC_DEFAULT_HAIKU_MODEL=<served name>` | Background tasks go to the main model anyway with a bearer credential; make it explicit |
  | `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` | Adds request class, compaction and prompt-id headers. Trace grouping needs them |
  | `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` | Fewer side calls outside the endpoint |
  | `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | Removes the per-conversation attribution block from the system prompt. Decide per cell: it is part of what real users send |

## 2. Wire

- **[doc] Routes:**
  - `POST /v1/messages?beta=true`, always streamed;
  - `POST /v1/messages/count_tokens`, which is optional. Without it Claude Code estimates
    tokens from characters, and `/context` shows approximations;
  - `HEAD /api/hello`, a warm-up probe the endpoint may reject;
  - `GET /v1/models?limit=1000`, only with `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`. It
    keeps ids containing `claude` or `anthropic`, has a 3 s timeout, and treats a redirect as a
    failure.
- **[doc] Headers:**
  - `anthropic-version: 2023-06-01`;
  - `anthropic-beta`, an open, growing list;
  - `Authorization` and/or `x-api-key`;
  - `x-claude-code-session-id`, plus agent ids on sub-agent requests.
- **[doc] For a model id it does not recognize, Claude Code assumes a current Claude model and
  sends the full capability set:**
  - `thinking: {"type": "adaptive"}`;
  - `output_config` (effort, structured-output format, task budget);
  - `context_management` with its beta;
  - mid-conversation `role: "system"` entries in `messages`;
  - `cache_control` on system and message blocks;
  - beta tool fields (`strict`, `defer_loading`) and `tool_reference` blocks;
  - possibly the advisor tool.
- **[main] The request models use `extra="allow"`** and accept `thinking.adaptive`,
  `output_config`, `context_management` and `role: "system"` messages. The "rejected field"
  class, Codex's `service_tier` story, is largely preempted. **[measure]** Confirm with real
  bodies.
- **[doc] System prompt.** Its first block is an attribution block: client version plus a
  fingerprint, stable per conversation since v2.1.181. On a non-Anthropic endpoint it reaches
  the model and the prefix cache.

## 3. Tools

- **[doc]** Tools are declared in `tools` with `input_schema`. MCP tool search adds
  `defer_loading` and `tool_reference`.
- **[measure]**
  - Which server tools (web search, advisor) reach the endpoint.
  - Whether it ever sends `tool_choice` `any`/`tool` or `disable_parallel_tool_use`. **[main]**
    rejects all three with a 400.
- **[measure] What it sends back for a failed or unknown call:** a `tool_result` with
  `is_error: true`, and content strings, e.g. `<tool_use_error>...`. These become the profile's
  `outcomes`.

## 4. Conversation state

- **History.** The full history is sent every turn [doc].
- **Reasoning round-trip.**
  - **[doc]** Thinking blocks come back with their `signature`. If the endpoint rejects a
    signature (e.g. "bound to a different conversation"), Claude Code drops earlier thinking
    from later requests.
  - **[main]** No signature is emitted, and incoming signatures are dropped.
  - **[measure]** What Claude Code does with unsigned thinking blocks. Then decide per model
    how past reasoning is rendered: model profile `reasoning.history_policy`.
- **Identity.**
  - Session: `x-claude-code-session-id`.
  - Sub-agents: `x-claude-code-agent-id` and `x-claude-code-parent-agent-id`.
  - Request class, compaction trigger and prompt id need hint headers [doc].

## 5. Stream semantics

- **[doc] Full event sequence.** Claude Code reads events as they arrive and expects the whole
  sequence, through `message_delta` and `message_stop`. A body that ends cleanly after a block
  has started, but before `message_delta`, is a dropped connection: it shows "the response above
  may be incomplete" or retries.
- **[doc] Idle watchdog.** Claude Code aborts a stream when no bytes arrive for longer than its
  streaming idle watchdog, and pings are the keepalive.
  - **[main]** No `ping` is emitted.
  - **[measure]** The watchdog value, and what happens on a long prefill. This is the analog of
    Codex's 120 s cut.
- **[doc] Retries.**
  - It honors `retry-after` (integer seconds); above 60 it stops retrying.
  - It reads `x-should-retry`.
  - It retries once without a capability when a 400's wording matches the capability: the
    `thinking` type, mid-conversation system messages, `cache_control` on a system message,
    a thinking signature, the advisor tool, `output_config.effort` and `context_management`.
  - The logic matches error wording, so **our error messages are part of this contract**.
    **[measure]** The exact strings.

## 6. Context window

- **[doc]** For an unrecognized model id it assumes 200K, or 1M when the id carries `[1m]`.
  **[measure]** The model-config mechanism that declares the real window for a custom id. Set it
  to the served `max_seq_len`.
- **[doc] Reactive compaction.** It compacts when the API rejects a request as too long. Claude
  apps gateway uses the stable token `capability_rejected: prompt_too_long`.
  - **[main]** Over-length is an HTTP 400 with OpenAI wording and no "prompt is too long".
  - **[measure]** Whether Claude Code compacts on main's response. If it does not, this is the
    Codex 400 story again.

## 7. Known quirks and checks to enable

- **Background tasks.** Titles, classifiers and summaries go to the main model unless
  `ANTHROPIC_DEFAULT_HAIKU_MODEL` is set. They use structured outputs (`output_config.format`).
- **Direct egress.** The fast-mode check and the WebFetch domain safety check call
  `api.anthropic.com` directly. In a no-egress sandbox those errors are expected noise.
- **Checks:**
  - ANT-01 to ANT-14;
  - C-03 echo;
  - C-07 if parallel calls are ever disabled;
  - C-10 over-length;
  - R-01/R-02 against the measured watchdog;
  - H-01 to H-05 with the measured outcome strings.

## 8. Open S0 measurements (in order)

1. Record 10 sessions: the session catalog in `references/contract.md` section 2, sub-agent and
   compaction included. Run `summarize_capture.py`.
2. Measure the idle watchdog and the first-byte timeout: use the fake server with a
   `silent_gap`.
3. Over-length: main's 400 vs `prompt is too long` wording. Which one triggers compaction?
4. Thinking round-trip: unsigned blocks, a rejected signature.
5. Tool error strings: unknown tool, schema-invalid input, a failing command.
6. Whether `tool_choice`/`disable_parallel_tool_use` are ever sent.
7. The context-window declaration for a custom model id.
