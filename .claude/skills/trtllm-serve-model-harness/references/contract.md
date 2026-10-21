<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# S0: capture a harness's contract

Most of the campaign's serving bugs were harness-contract bugs:
- a custom tool delivered as a function;
- a field the harness always sends being rejected;
- an error code the harness does not understand;
- 120 s of silence the harness reads as a dead server.

Write the contract first. Adaptation (S1) and the validator's harness profile are built from
it.

**Outputs:**
- `contracts/<harness>[-<config>].md`, with every line marked [measured], [source] or [doc];
- `profiles/harness/<id>.yaml`, derived per `contracts/TEMPLATE.md` section 8.

The human signs off once per harness version and configuration.

## 1. Point the harness at a recorder

- **Never capture against a vendor endpoint with real keys.** Use a self-hosted target.
- **Record-only** (learn the first request, before any server supports the harness):

  ```bash
  python scripts/record_requests.py --out-dir capture/ --port 8398
  # point the harness at http://127.0.0.1:8398 (Codex: .../v1), with a dummy key
  ```

- **Forward mode** (a server already answers, e.g. trtllm-serve with a small model or a fake
  server). It relays and records both sides byte-exact:

  ```bash
  python scripts/record_requests.py --out-dir capture/ --port 8398 --forward http://127.0.0.1:8000
  ```

- **Or** run trtllm-serve with `TRTLLM_REQUEST_TRACE_DIR=<dir>`. Its request trace also records
  the requests it rejects. That trace writer is on the campaign branches, not yet on main.
- **The recorder records every route.** That covers `/v1/responses`, `/v1/messages` (plus
  `count_tokens`, `HEAD /api/hello`) and `/v1/chat/completions`.
- **Credentials.** The recorder redacts credential headers and forwards them only with
  `--forward-auth`. Bodies are recorded as sent, so treat captures like production traces.

## 2. Run a session catalog

Run sessions that exercise every path. At minimum:
- a plain question;
- one tool call;
- a forced multi-tool turn;
- a failing tool;
- a long tool output (or an image);
- an interrupt/cancel;
- an over-long context (paste until the harness complains);
- every reasoning-effort setting;
- a sub-agent, if the harness has them;
- a compaction, manual and automatic.

Each path is a field or event that otherwise shows up first in production.

## 3. Summarize the wire surface

```bash
python scripts/summarize_capture.py --out summary.json --markdown contract-skeleton.md capture/
```

- **What it reports:**
  - routes;
  - body keys with types and frequencies;
  - enumerated values;
  - every declared tool;
  - input item types, roles and content part types;
  - tool-output shapes;
  - metadata keys;
  - header names;
  - reply statuses, terminal events, error codes and the commonest SSE sequences.
- **What it never prints:** free text or argument values.
- **Today it understands Responses best.** For Anthropic and Chat captures, read message, tool
  and SSE shapes from the turn-record adapters once they land (`validate.md` section 8).

## 4. Read the source (or probe) for what the wire cannot show

Answer each of these with file:line:

| Question | Codex answer (from the campaign) |
|---|---|
| Which event it records items from | `output_item.done` |
| Idle timeouts before and between events | About 120 s of silence and the stream is cut |
| Which errors it retries, and how many times | An HTTP 400 is terminal in the source, yet the KF build re-sent it six times |
| What triggers compaction, and which error means "context full" | Only SSE `response.failed` with `context_length_exceeded` |
| How it matches a tool call to a declared tool | Bare vs namespaced names; custom vs function shape |
| What it replays every turn | Reasoning carriers, tool outputs |

- **Open-source harnesses** (Codex: codex-rs; pi: TypeScript): read the source.
- **Closed-source harnesses (Claude Code):** use the vendor docs, mark them [doc], then measure.
  Run the harness against the fake server of its protocol and inject one fault at a time:
  - silence for N seconds;
  - over-length errors of each wording;
  - unknown fields rejected;
  - signatures rejected.

  Record what it does: whether it retries, compacts, aborts, or drops a capability.

## 5. Write the contract and the profile

- **The contract.** Fill `contracts/TEMPLATE.md`.
  - `contracts/codex-kf.md` is a complete measured example.
  - `contracts/claude-code.md` and `contracts/pi.md` are doc-only starting points with their
    open measurements listed.
- **The profile.** Derive `profiles/harness/<id>.yaml`. Unknown fields stay `TODO`, which makes
  the checks that need them BLOCKED.
- **One profile per configuration.** A default-config Codex and KF's Codex are different
  contracts:
  - top-level `tools` vs an `additional_tools` item;
  - an `instructions` field vs `developer` messages;
  - `parallel_tool_calls` true vs false.

## 6. Traps

- **The model field is the client's, not yours.** KF's Codex sends `gpt-5.6-sol` whatever model
  serves it. Route on pools, not on the model name. Claude Code sends whatever
  `ANTHROPIC_MODEL` says, and its model discovery keeps only ids containing `claude` or
  `anthropic`.
- **Tools may not be where you expect.** Codex KF sends no top-level `tools`; tools come in an
  `additional_tools` input item. A server or checker that only reads `tools` sees a tool-less
  request.
- **Namespaced tools are written both ways by models.** Decide, and test, which spellings the
  server resolves.
- **The session key is not the conversation.**
  - Codex's `prompt_cache_key` is the parent session, and sub-agents share it: group by
    `client_metadata.thread_id`.
  - Claude Code puts sub-agents in `x-claude-code-agent-id` under one session id.
- **A wrapper changes the wire.** KF ran Codex behind LiteLLM, which can add, drop or rename
  fields. Capture the harness the way production will run it.
- **Error wording can be part of the contract.** Claude Code's capability retries match the
  400's message text. Codex compacts only on one error code.
- **Versions drift.** Re-run the capture when the harness version changes, and diff the
  summaries.
