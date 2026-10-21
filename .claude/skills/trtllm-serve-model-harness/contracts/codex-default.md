<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Codex, default configuration: pointing it at trtllm-serve

Harness profile: `profiles/harness/codex-default.yaml`.

Checked against codex-cli 0.155.1 on 2026-10-19. Re-check `codex exec --help` for other versions.

## Provider configuration

Codex talks to a custom provider through `model_providers`. These overrides need no config file
(`scripts/run_codex_sessions.sh` passes them):

```bash
codex exec --json --skip-git-repo-check -C <workdir> -s workspace-write -m <model> \
  -c 'model_provider="trtllm"' \
  -c 'model_providers.trtllm={name="trtllm-serve", base_url="http://<host>:<port>/v1", wire_api="responses", env_key="TRTLLM_API_KEY"}' \
  -o last_message.txt "<task prompt>"
```

| Setting | Notes |
|---|---|
| `wire_api="responses"` | Codex then POSTs `/v1/responses` with `stream: true`. Kernel Factory bakes the same into its Codex home template |
| `env_key` | Names the environment variable holding the key. Set it to a dummy when the endpoint needs no auth |
| `model_context_window` | Codex compacts when its own context count reaches this. Kernel Factory sets it to the served window, or to a ceiling when compaction is off |
| `model_auto_compact_token_limit` | Related compaction setting |
| `model_reasoning_effort` | Codex otherwise sends whatever its config has. With no value, the endpoint's default applies |
| `CODEX_HOME=<dir>` | Per run, so nothing reads or writes the user's `~/.codex`. Codex warns that it cannot create helper binaries under `/tmp`; that is harmless |

## What changes with the configuration

The same Codex sends a different contract depending on its configuration. Both of these were
captured with `scripts/record_requests.py` (see `references/contract.md`):

| | codex-cli 0.155.1, default config | Kernel Factory's Codex (via its LiteLLM gateway) |
|---|---|---|
| Tools | Top-level `tools`: `exec_command`, `write_stdin`, `view_image`, `multi_agent_v1.*`, goal tools, `web_search` (all JSON functions) | No top-level `tools`. An `additional_tools` input item: `functions.exec` (custom, Lark grammar), `wait`, `collaboration.*` |
| Instructions | `instructions` field | `developer` messages |
| `parallel_tool_calls` | `true` | `false` |
| Reasoning | `reasoning.summary: "auto"` | `reasoning: {"context": "all_turns"}` |
| Model id | as configured | `gpt-5.6-sol` whatever serves it |

Smoke-test the configuration that production will use. A pass on one says little about the other:
a custom `exec` with a grammar and a JSON `exec_command` exercise different parser paths.

## Kernel Factory specifics (from its agent entrypoint, captured 2026-10-05)

- Codex runs through an SDK runner with an in-pod LiteLLM gateway. Only `client_metadata` is
  forwarded as an extension field.
- The endpoint must be on KF's allowlist, which is the address the gateway runs on.
- Agents run in sandboxes with a platform `$API_TOKEN`. Expect that token in tool outputs (`curl -v`),
  and scan for it (`scripts/validate/scan_credentials.py`).
