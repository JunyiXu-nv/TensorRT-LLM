<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# S4: smoke-test the real harness

Replay checks single turns. It cannot show what the harness does with our replies.

The campaign's worst serving bug was invisible on the server side:
- every call was well-formed, but a custom tool went out as a `function_call`;
- Codex answered `aborted` to 32.1% of tool results;
- agents concluded their tools were broken.

A smoke run runs the harness itself, then reads the harness's verdict back from its own next
requests.

For a harness with no recorded traffic yet (Claude Code, pi), S4's wiring run is also how its
first corpus is made.

## 0. Isolation and safety

- **Agents run model-generated commands.** Use a disposable container or VM per run, with no
  access to the user's home, the NFS workspace or cluster credentials.
- **One configuration home per session,** never the user's:

  | Harness | Variable |
  |---|---|
  | Codex | `CODEX_HOME` |
  | Claude Code | `CLAUDE_CONFIG_DIR` |
  | pi | a fresh `HOME`, or pi's own directory variable |

  The dev host holds `~/.codex/auth.json` and Claude MCP credentials. None of it may be visible
  to a session.
- **No real credentials in the environment the agent sees.** Agents print environment and
  headers; the campaign leaked platform tokens through `curl -v`.
- **Restrict egress.** Claude Code's fast-mode check and WebFetch domain check call
  `api.anthropic.com` directly. Under no-egress they fail, and that is expected noise.

## 1. Wire it up without a model first (minutes, no GPU)

```bash
python scripts/validate/fake_responses_server.py --port 8399 --max-tool-turns 2 &
python scripts/record_requests.py --out-dir run/trace --port 8397 --forward http://127.0.0.1:8399 &
printf '%s\t%s\n' ./task_src "Run hello.py and tell me what it prints." > tasks.tsv
BASE_URL=http://127.0.0.1:8397/v1 MODEL=fake scripts/run_codex_sessions.sh tasks.tsv run/out
python scripts/validate/verdict.py --harness profiles/harness/codex-default.yaml \
    --model profiles/model/<id>.yaml --population smoke --out-dir run/verdict run/trace
```

How it behaves:
- The fake server answers with a call to the first declared tool, and with `--max-tool-turns`
  it ends the turn.
- Codex executes the call and finishes.

This was verified on 2026-10-19 with codex-cli 0.155.1: 2 sessions, exit 0, 6 streams and 4
calls, all conformant.

**Claude Code.** Same wiring, with `fake_agent_server.py` and `run_claude_sessions.sh`:

```bash
python scripts/validate/fake_agent_server.py --port 8400 --max-tool-turns 1 &
python scripts/record_requests.py --out-dir run/trace --port 8397 --forward http://127.0.0.1:8400 &
printf '%s\t%s\n' - "Read the file named x and summarize it." > tasks.tsv
BASE_URL=http://127.0.0.1:8397 MODEL=fake-model scripts/run_claude_sessions.sh tasks.tsv run/out
python scripts/validate/verdict.py --harness profiles/harness/claude-code.yaml \
    --model profiles/model/<id>.yaml --population smoke --out-dir run/verdict run/trace
```

- **Run every session under `env -i`.** A Claude Code parent session exports `CLAUDE_CODE_*`
  variables, including a messaging socket and token, and a child must not inherit them. The
  runner does this.
- **Verified 2026-10-22 with Claude Code 2.1.289:** 5 sessions, exit 0, every request and reply
  parsed, echo equal. The measured facts are in `contracts/claude-code.md` section 9.
- **`fake_agent_server.py --mode <fault>`** shows what the harness does with a broken reply, e.g.
  `undeclared_tool` and `invalid_input` gave the measured tool-error strings.

**pi.** It needs a runner (`pi --print` or `pi --mode json`) with a per-session home and a
`models.json` per `contracts/pi.md`. pi is not installed yet, so this is not written.

**A wiring failure is a harness configuration problem, not a model problem.** Fix it before
spending GPU time.

## 2. The real run

1. **Serve the model with tracing on:**

   ```bash
   TRTLLM_REQUEST_TRACE_DIR=run/trace trtllm-serve <model> --tool_parser <name> \
       --reasoning_parser <name> [--custom_tokenizer <name>] ...
   ```

   - Put a gateway in front of it if the run should look like production. Keep session affinity,
     and serve the model name the harness sends.
   - Configure the harness's context window to the served `max_seq_len`:
     - Codex: `model_context_window`;
     - Claude Code: its model config;
     - pi: `contextWindow`.

     Otherwise compaction triggers at the wrong point.
2. **Choose tasks the agent can complete locally.** Small repositories with scripted bugs work.
   Include tasks that force:
   - several tool calls;
   - a failing tool;
   - long outputs;
   - a long session that reaches compaction (set the window low to trigger it);
   - a sub-agent, where the harness has them.
3. **Run enough sessions to see rates, not anecdotes.**
   - Tens of sessions for wiring; hundreds before a campaign.
   - Set concurrency to reflect real load, since prefill under load is what trips harness
     timeouts.

## 3. Judge both sides

**Both sides at once.** `verdict.py --population smoke` judges the server side (`conformance`,
`runtime`) and the harness side (`harness-outcome`) from the same trace. Add
`scan_credentials.py` when tasks touch secrets.

**Harness side.** Each call's first reported outcome is read from the next request of its
thread. The strings come from the harness profile's `outcomes`: Codex's are measured, and Claude
Code's were measured 2026-10-22.

| Outcome | Meaning |
|---|---|
| `aborted` | The harness cancelled the call. The custom-vs-function shape is the usual cause |
| `unsupported` | The harness does not know the tool |
| `arguments_rejected` | The arguments do not match the tool schema |
| `unavailable_in_mode` | The tool is not available in the harness's current mode |
| `result` | Anything else: the call ran (success is not implied) |

Also counted:
- identical resends, which mean a reply the harness did not understand (e.g. a 400 where it
  expected an in-stream failure);
- client disconnects, often a first event later than the harness's idle cutoff;
- failed terminal codes;
- compaction requests, and whether they were answered substantively.

### Reference rates (Kimi-K3 under KF Codex, 2026-10-16, one hour, 151 threads)

| Outcome | Count | Rate |
|---|---|---|
| `result` | 21,792 of 21,807 | 99.93% |
| `unsupported` | 14 | 0.064% (the model calling tools nested inside `exec` directly) |
| `aborted` | 1 | 0.005% |
| Resend | 1 | |
| Disconnects | 10 of 7,372 replies | |

Before the bare-name fix, `aborted` was 32.1%.

### Exit criteria

Set these in the run's matrix file with the human before starting; `validate.md` section 3 has
the rate rules.

- **Zero** serving-shape and contract failures, and views equal on 100% of replies.
- **`aborted` and `arguments_rejected` below an agreed rate,** each case traced to the raw
  generation.
- **Zero identical resends** caused by our error mapping.
- **Every disconnect attributed** to the client, a timeout or the server.

## 4. What a smoke run still cannot show

Long-tail rates:
- a 1-in-187,895 defect (B009-01);
- a kernel race that corrupts 1 in 256 steps under load (the K3 `[PAD]` loops).

Both need 10^5–10^6 turns at production concurrency. That is S5: a large local soak or a Kernel
Factory campaign, still validated with the same checks.

## 5. Traps

- **Configuration changes the contract.** Smoke-test the configuration production uses
  (`contracts/codex-default.md` vs `contracts/codex-kf.md`).
- **A fake server that always calls a tool never ends the conversation.** Codex looped for 120 s
  and sent 1.9 GB of history. Use `--max-tool-turns`.
- **Harness logs are a secondary signal.**
  - Codex `--json`, Claude Code `stream-json` and pi's session JSONL explain individual sessions.
  - The rates come from the server's trace, which records exactly what the harness sent back.
- **Run the harness the way production does.** A wrapper (LiteLLM, a gateway) can add, drop or
  rename fields.
