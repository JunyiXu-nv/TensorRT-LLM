---
name: trtllm-serve-model-harness
description: >-
  Support a model x agent-harness pair on trtllm-serve and prove it, as an
  agent loop: capture the harness's wire contract, adapt the model's
  rendering, parsers and stop rules, replay recorded traffic, smoke-test the
  real harness, and let an independent validator issue fail-closed verdicts
  per cell (inventory, conformance, runtime, model behavior, harness outcome)
  over Responses, Anthropic Messages and Chat Completions traffic. Covers
  GLM-5.x, Kimi-K3 and DeepSeek V4.1 with Codex, Claude Code and pi, and any
  new model or harness. Triggers on: new model for agents, new harness,
  Codex / Claude Code / pi against trtllm-serve, harness contract, tool
  parser for agents, reasoning parser, replay traces, smoke test a harness,
  validate a serving change, stream/final mismatch, parallel_tool_calls,
  model x harness matrix, agent support loop.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Support a model x harness pair on trtllm-serve

One skill for the whole job. It replaces five campaign-era skills (harness-contract,
model-adapter, trace-replay, conformance-validate, harness-smoke) that were stages of this
one workflow and called each other's files.

**Where it comes from.** The TRT-LLM x Kernel Factory campaign (09-08 to 10-19) served GLM-5.x
and Kimi-K3 to Codex at about 24M requests. The expensive bugs were harness-contract bugs, not
model bugs:
- a custom tool delivered as a function, so Codex aborted 32.1% of tool results;
- a field Codex always sends rejected 6,300 times;
- an HTTP 400 where Codex needed an in-stream failure, so it resent forever;
- more than 120 s of silence during prefill, so the client cut the stream.

Each was cheap to fix once known. Each could have been found without production scale, given
the right inputs and the right checks. That is what this skill provides.

## 1. The matrix

- **A cell is `(model profile, harness profile)`.** The harness profile fixes the protocol,
  so pi over Chat Completions and pi over Anthropic Messages are different cells.
- **Work per axis, verify per cell.**
  - Model adaptation (renderer, tool and reasoning parsers, stop rules) is shared by every
    harness.
  - trtllm-serve has two output paths:
    - `/v1/responses` has its own path;
    - `/v1/messages` is converted onto the `/v1/chat/completions` path, so the two share parsers.
  - Harness quirks live in the contract and the harness profile.
- **Profiles are data.** Both live under `profiles/`:
  - `profiles/model/*.yaml`: markers, special and degenerate tokens, parsers, renderer source;
  - `profiles/harness/*.yaml`: protocol, routes, identity keys, outcome strings, timeouts,
    error mapping.

  Both are derived from S0 and S1 outputs, and the validator reads them. A new model or
  harness is a new profile, not new checker code.
- **State.** Copy `templates/matrix.yaml` to a run directory outside the repository. Each
  cell's status is computed from its latest verdicts, never typed by hand.

## 2. Roles

| Role | Does | Never does |
|---|---|---|
| adapter agent | Contracts, adaptation, fixes, replays, smoke runs, fix reports | Edits validator-owned files in a cycle, or writes a verdict |
| validator agent | Freezes evidence, runs and owns the checks and controls, issues full-tier verdicts | Runs the adapter's scripts as evidence, or fixes serving code |
| investigator (on demand) | Runtime/kernel root causes with env-gated diagnostics | Restarts serving |
| human | Resources, thresholds, external actions (pushes, PRs, issues), policy choices, entering S5 | — |

- **The validator should be a different session, ideally a different vendor's model.** In the
  campaign a Codex validator checking Claude's work caught defects the operator could not see:
  - a "492 passed" that was a `-k glm` subset;
  - wire receipts stored truncated;
  - a 1-in-187,895 `parallel_tool_calls` violation.

  It also failed all three runtime claims put to it.
- **Validator-owned files:** `scripts/validate/`, `references/validate.md`,
  `references/protocols.md`, `references/turn-ir.md`, `references/acceptance-cases.md`, and
  the thresholds in the run's matrix file. Changing any of them is a policy change:
  - a separate commit;
  - the control matrix re-run;
  - an iteration-log entry.

  It is never a fix to a failing cell.

## 3. Stages and exit gates

| Stage | Do | Exit gate | Read |
|---|---|---|---|
| S0 contract | Record the harness, read its source or docs, and write the contract and harness profile. Write the model format card and model profile | Human signs off once per harness version and configuration | `references/contract.md`, `contracts/` |
| S1 adapt | Golden renders against the model's own encoder; parser fault matrix; stop rules; regression tests | Renders byte-equal; zero `calls_differ`, `streams_disagree` and `text_lost`; every new test failed on the old code | `references/adapt.md` |
| S2 regress | L1 frame replay and L2 fake engine over existing corpora, for every model already supported | Fast-tier `conformance` PASS on every regression corpus | `references/replay.md` |
| S3 replay | L3: recorded harness requests, one turn at a time, against the new model | Full tier: `inventory` and `conformance` PASS; `model-behavior` within thresholds, or BLOCKED for insufficient n | `references/replay.md`, `references/validate.md` |
| S4 smoke | L4: the real harness on local tasks, in isolation | Full tier: S3's claims plus `runtime` and `harness-outcome` | `references/smoke.md` |
| S5 scale | Soak or a Kernel Factory campaign for long-tail rates | Human decision; same checks | `trtllm-kf-campaigns` and related skills (campaign branches) |

A harness with no recorded traffic (Claude Code or pi today) runs S4 wiring and recording
before S3. Its first real traffic comes from L4.

## 4. The loop (per cell)

1. **Pick work.** Choose the lowest-stage cell that is not PASS from the matrix status. Cells
   that share a model or path move together: one parser fix serves every harness.
2. **Run the stage** and write its evidence under the run directory: `cells/<cell>/<stage>/<iteration>/`.
3. **Fast verdict.** Run `scripts/validate/` on the evidence. Read the verdict JSON, not only
   the exit code (see section 7).
4. **On FAIL:**
   - make the smallest reproducer;
   - write a test that fails on the old code;
   - fix the responsible layer (renderer, parser, protocol path or harness configuration);
   - re-run S2 for every model the layer touches, then repeat the stage.
5. **At a stage exit:** freeze the evidence with a `MANIFEST.sha256` and hand it to the
   validator for the full tier. Update the status only from that verdict.
6. **Budget.** Allow at most 3 repair cycles per finding class. Stop when two consecutive cycles
   reproduce the same blocker with no new evidence. Adding evidence does not consume a cycle.
7. **Escalate to the human** for:
   - GPU or time beyond the matrix budget;
   - pushes, PRs or upstream issues;
   - a policy choice, e.g. what serving should do when the model breaks a constraint;
   - a threshold;
   - an exhausted budget.

## 5. Rules that are not negotiable

- **Fail closed.** Missing evidence is BLOCKED, never PASS and never NOT_APPLICABLE.
  NOT_APPLICABLE needs a reason based on a feature that is absent.
- **Never weaken a gate to pass,** and never rename an objective to close one.
- **Never print trace content.** Use locators `(file, line, trace_id)`, lengths and SHA-256.
  Traces carry leaked platform tokens and user code. Never commit a corpus.
- **Prove what served.** Use process-level identity: `build_id` on every record, plus
  `provenance.txt`. A moved checkout proves nothing.
- **A regression test must fail on the old code.** A collection error on the old code is not a
  failure.
- **Do not hide model behavior in serving.** Attribute with the raw generation first, and keep
  a conversation on one model.

## 6. Map

| File | Read when |
|---|---|
| `references/contract.md` | S0: capturing a harness and writing its contract and profile |
| `contracts/*.md` | The Codex (KF and default configs), Claude Code and pi contracts, and the template |
| `references/adapt.md` | S1: renderer, parsers, stop rules, fault matrix, fault catalog |
| `references/replay.md` | S2/S3: L1 frames, L2 fake engine, L3 per-turn replay, shadow traffic |
| `references/smoke.md` | S4: isolated real-harness runs and harness-side signals |
| `references/validate.md` | Any verdict: claims, the gate-to-check catalogue, rate rules, tiers, outputs |
| `references/turn-ir.md` | Writing or reading a protocol adapter |
| `references/protocols.md` | Wire rules for Responses, Anthropic Messages and Chat Completions |
| `references/acceptance-cases.md` | Controls that qualify the validator itself |
| `profiles/`, `templates/matrix.yaml` | Model and harness profiles; the run's matrix, thresholds and budget |
| `tests/test_validator.py` | Checking the validator after a change (`MH_RUN_CONTROLS=1` for the matrix) |

## 7. Scripts

| Script | Stage | Scope |
|---|---|---|
| `scripts/validate/verdict.py` | all | **The validator.** Responses, Anthropic Messages and Chat Completions; claims, rate bounds, exit codes 0/1/2/3 |
| `scripts/validate/sdk_diff.py` | all | A second reading of every reply by the official `anthropic` and `openai` SDKs (check C-12); run by `verdict.py` |
| `scripts/validate/native.py`, `native_goldens.py` | all | The raw view (C-13): the model's own text read in its own format; goldens rendered by each checkpoint's encoder |
| `scripts/validate/usage.py` | all | Usage rules (C-14): wire fields, totals, `count_tokens`, the engine's and the server's own counts |
| `scripts/validate/requirements.txt` | all | The validator's environment, SDK versions pinned: `python3 -m venv <dir> && <dir>/bin/pip install -r requirements.txt` |
| `scripts/validate/controls.py` | qualification | Control matrix for all three protocols plus cross cases; writes `controls.json` bound to the validator tree |
| `scripts/validate/fake_agent_server.py`, `fake_responses_server.py` | controls, S4 wiring | Model-free servers, correct or broken on purpose |
| `scripts/validate/mutate_traces.py` | qualification | Seeded faults on recorded Responses replies (negative controls on real traffic) |
| `scripts/validate/scan_credentials.py` | all | Any shard |
| `scripts/validate/check_stream.py`, `check_tool_calls.py`, `harness_signals.py`, `run_controls.sh` | Step 3 | Responses-only reference implementation, kept for calibration |
| `scripts/record_requests.py` | S0, S4 | Any route; record-only or forward; per-chunk timing |
| `scripts/summarize_capture.py` | S0 | Mostly Responses |
| `scripts/fault_mutations.py`, `scripts/chunk_equivalence.py` | S1 | Any tool parser; card-driven |
| `scripts/build_replay_corpus.py`, `scripts/replay_requests.py` | S3 | All three routes (`--route`); thread keys and reasoning transforms per protocol |
| `scripts/count_tokens_corpus.py` | S4 | A `count_tokens` probe from recorded Messages requests, replayed into the same trace (USG-05) |
| `scripts/run_codex_sessions.sh`, `scripts/run_claude_sessions.sh` | S4 | Codex; Claude Code (`env -i`, isolated config). A pi runner is not written yet |

- **Tests.** `tests/test_validator.py` holds CPU-only tests. Set `MH_RUN_CONTROLS=1` to run
  the full control matrix.
- **Run the validator in its own environment** (`scripts/validate/requirements.txt`). The SDK
  versions are part of what a verdict means: without an SDK, C-12 is BLOCKED for its protocols,
  and a control receipt qualifies only the SDK versions it was produced with.
- **Read the verdict JSON, not only the exit code.** A BLOCKED claim names its blocker in
  `claim_blockers`. Typical blockers:
  - unset thresholds;
  - a harness profile that is not yet `measured`;
  - missing timing or `build_id`.
