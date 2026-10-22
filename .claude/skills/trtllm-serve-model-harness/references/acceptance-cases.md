<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Acceptance cases: controls that qualify the validator

Validator-owned. A check that never fires proves nothing. A validator version, identified by
its tree SHA-256, issues verdicts only after its control matrix passes (check V-01).

## Running the matrix

```bash
python scripts/validate/controls.py --out-dir <dir>      # all protocols + cross cases -> <dir>/controls.json
MH_RUN_CONTROLS=1 pytest .claude/skills/trtllm-serve-model-harness/tests
```

`controls.py` builds its own synthetic corpora and control harness profiles, so no trace is
read. 2026-10-22: 54 of 54 cases passed (Responses 9, Anthropic 20, Chat 16, cross 9).

## 0. How a control is judged

- **Every fault mode declares its expected set.** That is the check it must trip, plus its
  declared consequences. For example, `delta_differs` trips `delta_done_mismatch` and also
  `done_item_mismatch`.
- **A fault passes when it trips its expected set and nothing outside it.** `ok` and the other
  normal controls must trip nothing.
- **Normal controls also run on real traffic.** Use the last known-good window: today that is
  campaign batch-010 for Responses.
- **Wording-only tests do not count.** A test that only searches for text or echoes the
  expected status does not qualify a check.
- **After any edit to a check, re-run the whole matrix.** An early status check failed every
  K3 reply, and only a normal control on real traffic exposed it.

## 1. Responses: `scripts/validate/fake_responses_server.py`

`controls.py` runs these through `verdict.py`. The reference checkers still run them with
`scripts/validate/run_controls.sh <corpus.jsonl> <out> [python]`, given any recorded or
synthetic Responses requests with tool declarations. On 2026-10-21, after the move into
this skill, a 5-request synthetic corpus passed 9/9.

| Mode | Breaks | Expected set |
|---|---|---|
| `ok` | nothing | none |
| `overlength` | nothing: a correct `response.failed` with `context_length_exceeded` | none |
| `drop_part_done` | a `content_part.done` is missing | `unpaired_content_part` |
| `snapshot_text_differs` | the snapshot text differs from the stream | `item_content_differs` |
| `delta_differs` | the joined deltas differ from the done event | `delta_done_mismatch`, `done_item_mismatch` |
| `no_terminal` | there is no terminal event | `no_terminal_event` |
| `custom_as_function` | a custom tool is sent as `function_call` | `type_mismatch` |
| `two_calls` | two calls under `parallel_tool_calls=false` | `ptc_violation` |
| `prefixed_name` | a namespace-prefixed name | `prefixed_name_of_declared` |

## 2. Anthropic Messages: `scripts/validate/fake_agent_server.py`

The fault mode is chosen per request with the `x-fake-mode` header.

| Mode | Breaks | Expected set |
|---|---|---|
| `ok` | nothing; includes `ping` events and one `tool_use` | none |
| `ok_thinking` | nothing; includes a thinking block with `signature_delta` | none |
| `overlength_profile` | nothing: the over-length error in the shape the harness profile requires | none |
| `no_message_delta` | the body ends after `content_block_stop` | ANT-08, ANT-10 |
| `no_message_stop` | there is no `message_stop` | ANT-10 |
| `delta_before_start` | a delta arrives before its `content_block_start` | ANT-02 |
| `interleaved_blocks` | block 1 starts before block 0 stops | ANT-03 |
| `delta_type_mismatch` | a `text_delta` inside a `tool_use` block | ANT-04 |
| `bad_partial_json` | the joined `partial_json` is not a JSON object | ANT-06 |
| `duplicate_tool_use_id` | two `tool_use` blocks share an id | ANT-06 |
| `stop_reason_mismatch` | a `tool_use` block with `stop_reason: end_turn` | ANT-09 |
| `two_tools_disable_parallel` | two `tool_use` blocks under `disable_parallel_tool_use` | C-07 |
| `undeclared_tool` | the call names a tool the request did not declare | M-01 (`undeclared`); conformance must stay PASS |
| `invalid_input` | the call's input misses a required property | M-01 (`payload_invalid`) |
| `html_body` | an HTML page instead of SSE or JSON | WIRE-01 |
| `json_for_stream` | a JSON message for a `stream: true` request | WIRE-03 |
| `error_untyped` | an error event without `error.type` | ANT-11 |
| `content_after_stop` | an event after `message_stop` | ANT-10 |
| `silent_gap` | no bytes, ping included, for longer than the profile's idle watchdog | R-02 |
| `overlength_400_wrong_wording` | over-length as a 400 whose wording the harness does not recognize | C-10, when the profile requires recognizable wording |

## 3. Chat Completions: `scripts/validate/fake_agent_server.py`

| Mode | Breaks | Expected set |
|---|---|---|
| `ok` | nothing; one tool call with fragmented arguments | none |
| `ok_reasoning` | nothing; `reasoning_content` deltas, then content | none |
| `overlength_profile` | nothing: the over-length error in the harness profile's shape | none |
| `no_done` | there is no `[DONE]` | CHT-09 |
| `two_finish_reasons` | a second non-null `finish_reason` | CHT-06 |
| `fragment_without_id` | a tool-call index whose first fragment has no `id` or `name` | CHT-04 |
| `arguments_not_object` | the joined arguments are not a JSON object | CHT-05 |
| `finish_reason_mismatch` | tool calls with `finish_reason: stop` | CHT-07 |
| `duplicate_tool_call_id` | two indices share an id | CHT-04 |
| `two_calls_ptc_false` | two indices under `parallel_tool_calls: false` | C-07 |
| `reasoning_in_content` | think markup in `delta.content` | C-09 (flag), M-03 |
| `model_changes` | `model` differs between chunks | CHT-02 |
| `invalid_input` | the call's arguments miss a required property | M-01 (`payload_invalid`) |
| `html_body` | an HTML page instead of SSE or JSON | WIRE-01 |
| `json_for_stream` | a JSON completion for a `stream: true` request | WIRE-03 |
| `silent_gap` | no bytes for longer than the idle watchdog | R-02 |

## 4. Cross-protocol cases

`controls.py` implements X-01 to X-06 and X-08 to X-10. X-07 needs H-05, and X-11 and X-12
need the loop's invalidation log and phase ownership; they are not implemented yet.

Build these as fixture traces with an adapter-produced next turn. No server is needed.

| Case | Stimulus | Expected decision |
|---|---|---|
| X-01 echo mismatch | The harness's next request carries a different payload under our call id | C-03 FAIL |
| X-02 echo lossless | The next request re-serializes our JSON arguments with different spacing and key order | C-03 PASS (kit C06) |
| X-03 pending at cutoff | A delivered call has no result before the window's cutoff | I-08 counts it as pending; not a failure (kit C07) |
| X-04 outcome strings | Tool results carry each `outcomes` string in the harness profile | H-01 classifies each correctly; an unknown string becomes `result` with a count of unmatched strings |
| X-05 unknown outcome profile | The harness profile has no `outcomes` | H-01 is BLOCKED, not PASS |
| X-06 vacuous turn | The reply is only whitespace and EOS, with no call | M-06 counts it (kit C13). A reply with an empty text but a valid call is not vacuous |
| X-07 compaction placeholder | The harness's compaction request is answered with an unexecuted tool tag in place of a summary | H-05 flags it (kit C09) |
| X-08 client disconnect | The reply has no terminal event and the transport status is a client disconnect | C-04 is not failed; H-03 attributes it |
| X-09 degenerate stop | A `[PAD]` run is stopped and the reply fails with no output | R-04 PASS; M-02 counts the stop |
| X-10 rate bound | 0 events in 20 sessions against a 1% threshold | BLOCKED (insufficient n), not PASS (kit C25) |
| X-11 invalidated assurance | A control proves an old check skipped a case; the hashes are unchanged | The dependent PASS verdicts are recorded as invalidated (kit C30) |
| X-12 stale owner | Two loop controllers act on the same ready stage | Only one acts; the stale owner's write is refused (kit C28) |

## 5. Calibration with seeded bugs (Step 3)

**Trace-level seeded faults (done 2026-10-22).** `scripts/validate/mutate_traces.py` copies
recorded replies whose single call is valid, and gives each copy one fault in the shape of a
campaign bug, under a 0700 directory removed at exit. On batch-010:
- 194 K3 replies and 197 GLM replies;
- every fault was caught on every copy;
- the unmutated control stayed clean.

See `validate.md` section 8.

**Code-level seeded bugs (open).** Revert the fix, rebuild, and serve.

Revert one campaign fix at a time on a throwaway branch, rebuild the evidence, and confirm
detection. A miss is a validator defect.

| Revert | Expected detection |
|---|---|
| `b52237f18c` bare and qualified tool names | C-06 `type_mismatch` or `prefixed_name_of_declared`; H-01 `aborted` in a Codex smoke run |
| `d66713abe7` `service_tier` accepted | Replies with HTTP 400 for the Codex corpus; H-02 resends |
| `d1b379e870` in-stream `context_length_exceeded` | C-10 FAIL; H-04 FAIL |
| `f8efc82403` / `78ac9978b2` keepalive and early-open | R-01 / R-02 against the 120 s cutoff, with a slow fake engine |
| `a3e176fef0` `single_call_stop` | C-07 FAIL with a two-call model sample |
| `be7746ef53` / `e0a8699934` snapshot equals stream | C-02 FAIL |
| `a13ecfbb50` degenerate stop | R-04 FAIL with a `[PAD]` run sample |

These commits are on the campaign branches (`dev-kf-trtllm-agent-evaluation`,
`dev-junyix-feat-kimi-k3-serving`). Calibrate on a branch that carries them. The open PR #19883
carries some of them onto main.
