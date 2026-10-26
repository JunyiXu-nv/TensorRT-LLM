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
read. 2026-10-26: 130 of 130 cases passed (Responses 15, Anthropic 24, Chat 26, cross 65). Run it
with the validator's environment (`scripts/validate/requirements.txt`): the receipt binds the
SDK versions as well as the validator tree.

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
- **A second reader earns its place with faults only it reads.** When C-12 was added, five
  modes (`no_event_lines`, `created_without_output`, `wrong_output_index`, `tool_index_gap`,
  `repeated_call_id`) tripped C-12 and nothing else. Each then became an adapter rule, so their
  expected sets now hold both. Any fault the SDK also reads trips C-12 as well, like
  `delta_before_start`.

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
| `unknown_event` | an event type no rule covers (`response.reasoning_summary_text.delta`) | C-11, WIRE-05 |
| `created_without_output` | `response.created` carries a response without `output` | RSP-01, C-12 (SDK-01) |
| `wrong_output_index` | the message's part and text events carry the reasoning item's `output_index` | RSP-15, C-12 (SDK-01) |
| `usage_missing` | `response.completed` has no `usage` (`ok` reports correct usage) | C-14 (USG-01) |
| `usage_total_mismatch` | `total_tokens` is one more than input plus output | C-14 (USG-02) |
| `error_string` | HTTP 400 whose `error` is a bare string, as main answered pi's `store` | RSP-16 |

## 2. Anthropic Messages: `scripts/validate/fake_agent_server.py`

The fault mode is chosen per request with the `x-fake-mode` header.

| Mode | Breaks | Expected set |
|---|---|---|
| `ok` | nothing; includes `ping` events and one `tool_use` | none |
| `ok_thinking` | nothing; includes a thinking block with `signature_delta` | none |
| `overlength_profile` | nothing: the over-length error in the shape the harness profile requires | none |
| `no_message_delta` | the body ends after `content_block_stop` | ANT-08, ANT-10 |
| `no_message_stop` | there is no `message_stop` | ANT-10 |
| `delta_before_start` | a delta arrives before its `content_block_start` | ANT-02, C-12 (SDK-01) |
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
| `unknown_block` | a `server_tool_use` block, which no rule covers | C-11, WIRE-05 |
| `no_event_lines` | every frame is a bare `data:` line | ANT-15, C-12 (SDK-01: the SDK sees no `message_start`) |
| `usage_zero_output` | text and a call delivered, `output_tokens: 0` | C-14 (USG-04) |
| `count_tokens_off` | `count_tokens` answers 3 more than the turn billed; every Messages corpus entry is preceded by its count | C-14 (USG-05) |
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
| `unknown_delta_field` | a `delta.refusal` field, which no rule covers | C-11, WIRE-05 |
| `bad_extension_field` | `avg_decoded_tokens_per_iter` as a string (`ok` sends it as a number, as trtllm-serve does) | CHT-13 |
| `tool_index_gap` | the first call is announced at index 1 | CHT-04, C-12 (SDK-01) |
| `repeated_call_id` | later fragments repeat the call's `id` and `name` | CHT-04, C-12 (SDK-02: openai-python concatenates them) |
| `usage_cached_exceeds` | `prompt_tokens_details.cached_tokens` above `prompt_tokens` | C-14 (USG-03) |
| `usage_requested_missing` | `include_usage` asked for, no usage chunk | C-14 (USG-01) |
| `two_choices` | nothing: `n: 2`, both choices streamed and finished | none |
| `choice_missing` | `n: 2`, only choice 0 streamed | CHT-14 |
| `choice_without_finish` | `n: 2`, choice 1 never finishes | CHT-06 |
| `error_string` | HTTP 400 whose `error` is a bare string | CHT-15 |
| `silent_gap` | no bytes for longer than the idle watchdog | R-02 |

## 4. Cross-protocol cases

`controls.py` implements X-01 to X-06, X-08 to X-10 and X-13 to X-68. X-07 needs H-05, and
X-11 and X-12 need the loop's invalidation log and phase ownership; they are not implemented
yet.

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
| X-13 raw equal | The reply delivers exactly what the GLM-format raw record says | C-13 PASS over 1 joined reply |
| X-14 call lost | The raw record has a call; the reply delivers none | C-13 FAIL (RAW-01) |
| X-15 arguments rewritten | The delivered call's argument differs from the written one | C-13 FAIL (RAW-02) |
| X-16 content rewritten | The delivered text lost a character the model wrote | C-13 FAIL (RAW-03) |
| X-17 model broke its format | The raw call is never closed | M-07 flagged; C-13 does not judge the reply |
| X-18 first call only | Two written calls under `disable_parallel_tool_use`; the first is delivered | C-13 PASS (policy) |
| X-19 no raw output | The same trace without `--raw` | C-13 NOT_APPLICABLE |
| X-20 ambiguous engine id | Two raw records share one `disagg_request_id` | Nothing joins; C-13 BLOCKED |
| X-21 text echo equal | Messages: signed thinking and text come back verbatim at the appended position | C-03 PASS, one text echo equal |
| X-22 text echo differs | The echoed text lost a character | C-03 FAIL (`field: text`) |
| X-23 signature dropped | The echoed thinking lost its signature | C-03 FAIL (`field: reasoning`) |
| X-24 echo policy unknown | The harness profile's `echo.text` is TODO | C-03 BLOCKED |
| X-25 Chat text echo | The echoed assistant `content` differs | C-03 FAIL |
| X-26 server prompt count | The trace's prompt count differs from the billed input | C-14 FAIL (USG-07) |
| X-27 declared offset | Kimi-K3: the trace's prompt count is the billed input plus 3 | C-14 PASS |
| X-28 engine output count | The raw record counts 7 output tokens; the reply billed 1 | C-14 FAIL (USG-06) |
| X-29 next prompt shrank | The turn after an echoed call bills 1 input token where the turn billed 50 | C-14 FAIL (USG-08) |
| X-30 history cleared on request | The same, with `context_management` on the next request | C-14 not tripped |
| X-31 protocol differential, equal | One turn over Chat and over Messages: same request, billed input and reply | `protocol_diff.py` exit 0, `equal` |
| X-32 protocol differential, render | The same, but Messages billed 9 input tokens and Chat 7 | exit 1, `server_render` |
| X-33 protocol differential, harness | The two requests carry different system prompts | exit 0, `harness`: nothing is attributed to the server |
| X-34 protocol differential, divergence | Same request and billed input, other text from the first character | exit 2, `divergent_generation`: two generations, not a protocol path |
| X-35 protocol differential, conversion | The same text delivered as reasoning on one path and as content on the other | exit 1, `server_output` |
| X-36 protocol differential, reasoning divergence | Two turns. Turn 1 generates other reasoning per path; turn 2 replays each path's own and is billed differently | exit 2, turn 2 `after_divergence`: the bill follows the replayed reasoning |
| X-37 protocol differential, harness replay | Two turns. Turn 1 agrees; the harness replays other reasoning over one path on turn 2, billed differently | exit 0, turn 2 `harness` |
| X-38 protocol differential, render after replay | Two turns. Everything replayed agrees and turn 2 is billed differently | exit 1, turn 2 `server_render` |
| X-39 declared rewrite, schema default | Messages: the echoed Edit adds `replace_all: false`, its declared default; the profile declares `schema_defaults` | C-03 PASS, one `echo_rewritten` |
| X-40 undeclared rewrite | The same echo under a profile that declares no rewrite | C-03 FAIL |
| X-41 added value is not the default | The echoed Edit adds `replace_all: true` | C-03 FAIL |
| X-42 declared rewrite, cd prefix | We delivered `cd /w && ls`; the echo is `ls`; the profile declares `cd_cwd_prefix` | C-03 PASS, one `echo_rewritten` |
| X-43 cd added by the harness | We delivered `ls`; the echo is `cd /w && ls`. The cut is one-sided | C-03 FAIL |
| X-44 cd into another directory | We delivered `cd /w && ls`; the echo is `cd /v && ls` | C-03 FAIL |
| X-45 unknown rewrite | The profile declares a rewrite the validator does not know | the pipeline refuses the profile (no silent pass) |
| X-46 raw join by message id | An aggregated server's raw record has no orchestrator id; the Anthropic reply is named `msg_5_<hex>` | C-13 PASS over 1 joined reply |
| X-47 answered twice | The same request is answered twice (a retry after the first answer was lost in transit); the next turn echoes the second | C-03 PASS, one superseded answer, one text echo equal |
| X-48 asked again after an empty turn | The harness drops an empty answer and asks again at the same position, with a text block appended to its last user message (Claude Code); the next turn echoes the second answer | C-03 PASS: one superseded answer, one of them re-asked, one text echo equal |
| X-49 changed, not asked again | As X-48, but the second request changes the last user message instead of appending to it | C-03 FAIL: the first answer is still judged, and its echo differs |
| X-50 the recorder's own replies | The recorder answers 502 while the server is down, and loses another stream mid-reply (`upstream_error`) | C-01 and C-04 not failed; R-06 FAIL, count 2 |
| X-51 a lost request | A request has no response record at all | inventory BLOCKED, one `I-01:request_without_response` |
| X-52 the recorder stopped | A request's record says `recorder_shutdown` | pending at the capture's end: not judged, inventory not blocked |
| X-53 a declared capability rejection | The server rejects `thinking.display` with the wording the harness profile declares; the harness retries without the field | C-16 FLAGGED with the field, conformance PASS, no H-02 |
| X-54 an undeclared rejection | The same 400 with a wording the profile does not declare | C-16 FAIL |
| X-55 an operator probe | A request carries `x-mh-probe` | counted as a probe, not a reply of the cell |
| X-56 a rejected `count_tokens` | `count_tokens` answered 400 with an undeclared wording | C-16 FAIL |
| X-57 cut before the first byte | Against a 30 s first-byte cutoff, one client gave up after 35 s and one after 5 s, both before the reply | R-01 FAIL for the first only |
| X-58 cuts attributed | The same two cuts | H-03 FLAGGED, count 2 |
| X-59 an answer in the next hour | A window's last request is answered in the next hour's shard | without `--responses` inventory BLOCKED; with it PASS, and a response there for another request is ignored |
| X-60 a blind degenerate check | A clean reply, judged with a model profile that names no degenerate tokens, under a lenient threshold | M-02 BLOCKED, not PASS |
| X-61 a degenerate check that can see | The same reply with a profile that names them (K3) | M-02 PASS |
| X-62 parallel sub-agents at one position | Two sub-agents of a session ask with different first messages and the same last one; the second's echo arrives first | C-03 PASS, both texts echo equal (found by call ids) |
| X-63 a side request before the echo | A small request without tools comes between a turn and the request that sends its call back, which grew | No USG-08; one pair |
| X-64 compaction | The request that sends the call back replaced the history with a summary and bills less | No USG-08; `next_input_shrank_by_compaction` 1 |
| X-65 clearing that removes nothing | The echo bills less and carries `clear_thinking` with `keep: all` | USG-08 |
| X-66 clearing that may remove history | The echo bills less and carries `clear_tool_uses` | No USG-08 |
| X-67 a request without thinking (V4.1) | Raw output "OK" with no `</think>` for a Messages request without `thinking` | C-13 PASS on 1 reply, no M-07 |
| X-68 the same output with thinking on | The same raw output for a request with `thinking: adaptive` | M-07 (unclosed think) |

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
