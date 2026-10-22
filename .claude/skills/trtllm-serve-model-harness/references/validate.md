<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Validation: claims, checks, verdicts

Validator-owned. Changing this file is a policy change (see `SKILL.md` section 2).

## 0. Where the rules come from

- **The sft-data-validation kit v3.1** (`tmp/sft-data-validation-kit/`).
  - It has 20 gates, a claim/verdict algebra, and a loop and evidence contract.
  - The campaign's independent validator used it on 11 reports (discovery-001, batch-001 to
    010), which hold 54 finding records. Their primary gates were G01 14, G02 9, G04 8, G19 6,
    G05 5, G03 4, G12 3, G20 3, G06 1 and G17 1.
  - G09/G10 (evaluation) and G13–G15 (training) never fired. Only the gates that fired carry
    over here.
- **The campaign's serving verdict rules:** three claim levels, native-text-first attribution,
  the deployment check, window discipline.
- **What is new here:**
  - wire rules for three protocols (`protocols.md`);
  - a second reading of every reply by the official SDKs, compared with ours (C-12);
  - one turn record that every check reads (`turn-ir.md`);
  - per-harness outcome classes;
  - rate rules with intervals;
  - two tiers and exit codes, so an agent loop can stop on them.

## 1. Claims

A verdict is issued per cell, per claim and per evaluated population (a frozen window, a replay
run or a smoke run). Each claim states its population, its unit and its denominators.

| Claim | Question | Verdict style |
|---|---|---|
| `inventory` | What was evaluated, what is missing, and which code served it | Reconciliation. Missing data is stated, never "complete" |
| `conformance` | Did trtllm-serve speak this harness's protocol correctly, for each request's own declarations? | Hard: any confirmed violation is FAIL |
| `runtime` | Is the running process the code under test, and does it behave correctly under overload, over-length and silence? | Hard, using controls with normal and adverse cases |
| `model-behavior` | With serving correct, how often does the model misbehave in this harness? | Rate against a threshold (section 3) |
| `harness-outcome` | Does the harness accept our replies: no aborts, rejections or resends caused by us? | Rate against a threshold, plus hard zero for resends caused by our error mapping |
| `task-outcome` (L4, optional) | Did tasks succeed? | Needs a qualified oracle (kit G09/G10); reference agreement alone is not task correctness |
| `comparison` (optional) | Is A better than B: models, harnesses, or us against a vendor API? | Comparable populations, with selection and dependence handled (kit G18) |

**Healthy serving is not a good model, and a good model is not task success.** These are
separate claims, and none of them substitutes for another.

## 2. Verdict algebra

- **Per claim:**
  - FAIL wins over everything;
  - otherwise, missing required evidence is BLOCKED;
  - PASS needs every applicable check to have run over a stated denominator;
  - NOT_APPLICABLE needs a reason based on an absent feature, e.g. no request set
    `parallel_tool_calls=false`.
- **Three claim levels, never merged:**
  - *flagged*: a screen fired;
  - *content change proven*: two views of one generation differ, or a replay reproduces the
    difference;
  - *failure proven*: a consumer-side causal chain exists, e.g. the harness answered `aborted`
    to the delivered call.

  Report each level with its own denominator.
- **Attribution starts from the native text.** If the anomaly is already in the raw generation,
  the model wrote it, and serving can only be guilty of making two views diverge or of rewriting
  bytes. Unknown causes stay unknown, and correlation is not attribution.
- **Corrections are additive.** Published verdicts are never edited. New evidence that disproves
  an assurance invalidates every verdict that depended on it (kit G20). Record the invalidation,
  then re-check or BLOCK the affected claims.

## 3. Rates and thresholds

- **Thresholds come from the run's matrix file and are set by the human before the run.** The
  kit sets no universal quality percentage and neither does this skill. An unknown threshold
  makes the claim BLOCKED, with the measurement reported.
- **The interval.** For `k` events in `n` units, compute the one-sided 95% Wilson interval:
  - PASS when its upper bound is at most the threshold;
  - FAIL when its lower bound exceeds the threshold;
  - otherwise BLOCKED (insufficient n).
- **Zero observed is not a zero rate.** With 0 events in n independent units, the 95% upper bound
  is about 3/n:
  - claiming below 0.1% needs about 3,000 units;
  - a 1-in-187,895 defect needs S5 scale.
- **Units.**
  - Declare the unit for every rate: call, turn or session.
  - Turns of one session are not independent, so the full tier also reports the session-level
    rate (sessions with at least one event) and uses it for the verdict.
- **Semantic review.** For claims judged by sampling, such as "compaction summaries are
  substantive" (kit G12), record the rubric, strata, sample IDs and seed. A model-judge score is
  not ground truth.

## 4. Gates mapped to checks

IDs: `I` inventory, `C` conformance, `R` runtime, `M` model behavior, `H` harness outcome,
`V` validator closure. The "Now" column says for which protocols `scripts/validate/verdict.py`
implements the check today ("all" = Responses, Anthropic Messages and Chat Completions).

| ID | Gate | Check | Verdict rule | Now |
|---|---|---|---|---|
| I-01 | G01 | Joins: request ↔ response ↔ raw ↔ next turn. Ids above 2^53 are compared as decimal strings | Unjoined records are counted. A required join that is missing gives BLOCKED_PARTIAL. An engine id naming two raw records joins nothing | all |
| I-02 | G01 | Counts reconciled per unit (requests, streams, calls, turns, sessions) against the writer sidecars | Mismatch is BLOCKED until explained | planned |
| I-03 | G01 | Freeze manifest present and verified (full tier) | Missing gives BLOCKED | planned |
| I-04 | G01 | Credential scan: digests and counts only | NEW digests are reported to the human and the contexts held | all (scan_credentials.py) |
| I-05 | G02 | `build_id` on every record, and equal to the declared revision | Missing or mixed gives BLOCKED for the claims that name a revision | all |
| I-06 | G02 | Cell identity resolvable: model profile and template hash, harness profile and version, protocol | Unresolved gives BLOCKED | partial (profiles bound) |
| I-07 | G02 | Capture boundary declared: recorder, request trace, replay or mirror | Missing gives BLOCKED | planned |
| I-08 | G05 | Every delivered call is answered next turn, or pending at the cutoff | Unanswered calls are counted; pending is not a failure | all |
| I-09 | G06 | Sessions, threads and sub-agents grouped by the harness profile's identity keys | Ungroupable gives BLOCKED for session-level rates | all |
| I-10 | G17 | Strata coverage: tool kinds, effort levels, context buckets, sub-agents, images, errors, compaction | An empty stratum BLOCKS any claim that needs it | planned |
| C-01 | G03 | Wire rules for the protocol (`protocols.md`) | Any violation is FAIL | all |
| C-02 | G03 | Stream equals the final snapshot item by item: ids, text, names, arguments, input | FAIL. N/A where the protocol has no streamed snapshot (Anthropic, Chat) | Responses (N/A elsewhere) |
| C-03 | G03 | Echo: what the harness sends back next turn equals what we delivered (call id, name, payload, text, reasoning). Calls are found by call id, Responses texts by item id, Messages and Chat texts at the position the reply was appended to (the message before it must be the one we answered) | FAIL when we delivered something other than what the harness recorded. Texts and reasoning are compared where the harness profile's `echo` says `exact`; a `TODO` there makes C-03 BLOCKED for echoed texts | all |
| C-04 | G03 | Exactly one terminal per reply, nothing after it | FAIL, except client disconnects, which are reported under H-03 | all |
| C-05 | G03 | Ids stable across stream, final, raw and echo | FAIL | all (present and unique) |
| C-06 | G04 | Calls against the request's own declarations, serving-shape classes: `type_mismatch`, `namespace_mismatch`, `prefixed_name_of_declared` | FAIL at more than 0 | all |
| C-07 | G04 | Parallel policy: `parallel_tool_calls=false` or `disable_parallel_tool_use` gives at most one call | FAIL at more than 0; still FAIL if the model omitted a close tag (a policy decision is pending, B009-01) | all |
| C-08 | G04 | `tool_choice` honored: none, required/any, named | FAIL at more than 0 | all |
| C-09 | G08 | Reasoning and content channels separated: no think markup in content, no answer text in the reasoning channel | Flagged, then proven against raw | all (flag) |
| C-10 | G19 | Over-length input answered the way the harness profile's error mapping requires | FAIL on the wrong shape (e.g. HTTP 400 where Codex needs an in-stream `response.failed`) | all |
| C-11 | G03 | Coverage: every protocol element in the reply has a rule (WIRE-05) | An unchecked element makes conformance BLOCKED, never PASS | all |
| C-12 | G03 | Independent reading: the official SDK (`anthropic`, `openai`) reads the same bytes, and its text, reasoning, calls, order, stop, ids and usage equal the adapter's (`protocols.md` section 5) | Any disagreement is FAIL. An SDK that is not installed makes it BLOCKED | all |
| C-13 | G03 | Raw view: what was delivered is what the model wrote, its raw record read in its own format (`turn-ir.md` section 4). RAW-01 calls, RAW-02 arguments, RAW-03 content, RAW-04 reasoning, RAW-05 a stopped reply that is not a prefix of what was written. Accepted: the first call under `parallel_tool_calls=false`, and repairs the model profile declares (`native.repairs`) | FAIL on any other difference. NOT_APPLICABLE without raw output; replies without a raw record are counted as not covered | all (needs `--raw`) |
| R-01 | G19 | First-event latency against the harness's first-byte cutoff | Rate over replies. FAIL if any reply exceeds the cutoff without a keepalive | all (needs timing) |
| R-02 | G19 | Longest silent gap between events against the idle watchdog. Needs per-frame timestamps | Same as R-01 | all (needs per-chunk timing) |
| R-03 | G19 | Error-mapping controls: overload, unsupported field, invalid tool choice, mid-stream failure | Per the acceptance cases | planned |
| R-04 | G19 | A degenerate stop fails the reply without output | FAIL if output reached the client | all |
| R-05 | G19 | Process-level identity of the deployment | Missing gives BLOCKED | all (build_id presence) |
| M-01 | G04 | Model call classes: `undeclared`, `nested_tool_called_directly`, `payload_invalid` | Rate, after a raw check | all |
| M-02 | G12 | Degenerate runs (model profile `degenerate`) | Flag; rate | all (profile) |
| M-03 | G12 | Markup leaks (model profile `markers`), quoted vs bare | Flag; bare hits proven against raw | all (profile) |
| M-04 | G08 | Reasoning round-trip: the harness echoes our carrier (encrypted content, signature) and the next render uses it per the model profile | Flag. The policy is per cell | planned |
| M-05 | G12 | Repetitive reasoning: zlib ratio below 0.10 over at least 2,000 chars | Flag | all |
| M-06 | G12 | Empty turn: no text, no call, no error. EOS-only is not an action | Rate | all |
| M-07 | G12 | The model broke its own format in the raw generation: an unclosed call or channel, a malformed name, a value not of its declared type, a second think close (`native.py` anomalies) | Flag; the reply's differences are the model's, not serving's | all (needs `--raw`) |
| H-01 | G05 | Outcome of each call from the next request (harness profile `outcomes`) | Rate of `aborted` and `arguments_rejected` against the threshold. Each case traced to raw | all (profile outcomes) |
| H-02 | G05 | Identical resends | Hard zero for resends caused by our error mapping | all |
| H-03 | G05 | Disconnects, each attributed to the client, a timeout or the server | Unattributed gives BLOCKED | all (by status) |
| H-04 | G07 | Over-length reaches the harness's compaction path | FAIL if the harness resends instead | all |
| H-05 | G07 | Compaction requests answered with substantive summaries | Sampled semantic review | planned |
| V-01 | G20 | The validator tree hash has a passing control receipt (`acceptance-cases.md`) | Without one, no verdict counts | all (controls.py receipt) |
| V-02 | G20 | The report binds the input manifest, profile hashes and validator hash | Missing gives BLOCKED | partial (profiles, tree hash) |
| V-03 | G20 | Invalidations consumed before a saved PASS is trusted | — | planned |

Kit gates that do not apply to a cell:
- G09–G11 apply only to `task-outcome` and performance claims;
- G13–G15 are about training;
- G16 applies only to replay-corpus de-duplication and held-out L4 tasks;
- G18 applies only when a `comparison` claim is requested.

## 5. Tiers

| | Fast | Full |
|---|---|---|
| Who runs it | Adapter (pre-check) and the loop, every iteration | Validator, at stage exits |
| Input | The iteration's evidence directory, live | Frozen copy with `MANIFEST.sha256` |
| Runs | `scripts/validate/*` with the cell's profiles | Its own re-derivation of key numbers from raw bytes, plus the scripts, plus controls |
| Time | Minutes | Hours |
| Output | `verdict.json`, exit code | Kit-style report: `run-manifest.json`, `validation-report.json`, `findings.jsonl`, `SUMMARY.md`, and a `verdict.json` |
| Counts for stage exit | No | Yes |

The campaign validator wrote 17 fresh scripts per batch, and its reports reached 2 GB. That
belongs in the full tier, and only at stage exits.

## 6. Outputs

**`verdict.json`** has one file per cell and population:

```json
{
  "schema": "trtllm-serve-model-harness/verdict@1",
  "cell": {"model": "deepseek-v41", "harness": "claude-code", "protocol": "anthropic-messages"},
  "tier": "fast",
  "population": {"kind": "replay", "path": "...", "manifest_sha256": null, "cutoff": null},
  "claims": {"conformance": "PASS", "model-behavior": "BLOCKED", "harness-outcome": "NOT_APPLICABLE"},
  "checks": [
    {"id": "C-07", "gate": "G04", "claim": "conformance", "status": "PASS",
     "count": 0, "unit": "reply", "denominator": 1834, "interval": null,
     "evidence": ["calls.json#ptc_violation"], "not_applicable_reason": null}
  ],
  "validator": {"tree_sha256": "...", "sdk_versions": {"anthropic": "2.1.0", "openai": "2.48.0"},
                "controls_receipt": "controls.json", "controls_passed": true},
  "profiles": {"model_sha256": "...", "harness_sha256": "..."},
  "thresholds_from": "matrix.yaml@<sha256>"
}
```

**Exit codes** (the aggregator, Step 2):

| Code | Meaning |
|---|---|
| 0 | Every requested claim is PASS or NOT_APPLICABLE |
| 1 | Any requested claim is FAIL |
| 2 | Otherwise, any requested claim is BLOCKED |
| 3 | Usage or internal error; no verdict |

**Gate records** (full tier) carry the kit's fields: `gate_id`, `catalogue_version`,
`evaluated_population`, `required_for_claim`, `applicability_reason`, `status`,
`check_revision`, `evidence_refs`, `failure_count`/`unit`/`denominator`, `unresolved_evidence`
and `limitations`. A missing count is unknown, not zero.

## 7. Evidence discipline

- **Freeze before you look.**
  - Use closed UTC hours, plus the preceding hour as join-only context.
  - Copy the writer sidecars, then the shards, then hash the copies.
  - Never hash live files and copy later.
- **Late partners.** A record whose partner landed after the window gets a line-level look-up
  in later closed hours: file, line, line SHA-256, file size and file SHA-256. Mark it as
  receipt-only.
- **Membership filters fail closed.** An unknown, missing or conflicting tag excludes the
  record.
- **No content in reports.**
  - Locators are `(file, line, trace_id[, item_id])`.
  - Texts are compared by length, first differing offset and SHA-256.
  - Excerpts stay off by default.
- **Regression evidence.** Only an executable failure on the old code followed by a pass on the
  new code counts. Name tests by node id, and run each `-k` selection separately.
- **Test the seam, not the halves.** The first deployed stable-call-id fix broke 286 of 287
  streams at the join between stream assembly and the snapshot rebuild.

## 8. Implementation

**Entry point: `scripts/validate/verdict.py`.** It handles all three protocols and runs the
three passes in `pipeline.py`:
1. **requests:** one small summary per request, plus the first appearance per thread of every
   echoed item and tool result;
2. **replies:** each reply joined to its request and parsed by its adapter (`proto_responses.py`,
   `proto_anthropic.py`, `proto_chat.py`, built on `turnlib.py`), then read again by the
   protocol's official SDK (`sdk_diff.py`, C-12), and held against its raw record when
   `--raw` is given (`native.py`, C-13);
3. **threads:** echo, outcomes, resends and over-length handling.

Constants (markers, degenerate ids, identity keys, outcome strings, timeouts, error mapping)
come from the profiles (`profiles.py`). Memory grows with requests and calls, not with history
length.

```bash
python scripts/validate/verdict.py --harness profiles/harness/<h>.yaml --model profiles/model/<m>.yaml \
    --matrix <run>/matrix.yaml --population replay|smoke|window [--build-id ID] \
    [--tier full --controls-receipt controls.json] [--requests <boundary hour dirs>] \
    --out-dir <out> <trace or replay dirs>
```

**Qualification: `scripts/validate/controls.py`.**
- It starts `fake_responses_server.py` and `fake_agent_server.py` and replays a synthetic corpus
  once per fault mode.
- Each mode must trip exactly its expected set.
- It also runs the cross cases X-01 to X-25: echo, outcomes, the raw view and text echo
  (some ids are not implemented yet, see `acceptance-cases.md` section 4).
- It writes `controls.json`, bound to the validator tree SHA-256 and to the installed SDK
  versions.
- 2026-10-22: 76/76 cases passed. The tests in `tests/test_validator.py` wrap it
  (`MH_RUN_CONTROLS=1` for the full matrix).

**Environment.** The validator runs from its own virtualenv built from
`scripts/validate/requirements.txt`, which pins `anthropic` and `openai` (C-12) besides
`zstandard`, `jsonschema`, `lark`, `PyYAML` and `aiohttp`. TensorRT-LLM's build environment
also installs `openai`, at whatever version its own requirements say, so it is not the
validator's environment.

**Real-harness check, 2026-10-22.** Claude Code 2.1.289 against `fake_agent_server.py` through
`record_requests.py` (`run_claude_sessions.sh`):
- the Anthropic adapter parsed every request and reply;
- echo matched;
- the three outcome strings were classified as measured (`contracts/claude-code.md` section 9).

**Reference implementation, kept for Step 3.** `check_stream.py`, `check_tool_calls.py`,
`harness_signals.py` and `run_controls.sh` are the Responses-only checkers that the campaign's
production windows were judged with. They stay as an independent second implementation for
calibration, and are retired once Step 3 has reproduced their results.

**Not implemented yet:**
- I-02 (sidecar reconciliation), I-03 (manifest verification), I-07 (capture-boundary
  declaration), I-10 (strata coverage);
- R-03 (error-mapping controls beyond over-length), M-04 (reasoning round-trip policy), H-05
  (compaction substance), V-03 (invalidation log);
- reasoning echo for Responses (Codex sends reasoning items back with `encrypted_content`, not
  the text; that is M-04);
- the raw view on main: the raw-output hook is on the campaign branches only, so a main build
  gives C-13 NOT_APPLICABLE;
- per-frame timing in the trtllm-serve request trace. Without it, R-02 is BLOCKED on production
  windows; replay and recorder captures carry chunk times.

**Step 3, calibration on campaign batch-010 (2026-10-22).** The frozen window is two closed
hours, 2026-10-16 T16 and T17, on jhb and lhr, with the T15 request shards as join context. The
reference checkers and `verdict.py` both ran on it, the latter as two cells: codex-kf x kimi-k3
and codex-kf x glm-5.

| Measure | Reference checkers | `verdict.py` (K3 + GLM) |
|---|---|---|
| Responses streams | 161,639 | 161,470 judged + 169 cut by the client |
| Client disconnects | 169 streams without a terminal event | 169 cut streams + 9 disconnects before any body (H-03) |
| Calls | 160,425 | 160,425 |
| Valid / nested / payload_invalid / undeclared | 160,254 / 121 / 29 / 21 | 160,254 / 121 / 29 / 21 |
| Serving shape (C-06), `parallel_tool_calls` (C-07), stream vs snapshot (C-02) | 0, 0, 0 | 0, 0, 0 |
| Bare markup in calls and messages (M-03) | 1,640 + 229 | 1,869 replies |
| Repetitive reasoning (M-05) | 2 | 2 |
| Echo of delivered calls (C-03) | — | 159,882 equal, 55 missing (all unanswered next turn, I-08) |
| New screens | — | C-09 think markup in content 125 (GLM); M-06 empty turns 46 (GLM) |

Three things surfaced while calibrating, each fixed:
- **Foreign traffic.** Gateway probes are Chat Completions requests with `max_tokens: 1`, 231
  of them. They are not this cell's protocol, so they are now counted and excluded. They were
  also the reference's 9 "resent bodies": among Codex traffic there are 0 identical resends.
- **The K3 serve extension** gives every streaming chat request default `stream_options`, so an
  unrequested usage chunk ends every Chat stream (rule CHT-08). This is recorded in
  `profiles/model/kimi-k3.yaml` for the pi x K3 cell.
- **Client disconnects.** A reply whose client disconnected is judged unless it actually lacks
  its terminal event. A disconnect before any byte is counted under H-03.

**Coverage of real traffic (C-11).** Every one of batch-010's 161,807 replies was parsed for
elements no rule covers:
- the 161,708 Responses replies had none;
- the 99 Chat probes all carried trtllm-serve's `avg_decoded_tokens_per_iter`.

That field was ignored silently before C-11 existed. Rule CHT-13 now covers it, and a rescan
found no unchecked element.

**Seeded faults on real traffic.** `mutate_traces.py` copied 194 K3 and 197 GLM recorded
replies and gave each copy one fault: custom-as-function, prefixed name, second call under
`parallel_tool_calls=false`, snapshot differs, terminal lost.
- Every copy was caught by its own check (C-06, C-06, C-07, C-02 and C-04 respectively).
- The unmutated sample tripped nothing.

**Still open in Step 3:** code-level seeded bugs. These need the campaign fixes reverted on a
serving build with a model, and are listed in `acceptance-cases.md` section 5.

**The SDK reading (C-12), 2026-10-22.** `sdk_diff.py` read every reply of batch-010 again with
`openai` 2.48.0 (`anthropic` 2.1.0 has no Anthropic traffic there):
- **Real traffic.** 161,482 Responses replies and 145 Chat replies, minus 180 cut by the client.
  There were no disagreements once the Chat reading took the accumulated snapshot, because the
  structured-output helper raises on `length` (`protocols.md` section 5). The whole window takes
  about 3 minutes with 6 processes.
- **The adapters' gaps.** Five fault modes that no rule caught were read only by C-12:
  `no_event_lines`, `created_without_output`, `wrong_output_index`, `tool_index_gap` and
  `repeated_call_id`. They became rules ANT-15, RSP-01 (`created_shape`), RSP-15 and CHT-04. On
  batch-010 the new rules fire on nothing: every `response.created` carries `output: []`, and every
  `output_index` and `content_index` is consistent.
- **A fixture the adapters had accepted.** The X-04 fixture opened with a `response.created`
  without `output`. The adapters passed it, and the SDK could not read it. It is fixed.

**Text echo (C-03), 2026-10-22.** Texts are now compared, not only counted:
- **Codex.** batch-010 ran with `echo.text: exact`. 36,567 message texts came back by item id,
  23,481 for K3 and 13,086 for GLM. None differed and 22 were missing. The profile now says
  `exact`.
- **Claude Code.** A wiring run against `fake_agent_server.py` (`ok_thinking`) echoed the signed
  thinking block and the text verbatim, at the appended position.
- **Continuity.** Claude Code had re-serialized the message before it. A first version keyed on
  raw message bytes and missed the echo, so the digest now ignores `cache_control` and the
  string-versus-parts form.

**The raw view (C-13), 2026-10-22.** Both cells ran again with `--raw` over batch-010's
`raw_output` trees:

| Measure | codex-kf x kimi-k3 | codex-kf x glm-5 |
|---|---|---|
| Replies joined to their raw record | 35,747 (28,782 from attempts without raw capture) | 96,952 (1 not) |
| C-13 violations | 0 | 0 |
| M-07, the model broke its format | 10 (9 close mismatches, 2 call indices, 1 value not of its type, 1 stray text) | 1,519 (1,419 second `</think>`, 58 malformed names, 57 unclosed calls, 19 unclosed think, ...) |
| Repairs (`native.repairs`) | — | 2 qualified names mapped to the declared tool |
| Custom tools with extra written arguments, dropped by design | 6 | 28 |

What the raw view settled:
- **GLM's second `</think>`.** In 1,283 of 1,419 replies the parser removed only the markers. In the
  other 136, one `</think>` reached the content: these are the C-09 flags, now proven to start in
  the model's own text.
- **Malformed names.** GLM wrote `<tool_call>final]</arg_value>...` and similar; the parser
  delivered such a call as text.
- **Engine ids repeat across server processes.** 4,780 (K3) and 35,748 (GLM) engine ids name two
  raw records, so they join nothing. `disagg_request_id` named exactly one record every time.
- **Readers.** Every encoder render in `tests/fixtures/native_goldens.json` reads back exactly (5
  cases x 3 models), and the DSML reader agrees with DeepSeek's own parser on all 5.
