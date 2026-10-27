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
- **Units.** The matrix's `thresholds.unit_for_rates` sets the unit for every rate check:
  - `session`: sessions with at least one event, out of all sessions;
  - `turn`: each check's own unit, a reply (one model output) or, for call-level checks
    (M-01, H-01), a call. Chosen by junyix on 2026-10-26 for the classes this skill judges: a
    tool-parser, reasoning-parser or protocol defect shows in the output where it happens, so
    every output is an occasion for it;
  - any other value is BLOCKED, never guessed.
- **What a turn-level bound leaves open.** The interval treats its units as independent. Turns
  of one session share their context, tools and conditions, so a defect that only some sessions'
  conditions trigger clusters in those sessions, and the turn-level bound is then optimistic.
  Every rate record therefore also carries `sessions` and `sessions_with_events`, and the
  variety of conditions still comes from the number and spread of sessions.
- **Semantic review.** For claims judged by sampling, such as "compaction summaries are
  substantive" (kit G12), record the rubric, strata, sample IDs and seed. A model-judge score is
  not ground truth.

## 4. Gates mapped to checks

IDs: `I` inventory, `C` conformance, `R` runtime, `M` model behavior, `H` harness outcome,
`V` validator closure. The "Now" column says for which protocols `scripts/validate/verdict.py`
implements the check today ("all" = Responses, Anthropic Messages and Chat Completions).

| ID | Gate | Check | Verdict rule | Now |
|---|---|---|---|---|
| I-01 | G01 | Joins: request ↔ response ↔ raw ↔ next turn. Ids above 2^53 are compared as decimal strings | Unjoined records are counted. A required join that is missing gives BLOCKED_PARTIAL: `verdict.py` blocks inventory and names the partial join (a request without any response record). For a window, pass the next hour's response shards with `--responses`: they answer the window's requests only, and `--requests` boundary requests are never counted as lost. A `recorder_shutdown` record is pending at the capture's end, not missing. Operator probes (header `x-mh-probe`, or the older `x-mh-selftest`) are counted and left out of the cell. An engine id naming two raw records joins nothing | all |
| I-02 | G01 | Counts reconciled per unit (requests, streams, calls, turns, sessions) against the writer sidecars | Mismatch is BLOCKED until explained | planned |
| I-03 | G01 | Freeze manifest present and verified (full tier) | Missing gives BLOCKED | planned |
| I-04 | G01 | Credential scan: digests and counts only, including `NAME=value` secrets in environment dumps (`secret_assignment`: names with TOKEN, SECRET, PASSWORD or API_KEY, also after a JSON-escaped newline) | NEW digests are reported to the human and the contexts held | all (scan_credentials.py) |
| I-05 | G02 | `build_id` on every record, and equal to the declared revision | Missing or mixed gives BLOCKED for the claims that name a revision | all |
| I-06 | G02 | Cell identity resolvable: model profile and template hash, harness profile and version, protocol | Unresolved gives BLOCKED | partial (profiles bound) |
| I-07 | G02 | Capture boundary declared: recorder, request trace, replay or mirror | Missing gives BLOCKED | planned |
| I-08 | G05 | Every delivered call is answered next turn, or pending at the cutoff | Unanswered calls are counted; pending is not a failure | all |
| I-09 | G06 | Sessions, threads and sub-agents grouped by the harness profile's identity keys | Ungroupable gives BLOCKED for session-level rates | all |
| I-10 | G17 | Strata coverage: tool kinds, effort levels, context buckets, sub-agents, images, errors, compaction | An empty stratum BLOCKS any claim that needs it | planned |
| C-01 | G03 | Wire rules for the protocol (`protocols.md`) | Any violation is FAIL | all |
| C-02 | G03 | Stream equals the final snapshot item by item: ids, text, names, arguments, input | FAIL. N/A where the protocol has no streamed snapshot (Anthropic, Chat) | Responses (N/A elsewhere) |
| C-03 | G03 | Echo: what the harness sends back next turn equals what we delivered (call id, name, payload, text, reasoning). Calls are found by call id, Responses texts by item id. Messages and Chat texts of a turn with calls are found in the first later request whose history holds those call ids (parallel sub-agents of one session can share a position and a previous message, and a compacted history moves the turn); texts of a turn without calls at the position the reply was appended to (the message before it must be the one we answered) | FAIL when we delivered something other than what the harness recorded. Texts and reasoning are compared where the harness profile's `echo` says `exact`, a reasoning carrier by its value where the protocol gives one (Anthropic's thinking signature, a Responses reasoning item's `encrypted_content`, found by item id); a `TODO` there makes C-03 BLOCKED for echoed texts and reasoning items. A reasoning item the next request leaves out counts as `reasoning_echo_missing`, as a text does as `text_echo_missing`. Call arguments are compared exactly, or under the rewrites the profile lists in `echo.tool_input` (`schema_defaults`: an argument at its declared default may be added or left out; `cd_cwd_prefix`: a leading `cd <dir> &&` we delivered may be cut). Those count as `echo_rewritten`, and an unknown rewrite name fails the run. A request answered more than once (re-sent after a cut, or retried by a proxy) is judged on its last completed answer; the earlier ones count as `superseded_answers`. So does an answer the harness asked again for: a later request at the same position with blocks appended to its last user message and nothing else changed (Claude Code after an empty turn; also counted as `reasked_answers`) | all |
| C-04 | G03 | Exactly one terminal per reply, nothing after it | FAIL, except client disconnects, which are reported under H-03 | all |
| C-05 | G03 | Ids stable across stream, final, raw and echo | FAIL | all (present and unique) |
| C-06 | G04 | Calls against the request's own declarations, serving-shape classes: `type_mismatch`, `namespace_mismatch`, `prefixed_name_of_declared` | FAIL at more than 0 | all |
| C-07 | G04 | Parallel policy: `parallel_tool_calls=false` or `disable_parallel_tool_use` gives at most one call | FAIL at more than 0; still FAIL if the model omitted a close tag (a policy decision is pending, B009-01). NOT_APPLICABLE when no request asked for at most one call | all |
| C-08 | G04 | `tool_choice` honored: none, required/any, named | FAIL at more than 0. NOT_APPLICABLE when no request set one of them | all |
| C-09 | G08 | Reasoning and content channels separated: no think markup in content, no answer text in the reasoning channel | Flagged, then proven against raw | all (flag) |
| C-10 | G19 | Over-length input answered the way the harness profile's error mapping requires | FAIL on the wrong shape (e.g. HTTP 400 where Codex needs an in-stream `response.failed`) | all |
| C-11 | G03 | Coverage: every protocol element in the reply has a rule (WIRE-05) | An unchecked element makes conformance BLOCKED, never PASS | all |
| C-12 | G03 | Independent reading: the official SDK (`anthropic`, `openai`) reads the same bytes, and its text, reasoning, calls, order, stop, ids and usage equal the adapter's (`protocols.md` section 5) | Any disagreement is FAIL. An SDK that is not installed makes it BLOCKED | all |
| C-14 | G19 | Usage (`protocols.md` section 6): present and typed, totals add up, details within their wholes, output reported when output was delivered (USG-01 to 04); `count_tokens` equals the input the same request billed (USG-05); the reported output equals the engine's count (USG-06) and the input the server's prompt count less the profile's offset (USG-07); the request that sends an echoed call back (found by the call ids, not the next request in time: a side request of the session can come first) bills no less input (USG-08), unless its history is shorter than the turn's (the harness compacted: `next_input_shrank_by_compaction`) or its `context_management` can remove history (clearing thinking with `keep: all` cannot) | FAIL on any violation. Rules with no data are listed in the record's `not_exercised` | all (USG-05 needs count_tokens traffic, USG-06 `--raw`, USG-07 a trace's prompt count, USG-08 a next turn) |
| C-15 | G03 | Protocol differential: one harness and task over several wires, greedy decoding, requests made equal (`smoke.md` section 3a). Per turn: the same request must bill the same input and get the same reply on every protocol path | FAIL on `server_render` or `server_output`; a turn whose requests differ is not judged | `protocol_diff.py` (separate exit code), smoke runs over at least two APIs |
| C-16 | G03 | The server accepts what the harness sends, `count_tokens` included: a 4xx other than over-length (C-10) and capacity (429) is a rejection | FAIL. A rejection whose wording the harness profile declares in `errors.capability_retries` (the harness drops that capability and retries) is FLAGGED with its field instead: each costs a round trip | all |
| C-13 | G03 | Raw view: what was delivered is what the model wrote, its raw record read in its own format (`turn-ir.md` section 4). RAW-01 calls, RAW-02 arguments, RAW-03 content, RAW-04 reasoning, RAW-05 a stopped reply that is not a prefix of what was written. Accepted: the first call under `parallel_tool_calls=false`, and repairs the model profile declares (`native.repairs`). A record is read in the mode its request asked for: a model profile with `native.prompt_opens_without_thinking` (V4.1: `content`) has a Messages request without `thinking` read from the content, not as an unclosed think (`read_without_thinking`) | FAIL on any other difference. NOT_APPLICABLE without raw output; replies without a raw record are counted as not covered | all (needs `--raw`) |
| R-01 | G19 | First-event latency against the harness's first-byte cutoff. A client that gave up before the first byte after waiting at least the cutoff counts too: the cutoff passed. The profile gives the cutoff exactly, or as a measured lower bound (`{at_least: s}`: that long is known to survive, in runs before the one judged; a bound taken from the judged run itself would pass it by construction) | Rate over replies. FAIL if any reply exceeds an exact cutoff without a keepalive. Against a lower bound, a reply that waits at least that long makes R-01 BLOCKED: the cut itself is unknown | all (needs timing) |
| R-02 | G19 | Longest silent gap between events against the idle watchdog, given exactly or as a measured lower bound. Needs per-frame timestamps. Keepalive pings count as events, so a stream that pings more often than the bound stays under it whatever the cut is | Same as R-01 | all (needs per-chunk timing) |
| R-03 | G19 | Error-mapping controls: overload, unsupported field, invalid tool choice, mid-stream failure | Per the acceptance cases | planned |
| R-04 | G19 | A degenerate stop fails the reply without output | FAIL if output reached the client | all |
| R-05 | G19 | Process-level identity of the deployment | Missing gives BLOCKED | all (build_id presence) |
| R-06 | G19 | The server answered every request it was sent: the recorder reached it and read each reply to its end (no record with `status: upstream_error`) | FAIL at more than 0. Such a reply is the recorder's own error, or the server's bytes up to the loss, so no conformance check judges it | all (recorder captures) |
| M-01 | G04 | Model call classes: `undeclared`, `nested_tool_called_directly`, `payload_invalid` | Rate, after a raw check | all |
| M-02 | G12 | Degenerate runs (model profile `degenerate`) | Flag; rate. BLOCKED, never PASS, when the model profile names no degenerate tokens: only K3's `[PAD]` fallback is looked for. With `--raw`, the raw text is read too, and only replies that could show the tokens count: a raw record that kept special tokens (its stop marker is there), or one where a run was found; the others are reported as `raw_unobservable` | all (profile) |
| M-03 | G12 | Markup leaks (model profile `markers`), quoted vs bare | Flag; bare hits proven against raw | all (profile) |
| M-04 | G08 | Reasoning round-trip: the harness echoes our carrier (encrypted content, signature) and the next render uses it per the model profile | Flag. The policy is per cell | planned |
| M-05 | G12 | Repetitive reasoning: zlib ratio below 0.10 over at least 2,000 chars | Flag | all |
| M-06 | G12 | Empty turn: no text, no call, no error. EOS-only is not an action | Rate | all |
| M-07 | G12 | The model broke its own format in the raw generation: an unclosed call or channel, a malformed name, a value not of its declared type, a second think close (`native.py` anomalies) | Flag; the reply's differences are the model's, not serving's | all (needs `--raw`) |
| H-01 | G05 | Outcome of each call from the next request (harness profile `outcomes`) | Rate of `aborted` and `arguments_rejected` against the threshold. Each case traced to raw. A rated outcome the profile does not name is BLOCKED (such a call would land in another class); one declared absent with `{none: reason}` is NOT_APPLICABLE | all (profile outcomes) |
| H-02 | G05 | Identical resends | Hard zero for resends caused by our error mapping | all |
| H-03 | G05 | Disconnects, each attributed to the client, a timeout or the server | Unattributed gives BLOCKED. Client-side cuts are flagged with their status; the server's are R-06 | all (by status; a flag record) |
| H-04 | G07 | Over-length reaches the harness's compaction path | FAIL if the harness resends instead. NOT_APPLICABLE when no over-length reply was followed by a next turn | all |
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
- It also runs the cross cases X-01 to X-89: echo, outcomes, the raw view, text echo, declared
  echo rewrites, usage, the protocol differential, the recorder's own replies, lost requests, rejections, probes, cuts, boundary hours, blind degenerate checks, echo and usage pairing by call ids, the raw view per thinking mode, checks that tested nothing, the unit for rates, and client cuts known only from below
  (some ids are not implemented yet, see `acceptance-cases.md` section 4).
- It writes `controls.json`, bound to the validator tree SHA-256 and to the installed SDK
  versions.
- 2026-10-27: 152/152 cases passed. The tests in `tests/test_validator.py` wrap it
  (`MH_RUN_CONTROLS=1` for the full matrix).
- **Properties.** `tests/test_validator_properties.py` states the re-ask rule (C-03) and the
  declared-wording match (C-16) over generated inputs, and `tests/validator_properties.mutations`
  gates them the `trtllm-verify-logic` way: 2026-10-26, 9 of 9 planted bugs detected. A property
  counts only with its gate.

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
- the raw view on main: the raw-output hook is on the campaign branches only, so a main build
  gives C-13 NOT_APPLICABLE unless the hook is cherry-picked onto it (`smoke.md` section 3c);
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

**The pi protocol differential (C-15), 2026-10-22.** pi 1.4.0 against a main build
(`1.4.0rc2`) serving Qwen3-14B on one H200, every request through `record_requests.py` with
greedy decoding, four read-only tasks over each of pi's three APIs (`smoke.md` section 3a; the
details are in `contracts/pi.md` section 10):

| Turn, pair | Attribution | Cause, found by rendering both requests offline through the served build |
|---|---|---|
| 1, Chat and the others (pi's defaults) | `server_rejected` | The Chat route answers pi's `store: false` with HTTP 400 `extra_forbidden`, in a bare-string error body that names a server source line (CHT-15) |
| 1, Chat and Responses | `equal` in 4 of 4 tasks | Token-identical prompts and identical greedy replies, once KV-cache block reuse is off. With reuse on, 2 of 4 tasks gave `divergent_generation` from token-identical prompts |
| 1, Chat and Messages | `harness_literal` in 4 of 4 | pi writes tool schemas in another key order over Messages; the server keeps it (one token) |
| 2, Chat and Responses | `server_render` in 4 of 4 | The Responses route renders pi's echoed reasoning item and function call as two assistant messages; the Chat route as one (5 tokens) |

Each run was also judged on its own:
- **Chat**, with `compat.supportsStore: false`: conformance PASS.
- **Responses**: RSP-07, RSP-12 and RSP-10 / C-02. The campaign branches carry fixes for all three.
- **Messages**: USG-05 and C-03.
  - `count_tokens` renders user text blocks as an empty message, so it answers 10 to 17 tokens
    short.
  - With no signature on thinking blocks, pi replays the model's reasoning as text.
- **C-12** read every successful reply of all three APIs and found no disagreement.

The differential corrected itself twice. It first attributed the Messages one-token difference to
the server, because it compared tool schemas without their key order; it now compares them
literally. It first counted the divergent generations as `server_output`; it now separates those.

**Usage (C-14), 2026-10-22.** `usage.py` over batch-010's two cells:
- **Wire rules.** USG-01 to USG-04: no violation in 64,561 K3 and 96,969 GLM replies.
- **The engine's count.** Every completed reply with a raw record reported exactly the engine's
  `num_output_tokens` (35,746 K3, 96,952 GLM), so USG-06 is an equality.
- **The server's prompt count.** GLM reported exactly the trace's prompt count. Every K3 reply
  reported 3 less, which is Kimi's accounting of the generation opener (`openai_server.py`,
  `num_prompt_tokens_offset`); the K3 profile declares it.
- **count_tokens.** batch-010 is Codex traffic, so USG-05 is exercised by the controls only
  (`count_tokens_off`). Claude Code 2.1.289 did not call it in the wiring runs; a probe
  (`scripts/count_tokens_corpus.py`, replayed into the trace) paired both turns of a wiring run.
- **The next turn.** After a turn whose calls came back exactly, the next turn billed more input
  in all 63,653 K3 and 95,742 GLM cases, so USG-08 is a rule, not a flag.

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

**An A/B of the pi x main fixes, 2026-10-23 (`references/smoke.md` section 3b).** One GB300 node,
Qwen3-30B-A3B, pi 1.4.0. A is main at 9bb7931d6e; B is the same main plus PR #19883 and the four
fixes (Chat `store`, the validation error body, signed thinking blocks, snapshot status). Both
sides installed the same precompiled CI build (`TRTLLM_PRECOMPILED_LOCATION`), so they differ only
in Python. Each side ran pi over the three wires in two modes, a count_tokens probe, and a replay of
the 10-22 pi corpus with four invalid requests added.

| Population | A (main) | B (main + fixes) |
|---|---|---|
| Chat, pi defaults (`store: false`) | CHT-15 4: every request 400 with a bare string | 0, all served |
| Responses | RSP-07 8, RSP-12 13, RSP-10 8 | 0 |
| Messages | C-03 4 (diff) / 6 (real): unsigned thinking replayed as text | 0 |
| Replay, Chat | CHT-15 6 (store and invalid requests) | 0 |
| Replay, Responses | RSP-07 8, RSP-12 13, RSP-10 8, RSP-16 1 | 0 |
| Responses billed input on turn 2, against Chat for the same conversation | +5 (+9 once) | equal |
| count_tokens (USG-05) | 0 | 0 |

- **count_tokens was already fixed on main** by #19964 (shared render pipeline, 10-21): the 10-22
  USG-05 finding (8 of 8 on 807562c203) does not reproduce on 9bb7931d6e.
- **The model was not deterministic.** With greedy decoding and no block reuse, Qwen3-30B-A3B (MoE)
  still gave other reasoning to the same prompt on turn 1 of every task. Such a turn makes the
  rest of the S4 differential inconclusive (exit 2). Render-level evidence then comes from the S3
  replay, where both paths get the same requests.
- **A validator fix came out of it:** a turn whose requests differed only in replayed reasoning was
  attributed to the server (`server_render`). It is now `after_divergence` or `harness` (controls
  X-36 to X-38).
- **A new defect came out of it too:** every streamed `/v1/messages` request logged a GeneratorExit
  traceback and lost its request metrics, because the Anthropic reframer stopped reading the chat
  stream at `[DONE]`. Neither the main branch nor the campaign branches had a fix. A second A/B
  (17 streamed Messages requests, per-request metrics on) gave 17 tracebacks and 0 metric records
  without the fix, and 0 and 17 with it.
- **Unit tests:** each fix's new tests fail on the code it fixes by assertion and pass on the fix.
  The CPU serve suite on main plus every fix fails exactly the tests main fails, all for a missing
  `LLM_MODELS_ROOT`.
- **Upstream PRs (10-23):** #20864 (Chat `store`), #20865 (validation error body), #20866 (signed
  thinking), #20867 (Messages stream metrics), #20868 (snapshot status, a draft stacked on #19883).

**DeepSeek-V4.1-Flash x Anthropic Messages, 2026-10-23.** One GB300 node, TP4/EP4 with attention DP,
`--tool_parser deepseek_v41 --reasoning_parser deepseek_v41`, `max_num_tokens: 65536` without
chunked prefill, so every prompt served fit in 65,536 tokens. Do not copy that setting: with chunked
prefill on, it loses everything before the last 65,536 tokens of a longer prompt
(`references/smoke.md` section 3c).
- Workload: Claude Code 2.1.289, six coding tasks (Read, Glob, Grep, Edit, Write, Bash, up to 15
  turns); pi 1.4.0, eight tasks in the greedy and default modes, with a count_tokens probe.
- Sides, all from the same precompiled CI build:
  - A: main 9bb7931d6e;
  - B6: + #19883 and #20864 to #20868;
  - B7: B6 + #20873 and #20874.

| C-03 failures (echoed turns) | A | B6 | B7 |
|---|---|---|---|
| Claude Code | 21 | 18 | 0 (14 text echoes and 23 calls equal, 5 declared rewrites) |
| pi, greedy | 4 | 4 | 0 |
| pi, default | 5 (2 of them unsigned thinking replayed as text) | 5 | 0 |

- **Whitespace-only text blocks.** Every remaining failure was a whitespace-only text block that
  the client dropped when it sent the turn back. DeepSeek's DSML layout puts a newline after each
  invoke, which the shared streaming parser emitted as content (#20873). The `\n\n` before a
  tool-call block reached Messages as a text block of its own (#20874).
- **Claude Code's own rewrites.** Before the profile declared `echo.tool_input`, Claude Code's
  rewrites (Edit `replace_all: false`, the `cd <cwd> &&` cut) were judged server failures: 6 on
  A and 4 on B6.
- **Task outcomes.** All six Claude Code tasks were done correctly on every side.

**DeepSeek-V4.1-Flash x Claude Code in a Kernel Factory campaign, 2026-10-23 (`claude-code-kf`
x `deepseek-v41`).** One GB300 node served B7 plus the raw-output hook, the fleets' way:
`max_seq_len: 1048576`, chunked prefill, `max_num_tokens: 8192`, block reuse on. Three agents per
round worked a GPU-kernel optimization problem under a 3 h cap.
Both trials were judged at the full tier on 2026-10-26 by an independent validator session, on
frozen snapshots (`MANIFEST.sha256` verified), with validator 789bcac57e and its control receipt
(112/112). The same session judged them again with validator 56bd2b2b0e (116/116); the verdicts
below are from that run, and where it differs from the first, the difference is named.
- **The first attempt lost every agent.** At `max_seq_len: 262144` and `max_num_tokens: 65536`,
  all three derailed on their first request past 65,536 tokens (`references/smoke.md` section 3c).
  Full tier, without `--raw` because that build had no raw-output hook:
  - `inventory` and `conformance` PASS over 60 replies and 70 calls. C-16 is FLAGGED: 4
    `thinking.display` rejections, declared as a capability Claude Code retries without;
  - `runtime` FAIL on R-06: the server never answered three requests after all four ranks died
    of SIGBUS at 07:25Z, when a probe job reinstalled into its checkout (section 5 of `smoke.md`);
  - one empty turn at 68,459 tokens, which Claude Code dropped and asked again for (X-48).
- **The rerun held.** Six agent runs (round 1 complete, round 2 cut by the cap) sent 534 Messages
  requests and 7 `count_tokens` requests, with prompts up to 167k tokens. The 321 requests past
  65,536 tokens ended in `tool_use` 309 times and `end_turn` 12 times. All three round-1 agents
  finished; the best kernel ran 4.61x faster than the baseline. Full tier, with `--raw`:
  - `inventory` BLOCKED: 6 of 534 Messages requests have no response record (I-01). With
    validator 789bcac57e `verdict.py` read PASS, because it counted no request left without a
    response; the validator agent read it as partial, and now both do;
  - `conformance` BLOCKED: every check passed over 528 replies and 623 calls except C-13, which
    joined none of 515 replies to its raw record, since this server's message ids carry no engine
    id. An earlier fast verdict without `--raw` read C-13 as NOT_APPLICABLE and the claim as
    PASS, which overstated it: the raw output existed. C-16 is FLAGGED: 13 declared
    `thinking.display` rejections;
  - `runtime`, `model-behavior` and `harness-outcome` BLOCKED on records without `build_id`,
    unmeasured timeouts, a partial harness profile and unset thresholds;
  - measured for the thresholds: first byte p50 0.38 s and p99 2.81 s; six streams silent for
    22.7-31.7 s, with no `ping` keepalive in any stream; 13 of 534 requests (2.4%) rejected for
    `thinking.display` and retried.
- **Found on the way:**
  - the agent's `thinking.display` was rejected once per session, and Claude Code retried
    without it (#20881);
  - Anthropic message ids carried no engine id, so no reply could be joined to its raw record
    (#20936, and the validator's `msg_<id>_` join, X-46);
  - the recorder lost the streams a client cut after the server's headers, and wrote nothing for
    a request the client gave up on before them (both fixed in `record_requests.py`);
  - a request answered twice was judged on its first answer (now `superseded_answers`, X-47);
  - an answer Claude Code dropped and asked again for was judged against the next echo (X-48),
    and the recorder's own 502s were judged as the server's replies (R-06, X-50);
  - requests without a response record were not counted (I-01, X-51), and nothing reported the
    `thinking.display` rejections (C-16, X-53 and X-54).
- **The six lost requests, read against the server (2026-10-26).** The raw records and the server
  log, joined by engine id, say it was not prefill: the server's first token came 0.9 s (p50) and
  4.2 s (max) after arrival over 514 replies.
  - Four of the six were served in 0.3-5.7 s and ran to completion; their replies were lost
    between the recorder and the client (the recorder logged no-route and timeout errors toward
    the client).
  - Two were accepted (HTTP 200 committed) but never produced a token while later requests ran.
    They were aborted on disconnect after 222 s and 63 s. Cause unknown.
  - The next run samples `/metrics` every 2 s to tell a request queued in the engine from one that
    never reached it.

**The same cell, 2026-10-26 (trial 3).** The build added #20881 and #20936; the recorder had
`--build-id` and the cut fixes. Eight agents worked the same problem. KF never started round 2,
so there are 8 agent runs, 651 requests and 636 streamed Messages replies; the best kernel ran 4.65x.
Fast verdict, then the full tier by an independent session (validator 9b70d5f394, 123/123):
- `inventory` PASS: every request answered, `build_id` on all 1,302 records.
- `conformance` FAIL on C-13 alone. Every reply joined its raw record (the #20936 ids), and the 225
  that differ from what the model wrote differ only in whitespace:
  - 189 delivered texts end with the blank line the DeepSeek encoders write before the tool-call
    block (the streaming DSML parser delivered it as content);
  - 72 string arguments in 58 calls lost their edge whitespace (the parser stripped them):
    `Write.content` its last newline, `Edit` strings their indentation, `Bash.command` a space.
  - Both are fixed in the parser (#20873's second commit; a parameter-value branch), each gated
    by round-trip properties against the server's own V4.1 renderer and six planted bugs.
- `runtime` BLOCKED (KF's idle cut unknown; longest silence 32.75 s, no keepalive); first byte
  max 14.9 s, no cuts, no aborts, no request stuck in the engine queue. `model-behavior` and
  `harness-outcome` BLOCKED: 8 runs against 2% thresholds.
- **Validator defects the full tier found, fixed the same day (controls X-62 to X-68, 130/130;
  six planted bugs detected):**
  - one C-03 and one USG-08 were false: both took the wrong request as the turn's echo. A
    parallel sub-agent shared the reply's position, and a side request came first. Echoes of turns
    with calls are now found by call ids;
  - USG-08 was excused by any `context_management`, which Claude Code sends on every thinking
    request (`clear_thinking`, `keep: all`). Now only clearing that can remove history excuses a
    shrink, and a compaction (shorter history) is counted, not judged;
  - 8 M-07 flags were replies to requests without `thinking`, which V4.1 renders in chat mode.
    The raw view now reads each record in its request's mode, which also brought them into C-13.
