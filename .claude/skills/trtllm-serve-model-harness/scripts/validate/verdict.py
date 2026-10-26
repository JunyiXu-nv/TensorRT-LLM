#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fail-closed verdicts for one cell over one population (references/validate.md).

Reads request-trace shaped input of any of the three protocols (trace windows,
replay output, recorder captures), runs the turn-record checks, and writes:

    <out-dir>/report.json    counts, denominators, rule breakdowns, locators
    <out-dir>/verdict.json   one verdict per requested claim, per-check records

Exit code: 0 every requested claim PASS or NOT_APPLICABLE; 1 any FAIL;
2 otherwise any BLOCKED; 3 usage or internal error (no verdict).

Thresholds for rate claims come from the run's matrix file (--matrix). An
unset threshold makes its claim BLOCKED, never PASS. Nothing in the outputs
carries record content: locators, counts, tool names and hashes only.

usage: verdict.py --harness H.yaml [--model M.yaml] [--matrix matrix.yaml]
                  [--population replay|smoke|window|controls] [--claims a,b,...]
                  [--build-id ID] [--tier fast|full] [--controls-receipt controls.json]
                  [--requests PATH ...] [--responses PATH ...] [--raw PATH ...] [--workers N]
                  --out-dir DIR PATH [PATH ...]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline  # noqa: E402
import profiles as prof  # noqa: E402
import sdk_diff  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(os.path.dirname(HERE))
VALIDATOR_REFERENCES = ("validate.md", "protocols.md", "turn-ir.md", "acceptance-cases.md")
CLAIMS = ("inventory", "conformance", "runtime", "model-behavior", "harness-outcome")
DEFAULT_CLAIMS = {
    "replay": ("inventory", "conformance", "model-behavior"),
    "smoke": ("inventory", "conformance", "runtime", "model-behavior", "harness-outcome"),
    "window": ("inventory", "conformance", "model-behavior", "harness-outcome"),
    "controls": ("inventory", "conformance", "runtime"),
}
# check id -> (claim, kind, unit, threshold key). kind: hard | rate | flag.
CHECKS = {
    "C-01": ("conformance", "hard", "reply", None),
    "C-02": ("conformance", "hard", "reply", None),
    "C-03": ("conformance", "hard", "call", None),
    "C-04": ("conformance", "hard", "reply", None),
    "C-05": ("conformance", "hard", "reply", None),
    "C-06": ("conformance", "hard", "call", None),
    "C-07": ("conformance", "hard", "reply", None),
    "C-08": ("conformance", "hard", "reply", None),
    "C-09": ("conformance", "flag", "reply", None),
    "C-10": ("conformance", "hard", "reply", None),
    "C-11": ("conformance", "coverage", "reply", None),
    "C-12": ("conformance", "hard", "reply", None),
    "C-13": ("conformance", "hard", "reply", None),
    "C-14": ("conformance", "hard", "reply", None),
    "C-16": ("conformance", "hard", "reply", None),
    "R-01": ("runtime", "hard", "reply", None),
    "R-02": ("runtime", "hard", "reply", None),
    "R-04": ("runtime", "hard", "reply", None),
    "R-06": ("runtime", "hard", "reply", None),
    "M-01:undeclared": ("model-behavior", "rate", "call", "undeclared_call_rate"),
    "M-01:payload_invalid": ("model-behavior", "rate", "call", "payload_invalid_rate"),
    "M-01:server_tool_called": ("model-behavior", "flag", "call", None),
    "M-02": ("model-behavior", "rate", "reply", "degenerate_rate"),
    "M-03": ("model-behavior", "rate", "reply", "markup_leak_rate"),
    "M-05": ("model-behavior", "flag", "reply", None),
    "M-06": ("model-behavior", "rate", "reply", "empty_turn_rate"),
    "M-07": ("model-behavior", "flag", "reply", None),
    "H-01:aborted": ("harness-outcome", "rate", "call", "aborted_rate"),
    "H-01:arguments_rejected": ("harness-outcome", "rate", "call", "arguments_rejected_rate"),
    "H-01:unsupported": ("harness-outcome", "flag", "call", None),
    "H-01:unavailable_in_mode": ("harness-outcome", "flag", "call", None),
    "H-02": ("harness-outcome", "hard", "request", None),
    "H-03": ("harness-outcome", "flag", "reply", None),
    "H-04": ("harness-outcome", "hard", "reply", None),
}
Z_ONE_SIDED_95 = 1.6448536269514722


def wilson(k: int, n: int):
    """One-sided 95% Wilson bounds (lower, upper) for k events in n units."""
    if n <= 0:
        return None
    z = Z_ONE_SIDED_95
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def tree_sha256() -> str:
    """Identity of the validator: every file under scripts/validate plus its references."""
    entries = []
    for name in sorted(os.listdir(HERE)):
        path = os.path.join(HERE, name)
        if os.path.isfile(path) and not name.endswith(".pyc"):
            with open(path, "rb") as fh:
                entries.append(f"scripts/validate/{name}:{hashlib.sha256(fh.read()).hexdigest()}")
    for name in VALIDATOR_REFERENCES:
        path = os.path.join(SKILL, "references", name)
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                entries.append(f"references/{name}:{hashlib.sha256(fh.read()).hexdigest()}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def check_records(report: dict, thresholds: dict, harness: prof.Profile, population: str) -> list:
    counts = report["counts"]
    replies = counts.get("den:replies", 0)
    calls = counts.get("den:calls", 0)
    streams = counts.get("den:streams", 0)
    sessions = report.get("sessions", 0)
    unit_pref = (thresholds or {}).get("unit_for_rates") or "session"
    outcome = report.get("outcome_classes", {})
    answered = sum(
        v for k, v in outcome.items() if k not in ("pending", "unanswered", "unknown_profile")
    )
    records = []
    checks = dict(CHECKS)
    # Any other outcome class of the harness profile (tool_error, error, ...) is reported as a flag.
    for cls in outcome:
        if (
            cls not in ("result", "pending", "unanswered", "unknown_profile")
            and f"H-01:{cls}" not in checks
        ):
            checks[f"H-01:{cls}"] = ("harness-outcome", "flag", "call", None)
    for check, (claim, kind, unit, key) in checks.items():
        count = counts.get(check, 0)
        if check.startswith("H-01:"):
            count = outcome.get(check.split(":", 1)[1], 0)
        denominator = {
            "reply": replies,
            "call": calls,
            "request": counts.get("inventory:requests", 0),
        }[unit]
        if check.startswith("H-01:"):
            denominator = answered
        record = {"id": check, "claim": claim, "kind": kind, "count": count, "unit": unit,
                  "denominator": denominator, "status": None, "reason": None,
                  "examples": report["examples"].get(check, [])}  # fmt: skip
        if kind == "coverage":
            elements = sorted(
                k.split(":", 2)[2] for k in counts if k.startswith(check + ":element:")
            )
            record["status"] = "BLOCKED" if count else "PASS"
            if count:
                record["reason"] = (
                    "protocol elements this validator has no rule for: " + ", ".join(elements)[:300]
                )
        elif kind == "flag":
            record["status"] = "FLAGGED" if count else "PASS"
        elif kind == "hard":
            record["status"] = "FAIL" if count else "PASS"
        else:
            threshold = None
            group = "harness_outcome" if claim == "harness-outcome" else "model_behavior"
            if isinstance(thresholds.get(group), dict):
                threshold = thresholds[group].get(key)
            k, n = count, denominator
            if unit_pref == "session" and not sessions:
                record.update(status="BLOCKED", reason="session-level rate but no session identity",
                              unit="session", threshold=threshold, interval=None)  # fmt: skip
                records.append(record)
                continue
            if unit_pref == "session":
                k, n = report.get("sessions_with", {}).get(check, 0), sessions
                record["unit"] = "session"
                if check.startswith("H-01:"):
                    k = report.get("sessions_with", {}).get(check, 0)
            record.update(count=k, denominator=n)
            bounds = wilson(k, n)
            record["interval"] = [round(b, 6) for b in bounds] if bounds else None
            record["threshold"] = threshold
            if not isinstance(threshold, (int, float)):
                record["status"], record["reason"] = "BLOCKED", "threshold not set in the matrix"
            elif bounds is None:
                record["status"], record["reason"] = "BLOCKED", "no units observed"
            elif bounds[1] <= threshold:
                record["status"] = "PASS"
            elif bounds[0] > threshold:
                record["status"] = "FAIL"
            else:
                record["status"], record["reason"] = "BLOCKED", "insufficient n for the threshold"
        records.append(record)

    by_id = {r["id"]: r for r in records}
    # Evidence the checks need (fail closed).
    if counts.get("I-01:response_without_request", 0):
        for cid in ("C-03", "C-06", "C-07", "C-08"):
            if by_id[cid]["status"] == "PASS":
                by_id[cid].update(status="BLOCKED", reason="replies without their request: pass the "
                                  "boundary hour's request shards with --requests")  # fmt: skip
    no_next = counts.get("C-03:no_next_turn", 0)
    echoed = (counts.get("C-03:echo_equal", 0) + counts.get("C-03:echo_rewritten", 0)
              + counts.get("C-03:text_echo_equal", 0) + by_id["C-03"]["count"])  # fmt: skip
    if echoed == 0:
        reason = ("replay has no next turn" if population == "replay"
                  else "no delivered call was echoed by a next turn")  # fmt: skip
        if by_id["C-03"]["status"] == "PASS":
            by_id["C-03"].update(status="NOT_APPLICABLE" if population in ("replay", "controls")
                                 or calls == 0 else "BLOCKED", reason=reason)  # fmt: skip
    by_id["C-03"]["no_next_turn"] = no_next
    by_id["C-03"]["echo_missing"] = counts.get("C-03:echo_missing", 0)
    # Calls the harness sent back changed only as its profile declares (`echo.tool_input`).
    by_id["C-03"]["echo_rewritten"] = counts.get("C-03:echo_rewritten", 0)
    # Answers to a request the client sent again and was answered again: not what it went on with.
    by_id["C-03"]["superseded_answers"] = counts.get("C-03:superseded_answers", 0)
    # Of those, answers the harness dropped and asked again for with more appended (an empty turn).
    by_id["C-03"]["reasked_answers"] = counts.get("C-03:reasked_answers", 0)
    for key in ("text_echo_equal", "text_echo_missing", "text_echo_unchecked"):
        by_id["C-03"][key] = counts.get("C-03:" + key, 0)
    if counts.get("C-03:text_echo_unchecked", 0) and by_id["C-03"]["status"] == "PASS":
        by_id["C-03"].update(status="BLOCKED", reason="harness profile `echo` is TODO: "
                             f"{counts['C-03:text_echo_unchecked']} echoed texts were not compared")  # fmt: skip
    unread = sorted(k.rsplit(":", 1)[1] for k in counts if k.startswith("C-12:sdk_unavailable:"))
    if unread and by_id["C-12"]["status"] == "PASS":
        by_id["C-12"].update(status="BLOCKED", reason="no SDK reading for " + ", ".join(unread)
                             + ": install scripts/validate/requirements.txt")  # fmt: skip
    elif not counts.get("den:sdk_read", 0) and by_id["C-12"]["status"] == "PASS":
        by_id["C-12"].update(status="NOT_APPLICABLE", reason="no reply with a success status")
    for key in ("usage_vs_trace_prompt", "usage_vs_engine_output", "count_tokens_pairs",
                "next_input_pairs"):  # fmt: skip
        by_id["C-14"][key] = counts.get("den:" + key, 0)
    by_id["C-14"]["count_tokens_requests"] = counts.get("inventory:count_tokens_requests", 0)
    joined, unjoined = counts.get("den:raw_joined", 0), counts.get("C-13:unjoined", 0)
    by_id["C-13"].update(denominator=joined, unjoined=unjoined)
    if by_id["C-13"]["status"] == "PASS":
        if not counts.get("den:raw_shards", 0):
            by_id["C-13"].update(status="NOT_APPLICABLE", reason="no raw output given (--raw): "
                                 "what the model wrote is not known")  # fmt: skip
        elif counts.get("C-13:no_native_reader", 0) or not counts.get("den:raw_records", 0):
            by_id["C-13"].update(status="BLOCKED", reason="raw output given, but the model profile "
                                 "has no native reader or no raw record was read")  # fmt: skip
        elif not joined:
            by_id["C-13"].update(status="BLOCKED", reason="no reply joined its raw record")
        elif unjoined:
            by_id["C-13"]["reason"] = f"{unjoined} replies had no raw record; they are not covered"
    # C-16: rejections the harness profile declares as capability retries are flagged, not failed.
    declared = counts.get("C-16:declared", 0)
    by_id["C-16"]["declared_rejections"] = declared
    by_id["C-16"]["declared_fields"] = {
        k.split(":", 2)[2]: v for k, v in counts.items() if k.startswith("C-16:declared_field:")
    }
    if declared and by_id["C-16"]["status"] == "PASS":
        by_id["C-16"].update(status="FLAGGED", reason=f"{declared} rejections the harness profile "
                             "declares as capability retries: each costs a round trip")  # fmt: skip
    if counts.get("C-10:profile_unknown", 0):
        by_id["C-10"].update(status="BLOCKED", reason="harness profile errors.overlength is TODO")
    elif counts.get("den:overlength", 0) == 0 and by_id["C-10"]["status"] == "PASS":
        by_id["C-10"].update(status="NOT_APPLICABLE", reason="no over-length reply observed")
    if replies == 0:
        for record in records:
            if record["claim"] == "conformance" and record["status"] == "PASS":
                record.update(status="BLOCKED", reason="no replies")
    for rid, cut_key in (("R-01", "timeouts.first_event_s"), ("R-02", "timeouts.idle_s")):
        no_timing = counts.get(rid + ":no_timing", 0)
        cut = prof.known(harness.get_path(cut_key))
        if by_id[rid]["status"] == "PASS":
            if streams == 0:
                by_id[rid].update(status="NOT_APPLICABLE", reason="no streamed replies")
            elif no_timing:
                by_id[rid].update(status="BLOCKED", reason=f"{no_timing} streamed replies without "
                                  "timing (record per-chunk times)")  # fmt: skip
            elif not isinstance(cut, (int, float)):
                by_id[rid].update(status="BLOCKED", reason=f"harness profile {cut_key} is TODO")
    return records


def claim_verdicts(records: list, report: dict, requested: list, harness: prof.Profile,
                   model: prof.Profile, build_id, tier: str, controls_ok: bool) -> dict:  # fmt: skip
    counts = report["counts"]
    verdicts, reasons = {}, {}
    for claim in requested:
        mine = [r for r in records if r["claim"] == claim]
        statuses = [r["status"] for r in mine]
        blocked = []
        if claim == "inventory":
            if not counts.get("inventory:requests") or not counts.get("inventory:responses"):
                blocked.append("no requests or no responses")
            if not harness:
                blocked.append("no harness profile")
            if build_id and (
                counts.get("I-05:other_build_id") or counts.get("I-05:missing_build_id")
            ):
                blocked.append("records missing or not matching --build-id")
            if counts.get("I-01:request_without_response"):
                blocked.append(f"partial: {counts['I-01:request_without_response']} requests "
                               "without a response record (I-01); for a window, pass the next "
                               "hour's response shards with --responses")  # fmt: skip
            verdicts[claim] = "BLOCKED" if blocked else "PASS"
        else:
            if claim == "harness-outcome":
                if harness.get("status") != "measured":
                    blocked.append(
                        f"harness profile status is {harness.get('status')!r}, not measured"
                    )
                if prof.outcome_rules(harness) is None:
                    blocked.append("harness profile outcomes are TODO")
                if counts.get("I-09:turn_without_thread"):
                    blocked.append("turns without thread identity")
            if claim == "runtime" and counts.get("I-05:missing_build_id"):
                blocked.append("records without build_id: deployed identity unproven (R-05)")
            if claim == "model-behavior" and not model:
                blocked.append("no model profile (markers and degenerate ids unknown)")
            if "FAIL" in statuses:
                verdicts[claim] = "FAIL"
            elif blocked or "BLOCKED" in statuses:
                verdicts[claim] = "BLOCKED"
            elif mine and all(s == "NOT_APPLICABLE" for s in statuses):
                verdicts[claim] = "NOT_APPLICABLE"
            else:
                verdicts[claim] = "PASS"
        if tier == "full" and not controls_ok and verdicts[claim] != "FAIL":
            verdicts[claim] = "BLOCKED"
            blocked.append("no passing control receipt for this validator tree (V-01)")
        reasons[claim] = blocked
    return {"claims": verdicts, "claim_blockers": reasons}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--harness", required=True)
    ap.add_argument("--model")
    ap.add_argument("--matrix")
    ap.add_argument("--population", default="replay", choices=sorted(DEFAULT_CLAIMS))
    ap.add_argument("--claims", help="comma-separated; default depends on --population")
    ap.add_argument("--build-id")
    ap.add_argument("--tier", default="fast", choices=("fast", "full"))
    ap.add_argument("--controls-receipt")
    ap.add_argument("--requests", nargs="*", default=[])
    ap.add_argument("--responses", nargs="*", default=[],
                    help="next hours' response shards, read only to answer this population")  # fmt: skip
    ap.add_argument("--raw", nargs="*", default=[],
                    help="raw-*.jsonl shards or directories: what the model wrote (C-13)")  # fmt: skip
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                    help="processes for the request and reply passes (1 = serial)")  # fmt: skip
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    try:
        harness = prof.load_profile(args.harness)
        model = prof.load_profile(args.model)
        thresholds = prof.load_thresholds(args.matrix)
        requested = args.claims.split(",") if args.claims else list(DEFAULT_CLAIMS[args.population])
        unknown = [c for c in requested if c not in CLAIMS]
        if unknown:
            raise ValueError(f"unknown claims {unknown}; choose from {CLAIMS}")
        report = pipeline.run(
            args.paths,
            harness,
            model,
            args.requests,
            args.build_id,
            args.workers,
            args.raw,
            args.responses,
        )
        tree = tree_sha256()
        sdks = sdk_diff.versions()
        controls_ok = False
        if args.controls_receipt:
            with open(args.controls_receipt) as fh:
                receipt = json.load(fh)
            # A receipt qualifies one validator tree read by one set of SDK versions.
            controls_ok = (
                bool(receipt.get("passed"))
                and receipt.get("validator_tree_sha256") == tree
                and receipt.get("sdk_versions") == sdks
            )
        records = check_records(report, thresholds, harness, args.population)
        verdict = claim_verdicts(records, report, requested, harness, model, args.build_id,
                                 args.tier, controls_ok)  # fmt: skip
    except (OSError, ValueError) as error:
        print(f"verdict.py: {type(error).__name__}: {error}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 - an internal error must never look like a verdict
        traceback.print_exc()
        return 3

    os.makedirs(args.out_dir, exist_ok=True)
    # Every check that counted at least one event, plus every wire rule broken: what the
    # control matrix compares with each fault mode's expected set.
    tripped = sorted({r["id"] for r in records if r["count"] or r["status"] == "FLAGGED"}
                     | {rule for rules in report["by_rule"].values() for rule in rules})  # fmt: skip
    out = {
        "schema": "trtllm-serve-model-harness/verdict@1",
        "cell": {
            "model": model.get("id"),
            "harness": harness.get("id"),
            "protocol": harness.get("protocol"),
        },  # fmt: skip
        "tier": args.tier,
        "population": {"kind": args.population, "paths": args.paths},
        "claims": verdict["claims"],
        "claim_blockers": verdict["claim_blockers"],
        "checks": [{k: v for k, v in r.items() if k != "examples"} for r in records],
        "tripped": tripped,
        "validator": {
            "tree_sha256": tree,
            "sdk_versions": sdks,
            "sdk_pinned": sdk_diff.pinned(),
            "controls_receipt": args.controls_receipt,
            "controls_passed": controls_ok,
        },  # fmt: skip
        "profiles": {
            "model": model.path,
            "model_sha256": model.sha256,
            "harness": harness.path,
            "harness_sha256": harness.sha256,
        },  # fmt: skip
        "thresholds_from": args.matrix,
    }
    report_out = {"schema": "trtllm-serve-model-harness/report@1", **report,
                  "checks": records}  # fmt: skip
    with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
        json.dump(report_out, fh, indent=1, sort_keys=True)
    with open(os.path.join(args.out_dir, "verdict.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(json.dumps({"claims": verdict["claims"], "tripped": tripped}))
    statuses = [verdict["claims"][c] for c in requested]
    if "FAIL" in statuses:
        return 1
    if "BLOCKED" in statuses:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
