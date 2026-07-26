#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a per-turn report from a content-free Anthropic audit JSONL file.

This analyzer intentionally uses only ``TRTLLM_ANTHROPIC_AUDIT_LOG``. With
benchmark LCP tracking enabled on the server, it reports adjacent-input reuse
without persisting prompt bodies or token IDs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 6)


def load_audit_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: audit record must be a JSON object")
            records.append(record)
    return records


def _session_key(record: dict[str, Any]) -> str:
    session_id = record.get("client_session_id")
    return str(session_id) if session_id else "unidentified-session"


def _turn_metrics(record: dict[str, Any], session_id: str, turn_index: int) -> dict[str, Any]:
    usage = record.get("usage") or {}
    input_tokens = usage.get("input_tokens")
    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_creation = usage.get("cache_creation_input_tokens") or 0
    output_tokens = usage.get("output_tokens")
    isl_total = None
    isl_computed = None
    if input_tokens is not None:
        isl_total = input_tokens + cache_read + cache_creation
        isl_computed = isl_total - cache_read
    prompt_lcp_tokens = record.get("prompt_lcp_tokens")
    lcp_prompt_tokens = record.get("lcp_prompt_tokens")

    response = record.get("response") or {}
    emitted_calls = response.get("tool_calls_emitted") or []
    returned_results = record.get("tool_results_in_last_message") or []
    return {
        "session_id": session_id,
        "turn_index": turn_index,
        "audit_request_id": record.get("audit_request_id"),
        "anthropic_message_id": record.get("anthropic_message_id"),
        "openai_response_id": record.get("openai_response_id"),
        "engine_request_id": record.get("engine_request_id"),
        "disagg_request_id": record.get("disagg_request_id"),
        "ctx_request_id": record.get("ctx_request_id"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "status": record.get("status"),
        "server_ttft_ms": record.get("server_ttft_ms"),
        "server_total_ms": record.get("duration_ms"),
        "isl_total": isl_total,
        "isl_cached": cache_read if input_tokens is not None else None,
        "isl_computed": isl_computed,
        "osl": output_tokens,
        "actual_cache_hit_ratio": _ratio(cache_read, isl_total),
        "lcp_prompt_tokens": lcp_prompt_tokens,
        "lcp_matches_reported_isl": (
            lcp_prompt_tokens == isl_total
            if lcp_prompt_tokens is not None and isl_total is not None
            else None
        ),
        "previous_prompt_tokens": record.get("previous_prompt_tokens"),
        "prompt_lcp_tokens": prompt_lcp_tokens,
        "previous_prompt_retention_ratio": record.get("previous_prompt_retention_ratio"),
        "current_reuse_opportunity_ratio": record.get("current_reuse_opportunity_ratio"),
        "input_only_cache_realization_ratio": _ratio(cache_read, prompt_lcp_tokens),
        "actual_cached_exceeds_input_lcp": (
            cache_read > prompt_lcp_tokens if prompt_lcp_tokens is not None else None
        ),
        "tool_calls_emitted": len(emitted_calls),
        "tool_call_names": [call.get("name") for call in emitted_calls],
        "tool_results_returned": len(returned_results),
        "tool_result_errors": sum(1 for result in returned_results if result.get("is_error")),
    }


def analyze_records(
    records: Iterable[dict[str, Any]],
    session_id_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        session_id = _session_key(record)
        if session_id_filter is not None and session_id != session_id_filter:
            continue
        grouped[session_id].append(record)

    turns: list[dict[str, Any]] = []
    tool_loops: list[dict[str, Any]] = []
    for session_id, session_records in sorted(grouped.items()):
        session_records.sort(
            key=lambda record: (
                (
                    parsed.timestamp()
                    if (parsed := _parse_timestamp(record.get("started_at")))
                    else float("inf")
                ),
                str(record.get("audit_request_id") or ""),
            )
        )
        pending_calls: dict[str, tuple[int, dict[str, Any], dict[str, Any]]] = {}
        for turn_index, record in enumerate(session_records, start=1):
            turn = _turn_metrics(record, session_id, turn_index)
            turns.append(turn)

            for result in record.get("tool_results_in_last_message") or []:
                tool_use_id = result.get("tool_use_id")
                pending = pending_calls.pop(tool_use_id, None)
                if pending is None:
                    continue
                response_turn_index, response_record, call = pending
                response_finished = _parse_timestamp(response_record.get("finished_at"))
                result_request_started = _parse_timestamp(record.get("started_at"))
                gap_ms = None
                if response_finished is not None and result_request_started is not None:
                    gap_ms = round(
                        (result_request_started - response_finished).total_seconds() * 1000,
                        3,
                    )
                tool_loops.append(
                    {
                        "session_id": session_id,
                        "tool_use_id": tool_use_id,
                        "tool_name": call.get("name"),
                        "response_turn_index": response_turn_index,
                        "result_turn_index": turn_index,
                        "response_finished_at": response_record.get("finished_at"),
                        "result_request_started_at": record.get("started_at"),
                        "client_tool_roundtrip_gap_ms": gap_ms,
                        "tool_result_is_error": bool(result.get("is_error")),
                        "tool_result_content_chars": result.get("content_chars"),
                        "matched": True,
                    }
                )

            response = record.get("response") or {}
            for call in response.get("tool_calls_emitted") or []:
                tool_use_id = call.get("id")
                if tool_use_id:
                    pending_calls[tool_use_id] = (turn_index, record, call)

        for tool_use_id, (response_turn_index, response_record, call) in pending_calls.items():
            tool_loops.append(
                {
                    "session_id": session_id,
                    "tool_use_id": tool_use_id,
                    "tool_name": call.get("name"),
                    "response_turn_index": response_turn_index,
                    "result_turn_index": None,
                    "response_finished_at": response_record.get("finished_at"),
                    "result_request_started_at": None,
                    "client_tool_roundtrip_gap_ms": None,
                    "tool_result_is_error": None,
                    "tool_result_content_chars": None,
                    "matched": False,
                }
            )

    return turns, tool_loops


def _format_number(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def render_markdown(turns: list[dict[str, Any]], tool_loops: list[dict[str, Any]]) -> str:
    lines = [
        "# Anthropic Audit Report",
        "",
        "This report uses the content-free server audit. It reports actual cache ",
        "usage, adjacent-input reuse opportunity, and coarse server-observed ",
        "tool-loop gaps; it does not report Claude CLI-visible timing.",
        "",
        "## Turns",
        "",
        "| Session | Turn | Status | TTFT ms | Total ms | ISL | Cached ISL | OSL | "
        "Actual hit | LCP | Prev retained | Current reusable | Cache realization | "
        "Tools | Tool errors |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for turn in turns:
        tool_names = ", ".join(name for name in turn["tool_call_names"] if name) or "-"
        lines.append(
            "| {session} | {turn} | {status} | {ttft} | {total} | {isl} | "
            "{cached} | {osl} | {hit} | {lcp} | {retained} | {reusable} | "
            "{realization} | {tools} | {errors} |".format(
                session=turn["session_id"],
                turn=turn["turn_index"],
                status=turn["status"] or "-",
                ttft=_format_number(turn["server_ttft_ms"]),
                total=_format_number(turn["server_total_ms"]),
                isl=_format_number(turn["isl_total"], 0),
                cached=_format_number(turn["isl_cached"], 0),
                osl=_format_number(turn["osl"], 0),
                hit=_format_percent(turn["actual_cache_hit_ratio"]),
                lcp=_format_number(turn["prompt_lcp_tokens"], 0),
                retained=_format_percent(turn["previous_prompt_retention_ratio"]),
                reusable=_format_percent(turn["current_reuse_opportunity_ratio"]),
                realization=_format_percent(turn["input_only_cache_realization_ratio"]),
                tools=tool_names.replace("|", "\\|"),
                errors=turn["tool_result_errors"],
            )
        )

    lines.extend(
        [
            "",
            "## Tool Loops",
            "",
            "| Session | Tool-use turn | Result turn | Tool | Result error | Coarse gap ms | Matched |",
            "| --- | ---: | ---: | --- | --- | ---: | --- |",
        ]
    )
    for loop in tool_loops:
        lines.append(
            "| {session} | {response_turn} | {result_turn} | {tool} | "
            "{error} | {gap} | {matched} |".format(
                session=loop["session_id"],
                response_turn=loop["response_turn_index"],
                result_turn=_format_number(loop["result_turn_index"], 0),
                tool=(loop["tool_name"] or "-").replace("|", "\\|"),
                error=_format_number(loop["tool_result_is_error"], 0),
                gap=_format_number(loop["client_tool_roundtrip_gap_ms"]),
                matched=str(loop["matched"]).lower(),
            )
        )
    lines.extend(["", "## Validation Flags", ""])
    flags = []
    for turn in turns:
        turn_label = f"{turn['session_id']} turn {turn['turn_index']}"
        if turn["lcp_matches_reported_isl"] is False:
            flags.append(f"- {turn_label}: LCP prompt length does not match reported ISL.")
        if turn["actual_cached_exceeds_input_lcp"]:
            flags.append(f"- {turn_label}: actual cached ISL exceeds adjacent-input LCP.")
    lines.extend(flags or ["No LCP consistency flags."])
    lines.append("")
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_log", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory (default: <audit-log-stem>-analysis)",
    )
    parser.add_argument(
        "--session-id",
        help="Analyze one client_session_id instead of every session",
    )
    args = parser.parse_args()

    output_dir = args.out or args.audit_log.with_name(f"{args.audit_log.stem}-analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    turns, tool_loops = analyze_records(load_audit_records(args.audit_log), args.session_id)
    _write_jsonl(output_dir / "turns.jsonl", turns)
    _write_jsonl(output_dir / "tool_loops.jsonl", tool_loops)
    (output_dir / "REPORT.md").write_text(render_markdown(turns, tool_loops), encoding="utf-8")
    print(f"wrote {len(turns)} turns and {len(tool_loops)} tool loops to {output_dir}")


if __name__ == "__main__":
    main()
