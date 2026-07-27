# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv

from examples.serve.anthropic_compatibility.analyze_audit import (
    _write_timeline_csv,
    analyze_records,
    build_timeline_rows,
)


def _record(
    *,
    session_id,
    audit_request_id,
    engine_request_id,
    started_at,
    finished_at,
    emitted_calls=None,
    returned_results=None,
):
    return {
        "client_session_id": session_id,
        "client_session_source": "header:x-claude-code-session-id",
        "audit_request_id": audit_request_id,
        "engine_request_id": engine_request_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "completed",
        "duration_ms": 1000.0,
        "server_ttft_ms": 100.0,
        "usage": {
            "input_tokens": 10,
            "cache_read_input_tokens": 90,
            "cache_creation_input_tokens": 0,
            "output_tokens": 20,
        },
        "prompt_lcp_tokens": 90,
        "lcp_prompt_tokens": 100,
        "response": {
            "thinking_chars": 12,
            "text_chars": 3,
            "tool_calls_emitted": emitted_calls or [],
        },
        "tool_results_in_last_message": returned_results or [],
        "history_message_count": 3,
        "history_content_block_counts": {"text": 2},
        "tool_definition_count": 1,
        "message_capture_file": f"requests/{audit_request_id}.json.gz",
    }


def test_timeline_orders_multiple_sessions_and_attaches_tool_gap(tmp_path):
    records = [
        _record(
            session_id="session-a",
            audit_request_id="a-2",
            engine_request_id="3",
            started_at="2026-01-01T00:00:05+00:00",
            finished_at="2026-01-01T00:00:06+00:00",
            returned_results=[
                {
                    "tool_use_id": "tool-1",
                    "content_chars": 42,
                    "is_error": False,
                }
            ],
        ),
        _record(
            session_id="session-b",
            audit_request_id="b-1",
            engine_request_id="2",
            started_at="2026-01-01T00:00:01.500000+00:00",
            finished_at="2026-01-01T00:00:01.800000+00:00",
        ),
        _record(
            session_id="session-a",
            audit_request_id="a-1",
            engine_request_id="1",
            started_at="2026-01-01T00:00:01+00:00",
            finished_at="2026-01-01T00:00:02+00:00",
            emitted_calls=[
                {
                    "id": "tool-1",
                    "name": "Bash",
                    "input_json_bytes": 17,
                }
            ],
        ),
    ]

    turns, tool_loops = analyze_records(records)
    timeline = build_timeline_rows(turns, tool_loops)

    assert [
        (
            row["global_request_index"],
            row["session_id"],
            row["session_turn_index"],
        )
        for row in timeline
    ] == [
        (1, "session-a", 1),
        (2, "session-b", 1),
        (3, "session-a", 2),
    ]

    tool_turn = timeline[0]
    assert tool_turn["tool_call_names"] == ["Bash"]
    assert tool_turn["tool_call_input_json_bytes"] == 17
    assert tool_turn["tool_result_turns"] == [2]
    assert tool_turn["tool_result_content_chars"] == 42
    assert tool_turn["tool_loop_gap_ms"] == 3000.0
    assert tool_turn["tool_loop_matched"] is True
    assert tool_turn["osl_model_tokens"] == 20
    assert tool_turn["server_decode_ms"] == 900.0
    assert tool_turn["output_tps_per_user"] == 19 / 0.9

    output = tmp_path / "timeline.csv"
    _write_timeline_csv(output, timeline)
    with output.open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))

    assert [row["global_request_index"] for row in csv_rows] == ["1", "2", "3"]
    assert csv_rows[0]["tool_call_names"] == '["Bash"]'
    assert csv_rows[0]["tool_loop_gap_ms"] == "3000.0"
    assert float(csv_rows[0]["output_tps_per_user"]) == 19 / 0.9
