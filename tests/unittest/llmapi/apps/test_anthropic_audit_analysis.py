# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the content-free Anthropic audit V0 analyzer."""

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[4] / "examples/serve/anthropic_compatibility/analyze_audit.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_anthropic_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def test_analyze_records_computes_turn_and_tool_loop_metrics():
    records = [
        {
            "audit_request_id": "audit-1",
            "client_session_id": "session-1",
            "started_at": "2026-07-21T10:00:00+00:00",
            "finished_at": "2026-07-21T10:00:02+00:00",
            "duration_ms": 2000.0,
            "server_ttft_ms": 500.0,
            "status": "completed",
            "usage": {
                "input_tokens": 80,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 0,
                "output_tokens": 10,
            },
            "response": {
                "tool_calls_emitted": [
                    {
                        "id": "toolu-1",
                        "name": "Bash",
                        "input_json_bytes": 24,
                    }
                ]
            },
            "tool_results_in_last_message": [],
        },
        {
            "audit_request_id": "audit-2",
            "client_session_id": "session-1",
            "started_at": "2026-07-21T10:00:05+00:00",
            "finished_at": "2026-07-21T10:00:06+00:00",
            "duration_ms": 1000.0,
            "server_ttft_ms": 250.0,
            "lcp_prompt_tokens": 200,
            "previous_prompt_tokens": 100,
            "prompt_lcp_tokens": 80,
            "previous_prompt_retention_ratio": 0.8,
            "current_reuse_opportunity_ratio": 0.4,
            "status": "completed",
            "usage": {
                "input_tokens": 40,
                "cache_read_input_tokens": 160,
                "cache_creation_input_tokens": 0,
                "output_tokens": 5,
            },
            "response": {"tool_calls_emitted": []},
            "tool_results_in_last_message": [
                {
                    "tool_use_id": "toolu-1",
                    "is_error": False,
                    "content_chars": 18,
                }
            ],
        },
    ]

    turns, tool_loops = analyzer.analyze_records(reversed(records))

    assert [turn["turn_index"] for turn in turns] == [1, 2]
    assert turns[0]["isl_total"] == 100
    assert turns[0]["isl_cached"] == 20
    assert turns[0]["isl_computed"] == 80
    assert turns[0]["osl"] == 10
    assert turns[0]["actual_cache_hit_ratio"] == 0.2
    assert turns[1]["actual_cache_hit_ratio"] == 0.8
    assert turns[1]["prompt_lcp_tokens"] == 80
    assert turns[1]["previous_prompt_retention_ratio"] == 0.8
    assert turns[1]["current_reuse_opportunity_ratio"] == 0.4
    assert turns[1]["input_only_cache_realization_ratio"] == 2.0
    assert turns[1]["actual_cached_exceeds_input_lcp"] is True
    assert tool_loops == [
        {
            "session_id": "session-1",
            "tool_use_id": "toolu-1",
            "tool_name": "Bash",
            "response_turn_index": 1,
            "result_turn_index": 2,
            "response_finished_at": "2026-07-21T10:00:02+00:00",
            "result_request_started_at": "2026-07-21T10:00:05+00:00",
            "client_tool_roundtrip_gap_ms": 3000.0,
            "tool_result_is_error": False,
            "tool_result_content_chars": 18,
            "matched": True,
        }
    ]


def test_analyze_records_reports_unmatched_tool_use():
    turns, tool_loops = analyzer.analyze_records(
        [
            {
                "audit_request_id": "audit-1",
                "started_at": "2026-07-21T10:00:00+00:00",
                "finished_at": "2026-07-21T10:00:02+00:00",
                "status": "completed",
                "response": {"tool_calls_emitted": [{"id": "toolu-1", "name": "Read"}]},
            }
        ]
    )

    assert len(turns) == 1
    assert tool_loops[0]["matched"] is False
    assert tool_loops[0]["result_turn_index"] is None
