#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""raw_vs_parsed.py finds the raw dumps in both layouts serve.sh produces.

Without capture the hook writes ``<attempt>/raw_output/<hour>/raw-<pid>.jsonl``;
with capture on it writes beside the trace, ``<attempt>/request_trace/<hour>/``.
The analysis must locate the attempt and load the records from either.
"""
import importlib.util
import json
import os
from pathlib import Path

ANALYSIS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "raw_vs_parsed.py")
SPEC = importlib.util.spec_from_file_location("raw_vs_parsed_under_test", ANALYSIS)
rvp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rvp)


def _write_raw(directory: Path, request_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw-123.jsonl").write_text(json.dumps({"request_id": request_id, "text": "hi"}) + "\n")


def test_legacy_raw_output_layout(tmp_path):
    attempt = tmp_path / "attempt-001"
    _write_raw(attempt / "raw_output" / "2026-10-13T20", "req-legacy")
    assert rvp.find_attempt(attempt / "raw_output" / "2026-10-13T20") == attempt
    assert set(rvp.load_raw(attempt)) == {"req-legacy"}


def test_capture_layout_beside_the_trace(tmp_path):
    attempt = tmp_path / "attempt-002"
    trace_hour = attempt / "request_trace" / "2026-10-13T20"
    _write_raw(trace_hour, "req-capture")
    (trace_hour / "requests-123.jsonl").write_text("{}\n")   # the trace files it sits beside are not raw dumps
    assert rvp.find_attempt(trace_hour) == attempt
    assert set(rvp.load_raw(attempt)) == {"req-capture"}


def test_both_layouts_are_read_together(tmp_path):
    attempt = tmp_path / "attempt-003"
    _write_raw(attempt / "raw_output" / "2026-10-13T20", "a")
    _write_raw(attempt / "request_trace" / "2026-10-13T21", "b")
    assert set(rvp.load_raw(attempt)) == {"a", "b"}


def test_no_raw_dumps_means_no_attempt(tmp_path):
    (tmp_path / "attempt-004" / "request_trace" / "2026-10-13T20").mkdir(parents=True)
    assert rvp.find_attempt(tmp_path / "attempt-004") is None
