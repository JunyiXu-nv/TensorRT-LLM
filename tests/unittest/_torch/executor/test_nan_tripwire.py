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
"""Tests for the env-gated per-token first non-finite mark (nan_tripwire)."""

import pytest
import torch

from tensorrt_llm._torch import nan_tripwire as tw

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
HIDDEN = 7168


@pytest.fixture
def tripwire(monkeypatch):
    monkeypatch.setenv(tw.ENV_TRIPWIRE, "all")
    tw.NanTripwire.reset_for_tests()
    yield tw.NanTripwire.get()
    tw.NanTripwire.reset_for_tests()


def test_layer_spec_and_stage_names():
    assert tw.parse_layers("all") is None and tw.parse_layers("1") is None
    assert tw.parse_layers("36-38, 92") == {36, 37, 38, 92}
    assert tw.stage_name(tw.stage_id(39, 3)) == "L39.moe"
    assert tw.stage_name(tw.NONE) == "-"


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(tw.ENV_TRIPWIRE, raising=False)
    tw.NanTripwire.reset_for_tests()
    assert tw.NanTripwire.get() is None
    tw.NanTripwire.reset_for_tests()


@needs_gpu
def test_marks_record_each_rows_first_nonfinite_stage(tripwire):
    x = torch.zeros(6, HIDDEN, dtype=torch.bfloat16, device="cuda")
    tripwire.begin_step([10, 11], torch.tensor([4, 2]), False)
    tripwire.mark(x, tw.stage_id(38, 4))
    x[2, 100] = float("nan")
    tripwire.mark(x, tw.stage_id(39, 1))
    x[4, HIDDEN - 1] = float("inf")
    tripwire.mark(x, tw.stage_id(39, 3))
    tripwire.mark(x, tw.stage_id(40, 0))  # rows 2 and 4 keep their first stage
    tripwire.end_step()
    (step,) = tripwire.report(10)["steps"]
    assert step["request_rows"] == [0, 4]
    assert step["request_first"] == ["-", "-", "L39.attn", "-"]
    assert step["other_nonfinite_rows"] == 1 and step["others"] == [(11, 4, "L39.moe")]
    line = tw.summarize_report(tripwire.report(10))
    assert line.startswith("0 clean steps before | step -0: r0:- r1:- r2:L39.attn r3:-")
    assert "+1 other rows (11/4:L39.moe)" in line


@needs_gpu
def test_ring_reports_steps_oldest_first(tripwire):
    x = torch.zeros(3, HIDDEN, dtype=torch.bfloat16, device="cuda")
    for step in range(5):
        tripwire.begin_step([7, 8], [2, 1], False)
        if step == 3:
            x[1, 0] = float("nan")
        tripwire.mark(x, tw.stage_id(43, 1))
        tripwire.end_step()
    report = tripwire.report(7)
    assert [s["back"] for s in report["steps"]] == [4, 3, 2, 1, 0]
    assert report["steps"][3]["request_first"] == ["-", "L43.attn"]
    assert tw.summarize_report(report).startswith("3 clean steps before | step -1:")
    # A request missing from the recent layouts is reported as such.
    assert (
        tw.summarize_report(tripwire.report(99)) == "clean in the last 0 steps holding the request"
    )


@needs_gpu
def test_marks_run_inside_a_cuda_graph(tripwire):
    x = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        tripwire.mark(x, tw.stage_id(5, 2))  # warm-up outside capture
        with torch.cuda.graph(graph, stream=stream):
            tripwire.mark(x, tw.stage_id(5, 2))
    torch.cuda.synchronize()
    tripwire.cur.fill_(tw.NONE)
    x[3, 7] = float("nan")
    tripwire.begin_step([1], [4], False)
    graph.replay()
    tripwire.end_step()
    (step,) = tripwire.report(1)["steps"]
    assert step["request_first"] == ["-", "-", "-", "L5.moe_in"]


@needs_gpu
def test_oversized_or_unsupported_inputs_are_skipped(tripwire):
    big = torch.full((tw.MAX_ROWS + 1, 8), float("nan"), dtype=torch.bfloat16, device="cuda")
    ints = torch.zeros(4, 8, dtype=torch.int32, device="cuda")
    strided = torch.full((4, 16), float("nan"), dtype=torch.bfloat16, device="cuda")[:, ::2]
    tripwire.begin_step([1], [4], False)
    for tensor in (big, ints, strided):
        tripwire.mark(tensor, tw.stage_id(0, 0))
    tripwire.end_step()
    (step,) = tripwire.report(1)["steps"]
    assert step["request_first"] == ["-"] * 4
