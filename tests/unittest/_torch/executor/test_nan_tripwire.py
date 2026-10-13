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


@pytest.fixture
def tripwire_mag(monkeypatch):
    monkeypatch.setenv(tw.ENV_TRIPWIRE, "all")
    monkeypatch.setenv(tw.ENV_MAG, "1")
    tw.NanTripwire.reset_for_tests()
    yield tw.NanTripwire.get()
    tw.NanTripwire.reset_for_tests()


@needs_gpu
def test_magnitude_profile_shows_what_preceded_the_first_nonfinite_mark(tripwire_mag):
    wire = tripwire_mag
    assert wire.with_mag
    x = torch.ones(4, HIDDEN, dtype=torch.bfloat16, device="cuda")
    wire.begin_step([5], [4], False)
    wire.mark(x, tw.stage_id(38, 4))
    wire.end_step()
    wire.begin_step([5], [4], False)
    wire.mark(x, tw.stage_id(38, 4))
    x[1, 10] = 3072.0
    wire.mark(x, tw.stage_id(39, 1))
    x[1, 11] = float("nan")
    wire.mark(x, tw.stage_id(39, 3))
    wire.end_step()
    report = wire.report(5)
    clean, bad = report["steps"]
    assert [r["first"] for r in clean["magnitudes"]] == ["-"] * 4
    (row,) = [r for r in bad["magnitudes"] if r["row"] == 1]
    assert row["first"] == "L39.moe" and row["peak"] == 3072.0 and row["peak_at"] == "L39.attn"
    assert row["last_before_first"] == [["L38.out", 1.0], ["L39.attn", 3072.0]]
    assert row["all"][-1] == [tw.mag_index(tw.stage_id(39, 3)), 3072.0]
    line = tw.summarize_report(report)
    assert "mag r1 before L39.moe: L38.out 1 L39.attn 3.07e+03 (peak 3.07e+03 at L39.attn)" in line


@needs_gpu
def test_magnitude_marks_run_inside_a_cuda_graph(tripwire_mag):
    x = torch.full((4, HIDDEN), 2.0, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        tripwire_mag.mark(x, tw.stage_id(7, 1))
        with torch.cuda.graph(graph, stream=stream):
            tripwire_mag.mark(x, tw.stage_id(7, 1))
    torch.cuda.synchronize()
    tripwire_mag.cur.fill_(tw.NONE)
    x[0, 0] = float("inf")
    tripwire_mag.begin_step([3], [4], False)
    graph.replay()
    tripwire_mag.end_step()
    (step,) = tripwire_mag.report(3)["steps"]
    assert step["request_first"] == ["L7.attn", "-", "-", "-"]
    assert step["magnitudes"][0]["all"] == [[tw.mag_index(tw.stage_id(7, 1)), 2.0]]


def test_rank_stage_names_and_execution_order():
    part = tw.stage_id(15, tw.kind_of("mla_part"))
    assert tw.stage_name(part) == "L15.mla_part"
    assert tw.mag_index(part) == -1  # per-rank marks carry no magnitude
    order = [
        "L14.out",
        "L15.attn_in",
        "L15.mla_q",
        "L15.mla_part",
        "L15.mla_recv",
        "L15.mla_comb",
        "L15.mla_out",
        "L15.attn_part",
        "L15.attn",
        "L15.moe_in",
        "L15.moe_part",
        "L15.moe_shared",
        "L15.moe",
        "L15.out",
        "L16.attn_in",
    ]
    assert sorted(order, key=tw.mark_order) == order


@needs_gpu
def test_skip_mask_and_row_offset(tripwire):
    x = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device="cuda")
    x[0, 1] = float("nan")
    x[2, 3] = float("nan")
    skip = torch.tensor([True, False, False, False], device="cuda")
    tripwire.begin_step([1, 2], [8, 8], False)
    sid = tw.stage_id(15, tw.kind_of("mla_part"))
    # These four rows are batch rows 8..11 (the second request).
    tripwire.mark(x, sid, skip=skip, row_offset=8)
    tripwire.end_step()
    (step,) = tripwire.report(2)["steps"]
    assert step["request_first"] == ["-", "-", "L15.mla_part", "-", "-", "-", "-", "-"]
    assert step["other_nonfinite_rows"] == 0


@needs_gpu
def test_softmax_maxima_may_be_minus_infinity(tripwire):
    stats = torch.zeros(3, 2 * 64, dtype=torch.float32, device="cuda")
    stats[0, 0] = float("-inf")  # an entry the sender sanitized
    stats[1, 0] = float("inf")
    stats[2, 1] = float("nan")
    recv = tw.stage_id(15, tw.kind_of("mla_recv"))
    tripwire.begin_step([1], [3], False)
    tripwire.mark(stats, recv, allow_neg_inf=True)
    tripwire.end_step()
    assert tripwire.report(1)["steps"][0]["request_first"] == ["-", "L15.mla_recv", "L15.mla_recv"]
    tripwire.begin_step([1], [3], False)
    tripwire.mark(stats, recv)
    tripwire.end_step()
    assert tripwire.report(1)["steps"][-1]["request_first"] == ["L15.mla_recv"] * 3


@needs_gpu
def test_contiguous_tensors_are_read_per_leading_row(tripwire):
    # [tokens, heads, rank] as the MLA decode output and query are laid out.
    x = torch.zeros(3, 4, 64, dtype=torch.bfloat16, device="cuda")
    x[1, 3, 63] = float("inf")
    tripwire.begin_step([1], [3], False)
    tripwire.mark(x, tw.stage_id(3, tw.kind_of("mla_q")))
    tripwire.end_step()
    assert tripwire.report(1)["steps"][0]["request_first"] == ["-", "L3.mla_q", "-"]


@needs_gpu
def test_inputs_a_mark_cannot_read_in_place_are_skipped(tripwire):
    base = torch.full((4 * 64 + 1,), float("nan"), dtype=torch.bfloat16, device="cuda")
    unaligned = base[1:].view(4, 64)
    nan = torch.full((4, 64), float("nan"), dtype=torch.bfloat16, device="cuda")
    short_skip = torch.zeros(2, dtype=torch.bool, device="cuda")
    int_skip = torch.zeros(4, dtype=torch.int32, device="cuda")
    sid = tw.stage_id(0, tw.kind_of("mla_part"))
    tripwire.begin_step([1], [4], False)
    tripwire.mark(unaligned, sid)
    tripwire.mark(nan, sid, skip=short_skip)
    tripwire.mark(nan, sid, skip=int_skip)
    tripwire.mark(nan, sid, row_offset=tw.MAX_ROWS - 2)  # would run past the record
    tripwire.mark(nan.view(2, 2, 64).transpose(0, 1), sid)  # not contiguous
    tripwire.end_step()
    assert tripwire.report(1)["steps"][0]["request_first"] == ["-"] * 4


@needs_gpu
def test_magnitudes_stop_before_a_first_rank_mark(tripwire_mag):
    wire = tripwire_mag
    x = torch.ones(2, HIDDEN, dtype=torch.bfloat16, device="cuda")
    wire.begin_step([5], [2], False)
    wire.mark(x, tw.stage_id(38, 4))
    part = torch.zeros(2, 64 * 512, dtype=torch.bfloat16, device="cuda")
    part[1, 7] = float("nan")
    wire.mark(part, tw.stage_id(39, tw.kind_of("mla_part")))
    x[1, 0] = float("nan")
    wire.mark(x, tw.stage_id(39, 1))
    wire.end_step()
    (step,) = wire.report(5)["steps"]
    assert step["request_first"] == ["-", "L39.mla_part"]
    (row,) = [r for r in step["magnitudes"] if r["row"] == 1]
    assert row["last_before_first"] == [["L38.out", 1.0]]
    assert "mag r1 before L39.mla_part: L38.out 1" in tw.summarize_report(wire.report(5))


@needs_gpu
def test_rank_marks_with_a_skip_mask_run_inside_a_cuda_graph(tripwire):
    part = torch.zeros(4, 64 * 512, dtype=torch.bfloat16, device="cuda")
    stats = torch.zeros(4, 64 * 2, dtype=torch.float32, device="cuda")
    skip = torch.zeros(4, dtype=torch.bool, device="cuda")
    part_id = tw.stage_id(7, tw.kind_of("mla_part"))
    recv_id = tw.stage_id(7, tw.kind_of("mla_recv"))
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    def marks():
        tripwire.mark(part, part_id, skip=skip, row_offset=4)
        tripwire.mark(stats, recv_id, allow_neg_inf=True, row_offset=4)

    with torch.cuda.stream(stream):
        marks()  # warm-up outside capture
        with torch.cuda.graph(graph, stream=stream):
            marks()
    torch.cuda.synchronize()
    tripwire.cur.fill_(tw.NONE)
    part[0, 0] = float("nan")  # skipped below
    part[1, 0] = float("nan")
    stats[2, 0] = float("-inf")  # allowed
    stats[3, 1] = float("nan")
    skip[0] = True
    tripwire.begin_step([1, 2], [4, 4], False)
    graph.replay()
    tripwire.end_step()
    (step,) = tripwire.report(2)["steps"]
    assert step["request_first"] == ["-", "L7.mla_part", "-", "L7.mla_recv"]


@needs_gpu
def test_fp8_query_bytes_are_read_by_bit_pattern(tripwire):
    # [tokens, heads, kv_lora_rank + rope] e4m3 bytes, as MLA's quant_q_buffer.
    q = torch.zeros(3, 4, 576, dtype=torch.uint8, device="cuda")
    q[1, 2, 10] = 0x7F  # NaN
    q[2, 0, 0] = 0xFE  # -448, the largest finite magnitude
    q_id = tw.stage_id(3, tw.kind_of("mla_q"))
    tripwire.begin_step([1], [3], False)
    tripwire.mark(q, q_id, fp8=True)
    tripwire.mark(q, tw.stage_id(2, 0))  # bytes without fp8=True have no known encoding
    tripwire.end_step()
    assert tripwire.report(1)["steps"][0]["request_first"] == ["-", "L3.mla_q", "-"]
    tripwire.begin_step([1], [3], False)
    tripwire.mark(q.view(torch.float8_e4m3fn), q_id, row_offset=0)
    tripwire.end_step()
    assert tripwire.report(1)["steps"][-1]["request_first"] == ["-", "L3.mla_q", "-"]


@needs_gpu
def test_fp8_marks_run_inside_a_cuda_graph(tripwire):
    q = torch.zeros(4, 96 * 576, dtype=torch.uint8, device="cuda")
    q_id = tw.stage_id(11, tw.kind_of("mla_q"))
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        tripwire.mark(q, q_id, fp8=True, row_offset=4)
        with torch.cuda.graph(graph, stream=stream):
            tripwire.mark(q, q_id, fp8=True, row_offset=4)
    torch.cuda.synchronize()
    tripwire.cur.fill_(tw.NONE)
    q[3, 96 * 576 - 1] = 0xFF
    tripwire.begin_step([1, 2], [4, 4], False)
    graph.replay()
    tripwire.end_step()
    (step,) = tripwire.report(2)["steps"]
    assert step["request_first"] == ["-", "-", "-", "L11.mla_q"]
