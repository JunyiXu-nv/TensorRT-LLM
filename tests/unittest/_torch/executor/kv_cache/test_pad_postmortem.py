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
"""Tests for the env-gated post-mortem of requests that stop on a degenerate token."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache import pad_postmortem as pm

PAD = 163839
TPB = 64
ROW = 576
E4M3_NAN = 0x7F


def _request(tokens, prompt_len, request_id=7, global_prompt_len=None):
    """``tokens`` is the rank's token list; under Helix it starts with the rank's prompt share."""
    return SimpleNamespace(
        py_request_id=request_id,
        py_prompt_len=prompt_len,
        total_input_len_cp=prompt_len if global_prompt_len is None else global_prompt_len,
        is_dummy=False,
        get_num_tokens=lambda beam: len(tokens),
        get_token=lambda beam, pos: tokens[pos],
    )


def _manager(num_pages, pages_of, *, cp_size=1, cp_rank=0, state_slot=None, device="cpu"):
    """KVCacheManagerV2 stand-in: attention layer 0 with an fp8 MLA pool, layer 1 a state layer."""
    pool = torch.zeros(num_pages, 1, TPB, 1, ROW, dtype=torch.float8_e4m3fn, device=device)
    ssm = torch.zeros(4, 2, 3, device=device)
    conv = torch.zeros(4, 5, device=device)
    return SimpleNamespace(
        tokens_per_block=TPB,
        layer_offsets={10: 0, 11: 1},
        _is_local_mamba_layer=lambda local: local == 1,
        get_buffers=lambda layer_id: pool if layer_id == 10 else None,
        get_batch_cache_indices=lambda ids, layer_id: [pages_of[ids[0]] + [-1, -1]],
        _has_cp_helix=cp_size > 1,
        _helix_cp_size=cp_size,
        _helix_cp_rank=cp_rank,
        _request_id_to_state_index={} if state_slot is None else {7: state_slot},
        all_ssm_states=[ssm],
        all_conv_states=[conv],
        pool=pool,
    )


def _poison(pool, page, slots, lo=0, hi=ROW):
    pool.view(torch.uint8)[page, 0, slots, 0, lo:hi] = E4M3_NAN


def test_local_slot_to_global_matches_round_robin_pages():
    cp_size, tpb = 4, 4
    for cp_rank in range(cp_size):
        owned = [p for p in range(10 * cp_size * tpb) if (p // tpb) % cp_size == cp_rank]
        for local, position in enumerate(owned):
            assert pm.local_slot_to_global(local, tpb, cp_size, cp_rank) == position
    assert pm.local_slot_to_global(130, 64, 1, 0) == 130


def test_non_finite_mask_reads_fp8_by_bit_pattern():
    values = torch.zeros(4, dtype=torch.float8_e4m3fn)
    values.view(torch.uint8)[1] = 0x7F
    values.view(torch.uint8)[2] = 0xFF
    values.view(torch.uint8)[3] = 0x7E  # 448, the largest finite value
    assert pm.non_finite_mask(values).tolist() == [False, True, True, False]
    floats = torch.tensor([0.0, float("nan"), float("inf"), 1.0])
    assert pm.non_finite_mask(floats).tolist() == [False, True, True, False]


def test_clean_request_reports_nothing():
    mgr = _manager(4, {7: [2, 3]}, state_slot=1)
    report = pm.scan_request(mgr, _request([1] * 100 + [PAD], prompt_len=70), [PAD])
    assert report["trigger_pos"] == 100 and report["written"] == 100
    (layer,) = report["layers"]
    assert layer["layer"] == 10 and layer["pages"] == 2
    assert layer["valid_nonfinite_slots"] == 0 and layer["tail_nonfinite_slots"] == 0
    assert layer["first_valid_nonfinite_pos"] is None and layer["tail_slots"] == 28
    assert layer["valid_nonfinite_positions"] == [] and layer["valid_saturated_slots"] == 0
    assert report["state"] == [{"state_layer": 0, "ssm_nonfinite": 0, "conv_nonfinite": 0}]
    assert "earliest none" in pm.summarize(report)


def test_written_tail_page0_and_state_are_told_apart():
    mgr = _manager(5, {7: [2, 3]}, state_slot=1)
    _poison(mgr.pool, 3, slice(10, 12))  # written positions 74, 75
    _poison(mgr.pool, 3, slice(50, 51))  # past the 101 written tokens: tail, latent
    _poison(mgr.pool, 3, slice(60, 61), lo=520)  # tail, rope dims only
    _poison(mgr.pool, 0, slice(0, 64))  # another request's page 0
    mgr.all_ssm_states[0][1, 0, 0] = float("nan")
    report = pm.scan_request(mgr, _request([1] * 100 + [PAD], prompt_len=70), [PAD])
    (layer,) = report["layers"]
    assert layer["valid_nonfinite_slots"] == 2
    assert layer["first_valid_nonfinite_pos"] == 74
    assert layer["valid_nonfinite_positions"] == [74, 75]
    assert layer["first_row_nonfinite"] == ROW and layer["first_row_latent_nonfinite"] == 512
    assert layer["tail_nonfinite_slots"] == 2
    assert layer["tail_latent_nonfinite_slots"] == 1
    assert layer["page0_nonfinite_slots"] == 64
    assert report["state"][0]["ssm_nonfinite"] == 1
    line = pm.summarize(report)
    assert "earliest pos 74 (first at layer 10, 576/576 of its row)" in line
    assert "first pos by layer [10:74]" in line
    assert "tail latent NaN: layers 10 (rope only: -)" in line
    assert "page0 NaN: layers 10" in line and "state NaN (slot 1): 0" in line


def test_partial_row_and_saturation_point_at_a_cast():
    mgr = _manager(4, {7: [2, 3]})
    raw = mgr.pool.view(torch.uint8)
    raw[2, 0, 20, 0, 5] = 0x7E  # +448, position 20
    raw[2, 0, 25, 0, 100] = 0xFE  # -448, position 25
    _poison(mgr.pool, 2, slice(30, 31), lo=530, hi=532)  # two rope elements, position 30
    report = pm.scan_request(mgr, _request([1] * 100 + [PAD], prompt_len=70), [PAD])
    (layer,) = report["layers"]
    assert layer["valid_nonfinite_positions"] == [30]
    assert layer["first_row_nonfinite"] == 2 and layer["first_row_latent_nonfinite"] == 0
    assert layer["valid_saturated_slots"] == 2 and layer["first_valid_saturated_pos"] == 20
    line = pm.summarize(report)
    assert "(first at layer 10, 2/576 of its row)" in line
    assert "fp8 at max magnitude: layers 10" in line


def test_scan_in_chunks_matches_one_pass(monkeypatch):
    mgr = _manager(6, {7: [2, 5, 3]})
    _poison(mgr.pool, 5, slice(20, 21))  # position 84
    _poison(mgr.pool, 3, slice(40, 42))  # past the 150 written tokens
    request = _request([1] * 150 + [PAD], prompt_len=70)
    one_pass = pm.scan_request(mgr, request, [PAD])
    monkeypatch.setattr(pm, "SCAN_CHUNK_PAGES", 1)
    assert pm.scan_request(mgr, request, [PAD]) == one_pass
    (layer,) = one_pass["layers"]
    assert layer["first_valid_nonfinite_pos"] == 84 and layer["tail_nonfinite_slots"] == 2
    assert layer["first_tail_nonfinite_local_slot"] == 168


def test_helix_rank_maps_local_slots_to_global_positions():
    # cp 4, rank 2: local page k holds global block 4k + 2. A 500-token prompt gives rank 2
    # blocks 2 and 6 (positions 128-191 and 384-447), so its token list starts with 128
    # prompt tokens; the 101 generated tokens (positions 500-600) all land on other ranks.
    mgr = _manager(6, {7: [4, 5]}, cp_size=4, cp_rank=2)
    tokens = [1] * 128 + [1] * 100 + [PAD]
    _poison(mgr.pool, 5, slice(3, 4))  # local slot 67 -> block 6, slot 3 -> position 387
    request = _request(tokens, prompt_len=128, global_prompt_len=500)
    report = pm.scan_request(mgr, request, [PAD])
    (layer,) = report["layers"]
    assert (report["prompt_len"], report["num_tokens"], report["written"]) == (500, 601, 600)
    assert report["trigger_pos"] == 600 and report["local_written"] == 128
    assert layer["first_valid_nonfinite_pos"] == 6 * 64 + 3
    assert layer["tail_slots"] == 0


def test_trigger_in_the_prompt_is_ignored(monkeypatch):
    monkeypatch.setenv(pm.ENV_ENABLE, "1")
    postmortem = pm.PadPostmortem.from_env()
    mgr = _manager(4, {7: [2, 3]})
    replayed = _request([1] * 68 + [PAD, 1] + [2, 3], prompt_len=70)
    assert postmortem.maybe_scan(mgr, replayed, rank=0) is None
    report = postmortem.maybe_scan(mgr, _request([1] * 70 + [2, PAD], prompt_len=70), rank=0)
    assert report["trigger_pos"] == 71


def test_postmortem_is_off_by_default_and_respects_its_trigger(monkeypatch):
    monkeypatch.delenv(pm.ENV_ENABLE, raising=False)
    assert not pm.PadPostmortem.from_env().enabled

    monkeypatch.setenv(pm.ENV_ENABLE, "1")
    monkeypatch.setenv(pm.ENV_MAX, "1")
    postmortem = pm.PadPostmortem.from_env()
    mgr = _manager(4, {7: [2, 3]})
    assert postmortem.maybe_scan(mgr, _request([1] * 100, prompt_len=70), rank=0) is None
    report = postmortem.maybe_scan(mgr, _request([1] * 100 + [PAD, 5], prompt_len=70), rank=3)
    assert report is not None and report["rank"] == 3 and report["trigger_pos"] == 100
    # The per-process budget is spent.
    assert postmortem.maybe_scan(mgr, _request([1] * 100 + [PAD], prompt_len=70), rank=0) is None


def test_postmortem_never_raises(monkeypatch):
    monkeypatch.setenv(pm.ENV_ENABLE, "1")
    postmortem = pm.PadPostmortem.from_env()
    mgr = _manager(4, {7: [2, 3]})

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    mgr.get_batch_cache_indices = broken
    assert postmortem.maybe_scan(mgr, _request([1] * 100 + [PAD], prompt_len=70), rank=0) is None


def test_postmortem_writes_a_json_report(monkeypatch, tmp_path):
    monkeypatch.setenv(pm.ENV_ENABLE, "1")
    monkeypatch.setenv(pm.ENV_DIR, str(tmp_path))
    postmortem = pm.PadPostmortem.from_env()
    postmortem.maybe_scan(_manager(4, {7: [2, 3]}), _request([1] * 100 + [PAD], 70), rank=5)
    assert (tmp_path / "padnan-req7-rank5.json").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for KVCacheManagerV2")
def test_scan_indexes_pages_like_the_real_manager():
    """Against a real fp8 KVCacheManagerV2: the scan finds a NaN planted through the same
    page indexing the attention kernels use, at the right position, and nothing else."""
    import tensorrt_llm
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
    from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.sampling_params import SamplingParams

    lengths = {0: 70, 1: 133}
    manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=16 * TPB, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=ROW,
        tokens_per_block=TPB,
        max_seq_len=256,
        max_batch_size=2,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
    )
    try:
        stream = torch.cuda.current_stream().cuda_stream
        requests = {}
        for req_id, length in lengths.items():
            req = LlmRequest(
                request_id=req_id,
                max_new_tokens=4,
                input_tokens=[1] * length,
                sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
                is_streaming=False,
            )
            req.add_new_token(PAD, 0)
            kv_cache = manager._create_kv_cache(req_id, None, None)
            assert kv_cache.resume(stream)
            kv_cache.capacity = length + 1
            requests[req_id] = req
        manager.get_buffers(0).view(torch.uint8).zero_()
        pages = [p for p in manager.get_batch_cache_indices([1], 0)[0] if p >= 0]
        pool = manager.get_buffers(0, kv_layout="NHD").view(torch.uint8)
        pool[pages[1], 0, 5, 0, :] = E4M3_NAN  # request 1, position 69
        torch.cuda.synchronize()

        report = pm.scan_request(manager, requests[1], [PAD])
        (layer,) = report["layers"]
        assert layer["pages"] == 3 and layer["first_valid_nonfinite_pos"] == 69
        assert layer["valid_nonfinite_slots"] == 1 and layer["tail_nonfinite_slots"] == 0
        other = pm.scan_request(manager, requests[0], [PAD])
        assert other["layers"][0]["valid_nonfinite_slots"] == 0
    finally:
        manager.shutdown()
