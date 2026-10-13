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
"""The per-rank NaN tripwire marks of the Helix MLA generation path name the rank that made a
non-finite value and the rank that received it."""

import os
import pickle
import sys
import weakref

import _torch.attention.multi_gpu.helix_test_utils as helix_utils
import _torch.attention.multi_gpu.test_mla_helix as mla_helix
import cloudpickle
import pytest
import torch
from _torch.attention.multi_gpu.helix_test_utils import (
    CACHE_TYPE_SELFKONLY,
    activate_all_ranks_for_context,
    copy_weights_for_cp,
    create_helix_gen_metadata,
    parse_comms_medium,
    run_single_rank,
    setup_kv_and_metadata,
    split_inputs_for_rank,
)
from _torch.attention.multi_gpu.test_mla_helix import RopeConfig, Scenario, _generate_random_weights
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch import nan_tripwire as tw
from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import CpType, Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(helix_utils)
cloudpickle.register_pickle_by_value(mla_helix)
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# The made-up fault: rank 1's decode output for this generation row, in head 0. Head 0 belongs
# to the head chunk rank 0 combines, so rank 0 receives it through the all-to-all.
BAD_ROW = 3
BAD_RANK = 1


def _poison_the_allocator():
    """Leave 0xFF (NaN in bf16, fp32 and fp8) in the blocks the caching allocator hands out
    next, in both of its pools, as serving's scratch memory holds old bytes."""
    torch.cuda.synchronize()
    large = torch.full((512 << 20,), 0xFF, dtype=torch.uint8, device="cuda")
    small = [torch.full((256 << 10,), 0xFF, dtype=torch.uint8, device="cuda") for _ in range(64)]
    torch.cuda.synchronize()
    del large, small


def _mla(scenario, pos_embd_params, config=None):
    kwargs = {} if config is None else {"config": config}
    return MLA(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        num_key_value_heads=scenario.num_kv_heads,
        qk_nope_head_dim=scenario.qk_nope_head_dim,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        v_head_dim=scenario.v_head_dim,
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=scenario.kv_lora_rank,
        predicted_tokens_per_seq=scenario.predicted_tokens_per_seq,
        max_position_embeddings=scenario.max_position_embeddings,
        bias=scenario.bias,
        pos_embd_params=pos_embd_params,
        layer_idx=0,
        dtype=scenario.dtype,
        **kwargs,
    ).cuda()


@torch.inference_mode
def _tripwire_worker(rank, world_size, scenario, use_nccl_for_alltoall, fifo_version):
    os.environ[tw.ENV_TRIPWIRE] = "all"
    tw.NanTripwire.reset_for_tests()
    wire = tw.NanTripwire.get()
    assert wire is not None

    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling=None,
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )
    torch.manual_seed(42)
    input_ctx = torch.empty(
        scenario.batch * scenario.ctx_len, scenario.hidden_size, dtype=scenario.dtype, device="cuda"
    ).uniform_(-1, 1)
    input_gen = torch.empty(
        scenario.batch, scenario.hidden_size, dtype=scenario.dtype, device="cuda"
    ).uniform_(-1, 1)
    position_ids_ctx = torch.arange(scenario.ctx_len, dtype=torch.int, device="cuda").repeat(
        scenario.batch
    )
    position_ids_gen = torch.full(
        (scenario.batch,), scenario.ctx_len, dtype=torch.int, device="cuda"
    )
    # Same seed on every rank, so every rank draws the same weights.
    weights_source = _mla(scenario, pos_embd_params)
    _generate_random_weights(weights_source)
    weights = weights_source.state_dict()
    del weights_source

    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        cp_size=world_size,
        cp_config={
            "cp_type": CpType.HELIX,
            "use_nccl_for_alltoall": use_nccl_for_alltoall,
            "fifo_version": fifo_version,
        },
    )
    extra_attrs = dict()
    config = ModelConfig(mapping=mapping)
    config.extra_attrs = extra_attrs
    mla = _mla(scenario, pos_embd_params, config)
    copy_weights_for_cp(weights, "o_proj.weight", 1, rank, world_size)
    copy_weights_for_cp(weights, "v_b_proj", 0, rank, world_size)
    mla.load_state_dict(weights)
    assert mla._tripwire is wire

    kv_cache_manager, attn_metadata = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=scenario.kv_lora_rank + scenario.qk_rope_head_dim,
    )
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)
    ctx_len_per_gpu = scenario.ctx_len // world_size
    input_ctx_rank, position_ids_ctx_rank = split_inputs_for_rank(
        input_ctx, position_ids_ctx, scenario, rank, world_size
    )
    activate_all_ranks_for_context(attn_metadata, position_ids_ctx_rank)
    ctx_output = input_ctx_rank.new_empty(
        [input_ctx_rank.shape[0], mla.num_heads_tp * mla.v_head_dim], dtype=input_ctx_rank.dtype
    )
    mla.forward_impl(position_ids_ctx_rank, input_ctx_rank, attn_metadata, attn_output=[ctx_output])

    for req_id in range(scenario.batch):
        kv_cache_manager.impl.add_token(req_id)
    attn_metadata = create_helix_gen_metadata(
        scenario.batch,
        ctx_len_per_gpu,
        kv_cache_manager,
        [rank != world_size - 1] * scenario.batch,
        position_ids_gen,
        enable_context_mla_with_cached_kv=True,
    )
    extra_attrs["attention_metadata"] = weakref.ref(attn_metadata)

    def generation_step():
        _poison_the_allocator()
        wire.cur.fill_(tw.NONE)
        with model_extra_attrs(extra_attrs):
            mla(position_ids_gen, input_gen, attn_metadata)
        torch.cuda.synchronize()
        return [tw.stage_name(int(s)) for s in wire.cur[: scenario.batch].tolist()]

    clean = generation_step()

    decode = mla.mqa.forward

    def faulty_decode(*args, **kwargs):
        out = decode(*args, **kwargs)
        if rank == BAD_RANK:
            out[BAD_ROW, : mla.kv_lora_rank] = float("nan")
        return out

    mla.mqa.forward = faulty_decode
    faulty = generation_step()
    mla.mqa.forward = decode
    # The receiver's raw capture (FIFO paths only) names the sending rank and the bad span.
    captures = []
    if not use_nccl_for_alltoall and int(wire.recv_count.item()):
        captures = [
            (c["row"], c["src"], c["o_first"], c["o_count"], c["s_count"])
            for c in wire._recv_captures(int(wire.recv_count.item()))
        ]
    kv_cache_manager.shutdown()
    tw.NanTripwire.reset_for_tests()
    return rank, clean, faulty, captures


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
@pytest.mark.parametrize(
    "comms_medium, receiver_mark", [("fifo_v2", "L0.mla_recv"), ("nccl", "L0.mla_comb")]
)
def test_rank_marks_name_the_maker_and_the_receiver(comms_medium, receiver_mark):
    scenario = Scenario(batch=8, ctx_len=1024)
    use_nccl_for_alltoall, fifo_version = parse_comms_medium(comms_medium)
    world_size = 2
    with MPIPoolExecutor(max_workers=world_size) as executor:
        results = list(
            executor.map(
                run_single_rank,
                *zip(
                    *[
                        (
                            _tripwire_worker,
                            world_size,
                            scenario,
                            use_nccl_for_alltoall,
                            fifo_version,
                        )
                    ]
                    * world_size
                ),
            )
        )
    by_rank = {rank: (clean, faulty, captures) for rank, clean, faulty, captures in results}
    for clean, _, _ in by_rank.values():
        assert clean == ["-"] * scenario.batch
    expected = {BAD_RANK: "L0.mla_part", 1 - BAD_RANK: receiver_mark}
    for rank, (_, faulty, _) in by_rank.items():
        want = ["-"] * scenario.batch
        want[BAD_ROW] = expected[rank]
        assert faulty == want, (rank, faulty)
    if comms_medium == "fifo_v2":
        # The receiver holds head 0's chunk: the whole first head (kv_lora_rank values) of the
        # entry it got from the faulty rank is NaN; nothing else, and no stats.
        assert by_rank[1 - BAD_RANK][2] == [(BAD_ROW, BAD_RANK, 0, scenario.kv_lora_rank, 0)]
        assert by_rank[BAD_RANK][2] == []
