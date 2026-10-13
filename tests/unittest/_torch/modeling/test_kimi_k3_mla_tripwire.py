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
"""The NaN tripwire's K3 MLA marks read only what the generation path wrote.

Serving allocates its MLA scratch with ``torch.empty`` from a pool that holds old bytes, and an
old ``0xFF`` byte reads as NaN in bf16, fp32 and fp8 alike. So a clean generation step must not
mark anything even when every new allocation starts out as ``0xFF``.
"""

import weakref
from types import SimpleNamespace

import pytest
import torch
from _torch.attention.multi_gpu.helix_test_utils import CACHE_TYPE_SELFKONLY, setup_kv_and_metadata
from _torch.modeling.kimi_k3_mla_reference import KimiMLAAttention
from _torch.modeling.test_kimi_k3_checkpoint import load_mla_checkpoint
from _torch.modeling.test_kimi_k3_mla import _checkpoint_weights
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch import nan_tripwire as tw
from tensorrt_llm._torch.attention.backends.interface import KVCacheParams
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiLinearForCausalLM
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _poison_the_allocator() -> None:
    """Leave 0xFF in the blocks the caching allocator hands out next, in both of its pools."""
    torch.cuda.synchronize()
    large = torch.full((512 << 20,), 0xFF, dtype=torch.uint8, device="cuda")
    small = [torch.full((256 << 10,), 0xFF, dtype=torch.uint8, device="cuda") for _ in range(64)]
    torch.cuda.synchronize()
    del large, small


@pytest.fixture
def tripwire(monkeypatch):
    monkeypatch.setenv(tw.ENV_TRIPWIRE, "all")
    tw.NanTripwire.reset_for_tests()
    yield tw.NanTripwire.get()
    tw.NanTripwire.reset_for_tests()


@skip_pre_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "kv_cache_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16_cache", "fp8_cache"]
)
def test_a_clean_generation_step_marks_nothing_on_stale_memory(tripwire, kv_cache_dtype):
    # Another test's poisoned blocks must not end up in this test's KV pool (see below).
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    config = KimiLinearConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=256,
        num_hidden_layers=1,
        attn_res_block_size=1,
        torch_dtype=torch.bfloat16,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1], "num_heads": 32},
        num_attention_heads=32,
        num_key_value_heads=32,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        mla_use_output_gate=True,
        max_position_embeddings=512,
    )
    config._attn_implementation = "eager"
    reference = KimiMLAAttention(config, layer_idx=0).bfloat16().eval()
    weights = _checkpoint_weights(reference, fp8=False)
    fp8_cache = kv_cache_dtype == torch.float8_e4m3fn
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8 if fp8_cache else None),
    )
    with torch.device("cuda"):
        model = KimiLinearForCausalLM(model_config).eval()
    projections = {
        f"{name}.{suffix}": value
        for name, pair in weights.items()
        for suffix, value in pair.items()
    }
    projections.update(
        {name: value for name, value in reference.named_parameters() if value.ndim == 1}
    )
    load_mla_checkpoint(model, projections, "")
    runtime = model.model.layers[0].self_attn
    for module in runtime.modules():
        if callable(getattr(module, "post_load_weights", None)):
            module.post_load_weights()
    assert runtime.mixer._tripwire is tripwire
    assert getattr(runtime.mixer.mqa, "has_fp8_kv_cache", False) == fp8_cache

    batch, context_length = 2, 128
    generator = torch.Generator(device="cuda").manual_seed(1234)
    hidden = torch.randn(
        batch,
        context_length + 1,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    context = hidden[:, :context_length].reshape(-1, config.hidden_size).contiguous()
    scenario = SimpleNamespace(
        ctx_len=context_length,
        batch=batch,
        kv_cache_tokens_per_block=64,
        num_layers=1,
        kv_cache_dtype=kv_cache_dtype,
    )
    manager, metadata = setup_kv_and_metadata(
        scenario,
        mapping,
        cache_type=CACHE_TYPE_SELFKONLY,
        num_kv_heads=1,
        head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
    )
    # Serving scrubs its pool at warmup. Stale NaN bytes past a page's written slots do reach
    # the decode output (0 * NaN), a different, known issue that this test must not trip.
    manager.get_buffers(0).zero_()
    try:
        extra_attrs = {
            "attention_metadata": weakref.ref(metadata),
            "mla_layers": {runtime.mixer.layer_idx_str: weakref.ref(runtime.mixer)},
        }
        with model_extra_attrs(extra_attrs):
            runtime(context, metadata)

        for request_id in range(batch):
            manager.impl.add_token(request_id)
        metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.ones(batch, dtype=torch.int),
            request_ids=list(range(batch)),
            max_num_requests=batch,
            num_contexts=0,
            prompt_lens=[context_length] * batch,
            max_num_tokens=batch,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[context_length] * batch
            ),
            mapping=mapping,
            enable_context_mla_with_cached_kv=True,
        )
        metadata.prepare()
        extra_attrs["attention_metadata"] = weakref.ref(metadata)
        _poison_the_allocator()
        tripwire.cur.fill_(tw.NONE)
        with model_extra_attrs(extra_attrs):
            out = runtime(hidden[:, -1].contiguous(), metadata)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        marked = [tw.stage_name(int(s)) for s in tripwire.cur[: 4 * batch].tolist()]
        assert marked == ["-"] * (4 * batch)
    finally:
        manager.shutdown()
