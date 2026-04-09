/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "IndexerKCacheGather.h"
#include "indexerKCacheCommon.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
using indexer::HEAD_DIM;
using indexer::PER_TOKEN_SIZE;
using indexer::SCALE_SIZE;
using indexer::flatIndexToMemoryOffset;

constexpr int WARP_SIZE = 32;
constexpr int VEC_SIZE = 4;
constexpr int TOKENS_PER_BLOCK = 8;

} // anonymous namespace

/**
 * Optimized gather kernel for FP8 K values and scales from paged indexer K cache.
 *
 * Optimizations:
 *   1. Multi-token blocks: 8 tokens/block (256 threads) for warp-level latency hiding.
 *   2. Compile-time d3=132: enables fast multiplication-based division (~4 cycles)
 *      instead of general-purpose int64 division (20+ cycles).
 *   3. Power-of-2 d1 (block_size): d1 division replaced by bitwise AND/shift.
 *   4. __ldg for read-only cache and slot mapping loads.
 *   5. All threads compute own offset in SIMT lockstep (no shuffle serialization).
 */
__global__ __launch_bounds__(WARP_SIZE* TOKENS_PER_BLOCK) void indexerKCacheGatherUnifiedKernel(
    uint8_t* __restrict__ k_fp8_out, uint8_t* __restrict__ k_scale_out, uint8_t const* __restrict__ k_cache,
    int64_t const* __restrict__ slot_mapping_fp8, int64_t const* __restrict__ slot_mapping_scale, int32_t num_tokens,
    int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_3, int32_t d1_mask, int32_t d1_shift)
{
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int32_t token_idx = blockIdx.x * TOKENS_PER_BLOCK + warp_in_block;

    if (token_idx >= num_tokens)
    {
        return;
    }

    int64_t flat_idx_fp8_base = __ldg(&slot_mapping_fp8[token_idx]);
    int64_t flat_idx_scale_base = __ldg(&slot_mapping_scale[token_idx]);

    if (flat_idx_fp8_base < 0 || flat_idx_scale_base < 0)
    {
        return;
    }

    // Each thread computes its own memory offset (SIMT lockstep, no extra cost)
    int32_t head_dim_idx = lane * VEC_SIZE;
    int64_t flat_idx = flat_idx_fp8_base + head_dim_idx;
    int64_t src_offset
        = flatIndexToMemoryOffset(flat_idx, d1_mask, d1_shift, cache_stride_0, cache_stride_1, cache_stride_3);
    int64_t dst_offset = static_cast<int64_t>(token_idx) * HEAD_DIM + head_dim_idx;

    // Gather FP8 data: 4 bytes per thread, 128 bytes per warp
    *reinterpret_cast<uint32_t*>(&k_fp8_out[dst_offset])
        = __ldg(reinterpret_cast<uint32_t const*>(&k_cache[src_offset]));

    // Lane 0: gather scale (4 bytes)
    if (lane == 0)
    {
        int64_t src_offset_scale = flatIndexToMemoryOffset(
            flat_idx_scale_base, d1_mask, d1_shift, cache_stride_0, cache_stride_1, cache_stride_3);
        int64_t dst_offset_scale = static_cast<int64_t>(token_idx) * SCALE_SIZE;
        *reinterpret_cast<uint32_t*>(&k_scale_out[dst_offset_scale])
            = __ldg(reinterpret_cast<uint32_t const*>(&k_cache[src_offset_scale]));
    }
}

void invokeIndexerKCacheGather(uint8_t* k_fp8_out, uint8_t* k_scale_out, uint8_t const* k_cache,
    int64_t const* slot_mapping_fp8, int64_t const* slot_mapping_scale, int32_t num_tokens, int32_t head_dim,
    int32_t scale_size, int32_t cache_dim_0, int32_t cache_dim_1, int32_t cache_dim_2, int32_t cache_dim_3,
    int64_t cache_stride_0, int64_t cache_stride_1, int64_t cache_stride_2, int64_t cache_stride_3, cudaStream_t stream)
{
    if (num_tokens == 0)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(
        head_dim == HEAD_DIM, "head_dim must equal 128 for DeepSeek-V3 indexer cache (got %d)", head_dim);
    TLLM_CHECK_WITH_INFO(
        scale_size == SCALE_SIZE, "scale_size must equal 4 bytes (1 float32 scale per token, got %d)", scale_size);
    TLLM_CHECK_WITH_INFO(cache_dim_2 == 1, "Optimized gather requires cache_dim_2 == 1 (got %d)", cache_dim_2);
    TLLM_CHECK_WITH_INFO(cache_dim_3 == PER_TOKEN_SIZE,
        "Optimized gather requires cache_dim_3 == %d (head_dim + scale_size, got %d)", PER_TOKEN_SIZE, cache_dim_3);
    TLLM_CHECK_WITH_INFO((cache_dim_1 & (cache_dim_1 - 1)) == 0,
        "Optimized gather requires cache_dim_1 (block_size=%d) to be a power of 2", cache_dim_1);

    // Pre-compute mask and shift for power-of-2 block_size division
    int32_t d1_mask = cache_dim_1 - 1;
    int32_t d1_shift = __builtin_ctz(cache_dim_1);

    constexpr int32_t THREADS_PER_BLOCK = WARP_SIZE * TOKENS_PER_BLOCK; // 256

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((num_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK);

    indexerKCacheGatherUnifiedKernel<<<grid, block, 0, stream>>>(k_fp8_out, k_scale_out, k_cache, slot_mapping_fp8,
        slot_mapping_scale, num_tokens, cache_stride_0, cache_stride_1, cache_stride_3, d1_mask, d1_shift);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

TRTLLM_NAMESPACE_END
