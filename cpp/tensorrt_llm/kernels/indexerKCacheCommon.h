/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace indexer
{

/// Shared constants for indexer K-cache gather/scatter kernels.
/// Specialized for the DeepSeek-V3.2 layout where d2=1 and
/// d3 = PER_TOKEN_SIZE = HEAD_DIM + SCALE_SIZE = 132.

constexpr int32_t HEAD_DIM = 128;
constexpr int32_t SCALE_SIZE = 4;
constexpr int32_t PER_TOKEN_SIZE = HEAD_DIM + SCALE_SIZE; // 132

/**
 * Unravel a flat element index into a byte offset for the indexer K cache.
 * Specialized for the DeepSeek-V3.2 layout: d2=1, d3=132 (compile-time constants).
 * This enables the compiler to use multiplication-based integer division (4 cycles)
 * instead of general-purpose int64 division (20+ cycles) for the d3 unravel.
 * Block size (d1) must be power of 2 -> bitwise AND/shift instead of integer division.
 *
 * @param flat_idx   Flat element index into the 4-D cache.
 * @param d1_mask    (block_size - 1) for bitwise AND.
 * @param d1_shift   log2(block_size) for right shift.
 * @param s0         Stride of dimension 0 (in bytes).
 * @param s1         Stride of dimension 1 (in bytes).
 * @param s3         Stride of dimension 3 (in bytes).
 */
__device__ __forceinline__ int64_t flatIndexToMemoryOffset(
    int64_t flat_idx, int32_t d1_mask, int32_t d1_shift, int64_t s0, int64_t s1, int64_t s3)
{
    // d3 = PER_TOKEN_SIZE = 132 (compile-time constant -> fast multiply-based reduction)
    int32_t i3 = flat_idx % PER_TOKEN_SIZE;
    flat_idx /= PER_TOKEN_SIZE;
    // d2 = 1: skip (always 0)
    // d1 is power of 2 -> bitwise AND/shift instead of integer division
    int32_t i1 = static_cast<int32_t>(flat_idx) & d1_mask;
    int32_t i0 = static_cast<int32_t>(flat_idx) >> d1_shift;
    return i0 * s0 + i1 * s1 + i3 * s3;
}

} // namespace indexer
} // namespace kernels

TRTLLM_NAMESPACE_END
