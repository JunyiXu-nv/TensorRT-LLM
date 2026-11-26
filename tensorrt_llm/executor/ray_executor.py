# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for ``tensorrt_llm.executor.ray_executor``."""

import warnings

from tensorrt_llm.engine_client.ray_executor import RayExecutor  # noqa: F401

warnings.warn(
    "tensorrt_llm.executor.ray_executor has moved to "
    "tensorrt_llm.engine_client.ray_executor and will be removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

__all__ = ["RayExecutor"]
