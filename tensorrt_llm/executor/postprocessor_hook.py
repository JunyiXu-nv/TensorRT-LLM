# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for ``tensorrt_llm.executor.postprocessor_hook``."""

import warnings

from tensorrt_llm.engine_client.postprocessor_hook import (  # noqa: F401
    PostProcessorHook,
    PostProcessorHookChunk,
    PostProcessorHookVerdict,
    apply_post_processor_hook,
    emit,
    load_post_processor_hook,
    suppress,
    terminate,
)

warnings.warn(
    "tensorrt_llm.executor.postprocessor_hook has moved to "
    "tensorrt_llm.engine_client.postprocessor_hook and will be removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "PostProcessorHook",
    "PostProcessorHookChunk",
    "PostProcessorHookVerdict",
    "apply_post_processor_hook",
    "emit",
    "load_post_processor_hook",
    "suppress",
    "terminate",
]
