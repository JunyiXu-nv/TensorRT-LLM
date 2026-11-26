# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for ``tensorrt_llm.executor.result``.

Results moved to ``tensorrt_llm.engine_client.result``, log-probability types
to ``tensorrt_llm.executor.logprobs``, and ``ResponseWrapper`` / ``get_metrics_dict``
to ``tensorrt_llm.executor.utils``.
"""

import importlib
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm.engine_client.result import (
        CompletionOutput,
        DetokenizedGenerationResultBase,
        GenerationResult,
        GenerationResultBase,
        IterationResult,
    )
    from tensorrt_llm.executor.logprobs import (
        Logprob,
        LogProbsResult,
        SimpleTokenLogprobs,
        TokenLogprobs,
        compute_prompt_logprobs,
    )
    from tensorrt_llm.executor.utils import ResponseWrapper, get_metrics_dict

_LAZY_NAMES: dict[str, tuple[str, str]] = {
    "CompletionOutput": ("tensorrt_llm.engine_client.result", "CompletionOutput"),
    "DetokenizedGenerationResultBase": (
        "tensorrt_llm.engine_client.result",
        "DetokenizedGenerationResultBase",
    ),
    "GenerationResult": ("tensorrt_llm.engine_client.result", "GenerationResult"),
    "GenerationResultBase": ("tensorrt_llm.engine_client.result", "GenerationResultBase"),
    "IterationResult": ("tensorrt_llm.engine_client.result", "IterationResult"),
    "Logprob": ("tensorrt_llm.executor.logprobs", "Logprob"),
    "TokenLogprobs": ("tensorrt_llm.executor.logprobs", "TokenLogprobs"),
    "SimpleTokenLogprobs": ("tensorrt_llm.executor.logprobs", "SimpleTokenLogprobs"),
    "LogProbsResult": ("tensorrt_llm.executor.logprobs", "LogProbsResult"),
    "reduce_prompt_logprobs": ("tensorrt_llm.executor.logprobs", "reduce_prompt_logprobs"),
    "prompt_logprobs_from_reduced": (
        "tensorrt_llm.executor.logprobs",
        "prompt_logprobs_from_reduced",
    ),
    "compute_prompt_logprobs": ("tensorrt_llm.executor.logprobs", "compute_prompt_logprobs"),
    "ResponseWrapper": ("tensorrt_llm.executor.utils", "ResponseWrapper"),
    "get_metrics_dict": ("tensorrt_llm.executor.utils", "get_metrics_dict"),
    "_build_perf_metrics_dict": ("tensorrt_llm.executor.utils", "_build_perf_metrics_dict"),
}

_WARNED = False


def __getattr__(name: str):
    global _WARNED
    if name in _LAZY_NAMES:
        mod_name, attr_name = _LAZY_NAMES[name]
        if not _WARNED:
            _WARNED = True
            warnings.warn(
                f"tensorrt_llm.executor.result.{name} has moved to {mod_name}.{attr_name} "
                "and this compatibility shim will be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )
        return getattr(importlib.import_module(mod_name), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_NAMES))


__all__ = [
    "CompletionOutput",
    "GenerationResultBase",
    "DetokenizedGenerationResultBase",
    "GenerationResult",
    "IterationResult",
    "Logprob",
    "TokenLogprobs",
    "SimpleTokenLogprobs",
    "LogProbsResult",
    "ResponseWrapper",
    "get_metrics_dict",
    "compute_prompt_logprobs",
]
