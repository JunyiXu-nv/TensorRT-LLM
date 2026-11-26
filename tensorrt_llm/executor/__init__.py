# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import warnings

from .request import *
from .utils import EngineDeadError, PostprocWorkerConfig, RequestError

_LAZY_NAMES: dict[str, tuple[str, str]] = {
    "GenerationExecutorWorker":
    ("tensorrt_llm.executor.worker", "GenerationExecutorWorker"),
    "GenerationExecutor":
    ("tensorrt_llm.engine_client.executor", "GenerationExecutor"),
    "CppExecutorError":
    ("tensorrt_llm.engine_client.executor", "CppExecutorError"),
    "PostprocWorker":
    ("tensorrt_llm.engine_client.postproc_worker", "PostprocWorker"),
    "GenerationExecutorProxy": ("tensorrt_llm.engine_client.proxy",
                                "GenerationExecutorProxy"),
    "GenerationExecutorFrontendProxy": ("tensorrt_llm.engine_client.proxy",
                                        "GenerationExecutorFrontendProxy"),
    "CompletionOutput": ("tensorrt_llm.engine_client.result",
                         "CompletionOutput"),
    "GenerationResultBase": ("tensorrt_llm.engine_client.result",
                             "GenerationResultBase"),
    "DetokenizedGenerationResultBase": ("tensorrt_llm.engine_client.result",
                                        "DetokenizedGenerationResultBase"),
    "GenerationResult": ("tensorrt_llm.engine_client.result",
                         "GenerationResult"),
    "IterationResult": ("tensorrt_llm.engine_client.result", "IterationResult"),
    "Logprob": ("tensorrt_llm.executor.logprobs", "Logprob"),
    "LogProbsResult": ("tensorrt_llm.executor.logprobs", "LogProbsResult"),
    "ResponseWrapper": ("tensorrt_llm.executor.utils", "ResponseWrapper"),
}

_WARNED: set[str] = set()


def __getattr__(name: str):
    if name in _LAZY_NAMES:
        mod_name, attr_name = _LAZY_NAMES[name]
        if mod_name != "tensorrt_llm.executor.worker" and name not in _WARNED:
            _WARNED.add(name)
            warnings.warn(
                f"tensorrt_llm.executor.{name} has moved to {mod_name}.{attr_name} "
                "and will be removed from tensorrt_llm.executor in a future release.",
                FutureWarning,
                stacklevel=2,
            )
        return getattr(importlib.import_module(mod_name), attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_NAMES))


__all__ = [
    "PostprocWorker",
    "PostprocWorkerConfig",
    "GenerationRequest",
    "LoRARequest",
    "PromptAdapterRequest",
    "GenerationExecutorWorker",
    "GenerationExecutorProxy",
    "RequestError",
    "EngineDeadError",
    "CompletionOutput",
    "GenerationResultBase",
    "DetokenizedGenerationResultBase",
    "GenerationResult",
    "IterationResult",
]
