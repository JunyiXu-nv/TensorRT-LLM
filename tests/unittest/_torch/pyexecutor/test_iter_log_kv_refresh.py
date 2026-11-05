# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How often the iteration log rebuilds its KV cache stats.

The default keeps one stats build for ``ITER_LOG_KV_UTIL_REFRESH_ITERS``
iterations. ``TLLM_ITER_LOG_KV_REFRESH_ITERS`` overrides it, so a deployment
whose log analysis differences the cumulative KV counters of consecutive lines
can refresh on every line; a value that is not a positive integer keeps the
default.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.profiling import (
    ITER_LOG_KV_REFRESH_ITERS_ENV_VAR_NAME,
    ITER_LOG_KV_UTIL_REFRESH_ITERS,
    iter_log_kv_refresh_iters,
)

pytestmark = pytest.mark.cpu_only


def test_unset_keeps_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ITER_LOG_KV_REFRESH_ITERS_ENV_VAR_NAME, raising=False)
    assert iter_log_kv_refresh_iters() == ITER_LOG_KV_UTIL_REFRESH_ITERS


@pytest.mark.parametrize("raw, expected", [("1", 1), ("4", 4), ("64", 64)])
def test_a_positive_value_overrides_the_default(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv(ITER_LOG_KV_REFRESH_ITERS_ENV_VAR_NAME, raw)
    assert iter_log_kv_refresh_iters() == expected


@pytest.mark.parametrize("raw", ["0", "-3", "every", ""])
def test_a_value_that_is_not_a_positive_integer_keeps_the_default(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(ITER_LOG_KV_REFRESH_ITERS_ENV_VAR_NAME, raw)
    assert iter_log_kv_refresh_iters() == ITER_LOG_KV_UTIL_REFRESH_ITERS
