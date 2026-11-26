# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layering, pickle-free wire, and compatibility shim checks for ``tensorrt_llm.engine_client``."""

from __future__ import annotations

import ast
import importlib
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.cpu_only
def test_proto_zmq_wire_has_no_pickle_imports() -> None:
    targets = [
        REPO_ROOT / "tensorrt_llm" / "contracts" / "serve_engine" / "v1" / "zmq_wire.py",
        REPO_ROOT / "tensorrt_llm" / "engine_client" / "transport.py",
        REPO_ROOT / "tensorrt_llm" / "engine_client" / "transport_zmq.py",
        REPO_ROOT / "tensorrt_llm" / "engine_client" / "frame_decoder.py",
        REPO_ROOT / "tensorrt_llm" / "engine_client" / "capabilities.py",
        REPO_ROOT / "tensorrt_llm" / "engine_client" / "request_mapper.py",
        REPO_ROOT / "tensorrt_llm" / "engine_client" / "facade.py",
    ]
    for path in targets:
        mods = _imported_modules(path)
        assert not any(
            m == "pickle" or m.startswith("pickle.") or m == "cloudpickle" for m in mods
        ), f"Pickle import forbidden in {path}: {mods}"


@pytest.mark.cpu_only
def test_engine_client_proto_modules_do_not_import_serve_or_llm() -> None:
    new_modules = [
        "interface.py",
        "capabilities.py",
        "request_mapper.py",
        "frame_decoder.py",
        "transport.py",
        "transport_zmq.py",
        "facade.py",
    ]
    for name in new_modules:
        path = REPO_ROOT / "tensorrt_llm" / "engine_client" / name
        mods = _imported_modules(path)
        forbidden = [
            m
            for m in mods
            if m.startswith("tensorrt_llm.serve") or m.startswith("tensorrt_llm.llmapi.llm")
        ]
        assert not forbidden, f"{name} must not import serve/llmapi.llm: {forbidden}"


@pytest.mark.cpu_only
def test_proto_bridge_does_not_import_frontend_packages() -> None:
    path = REPO_ROOT / "tensorrt_llm" / "executor" / "proto_bridge.py"
    mods = _imported_modules(path)
    forbidden = [
        m
        for m in mods
        if m.startswith("tensorrt_llm.engine_client")
        or m.startswith("tensorrt_llm.serve")
        or m.startswith("tensorrt_llm.llmapi")
    ]
    assert not forbidden, f"executor/proto_bridge.py must not import frontend: {forbidden}"


@pytest.mark.cpu_only
def test_executor_result_shim_identity_and_warning() -> None:
    import tensorrt_llm.engine_client.result as new_result
    import tensorrt_llm.executor.logprobs as new_lp
    import tensorrt_llm.executor.utils as new_utils

    old_mod = importlib.import_module("tensorrt_llm.executor.result")
    old_mod._WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert old_mod.GenerationResult is new_result.GenerationResult
        assert old_mod.Logprob is new_lp.Logprob
        assert old_mod.ResponseWrapper is new_utils.ResponseWrapper
    assert any(issubclass(w.category, FutureWarning) for w in caught)
