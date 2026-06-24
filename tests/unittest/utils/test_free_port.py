# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the shared CI-aware free-port helper get_free_port_in_ci."""

import socket

import pytest

from tensorrt_llm import _utils
from tensorrt_llm._utils import get_free_port_in_ci


@pytest.fixture(autouse=True)
def _reset_ports_in_use():
    saved = set(_utils.PORTS_IN_USE)
    _utils.PORTS_IN_USE.clear()
    yield
    _utils.PORTS_IN_USE.clear()
    _utils.PORTS_IN_USE.update(saved)


def _bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False


def test_get_free_port_in_ci_unique():
    """Repeated calls within a process never hand out the same port twice."""
    ports = [get_free_port_in_ci() for _ in range(50)]
    assert len(set(ports)) == len(ports), "duplicate port handed out"
    assert _utils.PORTS_IN_USE.issuperset(ports)


def test_respects_container_port_range(monkeypatch):
    """When CI exports a port range, picks fall inside it."""
    start, num = 49200, 64
    monkeypatch.setenv("CONTAINER_PORT_START", str(start))
    monkeypatch.setenv("CONTAINER_PORT_NUM", str(num))
    for _ in range(20):
        port = get_free_port_in_ci()
        assert start <= port < start + num, f"{port} outside CI range"


def test_falls_back_when_range_exhausted(monkeypatch):
    """A fully used-up range falls back to a system-assigned ephemeral port."""
    start, num = 49300, 4
    monkeypatch.setenv("CONTAINER_PORT_START", str(start))
    monkeypatch.setenv("CONTAINER_PORT_NUM", str(num))
    # Mark the whole range as already in use.
    _utils.PORTS_IN_USE.update(range(start, start + num))
    port = get_free_port_in_ci()
    assert port not in range(start, start + num)
    assert _bindable(port)


def test_no_range_uses_system_port(monkeypatch):
    monkeypatch.delenv("CONTAINER_PORT_START", raising=False)
    monkeypatch.delenv("CONTAINER_PORT_NUM", raising=False)
    port = get_free_port_in_ci()
    assert isinstance(port, int) and port > 0
    assert _bindable(port)
