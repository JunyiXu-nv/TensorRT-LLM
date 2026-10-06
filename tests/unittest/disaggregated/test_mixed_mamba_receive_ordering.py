# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A disaggregated receive must not land in memory an in-flight step still writes.

``MixedMambaHybridCacheManager`` (separate V1 KV pool + Python recurrent-state
pool; the Kimi K3 PD-disagg manager) frees a finished request's KV blocks and
state slot as soon as its response is handled. With the overlap scheduler that
happens while the step launched afterwards -- which still contains the finished
request and writes its recurrent state in place -- is running, and the state
pool hands slots out LIFO, so the next admission usually receives exactly that
slot. Local compute that reuses it is ordered behind the step on the execution
stream; a disaggregated receive is not: the context server's RDMA write (or
the bounce scatter on its own stream) lands independently and the step's late
write then overwrites the transferred state. The manager therefore records a
fence when it releases memory and waits for it before a generation-init
admission publishes its receive destinations.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    MambaCacheManager,
    MixedMambaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


def _manager(stream=None) -> MixedMambaHybridCacheManager:
    """A Mixed manager with its storage stubbed out; only the hooks under test run."""
    manager = object.__new__(MixedMambaHybridCacheManager)
    manager._stream = stream if stream is not None else Mock(name="execution_stream")
    manager._release_fences = ()
    manager.kv_connector_manager = None
    return manager


def _request(request_id: int, *, disagg_gen_init: bool) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        is_disagg_generation_init_state=disagg_gen_init,
        is_last_context_chunk=True,
    )


def _batch(*requests) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = list(requests)
    return batch


@pytest.fixture
def stubbed_storage(monkeypatch: pytest.MonkeyPatch) -> list:
    """Replace both pools' allocation and release with event recorders."""
    events: list = []
    monkeypatch.setattr(
        MambaCacheManager, "prepare_resources", lambda self, batch: events.append("alloc_state")
    )
    monkeypatch.setattr(
        KVCacheManager, "prepare_resources", lambda self, batch: events.append("alloc_kv")
    )
    monkeypatch.setattr(
        MambaCacheManager, "free_resources", lambda self, request: events.append("free_state")
    )
    monkeypatch.setattr(
        KVCacheManager,
        "free_resources",
        lambda self, request, pin_on_release=False: events.append("free_kv"),
    )
    return events


@pytest.fixture
def mock_cuda(monkeypatch: pytest.MonkeyPatch, stubbed_storage: list) -> SimpleNamespace:
    """CPU stand-ins for the CUDA stream/event calls the fence makes."""
    current_stream = Mock(name="current_stream")
    created: list = []

    def make_event():
        event = Mock(name=f"fence{len(created)}")
        event.record.side_effect = lambda stream: stubbed_storage.append(("record", stream))
        event.synchronize.side_effect = lambda: stubbed_storage.append(("wait", event))
        created.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: current_stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return SimpleNamespace(current_stream=current_stream, events=created)


def _executor(manager, publish) -> SimpleNamespace:
    executor = SimpleNamespace(
        resource_manager=SimpleNamespace(
            resource_managers={ResourceManagerType.KV_CACHE_MANAGER: manager}
        ),
    )
    executor._recv_disagg_gen_cache = publish
    return executor


@pytest.mark.cpu_only
def test_release_records_fence_after_both_pools_are_freed(stubbed_storage, mock_cuda):
    manager = _manager()

    MixedMambaHybridCacheManager.free_resources(manager, _request(1, disagg_gen_init=False))

    assert stubbed_storage == [
        "free_state",
        "free_kv",
        ("record", manager._stream),
        ("record", mock_cuda.current_stream),
    ]
    assert manager._release_fences == tuple(mock_cuda.events)


@pytest.mark.cpu_only
def test_disagg_admission_waits_for_release_fence_before_publishing(stubbed_storage, mock_cuda):
    """Generation-init admission: allocate, wait for released memory, then publish."""
    manager = _manager()
    finished, admitted = _request(1, disagg_gen_init=False), _request(2, disagg_gen_init=True)
    MixedMambaHybridCacheManager.free_resources(manager, finished)
    stubbed_storage.clear()
    publish = Mock(side_effect=lambda requests: stubbed_storage.append("publish"))

    PyExecutor._prepare_disagg_gen_init(_executor(manager, publish), [admitted])

    assert stubbed_storage == [
        "alloc_state",
        "alloc_kv",
        ("wait", mock_cuda.events[0]),
        ("wait", mock_cuda.events[1]),
        "publish",
    ]
    publish.assert_called_once_with([admitted])
    assert manager._release_fences == ()


@pytest.mark.cpu_only
def test_fence_is_consumed_once(stubbed_storage, mock_cuda):
    manager = _manager()
    MixedMambaHybridCacheManager.free_resources(manager, _request(1, disagg_gen_init=False))
    executor = _executor(manager, Mock())

    PyExecutor._prepare_disagg_gen_init(executor, [_request(2, disagg_gen_init=True)])
    PyExecutor._prepare_disagg_gen_init(executor, [_request(3, disagg_gen_init=True)])

    waits = [event for event in stubbed_storage if isinstance(event, tuple) and event[0] == "wait"]
    assert len(waits) == 2  # one per stream of the single release, not repeated


@pytest.mark.cpu_only
def test_local_context_admission_does_not_wait(stubbed_storage, mock_cuda):
    """Local prefill reuse is ordered on the execution stream; no host wait."""
    manager = _manager()
    MixedMambaHybridCacheManager.free_resources(manager, _request(1, disagg_gen_init=False))

    MixedMambaHybridCacheManager.prepare_resources(
        manager, _batch(_request(2, disagg_gen_init=False))
    )

    for event in mock_cuda.events:
        event.synchronize.assert_not_called()
    assert manager._release_fences == tuple(mock_cuda.events)


@pytest.mark.cpu_only
def test_admission_without_prior_release_does_not_wait(stubbed_storage, mock_cuda):
    manager = _manager()
    publish = Mock()

    PyExecutor._prepare_disagg_gen_init(
        _executor(manager, publish), [_request(2, disagg_gen_init=True)]
    )

    assert mock_cuda.events == []
    publish.assert_called_once()


@pytest.mark.cpu_only
def test_failed_fence_does_not_publish_receive(stubbed_storage, mock_cuda):
    manager = _manager()
    MixedMambaHybridCacheManager.free_resources(manager, _request(1, disagg_gen_init=False))
    mock_cuda.events[0].synchronize.side_effect = RuntimeError("fence failed")
    publish = Mock()

    with pytest.raises(RuntimeError, match="fence failed"):
        PyExecutor._prepare_disagg_gen_init(
            _executor(manager, publish), [_request(2, disagg_gen_init=True)]
        )

    publish.assert_not_called()
    assert manager._release_fences == tuple(mock_cuda.events)


@pytest.mark.cpu_only
def test_release_during_graph_capture_records_no_fence(stubbed_storage, mock_cuda, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    manager = _manager()

    MixedMambaHybridCacheManager.free_resources(manager, _request(1, disagg_gen_init=False))

    assert mock_cuda.events == []
    assert manager._release_fences == ()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_receive_into_recycled_slot_survives_inflight_step(stubbed_storage):
    """The finished request's in-flight step must not overwrite received state.

    ``slot`` stands for the recurrent-state slot: the step still running on
    the execution stream writes the finished request's state there, and the
    receive for the next admission (an independently ordered RDMA writer,
    modelled by a second stream) targets the same recycled slot.
    """
    execution_stream = torch.cuda.Stream()
    receive_stream = torch.cuda.Stream()
    slot = torch.zeros(4096, dtype=torch.int32, device="cuda")
    finished_state = torch.full_like(slot, 17)
    received_state = torch.full_like(slot, 29)
    torch.cuda.synchronize()

    def launch_step() -> None:
        with torch.cuda.stream(execution_stream):
            torch.cuda._sleep(100_000_000)
            slot.copy_(finished_state)

    def receive(_requests) -> None:
        with torch.cuda.stream(receive_stream):
            slot.copy_(received_state)

    # Establish that this schedule exposes the overwrite without the fence.
    launch_step()
    receive(None)
    torch.cuda.synchronize()
    assert torch.equal(slot, finished_state)

    slot.zero_()
    torch.cuda.synchronize()
    manager = _manager(execution_stream)
    launch_step()
    MixedMambaHybridCacheManager.free_resources(manager, _request(1, disagg_gen_init=False))
    PyExecutor._prepare_disagg_gen_init(
        _executor(manager, Mock(side_effect=receive)), [_request(2, disagg_gen_init=True)]
    )
    torch.cuda.synchronize()
    assert torch.equal(slot, received_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fence_does_not_wait_for_work_after_release(stubbed_storage):
    """Only work queued before the release is awaited, not later steps."""
    execution_stream = torch.cuda.Stream()
    manager = _manager(execution_stream)
    MixedMambaHybridCacheManager.free_resources(manager, _request(1, disagg_gen_init=False))
    later_step = torch.cuda.Event()
    with torch.cuda.stream(execution_stream):
        torch.cuda._sleep(500_000_000)
        later_step.record()
    completed_at_publication = []
    publish = Mock(side_effect=lambda _: completed_at_publication.append(later_step.query()))

    try:
        PyExecutor._prepare_disagg_gen_init(
            _executor(manager, publish), [_request(2, disagg_gen_init=True)]
        )
        assert completed_at_publication == [False]
    finally:
        later_step.synchronize()
