# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""RouteTraceWriter: the ADP route trace must never block the executor loop.

A stalled flush of this file on a shared filesystem once held the executor
loop long enough for the hang detector to kill a serving instance (2026-10-08).
"""

import threading
import time

from tensorrt_llm._torch.pyexecutor.route_trace_writer import RouteTraceWriter, open_route_trace


class _StuckFile:
    """Stands in for a file on a stalled filesystem: write blocks until released."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.lines = []

    def write(self, line):
        self.entered.set()
        self.release.wait()
        self.lines.append(line)

    def flush(self):
        pass

    def close(self):
        pass


class _FailingFile:
    def write(self, line):
        raise OSError("Input/output error")

    def flush(self):
        pass

    def close(self):
        pass


def _writer_on(fake, tmp_path, max_pending):
    writer = RouteTraceWriter(str(tmp_path / "adp_route_trace.jsonl"), max_pending=max_pending)
    # Swap the real file out before anything is written to it; the thread
    # only touches _file once a line arrives.
    writer._file.close()
    writer._file = fake
    return writer


def test_lines_are_written_in_order_and_flushed_by_close(tmp_path):
    path = tmp_path / "adp_route_trace.jsonl"
    writer = RouteTraceWriter(str(path))
    for i in range(100):
        writer.write(f'{{"iter": {i}}}\n')
    assert writer.close(timeout=5.0)
    lines = path.read_text().splitlines()
    assert lines == [f'{{"iter": {i}}}' for i in range(100)]
    assert writer.dropped == 0


def test_write_does_not_block_while_the_file_is_stuck(tmp_path):
    stuck = _StuckFile()
    writer = _writer_on(stuck, tmp_path, max_pending=4)
    writer.write("0\n")
    assert stuck.entered.wait(5.0), "the writer thread never reached the file"
    started = time.monotonic()
    for i in range(1, 50):
        writer.write(f"{i}\n")
    elapsed = time.monotonic() - started
    # One line is held by the blocked write, four wait in the queue, and the
    # rest are dropped -- all without the caller waiting on the file.
    assert elapsed < 1.0, elapsed
    assert writer.dropped == 50 - 1 - 4
    stuck.release.set()
    assert writer.close(timeout=5.0)
    assert len(stuck.lines) == 5


def test_close_gives_up_on_a_writer_that_stays_stuck(tmp_path):
    stuck = _StuckFile()
    writer = _writer_on(stuck, tmp_path, max_pending=4)
    writer.write("held\n")
    started = time.monotonic()
    assert writer.close(timeout=0.5) is False
    assert time.monotonic() - started < 2.0
    # Writes after close are ignored rather than queued behind a dead writer.
    writer.write("late\n")
    # Let the abandoned writer finish, so the test leaves no thread behind.
    stuck.release.set()
    writer._thread.join(5.0)
    assert not writer._thread.is_alive()
    assert stuck.lines == ["held\n"]


def test_write_errors_are_counted_not_raised(tmp_path):
    writer = _writer_on(_FailingFile(), tmp_path, max_pending=16)
    for i in range(3):
        writer.write(f"{i}\n")
    assert writer.close(timeout=5.0)
    assert writer.write_errors == 3


def test_an_unopenable_path_yields_no_writer(tmp_path):
    assert open_route_trace(str(tmp_path / "missing" / "trace.jsonl")) is None
