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
"""A line writer for the ADP route trace that never blocks the executor loop.

The route trace (``TRTLLM_ADP_ROUTE_TRACE``) used to be written and flushed by
rank 0 inside the executor loop. Its file lives in the serving attempt's
directory on a shared filesystem, and a write there can stall for minutes: on
2026-10-08 an OST recovery stalled one flush long enough for the hang detector
to declare the executor hung and kill a serving instance.

Lines are handed to a daemon thread through a bounded queue instead. The
executor-side ``write`` only enqueues; when the queue is full the line is
dropped and counted, so a filesystem that stays stuck costs trace lines rather
than the instance. ``flush`` is a no-op for the same reason: the writer thread
flushes whenever it has drained the queue.
"""

import queue
import threading
import time
from typing import Optional

from tensorrt_llm.logger import logger

# Lines held while the file is slow. One line is one routing batch; at the
# fleet's rate of a few batches per second this covers several minutes of
# stall before anything is dropped.
DEFAULT_MAX_PENDING = 4096

# Write failures are reported at most this often, with the count since the
# last report, so a failing filesystem cannot flood the log.
_ERROR_LOG_INTERVAL_S = 60.0

_STOP = object()


class RouteTraceWriter:
    """Append lines to ``path`` on a background thread."""

    def __init__(self, path: str, max_pending: int = DEFAULT_MAX_PENDING):
        # Opened here, on the caller's thread, so a path that cannot be
        # opened fails at construction exactly as the plain file did.
        self._file = open(path, "a", encoding="utf-8")
        self.path = path
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=max_pending)
        self.dropped = 0
        self.write_errors = 0
        self._errors_unreported = 0
        self._last_error_log = 0.0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="adp-route-trace-writer", daemon=True
        )
        self._thread.start()

    def write(self, line: str) -> None:
        """Queue one line. Never blocks; drops and counts when the queue is full."""
        if self._closed:
            return
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    def flush(self) -> None:
        """No-op: the writer thread flushes whenever it has drained the queue."""

    def close(self, timeout: float = 5.0) -> bool:
        """Stop after writing what is queued; give up after ``timeout`` seconds.

        Returns whether the writer finished. A writer stuck on the filesystem
        is left behind as a daemon thread rather than holding up shutdown.
        """
        if self._closed:
            return not self._thread.is_alive()
        self._closed = True
        deadline = time.monotonic() + timeout
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(max(0.0, deadline - time.monotonic()))
        finished = not self._thread.is_alive()
        if self.dropped or self.write_errors or not finished:
            logger.warning(
                f"ADP route trace {self.path}: {self.dropped} line(s) dropped "
                f"while the file was slow, {self.write_errors} write error(s)"
                f"{'' if finished else ', writer still blocked at close'}"
            )
        return finished

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                break
            try:
                self._file.write(item)
                if self._queue.empty():
                    self._file.flush()
            except (OSError, ValueError) as e:
                self._note_error(e)
        try:
            self._file.flush()
            self._file.close()
        except (OSError, ValueError) as e:
            self._note_error(e)

    def _note_error(self, error: Exception) -> None:
        self.write_errors += 1
        self._errors_unreported += 1
        now = time.monotonic()
        if now - self._last_error_log >= _ERROR_LOG_INTERVAL_S:
            logger.warning(
                f"Failed to write ADP route trace {self.path}: "
                f"{error} ({self._errors_unreported} error(s) "
                f"since the last report)"
            )
            self._last_error_log = now
            self._errors_unreported = 0


def open_route_trace(path: str) -> Optional[RouteTraceWriter]:
    """A writer for ``path``, or None (with a warning) if it cannot be opened."""
    try:
        return RouteTraceWriter(path)
    except OSError as e:
        logger.warning(f"Failed to open ADP route trace {path}: {e}")
        return None
