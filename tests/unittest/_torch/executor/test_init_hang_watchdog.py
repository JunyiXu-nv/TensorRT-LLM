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
"""InitHangWatchdog timer/phase behavior and its hard-kill on a wedged init (no GPU)."""

import os
import signal
import subprocess
import sys
import time
from unittest import mock

from tensorrt_llm._torch.pyexecutor.hang_detector import InitHangWatchdog


def _wait_for(predicate, deadline_s: float) -> bool:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_fires_after_phase_timeout():
    wd = InitHangWatchdog(timeout=1)
    fired = []
    with mock.patch.object(wd, "_fire", lambda *a: fired.append(a)):
        wd.arm("phase-a")
        assert _wait_for(lambda: fired, 3.0), "watchdog did not fire"
    assert fired[0][0] == "phase-a"


def test_cancel_prevents_fire():
    wd = InitHangWatchdog(timeout=1)
    fired = []
    with mock.patch.object(wd, "_fire", lambda *a: fired.append(a)):
        wd.arm("phase-a")
        wd.cancel()
        time.sleep(1.5)
    assert fired == []
    # cancel() is idempotent and safe on an unarmed watchdog.
    wd.cancel()


def test_checkpoint_gives_each_phase_a_fresh_window():
    wd = InitHangWatchdog(timeout=1)
    fired = []
    with mock.patch.object(wd, "_fire", lambda *a: fired.append(a)):
        wd.arm("phase-a")
        for phase in ("phase-b", "phase-c", "phase-d"):
            time.sleep(0.5)  # < timeout: progress within the window
            wd.checkpoint(phase)
        assert fired == []
        wd.cancel()


def test_zero_timeout_disables():
    wd = InitHangWatchdog(timeout=0)
    fired = []
    with mock.patch.object(wd, "_fire", lambda *a: fired.append(a)):
        wd.arm("phase-a")
        time.sleep(0.5)
    assert not wd.enabled
    assert fired == []


def test_env_zero_disables_module_instance():
    with mock.patch.dict(os.environ, {"TLLM_INIT_HANG_TIMEOUT": "0"}):
        wd = InitHangWatchdog()
        assert not wd.enabled
    with mock.patch.dict(os.environ, {"TLLM_INIT_HANG_TIMEOUT": "not-a-number"}):
        wd = InitHangWatchdog()
        assert wd.enabled  # falls back to the default


def test_wedged_init_is_hard_killed():
    """A process wedged during init is SIGKILL'd at the phase timeout.

    With MPI unavailable, propagate_hard_kill falls back to self-SIGKILL,
    which the parent observes as returncode -SIGKILL (== -9).
    """
    script = (
        "from tensorrt_llm._torch.pyexecutor.hang_detector import InitHangWatchdog\n"
        "import time\n"
        "wd = InitHangWatchdog(timeout=2)\n"
        "wd.arm('wedged-init')\n"
        "time.sleep(600)  # simulate a wedged init phase\n"
    )
    env = {**os.environ, "TLLM_DISABLE_MPI": "1"}
    # Generous timeout: the subprocess pays a cold `import tensorrt_llm`.
    proc = subprocess.run([sys.executable, "-c", script], env=env, timeout=300, capture_output=True)
    assert proc.returncode == -signal.SIGKILL, (
        f"expected self-SIGKILL (-9), got {proc.returncode}; stderr tail: {proc.stderr[-500:]!r}"
    )
