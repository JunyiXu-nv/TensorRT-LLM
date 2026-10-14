#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A preempted job's controller advertises its new end time and keeps serving.

Under PreemptParameters=send_user_signal, a job submitted with
--signal=B:USR1@<s> is preempted by a USR1 to the batch shell alone and runs on
for its QoS GraceTime, with Slurm's EndTime moved to the end of that grace. The
controller has to survive the signal and publish the new end time in its fleet
registration, so the gateway drains the backend instead of routing to it until
the job is killed.

These drive the real controller (`serve.sh run`) against the stubbed scheduler
in launcher_stubs.py, with a squeue stub whose end time the test moves.
"""

import json
import os
import signal
import subprocess
import time

import pytest
from launcher_stubs import SERVE_SH, make_stub_bin, stub_env, write_deployment

START_END = 4102444800  # 2100-01-01T00:00:00Z, the end time the job starts with
GRACE_END_TEXT = "2030-01-01T00:20:00"  # squeue's %e after the preemption (TZ=UTC)
GRACE_END = 1893457200

# %T: running until the test removes the job; %e: whatever the test last wrote.
SQUEUE = r"""#!/usr/bin/env python3
import json, os, sys
with open(os.environ["STUB_LOG"], "a") as handle:
    handle.write(json.dumps({"cmd": "squeue", "argv": sys.argv[1:]}) + "\n")
if "%e" in sys.argv:
    path = os.environ["STUB_END_FILE"]
    if os.path.exists(path):
        sys.stdout.write(open(path).read())
    sys.exit(0)
if not os.path.exists(os.environ["STUB_GONE_FILE"]):
    print("RUNNING")
"""


@pytest.fixture
def world(tmp_path):
    root = str(tmp_path)
    bin_dir, log = make_stub_bin(root)
    squeue = os.path.join(bin_dir, "squeue")
    with open(squeue, "w") as handle:
        handle.write(SQUEUE)
    os.chmod(squeue, 0o755)
    env = stub_env(
        root,
        bin_dir,
        log,
        SLURM_JOB_NODELIST="node1",
        SLURM_JOB_END_TIME=str(START_END),
        STUB_END_FILE=os.path.join(root, "squeue_end"),
        STUB_GONE_FILE=os.path.join(root, "job_gone"),
        TZ="UTC",
    )
    record = os.path.join(root, "var", "_fleet", "kffleet_glm5.3", "4242.json")
    return root, env, record


def read_record(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def wait_for(predicate, timeout=30.0, step=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(step)
    return None


def start_controller(root, env):
    deployment = write_deployment(root)
    return subprocess.Popen(
        ["bash", SERVE_SH, "run", "--yaml", deployment, "--label", "i00"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_controller(proc, env):
    open(env["STUB_GONE_FILE"], "w").close()
    try:
        out, _ = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        pytest.fail("controller did not exit once the job was gone:\n" + out)
    return proc.returncode, out


def test_usr1_advertises_the_moved_end_time_and_keeps_serving(world):
    root, env, record = world
    proc = start_controller(root, env)
    try:
        first = wait_for(lambda: read_record(record))
        assert first, "no fleet registration appeared"
        assert first["end_time"] == START_END, first

        # The preemption: Slurm moves the end time, then signals the batch shell.
        with open(env["STUB_END_FILE"], "w") as handle:
            handle.write(GRACE_END_TEXT + "\n")
        os.kill(proc.pid, signal.SIGUSR1)

        moved = wait_for(lambda: (read_record(record) or {}).get("end_time") == GRACE_END)
        assert moved, "registration still says %r" % read_record(record)
        assert proc.poll() is None, "the controller died on USR1"
    finally:
        rc, out = stop_controller(proc, env)
    assert rc == 0, out
    assert "Slurm moved the end of job 4242 to 2030-01-01 00:20:00" in out, out


def test_usr1_with_an_unmoved_end_time_changes_nothing(world):
    """The signal also comes <s> seconds before an ordinary wall clock."""
    root, env, record = world
    proc = start_controller(root, env)
    try:
        assert wait_for(lambda: read_record(record)), "no fleet registration appeared"
        with open(env["STUB_END_FILE"], "w") as handle:
            handle.write("2100-01-01T00:00:00\n")
        os.kill(proc.pid, signal.SIGUSR1)
        # Five checks at the stubbed 0.05 s loop: well inside a second.
        time.sleep(2)
        assert proc.poll() is None, "the controller died on USR1"
        assert read_record(record)["end_time"] == START_END, read_record(record)
    finally:
        rc, out = stop_controller(proc, env)
    assert rc == 0, out
    assert "USR1 from Slurm, but the end time has not moved" in out, out
    assert "Slurm moved the end of job" not in out, out


def test_a_later_end_time_is_not_advertised(world):
    """Only an earlier end is a preemption; never extend the advertised life."""
    root, env, record = world
    proc = start_controller(root, env)
    try:
        assert wait_for(lambda: read_record(record)), "no fleet registration appeared"
        with open(env["STUB_END_FILE"], "w") as handle:
            handle.write("2101-01-01T00:00:00\n")
        os.kill(proc.pid, signal.SIGUSR1)
        time.sleep(2)
        assert proc.poll() is None, "the controller died on USR1"
        assert read_record(record)["end_time"] == START_END, read_record(record)
    finally:
        rc, out = stop_controller(proc, env)
    assert rc == 0, out
