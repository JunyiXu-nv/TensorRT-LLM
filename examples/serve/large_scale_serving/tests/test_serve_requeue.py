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
"""A requeued job must not overwrite the record of the run before it.

Slurm requeues a job under the same job id, and the batch script starts a new
controller. Its run directory is named <user>_<MMDDHH>_<jobid>_<name>_<label>,
so a requeue within the same hour lands on the run directory of the run that
just died. The new run then started over at attempt-001 and truncated that
attempt's launcher, server and worker logs. On 10-09 that erased the only
server-side record of two lhr runs that ended in NODE_FAIL. The job's sbatch log
had the same problem one level up: Slurm truncates --output on every requeue
unless the job asks to append.

These drive the real controller (`serve.sh run`) twice under one job id against
the stubbed scheduler in launcher_stubs.py, and `serve.sh submit` once.
"""

import os
import re

import pytest
from launcher_stubs import calls, make_stub_bin, serve_sh, stub_env, write_deployment


@pytest.fixture
def world(tmp_path):
    root = str(tmp_path)
    bin_dir, log = make_stub_bin(root)
    env = stub_env(root, bin_dir, log, SLURM_JOB_NODELIST="node1", SLURM_JOB_END_TIME="4102444800")
    return root, env, log


def run_controller(env, deployment):
    done = serve_sh(env, "run", "--yaml", deployment, "--label", "i00", timeout=180)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    return re.search(r"^run dir: (\S+)$", done.stdout, re.M).group(1), done.stdout


def read(path):
    with open(path) as handle:
        return handle.read()


def test_a_requeued_run_keeps_the_previous_runs_attempts_and_metadata(world):
    root, env, _ = world
    deployment = write_deployment(root)

    first_dir, _ = run_controller(env, deployment)
    first_attempt = os.path.join(first_dir, "attempt-001")
    assert os.path.isdir(first_attempt)
    # What the dead run left behind, as a requeue would find it.
    with open(os.path.join(first_attempt, "server.log"), "a") as handle:
        handle.write("first run: slurmstepd: *** STEP CANCELLED DUE TO NODE FAILURE ***\n")
    first_launcher_log = read(os.path.join(first_attempt, "launcher.log"))
    first_metadata = read(os.path.join(first_dir, "run_metadata.txt"))

    # The requeue: same job id, and the stubbed queue reports it running again.
    os.remove(env["STUB_SERVE_MARKER"])
    second_dir, stdout = run_controller(env, deployment)
    if second_dir != first_dir:
        pytest.skip("the two runs straddled an hour boundary, so their run dirs differ anyway")

    assert "this run starts at attempt 2" in stdout, stdout
    assert os.path.isdir(os.path.join(second_dir, "attempt-002"))
    assert read(os.path.join(first_attempt, "launcher.log")) == first_launcher_log
    assert "NODE FAILURE" in read(os.path.join(first_attempt, "server.log"))
    assert (
        read(os.path.join(second_dir, "control", "current_attempt_dir"))
        .strip()
        .endswith("attempt-002")
    )
    kept = [name for name in os.listdir(second_dir) if name.startswith("run_metadata.txt.before-")]
    assert len(kept) == 1, os.listdir(second_dir)
    assert read(os.path.join(second_dir, kept[0])) == first_metadata
    assert any(name.startswith("deployment.yaml.before-") for name in os.listdir(second_dir))
    assert os.path.isfile(os.path.join(second_dir, "deployment.yaml"))


def test_a_first_run_starts_at_attempt_one(world):
    root, env, _ = world
    run_dir, stdout = run_controller(env, write_deployment(root))
    assert "already holds" not in stdout
    assert sorted(n for n in os.listdir(run_dir) if n.startswith("attempt-")) == ["attempt-001"]
    assert not [n for n in os.listdir(run_dir) if ".before-" in n]


def test_submit_asks_slurm_to_append_to_the_job_log(world):
    root, env, log = world
    done = serve_sh(env, "submit", "--yaml", write_deployment(root))
    assert done.returncode == 0, done.stdout + done.stderr
    (argv,) = calls(log, "sbatch")
    assert "--open-mode" in argv and argv[argv.index("--open-mode") + 1] == "append", argv
