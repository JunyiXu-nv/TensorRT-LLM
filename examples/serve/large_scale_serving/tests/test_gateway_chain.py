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
"""The gateway chain queues one successor per job, even when Slurm requeues the job.

gateway.sbatch queues its successor at start. A job that Slurm requeues after a
node failure runs the script again under the same id; on jhb 10-15 the SGLang
gateway 995609 did, and two successors waited on it. queue_successor is taken
out of the script and run against stub squeue/sbatch commands.
"""

import os
import re
import stat
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SBATCH_SCRIPT = os.path.join(HERE, os.pardir, "gateway.sbatch")


def queue_successor_source():
    text = open(SBATCH_SCRIPT).read()
    m = re.search(r"^queue_successor\(\) \{\n.*?^\}\n", text, re.S | re.M)
    assert m, "queue_successor() not found in gateway.sbatch"
    return m.group(0)


def run(tmp_path, squeue_lines, squeue_rc=0, job_id="995609"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch.log"
    (bin_dir / "squeue").write_text(
        "#!/bin/bash\nprintf '%s' \"$SQUEUE_OUT\"\nexit ${SQUEUE_RC:-0}\n"
    )
    (bin_dir / "sbatch").write_text(f'#!/bin/bash\necho "$@" >> {log}\necho 2000001\n')
    for name in ("squeue", "sbatch"):
        path = bin_dir / name
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    script = queue_successor_source() + "queue_successor\n"
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SQUEUE_OUT="".join(line + "\n" for line in squeue_lines),
        SQUEUE_RC=str(squeue_rc),
        SLURM_JOB_ID=job_id,
        NODE="cpu-0034",
        LOG_DIR=str(tmp_path),
        SELF="/path/gateway.sbatch",
        USER="tester",
    )
    out = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30
    )
    submitted = log.read_text().splitlines() if log.exists() else []
    return out.stdout, submitted


def test_a_requeued_job_does_not_queue_a_second_successor(tmp_path):
    out, submitted = run(tmp_path, ["1003045 afterany:995609(unfulfilled)", "981854 (null)"])
    assert submitted == [], submitted
    assert "successor already queued: 1003045" in out, out


@pytest.mark.parametrize(
    "lines",
    [[], ["1015490 afterany:981854(unfulfilled)"], ["1200000 afterany:9956090(unfulfilled)"]],
    ids=["no-dependents", "other-chain", "longer-id-sharing-a-prefix"],
)
def test_a_first_start_queues_one_successor(tmp_path, lines):
    out, submitted = run(tmp_path, lines)
    assert len(submitted) == 1, submitted
    assert "--dependency=afterany:995609" in submitted[0] and "--nodelist=cpu-0034" in submitted[0]
    assert "successor queued: 2000001 on cpu-0034" in out, out


def test_a_shorter_id_is_not_matched_by_a_longer_one(tmp_path):
    out, submitted = run(tmp_path, ["1003045 afterany:995609(unfulfilled)"], job_id="99560")
    assert len(submitted) == 1, (out, submitted)


def test_an_unanswered_squeue_queues_as_before(tmp_path):
    out, submitted = run(tmp_path, [], squeue_rc=1)
    assert len(submitted) == 1, (out, submitted)
