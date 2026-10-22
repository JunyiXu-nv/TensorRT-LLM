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
"""A submit from inside another job must not hand that job's memory limit to the new one.

The gateway submits relays from inside its own Slurm job, and so does a keeper.
sbatch exports the submitter's environment into the job it creates, and every
srun step in that job reads SLURM_MEM_PER_NODE (or _PER_CPU / _PER_GPU) as its
--mem. On 2026-10-22 the jhb gateway job held 96G. Relays it submitted were
granted 900G per node, but every worker step ran under 96G, and the context
ranks were OOM-killed while loading weights, attempt after attempt.
"""

import json
import os

from launcher_stubs import make_stub_bin, serve_sh, stub_env, write_deployment

SBATCH_ENV_RECORDER = r"""#!/usr/bin/env python3
import json, os
with open(os.environ["SBATCH_ENV_LOG"], "w") as handle:
    json.dump({k: v for k, v in os.environ.items() if k.startswith("SLURM_MEM")}, handle)
print("Submitted batch job 4343")
"""


def test_submit_drops_the_enclosing_jobs_memory_variables(tmp_path):
    root = str(tmp_path)
    bin_dir, log = make_stub_bin(root)
    sbatch = os.path.join(bin_dir, "sbatch")
    with open(sbatch, "w") as handle:
        handle.write(SBATCH_ENV_RECORDER)
    os.chmod(sbatch, 0o755)
    recorded = os.path.join(root, "sbatch_env.json")
    env = stub_env(
        root,
        bin_dir,
        log,
        SBATCH_ENV_LOG=recorded,
        SLURM_MEM_PER_NODE="98304",
        SLURM_MEM_PER_CPU="4096",
        SLURM_MEM_PER_GPU="24576",
    )
    done = serve_sh(env, "submit", "--yaml", write_deployment(root))
    assert done.returncode == 0, done.stdout + done.stderr
    with open(recorded) as handle:
        assert json.load(handle) == {}, "sbatch saw the submitter's memory limit"
