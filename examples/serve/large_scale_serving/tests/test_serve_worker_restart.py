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
"""server.worker_restart: one dead disaggregated worker is restarted alone.

With the switch on, launch_disagg starts an etcd beside the proxy, the proxy finds its
workers there instead of in a static url list, every worker states its role, and a
worker that exits is started again on its own nodes and port while the rest keep
running -- until it has done so max_per_hour times, when the attempt ends as it always
did. Off, nothing about the launch changes.

The stubs come from launcher_stubs.py; srun is replaced here so that the registry and
the proxy outlive the test's interesting part and one chosen worker exits at once.
"""

import os

import pytest
import yaml
from launcher_stubs import _write, calls, make_stub_bin, serve_sh, stub_env, write_engine_config

# Every srun is recorded. Etcd, the proxy and the healthy workers stay up for
# STUB_LIFE seconds; the worker whose port is STUB_DYING_PORT exits at once.
SRUN = r"""#!/usr/bin/env python3
import json, os, sys, time
argv = sys.argv[1:]
with open(os.environ["STUB_LOG"], "a") as handle:
    handle.write(json.dumps({"cmd": "srun", "argv": argv}) + "\n")
if os.environ.get("STUB_DYING_PORT") in argv:
    sys.exit(1)
time.sleep(float(os.environ.get("STUB_LIFE", "2")))
"""


@pytest.fixture
def world(tmp_path):
    root = str(tmp_path)
    bin_dir, log = make_stub_bin(root)
    _write(os.path.join(bin_dir, "srun"), SRUN)
    # The registry's /health, polled before the proxy starts.
    _write(os.path.join(bin_dir, "curl"), "#!/bin/sh\nexit 0\n")
    return root, bin_dir, log


def write_disagg_deployment(root, worker_restart=None):
    """Two context workers and one generation worker, one rank each, three nodes."""
    for path in ("repo", "model"):
        os.makedirs(os.path.join(root, path), exist_ok=True)
    topology = os.path.join(root, "topology.yaml")
    with open(topology, "w") as handle:
        yaml.safe_dump(
            {
                "context_servers": {"num_instances": 2, "router": {"type": "conversation"}},
                "generation_servers": {"num_instances": 1},
            },
            handle,
        )
    server = {
        "port": 8400,
        "capture": False,
        "install_repo": False,
        "disagg": {
            "config": topology,
            "ctx": {"config": write_engine_config(os.path.join(root, "ctx.yaml")), "instances": 2},
            "gen": {"config": write_engine_config(os.path.join(root, "gen.yaml")), "instances": 1},
        },
    }
    if worker_restart is not None:
        server["worker_restart"] = worker_restart
    cfg = {
        "cluster_name": "kffleet",
        "repo_dir": os.path.join(root, "repo"),
        "model": {"name": "glm5.3", "path": os.path.join(root, "model"), "tool_parser": "fake"},
        "slurm": {"account": "acct", "partition": "part", "time": "01:00:00", "gpus_per_node": 1},
        "container": {"image": os.path.join(root, "image.sqsh"), "mounts": [root + ":" + root]},
        "server": server,
        "trace": {"root": os.path.join(root, "var")},
    }
    path = os.path.join(root, "deployment.yaml")
    with open(path, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return path


def make_disagg_attempt(root):
    """What start_attempt leaves behind: control/nodes and the engine config snapshots."""
    run_dir = os.path.join(root, "var", "2026-11", "03", "tester_110304_4242_kffleet_glm5.3_w00")
    os.makedirs(os.path.join(run_dir, "control"), exist_ok=True)
    with open(os.path.join(run_dir, "control", "nodes"), "w") as handle:
        handle.write("node1\nnode2\nnode3\n")
    attempt = os.path.join(run_dir, "attempt-001")
    os.makedirs(attempt, exist_ok=True)
    for role in ("ctx", "gen"):
        write_engine_config(os.path.join(attempt, "%s_config.yaml" % role))
    return attempt


def launch(world, worker_restart=None, **env_extra):
    root, bin_dir, log = world
    deployment = write_disagg_deployment(root, worker_restart)
    attempt = make_disagg_attempt(root)
    env = stub_env(root, bin_dir, log, **env_extra)
    result = serve_sh(env, "launch", "--yaml", deployment, "--attempt-dir", attempt)
    return result, attempt


def serving_steps(log):
    """The srun argvs of the registry, the proxy and the workers.

    Each step runs in the background and is recorded when its stub starts, so
    the record's order is not the launch order, and a step started just before
    the launch is torn down may not be recorded at all.
    """
    return [argv for argv in calls(log, "srun") if "-lc" in argv]


def role_of(argv):
    script = argv[argv.index("-lc") + 1]
    if "exec etcd" in script:
        return "etcd"
    if "trtllm-serve disaggregated" in script:
        return "proxy"
    port = argv[argv.index("_") + 2]
    return {"8401": "ctx-0", "8402": "ctx-1", "8403": "gen-0"}[port]


def test_off_keeps_the_static_url_list_and_ends_on_the_first_exit(world):
    result, attempt = launch(world, STUB_DYING_PORT="8402")
    # The failed worker's status ends the launch (set -e under pipefail), as it
    # always has; nothing is started again.
    assert result.returncode != 0
    steps = serving_steps(world[2])
    roles = [role_of(argv) for argv in steps]
    assert {"proxy", "ctx-1"} <= set(roles) <= {"proxy", "ctx-0", "ctx-1", "gen-0"}
    assert len(roles) == len(set(roles))
    assert not any("--server_role" in argv for argv in steps)
    with open(os.path.join(attempt, "disagg_config.yaml")) as handle:
        proxy_cfg = yaml.safe_load(handle)
    assert "disagg_cluster" not in proxy_cfg
    assert proxy_cfg["context_servers"]["urls"] == ["node1:8401", "node2:8402"]
    assert proxy_cfg["generation_servers"]["urls"] == ["node3:8403"]
    assert "restarting it alone" not in result.stdout
    for role in ("ctx", "gen"):
        with open(os.path.join(attempt, "%s_config.yaml" % role)) as handle:
            assert "disagg_cluster" not in yaml.safe_load(handle)


def test_on_puts_a_registry_beside_the_proxy_and_registers_every_worker(world):
    result, attempt = launch(
        world, {"enabled": True, "max_per_hour": 2, "min_ctx": 1}, STUB_DYING_PORT="none"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    steps = {role_of(argv): argv for argv in serving_steps(world[2])}
    assert sorted(steps) == ["ctx-0", "ctx-1", "etcd", "gen-0", "proxy"]
    etcd = steps["etcd"]
    assert etcd[etcd.index("--nodelist") + 1] == "node1"
    script = etcd[etcd.index("-lc") + 1]
    assert "--name 'kffleet_glm5.3-4242'" in script
    assert "http://0.0.0.0:8490" in script
    roles = {
        worker: steps[worker][steps[worker].index("--server_role") + 1]
        for worker in ("ctx-0", "ctx-1", "gen-0")
    }
    assert roles == {"ctx-0": "context", "ctx-1": "context", "gen-0": "generation"}
    assert "--server_role" not in steps["proxy"]

    with open(os.path.join(attempt, "disagg_config.yaml")) as handle:
        proxy_cfg = yaml.safe_load(handle)
    assert "urls" not in proxy_cfg["context_servers"]
    assert "urls" not in proxy_cfg["generation_servers"]
    assert proxy_cfg["disagg_cluster"] == {
        "cluster_uri": "etcd://node1:8490",
        "cluster_name": "kffleet_glm5.3-4242",
        "minimal_instances": {"context_servers": 1, "generation_servers": 1},
        "heartbeat_interval_sec": 5,
        "inactive_timeout_sec": 30,
    }
    for role in ("ctx", "gen"):
        with open(os.path.join(attempt, "%s_config.yaml" % role)) as handle:
            worker_cfg = yaml.safe_load(handle)
        assert worker_cfg["disagg_cluster"]["cluster_uri"] == "etcd://node1:8490"
        assert worker_cfg["disagg_cluster"]["cluster_name"] == "kffleet_glm5.3-4242"


def test_a_dead_worker_is_restarted_alone_until_its_hourly_budget(world):
    result, attempt = launch(
        world, {"enabled": True, "max_per_hour": 2}, STUB_DYING_PORT="8402", STUB_LIFE="3"
    )
    assert result.returncode != 0
    steps = serving_steps(world[2])
    ctx1 = [argv for argv in steps if role_of(argv) == "ctx-1"]
    # Started once, then twice more on its own node; nothing else was restarted.
    assert len(ctx1) == 3
    assert {argv[argv.index("--nodelist") + 1] for argv in ctx1} == {"node2"}
    assert [role_of(argv) for argv in steps].count("ctx-0") == 1
    assert [role_of(argv) for argv in steps].count("gen-0") == 1
    assert result.stdout.count("worker ctx-1 exited (rc=1); restarting it alone") == 2
    assert "after 2 restart(s) in the last hour" in result.stdout
    assert "tearing down the rest" in result.stdout
    # Its leftovers are killed on its own node before each restart, never elsewhere.
    pkill_hosts = [argv[0] for argv in calls(world[2], "ssh") if "mgmn_" in " ".join(argv)]
    assert pkill_hosts.count("node2") >= 4  # TERM and KILL, twice
    with open(os.path.join(os.path.dirname(attempt), "control", "state")) as handle:
        assert handle.read().startswith("running attempt 1, 2 worker restart(s), last ctx-1 at ")


def test_the_registry_needs_a_disaggregated_deployment(world):
    root, bin_dir, log = world
    deployment = write_disagg_deployment(root, {"enabled": True})
    with open(deployment) as handle:
        cfg = yaml.safe_load(handle)
    cfg["server"].pop("disagg")
    cfg["server"]["config"] = write_engine_config(os.path.join(root, "engine.yaml"))
    with open(deployment, "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    result = serve_sh(stub_env(root, bin_dir, log), "resolve", "--yaml", deployment)
    assert result.returncode != 0
    assert "server.worker_restart needs a disaggregated deployment" in result.stdout + result.stderr
