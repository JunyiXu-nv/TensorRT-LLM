<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Anthropic Compatibility Serving

`serve.sh` is the single entry point. One deployment YAML fully describes a
cluster + model pair; everything else is derived.

```bash
./serve.sh submit --yaml deployments/computelab_glm5.2.yaml --label bringup
./serve.sh status  <run_dir>
./serve.sh restart <run_dir>
./serve.sh quit    <run_dir>
```

Inside an existing `salloc` allocation, skip `sbatch` and run the controller
directly:

```bash
nohup ./serve.sh run --yaml deployments/computelab_glm5.2.yaml \
    > controller.log 2>&1 &
```

The controller owns the server: when it exits, it tears the server down. Under
`sbatch` that is what you want, since the controller *is* the job. From an
interactive shell, detach it with `nohup` — otherwise closing the terminal
sends SIGHUP and kills a server that may still be loading weights.

## Layout

```text
serve.sh                            launcher: submit / run / launch / start|restart|stop|quit|status
deployments/
  computelab_glm5.2.yaml            one file per cluster + model pair
  computelab_deepseek_v4.yaml
  server_configs/                   trtllm-serve --config files (one per model)
analysis/                           audit-log analysis and plotting
```

## Supported models

`model.name` must be one of `glm5.2` or `deepseek_v4` — the whitelist lives in
`serve.sh`, so a typo cannot silently produce a job, container and trace
directory that look almost right. Everything else about the model comes from the
YAML:

```yaml
cluster_name: computelab

model:
  name: deepseek_v4
  path: /home/scratch.trt_llm_data/llm-models/DeepSeek-V4-Pro
  tool_parser: deepseek_v4

server:
  config: server_configs/deepseek_v4_agg_tep8.yaml
  extra_args: []      # extra trtllm-serve flags, if a model ever needs them
```

Adding a checkpoint means one new deployment YAML, one file under
`server_configs/`, and one entry in the `serve.sh` whitelist.

## What the YAML holds, and what it does not

Everything the run needs is in the deployment YAML: `cluster_name`, `repo_dir`,
`model.{name, path, tool_parser}`, `slurm.{account, partition, reservation, qos,
segment, time, nodes, gpus_per_node, extra_args}`, `container.{image, mounts}`,
`server.{config, port, install_repo, numactl, extra_args, env}`, and
`trace.root`.

The layout is explicit: `slurm.nodes × slurm.gpus_per_node` gives `ntasks`, and
the server config's `tensor_parallel_size × pipeline_parallel_size ×
context_parallel_size` must agree with it. A mismatch is a hard error, so
changing TP without changing the node count fails at submit time rather than
half-way through a job.

These are derived and must not be written by hand:

| Derived | From |
|---|---|
| `ntasks`, `ntasks-per-node`, `gres` | `slurm.nodes` and `slurm.gpus_per_node` |
| Slurm job name, container name, `--output` | `<cluster_name>_<model.name>` + job ID |
| pkill pattern used to clear a previous server | `basename(model.path)` — that is what appears on the `trtllm-serve` command line, unlike `model.name` |
| Trace directory | `${USER}_$(date +%m%d%H)_${SLURM_JOB_ID}_<cluster_name>_<model.name>[_<label>]` |
| `TRTLLM_ANTHROPIC_AUDIT_LOG`, `TRTLLM_ANTHROPIC_BENCH_CAPTURE_DIR`, `TRTLLM_ANTHROPIC_LCP_TRACKING` | always on, pointed at the attempt directory |

One code path covers any node count, so a one-node and a two-node layout differ
only by `slurm.nodes`.

`server.env` overrides the built-in defaults: `TLLM_LOG_LEVEL=INFO`,
`TRTLLM_{SERVER,WORKER}_DISABLE_GC=1`, `TRTLLM_ENABLE_PDL=1`,
`ENROOT_ALLOW_DEV=yes`, `NCCL_GRAPH_MIXING_SUPPORT=0`, `MIMALLOC_PURGE_DELAY=0`.

## Run directory

```text
<trace.root>/serli_073023_3378564_computelab_glm5.2_bringup/
├── deployment.yaml  run_metadata.txt  server_url
├── control/                 job_id, nodes, attempt, state, current_attempt_dir
└── attempt-001/
    ├── launch_cmd.sh        the srun command lines this attempt actually ran
    ├── server_config.yaml   snapshot of the config this attempt served with
    ├── launcher.log  install.log  server.log
    ├── anthropic_audit.jsonl
    └── anthropic_message_capture/
```

The server config is snapshotted per attempt rather than per run: it can be
edited between a `stop` and the next `start`, and the attempt copy is the one
that ran.

`run_metadata.txt` records the branch, commit, model path, container, topology,
and node list, so a run is reproducible from the directory alone.

> The capture directory holds raw `/v1/messages` bodies. Treat it as sensitive.
