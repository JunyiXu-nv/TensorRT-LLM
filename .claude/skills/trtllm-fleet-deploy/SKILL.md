---
name: trtllm-fleet-deploy
description: >-
  Stand up a fleet of disaggregated trtllm-serve deployments on a SLURM cluster
  behind the long-lived gateway in examples/serve/large_scale_serving, including
  the cross-cluster case where the gateway does not run on the cluster it serves.
  Use when bringing the fleet up somewhere new, adding a cluster to an existing
  gateway, diagnosing a backend that registers but never turns healthy, or
  planning fleet size against preemption. Triggers on: deploy the fleet, fleetctl
  up, bring up disagg servers, gateway sees no backends, backend unhealthy,
  cross-cluster routing, fleet registration, flashinfer cold start, preemption
  recovery, kf-gateway.
tags: [infrastructure, serving, slurm, deployment]
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# Deploying the serving fleet

Everything here is measured, not inferred. Where a step says what a failure
looks like, that is because it happened and the symptom was not obvious.

`examples/serve/large_scale_serving/` holds the tooling: `serve.sh` runs one
disaggregated deployment, `fleetctl` runs many, `gateway.py` routes across them,
`fleet-sync` and `fleetctl-remote` bridge a gateway that is not on the cluster.
`NEW_CLUSTER.md` is the same procedure as a checklist for a human; this is the
version to execute.

## The shape of it

```
     agent client                     one stable URL
          │                                 │
          ▼                          ┌──────┴──────┐
   Kernel Factory  ───────────────▶  │   gateway   │  conversation affinity,
                                     └──────┬──────┘  health, draining
                          ┌─────────────────┼─────────────────┐
                       instance          instance          instance
                      (6 ctx + 1 gen, one SLURM job each)
```

The gateway wants to outlive the scheduler, because the only thing it exists to
provide is an address that does not move, and a job has a wall clock. So it
normally runs on a long-lived host **outside** the cluster. That is what makes
discovery the interesting part.

## Phase 0 — decide the architecture before editing anything

**One hard requirement, no fallback:** the gateway host must reach the cluster's
compute nodes on the serving ports. It probes every backend and proxies every
request. Test it before anything else — a closed port is fine, a timeout is not:

```bash
# from the gateway host, against any compute node IP
timeout 8 bash -c 'exec 3<>/dev/tcp/<compute-ip>/8400'
#   "connection refused" -> routable, nothing listening yet.  GOOD
#   timeout               -> filtered.  The gateway cannot live off-cluster.
```

If that fails, the gateway has to run inside the cluster with `gateway.sbatch`,
and you accept the chain: a successor submitted at every start, a gap at each
handover, and a thrash guard that will refuse to continue.

**Then pick the discovery rung.** Stop at the first that works:

| # | Test | Mechanism |
|---|---|---|
| 1 | compute node → `http://<gw>:8333/_gateway/health` | uncomment `gateway.register_url`; each job POSTs. Nothing else to do. |
| 2 | login node → same | a bridge on the login node reads the fleet dir locally and POSTs each record |
| 3 | neither | **gateway pulls**: run `fleet-sync` on the gateway host. No code change at all. |

Rung 3 needs only gateway-host → cluster SSH, which is the direction already
required, so it always works. Measured on aga: rungs 1 and 2 both failed
(login-node curl returned nothing), rung 3 worked first try.

## Phase 1 — survey the cluster

```bash
ssh <login-node> 'bash -s' < probe.sh     # see "Login shells are tcsh" below
```

Collect: partition and its `MaxTime`, idle node count, account and QoS,
`PreemptExemptTime` and `PreemptMode`, `gpus_per_node` from `scontrol show node`,
checkpoint and image paths, quota.

**Resolve every path with `readlink -f` before using it.** On aga the whole of
`/lustre/fsw` is a symlink into `/scratch/fsw`, where the filesystem actually
is. Configuring the symlink side and mounting it gives the container links
pointing at a path it has nothing mounted on — and a broken symlink inside a
container reads as an empty directory, not as an error.

## Phase 2 — write the fleet config

Copy an existing `deployments/fleet_*.yaml`. Per cluster:

- `cluster_name` — distinct per cluster unless you want the fleets merged into
  one pool. It keys the registration directory. Before merging, check Slurm job
  ids cannot collide across the clusters: the registration file is named after
  the job id.
- `repo_dir`, `model.path`, `container.image` — resolved paths
- `container.mounts` — the **real** filesystem, not a symlink to it
- `slurm.account` / `partition` / `qos` / `gpus_per_node`
- `trace.root` — `var/` inside the checkout; gitignored, and what
  `gateway.sbatch` defaults `GW_VAR` to so the two meet without being told twice
- `trace.request_root` — only if another team consumes the request traces
- **`FLASHINFER_WORKSPACE_BASE`** on shared storage — see below

`num_workers` in a topology YAML is **nodes, not workers**: a TP4 context worker
is one node and a TP8 generation worker is two, so 6P1D reads as 8 while having
7 workers. `fleetctl` derives the same number independently and `serve.sh`
refuses to submit when they disagree.

Validate locally before copying over: `./fleetctl --config <cfg> render`.

## Phase 3 — warm the FlashInfer cache with one instance

A cold start is ~36 min, of which ~30 is FlashInfer JIT (cc1plus on
`trtllm_batched_gemm_runner` alone runs 8-13 min). With the cache shared and
warm it is ~20.5 min, and the remainder — weights, CUDA graph capture, MoE
warmup — is not compressible.

The cache defaults under `$HOME` inside the container and dies with it. Two
equivalent ways to move it, differing only in path:

- `FLASHINFER_WORKSPACE_BASE=<shared>` in `server.env`
- mount `<shared>:/root/.cache/flashinfer`

`FLASHINFER_CACHE_DIR` is `$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer`, or
`$HOME/...` when the variable is unset — which is why the mount works.

**It must be writable**: a cache *hit* still rewrites ~31 files, so a read-only
mount fails. Setting the variable also disables the per-rank isolation in
`mpi_session.py`, which is the point: that bootstrap gives each rank
`rank-<n>` under an flock and skips ahead by the world size when a slot is
taken, so four 32-rank instances would occupy `rank-0` through `rank-127` and
compile the same modules 128 times over.

```bash
./fleetctl --config <cfg> up --only a      # one instance; wait for it to serve
```

Bring up the rest only after this one is healthy. Concurrent first-compiles into
one directory — and flock semantics on Lustre — is the part nobody has measured.

## Phase 4 — the gateway and the bridge

Run the gateway as its own minimal container, not inside a dev container:

```bash
SRV=<checkout>/examples/serve/large_scale_serving
docker run -d --name kf-gateway --restart unless-stopped --network host \
  --user "$(id -u):$(id -g)" -v "$SRV:/srv" python:3.12-slim \
  python3 /srv/gateway.py \
    --fleet-dir /srv/var/_fleet/<cluster>_<model> \
    --users /srv/var/gw/users.txt \
    --router-state /srv/var/gw/router_state.json \
    --host 0.0.0.0 --port 8333 --no-relay --log-level INFO
```

Three flags are load-bearing. `--user` because the container would otherwise be
root against root-squashing NFS and every write fails silently — the tell is a
`router_state.json` that never appears. `--network host` because a bridge
network binds a private address. `--restart unless-stopped` because the URL not
moving is the whole point.

Mounting the checkout rather than copying `gateway.py` means editing is
`vim` then `docker restart kf-gateway`. A copy went three weeks stale once,
running none of the patches, and nothing said so.

For rung 3, add the syncer alongside it (needs an image with rsync and
openssh-client, and a passwd entry for the uid — see the trap below):

```bash
docker run -d --name fleet-sync --restart unless-stopped \
  --user "$(id -u):$(id -g)" -e HOME=/home/user \
  -e FLEET_SYNC_SOURCE="<user>@<login>:<trace.root>/_fleet/<cluster>_<model>/" \
  -e FLEET_SYNC_DEST=/srv/var/_fleet/<cluster>_<model> \
  -v "$SRV:/srv" -v "$HOME/.ssh:/home/user/.ssh:ro" \
  fleet-sync:latest /srv/fleet-sync
```

For preemption recovery, `--fleet-config` plus `--fleetctl` pointed at
`fleetctl-remote` with `FLEETCTL_SSH_HOST` and `FLEETCTL_REMOTE_DIR` set.
`--fleet-config` is the path **on the remote host** and is deliberately not
checked locally. Look for `preemption recovery ready:` in the startup log; a
warning there means nothing will bring a preempted instance back.

## Phase 5 — the rest of the fleet, then verify

```bash
./fleetctl --config <cfg> up          # stagger it; see the I/O note
./fleetctl --config <cfg> status
curl -H "x-api-key: $USER" http://<gw>:8333/_gateway/fleet
```

A backend must appear **and** turn `healthy=true`. Registered but never healthy
is the signature failure — see the table below.

Cold start reads the checkpoint once per worker: seven workers per 6P1D
instance at 433 GiB is ~3 TiB per instance, so a 27-instance fleet is ~92 TiB
against a `start_timeout` of 7200 s. `fleetctl up` loops sbatch with no
throttle. Bring them up in waves.

## Traps, with the symptom each produces

| Symptom | Cause |
|---|---|
| `Illegal variable name` on any remote command | Login shell is tcsh. Feed scripts in: `ssh host 'bash -s' < script.sh` |
| Backend registers, `/health` is 200 on the cluster, gateway says `healthy=False` | Registration carried the **node name** and the gateway cannot resolve it. `serve.sh` registers an IPv4 address; if you touch that code, keep `getent ahostsv4` — `getent hosts` can return IPv6, and the gateway parses registrations with `^http://([^:/]+):(\d+)$`, which rejects anything containing colons. It then reads as a backend that never appeared. |
| Container writes nothing to NFS, no error | Container is root, NFS root-squashes it. `--user $(id -u):$(id -g)` |
| `No user exists for uid <n>` | OpenSSH refuses to run without a passwd entry. Add the uid to the image. |
| ssh `Permission denied` from a container that works on the host | The container's username for that uid is not yours. Put `user@` in the destination. |
| A directory is empty where it should have files | A symlink pointing outside the container's mounts, or a filesystem the cluster cannot see. Neither is an error. |
| `up` right after `down` says "already has a job" | `scancel` is asynchronous. Wait for `squeue` to clear. |
| Shell script dies at a `${VAR:?message}` line | An apostrophe in the message. Bash parses that word with quoting active. |
| `pgrep -f <pattern>` finds a process you just killed | The pattern is in your own script's argv. Check with `ps -eo comm,args`. |
| UCX `bind(addr=fdcd:...) failed` / `uct_iface_open(tcp/rdma_vf_rail0)` | Noise. OpenMPI picked an IPv6-only RoCE VF for the UCX PML and falls back; KV transfer is fine. Do not tune `UCX_TLS` for it. |
| Every node downloading the same 200 MB wheels | `install_repo: true` installs per node, per start. Worth pointing `PIP_CACHE_DIR` at shared storage. |

## Sizing against preemption

Availability is the exemption over the exemption plus the recovery, and
recovery is requeue latency **plus a cold start**:

```
4h05m / (4h05m + 21-36 min requeue + 20.5 min start) ≈ 81-86%
```

Then subtract a burst reserve: one scheduling decision has taken three
instances within four seconds, so running at the full derated number leaves
that event nowhere to go. For 27 instances: 27 × 0.83 ≈ 22, minus 3 ≈ **19
usable**.

Concurrency anchor, measured: 4 × 6P1D carried 200 concurrent campaigns at
11.4 req/s peak — 50 campaigns and 2.84 req/s per instance, but with TTFT p50
swinging 0.6-38 s, so that is "survived", not "comfortable". Start at ~35 per
instance. The gateway itself is not the constraint: backend count is free
(4/16/31/46 backends all cost ~10 ms/request), and it runs healthy to ~170
req/s and saturates near 420.

## Keeping this current

Every row in the traps table came from a deployment that failed in a way that
was not obvious. When a new one appears, add it with its **symptom** rather
than its cause — the symptom is what the next person will be searching for.
Update the measured numbers rather than averaging them in; if a cold start
changes, say what changed it.
