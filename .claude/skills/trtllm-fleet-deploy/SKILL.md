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
`fleet-sync`, `fleetctl-remote` and `slurm-remote` bridge a gateway that is
not on the cluster.
`NEW_CLUSTER.md` is the same procedure as a checklist for a human; this is the
version to execute.

## Ask for these before starting

Four things vary per deployment and cannot be guessed. Propose the default and
let the user correct it rather than interrogating them:

| Ask | Default to propose |
|---|---|
| **Which cluster** | none — there is no safe guess. `oci-jhb-slurm-1` and `oci-aga-slurm-1` are the two in use; they share no filesystem and each needs its own `fleet_*.yaml` |
| **How many instances, and which shapes** | a 4-instance 6P1D fleet for a first bring-up; 9 each of 6P1D/7P1D/8P1D (27 instances, 243 nodes, 972 GPUs) for the full run |
| **Test or full** | test. A first bring-up on a cluster should be one instance, then four, before anything larger |
| **Is the FlashInfer cache already warm** | check rather than ask — look at the shared cache directory. Cold means phase 3 first |

**Do not ask about these.** They are settled, and re-opening them each time
costs a round trip for no decision:

- the gateway runs on ipp2-1730 at `10.176.206.169:8333` and is already up
- the model is GLM-5.2-NVFP4
- the discovery rung is decided by the network tests in phase 0, not by
  preference — run them and take the answer
- paths, account and partition follow from the cluster

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

**If Kernel Factory is the consumer, there is a second hard requirement, and it
is not about reachability.** `kf llm-endpoint create` validates the gateway's
resolved IP against an operator-managed allowlist and 400s outside it — so the
gateway's address decides whether the fleet is usable at all, independently of
whether everything works. Check it *before* choosing a host, with one throwaway
`create`; it costs a second and it is not deducible from any network test. On
aga this was found after the fleet was healthy and serving: every address on
aga is outside the list — login node, GPU compute, and both CPU partitions —
while two different subnets on jhb are inside it. The grain is the cluster, so
one probe per cluster answers it for the whole cluster, and hunting for a host
of the right kind within a blocked cluster is wasted time.

**Then pick the discovery rung.** Stop at the first that works:

| # | Test | Mechanism |
|---|---|---|
| 1 | compute node → `http://<gw>:8333/_gateway/health` | uncomment `gateway.register_url`; each job POSTs. Nothing else to do. |
| 2 | login node → same | a bridge on the login node reads the fleet dir locally and POSTs each record |
| 3 | neither | **gateway pulls**: run `fleet-sync` on the gateway host. No code change at all. |

Rung 3 needs only gateway-host → cluster SSH, which is the direction already
required, so it always works. Measured on aga: against a gateway on ipp2,
rungs 1 and 2 both failed and rung 3 worked first try — but against a gateway
on HSG, aga's *compute* nodes reached it directly, so rung 1 was available.

Which is the point: the rung is a property of **the pair of networks**, not of
the fleet's cluster. Re-test it when the gateway moves; a recorded "rung 3 on
this cluster" is only true for the gateway it was measured against.

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
warm it is **14.5 min** — measured on aga with three instances starting at
once — and the remainder, weights plus CUDA graph capture plus MoE warmup, is
not compressible. Weight loading dominates it: 1961 shards per rank, and the
ranks spread out over several minutes as Lustre serves them.

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

Bring up the rest only after this one is healthy — but once it is, they can go
together. Three at once into a warm shared cache on Lustre was measured on aga:
all three healthy in 14.5 min, 11 of the cache's 14 files rewritten during the
overlap, no stall and no corruption. So concurrent *hits* are fine, and the
`rank-*` directories stayed absent, which is the check that the isolation is
really off.

Concurrent **first**-compiles into one directory is still unmeasured — every
instance in that run was a hit. Warm the cache with one instance first and the
question does not arise.

## Phase 4 — the gateway and the bridge

**Choosing the host.** It needs three properties, and they are independent:
reachability to the fleet's compute nodes, a KF-allowlisted address if Kernel
Factory is the consumer, and longevity. A 7-day CPU node satisfies the last
without a shared login node's exposure — `gateway.sbatch` already targets
`--partition=cpu --qos=cpu-long --time=7-00:00:00`, and because its successor
is submitted with `--nodelist=$NODE`, **the address survives the handover**, so
a registered KF endpoint does not need repointing every seven days.

Set `GW_REMOTE_HOST` and `GW_REMOTE_DIR` when the fleet is on another cluster.
One switch, on purpose: it redirects fleetctl, squeue and the fleet directory
together, and any subset of those is wrong in a way that looks fine.

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

For preemption recovery, **three** flags, because recovery is a query and an
action and off-cluster they need separate bridges:

```
--fleet-config  <path as the login node sees it>     # not checked locally
--fleetctl      /srv/fleetctl-remote                 # the action: fleetctl up
--slurm-wrapper /srv/slurm-remote                    # the query: is it really gone?
```

with `FLEETCTL_SSH_HOST`, `FLEETCTL_REMOTE_DIR` and `SLURM_SSH_HOST` in the
environment. Wrapping only the action is the trap: nothing is submitted until
squeue says the scheduler has no record of the job, and squeue runs directly
rather than through fleetctl.

Look for `preemption recovery ready:` in the startup log — it names the
transport it proved, so `(scheduler reachable via /srv/slurm-remote)` is the
part worth reading. A warning there means nothing will bring a preempted
instance back.

This also means the gateway container needs ssh, a key, and a passwd entry for
its uid — so run it on the same image as `fleet-sync` rather than on bare
`python:3.12-slim`.

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
| `authentication_error` from Kernel Factory, or any unauthenticated caller | The users file has no `anonymous` entry. `kf llm-endpoint` stores and sends no credential by design, so a BYO LLM endpoint must answer unauthenticated requests. Add `anonymous`; the file is reread on change, no restart. This costs attribution, not security — the username was always the key and always guessable. |
| `Illegal variable name` on any remote command | Login shell is tcsh. Feed scripts in: `ssh host 'bash -s' < script.sh` |
| Backend registers, `/health` is 200 on the cluster, gateway says `healthy=False` | Registration carried the **node name** and the gateway cannot resolve it. `serve.sh` registers an IPv4 address; if you touch that code, keep `getent ahostsv4` — `getent hosts` can return IPv6, and the gateway parses registrations with `^http://([^:/]+):(\d+)$`, which rejects anything containing colons. It then reads as a backend that never appeared. |
| Container writes nothing to NFS, no error | Container is root, NFS root-squashes it. `--user $(id -u):$(id -g)` |
| `No user exists for uid <n>` | OpenSSH refuses to run without a passwd entry. Add the uid to the image. |
| ssh `Permission denied` from a container that works on the host | The container's username for that uid is not yours. Put `user@` in the destination. |
| A directory is empty where it should have files | A symlink pointing outside the container's mounts, or a filesystem the cluster cannot see. Neither is an error. |
| Log says `preemption recovery ready`, but a preempted instance is never replaced | Only the action was bridged. `fleetctl up` goes through `fleetctl-remote`, but the gateway gates it on `squeue -j <id>`, which ran locally — off-cluster that is `OSError`, which reads as "cannot tell", and every caller correctly declines to act on an answer it did not get. Silent forever. Add `--slurm-wrapper slurm-remote`; the startup probe now exercises both halves. |
| Duplicate instances appear after a recovery | Two gateways had `--fleet-config` set against the same fleet. Recovery is a *write*: exactly one gateway may have it enabled. When migrating, disable it on the old one **before** enabling it on the new one — both will see the same backends vanish and both will call `fleetctl up`. |
| `fleetctl up` submits an instance that already has a job | Its "already has a job" check reads the registration files, and a job writes its own only once it is up. In the minutes-long window between `sbatch` and that write, a second `fleetctl up` sees the slot as empty. Harmless when a human runs it twice; automatic recovery makes the race routine. Check `fleetctl status` for a repeated instance letter after any recovery. |
| `up` right after `down` says "already has a job" | `scancel` is asynchronous. Wait for `squeue` to clear. |
| Shell script dies at a `${VAR:?message}` line | An apostrophe in the message. Bash parses that word with quoting active. |
| `pgrep -f <pattern>` finds a process you just killed | The pattern is in your own script's argv. Check with `ps -eo comm,args`. |
| UCX `bind(addr=fdcd:...) failed` / `uct_iface_open(tcp/rdma_vf_rail0)` | Noise. OpenMPI picked an IPv6-only RoCE VF for the UCX PML and falls back; KV transfer is fine. Do not tune `UCX_TLS` for it. |
| Every node downloading the same 200 MB wheels | `install_repo: true` installs per node, per start. Worth pointing `PIP_CACHE_DIR` at shared storage. |

## Sizing against preemption

Availability is the exemption over the exemption plus the recovery, and
recovery is requeue latency **plus a cold start**:

```
4h05m / (4h05m + 21-36 min requeue + 14.5 min start) ≈ 83-87%
```

Then subtract a burst reserve: one scheduling decision has taken three
instances within four seconds, so running at the full derated number leaves
that event nowhere to go. For 27 instances: 27 × 0.85 ≈ 23, minus 3 ≈ **20
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
