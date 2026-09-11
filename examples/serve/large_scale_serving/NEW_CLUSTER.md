# Bringing this up on a cluster it has not run on

Work down the list. Each item says what to run and what a pass looks like,
because several of these fail in ways that are quiet rather than loud -- a
directory that is empty instead of a permission error, a gateway that answers
200 while routing nowhere, a recovery path that execs fine and then does
nothing.

`FLEET.md` covers the shape of the fleet itself; this covers everything around
it. Phases 1 and 2 decide the architecture, so do them before editing anything.

---

## 1. Access and network

The gateway proxies every request and probes every backend, so it has to reach
the serving ports. Nothing else on this list matters until that is true.

- [ ] **SSH to the login node.** `ssh <user>@<login-node> hostname`

- [ ] **The gateway host can reach a compute node's serving port.** Bring up one
      instance first (phase 5) if there is nothing listening yet, then from the
      gateway host: `curl -m 10 http://<compute-node>:8400/health`

      **No fallback.** If this fails the gateway cannot live off-cluster, and
      the only remaining option is to run it inside the cluster with
      `gateway.sbatch` and accept the wall clock, the handover gap and the
      thrash guard that comes with the chain.

- [ ] **Pick the discovery rung.** How the gateway learns an address exists.
      Test in order and stop at the first that works:

      1. From a *compute* node: `curl -m 10 http://<gateway>:8333/_gateway/health`
         → uncomment `gateway.register_url` in `fleet.yaml`. Nothing else to do.
      2. From the *login* node, same curl → run a bridge there: it reads the
         fleet directory locally and POSTs each record to `/_gateway/backend`.
         Login-node egress is often less restricted than a compute node's.
      3. Neither → the gateway pulls instead, which needs only the direction
         already proven above:

             rsync -a --delete --timeout=20 \
               <user>@<login>:<trace.root>/_fleet/<cluster>_<model>/ \
               <gateway fleet dir>/

         every ~10 s. No change to `serve.sh` or `gateway.py`: the file drop
         keeps working on the cluster and the gateway still just reads a
         directory. Heartbeats arrive intact so `--stale-after` behaves, and
         `--delete` gives correct deregistration.

- [ ] **Scheduler limits.** Confirm the account and QoS can hold the whole fleet
      at once: `sacctmgr show assoc user=$USER format=account,qos,maxjobs,maxsubmit`
      and the partition's `MaxTRESPU`. A QoS with `DenyOnLimit` rejects rather
      than queues, so an oversized fleet fails at submit rather than waiting.

- [ ] **Preemption policy.** `scontrol show partition <p> | grep -i preempt` and
      the QoS's `PreemptExemptTime`. This sets the availability ceiling: with a
      4 h exemption and a 20-40 min requeue, plan on ~89% and keep a burst
      reserve on top (see the capacity note in phase 8).

## 2. Filesystems

- [ ] **`trace.root` is writable and roomy.** Everything the deployment writes
      lands here: run directories, engine logs, perf metrics, sbatch logs and
      the fleet registrations. Default is `var/` inside the checkout.
      Engine logs alone reach ~150 GB per instance over seven days.

- [ ] **`trace.request_root`, if the traces go elsewhere.** `mkdir -p` it and
      write a file by hand to prove the account can. On failure `serve.sh` keeps
      the traces under the attempt and prints a warning rather than failing the
      run -- so the symptom is *missing traces*, not a crash.

- [ ] **The gateway host and the cluster share no filesystem.** Check rather
      than assume: `ls /lustre /scratch/fsw` on the gateway host. If they do
      share one, the file drop works unchanged and phase 1's rung 1 is moot.

## 3. Configuration

- [ ] **The six values in `deployments/fleet.yaml`** — `model.path`,
      `container.image`, `repo_dir`, `trace.root`, `slurm.account`,
      `slurm.partition`. `fleetctl up` checks the first three before submitting
      and names the wrong ones; the account and partition surface as a
      scheduler rejection.

- [ ] **`gpus_per_node` matches reality.** `scontrol show node <n> | grep Gres`.
      Node counts are derived from it, so a wrong value produces a fleet that
      will not schedule.

- [ ] **Rack topology decides the transport.** Find out whether one instance's
      ctx and gen workers land in the same rack.

      - Same rack: `UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp` in `server.env`.
      - Across racks: leave `UCX_TLS` unset (`serve.sh` unsets it when the YAML
        does not set it) and add `UCX_IB_MLX5_DEVX: "n"`,
        `UCX_NET_DEVICES: "rdma_vf_rail0:1,...rail3:1"`,
        `UCX_IB_TRAFFIC_CLASS: "96"`, `TRTLLM_NIXL_NUM_THREADS: "1"`;
        then drop `max_tokens_in_buffer` from the ctx configs and both
        `max_tokens_in_buffer` and `kv_cache_bounce_size_mb` from the gen ones.

      Note the half-configured state is easy to reach: `UCX_TLS` unset is right
      for RDMA, but `UCX_NET_DEVICES` stays on whatever the enroot hook injected
      (`eth0`) unless it is set.

- [ ] **`num_workers` in each topology YAML is nodes, not workers.** A TP4
      context worker is one node and a TP8 generation worker is two, so 6P1D
      reads as 8 while having 7 workers. `fleetctl` derives the same number
      independently and `serve.sh` refuses to submit when they disagree — so
      this is a cross-check, and disagreement means one of them is wrong.

- [ ] **`./fleetctl --config deployments/fleet.yaml render`** — topology only,
      no submission. Node counts per instance should match expectation.

## 4. The gateway

- [ ] **Users file.** One username per line; the username *is* the key. An
      unedited example admits nobody, which is deliberate.

- [ ] **Start it.** Under docker, three flags are load-bearing:

      - `--user $(id -u):$(id -g)` — without it the container is root, NFS
        root-squashes it, and every write fails EACCES. The tell is a
        `router_state.json` that never appears.
      - `--network host` — otherwise it binds a private address.
      - `--restart unless-stopped` — the URL not moving is the entire point.

- [ ] **Reachable at the *machine's* address, not just localhost.**
      `curl http://<ip>:8333/_gateway/health` → 200, and no key → 401.

- [ ] **Preemption recovery can actually run.** `fleetctl` calls squeue and
      sbatch, so it only works where the scheduler is. From off-cluster, point
      `--fleetctl` at `fleetctl-remote` with `FLEETCTL_SSH_HOST` and
      `FLEETCTL_REMOTE_DIR` set, and give `--fleet-config` **the path as that
      host sees it** — it is not checked locally, on purpose.

      Pass: `preemption recovery ready:` in the startup log. A warning there
      means nothing will bring a preempted instance back.

## 5. One instance

Do not skip to the full fleet. A single instance surfaces every config error
for the price of one allocation.

- [ ] `./fleetctl --config deployments/fleet.yaml up --only <name>`
- [ ] It registers: the file appears in the fleet directory, or
      `curl -H "x-api-key: $USER" http://<gw>:8333/_gateway/fleet` shows it.
- [ ] It answers through the gateway, not just directly.
- [ ] Traces are being written where phase 2 said they would be.
- [ ] `roll` it once and watch traffic not drop.

## 6. The fleet

- [ ] **Stagger the bring-up.** `fleetctl up` loops sbatch with no throttle.
      Cold start reads the checkpoint once per worker — seven workers per 6P1D
      instance at 433 GiB is ~3 TiB per instance, and a 27-instance fleet is
      ~92 TiB against a `start_timeout` of 7200 s. Bring them up in waves.
- [ ] `./fleetctl status` — configured against running.
- [ ] The gateway sees all of them, and `least_conversations` is spreading.

## 7. Kernel Factory

- [ ] `kf auth login --manual` — the token lasts about an hour and **BYO-LLM
      refuses service accounts**, so `kf auth ssa` is not an option. Long runs
      need the token renewed by a human; the launcher must stop rather than
      claim work it cannot start.
- [ ] `kf llm-endpoint create --name <name> --url http://<gateway>:8333`
      Plain http to a private address is accepted; no path, origin only.
- [ ] One campaign end to end before the queue is opened.

## 8. Campaigns

- [ ] **`--max-duration` is not the per-agent deadline.** `--effort` sets that
      (low 1 h, medium 2 h, high/max 4 h). `--max-duration` is a wall clock for
      the whole campaign and will cut a healthy one off mid-round.
- [ ] **Start conservative.** Measured anchor: 4×6P1D carried 200 concurrent
      campaigns at 11.4 req/s peak — 50 campaigns and 2.84 req/s per instance,
      with TTFT p50 swinging 0.6-38 s, so that is "survived", not "comfortable".
      Derate for availability (~89%) and keep a reserve for the three-instance
      loss a single scheduling decision has produced.
- [ ] `./kfrun --concurrency <n> --endpoint <name>` and watch the first round
      reconcile before walking away.
- [ ] **Check the queue is not stranding work.** `./kfq status` — anything stuck
      in `submitted` with nothing running against it means a claim was made and
      never reported.

## 9. Once it is running

- [ ] Error rate by cause. ~10% is the preemption baseline; above that, look at
      `by_route` — one route failing much harder than the others is a bug, not
      the scheduler.
- [ ] `cache_hit_ratio` in the rollups. Low means conversation affinity is not
      working, which is the whole reason the gateway exists.
- [ ] `pinned` against `--sticky-capacity`, and `rehomed` for sudden jumps.
- [ ] Per-node throughput, not raw TTFT, when comparing instance shapes — a
      bigger shape under equal load flatters itself.
