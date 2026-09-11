# Running several disaggregated deployments as one pool

`serve.sh` brings up one deployment. `fleetctl` runs several of them behind a
single gateway, each configured on its own, and is what makes the gateway's
conversation routing worth having: with one backend there is nothing to route
between.

```
./fleetctl up                bring up every instance that is not running
./fleetctl up --only a b     bring up named ones
./fleetctl status            configured instances against running jobs
./fleetctl roll a            replace one without a gap in service
./fleetctl down a            stop one
./fleetctl render            write the per-instance files without submitting
```

## What makes them a pool

Every instance registers into the **same** fleet directory. `serve.sh` names
each registration after its job id, so they do not collide, and the gateway
discovers the directory rather than any one file. Adding an instance is
therefore a submission and nothing else — the gateway needs no reconfiguring
and picks it up on its next sweep.

This is the opposite of what a single deployment wants. `agadisagg_glm5.2.yaml`
warns that sharing `cluster_name` + `model.name` makes two deployments register
as backends of each other; here that is the point.

## Instances are independent

They differ in prefill/decode ratio, and the right ratio is not settled — agent
turns replay their whole history, so this workload is prefill-heavy in a way
ordinary chat is not, and the only way to find the ratio is to run more than
one and compare. `fleet.yaml` therefore carries a `defaults:` block and a list
of instances, each free to override the topology and the engine configs.

Node counts are derived, never declared:

    world_size = ctx_ranks x ctx_instances + gen_ranks x gen_instances

`serve.sh` refuses to submit when `nodes x gpus_per_node` disagrees with that
sum, and deriving it here means a topology change cannot leave a stale node
count behind.

## What can and cannot be changed under load

Engine settings — tensor parallelism, speculative decoding, the memory
fraction — are read once, when the weights load. Changing them means replacing
the instance, which is what `roll` does: start the replacement, wait for it to
serve, then stop the old one. Traffic never stops, because the gateway keeps
routing to whatever is healthy and re-homes a conversation whose backend goes
away.

Routing itself *is* changeable under load, through the gateway rather than
here: `/_gateway/route`, `/_gateway/pin`, `/_gateway/drain`.

## Bringing it up somewhere else

Everything needed is here: `serve.sh` runs one deployment, `fleetctl` runs
several, `gateway.sbatch` runs the gateway in front of them, and
`deployments/server_configs/` holds the engine and topology files.

What is *not* here is anything that names your cluster. `deployments/fleet.yaml`
carries the paths of the cluster it was written on, and six values have to
change:

    defaults.model.path          the checkpoint
    defaults.container.image     the .sqsh
    defaults.repo_dir            this checkout
    defaults.trace.root          <repo_dir>/examples/serve/large_scale_serving/var
    defaults.trace.request_root  where the request traces go, if not under trace.root
    defaults.slurm.account       your account
    defaults.slurm.partition     a partition long enough for slurm.time

`trace.root` is a path rather than something derived because serve.sh takes it
literally, but it is meant to stay as `var/` inside the checkout: that is one
directory the deployment writes instead of five scattered ones, and it is what
`gateway.sbatch` defaults `GW_VAR` to, so the gateway and the serving jobs meet
without either being told twice. It holds the run directories, the engine logs,
the perf metrics, the sbatch logs and the fleet registrations, and it is
gitignored.

`trace.request_root` is the one exception. The per-request and per-response
traces are the output another team reads, so they can be sent elsewhere;
serve.sh symlinks `<attempt>/request_trace` at it, which keeps every reader
here working unchanged.

`fleetctl up` checks the first three before submitting anything and names the
ones that are wrong. The account and partition it cannot check, so a wrong one
surfaces as a scheduler rejection.

Then:

    ./fleetctl --config deployments/fleet.yaml render     # topology only, no submit
    ./fleetctl --config deployments/fleet.yaml up
    GW_ROOT=<your root> GW_ACCOUNT=<your account> \
        sbatch -o <your root>/gw/log/gateway.%j.out gateway.sbatch

`gateway.sbatch` names nothing itself; `GW_ROOT` is the only required setting
and the rest derive from it. Add `GW_FLEET_CONFIG=deployments/fleet.yaml` to
let the gateway bring back instances the scheduler has dropped entirely --- see
"Preemption" in the README for why that is narrower than it sounds.

The gateway watches `<trace.root>/_fleet/<cluster_name>_<model.name>`, which is
where serve.sh writes registrations, so `gateway.sbatch` and `fleet.yaml` have
to agree on that path.

## Sizing

A worker must fit on one node. The reference 6P1D shape is TP4 per worker,
which needs 8 nodes of 4 GPUs; the 1P1D shape needs 2. A checkpoint too large
for one node's GPUs forces a worker to span nodes, and that is worth avoiding
for reasons beyond throughput -- a single MPI world spread across nodes was
what made GLM-5.3 (1.4 TB, TP8, two nodes per worker) fail to start about one
time in five here, always with a null-pointer segfault on the second node.
GLM-5.2-NVFP4 is 433 GiB and fits TP4 on one node, and has not done it once.
