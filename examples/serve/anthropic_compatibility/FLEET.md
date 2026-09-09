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

## Dependencies not in this repository

`fleetctl` shells out to `serve.sh`, which is not checked in here — it belongs
to the reference GLM-5.2 deployment and is maintained with it. Put a copy (or a
symlink) beside `fleetctl`, along with the `server_configs/` the instances
name. `fleetctl render` will tell you what is missing before anything is
submitted.
