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

`NEW_CLUSTER.md` is the checklist for this, and covers the parts that are not
about the fleet -- network reachability, how the gateway is told a backend
exists, the transport the rack layout implies, and the Kernel Factory wiring.
What follows is the fleet's own share of it.

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

## What an attempt leaves under `trace.request_root`

One directory per attempt, self-describing: everything a training export
needs to trust a record is in the record or beside it, and nothing has to be
looked up in this deployment's `var/` tree.

    <request_root>/<run>/<attempt>/
      provenance.txt              copied from the attempt (commit, dirty, py_manifest_sha256, chat template)
      <UTC hour>/
        requests-<pid>.jsonl      body as sent + thread, kf, gateway, sampling, body_sha256, build_id
        responses-<pid>.jsonl     stream as received + prompt_tokens, input_replay, response_id,
                                  usage, max_output_tokens (as submitted), output_call_ids, build_id
        raw-<pid>.jsonl           native model text per output + token_ids, finish_reason, flags, build_id
        normalize-<pid>.jsonl     per-call_id record of what the parser / Responses layer repaired
      _writer/<generation>.json   writer accounting + build_id + provenance

The fields, in short (the module docstrings of `tensorrt_llm/serve/request_trace.py`
and `tensorrt_llm/executor/raw_output_hook.py` are the reference):

- `thread` is the harness's conversation identity (`client_metadata.thread_id`
  and friends). `session` is the *router* key -- for Codex, the parent session
  -- and a sub-agent shares it, so group conversations on `thread.thread_id`.
- `prompt_tokens` is `{count, sha256, encoding: int32le}` of the token ids the
  engine was given: `sha256(numpy.asarray(ids, "<i4").tobytes())`. It is what
  proves which context the model saw; an export re-renders the body and must
  reproduce it. The ids themselves are stored too when
  `TRTLLM_REQUEST_TRACE_PROMPT_TOKEN_IDS=1` is in `server.env`.
- `input_replay` is what the server changed when it rendered the body
  (namespaces restored, replayed reasoning decoded, items dropped, orphan tool
  results noted). Counts and call ids, not copies.
- `max_output_tokens` on the response line is the cap the generation was
  submitted with. `TLLM_RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS=<n>` in
  `server.env` caps requests that set none (Codex never does); the request
  line's `sampling` still shows what the client sent. The cap is applied
  where the request is rendered -- the workers, not the proxy -- and
  `server.env` reaches every process, so set it in the deployment YAML rather
  than on one host. The engine may still lower a cap to fit `max_seq_len`
  minus the prompt; `incomplete_reason` and the raw record's `finish_reason`
  say whether a cap bound. For data generation leave it unset: a cap of our
  own truncates real outputs (GLM-5.2 completes 40-70k-token calls on kernel
  tasks), and a loop is better marked at export time from the raw record than
  cut at serving time.
- `build_id` is `py_manifest_sha256[:12]` from the attempt's provenance, on
  every line, so records stay attributable across restarts.
- raw `token_ids` and `finish_reason` are the engine's own; `flags` are cheap
  scans of the text (`<|...|>` literals with counts, a repetition measure,
  dots-only reasoning) meant as signals for an export, not verdicts.

Raw output follows the traces only when `server.capture` is on; without it the
hook keeps writing under `<attempt>/raw_output` as before.

## Sizing

A worker must fit on one node. The reference 6P1D shape is TP4 per worker,
which needs 8 nodes of 4 GPUs; the 1P1D shape needs 2. A checkpoint too large
for one node's GPUs forces a worker to span nodes, and that is worth avoiding
for reasons beyond throughput -- a single MPI world spread across nodes was
what made GLM-5.3 (1.4 TB, TP8, two nodes per worker) fail to start about one
time in five here, always with a null-pointer segfault on the second node.
GLM-5.2-NVFP4 is 433 GiB and fits TP4 on one node, and has not done it once.
