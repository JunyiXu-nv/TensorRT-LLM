# Large-scale serving

Several disaggregated TensorRT-LLM deployments behind one gateway, run on SLURM
as a long-lived service rather than as a benchmark.

This directory holds everything needed to bring that up. It exists because the
hard part of serving at this scale is not any one server — it is that the
address must stay put for days, instances get preempted and come back, and a
conversation has to keep landing on the machine that holds its KV cache.

```
             one stable URL
                   │
            ┌──────┴───────┐
            │   gateway    │   conversation routing, health, draining
            └──────┬───────┘   gateway.py, one CPU node, chained
      ┌────────┬───┴────┬────────┐
      │        │        │        │
   inst a   inst b   inst c   inst d      fleetctl + serve.sh
      │                                   one SLURM job each
  ┌───┴────────────────┐
  │ proxy → 6×ctx 1×gen│                  disaggregated: prefill and
  └────────────────────┘                  decode are separate workers
```

## Bring-up

```bash
# 0. Say who may call the gateway, and what the fleet looks like.
$EDITOR deployments/gateway_users.txt   # ships admitting nobody; add usernames
$EDITOR deployments/fleet.yaml            # account, partition, image, model, instances

# 1. The gateway. Submit once; it re-submits itself (see "The chain" below).
#    GW_ROOT is the only required setting; everything else defaults under it.
#    sbatch exports the environment into the job, which is also how each
#    setting reaches the successors.
GW_ROOT=/scratch/.../users/$USER GW_ACCOUNT=<account> \
    sbatch -o /scratch/.../users/$USER/gw/log/gateway.%j.out gateway.sbatch
#    -> writes GATEWAY_URL=http://<node>:8333 into its log

# 2. The instances.
./fleetctl --config deployments/fleet.yaml up

# 3. Watch them arrive. Registration is by file drop, so this is the truth.
./fleetctl --config deployments/fleet.yaml status
curl -s http://<node>:8333/_gateway/fleet -H "x-api-key: <username>" | python3 -m json.tool
```

Day-to-day:

```bash
./fleetctl ... roll a     # replace one instance without dropping traffic
./fleetctl ... down a     # stop one
./fleetctl ... status     # what is running against what is configured
python3 test_router.py    # 67 routing tests, no cluster needed
```

## What is here

| file | runs where | role |
|---|---|---|
| `gateway.py` | one CPU node | the router: conversation stickiness, health, draining, the public API |
| `gateway.sbatch` | login node | submits the gateway as a self-renewing chain |
| `test_router.py` | anywhere | routing tests, pure stdlib, no GPUs |
| `serve.sh` | login node | brings up **one** disaggregated deployment from one YAML |
| `fleetctl` | login node | brings up **N** of them and keeps them at the configured count |
| `fleetctl-remote` | the gateway's host | runs `fleetctl` over ssh, for a gateway that is not on the cluster |
| `slurm-remote` | the gateway's host | runs `squeue` over ssh, so recovery can tell a requeued job from a gone one |
| `FLEET.md` | — | why the fleet is shaped the way it is |
| `NEW_CLUSTER.md` | — | checklist for standing this up somewhere it has not run |
| `deployments/` | — | the inputs you edit |
| `generated/` | — | per-instance YAMLs `fleetctl` derives; not checked in |

`fleetctl` and `serve.sh` must stay in the same directory (`fleetctl` finds
`serve.sh` next to itself), and `gateway.py` must stay one level above
`deployments/` (`serve.sh` resolves it that way). The layout is load-bearing.

## The chain

A SLURM job cannot outlive the partition's wall clock — seven days here. But
the whole point of the gateway is that its URL does not move, and a URL here is
a node. So `gateway.sbatch` submits its own successor:

```
--dependency=afterany:$SLURM_JOB_ID    start the moment this one ends, for any reason
--nodelist=$NODE                       same node, so the address is unchanged
--router-state .../router_state.json   successor inherits the pin table
```

Three details are deliberate:

- **The successor is submitted at the start, not the end.** A job that is
  killed, preempted, or hits its wall clock never runs its own cleanup — and
  those are exactly the cases where the chain must not break.
- **It refuses to extend a thrashing chain.** More than five starts in thirty
  minutes and it stops rather than submitting a job that will also die
  immediately. Losing the chain costs us; a submission loop costs the cluster.
- **It waits for `:8333` to be released** before binding. The predecessor may
  still be shutting down, and this job already holds the node, so waiting is
  the only option that does not break the chain it was started to continue.

**Changing this file does not change queued jobs.** SLURM copies the script at
submit time, so an already-queued successor still runs the old text. After
editing, submit a replacement *before* cancelling the old one, and check
`scontrol write batch_script <id> -` to see what will actually run.

This is most of why the script carries no paths of its own: a `GW_*` variable
is read when the job runs, so it can be changed without the copy-at-submit
problem, while an edit to the script text cannot reach a successor that is
already queued. `GW_ROOT` is required and everything else derives from it --
`GW_REPO`, `GW_STATE_DIR`, `GW_RUN_DIR`, `GW_LOG_DIR`, `GW_SELF`, `GW_PORT`,
`GW_FLEET_CONFIG`, `GW_EXTRA_ARGS`; the header of `gateway.sbatch` lists them
with their defaults. Scheduler options come from `SBATCH_ACCOUNT` and friends,
which SLURM resolves as command line > environment > `#SBATCH` directive.

## Conversation routing

The gateway pins a conversation to a backend and keeps it there, because an
agent turn replays its whole history and the KV cache for that prefix lives on
one machine. It identifies a conversation by trying, in order:

```
header:x-conversation-id / x-session-id / session-id / thread-id
body:prompt_cache_key / previous_response_id / conversation_id
body:client_metadata.session_id / .thread_id / metadata.session_id / .conversation_id
prefix          ← sha256 of instructions + the first two input entries
```

Configurable per deployment, so a client that appears later does not need a
code change. The last entry is a fallback for clients that send no identity at
all: only the *first two* entries are hashed, because anything further folds in
text that grows every turn and the key would then change under the session it
is supposed to pin.

Measured on this deployment: 702 conversations, 13839 requests, zero that
crossed backends. Prefer sending a real id — `prefix` works only as long as the
opening of the conversation is stable, and a client that truncates old history
to fit a context window silently re-keys and loses its cache.

## Adding and removing instances at runtime

Registration is a file in `--fleet-dir`, discovered every five seconds. Nothing
else couples a backend to the gateway, so a serving job joins by writing one
and leaves by deleting it — which is how instances survive preemption: they
come back on different nodes, under the same job ids, and the gateway follows.

Two endpoints write that file for you, for backends the fleet launcher does not
own — a hand-started server, a differently shaped instance, a borrowed one:

```bash
curl -X POST http://$GW/_gateway/backend -H "x-api-key: $USER" \
     -d '{"job_id": "spare-1", "url": "http://node:8500"}'
# 503 unless it answers /health. Add "probe": false to register one still loading.

curl -X POST http://$GW/_gateway/backend/remove -H "x-api-key: $USER" \
     -d '{"job_id": "spare-1"}'
# 202 while conversations are still pinned there: it stops taking new ones and
# says how many remain. Add "force": true to remove it now, "drain": false to
# skip the wait entirely.
```

They write the same file the launcher does rather than keeping a second list in
memory. Two sources of truth for "what is in the fleet" would disagree within
one sweep, and the sweep would win.

The gateway **does not check what a backend serves.** No model name, no
parallelism, no prefill/decode ratio — any address that speaks the API can
join, which is what lets differently configured instances share a pool. It also
means registering an instance running a different model silently routes
conversations to the wrong model. Keeping a pool homogeneous is the caller's
job.

## Custom routing policies

Built-in: `least_conversations` (default), `least_inflight`, `round_robin`,
`longest_lived`. Switch at runtime; already-pinned conversations stay put, so
switching back leaves nothing behind:

```bash
curl -X POST http://$GW/_gateway/route -H "x-api-key: $USER" \
     -d '{"policy": "least_inflight"}'
```

For anything else, `--policy-dir` is a directory of `.py` files. `mypolicy.py`
supplies the policy `mypolicy`:

```python
def select(accepting):
    """accepting: {job_id: (conversations, inflight, remaining_seconds)}"""
    return min(accepting, key=lambda j: accepting[j][0] + 0.3 * accepting[j][1])
```

Drop it in and it is loadable within five seconds — no restart. Edit it and the
new version replaces the old one on the next sweep.

**A directory rather than an HTTP endpoint, deliberately.** A policy is code,
and this gateway authenticates with a username its own users file describes as
guessable. Accepting code over HTTP would make "knows a colleague's name"
enough to run anything on the gateway node. Writing to the shared filesystem is
a much higher bar — and the same bar registering a backend already sets.

A custom policy runs on the request path, so it is contained rather than
trusted. Raising, or naming a backend that is not on offer, counts against it;
after three strikes it is dropped and placement falls back to
`least_conversations`. Editing the file clears the count — the way to re-enable
a policy is to fix it, not to restart the gateway.

## Access

`gateway_users.txt` is an allowlist, one username per line, reread on change.
**The username is the key** — callers set `x-api-key` to their own name. That
makes the access log an attribution trail, not proof of identity: a name is
guessable. Fine among colleagues on an internal network; do not put this on a
public one, and do not treat the log as an audit record.

**A caller that cannot send a key needs the `anonymous` entry.** Kernel Factory
is one: `kf llm-endpoint` stores and sends no credential, by design, so a BYO
LLM endpoint has to answer an unauthenticated request. Without that line every
agent call is a 401, and it surfaces as an `authentication_error` that says
nothing about the fleet behind it.

An unedited copy of the example file admits nobody, on purpose. A live
placeholder like `your-username` would be the first thing an unwelcome caller
tries.

## Preemption

Instances are preempted; plan for it rather than against it. On this cluster
`PreemptType=preempt/qos` with `PreemptMode=REQUEUE`, and the `normal` QoS
preempts *within itself* by priority, so a higher-priority job in the same QoS
can take your nodes. `PreemptExemptTime` gives a grace period — four hours here
— after which a running instance is fair game.

Requeue is automatic. In four observed events SLURM brought instances back on
its own in 21 and 36 minutes; the worst was three instances taken by a single
scheduling decision within four seconds. Restarting by hand during the grace
period wastes the exemption you have already earned, so the useful reflexes are:
let SLURM requeue, keep more instances than you need, and make the gateway drop
a dead backend quickly rather than trying to prevent the death.

Requeue is not guaranteed, though, and an instance the scheduler has forgotten
never comes back on its own -- the fleet just gets smaller until somebody
notices. Setting `GW_FLEET_CONFIG` to a `fleet.yaml` lets the gateway close
that gap: a backend that disappears is remembered, and once it has been absent
for `--recover-grace` the gateway asks `squeue` about the job. A job that is
still queued -- which is what a requeue looks like, since REQUEUE keeps the job
id -- is left entirely alone. Only a job the scheduler has no record of at all
triggers `fleetctl up`, which reconciles against `squeue` itself and so cannot
start a second copy of anything. Off unless the variable is set: it is the one
setting that lets the gateway submit jobs.

## Analysing a run

`analysis/` turns an attempt directory into per-request and per-iteration
tables, an HTML report, and a small JSON rollup:

```bash
python3 analysis/report.py <attempt_dir> --hour 2026-09-10T06 --json
python3 analysis/rollup_store.py --trace-root .../deploy/trace --store .../wb/rollups
```

Two things about where and how it runs, both learned the hard way.

**Not on the login node.** It reads gigabytes and the login node caps a process
at 8 GB of address space for everyone, so the report dies with a MemoryError
partway through the engine logs. The data-copier node
(`oci-jhb-slurm-1-dc-01`) sees the same filesystem, sets no cap, and is not
shared with a hundred interactive sessions.

**One hour costs one hour.** Engine logs are appended to for the life of the
instance -- 254 MB at two hours old, 12 GB at eighteen, and around 150 GB for
one that lasts the seven days the partition allows -- and the analysis of any
single hour was reading all of them. `analysis/.engine_index/`, written beside
the logs, holds a checkpoint every 64 MB plus the lifetime KV capacity peaks
that a window cannot see, so a later run seeks to just before its window and
reads only what it needs. `--no-index` forces the full pass, which is how the
index is checked against it.

## Monitoring

A live dashboard for exactly this deployment — fleet health, traffic by route
and instance, routing state, 5xx by cause, instance timelines — is kept
separately, since it wants a web host rather than a login node:

    ssh://git@gitlab-master.nvidia.com:12051/junyix/trtllm-serving-dashboard.git

Its collectors run on the login node and read the same fleet directory and
gateway log this deployment writes.
