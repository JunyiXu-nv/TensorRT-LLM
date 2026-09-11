#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --qos=cpu-long
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=trtllm-gateway
#
# The gateway that an agent client is pointed at, run as a chain rather than as
# a single job.
#
# The client holds one URL, and that URL is a scheduler-allocated node. Seven
# days is the longest a job may live here, so the address this deployment
# promises to be stable is guaranteed to move roughly once a week unless
# something puts it back. Two things keep it in place:
#
#   --nodelist pins the successor to the same node, so the IP is unchanged.
#   --dependency=afterany makes it start the moment the predecessor ends,
#     rather than sitting behind whatever else wants a CPU node.
#
# The successor is submitted from inside this job, so the chain continues
# without anybody remembering to extend it. It is submitted at the START, not
# at the end: a job that is killed, preempted or hits its wall clock never runs
# its own cleanup, and those are exactly the cases where the chain must not
# break.
#
# The pin table is shared through --router-state, so the successor inherits
# every conversation's placement and the handover costs no prefix cache. The
# gap is whatever the scheduler takes to start the next job, during which the
# gateway answers nothing and clients retry.
#
# ---------------------------------------------------------------------------
# Configuration
#
# Nothing site-specific is written into this file. Everything below reads an
# environment variable, and sbatch exports the submitting environment into the
# job by default -- which is also what carries a setting down the chain, since
# each successor is submitted from inside a job that already has it. Set them
# once at the first submit and the whole chain inherits them:
#
#   GW_ROOT=/scratch/.../users/me \
#   GW_ACCOUNT=my_account \
#     sbatch -o /scratch/.../users/me/gw/log/gateway.%j.out gateway.sbatch
#
#   GW_ROOT         required; everything else defaults underneath it
#   GW_REPO         this checkout            (default $GW_ROOT/TensorRT-LLM)
#   GW_STATE_DIR    fleet dir, users, pins   (default $GW_ROOT/wb)
#   GW_RUN_DIR      node file, chain starts  (default $GW_ROOT/gw)
#   GW_LOG_DIR      successor --output       (default $GW_RUN_DIR/log)
#   GW_SELF         script the chain resubmits   (default this file in GW_REPO)
#   GW_PORT         listen port              (default 8333)
#   GW_FLEET_CONFIG fleet.yaml for preemption recovery (default: off)
#   GW_MAX_STARTS   chain-thrash threshold   (default 5 in 30 minutes)
#   GW_EXTRA_ARGS   appended to the gateway command line
#   GW_NO_CHAIN     set to anything to run once without a successor
#
# The scheduler options are separate, because #SBATCH directives are read
# before this shell runs and cannot see a variable. SLURM resolves them as
# command line > environment > directive, and the SBATCH_* environment
# variables propagate down the chain the same way the GW_* ones do:
#
#   GW_ACCOUNT -> SBATCH_ACCOUNT, or export SBATCH_PARTITION / SBATCH_QOS /
#   SBATCH_TIMELIMIT / SBATCH_JOB_NAME directly. There is no SBATCH_OUTPUT, so
#   the first job's log path is an `sbatch -o` argument; successors get
#   GW_LOG_DIR, which is why that one is a variable rather than a directive.
set -uo pipefail

if [ -z "${GW_ROOT:-}" ]; then
    echo "GW_ROOT is not set. It is the one path everything else derives from;" >&2
    echo "see the configuration block at the top of this file." >&2
    exit 2
fi

REPO="${GW_REPO:-$GW_ROOT/TensorRT-LLM}"
STATE_DIR="${GW_STATE_DIR:-$GW_ROOT/wb}"
RUN_DIR="${GW_RUN_DIR:-$GW_ROOT/gw}"
LOG_DIR="${GW_LOG_DIR:-$RUN_DIR/log}"
PORT="${GW_PORT:-8333}"
MAX_STARTS="${GW_MAX_STARTS:-5}"
HERE="$REPO/examples/serve/large_scale_serving"
# Not $0: SLURM runs a spool copy of this script, so $0 is a path that exists
# only for the life of the job and cannot be resubmitted.
SELF="${GW_SELF:-$HERE/gateway.sbatch}"

# Exported so the sbatch that queues the successor inherits it, rather than
# each link in the chain having to be told again.
[ -n "${GW_ACCOUNT:-}" ] && export SBATCH_ACCOUNT="$GW_ACCOUNT"

NODE=$(hostname -s)
mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "GATEWAY_NODE=$NODE"
echo "GATEWAY_URL=http://$NODE:$PORT"
echo "$NODE" > "$RUN_DIR/node"

for path in "$SELF" "$HERE/gateway.py"; do
    if [ ! -f "$path" ]; then
        echo "WARNING: $path does not exist; check GW_REPO/GW_SELF" >&2
    fi
done

# A chain that submits its successor at the start will, if the gateway dies
# immediately, submit a successor that also dies immediately -- a submission
# loop against a shared scheduler. So record every start and refuse to extend
# the chain when it is clearly thrashing. Refusing costs the chain; continuing
# costs everyone else on the cluster, which is worse.
STARTS="$RUN_DIR/chain_starts"
date +%s >> "$STARTS"
RECENT=$(awk -v cutoff="$(( $(date +%s) - 1800 ))" '$1 > cutoff' "$STARTS" 2>/dev/null | wc -l)
tail -200 "$STARTS" > "$STARTS.tmp" 2>/dev/null && mv -f "$STARTS.tmp" "$STARTS"

if [ "$RECENT" -gt "$MAX_STARTS" ]; then
    echo "REFUSING to queue a successor: $RECENT chain starts in the last 30 min."
    echo "The gateway is failing fast and the chain would spin. Fix it, then"
    echo "restart the chain by hand with: sbatch --nodelist=$NODE $SELF"
elif [ -z "${GW_NO_CHAIN:-}" ]; then
    # --output is passed rather than left to the directive: there is no
    # SBATCH_OUTPUT, so this is the only way a successor's log follows the
    # configured directory instead of wherever the first submit happened to
    # put it.
    NEXT=$(sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID" \
        --nodelist="$NODE" \
        --output="$LOG_DIR/gateway.%j.out" \
        "$SELF" 2>&1) && echo "successor queued: $NEXT on $NODE" \
        || echo "WARNING: could not queue a successor: $NEXT"
fi

# The predecessor may still be releasing the port. Wait rather than exit: this
# job already holds the node, so there is nothing to fall back to and failing
# here would break the chain it was started to continue.
for attempt in $(seq 1 60); do
    if ! ss -lnt 2>/dev/null | grep -q ":$PORT "; then
        break
    fi
    [ "$attempt" = 1 ] && echo "waiting for :$PORT to be released by the predecessor"
    sleep 5
done

# Preemption recovery is off unless a fleet.yaml is named: it is the only
# option here that lets the gateway submit jobs, so it is opt-in rather than
# something a bare GW_ROOT switches on.
RECOVERY=()
if [ -n "${GW_FLEET_CONFIG:-}" ]; then
    RECOVERY=(--fleet-config "$GW_FLEET_CONFIG")
    echo "preemption recovery enabled against $GW_FLEET_CONFIG"
fi

exec python3 "$HERE/gateway.py" \
    --fleet-dir "$STATE_DIR/fleet" \
    --users "$STATE_DIR/users.txt" \
    --host 0.0.0.0 --port "$PORT" \
    --router-state "$STATE_DIR/router_state.json" \
    "${RECOVERY[@]}" \
    --no-relay --log-level INFO \
    ${GW_EXTRA_ARGS:-}
