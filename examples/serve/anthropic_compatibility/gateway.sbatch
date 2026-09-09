#!/bin/bash
#SBATCH --account=coreai_kf_dev
#SBATCH --partition=cpu
#SBATCH --qos=cpu-long
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --job-name=kf-gateway-junyix
#SBATCH -o /scratch/fsw/portfolios/coreai/projects/coreai_kf_dev/users/junyix/gw/log/gateway.%j.out
#
# The gateway that Kernel Factory is pointed at, run as a chain rather than as
# a single job.
#
# KF holds one URL, and that URL is a scheduler-allocated node. Seven days is
# the longest a job may live here, so the address this deployment promises to
# be stable is guaranteed to move roughly once a week unless something puts it
# back. Two things keep it in place:
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
set -uo pipefail

T=/scratch/fsw/portfolios/coreai/projects/coreai_kf_dev/users/junyix
NODE=$(hostname -s)
SELF="$T/wb/gw_longrun.sbatch"

echo "GATEWAY_NODE=$NODE"
echo "GATEWAY_URL=http://$NODE:8333"
echo "$NODE" > "$T/gw/node"

# A chain that submits its successor at the start will, if the gateway dies
# immediately, submit a successor that also dies immediately -- a submission
# loop against a shared scheduler. So record every start and refuse to extend
# the chain when it is clearly thrashing. Refusing costs the chain; continuing
# costs everyone else on the cluster, which is worse.
STARTS="$T/gw/chain_starts"
date +%s >> "$STARTS"
RECENT=$(awk -v cutoff="$(( $(date +%s) - 1800 ))" '$1 > cutoff' "$STARTS" 2>/dev/null | wc -l)
tail -200 "$STARTS" > "$STARTS.tmp" 2>/dev/null && mv -f "$STARTS.tmp" "$STARTS"

if [ "$RECENT" -gt 5 ]; then
    echo "REFUSING to queue a successor: $RECENT chain starts in the last 30 min."
    echo "The gateway is failing fast and the chain would spin. Fix it, then"
    echo "restart the chain by hand with: sbatch --nodelist=$NODE $SELF"
elif [ -z "${GW_NO_CHAIN:-}" ]; then
    NEXT=$(sbatch --parsable \
        --dependency=afterany:"$SLURM_JOB_ID" \
        --nodelist="$NODE" \
        "$SELF" 2>&1) && echo "successor queued: $NEXT on $NODE" \
        || echo "WARNING: could not queue a successor: $NEXT"
fi

# The predecessor may still be releasing the port. Wait rather than exit: this
# job already holds the node, so there is nothing to fall back to and failing
# here would break the chain it was started to continue.
for attempt in $(seq 1 60); do
    if ! ss -lnt 2>/dev/null | grep -q ':8333 '; then
        break
    fi
    [ "$attempt" = 1 ] && echo "waiting for :8333 to be released by the predecessor"
    sleep 5
done

exec python3 "$T/TensorRT-LLM/examples/serve/anthropic_compatibility/gateway.py" \
    --fleet-dir "$T/wb/fleet" \
    --users "$T/wb/users.txt" \
    --host 0.0.0.0 --port 8333 \
    --router-state "$T/wb/router_state.json" \
    --no-relay --log-level INFO
