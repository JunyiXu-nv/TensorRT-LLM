#!/usr/bin/env bash
# Build any hourly rollups the store is missing. Meant for cron, on a node that
# can see the trace filesystem and is not the login node -- the login node caps
# a process at 8 GB of address space and the analysis needs more than that
# briefly, and it is shared with a hundred interactive sessions besides.
#
#   20 * * * * /path/to/analysis/hourly-rollups.sh
#
# Safe to run at any cadence and from more than one place at once:
# rollup_store.py takes a lock and skips hours it already has. LIMIT bounds one
# invocation, so a backlog is worked through over several rather than in one
# long occupation of a core.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=${KF_BASE:-/scratch/fsw/portfolios/coreai/projects/coreai_kf_dev/users/junyix}
TRACE=${KF_TRACE_ROOT:-$BASE/deploy/trace}
STORE=${KF_ROLLUP_STORE:-$BASE/wb/rollups}
LOG=${KF_ROLLUP_LOG:-$STORE/hourly.log}
LIMIT=${KF_ROLLUP_LIMIT:-4}
PYTHON=${KF_PYTHON:-python3}

mkdir -p "$STORE"
# Keep the log to the last few thousand lines rather than letting a year of
# hourly runs accumulate next to the files this is supposed to be producing.
if [ -f "$LOG" ] && [ "$(wc -l < "$LOG")" -gt 5000 ]; then
  tail -2000 "$LOG" > "$LOG.trim" && mv -f "$LOG.trim" "$LOG"
fi
{
  echo "--- $(date -u +%FT%TZ) on $(hostname -s) ---"
  "$PYTHON" "$HERE/rollup_store.py" --trace-root "$TRACE" --store "$STORE" --limit "$LIMIT"
  rc=$?
  # 75 is "someone else holds the lock", which for an hourly timer is normal
  # and not worth reporting as a failure. Anything else that is not success is.
  [ "$rc" -eq 0 ] || [ "$rc" -eq 75 ] || echo "rollup_store exited $rc"
} >> "$LOG" 2>&1
