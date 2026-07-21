#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Validate the merged HangDetector hard-kill + cross-rank propagation with a
# REAL TP=2 model: wedge one rank mid-decode and confirm ALL ranks are
# hard-killed (GPUs freed) within ~hang-timeout instead of hanging to the
# wall clock.
#
# Requires: >=2 GPUs, a built+editable TRT-LLM on the scratch branch
# `dev-junyix-test-single-rank-hang` (pure-Python fault injection, no rebuild),
# and LLM_MODELS_ROOT set.
#
# Usage:  LLM_MODELS_ROOT=/path bash run_single_rank_hang.sh
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
: "${LLM_MODELS_ROOT:?set LLM_MODELS_ROOT to your models dir}"

export REPRO_MODEL="${REPRO_MODEL:-$LLM_MODELS_ROOT/llama-3.1-model/Llama-3.1-8B-Instruct}"
export REPRO_TP="${REPRO_TP:-2}"
export TLLM_DEBUG_WEDGE_RANK="${TLLM_DEBUG_WEDGE_RANK:-1}"
export TLLM_DEBUG_WEDGE_AFTER_STEPS="${TLLM_DEBUG_WEDGE_AFTER_STEPS:-5}"
export TLLM_DEBUG_HANG_TIMEOUT="${TLLM_DEBUG_HANG_TIMEOUT:-30}"
OUTER="${OUTER_TIMEOUT:-360}"     # hard wall-clock cap for the whole repro
POLL="${POLL_INTERVAL:-3}"

gpu_used () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}'; }

echo "=== single-rank hang repro ==="
echo "model=$REPRO_MODEL tp=$REPRO_TP wedge_rank=$TLLM_DEBUG_WEDGE_RANK after_steps=$TLLM_DEBUG_WEDGE_AFTER_STEPS hang_timeout=${TLLM_DEBUG_HANG_TIMEOUT}s outer=${OUTER}s"
base=$(gpu_used); echo "GPU mem baseline (all GPUs summed): ${base} MiB"

t0=$(date +%s)
python "$HERE/repro_single_rank_hang.py" > /tmp/repro_driver.log 2>&1 &
drv=$!
echo "[runner] driver pid=$drv ; polling GPU mem + liveness (log: /tmp/repro_driver.log)"

peak=$base; kill_t=""; driver_exit_t=""
while :; do
  now=$(( $(date +%s) - t0 ))
  used=$(gpu_used)
  alive="dead"; kill -0 "$drv" 2>/dev/null && alive="alive"
  [ "$used" -gt "$peak" ] && peak=$used
  printf "  t=%3ss  gpu_used=%6s MiB  driver=%s\n" "$now" "$used" "$alive"
  # ranks freed GPUs = mem fell back near baseline AFTER having risen >2GB
  if [ -z "$kill_t" ] && [ "$peak" -gt $((base + 2000)) ] && [ "$used" -lt $((base + 1000)) ]; then
    kill_t=$now; echo "  >>> GPUs returned to ~baseline at t=${now}s  (all ranks hard-killed)"
  fi
  if [ "$alive" = "dead" ] && [ -z "$driver_exit_t" ]; then
    wait "$drv"; drc=$?; driver_exit_t=$now
    echo "  >>> driver exited rc=$drc at t=${now}s"
  fi
  # stop once both settled, or at outer cap
  { [ -n "$kill_t" ] && [ -n "$driver_exit_t" ]; } && break
  if [ "$now" -ge "$OUTER" ]; then echo "  >>> OUTER cap ${OUTER}s reached"; kill -9 "$drv" 2>/dev/null; break; fi
  sleep "$POLL"
done

echo; echo "=== verdict ==="
echo "peak GPU mem: ${peak} MiB (baseline ${base})"
echo "stray workers now: $(pgrep -af 'repro_single_rank_hang|trtllm' | grep -v run_single_rank_hang | wc -l)"
if [ -n "$kill_t" ]; then
  echo "PASS (ST-1): all ranks hard-killed / GPUs freed at t=${kill_t}s (<< ${OUTER}s wall-clock)."
else
  echo "FAIL (ST-1): GPUs NOT freed within ${OUTER}s -- ranks appear to be zombie-holding (hang not hard-killed)."
fi
if [ -n "$driver_exit_t" ]; then
  echo "driver (proxy) also exited at t=${driver_exit_t}s."
else
  echo "NOTE (ST-2 territory): driver/proxy did NOT unblock on its own (generate() stayed blocked) -- expected on main without the sticky-EngineDeadError PR."
fi
echo "--- driver log tail ---"; tail -120 /tmp/repro_driver.log
