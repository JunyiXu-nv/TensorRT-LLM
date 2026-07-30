#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-node variant: run within an existing salloc allocation.
# Usage:
#   REPO_DIR=... MODEL_PATH=... CONTAINER_IMAGE=... PORT=... \
#     bash start_server_1node.sh <node> <attempt_dir>
#
# Or via srun inside an existing allocation:
#   srun --jobid=<JOBID> bash start_server_1node.sh <node> <attempt_dir>

set -euo pipefail

NODE0="${1:?node is required}"
ATTEMPT_DIR="${2:?attempt directory is required}"
: "${SLURM_JOB_ID:?must run inside a SLURM allocation}"
: "${REPO_DIR:?REPO_DIR is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
: "${PORT:?PORT is required}"

CONFIG_FILE="${ATTEMPT_DIR}/agg_config.yaml"
AUDIT_LOG="${ATTEMPT_DIR}/anthropic_audit.jsonl"
BENCH_CAPTURE_DIR="${ATTEMPT_DIR}/anthropic_message_capture"
CONTAINER_NAME="deepseek-v4-pro-${SLURM_JOB_ID}"
USER_ROOT="/home/scratch.serli_gpu"
MOUNTS="${USER_ROOT}:${USER_ROOT},/home/scratch.trt_llm_data:/home/scratch.trt_llm_data,/raid:/raid"

cleanup_workers() {
    ssh "${NODE0}" \
        "pkill -TERM -f '[t]ensorrt_llm.llmapi.mgmn_' || true; pkill -TERM -f '[t]rtllm-llmapi-launch.*DeepSeek-V4-Pro' || true; pkill -TERM -f '[t]rtllm-serve.*DeepSeek-V4-Pro' || true" \
        || true
    sleep 5
    ssh "${NODE0}" \
        "pkill -KILL -f '[t]ensorrt_llm.llmapi.mgmn_' || true; pkill -KILL -f '[t]rtllm-llmapi-launch.*DeepSeek-V4-Pro' || true; pkill -KILL -f '[t]rtllm-serve.*DeepSeek-V4-Pro' || true" \
        || true
}

trap cleanup_workers EXIT
trap 'exit 0' INT TERM
cleanup_workers

echo "Installing branch checkout on ${NODE0}: $(git -C "${REPO_DIR}" branch --show-current)"
srun -l \
    --jobid "${SLURM_JOB_ID}" \
    --nodelist "${NODE0}" \
    --nodes 1 \
    --ntasks 1 \
    --ntasks-per-node 1 \
    --container-image "${CONTAINER_IMAGE}" \
    --container-name "${CONTAINER_NAME}" \
    --container-mounts "${MOUNTS}" \
    --no-container-mount-home \
    --mpi=pmix \
    --overlap \
    bash -lc "cd '${REPO_DIR}' && python3 -m pip install -e ." \
    |& tee "${ATTEMPT_DIR}/install.log"

echo "Starting DeepSeek-V4-Pro TP8/EP8 server (single node) at http://${NODE0}:${PORT}"
srun -l \
    --jobid "${SLURM_JOB_ID}" \
    --nodelist "${NODE0}" \
    --nodes 1 \
    --ntasks 8 \
    --ntasks-per-node 8 \
    --export="ALL,TLLM_LOG_LEVEL=INFO,TRTLLM_SERVER_DISABLE_GC=1,TRTLLM_WORKER_DISABLE_GC=1,TRTLLM_ENABLE_PDL=1,TRTLLM_ANTHROPIC_AUDIT_LOG=${AUDIT_LOG},TRTLLM_ANTHROPIC_LCP_TRACKING=1,TRTLLM_ANTHROPIC_BENCH_CAPTURE_DIR=${BENCH_CAPTURE_DIR},ENROOT_ALLOW_DEV=yes,NCCL_GRAPH_MIXING_SUPPORT=0,MIMALLOC_PURGE_DELAY=0" \
    --container-image "${CONTAINER_IMAGE}" \
    --container-name "${CONTAINER_NAME}" \
    --container-mounts "${MOUNTS}" \
    --no-container-mount-home \
    --mpi=pmix \
    --overlap \
    bash -lc '
        export CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"
        unset UCX_TLS
        exec trtllm-llmapi-launch numactl -m 0,1 \
            trtllm-serve "$1" \
            --host "$(hostname)" \
            --port "$2" \
            --config "$3" \
            --tool_parser "${TOOL_PARSER:-deepseek_v4}"
    ' _ "${MODEL_PATH}" "${PORT}" "${CONFIG_FILE}" \
    |& tee "${ATTEMPT_DIR}/server.log"
