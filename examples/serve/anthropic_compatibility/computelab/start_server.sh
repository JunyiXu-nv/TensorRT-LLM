#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

NODE0="${1:?first allocated node is required}"
NODE1="${2:?second allocated node is required}"
ATTEMPT_DIR="${3:?attempt directory is required}"
: "${SLURM_JOB_ID:?start_server.sh must run inside the persistent allocation}"
: "${REPO_DIR:?REPO_DIR is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
: "${PORT:?PORT is required}"

CONFIG_FILE="${ATTEMPT_DIR}/agg_config.yaml"
AUDIT_LOG="${ATTEMPT_DIR}/anthropic_audit.jsonl"
BENCH_CAPTURE_DIR="${ATTEMPT_DIR}/anthropic_message_capture"
CONTAINER_NAME="deepseek-v4-pro-${SLURM_JOB_ID}"
USER_ROOT="/home/scratch.serli_gpu"
MOUNTS="${USER_ROOT}:${USER_ROOT},/home/scratch.trt_llm_data:/home/scratch.trt_llm_data"

cleanup_workers() {
    local node
    for node in "${NODE0}" "${NODE1}"; do
        ssh "${node}" \
            "pkill -TERM -f '[t]ensorrt_llm.llmapi.mgmn_' || true; pkill -TERM -f '[t]rtllm-llmapi-launch.*DeepSeek-V4-Pro' || true; pkill -TERM -f '[t]rtllm-serve.*DeepSeek-V4-Pro' || true" \
            || true
    done
    sleep 5
    for node in "${NODE0}" "${NODE1}"; do
        ssh "${node}" \
            "pkill -KILL -f '[t]ensorrt_llm.llmapi.mgmn_' || true; pkill -KILL -f '[t]rtllm-llmapi-launch.*DeepSeek-V4-Pro' || true; pkill -KILL -f '[t]rtllm-serve.*DeepSeek-V4-Pro' || true" \
            || true
    done
}

trap cleanup_workers EXIT
trap 'exit 0' INT TERM
cleanup_workers

echo "Installing branch checkout on both nodes: $(git -C "${REPO_DIR}" branch --show-current)"
srun -l \
    --nodelist "${NODE0},${NODE1}" \
    --nodes 2 \
    --ntasks 2 \
    --ntasks-per-node 1 \
    --container-image "${CONTAINER_IMAGE}" \
    --container-name "${CONTAINER_NAME}" \
    --container-mounts "${MOUNTS}" \
    --no-container-mount-home \
    --mpi=pmix \
    --overlap \
    bash -lc "cd '${REPO_DIR}' && python3 -m pip install -e ." \
    |& tee "${ATTEMPT_DIR}/install.log"

echo "Starting DeepSeek-V4-Pro aggregated TP8/EP8 server at http://${NODE0}:${PORT}"
echo "WARNING: sensitive Anthropic request capture is enabled at ${BENCH_CAPTURE_DIR}"
srun -l \
    --nodelist "${NODE0},${NODE1}" \
    --nodes 2 \
    --ntasks 8 \
    --ntasks-per-node 4 \
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
            --tool_parser deepseek_v4
    ' _ "${MODEL_PATH}" "${PORT}" "${CONFIG_FILE}" \
    |& tee "${ATTEMPT_DIR}/server.log"
