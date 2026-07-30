#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-node variant for GLM-5.2-NVFP4: run within an existing salloc allocation.
# Usage:
#   REPO_DIR=... MODEL_PATH=... CONTAINER_IMAGE=... PORT=... \
#     bash start_server_1node_glm52.sh <node> <attempt_dir>
#
# Example invocation (umbriel-b200-145, job 3352521):
#   cd /home/scratch.serli_gpu/workspace/TensorRT-LLM
#   srun --jobid=3352521 --nodelist=umbriel-b200-145 --nodes=1 --ntasks=1 --overlap \
#     bash -c "
#       cd /home/scratch.serli_gpu/workspace/TensorRT-LLM
#       REPO_DIR=/home/scratch.serli_gpu/workspace/TensorRT-LLM \
#       MODEL_PATH=/home/scratch.trt_llm_data/llm-models/GLM-5.2-NVFP4 \
#       CONTAINER_IMAGE=/home/scratch.serli_gpu/workspace/containers/tensorrt-llm-pytorch-26.05-py3-x86_64-ubuntu24.04-trt10.16.1.11-skip-tritondevel-202607151440-16194.sqsh \
#       PORT=8333 \
#         bash examples/serve/anthropic_compatibility/computelab/start_server_1node_glm52.sh \
#           umbriel-b200-145 \
#           /home/scratch.serli_gpu/workspace/TensorRT-LLM/examples/serve/anthropic_compatibility/runs/glm52_modelbringup_3352521
#     "

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
CONTAINER_NAME="glm52-${SLURM_JOB_ID}"
USER_ROOT="/home/scratch.serli_gpu"
MOUNTS="${USER_ROOT}:${USER_ROOT},/home/scratch.trt_llm_data:/home/scratch.trt_llm_data,/raid:/raid"

cleanup_workers() {
    ssh "${NODE0}" \
        "pkill -TERM -f '[t]ensorrt_llm.llmapi.mgmn_' || true; pkill -TERM -f '[t]rtllm-llmapi-launch.*GLM' || true; pkill -TERM -f '[t]rtllm-serve.*GLM' || true" \
        || true
    sleep 5
    ssh "${NODE0}" \
        "pkill -KILL -f '[t]ensorrt_llm.llmapi.mgmn_' || true; pkill -KILL -f '[t]rtllm-llmapi-launch.*GLM' || true; pkill -KILL -f '[t]rtllm-serve.*GLM' || true" \
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

echo "Starting GLM-5.2-NVFP4 TP8/EP8 server (single node) at http://${NODE0}:${PORT}"
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
            --tool_parser "${TOOL_PARSER:-glm47}"
    ' _ "${MODEL_PATH}" "${PORT}" "${CONFIG_FILE}" \
    |& tee "${ATTEMPT_DIR}/server.log"
