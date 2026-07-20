#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Submit from the TensorRT-LLM repository root:
#   sbatch examples/serve/anthropic_compatibility/build_server.sh
# Optional overrides:
#   sbatch --export=ALL,MODEL_PATH=/path,CONTAINER_IMAGE=/path,RUN_DIR=/path,PORT=8333 \
#     examples/serve/anthropic_compatibility/build_server.sh

#SBATCH --job-name=deepseek-v4-pro-anthropic
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --segment=2
#SBATCH --time=04:00:00
#SBATCH --output=deepseek-v4-pro-anthropic-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch from the TensorRT-LLM repository root}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
EXAMPLE_DIR="${REPO_DIR}/examples/serve/anthropic_compatibility"
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/common/DeepSeek-V4-Pro}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/serli/containers/sw-tensorrt-docker+tensorrt-llm+pytorch-26.05-py3-sbsa-ubuntu24.04-trt10.16.1.11-skip-tritondevel-202607151440-16194.sqsh}"
PORT="${PORT:-8333}"
RUN_ROOT="$(cd "${REPO_DIR}/../.." && pwd)/runs"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/deepseek_v4_pro_anthropic_${SLURM_JOB_ID}}"
CONFIG_FILE="${RUN_DIR}/agg_config.yaml"
CONTAINER_NAME="deepseek-v4-pro-${SLURM_JOB_ID}"
MOUNTS="/lustre/fsw/portfolios/coreai/:/lustre/fsw/portfolios/coreai/,/lustre/fs1/portfolios/coreai/:/lustre/fs1/portfolios/coreai/"

mkdir -p "${RUN_DIR}"
cp "${EXAMPLE_DIR}/agg_config.yaml" "${CONFIG_FILE}"

{
    echo "origin=$(git -C "${REPO_DIR}" config --get remote.origin.url)"
    echo "branch=$(git -C "${REPO_DIR}" branch --show-current)"
    echo "commit=$(git -C "${REPO_DIR}" rev-parse HEAD)"
    echo "model=${MODEL_PATH}"
    echo "container=${CONTAINER_IMAGE}"
} > "${RUN_DIR}/server_metadata.txt"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#NODES[@]}" -ne 2 ]]; then
    echo "Expected exactly two allocated nodes; got ${#NODES[@]}" >&2
    exit 1
fi
SERVER_HOST="${NODES[0]}"
echo "http://${SERVER_HOST}:${PORT}" > "${RUN_DIR}/server_url"

cleanup_workers() {
    local node
    for node in "${NODES[@]}"; do
        ssh "${node}" \
            "pkill -TERM -f '[t]ensorrt_llm.llmapi.mgmn_' || true; pkill -TERM -f '[t]rtllm-llmapi-launch.*DeepSeek-V4-Pro' || true; pkill -TERM -f '[t]rtllm-serve.*DeepSeek-V4-Pro' || true" \
            || true
    done
    sleep 5
    for node in "${NODES[@]}"; do
        ssh "${node}" \
            "pkill -KILL -f '[t]ensorrt_llm.llmapi.mgmn_' || true; pkill -KILL -f '[t]rtllm-llmapi-launch.*DeepSeek-V4-Pro' || true; pkill -KILL -f '[t]rtllm-serve.*DeepSeek-V4-Pro' || true" \
            || true
    done
}

trap cleanup_workers EXIT INT TERM
cleanup_workers

echo "Installing branch checkout on both nodes: $(git -C "${REPO_DIR}" branch --show-current)"
srun -l \
    --nodes 2 \
    --ntasks 2 \
    --ntasks-per-node 1 \
    --container-image "${CONTAINER_IMAGE}" \
    --container-name "${CONTAINER_NAME}" \
    --container-mounts "${MOUNTS}" \
    --no-container-mount-home \
    --mpi=pmix \
    bash -lc "cd '${REPO_DIR}' && python3 -m pip install -e ." \
    |& tee "${RUN_DIR}/install.log"

echo "Starting DeepSeek-V4-Pro aggregated TP8/EP8 server at http://${SERVER_HOST}:${PORT}"
srun -l \
    --nodelist "${NODES[0]},${NODES[1]}" \
    --nodes 2 \
    --ntasks 8 \
    --ntasks-per-node 4 \
    --export="ALL,TLLM_LOG_LEVEL=INFO,TRTLLM_SERVER_DISABLE_GC=1,TRTLLM_WORKER_DISABLE_GC=1,TRTLLM_ENABLE_PDL=1,ENROOT_ALLOW_DEV=yes,NCCL_GRAPH_MIXING_SUPPORT=0,MIMALLOC_PURGE_DELAY=0" \
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
    |& tee "${RUN_DIR}/server.log"
