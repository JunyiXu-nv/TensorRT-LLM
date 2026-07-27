#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Submit from the TensorRT-LLM repository root:
#   sbatch examples/serve/anthropic_compatibility/glm52/build_server.sh
# The allocation remains alive after the server stops. Use server_control.sh to
# start, restart, stop, or quit without requesting new nodes.

#SBATCH --job-name=glm52-nvfp4-anthropic
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --segment=2
#SBATCH --time=04:00:00
#SBATCH --output=glm52-nvfp4-anthropic-%j.out

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch from the TensorRT-LLM repository root}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
EXAMPLE_DIR="${REPO_DIR}/examples/serve/anthropic_compatibility/glm52"
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/common/GLM-5.2-NVFP4}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/serli/containers/sw-tensorrt-docker+tensorrt-llm+pytorch-26.05-py3-sbsa-ubuntu24.04-trt10.16.1.11-skip-tritondevel-202607151440-16194.sqsh}"
PORT="${PORT:-8333}"
RUN_ROOT="${EXAMPLE_DIR}/runs"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/glm52_nvfp4_anthropic_${SLURM_JOB_ID}}"
CONFIG_FILE="${RUN_DIR}/agg_config.yaml"
CONTROL_DIR="${RUN_DIR}/control"
START_SCRIPT="${EXAMPLE_DIR}/start_server.sh"

mkdir -p "${CONTROL_DIR}"
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
printf '%s\n' "${SLURM_JOB_ID}" > "${CONTROL_DIR}/job_id"
printf '%s\n' "${NODES[@]}" > "${CONTROL_DIR}/nodes"
rm -f "${CONTROL_DIR}/start" "${CONTROL_DIR}/restart" \
    "${CONTROL_DIR}/stop" "${CONTROL_DIR}/quit"

server_pid=""
attempt=0

stop_server() {
    if [[ -z "${server_pid}" ]]; then
        return
    fi

    printf '%s\n' "stopping attempt ${attempt}" > "${CONTROL_DIR}/state"
    if kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}" 2>/dev/null || true
    fi
    wait "${server_pid}" 2>/dev/null || true
    server_pid=""
    rm -f "${CONTROL_DIR}/server_pid"
}

start_server() {
    local attempt_dir

    stop_server
    attempt=$((attempt + 1))
    attempt_dir="${RUN_DIR}/attempt-$(printf '%03d' "${attempt}")"
    mkdir -p "${attempt_dir}"
    cp "${CONFIG_FILE}" "${attempt_dir}/agg_config.yaml"

    printf '%s\n' "${attempt}" > "${CONTROL_DIR}/attempt"
    printf '%s\n' "${attempt_dir}" > "${CONTROL_DIR}/current_attempt_dir"
    printf '%s\n' "starting attempt ${attempt}" > "${CONTROL_DIR}/state"

    REPO_DIR="${REPO_DIR}" \
    MODEL_PATH="${MODEL_PATH}" \
    CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
    PORT="${PORT}" \
        bash "${START_SCRIPT}" "${NODES[0]}" "${NODES[1]}" "${attempt_dir}" \
        > "${attempt_dir}/launcher.log" 2>&1 &
    server_pid=$!
    printf '%s\n' "${server_pid}" > "${CONTROL_DIR}/server_pid"
    printf '%s\n' "running attempt ${attempt}" > "${CONTROL_DIR}/state"
    echo "Started attempt ${attempt} with launcher PID ${server_pid}"
}

trap stop_server EXIT
trap 'exit 0' INT TERM
touch "${CONTROL_DIR}/start"

while true; do
    if [[ -f "${CONTROL_DIR}/quit" ]]; then
        rm -f "${CONTROL_DIR}/quit"
        printf '%s\n' "quitting" > "${CONTROL_DIR}/state"
        exit 0
    fi

    if [[ -f "${CONTROL_DIR}/stop" ]]; then
        rm -f "${CONTROL_DIR}/stop"
        stop_server
        printf '%s\n' "stopped; allocation retained" > "${CONTROL_DIR}/state"
    fi

    if [[ -f "${CONTROL_DIR}/start" || -f "${CONTROL_DIR}/restart" ]]; then
        rm -f "${CONTROL_DIR}/start" "${CONTROL_DIR}/restart"
        start_server
    fi

    if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
        if wait "${server_pid}"; then
            server_rc=0
        else
            server_rc=$?
        fi
        printf '%s\n' "attempt ${attempt} exited with status ${server_rc}; allocation retained" \
            > "${CONTROL_DIR}/state"
        server_pid=""
        rm -f "${CONTROL_DIR}/server_pid"
    fi

    sleep 2
done