#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "${#}" -ne 2 ]]; then
    echo "usage: $0 RUN_DIR {start|restart|stop|status|quit}" >&2
    exit 2
fi

RUN_DIR="${1}"
ACTION="${2}"
CONTROL_DIR="${RUN_DIR}/control"

if [[ ! -d "${CONTROL_DIR}" ]]; then
    echo "Controller is not ready: ${CONTROL_DIR}" >&2
    exit 1
fi

case "${ACTION}" in
    start|restart|stop|quit)
        touch "${CONTROL_DIR}/${ACTION}"
        echo "Requested ${ACTION} for Slurm job $(cat "${CONTROL_DIR}/job_id")"
        ;;
    status)
        echo "job_id=$(cat "${CONTROL_DIR}/job_id" 2>/dev/null || echo unavailable)"
        echo "nodes=$(paste -sd, "${CONTROL_DIR}/nodes" 2>/dev/null || echo unavailable)"
        echo "attempt=$(cat "${CONTROL_DIR}/attempt" 2>/dev/null || echo 0)"
        echo "state=$(cat "${CONTROL_DIR}/state" 2>/dev/null || echo initializing)"
        echo "server_url=$(cat "${RUN_DIR}/server_url" 2>/dev/null || echo unavailable)"
        echo "current_attempt_dir=$(cat "${CONTROL_DIR}/current_attempt_dir" 2>/dev/null || echo unavailable)"
        ;;
    *)
        echo "Unknown action: ${ACTION}" >&2
        exit 2
        ;;
esac
