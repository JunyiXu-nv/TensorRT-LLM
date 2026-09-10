#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Unified launcher for the Anthropic-compatibility serving runs.
#
# One deployment YAML fully describes a cluster + model pair. Node count, task
# layout, container name, audit-capture paths, and the trace directory are all
# derived -- never spelled out by hand.
#
#   ./serve.sh submit --yaml deployments/computelab_glm5.2.yaml --label bringup
#   ./serve.sh status <run_dir>
#   ./serve.sh restart <run_dir>
#   ./serve.sh quit <run_dir>
#
# Inside an existing salloc allocation, skip sbatch and run the controller directly:
#   ./serve.sh run --yaml deployments/computelab_glm5.2.yaml

set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

die() {
    echo "serve.sh: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
usage: serve.sh <command> [options]

commands:
  submit  --yaml FILE [--label TEXT]     compose sbatch flags from the deployment
                                         YAML and submit
  run     --yaml FILE [--label TEXT]     controller loop; runs inside an allocation
                                         (sbatch target, or invoke directly under
                                         salloc)
  gateway --yaml FILE [--submit]         stable front door for the serving
                                         jobs; --submit runs it as a long
                                         CPU-only Slurm job instead of here
  gateway --yaml FILE --start            ask a running gateway to find GPUs and
                                         bring a serving job up
  gateway --yaml FILE --stop             ask it to release every serving job;
                                         only the gateway keeps running
  gateway --yaml FILE --status           what that gateway is doing
  start|restart|stop|quit|status RUN_DIR drive a controller already running
EOF
}

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Merges the deployment YAML with the built-in model table and the server config,
# then prints shell assignments for eval. Every derived value is computed here so
# that submit / run / launch cannot drift apart.
resolve() {
    python3 - "$1" <<'PY'
import os
import shlex
import sys

import yaml

scenario_path = os.path.abspath(sys.argv[1])
# Paths resolve against the deployment YAML, never against this script: sbatch
# executes a spool copy of it, so its own directory is meaningless there.
scenario_dir = os.path.dirname(scenario_path)

# model.name is constrained so a typo cannot silently produce a new container
# name, job name and trace directory that look almost right.
KNOWN_MODELS = ("glm5.2", "glm5.3", "deepseek_v4", "deepseek_v4_flash",
                "deepseek_v4_pro")

# Always-on audit capture is layered on top of these in launch().
ENV_DEFAULTS = {
    "TLLM_LOG_LEVEL": "INFO",
    "TRTLLM_SERVER_DISABLE_GC": "1",
    "TRTLLM_WORKER_DISABLE_GC": "1",
    "TRTLLM_ENABLE_PDL": "1",
    "TRTLLM_ANTHROPIC_LCP_TRACKING": "1",
    "ENROOT_ALLOW_DEV": "yes",
    "NCCL_GRAPH_MIXING_SUPPORT": "0",
    "MIMALLOC_PURGE_DELAY": "0",
}

def die(message):
    sys.stderr.write("serve.sh: config error: %s\n" % message)
    sys.exit(1)


if not os.path.isfile(scenario_path):
    die("deployment YAML not found: %s" % scenario_path)
with open(scenario_path) as handle:
    cfg = yaml.safe_load(handle) or {}


def section(key):
    return cfg.get(key) or {}


model = section("model")
slurm = section("slurm")
container = section("container")
server = section("server")
trace = section("trace")
# Optional throughout: a deployment YAML written before the gateway existed
# still resolves, so every field below reads through .get() with a default.
gateway = section("gateway")


def require(value, what):
    if value in (None, ""):
        die("%s is required in %s" % (what, os.path.basename(scenario_path)))
    return value


model_key = require(model.get("name"), "model.name")
if model_key not in KNOWN_MODELS:
    die("model.name %r is not one of: %s" % (model_key, ", ".join(KNOWN_MODELS)))

# The run name identifies cluster + model, and feeds the Slurm job name, the
# container name, and the trace directory.
cluster = cfg.get("cluster_name") or os.path.splitext(os.path.basename(scenario_path))[0]
name = "%s_%s" % (cluster, model_key)

model_path = require(model.get("path"), "model.path").rstrip("/")

def resolve_config(value, what):
    path = require(value, what)
    if not os.path.isabs(path):
        path = os.path.join(scenario_dir, path)
    if not os.path.isfile(path):
        die("server config not found: %s (relative paths resolve against %s)"
            % (path, scenario_dir))
    return path


def ranks_of(path):
    with open(path) as handle:
        engine = yaml.safe_load(handle) or {}
    return (int(engine.get("tensor_parallel_size", 1))
            * int(engine.get("pipeline_parallel_size", 1))
            * int(engine.get("context_parallel_size", 1)))


# gpus_per_node is a hardware fact and has to be stated. The node count is not:
# it follows from the parallel sizes and the instance counts, so it is derived
# below rather than written down a second time. A TP change or an extra context
# instance now resizes the allocation by itself, where before it needed a
# matching edit in the deployment YAML and only failed loudly if it missed one.
tasks_per_node = int(require(slurm.get("gpus_per_node"), "slurm.gpus_per_node"))

# server.disagg selects disaggregated serving: a proxy plus context and
# generation workers, instead of one aggregated server. Its absence keeps the
# aggregated path byte-for-byte, so existing deployments are unaffected.
disagg = server.get("disagg") or {}
if disagg:
    ctx = disagg.get("ctx") or {}
    gen = disagg.get("gen") or {}
    ctx_config = resolve_config(ctx.get("config"), "server.disagg.ctx.config")
    gen_config = resolve_config(gen.get("config"), "server.disagg.gen.config")
    ctx_ranks = ranks_of(ctx_config)
    gen_ranks = ranks_of(gen_config)
    # server.disagg.config names the proxy's own YAML -- router types,
    # num_workers, retry and schedule policy. Required, not optional: a
    # disaggregated deployment always has proxy behaviour, and leaving it
    # implicit meant that behaviour was whatever DisaggServerConfig defaulted
    # to, stated nowhere and invisible in review. The file may be as short as
    # the two num_instances counts.
    #
    # The four fields serve.sh must own (hostname, port, the two url lists and
    # the metrics directory) are injected at launch, because only the running
    # job knows the node names. Stating them here would be a value that is
    # always overwritten, so they are rejected instead of silently ignored.
    disagg_config = resolve_config(disagg.get("config"), "server.disagg.config")
    with open(disagg_config) as handle:
        proxy_cfg = yaml.safe_load(handle) or {}
    for key in ("hostname", "port", "perf_metrics_output_dir"):
        if key in proxy_cfg:
            die("%s is assigned by serve.sh at launch; remove it from %s"
                % (key, os.path.basename(disagg_config)))
    for section_name in ("context_servers", "generation_servers"):
        if "urls" in (proxy_cfg.get(section_name) or {}):
            die("%s.urls is assigned by serve.sh at launch; remove it "
                "from %s" % (section_name, os.path.basename(disagg_config)))

    # Instance counts run the other way: the proxy config is authoritative when
    # it states them, because that file is what the proxy validates its url list
    # against. server.disagg.{ctx,gen}.instances stays supported for deployments
    # with no proxy config, and a disagreement between the two is a typo that
    # would otherwise surface as a url-count mismatch inside trtllm-serve.
    def instances_of(role, section_name, stated):
        declared = (proxy_cfg.get(section_name) or {}).get("num_instances")
        if declared is None:
            return int(stated) if stated is not None else 1
        if stated is not None and int(stated) != int(declared):
            die("server.disagg.%s.instances is %s but %s.num_instances is %s "
                "in %s; they name the same thing"
                % (role, stated, section_name, declared,
                   os.path.basename(disagg_config)))
        return int(declared)

    ctx_instances = instances_of("ctx", "context_servers", ctx.get("instances"))
    gen_instances = instances_of("gen", "generation_servers", gen.get("instances"))
    # Whole-node ownership. The reference launcher also supports packing two
    # workers onto one node via a per-worker gpu_map, but that only matters
    # when a worker's rank count does not divide the node, and it costs the
    # simple CUDA_VISIBLE_DEVICES=$SLURM_LOCALID mapping used below.
    for role, role_ranks in (("ctx", ctx_ranks), ("gen", gen_ranks)):
        if role_ranks % tasks_per_node:
            die("server.disagg.%s needs %d ranks, which does not divide "
                "slurm.gpus_per_node %d; whole nodes per worker are required"
                % (role, role_ranks, tasks_per_node))
    world_size = ctx_ranks * ctx_instances + gen_ranks * gen_instances
    server_config = ""
    tp = pp = 0
else:
    server_config = resolve_config(server.get("config"), "server.config")
    ctx_config = gen_config = disagg_config = ""
    ctx_ranks = gen_ranks = ctx_instances = gen_instances = 0
    with open(server_config) as handle:
        engine = yaml.safe_load(handle) or {}
    tp = int(engine.get("tensor_parallel_size", 1))
    pp = int(engine.get("pipeline_parallel_size", 1))
    cp = int(engine.get("context_parallel_size", 1))
    world_size = tp * pp * cp

# Whole nodes only: one srun covers the allocation and CUDA_VISIBLE_DEVICES is
# just SLURM_LOCALID, so a rank total that does not fill its nodes has nowhere
# to put the remainder. The disaggregated path already required this per role,
# which makes the sum divide too; the aggregated path is checked here.
if world_size % tasks_per_node:
    die("%d ranks do not divide slurm.gpus_per_node %d; a whole number of "
        "nodes is required" % (world_size, tasks_per_node))
nodes = world_size // tasks_per_node
# slurm.nodes is no longer read, only checked. Deployments that still state it
# keep working, and one that states a stale value says so instead of silently
# meaning something else than it reads.
stated_nodes = slurm.get("nodes")
if stated_nodes is not None and int(stated_nodes) != nodes:
    die("slurm.nodes says %s but the configured parallel sizes need %d x %d "
        "= %d ranks; drop slurm.nodes, it is derived"
        % (stated_nodes, nodes, tasks_per_node, world_size))

# server.env is the whole environment: this script holds no defaults, so a
# variable that is not in the deployment YAML is not set at all. Always-on audit
# capture is layered on top in launch().
#
# Values are stringified, which makes YAML's type inference load-bearing:
# unquoted `yes`, `no`, `on` and `off` resolve to booleans and would reach the
# job as "True"/"False", and unquoted `0755` would lose its leading zero. Quote
# anything whose exact text matters.
env = dict(ENV_DEFAULTS)
for key, value in (server.get("env") or {}).items():
    env[str(key)] = str(value)

mounts = container.get("mounts") or []
if not mounts:
    die("container.mounts is required")

out = []


def emit(key, value):
    out.append("%s=%s" % (key, shlex.quote("" if value is None else str(value))))


def emit_array(key, values):
    out.append("%s=(%s)" % (key, " ".join(shlex.quote(str(v)) for v in values)))


emit("CFG_NAME", name)
emit("CFG_MODEL_KEY", model_key)
emit("CFG_MODEL_PATH", model_path)
emit("CFG_TOOL_PARSER", require(model.get("tool_parser"), "model.tool_parser"))
# pkill matches the whole command line, and what appears there is the checkpoint
# path -- not model.name, which never shows up in it.
emit("CFG_PROC_PATTERN", os.path.basename(model_path))
emit("CFG_SERVER_CONFIG", server_config)
emit("CFG_DISAGG", "1" if disagg else "0")
emit("CFG_CTX_CONFIG", ctx_config)
emit("CFG_GEN_CONFIG", gen_config)
# The proxy's own config, or "" when the deployment states no behaviour beyond
# the topology launch_disagg derives from the allocation.
emit("CFG_DISAGG_CONFIG", disagg_config)
emit("CFG_CTX_RANKS", ctx_ranks)
emit("CFG_GEN_RANKS", gen_ranks)
emit("CFG_CTX_INSTANCES", ctx_instances)
emit("CFG_GEN_INSTANCES", gen_instances)
# The proxy sits in _wait_for_all_servers_ready for the whole of the slowest
# worker's model load, so this has to exceed it; the CLI default is 180s, and
# slurm/benchmark/start_server.sh uses 7200 for both -t and -r.
emit("CFG_DISAGG_TIMEOUT", int(disagg.get("start_timeout", 7200)) if disagg else 0)
emit("CFG_REPO_DIR",
     cfg.get("repo_dir") or os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd())

emit("CFG_TP", tp)
emit("CFG_PP", pp)
emit("CFG_WORLD_SIZE", world_size)
emit("CFG_NODES", nodes)
emit("CFG_TASKS_PER_NODE", tasks_per_node)

emit("CFG_ACCOUNT", require(slurm.get("account"), "slurm.account"))
emit("CFG_PARTITION", require(slurm.get("partition"), "slurm.partition"))
emit("CFG_TIME", require(slurm.get("time"), "slurm.time"))
emit("CFG_RESERVATION", slurm.get("reservation") or "")
emit("CFG_QOS", slurm.get("qos") or "")
emit("CFG_SEGMENT", slurm.get("segment") or "")
emit_array("CFG_EXTRA_ARGS", slurm.get("extra_args") or [])

emit("CFG_IMAGE", require(container.get("image"), "container.image"))
emit("CFG_MOUNTS", ",".join(str(m) for m in mounts))

emit("CFG_PORT", server.get("port") or 8333)
emit("CFG_INSTALL_REPO", "1" if server.get("install_repo", True) else "0")
emit("CFG_NUMACTL", server.get("numactl") or "")
emit_array("CFG_SERVE_EXTRA_ARGS", server.get("extra_args") or [])
# srun --export is a comma-separated list, so a value containing a comma
# (UCX_TLS is a transport list) would be split into one bogus entry per
# element. Quoting does not help: srun does not strip quotes on this path --
# it hands the task TLLM_LOG_LEVEL='INFO', quotes included. So the two kinds
# are carried differently: comma-free values go in --export as before, and the
# rest are set on the srun process itself, which --export=ALL then propagates.
# The GPU nodes' enroot environ.d injects UCX_TLS=tcp and UCX_NET_DEVICES=eth0
# after srun's environment, so these two are exported inside the worker shell.
emit("CFG_UCX_TLS", env.pop("UCX_TLS", ""))
emit("CFG_UCX_NET_DEVICES", env.pop("UCX_NET_DEVICES", ""))
_plain = {k: v for k, v in env.items() if "," not in v}
_multi = {k: v for k, v in env.items() if "," in v}
emit("CFG_ENV", ",".join("%s=%s" % kv for kv in sorted(_plain.items())))
emit_array("CFG_ENV_MULTI", ["%s=%s" % kv for kv in sorted(_multi.items())])

emit("CFG_CAPTURE", "1" if server.get("capture", True) else "0")

trace_root = require(trace.get("root"), "trace.root")
emit("CFG_TRACE_ROOT", trace_root)

# --- gateway -------------------------------------------------------------
# The serving jobs register here and the gateway reads it; one file per job, so
# there is exactly one writer per file and no locking on the shared filesystem.
#
# Namespaced by cluster+model, not just trace.root: the deployments share a
# trace root, and a flat directory would let one model's gateway discover
# another model's backend. Election only compares end times, so it would route
# there happily and answer from the wrong model with nothing to show for it.
emit("CFG_FLEET_DIR", os.path.join(trace_root, "_fleet", name))

# sbatch runs a spool copy of serve.sh, so its own directory says nothing about
# where gateway.py lives. Resolve it against the deployment YAML instead, the
# same rule the rest of this resolver follows.
gateway_script = gateway.get("script") or os.path.join(
    os.path.dirname(scenario_dir), "gateway.py")
if not os.path.isabs(gateway_script):
    gateway_script = os.path.join(scenario_dir, gateway_script)
emit("CFG_GW_SCRIPT", gateway_script)
# serve.sh sits next to gateway.py by construction. The gateway resubmits
# through this path rather than through $0, which under sbatch is a spool copy.
emit("CFG_SERVE_SH", os.path.join(os.path.dirname(gateway_script), "serve.sh"))

gateway_users = gateway.get("users") or "gateway_users.txt"
if not os.path.isabs(gateway_users):
    gateway_users = os.path.join(scenario_dir, gateway_users)
emit("CFG_GW_USERS", gateway_users)

emit("CFG_GW_PORT", gateway.get("port") or 8333)
emit("CFG_GW_LEAD_TIME", gateway.get("lead_time") or 2700)
# The gateway needs no GPU and no reservation, so it does not inherit the
# serving job's partition or QoS -- those are usually reservation-bound.
emit("CFG_GW_ACCOUNT", gateway.get("account") or slurm.get("account") or "")
emit("CFG_GW_PARTITION", gateway.get("partition") or "")
emit("CFG_GW_QOS", gateway.get("qos") or "")
emit("CFG_GW_TIME", gateway.get("time") or "7-00:00:00")
emit_array("CFG_GW_EXTRA_ARGS", gateway.get("extra_args") or [])

print("\n".join(out))
PY
}

# `eval "$(resolve ...)"` would swallow a resolver failure, so stage the output
# in a variable first and let set -e see the non-zero status.
load_config() {
    local resolved
    resolved="$(resolve "${ARG_YAML}")"
    eval "${resolved}"
}

# TP/PP describe the single aggregated server and are meaningless once the
# deployment is split into context and generation workers with their own
# parallel sizes, so disaggregated runs report their instance layout instead.
topology_summary() {
    if [[ "${CFG_DISAGG}" == "1" ]]; then
        echo "${CFG_CTX_INSTANCES}xctx${CFG_CTX_RANKS} + ${CFG_GEN_INSTANCES}xgen${CFG_GEN_RANKS}"
    else
        echo "TP${CFG_TP} x PP${CFG_PP}"
    fi
}

parse_args() {
    ARG_YAML=""
    ARG_LABEL=""
    ARG_ATTEMPT_DIR=""
    ARG_SUBMIT=""
    ARG_CONTROL=""
    while (( $# )); do
        case "$1" in
            --yaml) ARG_YAML="${2:?--yaml needs a value}"; shift 2 ;;
            --label) ARG_LABEL="${2:?--label needs a value}"; shift 2 ;;
            --attempt-dir) ARG_ATTEMPT_DIR="${2:?--attempt-dir needs a value}"; shift 2 ;;
            --submit) ARG_SUBMIT="1"; shift ;;
            --start) ARG_CONTROL="start"; shift ;;
            --stop) ARG_CONTROL="stop"; shift ;;
            --status) ARG_CONTROL="status"; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done
    [[ -n "${ARG_YAML}" ]] || die "--yaml is required"
    ARG_YAML="$(readlink -f "${ARG_YAML}")"
}

# --start/--stop/--status drive a gateway; every other command parses the same
# flags and would silently ignore them. `serve.sh submit --yaml X --stop` would
# then allocate GPUs, which is the exact opposite of what was typed.
reject_control_flags() {
    [[ -z "${ARG_CONTROL}" ]] || die "--${ARG_CONTROL} only applies to \
'serve.sh gateway'; you probably meant: serve.sh gateway --yaml ${ARG_YAML} --${ARG_CONTROL}"
}

# --------------------------------------------------------------------------
# submit: build the sbatch command line from the YAML
# --------------------------------------------------------------------------
cmd_submit() {
    parse_args "$@"
    reject_control_flags
    load_config

    local log_dir="${CFG_TRACE_ROOT}/_sbatch_logs"
    mkdir -p "${log_dir}"

    local sbatch_args=(
        --job-name "${CFG_NAME}"
        --account "${CFG_ACCOUNT}"
        --partition "${CFG_PARTITION}"
        --nodes "${CFG_NODES}"
        --ntasks "${CFG_WORLD_SIZE}"
        --ntasks-per-node "${CFG_TASKS_PER_NODE}"
        --gres "gpu:${CFG_TASKS_PER_NODE}"
        --time "${CFG_TIME}"
        --output "${log_dir}/${CFG_NAME}-%j.out"
    )
    if [[ -n "${CFG_RESERVATION}" ]]; then
        sbatch_args+=(--reservation "${CFG_RESERVATION}")
    fi
    # QoS is otherwise inherited from whatever job the submit runs inside, which
    # fails if that QoS is denied on the target partition.
    if [[ -n "${CFG_QOS}" ]]; then
        sbatch_args+=(--qos "${CFG_QOS}")
    fi
    if [[ -n "${CFG_SEGMENT}" ]]; then
        sbatch_args+=(--segment "${CFG_SEGMENT}")
    fi
    sbatch_args+=(${CFG_EXTRA_ARGS[@]+"${CFG_EXTRA_ARGS[@]}"})

    echo "${CFG_NAME}: $(topology_summary) = ${CFG_WORLD_SIZE} ranks" \
         "-> ${CFG_NODES} node(s) x ${CFG_TASKS_PER_NODE} GPU(s)"
    echo "model: ${CFG_MODEL_PATH}"
    echo "trace root: ${CFG_TRACE_ROOT}"

    local forward=(run --yaml "${ARG_YAML}")
    if [[ -n "${ARG_LABEL}" ]]; then
        forward+=(--label "${ARG_LABEL}")
    fi
    sbatch "${sbatch_args[@]}" "${SCRIPT_PATH}" "${forward[@]}"
}

# --------------------------------------------------------------------------
# run: controller loop inside the allocation
# --------------------------------------------------------------------------
RUN_DIR=""
CONTROL_DIR=""
server_pid=""
attempt=0
FLEET_FILE=""
FLEET_URL=""
FLEET_END_TIME=0

# Registration for the gateway, which discovers backends by listing the fleet
# directory. One file per job means a single writer per file, so nothing has to
# lock; the rename makes each update atomic, so a reader never sees half a
# record.
#
# Nothing in here may be fatal. cmd_run is the controller of a live server and
# runs under `set -e`: if a stalled filesystem could fail this function, it
# could take the server down with it. The subshell absorbs both the exit status
# and any message.
write_fleet() {
    [[ -n "${FLEET_FILE}" ]] || return 0
    local state
    state="$(head -1 "${CONTROL_DIR}/state" 2>/dev/null | tr -d '"\\' || true)"
    # The suppression wraps the group rather than the redirect: a failing
    # redirect is reported by the shell setting it up, so `> file 2>/dev/null`
    # would still print. At one heartbeat per 10s that would flood the job log.
    {
        (
            printf '{"job_id":"%s","url":"%s","run_dir":"%s","state":"%s",' \
                "${SLURM_JOB_ID}" "${FLEET_URL}" "${RUN_DIR}" "${state:-unknown}"
            printf '"end_time":%s,"heartbeat":%s}\n' \
                "${FLEET_END_TIME}" "$(date +%s)"
        ) > "${FLEET_FILE}.tmp" \
            && mv -f "${FLEET_FILE}.tmp" "${FLEET_FILE}"
    } 2>/dev/null
    return 0
}

# Nothing ties this process to the allocation it serves: `run` is startable from
# outside the job by passing SLURM_JOB_ID and SLURM_JOB_NODELIST by hand, and
# Slurm reclaiming the nodes does not signal it. Such a controller keeps
# heartbeating a record that still carries the end time captured at startup, so
# a preempted 7-day allocation goes on advertising a week of remaining life. The
# gateway elects on end time, so that record outranks every live 4-hour backend,
# takes routing to a URL nobody serves, and reclaims the backend that was
# actually working. It also keeps `fleet.backends` non-empty, which is the
# condition the gateway's own recovery path waits on -- so nothing repairs it.
#
# Only a positive answer is acted on. A failing squeue means a busy or
# restarting slurmctld, which must never tear down a healthy server, so anything
# short of Slurm plainly saying this job is not running counts as alive.
job_is_gone() {
    local out rc=0
    out="$(squeue -h -j "${SLURM_JOB_ID}" -o '%T' 2>&1)" || rc=$?
    if (( rc != 0 )); then
        # An id Slurm no longer recognises is exactly what this guards against;
        # any other failure is an outage on Slurm's side, not on ours.
        [[ "${out,,}" == *"invalid job id"* ]]
        return
    fi
    # Empty output means the job left the queue. A state that is not one of
    # these is not serving either -- a requeued job runs its batch script
    # afresh, which starts a controller of its own.
    case "${out}" in
        *RUNNING*|*COMPLETING*|*CONFIGURING*) return 1 ;;
        *) return 0 ;;
    esac
}

# Deregistration is best effort by design: a job killed with SIGKILL never gets
# here, so the gateway ages entries out by heartbeat rather than trusting this.
clear_fleet() {
    [[ -n "${FLEET_FILE}" ]] || return 0
    rm -f "${FLEET_FILE}" "${FLEET_FILE}.tmp" 2>/dev/null || true
    return 0
}

on_exit() {
    stop_server
    clear_fleet
}

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

start_attempt() {
    local attempt_dir
    stop_server
    attempt=$((attempt + 1))
    attempt_dir="${RUN_DIR}/attempt-$(printf '%03d' "${attempt}")"
    mkdir -p "${attempt_dir}"
    # Snapshot every engine config this attempt runs on, so the attempt stays
    # replayable after the deployment YAML moves on. The generated
    # disagg_config.yaml is written by launch, which knows the node names.
    if [[ "${CFG_DISAGG}" == "1" ]]; then
        cp "${CFG_CTX_CONFIG}" "${attempt_dir}/ctx_config.yaml"
        cp "${CFG_GEN_CONFIG}" "${attempt_dir}/gen_config.yaml"
    else
        cp "${CFG_SERVER_CONFIG}" "${attempt_dir}/server_config.yaml"
    fi

    printf '%s\n' "${attempt}" > "${CONTROL_DIR}/attempt"
    printf '%s\n' "${attempt_dir}" > "${CONTROL_DIR}/current_attempt_dir"
    printf '%s\n' "starting attempt ${attempt}" > "${CONTROL_DIR}/state"

    "${SCRIPT_PATH}" launch \
        --yaml "${ARG_YAML}" --attempt-dir "${attempt_dir}" \
        > "${attempt_dir}/launcher.log" 2>&1 &
    server_pid=$!
    printf '%s\n' "${server_pid}" > "${CONTROL_DIR}/server_pid"
    printf '%s\n' "running attempt ${attempt}" > "${CONTROL_DIR}/state"
    echo "started attempt ${attempt} with launcher PID ${server_pid}"
}

cmd_run() {
    parse_args "$@"
    reject_control_flags
    : "${SLURM_JOB_ID:?serve.sh run must execute inside a Slurm allocation}"
    : "${SLURM_JOB_NODELIST:?serve.sh run must execute inside a Slurm allocation}"
    load_config

    # Validate the allocation before creating anything under the trace root.
    local nodes
    mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    if [[ "${#nodes[@]}" -ne "${CFG_NODES}" ]]; then
        die "allocation has ${#nodes[@]} node(s); ${CFG_NAME} needs ${CFG_NODES}" \
            "($(topology_summary) over ${CFG_TASKS_PER_NODE} GPUs per node)"
    fi

    # user_MMDDHH_slurmjob_jobname, shared across every attempt of this job.
    # Date-partitioned: <root>/runs/2026-08/19/serli_081914_... . The flat
    # layout reached 122 sibling directories, which made finding a specific run
    # a matter of reading job ids. Nothing globs the trace root itself -- the
    # gateway only reads _fleet/<deployment>/*.json, and every consumer takes
    # an absolute run directory -- so the extra levels are free.
    #
    # The runs/ level exists so that it, alone, can be a symlink: the run
    # directories are the bulk (~7 GiB a day) and can then be billed to a
    # different Lustre project quota than _fleet and _sbatch_logs, which are
    # kilobytes and must stay wherever the running gateway found them. One
    # trace.root either way -- the split, if any, is made on disk, not in YAML.
    RUN_DIR="${CFG_TRACE_ROOT}/$(date +%Y-%m)/$(date +%d)/${USER}_$(date +%m%d%H)_${SLURM_JOB_ID}_${CFG_NAME}"
    if [[ -n "${ARG_LABEL}" ]]; then
        RUN_DIR="${RUN_DIR}_${ARG_LABEL}"
    fi
    CONTROL_DIR="${RUN_DIR}/control"
    mkdir -p "${CONTROL_DIR}"

    # The server config is snapshotted per attempt, not here: it can be edited
    # between a stop and the next start, and the attempt copy is what ran.
    cp "${ARG_YAML}" "${RUN_DIR}/deployment.yaml"
    {
        echo "name=${CFG_NAME}"
        echo "model=${CFG_MODEL_KEY}"
        echo "model_path=${CFG_MODEL_PATH}"
        echo "container=${CFG_IMAGE}"
        echo "topology=$(topology_summary) on ${CFG_NODES}x${CFG_TASKS_PER_NODE}"
        echo "nodes=$(IFS=,; echo "${nodes[*]}")"
        echo "origin=$(git -C "${CFG_REPO_DIR}" config --get remote.origin.url)"
        echo "branch=$(git -C "${CFG_REPO_DIR}" branch --show-current)"
        echo "commit=$(git -C "${CFG_REPO_DIR}" rev-parse HEAD)"
    } > "${RUN_DIR}/run_metadata.txt"

    echo "http://${nodes[0]}:${CFG_PORT}" > "${RUN_DIR}/server_url"
    printf '%s\n' "${SLURM_JOB_ID}" > "${CONTROL_DIR}/job_id"
    printf '%s\n' "${nodes[@]}" > "${CONTROL_DIR}/nodes"
    rm -f "${CONTROL_DIR}/start" "${CONTROL_DIR}/restart" \
        "${CONTROL_DIR}/stop" "${CONTROL_DIR}/quit"

    # The gateway ranks backends by end time, so a successor automatically wins
    # the election once it is healthy. Slurm >= 20.11 puts the timestamp in the
    # job environment; ask the controller only when it is missing. Neither path
    # may fail the run -- an unknown end time costs the gateway its relay
    # timing, not the ability to route to this server.
    FLEET_END_TIME="${SLURM_JOB_END_TIME:-}"
    if [[ -z "${FLEET_END_TIME}" ]]; then
        FLEET_END_TIME="$(squeue -h -j "${SLURM_JOB_ID}" -o '%e' 2>/dev/null || true)"
        FLEET_END_TIME="$(date -d "${FLEET_END_TIME:-x}" +%s 2>/dev/null || echo 0)"
    fi
    FLEET_URL="http://${nodes[0]}:${CFG_PORT}"
    FLEET_FILE="${CFG_FLEET_DIR}/${SLURM_JOB_ID}.json"

    mkdir -p "${CFG_FLEET_DIR}" 2>/dev/null || true

    echo "run dir: ${RUN_DIR}"
    echo "server:  http://${nodes[0]}:${CFG_PORT}"
    echo "fleet:   ${FLEET_FILE} (ends $(date -d "@${FLEET_END_TIME}" '+%F %T' \
        2>/dev/null || echo unknown))"

    trap on_exit EXIT
    trap 'exit 0' INT TERM
    touch "${CONTROL_DIR}/start"
    write_fleet

    local tick=0
    while true; do
        if [[ -f "${CONTROL_DIR}/quit" ]]; then
            rm -f "${CONTROL_DIR}/quit"
            # Stop first: the EXIT trap would otherwise overwrite the final state.
            stop_server
            printf '%s\n' "quit; allocation released" > "${CONTROL_DIR}/state"
            exit 0
        fi

        if [[ -f "${CONTROL_DIR}/stop" ]]; then
            rm -f "${CONTROL_DIR}/stop"
            stop_server
            printf '%s\n' "stopped; allocation retained" > "${CONTROL_DIR}/state"
        fi

        if [[ -f "${CONTROL_DIR}/start" || -f "${CONTROL_DIR}/restart" ]]; then
            rm -f "${CONTROL_DIR}/start" "${CONTROL_DIR}/restart"
            start_attempt
        fi

        if [[ -n "${server_pid}" ]] && ! kill -0 "${server_pid}" 2>/dev/null; then
            local rc=0
            wait "${server_pid}" || rc=$?
            printf '%s\n' "attempt ${attempt} exited with status ${rc}; allocation retained" \
                > "${CONTROL_DIR}/state"
            server_pid=""
            rm -f "${CONTROL_DIR}/server_pid"
        fi

        # Every fifth turn, so the heartbeat lands every 10s while the loop
        # stays at 2s for the control files. The gateway only needs this to
        # notice a job that vanished without deregistering -- a backend that is
        # merely unhealthy is caught much sooner by /health probing.
        #
        # Written as an assignment on purpose: `(( tick++ ))` returns 1 when the
        # value is 0, which under `set -e` would kill the controller on the
        # first turn.
        tick=$(( (tick + 1) % 5 ))
        if [[ "${tick}" -eq 0 ]]; then
            # Checked before the heartbeat, never after: a controller that has
            # outlived its allocation must stop advertising rather than publish
            # one more record. clear_fleet runs here as well as in the EXIT
            # trap, so a stop_server that blocks cannot leave the registration
            # behind.
            if job_is_gone; then
                printf '%s\n' "allocation gone; controller exiting" \
                    > "${CONTROL_DIR}/state"
                clear_fleet
                echo "Slurm job ${SLURM_JOB_ID} is no longer running; exiting"
                exit 0
            fi
            write_fleet
        fi

        sleep 2
    done
}

# --------------------------------------------------------------------------
# launch: one server attempt (any node count)
# --------------------------------------------------------------------------
LAUNCH_NODES=()

cleanup_workers() {
    local signal node
    for signal in TERM KILL; do
        for node in "${LAUNCH_NODES[@]}"; do
            ssh "${node}" "
                pkill -${signal} -f '[t]ensorrt_llm.llmapi.mgmn_' || true
                pkill -${signal} -f '[t]rtllm-llmapi-launch.*${CFG_PROC_PATTERN}' || true
                pkill -${signal} -f '[t]rtllm-serve.*${CFG_PROC_PATTERN}' || true
                pkill -${signal} -f '[t]rtllm-serve disaggregated.*${CFG_NAME}' || true
            " || true
        done
        if [[ "${signal}" == "TERM" ]]; then
            sleep 5
        fi
    done
}

# printf %q quotes far more than a shell needs (commas included), which turns a
# long srun line into noise. Quote only what actually requires it.
shell_quote() {
    if [[ "$1" =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then
        printf '%s' "$1"
    else
        printf '%q' "$1"
    fi
}

# One flag per line, each keeping its value: the record is meant to be read as
# well as replayed.
format_cmd() {
    local arg line=""
    for arg in "$@"; do
        if [[ -z "${line}" ]]; then
            line="$(shell_quote "${arg}")"
        elif [[ "${arg}" == -* ]]; then
            printf '%s \\\n    ' "${line}"
            line="$(shell_quote "${arg}")"
        else
            line+=" $(shell_quote "${arg}")"
        fi
    done
    printf '%s\n' "${line}"
}

cmd_launch() {
    parse_args "$@"
    reject_control_flags
    [[ -n "${ARG_ATTEMPT_DIR}" ]] || die "--attempt-dir is required"
    : "${SLURM_JOB_ID:?serve.sh launch must execute inside a Slurm allocation}"
    load_config

    local attempt_dir="${ARG_ATTEMPT_DIR}"
    local run_dir nodelist
    run_dir="$(dirname "${attempt_dir}")"
    mapfile -t LAUNCH_NODES < "${run_dir}/control/nodes"
    nodelist="$(IFS=,; echo "${LAUNCH_NODES[*]}")"

    local config_file="${attempt_dir}/server_config.yaml"
    local container_name="${CFG_NAME}-${SLURM_JOB_ID}"

    # Upstream retired the pull-based /perf_metrics endpoint in favour of a
    # writer that appends JSONL from inside each serving process, so the
    # directory has to be named in the engine config rather than polled. Only
    # this attempt knows its own path, which is why it is appended to the
    # snapshot instead of living in the checked-in config.
    #
    # Appended here rather than beside the cp in start_attempt so that a
    # restart is enough to pick the change up: the controller is a long-lived
    # bash process still running whichever serve.sh it started with, while
    # launch is re-exec'd per attempt and always reads the current file.
    # Idempotent because a relaunch onto an existing attempt directory must not
    # leave two keys behind -- the later one would silently win.
    local perf_dir="${attempt_dir}/perf_metrics"
    local cfg
    mkdir -p "${perf_dir}"
    for cfg in "${attempt_dir}"/{ctx,gen,server}_config.yaml; do
        [[ -f "${cfg}" ]] || continue
        grep -q '^perf_metrics_output_dir:' "${cfg}" && continue
        # The leading newline guards a source config with no trailing one.
        printf '\nperf_metrics_output_dir: %s\n' "${perf_dir}" >> "${cfg}"
    done
    # Capture records raw /v1/messages bodies, which is what you want while
    # bringing a model up and not what you want once the URL is shared: every
    # user's prompts would land in this run directory. Both paths follow the
    # attempt directory; server.capture in the deployment YAML turns them off.
    local export_env="ALL,${CFG_ENV}"
    if [[ "${CFG_CAPTURE}" == "1" ]]; then
        # This checkout's request-trace dump. One variable, one directory:
        # tensorrt_llm/serve/request_trace.py writes <UTC hour>/requests-<pid>.jsonl
        # at handler entry and <UTC hour>/responses-<pid>.jsonl when the response
        # ends, e.g. 2026-09-03T14/. The client session is a field on each line,
        # not a directory, so grouping a conversation is a jq filter rather than
        # a listing; unattributable requests carry session "_no_session".
        #
        # The pid in the name matters here specifically: this variable goes into
        # export_env, so every rank of every srun below inherits it, and the hour
        # bucket is the same for all of them. Without the pid two nodes would
        # append to one file and splice each other's JSON lines.
        #
        # Both frontends are wired to it -- openai_disagg_server.py for the proxy
        # the clients actually reach, openai_server.py for each worker -- and it
        # stays inert while the variable is unset.
        #
        # It replaces TRTLLM_ANTHROPIC_AUDIT_LOG, TRTLLM_ANTHROPIC_BENCH_CAPTURE_DIR,
        # TRTLLM_OPENAI_BENCH_CAPTURE_DIR and TRTLLM_TOOL_PARSE_TRACE, which
        # this script used to set. Nothing in this checkout reads any of those
        # four, so they produced no files and no error -- the capture banner
        # printed and the run recorded nothing.
        #
        # Same exposure as the variables it replaces: the directory holds other
        # people's prompts once the URL is shared, so it stays gated on
        # server.capture rather than following the run unconditionally.
        export_env+=",TRTLLM_REQUEST_TRACE_DIR=${attempt_dir}/request_trace"
    fi
    # Attention-DP routing decisions, one JSON line per batch that routed
    # something. Content-free -- request ids, token counts and per-rank prefix
    # match lengths, no prompt text -- so unlike the request trace it is not
    # gated on server.capture and stays on once the URL is shared. Rank 0 is
    # the only writer; see PyExecutor._emit_route_trace.
    export_env+=",TRTLLM_ADP_ROUTE_TRACE=${attempt_dir}/adp_route_trace.jsonl"

    trap cleanup_workers EXIT
    trap 'exit 0' INT TERM
    cleanup_workers

    # Split so the disaggregated path can place each worker itself: everything
    # here is allocation-wide, and the node placement is appended per srun.
    local common_base=(
        -l
        --jobid "${SLURM_JOB_ID}"
        --container-image "${CFG_IMAGE}"
        --container-name "${container_name}"
        --container-mounts "${CFG_MOUNTS}"
        --no-container-mount-home
        --mpi=pmix
        --overlap
    )
    local common=(
        "${common_base[@]}"
        --nodelist "${nodelist}"
        --nodes "${CFG_NODES}"
    )

    # --export=ALL would carry the caller's GPU visibility into the allocation.
    # If the controller was started from another node, those UUIDs do not exist
    # here and enroot fails with "unknown device"; the ranks pick their own GPU
    # from SLURM_LOCALID anyway.
    # CFG_ENV_MULTI holds the env entries whose values contain commas and so
    # cannot ride in --export; setting them here puts them in srun's own
    # environment, which --export=ALL carries into every task.
    #
    # The two SLURM_*_PER_TASK variables leak the same way and are fatal rather
    # than merely wrong. The gateway runs under --cpus-per-task 4, so its
    # environment carries SLURM_TRES_PER_TASK=cpu=4; sbatch's default
    # --export=ALL copies that into the serving job it submits, where Slurm also
    # sets SLURM_CPUS_PER_TASK=1 for an allocation that asked for no such thing.
    # srun then refuses to run at all -- "cpus-per-task set by two different
    # environment variables" -- and the controller retries forever because a
    # launch that never starts looks exactly like one that crashed. Job 608568
    # burned two GB300 nodes over 260 attempts this way. Only the gateway's
    # successors were affected; a job submitted by hand from a login shell has
    # neither variable set. Unsetting both lets srun take its cpus-per-task from
    # the allocation, which is what a hand submission does.
    local clean_env=(env -u NVIDIA_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES
                     -u SLURM_CPUS_PER_TASK -u SLURM_TRES_PER_TASK
                     ${CFG_ENV_MULTI[@]+"${CFG_ENV_MULTI[@]}"})

    local install_cmd=(
        "${clean_env[@]}"
        srun "${common[@]}"
        --ntasks "${CFG_NODES}"
        --ntasks-per-node 1
        bash -lc "cd '${CFG_REPO_DIR}' && python3 -m pip install -e ."
    )
    local serve_cmd=(
        "${clean_env[@]}"
        srun "${common[@]}"
        --ntasks "${CFG_WORLD_SIZE}"
        --ntasks-per-node "${CFG_TASKS_PER_NODE}"
        --export="${export_env}"
        bash -lc '
            export CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"
            model="$1"; port="$2"; config="$3"; numa_node="$4"; parser="$5"
            ucx_tls="$6"; ucx_net="$7"
            shift 7
            # enroot environ.d sets UCX_TLS=tcp and UCX_NET_DEVICES=eth0 on the
            # GPU nodes after the srun environment; only an export here outranks
            # it. NET_DEVICES stays as the hook set it unless stated: the RoCE VFs
            # are IPv6-only and UCX cannot bind tcp on them (job 278099).
            if [[ -n "${ucx_tls}" ]]; then export UCX_TLS="${ucx_tls}"; else unset UCX_TLS; fi
            [[ -z "${ucx_net}" ]] || export UCX_NET_DEVICES="${ucx_net}"
            numa=()
            if [[ -n "${numa_node}" ]]; then
                numa=(numactl -m "${numa_node}")
            fi
            exec trtllm-llmapi-launch "${numa[@]}" \
                trtllm-serve "${model}" \
                --host "$(hostname)" \
                --port "${port}" \
                --config "${config}" \
                --tool_parser "${parser}" \
                "$@"
        ' _ "${CFG_MODEL_PATH}" "${CFG_PORT}" "${config_file}" "${CFG_NUMACTL}" "${CFG_TOOL_PARSER}" \
            "${CFG_UCX_TLS}" "${CFG_UCX_NET_DEVICES}" \
        ${CFG_SERVE_EXTRA_ARGS[@]+"${CFG_SERVE_EXTRA_ARGS[@]}"}
    )

    # Exactly what ran, quoted so it can be replayed by hand from this attempt.
    {
        echo "#!/usr/bin/env bash"
        echo "# ${CFG_NAME} ${SLURM_JOB_ID} $(basename "${attempt_dir}") on ${nodelist}"
        if [[ "${CFG_INSTALL_REPO}" == "1" ]]; then
            printf '\n# install:\n'
            format_cmd "${install_cmd[@]}"
        fi
        if [[ "${CFG_DISAGG}" == "1" ]]; then
            # serve_cmd describes the aggregated path and is not what runs
            # here; launch_disagg appends its own sruns to this file as it
            # starts them.
            printf '\n# serve (disaggregated):\n'
        else
            printf '\n# serve:\n'
            format_cmd "${serve_cmd[@]}"
        fi
    } > "${attempt_dir}/launch_cmd.sh"

    if [[ "${CFG_INSTALL_REPO}" == "1" ]]; then
        echo "installing $(git -C "${CFG_REPO_DIR}" branch --show-current) on ${nodelist}"
        "${install_cmd[@]}" |& tee "${attempt_dir}/install.log"
    fi

    echo "starting ${CFG_MODEL_KEY} at http://${LAUNCH_NODES[0]}:${CFG_PORT}"
    if [[ "${CFG_CAPTURE}" == "1" ]]; then
        echo "WARNING: Anthropic request capture is enabled under ${attempt_dir}"
    else
        echo "Anthropic request capture is disabled (server.capture: false)"
    fi
    if [[ "${CFG_DISAGG}" == "1" ]]; then
        launch_disagg "${attempt_dir}" "${export_env}" "${common_base[@]}"
        return
    fi
    "${serve_cmd[@]}" |& tee "${attempt_dir}/server.log"
}

# --------------------------------------------------------------------------
# disaggregated launch: one proxy plus one srun per worker instance
# --------------------------------------------------------------------------
#
# Unlike the aggregated path this is not a single srun. Every worker instance is
# its own MPI world -- its own `trtllm-llmapi-launch` -- so instances cannot
# share an srun, and the total is 1 + ctx_instances + gen_instances.
#
# Workers find the proxy through service discovery rather than a static url
# list, which matters for more than convenience: with `disagg_cluster` set,
# `/health` gates on ctx and gen worker counts (openai_disagg_service.is_ready
# -> disagg_auto_scaling.is_ready). Without it `is_ready()` returns True
# unconditionally, so the proxy would answer /health the moment it binds and
# the gateway would elect a backend that has no workers behind it.
launch_disagg() {
    local attempt_dir="$1"; shift
    local export_env="$1"; shift
    local common=("$@")

    # Node and port assignment happens up front, because the proxy's config has
    # to name every worker before any of them starts. serve.sh already owns
    # this information -- it placed the workers -- so the static url list costs
    # nothing here, and it buys the readiness semantics the gateway needs:
    # OpenAIDisaggregatedService.setup() blocks in the FastAPI lifespan on
    # _wait_for_all_servers_ready(), polling every worker's /health, so the
    # proxy does not answer at all until the whole deployment is up.
    #
    # Service discovery was the other option and is the wrong one here. The
    # built-in HTTP registry is rejected unless the cluster uri host and the
    # server host are both loopback (cluster_storage.py:33-43), which no
    # multi-node job can satisfy, and the etcd alternative adds a process, two
    # ports, a data directory that must not live on Lustre, and a component
    # whose death makes the deployment unhealthy -- to buy dynamic membership
    # that a fixed 1P1D topology never uses. Three of the four reference
    # launchers under examples/disaggregated use static urls; this follows
    # slurm/benchmark/submit.py.
    local proxy_node="${LAUNCH_NODES[0]}"
    local nodes_per_ctx=$((CFG_CTX_RANKS / CFG_TASKS_PER_NODE))
    local nodes_per_gen=$((CFG_GEN_RANKS / CFG_TASKS_PER_NODE))

    local ctx_urls=() gen_urls=() ctx_nodes=() gen_nodes=()
    local i port node_cursor=0 next_port=$((CFG_PORT + 1))
    for ((i = 0; i < CFG_CTX_INSTANCES; i++)); do
        ctx_nodes+=("$(IFS=,; echo "${LAUNCH_NODES[*]:node_cursor:nodes_per_ctx}")")
        ctx_urls+=("${LAUNCH_NODES[node_cursor]}:${next_port}")
        node_cursor=$((node_cursor + nodes_per_ctx)); next_port=$((next_port + 1))
    done
    for ((i = 0; i < CFG_GEN_INSTANCES; i++)); do
        gen_nodes+=("$(IFS=,; echo "${LAUNCH_NODES[*]:node_cursor:nodes_per_gen}")")
        gen_urls+=("${LAUNCH_NODES[node_cursor]}:${next_port}")
        node_cursor=$((node_cursor + nodes_per_gen)); next_port=$((next_port + 1))
    done

    # The proxy config is CFG_DISAGG_CONFIG -- router types, num_workers,
    # retry and schedule policy, all stated in YAML -- plus the four fields only
    # this function can know. Nothing about the proxy's behaviour is decided
    # here. Comments do not survive safe_dump, which is why the source of truth
    # is the deployment's file and this is a per-attempt artifact.
    #
    # perf_metrics_output_dir is set unconditionally, and enabling it on the
    # proxy is safe now: upstream replaced the pull-based /perf_metrics endpoint
    # with a writer that appends JSONL from inside each serving process, so the
    # controller no longer polls anything. The endpoint used to drain both
    # workers' queues on every GET and drop whatever failed to pair, which cost
    # ~99% of records because Claude Code streams and only non-streaming
    # responses paired. Workers now push their metrics back in response headers
    # (Server-Timing, X-TRTLLM-Step-Metrics) or an SSE event, and the proxy
    # files the joined ctx+gen record itself -- the pairing we used to redo
    # offline.
    CTX_URLS="${ctx_urls[*]}" GEN_URLS="${gen_urls[*]}" \
    python3 - "${CFG_DISAGG_CONFIG}" "${attempt_dir}/disagg_config.yaml" \
             "${proxy_node}" "${CFG_PORT}" "${attempt_dir}/perf_metrics" <<'PY'
import os
import sys

import yaml

template, out_path, hostname, port, metrics_dir = sys.argv[1:6]

with open(template) as handle:
    cfg = yaml.safe_load(handle) or {}
# Only a default: a proxy config is free to state it, and every deployment so
# far runs the PyTorch backend.
cfg.setdefault("backend", "pytorch")
cfg["hostname"] = hostname
cfg["port"] = int(port)
cfg["perf_metrics_output_dir"] = metrics_dir

for section_name, env_var in (("context_servers", "CTX_URLS"),
                              ("generation_servers", "GEN_URLS")):
    urls = os.environ[env_var].split()
    section = cfg.setdefault(section_name, {})
    # num_instances is re-stated rather than trusted: the resolver already made
    # the two agree, and trtllm-serve rejects a list whose length differs.
    section["num_instances"] = len(urls)
    section["urls"] = urls

with open(out_path, "w") as handle:
    yaml.safe_dump(cfg, handle, default_flow_style=False, sort_keys=False)
PY

    local pids=() names=()
    {
        printf '# proxy   %s:%s  start/request timeout %ss\n' \
            "${proxy_node}" "${CFG_PORT}" "${CFG_DISAGG_TIMEOUT}"
        printf '#   trtllm-serve disaggregated -c %s -t %s -r %s\n' \
            "${attempt_dir}/disagg_config.yaml" \
            "${CFG_DISAGG_TIMEOUT}" "${CFG_DISAGG_TIMEOUT}"
    } >> "${attempt_dir}/launch_cmd.sh"

    # -t must cover the slowest worker's model load, because the proxy spends
    # that whole time in _wait_for_all_servers_ready; the 180s default would
    # abort long before a large checkpoint finishes. -r matches it, following
    # slurm/benchmark/start_server.sh, which uses 7200 for both.
    "${clean_env[@]}" srun "${common[@]}" \
        --nodelist "${proxy_node}" --nodes 1 --ntasks 1 \
        --export="${export_env}" \
        bash -lc "exec trtllm-serve disaggregated \
            -c '${attempt_dir}/disagg_config.yaml' \
            -t '${CFG_DISAGG_TIMEOUT}' -r '${CFG_DISAGG_TIMEOUT}'" \
        |& tee "${attempt_dir}/server.log" &
    pids+=($!); names+=("proxy")

    local role config ranks instances url worker_nodes
    for role in ctx gen; do
        if [[ "${role}" == "ctx" ]]; then
            config="${attempt_dir}/ctx_config.yaml"
            ranks="${CFG_CTX_RANKS}"; instances="${CFG_CTX_INSTANCES}"
        else
            config="${attempt_dir}/gen_config.yaml"
            ranks="${CFG_GEN_RANKS}"; instances="${CFG_GEN_INSTANCES}"
        fi
        for ((i = 0; i < instances; i++)); do
            if [[ "${role}" == "ctx" ]]; then
                url="${ctx_urls[i]}"; worker_nodes="${ctx_nodes[i]}"
            else
                url="${gen_urls[i]}"; worker_nodes="${gen_nodes[i]}"
            fi
            port="${url##*:}"
            # No --server_role and no --disagg_cluster_uri: with a static url
            # list the proxy already knows which workers are context and which
            # are generation, and the reference simple_example passes neither.
            #
            # --tool_parser is generation-only. It configures the OpenAI HTTP
            # server rather than TorchLlmArgs, and the proxy drives context
            # workers with stream=False while keeping nothing but their
            # disaggregated_params, so a parser there would only ever see
            # deliberately truncated output -- the exact shape that used to
            # raise "Incomplete DSML invoke".
            local parser_args=()
            if [[ "${role}" == "gen" && -n "${CFG_TOOL_PARSER}" ]]; then
                parser_args=(--tool_parser "${CFG_TOOL_PARSER}")
            fi
            # Each worker is its own engine with its own rank 0, so both would
            # otherwise append to the one adp_route_trace.jsonl that cmd_launch
            # put in export_env and the two streams would interleave. Give each
            # its own file; the aggregated path keeps the undecorated name.
            local worker_env="${export_env//adp_route_trace.jsonl/adp_route_trace-${role}-${i}.jsonl}"
            worker_env="${worker_env//tool_parse_trace.jsonl/tool_parse_trace-${role}-${i}.jsonl}"
            "${clean_env[@]}" srun "${common[@]}" \
                --nodelist "${worker_nodes}" \
                --nodes "$([[ ${role} == ctx ]] && echo "${nodes_per_ctx}" || echo "${nodes_per_gen}")" \
                --ntasks "${ranks}" --ntasks-per-node "${CFG_TASKS_PER_NODE}" \
                --export="${worker_env}" \
                bash -lc '
                    export CUDA_VISIBLE_DEVICES="${SLURM_LOCALID}"
                    model="$1"; port="$2"; config="$3"; numa_node="$4"
                    ucx_tls="$5"; ucx_net="$6"
                    shift 6
                    # enroot environ.d sets UCX_TLS=tcp and UCX_NET_DEVICES=eth0 on
                    # the GPU nodes after the srun environment; only an export here
                    # outranks it. NET_DEVICES stays as the hook set it unless stated:
                    # the RoCE VFs are IPv6-only and UCX cannot bind tcp on them
                    # (job 278099). Pinning UCX_TLS once cost NIXL its CUDA support
                    # (registerMemory abort, job 513029); a deployment stating it
                    # opts into that.
                    if [[ -n "${ucx_tls}" ]]; then export UCX_TLS="${ucx_tls}"; else unset UCX_TLS; fi
                    [[ -z "${ucx_net}" ]] || export UCX_NET_DEVICES="${ucx_net}"
                    numa=()
                    if [[ -n "${numa_node}" ]]; then
                        numa=(numactl -m "${numa_node}")
                    fi
                    exec trtllm-llmapi-launch "${numa[@]}" \
                        trtllm-serve "${model}" \
                        --host "$(hostname)" \
                        --port "${port}" \
                        --config "${config}" \
                        "$@"
                ' _ "${CFG_MODEL_PATH}" "${port}" "${config}" "${CFG_NUMACTL}" \
                    "${CFG_UCX_TLS}" "${CFG_UCX_NET_DEVICES}" \
                ${parser_args[@]+"${parser_args[@]}"} \
                ${CFG_SERVE_EXTRA_ARGS[@]+"${CFG_SERVE_EXTRA_ARGS[@]}"} \
                |& tee "${attempt_dir}/${role}-${i}.log" &
            pids+=($!); names+=("${role}-${i}")
            echo "  ${role}-${i}: ${ranks} ranks on ${worker_nodes} -> ${url}"
            {
                printf '# %-7s %s  %s ranks  config=%s\n' \
                    "${role}-${i}" "${url}" "${ranks}" "$(basename "${config}")"
                printf '#   trtllm-llmapi-launch trtllm-serve %s --port %s --config %s %s\n' \
                    "${CFG_MODEL_PATH}" "${port}" "${config}" "${parser_args[*]}"
            } >> "${attempt_dir}/launch_cmd.sh"
        done
    done

    echo "disagg: proxy on ${proxy_node}:${CFG_PORT}, ${#pids[@]} sruns total"
    # Any component exiting makes the deployment incomplete, so surface it the
    # same way the aggregated path surfaces its single server dying: return,
    # let the trap tear the rest down, and let the controller decide.
    wait -n "${pids[@]}"
    local rc=$?
    echo "a disagg component exited (rc=${rc}); tearing down the rest"
    return "${rc}"
}

# --------------------------------------------------------------------------
# gateway: the address users hold, in front of whatever backend is current
# --------------------------------------------------------------------------
gateway_submit() {
    local log_dir="${CFG_TRACE_ROOT}/_sbatch_logs"
    mkdir -p "${log_dir}"
    [[ -n "${CFG_GW_PARTITION}" ]] \
        || die "gateway.partition is required for --submit"

    # No GPU, no reservation: the gateway only proxies, and a reservation-bound
    # QoS would reject it on a CPU partition anyway.
    local sbatch_args=(
        --job-name "${CFG_NAME}-gateway"
        --account "${CFG_GW_ACCOUNT}"
        --partition "${CFG_GW_PARTITION}"
        --nodes 1
        --ntasks 1
        --cpus-per-task 4
        --time "${CFG_GW_TIME}"
        --output "${log_dir}/${CFG_NAME}-gateway-%j.out"
    )
    if [[ -n "${CFG_GW_QOS}" ]]; then
        sbatch_args+=(--qos "${CFG_GW_QOS}")
    fi
    sbatch_args+=(${CFG_GW_EXTRA_ARGS[@]+"${CFG_GW_EXTRA_ARGS[@]}"})

    echo "gateway: ${CFG_GW_PARTITION} for ${CFG_GW_TIME} on port ${CFG_GW_PORT}"
    echo "URL lands in ${CFG_FLEET_DIR}/gateway_url once it starts"
    sbatch "${sbatch_args[@]}" "${CFG_SERVE_SH}" gateway --yaml "${ARG_YAML}"
}

# The gateway holds no GPUs until asked. Two POST endpoints decide that, and
# curl against them works just as well -- this exists so that driving a gateway
# uses the same `--yaml` the rest of this script does, instead of making
# somebody look the published address up by hand first.
# One curl, with the response body and the status code kept apart. Writes
# GW_BODY and GW_CODE; the return status says only whether curl itself ran, so
# an HTTP error still leaves its body -- which carries the reason -- readable.
GW_BODY=""
GW_CODE=""
gateway_curl() {
    local timeout="$1" response
    shift
    # Command substitution strips trailing newlines, and %{http_code} ends
    # without one, so the code is always the last line however the body ends.
    response="$(curl --silent --show-error --max-time "${timeout}" \
                     --write-out $'\n%{http_code}' "$@")" || return 1
    GW_CODE="${response##*$'\n'}"
    GW_BODY="${response%$'\n'*}"
    return 0
}

# The gateway holds no GPUs until asked. Two POST endpoints decide that, and
# curl against them works just as well -- this exists so that driving a gateway
# uses the same `--yaml` the rest of this script does, instead of making
# somebody look the published address up by hand first.
gateway_control() {
    local action="$1" url url_file seen
    url_file="${CFG_FLEET_DIR}/gateway_url"
    [[ -f "${url_file}" ]] || die "no gateway has published an address in ${url_file}
start one with: serve.sh gateway --yaml ${ARG_YAML} --submit"
    url="$(head -1 "${url_file}")"
    [[ -n "${url}" ]] || die "empty gateway address in ${url_file}"

    # gateway_url outlives the gateway that wrote it, and a pinned hostname can
    # come back hosting somebody else's. Releasing the wrong deployment's GPUs
    # is not a mistake worth being able to make, so ask who is answering before
    # sending anything that changes state. Skipped for --status, which changes
    # nothing and is the thing you reach for when you suspect the file is stale.
    if [[ "${action}" != "status" ]]; then
        gateway_curl 20 "${url}/_gateway/health" \
            || die "cannot reach the gateway at ${url}
that address came from ${url_file} and may be stale; check the gateway job:
  squeue -u ${USER:-$(id -un)} -n ${CFG_NAME}-gateway"
        [[ "${GW_CODE}" == 2?? ]] \
            || die "the gateway at ${url} answered HTTP ${GW_CODE} on /_gateway/health
refusing to ${action} anything until it says which deployment it serves"
        seen="$(printf '%s' "${GW_BODY}" \
                | sed -n 's/.*"deployment": *"\([^"]*\)".*/\1/p')"
        [[ -n "${seen}" ]] \
            || die "the gateway at ${url} did not name a deployment in /_gateway/health
it predates this check; upgrade it, or use curl -XPOST ${url}/_gateway/${action}_server"
        [[ "${seen}" == "${CFG_NAME}" ]] \
            || die "${url} is the gateway for '${seen}', not '${CFG_NAME}'
${url_file} is stale. Refusing to ${action} another deployment's serving jobs."
    fi

    # Long enough for stop to walk every backend: each one is released and then
    # watched until Slurm confirms the nodes came back.
    case "${action}" in
        start)  gateway_curl 180 --request POST "${url}/_gateway/start_server" ;;
        stop)   gateway_curl 180 --request POST "${url}/_gateway/stop_server" ;;
        status) gateway_curl 30 "${url}/_gateway/health" ;;
    esac || die "cannot reach the gateway at ${url}
check that its Slurm job is still running: squeue -u ${USER:-$(id -un)} -n ${CFG_NAME}-gateway"

    printf '%s\n' "${GW_BODY}"
    [[ "${GW_CODE}" == 2?? ]] || die "gateway answered HTTP ${GW_CODE}"
}

cmd_gateway() {
    parse_args "$@"
    load_config

    # Driving a gateway that is already running needs neither the gateway
    # script nor the users file, only the address that gateway published.
    if [[ -n "${ARG_CONTROL}" ]]; then
        gateway_control "${ARG_CONTROL}"
        return
    fi

    [[ -f "${CFG_GW_SCRIPT}" ]] || die "gateway script not found: ${CFG_GW_SCRIPT}"
    [[ -f "${CFG_GW_USERS}" ]] || die "users file not found: ${CFG_GW_USERS}
create it with one username per line; every request is checked against it"

    if [[ -n "${ARG_SUBMIT}" ]]; then
        gateway_submit
        return
    fi

    mkdir -p "${CFG_FLEET_DIR}"
    local url="http://$(hostname):${CFG_GW_PORT}"
    # The whole point is that this address outlives the serving jobs, so record
    # it somewhere findable instead of only in an sbatch log.
    echo "${url}" > "${CFG_FLEET_DIR}/gateway_url"

    echo "gateway for ${CFG_NAME}"
    echo "  ANTHROPIC_BASE_URL=${url}"
    echo "  users: ${CFG_GW_USERS}"
    echo "  fleet: ${CFG_FLEET_DIR}"
    echo "  idle: no serving job until 'serve.sh gateway --yaml ${ARG_YAML} --start'"
    exec python3 "${CFG_GW_SCRIPT}" \
        --fleet-dir "${CFG_FLEET_DIR}" \
        --users "${CFG_GW_USERS}" \
        --yaml "${ARG_YAML}" \
        --serve-sh "${CFG_SERVE_SH}" \
        --port "${CFG_GW_PORT}" \
        --lead-time "${CFG_GW_LEAD_TIME}"
}

# --------------------------------------------------------------------------
# control actions against a live run directory
# --------------------------------------------------------------------------
cmd_control() {
    local action="$1"
    local run_dir="${2:?usage: serve.sh ${1} RUN_DIR}"
    local control_dir="${run_dir}/control"
    [[ -d "${control_dir}" ]] || die "controller is not ready: ${control_dir}"

    case "${action}" in
        start|restart|stop|quit)
            touch "${control_dir}/${action}"
            echo "requested ${action} for Slurm job $(cat "${control_dir}/job_id")"
            ;;
        status)
            echo "job_id=$(cat "${control_dir}/job_id" 2>/dev/null || echo unavailable)"
            echo "nodes=$(paste -sd, "${control_dir}/nodes" 2>/dev/null || echo unavailable)"
            echo "attempt=$(cat "${control_dir}/attempt" 2>/dev/null || echo 0)"
            echo "state=$(cat "${control_dir}/state" 2>/dev/null || echo initializing)"
            echo "server_url=$(cat "${run_dir}/server_url" 2>/dev/null || echo unavailable)"
            echo "current_attempt_dir=$(cat "${control_dir}/current_attempt_dir" 2>/dev/null || echo unavailable)"
            ;;
    esac
}

main() {
    local command="${1:-}"
    [[ -n "${command}" ]] || { usage; exit 2; }
    shift
    case "${command}" in
        submit) cmd_submit "$@" ;;
        run) cmd_run "$@" ;;
        launch) cmd_launch "$@" ;;
        gateway) cmd_gateway "$@" ;;
        start|restart|stop|quit|status) cmd_control "${command}" "$@" ;;
        -h|--help|help) usage ;;
        *) usage; die "unknown command: ${command}" ;;
    esac
}

main "$@"
