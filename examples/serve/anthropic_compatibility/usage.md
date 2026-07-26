<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro with Claude Code

This is the short path for running the TensorRT-LLM Anthropic Messages API with
DeepSeek-V4-Pro. It is based on the successful run recorded in
`runs/disagg_anthropic_messages_v4_pro_features_20260719/STATUS.md`:

- GitHub branch: `wanqian-nv/TensorRT-LLM:serli_anthropic_messages`
- deployment: aggregated PyTorch backend, 2 nodes x 4 GPUs
- parallelism: TP8 / EP8
- model name exposed to Claude Code: `DeepSeek-V4-Pro`

Despite the historical run-directory name, the successful server was
**aggregated**, not disaggregated.

## 1. Clone the tested branch

```bash
git clone --recursive --branch serli_anthropic_messages --single-branch \
  git@github.com:wanqian-nv/TensorRT-LLM.git
cd TensorRT-LLM
```

The recorded successful run used commit `1162a1b089`. The launch script installs
the current checkout with `python3 -m pip install -e .` on both allocated nodes.

## 2. Allocate nodes and start the server

From the repository root:

```bash
JOB_ID=$(sbatch --parsable \
  examples/serve/anthropic_compatibility/build_server.sh)
JOB_ID="${JOB_ID%%;*}"
RUN_DIR="$(pwd)/examples/serve/anthropic_compatibility/runs/deepseek_v4_pro_anthropic_${JOB_ID}"

echo "JOB_ID=${JOB_ID}"
echo "RUN_DIR=${RUN_DIR}"
```

`build_server.sh` requests two nodes and four GPUs per node, installs this Git
checkout in the known working container, removes stale DeepSeek-V4-Pro TRT-LLM
workers, and launches the first aggregated-server attempt. The Slurm allocation
stays alive if that attempt exits or is stopped, so later attempts reuse the
same nodes. Follow startup with:

```bash
until [[ -f "${RUN_DIR}/control/current_attempt_dir" ]]; do sleep 2; done
ATTEMPT_DIR=$(cat "${RUN_DIR}/control/current_attempt_dir")
tail -f "${ATTEMPT_DIR}/launcher.log"
```

The server URL is written as soon as the job starts:

```bash
SERVER_URL=$(cat "${RUN_DIR}/server_url")
until curl -fsS "${SERVER_URL}/health"; do sleep 10; done
```

Control the server without releasing the allocation:

```bash
CONTROL=examples/serve/anthropic_compatibility/server_control.sh

bash "${CONTROL}" "${RUN_DIR}" status
bash "${CONTROL}" "${RUN_DIR}" restart
bash "${CONTROL}" "${RUN_DIR}" stop
bash "${CONTROL}" "${RUN_DIR}" start
```

Each `start` or `restart` copies `${RUN_DIR}/agg_config.yaml` into a new
`attempt-NNN/` directory before launching. Edit the run-root copy between
attempts to test a different configuration while preserving every attempt's
exact config and logs. `stop` stops only the current server; `quit` stops the
server and releases the allocation.

## 3. Verify the Anthropic route

```bash
curl -sS "${SERVER_URL}/v1/messages" \
  -H 'content-type: application/json' \
  -H 'x-api-key: test' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "DeepSeek-V4-Pro",
    "max_tokens": 64,
    "stream": false,
    "messages": [
      {"role": "user", "content": "Reply with ROUTE_OK only."}
    ]
  }'
```

The request model is deliberately `DeepSeek-V4-Pro`; do not use a Claude model
alias for this benchmark.

## 4. Start Claude Code with DeepSeek-V4-Pro

```bash
export ANTHROPIC_BASE_URL="${SERVER_URL}"
export ANTHROPIC_AUTH_TOKEN=test
export ANTHROPIC_DEFAULT_OPUS_MODEL=DeepSeek-V4-Pro
export ANTHROPIC_DEFAULT_SONNET_MODEL=DeepSeek-V4-Pro
export ANTHROPIC_DEFAULT_HAIKU_MODEL=DeepSeek-V4-Pro

claude --model DeepSeek-V4-Pro
```

`ANTHROPIC_AUTH_TOKEN` is a dummy local token required by Claude Code; this
server does not validate it. The three `ANTHROPIC_DEFAULT_*_MODEL` variables
also map Claude Code's default model classes to the V4 model name. Keeping
`--model DeepSeek-V4-Pro` makes the tested model explicit.

## Per-request Anthropic audit log

The example launcher enables a content-free JSONL audit log for every received
`/v1/messages` request:

```bash
ATTEMPT_DIR=$(cat "${RUN_DIR}/control/current_attempt_dir")
tail -f "${ATTEMPT_DIR}/anthropic_audit.jsonl" | jq .
```

One line represents one actual inference request, completed only after the
non-stream response or the final streaming SSE event. It includes distinct
OpenAI response, engine, disaggregated, and context request IDs when available;
an explicitly supplied client session header; server-side time to first
semantic streaming delta; total endpoint duration; reported `input_tokens`,
`cache_read_input_tokens`, and `output_tokens`; emitted tool calls (`id`, name,
argument byte count); and received `tool_result` blocks (`tool_use_id`, error
bit, text length). It also distinguishes tool results in the full history from
those in the final inbound message.

The log intentionally excludes prompt text, tool arguments, and tool-result
contents. Claude Code does not provide a protocol-standard session ID, so its
local session is recorded only when a caller/proxy supplies one of
`x-claude-session-id`, `x-claude-code-session-id`, or `x-session-id`.

The example launcher also enables adjacent-prompt tracking with
`TRTLLM_ANTHROPIC_LCP_TRACKING=1`. It keeps only the previous prompt's token
IDs in process memory (up to 16 sessions) and writes only content-free metrics
to the audit:

- `prompt_lcp_tokens`;
- `previous_prompt_retention_ratio = prompt_lcp_tokens / previous_prompt_tokens`;
- `current_reuse_opportunity_ratio = prompt_lcp_tokens / lcp_prompt_tokens`.

The first turn has no previous prompt, so these fields are null. When Claude
Code does not send a session header, all requests use `unidentified-session`;
only run one serial client session in that mode. The comparison state resets
whenever the frontend server restarts.

For another deployment, enable the same writer by exporting an absolute path
before starting `trtllm-serve`:

```bash
export TRTLLM_ANTHROPIC_AUDIT_LOG=/path/to/anthropic_audit.jsonl
export TRTLLM_ANTHROPIC_LCP_TRACKING=1
```

Verify the token-counting endpoint independently of generation:

```bash
curl -sS "${SERVER_URL}/v1/messages/count_tokens" \
  -H 'content-type: application/json' \
  -d '{
    "model": "DeepSeek-V4-Pro",
    "system": "You are concise.",
    "messages": [{"role": "user", "content": "hello"}]
  }' | jq .
```

The standard endpoint and `/v1/messages` share conversion, message parsing,
chat-template application, and generator preprocessing. In disaggregated mode,
the frontend forwards count requests to a context worker so the aggregation
layer never approximates with a different tokenizer.

Build the content-free per-turn and tool-loop report with:

```bash
python examples/serve/anthropic_compatibility/analyze_audit.py \
  /path/to/anthropic_audit.jsonl \
  --out /path/to/audit-analysis
```

The analyzer writes `turns.jsonl`, `tool_loops.jsonl`, and `REPORT.md`. It
derives total ISL, cached/computed ISL, OSL, actual cache-hit ratio, adjacent
input LCP, both theoretical reuse ratios, input-only cache realization, and
the coarse server-observed gap between a completed tool-use response and the
next request carrying the matching `tool_result`. It does not contain prompt
content or Claude CLI-visible timing.

## Sensitive request capture for offline prompt analysis

The example launcher also enables full `/v1/messages` request capture under:

```bash
ATTEMPT_DIR=$(cat "${RUN_DIR}/control/current_attempt_dir")
CAPTURE_DIR="${ATTEMPT_DIR}/anthropic_message_capture"
```

Each ingress capture is stored as:

```text
anthropic_message_capture/
└── requests/
    └── <audit-request-id>.json.gz
```

The compressed JSON contains the complete parsed request body and raw ordered
HTTP header pairs, including credentials. Its `audit_request_id` matches the
record in `anthropic_audit.jsonl`; that audit record also contains the relative
`message_capture_file` path. Inspect one request with:

```bash
gzip -cd "${CAPTURE_DIR}/requests/<audit-request-id>.json.gz" | jq .
```

This capture is intentionally separate from the content-free audit and works
only when `TRTLLM_ANTHROPIC_BENCH_CAPTURE_DIR` is set. Request data is queued
before generation and compressed by a background writer. The capture directory
is mode `0700`, and each file is mode `0600`. It contains sensitive headers,
prompts, source code, tool arguments, and tool results; keep it under the
Git-ignored `runs/` tree and delete it after analysis when it is no longer
needed.

No prompt token IDs are written. Use the captured bodies to compare adjacent
turns' `system`, ordered `tools`, and `messages`, while the existing audit
analyzer supplies the corresponding LCP and cache metrics.

## 5. Run client-tool benchmarks

First test one real Bash tool loop:

```bash
claude -p 'Use the Bash tool exactly once to run pwd. Do not infer the path. Return only BASH_OK:<the exact stdout without the trailing newline>.' \
  --model DeepSeek-V4-Pro \
  --allowedTools Bash \
  --output-format stream-json \
  --verbose \
  --max-turns 6
```

For the Claude Code Skill fixture:

```bash
cd examples/serve/anthropic_compatibility

claude -p 'Use the Skill tool exactly once to invoke client-tool-sentinel. Do not answer from its description and do not use any other tool. After the skill finishes, return only the exact token required by the skill.' \
  --model DeepSeek-V4-Pro \
  --allowedTools Skill \
  --output-format stream-json \
  --verbose \
  --max-turns 6
```

The Skill test passes only when the trace contains a real `Skill` tool call and
the final output is `SKILL_OK_83D1`. See the
[capability matrix](capability_matrix.md#p0-04-client-tool-prompt-benchmarks)
for the Bash, text editor, computer, memory, MCP, and Skill prompt catalog and
their pass criteria.

## Server parameter definitions

| Parameter | Working value | Meaning |
|---|---:|---|
| Slurm nodes / GPUs | 2 / 8 | Four GPUs on each of two NVL72 nodes. |
| server mode | `AGG` | One aggregated server; no disaggregated frontend or NIXL transfer. |
| `tensor_parallel_size` | 8 | Shards the model across all eight GPUs. |
| `moe_expert_parallel_size` | 8 | Shards MoE experts across all eight GPUs. |
| `enable_attention_dp` | `false` | Attention data parallelism is disabled. |
| `max_seq_len` / `max_input_len` | 1048576 / 1048575 | Experimental 1M-token total and input sequence limits, matching the checkpoint metadata. |
| `max_num_tokens` | 8192 | Maximum scheduled tokens per iteration. |
| `max_batch_size` | 64 | Maximum active request batch size. |
| KV cache | FP8, 80% free GPU memory | Uses block reuse and a 40 GiB host cache. |
| `custom_tokenizer` | `deepseek_v4` | Uses the DeepSeek-V4 tokenizer adapter. |
| `reasoning_parser` | `deepseek_v4` | Parses DeepSeek-V4 reasoning output. |
| `--tool_parser` | `deepseek_v4` | Parses model-generated tool calls; supplied by `build_server.sh`. |
| MTP | disabled | The verified run did not use a draft model. |

The full values are in [agg_config.yaml](agg_config.yaml). To override only the
machine-specific paths or port:

```bash
sbatch --export=ALL,MODEL_PATH=/path/to/DeepSeek-V4-Pro,CONTAINER_IMAGE=/path/to/image.sqsh,RUN_DIR=/path/to/logs,PORT=8333 \
  examples/serve/anthropic_compatibility/build_server.sh
```

Stop the server while retaining the nodes, or release the allocation:

```bash
bash examples/serve/anthropic_compatibility/server_control.sh "${RUN_DIR}" stop
bash examples/serve/anthropic_compatibility/server_control.sh "${RUN_DIR}" quit
```

`scancel "${JOB_ID}"` remains available as the immediate fallback.
