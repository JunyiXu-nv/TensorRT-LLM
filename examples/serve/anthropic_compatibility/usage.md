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

## 2. Build and start the server

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
workers, and launches the aggregated server. Follow startup with:

```bash
tail -f "${RUN_DIR}/server.log"
```

The server URL is written as soon as the job starts:

```bash
SERVER_URL=$(cat "${RUN_DIR}/server_url")
until curl -fsS "${SERVER_URL}/health"; do sleep 10; done
```

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
| `max_seq_len` / `max_input_len` | 131072 / 131071 | Maximum total and input sequence lengths. |
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

Stop the server and release its allocation with `scancel "${JOB_ID}"`.
