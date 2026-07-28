---
name: audit-trace-analyze
description: Analyze an Anthropic audit trace from a run directory. Runs analyze_audit.py, always generates a run-summary Markdown table plus per-turn and pooled-distribution dashboards, then inspects turns with actual cache-hit ratio below 80% from captured request bodies. Use when the user asks to analyze a run, compare performance distributions, check cache reuse, or investigate low cache hit rates.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Audit trace analysis skill

**Skill base directory** (all relative paths below are relative to this):
`TensorRT-LLM/examples/serve/anthropic_compatibility/`

---

## Step 0 — Resolve the run directory

If the user named a run (e.g. "repairbot", "model bringup", a SLURM job ID, or
a substring), find the matching directory under `runs/`. If ambiguous, list
candidates and ask. You need:

- `<run_dir>/anthropic_audit.jsonl`   — required; abort if missing
- `<run_dir>/anthropic_message_capture/requests/`  — required for anomaly drill-down

Set:
```
SKILL_BASE = TensorRT-LLM/examples/serve/anthropic_compatibility
RUN_DIR    = <run_dir>           # e.g. runs/deepseek_v4_pro_manual_3291398-repairbot-6507081
ANALYSIS   = <RUN_DIR>/analysis
```

---

## Step 1 — Run the audit analyzer

```bash
cd <SKILL_BASE>
python3 analyze_audit.py <RUN_DIR>/anthropic_audit.jsonl --out <ANALYSIS>
```

This produces:
- `<ANALYSIS>/timeline.csv`    — one row per request, all metrics
- `<ANALYSIS>/turns.jsonl`     — same data as JSONL
- `<ANALYSIS>/tool_loops.jsonl`
- `<ANALYSIS>/REPORT.md`
- `<ANALYSIS>/TIMELINE.md`

Report the summary line printed by the script (e.g. "wrote 84 timeline rows…").

---

## Step 2 — Generate dashboards

```bash
cd <SKILL_BASE>
python3 plot_dashboard.py <ANALYSIS> --title "<run label>"
```

This writes `<ANALYSIS>/dashboard.png`. Tell the user the output path.

The dashboard contains 8 panels:
1. ISL breakdown (cached + new) — stacked bar
2. OSL (output tokens)
3. TTFT (time to first token, seconds)
4. Total server latency (seconds)
5. Actual cache-hit ratio — with reuse opportunity overlay; **turns < 80% are
   highlighted in red**
6. Decode TPS/user
7. Tool calls per turn
8. Tool loop gap (client-side tool execution time)

Each panel has a mean/p50/p75/p99 stats box in the top-right corner.

Always generate the pooled distribution dashboard:

```bash
cd <SKILL_BASE>
python3 plot_distributions.py \
  --series "<run label>=<ANALYSIS>/timeline.csv" \
  --out-dir <ANALYSIS> \
  --title "<run label>"
```

This writes:

- `<ANALYSIS>/distribution_dashboard.png`
- `<ANALYSIS>/distribution_dashboard.html`
- `<ANALYSIS>/run_summary.md`

`run_summary.md` is mandatory for every analysis. It contains one column per
`--series` run and includes:

- Sessions, API requests, completed turns, failed requests, and cancellations
- Trace elapsed time (`max(finished_at) - min(started_at)`) and summed request
  latency (`sum(server_total_ms)`)
- Total processed tokens, total input ISL, cache-read ISL, new/computed ISL,
  and model output tokens
- Warm cache-hit ratio, warm TTFT p50/p95, and decode TPS/user p50
- Total tool calls, total tool-call time, tool-loop gap p50/p95, and tool-result
  error rate

Define total tool-call time as the sum of one matched
response-to-next-request gap per tool-calling turn. Do not count parallel tool
calls emitted by the same turn as separate elapsed-time intervals.

The distribution dashboard pools completed turns across sessions, excludes
missing values rather than replacing them with zero, and shows a histogram plus
ECDF with mean/p50/p75/p95 for:

1. Total ISL
2. Cached ISL
3. New/uncached ISL
4. OSL
5. TTFT
6. Total server latency
7. Decode latency
8. TPS/user
9. Actual cache-hit ratio
10. Matched tool-loop gap

For comparisons, repeat `--series "LABEL=/path/to/timeline.csv"` for every run.
Do not split the distribution by session unless the user explicitly requests it.

---

## Step 3 — Find low-reuse turns (< 80% cache hit)

Read `<ANALYSIS>/timeline.csv` with Python. Collect all rows where
`actual_cache_hit_ratio < 0.80`, excluding turns where the low hit is expected:

**Expected-low criteria (skip these):**
- `session_turn_index == 1` — first turn of a session always starts cold
- `session_turn_index == 2` AND `previous_prompt_retention_ratio` is blank or 0
  — second turn before LCP is established

For each **unexpected** low-reuse turn, capture:
```python
{
  "global_request_index": ...,
  "session_id": ...,
  "session_turn_index": ...,
  "started_at": ...,
  "actual_cache_hit_ratio": ...,       # the anomalous value
  "isl_total": ...,
  "isl_cached": ...,
  "isl_new": ...,
  "previous_prompt_retention_ratio": ...,
  "current_reuse_opportunity_ratio": ...,
  "prompt_lcp_tokens": ...,
  "message_capture_file": ...,         # path inside RUN_DIR/anthropic_message_capture/
}
```

If there are no unexpected low-reuse turns, report that and skip Step 4.

---

## Step 4 — Drill into each anomalous request body

For each turn from Step 3, load its captured request:

```python
import gzip, json
body_path = f"<RUN_DIR>/anthropic_message_capture/{row['message_capture_file']}"
with gzip.open(body_path) as f:
    capture = json.load(f)
body = capture["body"]
```

From `body`, extract:

**System prompt fingerprint:**
```python
sys_blocks = body.get("system", [])
sys_text = " ".join(b.get("text", "") for b in sys_blocks if isinstance(b, dict))
sys_len = len(sys_text)
sys_first_100 = sys_text[:100]
```

**Message history shape:**
```python
messages = body.get("messages", [])
msg_count = len(messages)
# For each message, get role + content length
msg_shape = [(m["role"], sum(len(b.get("text","")) for b in (m["content"] if isinstance(m["content"], list) else [{"text": str(m["content"])}]))) for m in messages]
```

**Tool definitions count:**
```python
tool_count = len(body.get("tools", []))
```

**Context change vs. previous turn:**
Look at the turn BEFORE this one in the timeline (same session, `session_turn_index - 1`).
Compare `isl_total` values. The delta is `isl_total[this] - isl_total[prev]`.

Then classify the root cause using this decision tree:

| Condition | Root cause label |
|---|---|
| `previous_prompt_retention_ratio` is blank or near 0 | **CONTEXT_RESET** — session restarted or full context drop |
| `previous_prompt_retention_ratio` < 0.5 | **LARGE_CONTEXT_DROP** — >50% of prior prompt was dropped; truncation or summarization |
| `current_reuse_opportunity_ratio` < 0.5 | **LOW_OVERLAP** — this turn's input doesn't share a long prefix with the previous one (context diverged or system prompt changed) |
| `isl_new` is large (> 20k) relative to `isl_cached` | **LARGE_NEW_INJECTION** — a big block of new content was prepended (e.g. tool result, new system message) that broke the cached prefix |
| `actual_cache_hit_ratio` < `current_reuse_opportunity_ratio` × 0.7 | **CACHE_EVICTION** — the opportunity was there but the server evicted the KV cache entry (capacity pressure or TTL) |
| None of the above | **UNKNOWN** |

---

## Step 5 — Return the anomaly report

Output a structured report in this format:

```
## Audit Trace Analysis — <run label>

### Summary
- Turns analyzed: <N>
- Sessions: <K>
- Low-reuse turns (< 80%, unexpected): <M>
- Dashboard: <ANALYSIS>/dashboard.png
- Distribution dashboard: <ANALYSIS>/distribution_dashboard.html
- Run summary: <ANALYSIS>/run_summary.md

### Per-turn stats (across all turns)
| Metric | mean | p50 | p75 | p99 |
|--------|------|-----|-----|-----|
| ISL total (tokens) | … | … | … | … |
| OSL (tokens) | … | … | … | … |
| TTFT (s) | … | … | … | … |
| Total latency (s) | … | … | … | … |
| Cache hit ratio (%) | … | … | … | … |
| Decode TPS | … | … | … | … |
| Tool loop gap (s, non-zero) | … | … | … | … |

### Anomalous turns (cache hit < 80%, unexpected)

For each anomalous turn:

#### Turn <global_index> — Session <S#> turn <session_turn_index> — <root_cause_label>

- **Started at**: <started_at>
- **Cache hit**: <actual_cache_hit_ratio>  (opportunity: <current_reuse_opportunity_ratio>)
- **ISL**: <isl_total> total / <isl_cached> cached / <isl_new> new
- **Previous retention**: <previous_prompt_retention_ratio>
- **LCP tokens**: <prompt_lcp_tokens>
- **Root cause**: <label> — <one sentence explanation>
- **Evidence**: <what you observed in the request body — system prompt length, message count, isl delta, etc.>

### Observations & recommendations

<2–4 bullet points with cross-turn patterns, e.g.:
- "Cache evictions cluster around turns X–Y, suggesting KV cache pressure from concurrent sessions"
- "CONTEXT_RESET turns all follow a >30s gap — likely client timeout / reconnect"
- "Large tool results (tool_result_content_chars > 10k) correlate with LOW_OVERLAP turns"
>
```

Do not truncate anomalous turns — report all of them. If M > 10, group identical
root causes together and show one representative example per group.
