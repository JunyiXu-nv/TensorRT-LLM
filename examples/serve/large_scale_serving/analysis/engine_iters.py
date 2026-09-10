# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-side iteration metrics from the worker logs (ctx-N.log / gen-N.log, or server.log).

Each worker prints one line per (rank, iteration):
    iter = 12, global_rank = 2, rank = 2, num_scheduled_requests = 1, ..., kv_hit_rate = 0.93,
    kv_reused_blocks = ..., kv_free_blocks = ..., host_step_time = 12.3ms, prev_device_step_time = 9.8ms,
    timestamp = 2026-09-09 06:46:55, states = {'num_ctx_tokens': 8192, 'num_generation_tokens': 0, ...}

Two grains are produced per worker:
  rank rows      one per (engine_instance, iter, rank)   -- what one GPU did
  instance rows  one per (engine_instance, iter)         -- the worker's ranks summed or averaged

`engine_instance` separates engine lifetimes inside one log: the KV-pool sizing dry run and the real
engine both count `iter` from 1, so rows are tagged by the restarts of that counter.

Attention-DP pad: a rank with no work is handed a dummy context request sized close to
max_num_tokens (exactly 8192 on older runs, 8190 with a speculative decoder configured), and it shows
up in `num_ctx_tokens` like real prefill. It is recognised by the four KV allocation counters
standing still since that rank's previous iteration (the dummy never commits blocks, a real chunk
always misses or reuses at least one) AND a size of at least PAD_MIN_FRACTION of the budget, which
keeps a sub-block real chunk from being mistaken for a pad. A pad counts as 0 tokens but stays in
the per-rank average, so utilization on 4 ranks with one busy rank reads (t + 0 + 0 + 0) / 4 /
max_num_tokens.

On a generation worker the idle rank is padded with one dummy decode request instead. It is
recognised by `num_scheduled_requests == 1` with `kv_cache_util == 0`: a real request keeps its KV
blocks pinned for its whole life, the dummy pins nothing. Its requests and tokens count as 0.

Cumulative log counters (kv_reused_blocks, kv_offload_blocks, ...) are differenced against the same
rank's previous iteration; `*_delta` is what moved in this iteration, `*_total` the counter itself.
"""
from __future__ import annotations

import ast
import re
import statistics
from collections import defaultdict
from datetime import tzinfo
from pathlib import Path

from common import parse_log_stamp, rank_skew, to_float

ITER_LINE = re.compile(r"\biter = \d+, global_rank = ")
PAD_COUNTERS = ("kv_reused_blocks", "kv_missed_blocks", "kv_alloc_total_blocks", "kv_alloc_new_blocks")
PAD_MIN_FRACTION = 0.9  # a pad is "about max_num_tokens"; the exact size differs between configs
TIER_COUNTERS = ("kv_offload_blocks", "kv_onboard_blocks", "kv_host_dropped_blocks")

RANK_COLUMNS = [
    "worker", "engine_instance", "iter", "rank", "timestamp",
    "scheduled_requests", "paused_requests",
    "ctx_tokens", "is_adp_pad", "ctx_tokens_real", "token_budget_util",
    "gen_tokens", "cached_kv_tokens", "decode_requests_real", "gen_tokens_real",
    "kv_hit_rate_iter", "kv_hit_rate_cum", "kv_reused_blocks_delta", "kv_missed_blocks_delta",
    "kv_reused_blocks_total", "kv_missed_blocks_total",
    "kv_cache_util", "kv_free_blocks", "kv_evictable_blocks", "kv_capacity_blocks", "kv_pool_filled_ratio",
    "kv_offload_blocks_delta", "kv_onboard_blocks_delta", "kv_host_dropped_blocks_delta",
    "kv_offload_blocks_total", "kv_onboard_blocks_total", "kv_host_dropped_blocks_total",
    "host_step_ms", "device_step_ms",
]
INSTANCE_COLUMNS = [
    "worker", "engine_instance", "iter", "timestamp", "ranks", "idle_ranks",
    "scheduled_requests_sum", "paused_requests_sum",
    # prefill (ctx workers)
    "busy_ranks", "has_prefill", "ctx_tokens_mean", "ctx_tokens_max", "token_budget_util", "ctx_tokens_rank_skew",
    # decode (gen workers)
    "has_decode", "decode_requests_sum", "batch_occupancy", "decode_requests_rank_skew",
    "gen_tokens_sum", "tokens_per_request", "cached_kv_tokens_sum",
    # KV cache
    "kv_hit_rate_iter", "kv_hit_rate_cum", "kv_cache_util_mean",
    "kv_free_blocks_sum", "kv_evictable_blocks_sum", "kv_capacity_blocks_sum",
    "kv_pool_filled_mean", "kv_pool_filled_rank_skew",
    "kv_offload_blocks_delta", "kv_onboard_blocks_delta", "kv_host_dropped_blocks_delta",
    # step time
    "host_step_ms_mean", "device_step_ms_mean", "device_step_rank_skew",
]
PREFILL_ONLY = ("busy_ranks", "has_prefill", "ctx_tokens_mean", "ctx_tokens_max", "token_budget_util", "ctx_tokens_rank_skew")
DECODE_ONLY = ("has_decode", "decode_requests_sum", "batch_occupancy", "decode_requests_rank_skew", "gen_tokens_sum",
               "tokens_per_request", "cached_kv_tokens_sum")


# ---------------------------------------------------------------- parsing
def parse_iter_log(path: Path, tz: tzinfo) -> list[dict]:
    """Raw per-(rank, iteration) entries of one worker log, tagged with engine_instance."""
    entries: list[dict] = []
    instance, last_iter_of_rank = 0, {}
    with path.open(encoding="latin-1", errors="replace") as handle:
        for line in handle:
            if not ITER_LINE.search(line):
                continue
            tail = line[line.find("iter = "):].rstrip()
            states: dict = {}
            marker = tail.find("states = ")
            if marker >= 0:
                try:
                    states = ast.literal_eval(tail[marker + len("states = "):])
                except (ValueError, SyntaxError):
                    states = {}
                tail = tail[:marker]
            fields = {}
            for chunk in tail.split(", "):
                key, sep, value = chunk.partition(" = ")
                if sep:
                    fields[key.strip()] = value.strip().rstrip(",")
            iteration, rank = int(fields["iter"]), int(fields.get("rank", -1))
            if iteration <= last_iter_of_rank.get(rank, 0):  # counter restarted: a new engine lifetime
                instance += 1
                last_iter_of_rank.clear()
            last_iter_of_rank[rank] = iteration
            entries.append({
                "engine_instance": instance, "iter": iteration, "rank": rank,
                "timestamp": parse_log_stamp(fields.get("timestamp"), tz),
                "scheduled_requests": int(fields.get("num_scheduled_requests", 0)),
                "paused_requests": int(to_float(fields.get("num_paused_requests")) or 0),
                "kv_cache_util": to_float(fields.get("kv_cache_util")),
                "kv_hit_rate_cum": to_float(fields.get("kv_hit_rate")),
                "kv_free_blocks": to_float(fields.get("kv_free_blocks")),
                "kv_evictable_blocks": to_float(fields.get("kv_evictable_blocks")),
                "host_step_ms": to_float(fields.get("host_step_time", "").rstrip("ms")),
                "device_step_ms": to_float(fields.get("prev_device_step_time", "").rstrip("ms")),
                "ctx_tokens": states.get("num_ctx_tokens"),
                "gen_tokens": states.get("num_generation_tokens"),
                "cached_kv_tokens": states.get("cached_kv_tokens"),
                **{key: to_float(fields.get(key)) for key in PAD_COUNTERS + TIER_COUNTERS},
            })
    return entries


# ---------------------------------------------------------------- per rank
def is_adp_pad(entry: dict, previous: dict | None, max_num_tokens: int | None) -> bool:
    """Near-budget chunk AND no KV counter moved since this rank's previous step (all zero on its first)."""
    if not max_num_tokens or (entry.get("ctx_tokens") or 0) < PAD_MIN_FRACTION * max_num_tokens:
        return False
    counters = [entry.get(key) for key in PAD_COUNTERS]
    if any(value is None for value in counters):
        return False
    if previous is None:
        return all(value == 0 for value in counters)
    return all(entry.get(key) == previous.get(key) for key in PAD_COUNTERS)


def is_gen_pad(entry: dict) -> bool:
    """Idle generation rank: exactly one scheduled request and no KV block pinned."""
    return entry["scheduled_requests"] == 1 and entry.get("kv_cache_util") == 0


def delta(entry: dict, previous: dict | None, key: str) -> float | None:
    if previous is None or entry.get(key) is None or previous.get(key) is None:
        return None
    return entry[key] - previous[key]


def ratio(top: float | None, bottom: float | None) -> float | None:
    return None if top is None or not bottom else top / bottom


def rank_rows(worker: str, role: str, entries: list[dict], max_num_tokens: int | None) -> list[dict]:
    """Per-rank rows with deltas and the pad flag. Capacity = peak of free + evictable per (instance, rank)."""
    entries = sorted(entries, key=lambda e: (e["engine_instance"], e["rank"], e["iter"]))
    capacity: dict[tuple, float] = {}
    for entry in entries:
        if entry["kv_free_blocks"] is not None and entry["kv_evictable_blocks"] is not None:
            key = (entry["engine_instance"], entry["rank"])
            capacity[key] = max(capacity.get(key, 0.0), entry["kv_free_blocks"] + entry["kv_evictable_blocks"])

    rows, previous_of_rank = [], {}
    for entry in entries:
        key = (entry["engine_instance"], entry["rank"])
        previous = previous_of_rank.get(key)
        pad = is_gen_pad(entry) if role == "gen" else is_adp_pad(entry, previous, max_num_tokens)
        real_tokens = 0 if pad else (entry["ctx_tokens"] or 0)
        reused, missed = delta(entry, previous, "kv_reused_blocks"), delta(entry, previous, "kv_missed_blocks")
        cap, free = capacity.get(key), entry["kv_free_blocks"]
        rows.append({
            "worker": worker, "engine_instance": entry["engine_instance"], "iter": entry["iter"],
            "rank": entry["rank"], "timestamp": entry["timestamp"],
            "scheduled_requests": entry["scheduled_requests"], "paused_requests": entry["paused_requests"],
            "ctx_tokens": entry["ctx_tokens"], "is_adp_pad": pad, "ctx_tokens_real": real_tokens,
            "token_budget_util": ratio(real_tokens, max_num_tokens),
            "gen_tokens": entry["gen_tokens"], "cached_kv_tokens": entry["cached_kv_tokens"],
            "decode_requests_real": 0 if pad else entry["scheduled_requests"],
            "gen_tokens_real": 0 if pad else (entry["gen_tokens"] or 0),
            "kv_hit_rate_iter": ratio(reused, (reused or 0) + (missed or 0)) if reused is not None and missed is not None else None,
            "kv_hit_rate_cum": entry["kv_hit_rate_cum"],
            "kv_reused_blocks_delta": reused, "kv_missed_blocks_delta": missed,
            "kv_reused_blocks_total": entry["kv_reused_blocks"], "kv_missed_blocks_total": entry["kv_missed_blocks"],
            "kv_cache_util": entry["kv_cache_util"],
            "kv_free_blocks": free, "kv_evictable_blocks": entry["kv_evictable_blocks"],
            "kv_capacity_blocks": cap,
            "kv_pool_filled_ratio": 1.0 - free / cap if cap and free is not None else None,
            **{f"{counter}_delta": delta(entry, previous, counter) for counter in TIER_COUNTERS},
            **{f"{counter}_total": entry.get(counter) for counter in TIER_COUNTERS},
            "host_step_ms": entry["host_step_ms"], "device_step_ms": entry["device_step_ms"],
        })
        previous_of_rank[key] = entry
    return rows


# ---------------------------------------------------------------- per instance
def _mean(values) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


def _sum(values) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) if kept else None


def instance_rows(rows: list[dict], role: str, max_num_tokens: int | None, max_batch_size: int | None) -> list[dict]:
    """Pool one worker's ranks per iteration. Prefill columns are filled for ctx workers, decode for gen."""
    by_iter: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_iter[(row["worker"], row["engine_instance"], row["iter"])].append(row)

    out = []
    for (worker, instance, iteration), ranks in sorted(by_iter.items()):
        tokens = [r["ctx_tokens_real"] for r in ranks]  # pads are 0 and stay in the denominator
        tokens_mean = statistics.fmean(tokens)
        batch = [r["decode_requests_real"] for r in ranks]  # idle-rank dummies are 0
        gen_tokens = sum(r["gen_tokens_real"] for r in ranks)
        reused, missed = _sum(r["kv_reused_blocks_delta"] for r in ranks), _sum(r["kv_missed_blocks_delta"] for r in ranks)
        cum_reused, cum_missed = _sum(r["kv_reused_blocks_total"] for r in ranks), _sum(r["kv_missed_blocks_total"] for r in ranks)
        row = {
            "worker": worker, "engine_instance": instance, "iter": iteration,
            "timestamp": ranks[0]["timestamp"], "ranks": len(ranks),
            "idle_ranks": sum(1 for r in ranks if r["is_adp_pad"]),
            "scheduled_requests_sum": sum(r["scheduled_requests"] for r in ranks),
            "paused_requests_sum": sum(r["paused_requests"] for r in ranks),
            "busy_ranks": sum(1 for t in tokens if t > 0), "has_prefill": max(tokens) > 0,
            "ctx_tokens_mean": tokens_mean, "ctx_tokens_max": max(tokens),
            "token_budget_util": ratio(tokens_mean, max_num_tokens),
            "ctx_tokens_rank_skew": rank_skew(tokens),
            "has_decode": sum(batch) > 0, "decode_requests_sum": sum(batch),
            "batch_occupancy": ratio(statistics.fmean(batch), max_batch_size),
            "decode_requests_rank_skew": rank_skew(batch),
            "gen_tokens_sum": gen_tokens,
            "tokens_per_request": ratio(gen_tokens, sum(batch)),
            "cached_kv_tokens_sum": _sum(r["cached_kv_tokens"] for r in ranks),
            "kv_hit_rate_iter": ratio(reused, (reused or 0) + (missed or 0)) if reused is not None and missed is not None else None,
            "kv_hit_rate_cum": ratio(cum_reused, (cum_reused or 0) + (cum_missed or 0)) if cum_reused is not None else None,
            "kv_cache_util_mean": _mean(r["kv_cache_util"] for r in ranks),
            "kv_free_blocks_sum": _sum(r["kv_free_blocks"] for r in ranks),
            "kv_evictable_blocks_sum": _sum(r["kv_evictable_blocks"] for r in ranks),
            "kv_capacity_blocks_sum": _sum(r["kv_capacity_blocks"] for r in ranks),
            "kv_pool_filled_mean": _mean(r["kv_pool_filled_ratio"] for r in ranks),
            "kv_pool_filled_rank_skew": rank_skew(r["kv_pool_filled_ratio"] for r in ranks),
            **{f"{counter}_delta": _sum(r[f"{counter}_delta"] for r in ranks) for counter in TIER_COUNTERS},
            "host_step_ms_mean": _mean(r["host_step_ms"] for r in ranks),
            "device_step_ms_mean": _mean(r["device_step_ms"] for r in ranks),
            "device_step_rank_skew": rank_skew(r["device_step_ms"] for r in ranks),
        }
        for key in DECODE_ONLY if role == "ctx" else PREFILL_ONLY if role == "gen" else ():
            row[key] = None  # a context-only worker has no decode batch, a generation worker no prefill chunk
        out.append(row)
    return out


# ---------------------------------------------------------------- driver
def worker_logs(attempt: Path) -> list[tuple[str, str, Path]]:
    """(worker name, role, path). Disaggregated runs name logs by role; an aggregated run has server.log."""
    logs = [(p.stem, "ctx", p) for p in sorted(attempt.glob("ctx-*.log"))]
    logs += [(p.stem, "gen", p) for p in sorted(attempt.glob("gen-*.log"))]
    if not logs and (attempt / "server.log").exists():
        logs = [("server", "mixed", attempt / "server.log")]
    return logs


def build_engine(attempt: Path, tz: tzinfo, window: tuple[float | None, float | None] = (None, None),
                 max_num_tokens: int | None = None, max_batch_size: int | None = None) -> dict:
    """Rank and instance rows of every worker. The window is applied after differencing, so the
    first surviving iteration is still a one-step delta and not the whole idle stretch before it."""
    result = {"ctx_rank": [], "ctx_iters": [], "gen_rank": [], "gen_iters": [], "workers": []}
    lo, hi = window
    for worker, role, path in worker_logs(attempt):
        entries = parse_iter_log(path, tz)
        ranks = rank_rows(worker, role, entries, None if role == "gen" else max_num_tokens)
        iters = instance_rows(ranks, role, max_num_tokens, max_batch_size)
        if lo is not None or hi is not None:
            ranks = [r for r in ranks if _inside(r["timestamp"], lo, hi)]
            iters = [r for r in iters if _inside(r["timestamp"], lo, hi)]
        bucket = "gen" if role == "gen" else "ctx"
        result[f"{bucket}_rank"] += ranks
        result[f"{bucket}_iters"] += iters
        result["workers"].append({"worker": worker, "role": role, "iter_lines": len(entries),
                                  "engine_instances": 1 + max((e["engine_instance"] for e in entries), default=0)})
    return result


def _inside(stamp: float | None, lo: float | None, hi: float | None) -> bool:
    if stamp is None:
        return False
    return (lo is None or stamp >= lo) and (hi is None or stamp <= hi)


if __name__ == "__main__":  # quick look: python3 engine_iters.py <attempt_dir>
    import sys
    from common import TIME_COLUMNS, local_tz, write_csv, yaml_scalar
    attempt = Path(sys.argv[1])
    budget = yaml_scalar([attempt / "ctx_config.yaml", attempt / "server_config.yaml"], "max_num_tokens")
    batch = yaml_scalar([attempt / "gen_config.yaml", attempt / "server_config.yaml"], "max_batch_size")
    engine = build_engine(attempt, local_tz(), max_num_tokens=budget, max_batch_size=batch)
    for name in ("ctx_rank", "ctx_iters", "gen_rank", "gen_iters"):
        cols = RANK_COLUMNS if name.endswith("rank") else INSTANCE_COLUMNS
        print(name, len(engine[name]), "->", write_csv(Path(f"{name}.csv"), engine[name], cols, TIME_COLUMNS))
    print(engine["workers"])
