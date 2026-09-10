# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One hour of one instance, reduced to something a dashboard can keep.

report.py writes CSVs and an HTML page for a person to read. This writes the
same hour as a small JSON object for a program to collect: a few hundred
kilobytes instead of a few hundred megabytes, so a dashboard can hold weeks of
them and assemble a rolling view without ever re-reading a trace.

The one thing that does not survive aggregation is a percentile. p99 of two
hours is not a function of each hour's p99, so every latency metric also
carries a bounded, evenly-spaced sample of its own values -- merge the samples,
then take the percentile. That is the same trick the dashboard already uses to
merge per-minute latency buckets, and it is honest about being approximate
rather than quietly wrong.
"""
from __future__ import annotations

import time
from collections import Counter

from common import stats

SCHEMA = 1

# Latency and size columns worth a percentile. Kept short on purpose: every
# entry costs a sample array in every hour a dashboard retains.
REQUEST_METRICS = (
    "ttft_ms", "e2e_ms", "decode_ms", "parse_ms", "proxy_dispatch_ms",
    "prefill_queue_ms", "prefill_ms", "gpu_prefill_ms",
    "kv_transfer_ms", "gen_decode_ms",
    "isl_total", "isl_cached", "isl_new", "osl", "cache_hit_ratio",
)
SAMPLE_BUDGET = 512


def sample(values, budget: int = SAMPLE_BUDGET) -> list[float]:
    """Up to `budget` values, evenly spaced through the sorted order.

    Sorted first so the sample spans the distribution rather than the hour: an
    incident in the last ten minutes should still be visible in a merge, and
    taking every Nth arrival would keep whichever minutes happened to be
    sampled instead.
    """
    kept = sorted(v for v in values if v is not None)
    if len(kept) <= budget:
        return kept
    step = len(kept) / budget
    return [kept[min(len(kept) - 1, int(i * step))] for i in range(budget)]


def _mean(values):
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


def last_per_worker(iters: list[dict], column: str) -> list:
    """The column's value on each worker's final iteration.

    Cumulative counters answer for a whole engine lifetime, so the reading that
    matters is the last one -- per worker, since these rows are six prefill
    workers interleaved and the last row overall belongs to just one of them.
    """
    last: dict = {}
    for row in iters:
        last[row["worker"]] = row.get(column)
    return list(last.values())


def engine_side(iters: list[dict], ranks: list[dict], role: str) -> dict:
    """What a dashboard plots for one side of the fleet, without the rows."""
    busy_key = "has_decode" if role == "gen" else "has_prefill"
    busy = [r for r in iters if r.get(busy_key)]
    out = {
        "iterations": len(iters),
        "busy_iterations": len(busy),
        "rank_rows": len(ranks),
        "pads_removed": sum(1 for r in ranks if r.get("is_adp_pad")),
        "workers": sorted({r["worker"] for r in iters}),
        # Cumulative over the engine's life, so it is read off each worker's
        # last iteration -- not the last row overall, which belongs to whichever
        # of six prefill workers happened to sort last.
        "kv_hit_rate_cum": _mean(last_per_worker(iters, "kv_hit_rate_cum")),
        "kv_cache_util_mean": _mean(r.get("kv_cache_util_mean") for r in busy),
        "kv_pool_filled_mean": _mean(r.get("kv_pool_filled_mean") for r in busy),
        "host_step_ms_mean": _mean(r.get("host_step_ms_mean") for r in busy),
        "device_step_ms_mean": _mean(r.get("device_step_ms_mean") for r in busy),
    }
    if role == "gen":
        out.update({
            "batch_occupancy_mean": _mean(r.get("batch_occupancy") for r in busy),
            "tokens_per_request_mean": _mean(r.get("tokens_per_request") for r in busy),
            "decode_requests_rank_skew_mean": _mean(r.get("decode_requests_rank_skew") for r in busy),
            "batch_occupancy_sample": sample(r.get("batch_occupancy") for r in busy),
            "tokens_per_request_sample": sample(r.get("tokens_per_request") for r in busy),
        })
    else:
        out.update({
            "token_budget_util_mean": _mean(r.get("token_budget_util") for r in busy),
            "ctx_tokens_rank_skew_mean": _mean(r.get("ctx_tokens_rank_skew") for r in busy),
            "token_budget_util_sample": sample(r.get("token_budget_util") for r in busy),
            "ctx_tokens_rank_skew_sample": sample(r.get("ctx_tokens_rank_skew") for r in busy),
            # Per worker, because six prefill workers doing unequal work is the
            # thing this fleet is most likely to be doing without saying so.
            "by_worker": {
                worker: {
                    "iterations": sum(1 for r in iters if r["worker"] == worker),
                    "busy_iterations": sum(1 for r in busy if r["worker"] == worker),
                    # Mean over ranks, summed over iterations: the per-rank
                    # token total, which is what compares across workers of
                    # equal width. Not the worker's absolute token count.
                    "ctx_tokens_per_rank": sum(r.get("ctx_tokens_mean") or 0 for r in iters if r["worker"] == worker),
                }
                for worker in sorted({r["worker"] for r in iters})
            },
        })
    return out


def build_rollup(attempt, hour, window, requests, engine, budgets, notes) -> dict:
    """The whole hour as one JSON-serialisable object."""
    ok = [r for r in requests if r["status"] == "completed"]
    routes = sorted({r["route"] for r in requests if r.get("route")})
    return {
        "schema": SCHEMA,
        "run": attempt.parent.name,
        "attempt": attempt.name,
        "hour": hour,
        "window": {"from": window[0], "to": window[1]},
        "generated_at": time.time(),
        "budgets": budgets,
        "notes": notes,
        "requests": {
            "n": len(requests),
            "completed": len(ok),
            "status": dict(Counter(r["status"] for r in requests)),
            "by_route": {
                route: {
                    "n": sum(1 for r in requests if r.get("route") == route),
                    "completed": sum(1 for r in ok if r.get("route") == route),
                    "ttft_ms_p50": stats(r["ttft_ms"] for r in ok if r.get("route") == route)["p50"],
                    "e2e_ms_p50": stats(r["e2e_ms"] for r in ok if r.get("route") == route)["p50"],
                }
                for route in routes
            },
            "stats": {m: stats(r[m] for r in ok) for m in REQUEST_METRICS},
            # Percentiles do not merge; samples do. See the module docstring.
            "samples": {m: sample(r[m] for r in ok) for m in REQUEST_METRICS},
        },
        "engine": {
            "ctx": engine_side(engine["ctx_iters"], engine["ctx_rank"], "ctx"),
            "gen": engine_side(engine["gen_iters"], engine["gen_rank"], "gen"),
        },
    }
