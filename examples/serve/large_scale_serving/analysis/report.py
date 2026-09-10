#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the per-request table, the engine iteration tables and one HTML report for a serving run.

    python3 analysis/report.py <attempt_dir> [--hour 2026-09-09T13] [--out DIR]
    python3 analysis/report.py <attempt_dir>/request_trace/2026-09-09T13        # same as --hour

With --hour only the requests of that UTC hour are read, and the engine logs are cut to the window
[first request start, last request finish] of those requests. Without it the whole attempt is used.

Output (default: _reports/<run name>[_<hour>]/ beside this checkout):
    requests.csv        one row per request                      (requests_table.py)
    ctx_iters.csv       one row per (ctx worker, iteration), ranks pooled
    ctx_rank_iters.csv  one row per (ctx worker, iteration, rank)
    gen_iters.csv / gen_rank_iters.csv   the same for the generation worker
    REPORT.html         percentile tables, per-instance engine tables, charts
"""
from __future__ import annotations

import argparse
import html
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import charts
from common import TIME_COLUMNS, local_tz, stats, utc_iso, write_csv, yaml_scalar
from engine_iters import INSTANCE_COLUMNS, RANK_COLUMNS, TIER_COUNTERS, build_engine
from requests_table import COLUMNS as REQUEST_COLUMNS
from rollup import build_rollup
from requests_table import build_requests

REPORTS_ROOT = Path(__file__).resolve().parent.parent / "_reports"

LATENCY_METRICS = [  # (column, label)
    ("parse_ms", "Body parse before the handler (arrival to trace hook)"),
    ("proxy_dispatch_ms", "Proxy pre-dispatch (arrival to ctx dispatch)"),
    ("ttft_ms", "TTFT (proxy arrival to first token)"), ("e2e_ms", "E2E (arrival to response end)"),
    ("decode_ms", "Decode (E2E minus TTFT)"), ("prefill_queue_ms", "Prefill queue (ctx arrival to scheduled)"),
    ("prefill_ms", "Prefill (ctx scheduled to first token)"), ("gpu_prefill_ms", "GPU prefill forward (ctx)"),
    ("kv_transfer_ms", "KV transfer (gen)"), ("gen_decode_ms", "Decode on gen (first to last token)"),
]
TOKEN_METRICS = [
    ("isl_total", "ISL total", 0), ("isl_cached", "ISL cached", 0), ("isl_new", "ISL new", 0), ("osl", "OSL", 0),
    ("cache_hit_ratio", "Cache hit ratio (cached / total)", 3),
    ("ctx_blocks_new", "ctx KV blocks newly allocated", 0), ("ctx_blocks_reused", "ctx KV blocks reused", 0),
    ("kv_transfer_bytes", "KV transfer bytes", 0),
]


# ---------------------------------------------------------------- formatting
def esc(text) -> str:
    return html.escape(str(text))


def num(value, digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def dur(ms) -> str:
    if ms is None:
        return "—"
    return f"{ms / 1000:,.2f} s" if abs(ms) >= 1000 else f"{ms:,.0f} ms"


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def table(header: list[str], rows: list[list], caption: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def stat_rows(rows: list[dict], metrics, fmt) -> list[list]:
    out = []
    for column, label, *rest in metrics:
        digits = rest[0] if rest else 0
        s = stats(r.get(column) for r in rows)
        f = (lambda v: fmt(v)) if fmt is dur else (lambda v, d=digits: num(v, d))
        out.append([esc(label), s["n"], f(s["mean"]), f(s["p50"]), f(s["p90"]), f(s["p99"]), f(s["max"])])
    return out


def mean(values) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


def window_hit_rate(rank_rows: list[dict]) -> float | None:
    """Blocks reused / (reused + missed) over the rows given, from the per-iteration deltas."""
    reused = sum(r["kv_reused_blocks_delta"] or 0 for r in rank_rows)
    missed = sum(r["kv_missed_blocks_delta"] or 0 for r in rank_rows)
    return reused / (reused + missed) if reused + missed else None


# ---------------------------------------------------------------- sections
def requests_section(rows: list[dict]) -> str:
    parts = ["<h2>1 · Requests</h2>"]
    by_route = Counter((r["route"], r["status"]) for r in rows)
    parts.append(table(["route", "status", "requests"],
                       [[esc(route), esc(status), n] for (route, status), n in sorted(by_route.items())]))
    cols = ["metric", "n", "mean", "p50", "p90", "p99", "max"]
    done = [r for r in rows if r["status"] == "completed"]
    parts.append(f"<h3>Latency</h3><p class='sub'>Completed requests only ({len(done)} of {len(rows)}); "
                 "an errored or disconnected stream has a truncated E2E.</p>" + table(cols, stat_rows(done, LATENCY_METRICS, dur)))
    parts.append("<h3>Tokens and KV blocks</h3>" + table(cols, stat_rows(done, TOKEN_METRICS, num)))
    parts.append('<p class="sub">ISL / OSL / cached come from the usage block the client received. '
                 "On /v1/responses the server currently reports cached_tokens equal to the whole prompt, so "
                 "isl_cached, isl_new and cache_hit_ratio on that route are not trustworthy yet; "
                 "ctx_blocks_reused × tokens_per_block is the engine-side measurement.</p>")

    per_route = []
    for route in sorted({r["route"] for r in rows}):
        sub = [r for r in rows if r["route"] == route]
        ok = [r for r in sub if r["status"] == "completed"]
        per_route.append([esc(route), len(sub), len(ok), dur(stats(r["ttft_ms"] for r in ok)["p50"]),
                          dur(stats(r["e2e_ms"] for r in ok)["p50"]), num(stats(r["isl_total"] for r in ok)["p50"]),
                          num(stats(r["osl"] for r in ok)["p50"]), num(mean(r["cache_hit_ratio"] for r in ok), 3)])
    parts.append("<h3>By route (completed requests)</h3>" + table(
        ["route", "requests", "completed", "TTFT p50", "E2E p50", "ISL p50", "OSL p50", "cache hit mean"], per_route))

    per_instance = []
    for inst in sorted({r["ctx_instance"] for r in rows if r["ctx_instance"]}):
        sub = [r for r in rows if r["ctx_instance"] == inst]
        ranks = Counter(r["routed_rank"] for r in sub)
        per_instance.append([esc(inst), len(sub), " / ".join(str(ranks.get(k, 0)) for k in sorted(ranks)),
                             dur(stats(r["ttft_ms"] for r in sub)["p50"]), dur(stats(r["prefill_ms"] for r in sub)["p50"]),
                             num(sum(r["ctx_blocks_new"] or 0 for r in sub)), num(sum(r["ctx_blocks_reused"] or 0 for r in sub))])
    parts.append("<h3>By context instance (from the routing trace)</h3>" + table(
        ["ctx instance", "requests", "per rank", "TTFT p50", "prefill p50", "Σ ctx blocks new", "Σ ctx blocks reused"], per_instance))

    parts.append('<div class="grid-2">' + "".join([
        charts.histogram([r["isl_total"] for r in done], "ISL total", "tokens", log_x=True),
        charts.histogram([r["isl_new"] for r in done], "ISL new (usage)", "tokens", log_x=True),
        charts.histogram([r["osl"] for r in done], "OSL", "tokens", log_x=True),
        charts.histogram([r["ttft_ms"] for r in done], "TTFT", "ms", log_x=True),
        charts.histogram([r["e2e_ms"] for r in done], "E2E", "ms", log_x=True),
        charts.histogram([r["prefill_ms"] for r in done], "Prefill (ctx)", "ms", log_x=True),
    ]) + "</div>")
    return "".join(parts)


def last_instance(rows: list[dict], worker: str) -> list[dict]:
    """Rows of the engine that served traffic: the highest engine_instance seen for the worker."""
    mine = [r for r in rows if r["worker"] == worker]
    if not mine:
        return []
    last = max(r["engine_instance"] for r in mine)
    return [r for r in mine if r["engine_instance"] == last]


def prefill_section(iters: list[dict], ranks: list[dict], max_num_tokens: int | None) -> str:
    parts = ["<h2>2 · Prefill workers (ctx)</h2>",
             f'<p class="sub">Token budget max_num_tokens = {num(max_num_tokens)} per rank per iteration. '
             "Attention-DP pads count as 0 tokens but stay in every rank average. Means are taken over "
             "iterations with prefill work; skew is (max − mean) / mean across the ranks of one iteration "
             "and reads ranks − 1 when a single rank is busy. Hit rate (window) is Σ reused / Σ (reused + missed) "
             "blocks over the iterations shown; hit rate (cum) is the log's own running ratio since engine start.</p>"]
    workers = sorted({r["worker"] for r in iters})
    summary, per_rank = [], []
    for worker in workers:
        rows, rrows = last_instance(iters, worker), last_instance(ranks, worker)
        busy = [r for r in rows if r["has_prefill"]]
        pads = sum(1 for r in rrows if r["is_adp_pad"])
        summary.append([esc(worker), len(rows), len(busy), pct(pads / len(rrows)) if rrows else "—",
                        num(mean(r["token_budget_util"] for r in busy), 3), num(mean(r["busy_ranks"] for r in busy), 2),
                        num(mean(r["ctx_tokens_rank_skew"] for r in busy), 2),
                        num(window_hit_rate(rrows), 3), num(rows[-1]["kv_hit_rate_cum"] if rows else None, 3),
                        pct(max((r["kv_pool_filled_ratio"] for r in rrows if r["kv_pool_filled_ratio"] is not None), default=None)),
                        num(mean(r["kv_cache_util"] for r in rrows), 3),
                        *[num(sum(r[f"{c}_delta"] or 0 for r in rows)) for c in TIER_COUNTERS],
                        dur(mean(r["host_step_ms"] for r in rrows)), dur(mean(r["device_step_ms"] for r in rrows))])
        total_real = sum(r["ctx_tokens_real"] for r in rrows) or 1
        for rank in sorted({r["rank"] for r in rrows}):
            mine = [r for r in rrows if r["rank"] == rank]
            per_rank.append([esc(worker), rank, num(sum(r["ctx_tokens_real"] for r in mine)),
                             pct(sum(r["ctx_tokens_real"] for r in mine) / total_real),
                             sum(1 for r in mine if r["is_adp_pad"]), num(window_hit_rate(mine), 3), num(mine[-1]["kv_hit_rate_cum"], 3),
                             pct(mine[-1]["kv_pool_filled_ratio"]), num(mine[-1]["kv_capacity_blocks"])])
    parts.append(table(["worker", "iterations", "with prefill", "pad share", "budget util", "busy ranks", "rank skew",
                        "hit rate (window)", "hit rate (cum, end)", "pool filled peak", "kv_cache_util", "Δ offload", "Δ onboard", "Δ host dropped",
                        "host step", "device step"], summary, "one row per context worker, ranks pooled"))
    parts.append(table(["worker", "rank", "Σ real ctx tokens", "share", "pads", "hit rate (window)", "hit rate (cum, end)",
                        "pool filled (end)", "capacity blocks"], per_rank, "per rank: where the prefill tokens landed"))
    for i, worker in enumerate(workers):
        rows, rrows = last_instance(iters, worker), last_instance(ranks, worker)
        by_rank = defaultdict(list)
        for r in rrows:
            by_rank[r["rank"]].append(r)
        series = lambda key: [(f"rank {k}", [(r["timestamp"], r[key]) for r in v]) for k, v in sorted(by_rank.items())]
        parts.append(f'<details {"open" if i == 0 else ""}><summary>{esc(worker)} charts</summary><div class="grid-2">'
                     + charts.line_chart([("pooled", [(r["timestamp"], r["token_budget_util"]) for r in rows])],
                                         f"{worker}: token budget utilization", "share of max_num_tokens", y_max=1.0)
                     + charts.line_chart(series("kv_hit_rate_cum"), f"{worker}: KV hit rate (cumulative) per rank", "reused / (reused + missed)", y_max=1.0)
                     + charts.line_chart(series("kv_pool_filled_ratio"), f"{worker}: KV pool filled per rank", "1 − free / capacity", y_max=1.0)
                     + charts.line_chart([("pooled mean", [(r["timestamp"], r["device_step_ms_mean"]) for r in rows])],
                                         f"{worker}: device step time", "ms", unit=" ms")
                     + "</div></details>")
    return "".join(parts)


def decode_section(iters: list[dict], ranks: list[dict], max_batch_size: int | None) -> str:
    parts = ["<h2>3 · Generation worker (gen)</h2>",
             f'<p class="sub">An idle attention-DP rank carries one dummy decode request; it is dropped here '
             f"(idle rank = one scheduled request and kv_cache_util 0). Batch occupancy = real requests per rank / "
             f"max_batch_size ({num(max_batch_size)}). tokens_per_request = generated tokens per real request per "
             "iteration: 1 without speculative decoding, accepted draft length + 1 with it. Means are over "
             "iterations with at least one real request.</p>"]
    for worker in sorted({r["worker"] for r in iters}):
        rows, rrows = last_instance(iters, worker), last_instance(ranks, worker)
        busy = [r for r in rows if r["has_decode"]]
        idle_share = sum(r["idle_ranks"] for r in rows) / sum(r["ranks"] for r in rows) if rows else None
        parts.append(table(["worker", "iterations", "with decode", "idle rank share", "batch occupancy", "real requests / iter",
                            "tokens per request", "kv_cache_util", "pool filled peak", "Δ offload", "Δ onboard", "Δ host dropped",
                            "host step", "device step"],
                           [[esc(worker), len(rows), len(busy), pct(idle_share), num(mean(r["batch_occupancy"] for r in busy), 3),
                             num(mean(r["decode_requests_sum"] for r in busy), 2), num(mean(r["tokens_per_request"] for r in busy), 2),
                             num(mean(r["kv_cache_util_mean"] for r in rows), 3),
                             pct(max((r["kv_pool_filled_ratio"] for r in rrows if r["kv_pool_filled_ratio"] is not None), default=None)),
                             *[num(sum(r[f"{c}_delta"] or 0 for r in rows)) for c in TIER_COUNTERS],
                             dur(mean(r["host_step_ms_mean"] for r in rows)), dur(mean(r["device_step_ms_mean"] for r in rows))]]))
        by_rank = defaultdict(list)
        for r in rrows:
            by_rank[r["rank"]].append(r)
        parts.append('<div class="grid-2">'
                     + charts.line_chart([("real requests, all ranks", [(r["timestamp"], r["decode_requests_sum"]) for r in rows])], f"{worker}: decode requests in flight", "requests")
                     + charts.line_chart([("pooled", [(r["timestamp"], r["tokens_per_request"]) for r in busy])], f"{worker}: tokens per request per iteration", "tokens")
                     + charts.line_chart([(f"rank {k}", [(r["timestamp"], r["kv_cache_util"]) for r in v]) for k, v in sorted(by_rank.items())], f"{worker}: kv_cache_util per rank", "share of blocks pinned", y_max=1.0)
                     + charts.line_chart([("pooled mean", [(r["timestamp"], r["device_step_ms_mean"]) for r in rows])], f"{worker}: device step time", "ms", unit=" ms")
                     + "</div>")
    return "".join(parts)


# ---------------------------------------------------------------- driver
def resolve_paths(target: Path, hour: str | None) -> tuple[Path, str | None]:
    """Accept an attempt dir, or a request_trace/<hour> dir which implies both."""
    target = target.resolve()
    if target.parent.name == "request_trace":
        return target.parent.parent, target.name
    return target, hour


def run_name(attempt: Path) -> str:
    return attempt.parent.name if re.fullmatch(r"attempt-\d+", attempt.name) else attempt.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="attempt directory, or its request_trace/<UTC hour> directory")
    parser.add_argument("--hour", default=None, help="UTC hour bucket to restrict to, e.g. 2026-09-09T13")
    parser.add_argument("--out", type=Path, default=None, help=f"output directory (default: {REPORTS_ROOT}/<run>[_<hour>])")
    parser.add_argument("--no-index", action="store_true",
                        help="read every log in full instead of resuming from the "
                             "checkpoint index. Slower on a long-lived instance; "
                             "use it to check the index against a full pass.")
    parser.add_argument("--json", action="store_true",
                        help="also write rollup.json: this hour reduced to percentiles, counts and "
                             "bounded samples, for a collector to keep. A few hundred kilobytes.")
    parser.add_argument("--json-only", action="store_true",
                        help="write only rollup.json and stop, skipping the CSVs and the HTML page")
    parser.add_argument("--log-tz", default=None, metavar="ZONE",
                        help="time zone of the worker logs' naive `timestamp` field, e.g. America/Los_Angeles "
                             "(default: this machine's zone; the workers and the analysis must agree)")
    args = parser.parse_args()
    log_tz = ZoneInfo(args.log_tz) if args.log_tz else local_tz()
    attempt, hour = resolve_paths(args.target, args.hour)
    out = args.out or REPORTS_ROOT / (run_name(attempt) + (f"_{hour}" if hour else ""))
    out.mkdir(parents=True, exist_ok=True)

    # B: requests
    requests, notes = build_requests(attempt, hour)
    starts = [r["started_at"] for r in requests if r["started_at"] is not None]
    ends = [r["finished_at"] for r in requests if r["finished_at"] is not None]
    window = (min(starts), max(ends)) if hour and starts and ends else (None, None)

    # A: engine iterations (worker logs are stamped in the worker's local zone; assumed to be this machine's)
    max_num_tokens = yaml_scalar([attempt / "ctx_config.yaml", attempt / "server_config.yaml"], "max_num_tokens")
    max_batch_size = yaml_scalar([attempt / "gen_config.yaml", attempt / "server_config.yaml"], "max_batch_size")
    tokens_per_block = yaml_scalar([attempt / "ctx_config.yaml", attempt / "server_config.yaml"], "tokens_per_block")
    # The index lives beside the logs it describes, so it travels with the run
    # and a second reader of the same attempt inherits the work of the first.
    index_dir = attempt / ".engine_index"
    engine = build_engine(attempt, log_tz, window, max_num_tokens, max_batch_size,
                          None if args.no_index else index_dir)

    real_tokens = sum(r["ctx_tokens_real"] for r in engine["ctx_rank"])
    perf_new_tokens = sum(r["ctx_blocks_new"] or 0 for r in requests) * (tokens_per_block or 0)
    pad_sizes = Counter(int(r["ctx_tokens"]) for r in engine["ctx_rank"] if r["is_adp_pad"])
    notes.append(f"Workers: {', '.join(w['worker'] for w in engine['workers'])}; budgets max_num_tokens={max_num_tokens}, "
                 f"max_batch_size={max_batch_size}, tokens_per_block={tokens_per_block}.")
    notes.append(f"Pad check: {sum(pad_sizes.values()):,} attention-DP pads removed (sizes seen: {dict(pad_sizes)}); "
                 f"real ctx tokens in window {real_tokens:,.0f} vs ctx blocks newly allocated × tokens_per_block "
                 f"{perf_new_tokens:,.0f} → ratio {real_tokens / perf_new_tokens:.4f}" if perf_new_tokens else
                 "Pad check skipped: no ctx KV block counts in the perf records.")
    if window[0] is not None:
        notes.append(f"Window: {utc_iso(window[0])} to {utc_iso(window[1])} (UTC), from the hour's requests; "
                     f"engine iterations outside it are dropped after differencing. Worker log stamps read as {log_tz}.")

    budgets = {"max_num_tokens": max_num_tokens, "max_batch_size": max_batch_size,
               "tokens_per_block": tokens_per_block}
    if args.json:
        # Written before the CSVs and the page, so a collector that only wants
        # this is not held up by the parts it will not read.
        payload = build_rollup(attempt, hour, window, requests, engine, budgets, notes)
        (out / "rollup.json").write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        print(f"  rollup -> {out / 'rollup.json'}")
        if args.json_only:
            return 0

    write_csv(out / "requests.csv", requests, REQUEST_COLUMNS, TIME_COLUMNS)
    write_csv(out / "ctx_iters.csv", engine["ctx_iters"], INSTANCE_COLUMNS, TIME_COLUMNS)
    write_csv(out / "ctx_rank_iters.csv", engine["ctx_rank"], RANK_COLUMNS, TIME_COLUMNS)
    write_csv(out / "gen_iters.csv", engine["gen_iters"], INSTANCE_COLUMNS, TIME_COLUMNS)
    write_csv(out / "gen_rank_iters.csv", engine["gen_rank"], RANK_COLUMNS, TIME_COLUMNS)

    title = f"{run_name(attempt)}" + (f" · {hour} UTC" if hour else "")
    page = [f"<!doctype html><meta charset=utf-8><title>{esc(title)}</title><style>{charts.CSS}</style>",
            f"<h1>{esc(title)}</h1><p class='sub'>{esc(attempt)}</p>",
            "<ul>" + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>",
            requests_section(requests),
            prefill_section(engine["ctx_iters"], engine["ctx_rank"], max_num_tokens),
            decode_section(engine["gen_iters"], engine["gen_rank"], max_batch_size)]
    (out / "REPORT.html").write_text("".join(page), encoding="utf-8")
    print(f"{len(requests)} requests, {len(engine['ctx_iters'])} ctx and {len(engine['gen_iters'])} gen iterations -> {out}")
    for note in notes:
        print("  note: " + note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
