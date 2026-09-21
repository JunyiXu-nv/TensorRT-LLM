# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engine-side iteration metrics from the worker logs (ctx-N.log / gen-N.log, or server.log of an aggregated server).

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

An aggregated deployment has one worker (server.log, role "mixed") whose ranks run prefill chunks
and decode steps in the same iteration: `num_ctx_requests` / `num_ctx_tokens` are the rank's
prefill work, `num_scheduled_requests - num_ctx_requests` its decode batch. Its idle rank is padded
like a generation worker's (one dummy decode request pinning nothing, never a near-budget context
chunk), so the generation rule applies, guarded by the rank having no context work.
`total_tokens_real` = real ctx tokens + decode token slots is what the scheduler counted against
max_num_tokens in that iteration.

`num_generation_tokens` is the decode token SLOTS of the iteration -- requests x (draft length + 1),
rounded up to the CUDA-graph batch size and, in a pure-decode attention-DP iteration, aligned across
ranks -- not the tokens the requests accepted. Accepted draft tokens are per request in the perf
records (speculative_decoding.acceptance_rate), not in the iteration log.

The KV allocation counters (kv_reused_blocks, kv_missed_blocks, kv_alloc_*_blocks) are cumulative
since engine start and are differenced against the same rank's previous iteration. The cross-tier
counters (kv_offload_blocks, kv_onboard_blocks, kv_host_dropped_blocks) are the opposite: the worker
drains them on every log line, so the printed value is what moved in THIS iteration. For both
families `*_delta` is the per-iteration movement and `*_total` the running total since engine start.
"""
from __future__ import annotations

import concurrent.futures
import gc
import multiprocessing
import re
import statistics
from collections import defaultdict

import engine_index
from datetime import tzinfo
from pathlib import Path

from common import parse_log_stamp, rank_skew, to_float

ITER_LINE = re.compile(r"\biter = \d+, global_rank = ")
# `states = {'num_ctx_requests': 0, 'num_ctx_tokens': 0, ...}`: every value is an int, so a regex reads the
# dict in a few microseconds where ast.literal_eval took tens (it was the largest per-line cost).
_STATE_INT = re.compile(r"'(\w+)':\s*(-?\d+)")
_STAMP_CACHE: dict[str, float | None] = {}   # the same second is stamped on every rank's line of every iteration
PAD_COUNTERS = ("kv_reused_blocks", "kv_missed_blocks", "kv_alloc_total_blocks", "kv_alloc_new_blocks")
PAD_MIN_FRACTION = 0.9  # a pad is "about max_num_tokens"; the exact size differs between configs
TIER_COUNTERS = ("kv_offload_blocks", "kv_onboard_blocks", "kv_host_dropped_blocks")

RANK_COLUMNS = [
    "worker", "engine_instance", "iter", "rank", "timestamp",
    "scheduled_requests", "paused_requests", "ctx_requests",
    "ctx_tokens", "is_adp_pad", "ctx_tokens_real", "token_budget_util",
    "gen_tokens", "cached_kv_tokens", "decode_requests_real", "gen_tokens_real",
    "total_tokens_real", "total_budget_util",
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
    "busy_ranks", "has_prefill", "ctx_requests_sum", "ctx_tokens_mean", "ctx_tokens_max", "token_budget_util",
    "ctx_tokens_rank_skew",
    # decode (gen workers)
    "has_decode", "decode_requests_sum", "batch_occupancy", "decode_requests_rank_skew",
    "gen_tokens_sum", "tokens_per_request", "cached_kv_tokens_sum",
    # both phases on one rank (aggregated workers)
    "decode_only", "total_tokens_mean", "total_budget_util",
    # KV cache
    "kv_hit_rate_iter", "kv_hit_rate_cum", "kv_cache_util_mean",
    "kv_free_blocks_sum", "kv_evictable_blocks_sum", "kv_capacity_blocks_sum",
    "kv_pool_filled_mean", "kv_pool_filled_rank_skew",
    "kv_offload_blocks_delta", "kv_onboard_blocks_delta", "kv_host_dropped_blocks_delta",
    # step time
    "host_step_ms_mean", "device_step_ms_mean", "device_step_rank_skew",
]
PREFILL_ONLY = ("busy_ranks", "has_prefill", "ctx_requests_sum", "ctx_tokens_mean", "ctx_tokens_max", "token_budget_util",
                "ctx_tokens_rank_skew")
DECODE_ONLY = ("has_decode", "decode_requests_sum", "batch_occupancy", "decode_requests_rank_skew", "gen_tokens_sum",
               "tokens_per_request", "cached_kv_tokens_sum")
MIXED_ONLY = ("decode_only", "total_tokens_mean", "total_budget_util")


# ---------------------------------------------------------------- parsing
def iter_log_entries(path: Path, tz: tzinfo, start_offset: int = 0,
                     start_instance: int = 0, report=None):
    """Raw per-(rank, iteration) entries of one worker log, tagged with engine_instance.

    A generator, so a caller that only wants a window never holds the run. The
    list form below is the same walk, kept for callers that do want it all.

    `start_offset` and `start_instance` resume from a checkpoint rather than
    from the beginning, which is what keeps the cost of one hour independent of
    how long the instance has been up. Resuming is safe because restarts are
    found by a rank's counter going backwards against a map that starts empty,
    and an empty map compares against zero -- no live iteration number is at or
    below that, so a resume cannot invent a restart it has already been told
    about.

    `report`, if given, is called as report(offset, timestamp, instance) once
    per line, which is how the index is extended by the same pass that reads.
    """
    instance, last_iter_of_rank = start_instance, {}
    with path.open(encoding="latin-1", errors="replace") as handle:
        if start_offset:
            handle.seek(start_offset)
        # readline rather than iteration: a text file being iterated refuses to
        # tell() its position ("telling position disabled by next() call"), and
        # the position is the whole point -- it is what the next run seeks to.
        while True:
            line = handle.readline()
            if not line:
                break
            if not ITER_LINE.search(line):
                continue
            tail = line[line.find("iter = "):].rstrip()
            states: dict = {}
            marker = tail.find("states = ")
            if marker >= 0:
                states = {key: int(value) for key, value in _STATE_INT.findall(tail, marker + 9)}
                tail = tail[:marker]
            fields = {}
            for chunk in tail.split(", "):
                key, sep, value = chunk.partition(" = ")
                if sep:
                    fields[key.strip()] = value.strip().rstrip(",")
            iteration, rank = int(fields["iter"]), int(fields.get("rank", -1))
            here = handle.tell()
            if iteration <= last_iter_of_rank.get(rank, 0):  # counter restarted: a new engine lifetime
                instance += 1
                last_iter_of_rank.clear()
            last_iter_of_rank[rank] = iteration
            raw_stamp = fields.get("timestamp")
            stamp = _STAMP_CACHE.get(raw_stamp) if raw_stamp in _STAMP_CACHE else None
            if raw_stamp not in _STAMP_CACHE:
                if len(_STAMP_CACHE) > 100_000:
                    _STAMP_CACHE.clear()
                stamp = _STAMP_CACHE[raw_stamp] = parse_log_stamp(raw_stamp, tz)
            if report is not None:
                report(here, stamp, instance)
            yield {
                "engine_instance": instance, "iter": iteration, "rank": rank,
                "timestamp": stamp,
                "scheduled_requests": int(fields.get("num_scheduled_requests", 0)),
                "paused_requests": int(to_float(fields.get("num_paused_requests")) or 0),
                "kv_cache_util": to_float(fields.get("kv_cache_util")),
                "kv_hit_rate_cum": to_float(fields.get("kv_hit_rate")),
                "kv_free_blocks": to_float(fields.get("kv_free_blocks")),
                "kv_evictable_blocks": to_float(fields.get("kv_evictable_blocks")),
                "host_step_ms": to_float(fields.get("host_step_time", "").rstrip("ms")),
                "device_step_ms": to_float(fields.get("prev_device_step_time", "").rstrip("ms")),
                "ctx_requests": states.get("num_ctx_requests"),
                "ctx_tokens": states.get("num_ctx_tokens"),
                "gen_tokens": states.get("num_generation_tokens"),
                "cached_kv_tokens": states.get("cached_kv_tokens"),
                **{key: to_float(fields.get(key)) for key in PAD_COUNTERS + TIER_COUNTERS},
            }


def parse_iter_log(path: Path, tz: tzinfo) -> list[dict]:
    """Every entry of one worker log, in file order."""
    return list(iter_log_entries(path, tz))


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


def is_mixed_pad(entry: dict) -> bool:
    """Idle rank of an aggregated worker: the generation dummy, on a rank with no context chunk either."""
    return is_gen_pad(entry) and not entry.get("ctx_requests") and not entry.get("ctx_tokens")


def pad_of(role: str, entry: dict, previous: dict | None, max_num_tokens: int | None) -> bool:
    if role == "gen":
        return is_gen_pad(entry)
    if role == "mixed":
        return is_mixed_pad(entry)
    return is_adp_pad(entry, previous, max_num_tokens)


def delta(entry: dict, previous: dict | None, key: str) -> float | None:
    if previous is None or entry.get(key) is None or previous.get(key) is None:
        return None
    return entry[key] - previous[key]


def ratio(top: float | None, bottom: float | None) -> float | None:
    return None if top is None or not bottom else top / bottom


def advance_tiers(totals: dict[str, float] | None, entry: dict) -> dict[str, float]:
    """Running totals of the cross-tier counters after this entry (they are printed per iteration)."""
    out = dict(totals) if totals else {counter: 0.0 for counter in TIER_COUNTERS}
    for counter in TIER_COUNTERS:
        if entry.get(counter) is not None:
            out[counter] += entry[counter]
    return out


def rank_row(worker: str, role: str, entry: dict, previous: dict | None,
             max_num_tokens: int | None, tiers: dict[str, float] | None = None) -> dict:
    """One rank row, from one entry, that rank's previous one and its tier totals so far.

    ``kv_capacity_blocks`` and ``kv_pool_filled_ratio`` are left None here:
    capacity is a peak over the engine's whole life, which no single entry
    knows. Both callers fill them with fill_capacity() once their pass is done.
    ``tiers`` is the other lifetime quantity, carried by the caller (and by the
    index across resumes) for the same reason.

    ``device_step_ms`` is written as printed, which is the span of the PREVIOUS
    iteration; both callers shift it back afterwards (realign_device_step).

    Extracted so the streaming and whole-file paths cannot drift into
    disagreeing about a column.
    """
    pad = pad_of(role, entry, previous, max_num_tokens)
    real_tokens = 0 if pad else (entry["ctx_tokens"] or 0)
    ctx_requests = entry.get("ctx_requests")
    # On an aggregated worker the scheduled count covers both phases and the
    # decode batch is what is left after the context requests. ctx and gen
    # workers keep the plain count, which is what their logs have always meant.
    if pad:
        decode_requests = 0
    elif role == "mixed":
        decode_requests = max(entry["scheduled_requests"] - (ctx_requests or 0), 0)
    else:
        decode_requests = entry["scheduled_requests"]
    gen_tokens = 0 if pad else (entry["gen_tokens"] or 0)
    total_tokens = real_tokens + gen_tokens
    reused, missed = delta(entry, previous, "kv_reused_blocks"), delta(entry, previous, "kv_missed_blocks")
    free = entry["kv_free_blocks"]
    return {
        "worker": worker, "engine_instance": entry["engine_instance"], "iter": entry["iter"],
        "rank": entry["rank"], "timestamp": entry["timestamp"],
        "scheduled_requests": entry["scheduled_requests"], "paused_requests": entry["paused_requests"],
        "ctx_requests": ctx_requests,
        "ctx_tokens": entry["ctx_tokens"], "is_adp_pad": pad, "ctx_tokens_real": real_tokens,
        "token_budget_util": ratio(real_tokens, max_num_tokens),
        "gen_tokens": entry["gen_tokens"], "cached_kv_tokens": entry["cached_kv_tokens"],
        "decode_requests_real": decode_requests,
        "gen_tokens_real": gen_tokens,
        "total_tokens_real": total_tokens,
        "total_budget_util": ratio(total_tokens, max_num_tokens),
        "kv_hit_rate_iter": ratio(reused, (reused or 0) + (missed or 0)) if reused is not None and missed is not None else None,
        "kv_hit_rate_cum": entry["kv_hit_rate_cum"],
        "kv_reused_blocks_delta": reused, "kv_missed_blocks_delta": missed,
        "kv_reused_blocks_total": entry["kv_reused_blocks"], "kv_missed_blocks_total": entry["kv_missed_blocks"],
        "kv_cache_util": entry["kv_cache_util"],
        "kv_free_blocks": free, "kv_evictable_blocks": entry["kv_evictable_blocks"],
        "kv_capacity_blocks": None,
        "kv_pool_filled_ratio": None,
        **{f"{counter}_delta": entry.get(counter) for counter in TIER_COUNTERS},
        **{f"{counter}_total": (tiers or {}).get(counter) for counter in TIER_COUNTERS},
        "host_step_ms": entry["host_step_ms"], "device_step_ms": entry["device_step_ms"],
    }


def fill_capacity(rows_by_rank: dict, capacity: dict) -> None:
    """Write each rank's lifetime peak into the rows kept for it. In place."""
    for key, kept in rows_by_rank.items():
        cap = capacity.get(key)
        for row in kept:
            free = row["kv_free_blocks"]
            row["kv_capacity_blocks"] = cap
            row["kv_pool_filled_ratio"] = 1.0 - free / cap if cap and free is not None else None


def realign_device_step(kept: list[dict]) -> None:
    """Move prev_device_step_time back onto the iteration it measured. In place; one rank's rows in iteration order.

    profile_step() (py_executor.py) records a CUDA event pair at the top of consecutive loops and reads
    the OTHER parity's pair, so the value printed on iteration N is the GPU-timeline span of iteration
    N − 1, while host_step_time on the same line is N's. Every row takes the value the next line of its
    rank printed, provided that line is the very next iteration; the last row of a rank keeps None.
    """
    for row, following in zip(kept, kept[1:]):
        row["device_step_ms"] = following["device_step_ms"] if following["iter"] == row["iter"] + 1 else None
    if kept:
        kept[-1]["device_step_ms"] = None


def rank_rows(worker: str, role: str, entries: list[dict], max_num_tokens: int | None) -> list[dict]:
    """Every rank row of one worker, held in memory. stream_rank_rows is the windowed form."""
    entries = sorted(entries, key=lambda e: (e["engine_instance"], e["rank"], e["iter"]))
    rows, previous_of_rank, capacity, by_rank = [], {}, {}, defaultdict(list)
    tiers: dict[tuple, dict[str, float]] = {}
    for entry in entries:
        key = (entry["engine_instance"], entry["rank"])
        free, evictable = entry["kv_free_blocks"], entry["kv_evictable_blocks"]
        if free is not None and evictable is not None:
            capacity[key] = max(capacity.get(key, 0.0), free + evictable)
        tiers[key] = advance_tiers(tiers.get(key), entry)
        row = rank_row(worker, role, entry, previous_of_rank.get(key), max_num_tokens, tiers[key])
        rows.append(row)
        by_rank[key].append(row)
        previous_of_rank[key] = entry
    fill_capacity(by_rank, capacity)
    for kept in by_rank.values():
        realign_device_step(kept)
    return rows


# ------------------------------------------------------- windowed streaming
# Ranks of one iteration are written within milliseconds of each other, but a
# window boundary can still fall between them. Rows are kept with this much
# slack so an iteration at the edge is pooled from all its ranks, and the exact
# window is applied to the finished tables.
ITERATION_SLACK_S = 5.0


def _consume(entry, worker, role, max_num_tokens, windowed, keep_lo, keep_hi,
             rows, previous_of_rank, last_row_of_rank, capacity, pending, tiers, marks) -> None:
    """One log line into the running state of stream_rank_rows (its loop body, kept separate for readability)."""
    key = (entry["engine_instance"], entry["rank"])
    free, evictable = entry["kv_free_blocks"], entry["kv_evictable_blocks"]
    if free is not None and evictable is not None:
        capacity[key] = max(capacity.get(key, 0.0), free + evictable)
    tiers[key] = advance_tiers(tiers.get(key), entry)
    if marks and marks[-1][3] is None:
        marks[-1][3] = engine_index.tiers_stored(tiers)
    previous = previous_of_rank.get(key)
    # The log prints the GPU span one loop late (realign_device_step): what this line carries is
    # the span of this rank's previous iteration. Hand it to that row when it was kept and is the
    # very next iteration -- also when this line itself falls outside the window -- and leave this
    # row's own cell for the next line to fill.
    earlier = last_row_of_rank.pop(key, None)
    if earlier is not None and earlier["iter"] == entry["iter"] - 1:
        earlier["device_step_ms"] = entry["device_step_ms"]
    if not windowed or _inside(entry["timestamp"], keep_lo, keep_hi):
        row = rank_row(worker, role, entry, previous, max_num_tokens, tiers[key])
        row["device_step_ms"] = None
        rows.append(row)
        pending[key].append(row)
        last_row_of_rank[key] = row
    previous_of_rank[key] = entry


def stream_rank_rows(worker: str, role: str, path: Path, tz: tzinfo,
                     max_num_tokens: int | None,
                     lo: float | None, hi: float | None,
                     index_dir: Path | None = None) -> tuple[list[dict], int, int]:
    """One pass over a worker log, materialising only the rows inside the window.

    The whole-file version of this holds two dicts per iteration line, and a
    four-hour run of seven workers is ten million of them -- about 19 GB, on a
    login node that caps a process at 8 GB of address space. Windowing a
    finished table cannot help, because the finished table is the thing that
    does not fit.

    Two quantities genuinely need the whole file, and neither needs it kept:

    the previous iteration of each rank, for the deltas. Carried forward in a
    dict holding one entry per rank, so the first row inside the window is
    still a one-step delta and not a jump from the beginning of the run -- the
    property the old post-filter was written to preserve.

    ``kv_capacity_blocks``, the peak of free + evictable over the engine's
    lifetime. A peak cannot be read off a window, so it is accumulated as two
    floats per rank and written into the kept rows after the pass. The rows it
    patches are only the kept ones, so the fixup is bounded by the window too.

    The cross-tier totals (offload / onboard / host dropped since engine start)
    are the same kind of quantity: the worker prints those counters drained per
    iteration, so the sum is carried per rank across the pass, and across
    resumes by storing it in every checkpoint of the index.

    The pass stops ITERATION_SLACK_S past the window's end: everything after it
    belongs to a later hour and gets read when that hour is asked for. So one
    hour costs one hour's slice of the log whatever the instance's age, and a
    backfill of N hours costs N slices instead of N²/2 (the previous pass read
    to EOF every time). The index is extended as far as the pass read; the next
    hour resumes from the last checkpoint before its own start, which this pass
    wrote if it read through it.

    Returns the rows, the number of iteration lines read, and how many engine
    lifetimes they covered (whole-file counts when there is no window).
    """
    keep_lo = None if lo is None else lo - ITERATION_SLACK_S
    keep_hi = None if hi is None else hi + ITERATION_SLACK_S
    # No window means keep everything, including rows whose timestamp did not
    # parse. _inside() calls those outside every window, which is right when
    # there is a window to be outside of and wrong when there is not -- and the
    # caller this replaced skipped the filter entirely in that case.
    windowed = lo is not None or hi is not None

    rows: list[dict] = []
    previous_of_rank: dict[tuple, dict] = {}
    last_row_of_rank: dict[tuple, dict] = {}   # the kept row still waiting for its device step
    capacity: dict[tuple, float] = {}
    pending: dict[tuple, list[dict]] = defaultdict(list)
    lines = 0
    instances = 0

    # Without an index this reads from the start, which is correct and was the
    # only behaviour. With one it resumes from the last checkpoint before the
    # window, which is what stops one hour costing more as the instance ages.
    data = engine_index.load(index_dir, path) if index_dir else None
    start_offset, start_instance = 0, 0
    tiers: dict[tuple, dict[str, float]] = {}
    if data is not None:
        start_offset, start_instance, tiers = engine_index.seek_point(data, keep_lo)
        capacity.update(engine_index.capacity_map(data))

    marks: list[list] = []
    last_mark = [start_offset]

    def note(offset, stamp, instance):
        # Called just before the entry ending at `offset` is yielded; the tier
        # snapshot is attached below, once that entry has been added to the totals.
        if offset - last_mark[0] >= engine_index.CHECKPOINT_BYTES:
            marks.append([offset, stamp, instance, None])
            last_mark[0] = offset

    # Millions of short-lived dicts; the cyclic collector's passes over them cost more than they free.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for entry in iter_log_entries(path, tz, start_offset, start_instance,
                                      note if data is not None else None):
            stamp = entry["timestamp"]
            if keep_hi is not None and stamp is not None and stamp > keep_hi:
                break
            _consume(entry, worker, role, max_num_tokens, windowed, keep_lo, keep_hi,
                     rows, previous_of_rank, last_row_of_rank, capacity, pending, tiers, marks)
            lines += 1
            instances = max(instances, entry["engine_instance"])
    finally:
        if gc_was_enabled:
            gc.enable()

    fill_capacity(pending, capacity)
    if data is not None:
        engine_index.merge_capacity(data, capacity)
        engine_index.merge_checkpoints(data, [tuple(m) for m in marks if m[3] is not None])
        try:
            # Only ever grows: a run that stopped early must not tell the next
            # one that less of the file has been indexed than actually has.
            data["indexed_bytes"] = max(data.get("indexed_bytes", 0), path.stat().st_size)
        except OSError:
            pass
        engine_index.save(index_dir, path, data)
    return rows, lines, instances + 1


# ---------------------------------------------------------------- per instance
def _mean(values) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


def _sum(values) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) if kept else None


def instance_rows(rows: list[dict], role: str, max_num_tokens: int | None, max_batch_size: int | None) -> list[dict]:
    """Pool one worker's ranks per iteration. Prefill columns are filled for ctx workers, decode for gen, all for mixed."""
    by_iter: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_iter[(row["worker"], row["engine_instance"], row["iter"])].append(row)

    out = []
    for (worker, instance, iteration), ranks in sorted(by_iter.items()):
        tokens = [r["ctx_tokens_real"] for r in ranks]  # pads are 0 and stay in the denominator
        tokens_mean = statistics.fmean(tokens)
        batch = [r["decode_requests_real"] for r in ranks]  # idle-rank dummies are 0
        gen_tokens = sum(r["gen_tokens_real"] for r in ranks)
        totals = [r["total_tokens_real"] for r in ranks]
        reused, missed = _sum(r["kv_reused_blocks_delta"] for r in ranks), _sum(r["kv_missed_blocks_delta"] for r in ranks)
        cum_reused, cum_missed = _sum(r["kv_reused_blocks_total"] for r in ranks), _sum(r["kv_missed_blocks_total"] for r in ranks)
        row = {
            "worker": worker, "engine_instance": instance, "iter": iteration,
            "timestamp": ranks[0]["timestamp"], "ranks": len(ranks),
            "idle_ranks": sum(1 for r in ranks if r["is_adp_pad"]),
            "scheduled_requests_sum": sum(r["scheduled_requests"] for r in ranks),
            "paused_requests_sum": sum(r["paused_requests"] for r in ranks),
            "busy_ranks": sum(1 for t in tokens if t > 0), "has_prefill": max(tokens) > 0,
            "ctx_requests_sum": _sum(r["ctx_requests"] for r in ranks),
            "ctx_tokens_mean": tokens_mean, "ctx_tokens_max": max(tokens),
            "token_budget_util": ratio(tokens_mean, max_num_tokens),
            "ctx_tokens_rank_skew": rank_skew(tokens),
            "has_decode": sum(batch) > 0, "decode_requests_sum": sum(batch),
            "batch_occupancy": ratio(statistics.fmean(batch), max_batch_size),
            "decode_requests_rank_skew": rank_skew(batch),
            "gen_tokens_sum": gen_tokens,
            "tokens_per_request": ratio(gen_tokens, sum(batch)),
            "cached_kv_tokens_sum": _sum(r["cached_kv_tokens"] for r in ranks),
            "decode_only": sum(batch) > 0 and max(tokens) == 0,
            "total_tokens_mean": statistics.fmean(totals),
            "total_budget_util": ratio(statistics.fmean(totals), max_num_tokens),
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
        for key in DECODE_ONLY + MIXED_ONLY if role == "ctx" else PREFILL_ONLY + MIXED_ONLY if role == "gen" else ():
            row[key] = None  # a context-only worker has no decode batch, a generation worker no prefill chunk
        out.append(row)
    return out


# ---------------------------------------------------------------- quotas
_QUOTA_LINES = (("device_quota_gib", re.compile(r"device quota set to ([0-9.]+)GiB")),
                ("host_quota_gib", re.compile(r"host cache quota set to ([0-9.]+)GiB")),
                ("kv_bytes_per_token", re.compile(r"kv size per token is (\d+) byte")))


def parse_quotas(path: Path, max_lines: int = 50_000) -> dict:
    """GPU and host KV quotas of the engine that served traffic, from the worker's startup lines.

    The memory-profiling dry run prints its own (smaller) quotas first, so the LAST match of each line
    within the startup region is the real engine's.
    """
    found: dict = {}
    with path.open(encoding="latin-1", errors="replace") as handle:
        for number, line in enumerate(handle):
            if number > max_lines:
                break
            if "quota set to" not in line and "kv size per token" not in line:
                continue
            for key, pattern in _QUOTA_LINES:
                hit = pattern.search(line)
                if hit:
                    found[key] = float(hit.group(1))
    return found


# ---------------------------------------------------------------- driver
def worker_logs(attempt: Path) -> list[tuple[str, str, Path]]:
    """(worker name, role, path). Disaggregated runs name logs by role; an aggregated run has server.log."""
    logs = [(p.stem, "ctx", p) for p in sorted(attempt.glob("ctx-*.log"))]
    logs += [(p.stem, "gen", p) for p in sorted(attempt.glob("gen-*.log"))]
    if not logs and (attempt / "server.log").exists():
        logs = [("server", "mixed", attempt / "server.log")]
    return logs


def _engine_job(job: tuple) -> tuple:
    """One worker log, start to finish: stream, pool per iteration, cut to the window, read the quotas."""
    worker, role, path, tz, max_num_tokens, max_batch_size, lo, hi, index_dir = job
    # Windowed while reading, not after: see stream_rank_rows. Rows arrive
    # with ITERATION_SLACK_S of margin so an iteration on the boundary is
    # still pooled from all of its ranks, and the exact window is applied
    # to both finished tables below.
    ranks, lines, instances = stream_rank_rows(
        worker, role, path, tz, None if role == "gen" else max_num_tokens, lo, hi, index_dir)
    iters = instance_rows(ranks, role, max_num_tokens, max_batch_size)
    if lo is not None or hi is not None:
        ranks = [r for r in ranks if _inside(r["timestamp"], lo, hi)]
        iters = [r for r in iters if _inside(r["timestamp"], lo, hi)]
    return worker, role, ranks, iters, lines, instances, parse_quotas(path)


def build_engine(attempt: Path, tz: tzinfo, window: tuple[float | None, float | None] = (None, None),
                 max_num_tokens: int | None = None, max_batch_size: int | None = None,
                 index_dir: Path | None = None, jobs: int | None = None) -> dict:
    """Rank and instance rows of every worker, restricted to the window.

    The window is applied while the logs are read rather than to a finished
    table, because the finished table is what does not fit: ten million
    iteration lines is about 19 GB of dicts against an 8 GB per-process cap on
    the login node. Differencing still crosses the boundary -- each rank's
    previous iteration is carried forward -- so the first surviving row is a
    one-step delta and not a jump from the start of the run, which is the
    property the old post-filter existed to preserve.
    """
    result = {"ctx_rank": [], "ctx_iters": [], "gen_rank": [], "gen_iters": [],
              "mixed_rank": [], "mixed_iters": [], "workers": [], "mode": "disagg"}
    lo, hi = window
    jobs_list = [(worker, role, path, tz, max_num_tokens, max_batch_size, lo, hi, index_dir)
                 for worker, role, path in worker_logs(attempt)]
    # The logs are independent files with independent index entries, so they are read in
    # parallel, one process each (a 6P1D attempt is seven of them). `jobs` caps the processes;
    # 1 keeps everything in this process, which is the form to debug in.
    workers_n = max(1, min(len(jobs_list), jobs or len(jobs_list)))
    if workers_n > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers_n,
                                                    mp_context=multiprocessing.get_context("fork")) as pool:
            outcomes = list(pool.map(_engine_job, jobs_list))
    else:
        outcomes = [_engine_job(job) for job in jobs_list]
    for worker, role, ranks, iters, lines, instances, quotas in outcomes:
        bucket = role if role in ("gen", "mixed") else "ctx"
        result[f"{bucket}_rank"] += ranks
        result[f"{bucket}_iters"] += iters
        result["workers"].append({"worker": worker, "role": role, "iter_lines": lines,
                                  "engine_instances": instances, **quotas})
    # One worker doing both phases is an aggregated deployment; a disaggregated one names its logs by role.
    result["mode"] = "agg" if any(w["role"] == "mixed" for w in result["workers"]) else "disagg"
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
    for name in ("ctx_rank", "ctx_iters", "gen_rank", "gen_iters", "mixed_rank", "mixed_iters"):
        cols = RANK_COLUMNS if name.endswith("rank") else INSTANCE_COLUMNS
        print(name, len(engine[name]), "->", write_csv(Path(f"{name}.csv"), engine[name], cols, TIME_COLUMNS))
    print(engine["workers"])
