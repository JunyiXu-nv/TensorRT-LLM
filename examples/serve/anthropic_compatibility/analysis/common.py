# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by the analysis scripts: time parsing, percentiles, CSV, yaml scalars."""
from __future__ import annotations

import csv
import gzip
import re
import statistics
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any, Iterable

GZIP_ABOVE_ROWS = 50_000  # iteration-grain CSVs get large; gzip them past this


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_iso(value: str | None) -> float | None:
    """ISO-8601 stamp (as written by request_trace) -> epoch seconds. Naive stamps are UTC."""
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def parse_log_stamp(value: str | None, tz: tzinfo) -> float | None:
    """'2026-09-09 06:46:55' from a worker log (naive, worker-local time) -> epoch seconds."""
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp()
    except ValueError:
        return None


def utc_iso(epoch: float | None) -> str:
    return "" if epoch is None else datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def percentile(sorted_values: list[float], q: float) -> float | None:
    """Nearest-rank percentile on an already sorted list."""
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[index]


def stats(values: Iterable[float | None]) -> dict[str, float | None]:
    """n / mean / p50 / p90 / p99 / max over the non-missing values."""
    kept = sorted(v for v in values if v is not None)
    if not kept:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    return {"n": len(kept), "mean": statistics.fmean(kept), "p50": percentile(kept, 0.50),
            "p90": percentile(kept, 0.90), "p99": percentile(kept, 0.99), "max": kept[-1]}


def rank_skew(values: Iterable[float | None]) -> float | None:
    """(max - mean) / mean across the ranks of one instance.

    0 means every rank carried the same amount; with n ranks and a single busy one it is n - 1.
    None below two ranks (a single sample has nothing to compare against).
    """
    kept = [v for v in values if v is not None]
    if len(kept) < 2:
        return None
    mean = statistics.fmean(kept)
    return (max(kept) - mean) / mean if mean else None


def write_csv(path: Path, rows: list[dict], columns: list[str], formatters: dict | None = None) -> Path:
    """Write rows in the given column order; gzip (with .gz suffix) when large. Missing -> empty.

    `formatters` maps a column to a function applied before writing (e.g. epoch -> ISO string).
    """
    formatters = formatters or {}
    if len(rows) > GZIP_ABOVE_ROWS:
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(formatters[k](row.get(k)) if k in formatters else row.get(k)) for k in columns})
    return path


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")  # 4 decimals is enough for ms and ratios
    return value


TIME_COLUMNS = {"started_at": utc_iso, "handler_entry_at": utc_iso, "finished_at": utc_iso, "timestamp": utc_iso}


def yaml_scalar(paths: Iterable[Path], key: str) -> int | None:
    """First `key: <int>` found in any of the yaml files (top level or nested, regex not a parser)."""
    for path in paths:
        if not path.exists():
            continue
        hit = re.search(rf"^\s*{re.escape(key)}:\s*(\d+)\s*$", path.read_text(), re.MULTILINE)
        if hit:
            return int(hit.group(1))
    return None


def local_tz() -> tzinfo:
    return datetime.now().astimezone().tzinfo
