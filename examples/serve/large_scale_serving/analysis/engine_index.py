# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A resumable index over a worker's iteration log, so one hour costs one hour.

Engine logs are appended to for the life of the instance, and the analysis of
any single hour was reading all of them: 254 MB for an instance two hours old,
12 GB for one eighteen hours old, and around 150 GB for one that lasts the
seven days the partition allows. The work needed is always the same one hour;
only the distance to it grows.

This keeps, per worker log, a small sidecar recording

    capacity     the peak of free + evictable blocks per (engine instance,
                 rank) -- a lifetime maximum, which a window cannot see and
                 which therefore has to be carried rather than recomputed;

    checkpoints  (byte offset, timestamp, engine instance) every so often, so
                 a reader can seek to just before a window instead of parsing
                 its way there;

    tail         where indexing stopped and in what state, so the next run
                 reads only what has been appended since.

A checkpoint stores no per-rank iteration numbers on purpose. Restarts are
detected by a rank's counter going backwards, compared against a map that
starts empty -- and an empty map compares against zero, which no live iteration
number is at or below. So resuming with an empty map cannot invent a restart,
and the count of the ones already passed is what the checkpoint carries.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# How often to drop a checkpoint. Small enough that a seek lands close to the
# window, large enough that a day of logs is a few hundred of them.
CHECKPOINT_BYTES = 64 * 1024 * 1024
INDEX_SCHEMA = 2


def index_path(index_dir: Path, log: Path) -> Path:
    # Named for the attempt and the worker: two attempts of the same job have
    # different logs and must not share one of these.
    return index_dir / ("%s__%s.index.json" % (log.parent.parent.name, log.stem))


def _blank(log: Path) -> dict:
    return {"schema": INDEX_SCHEMA, "worker": log.stem, "capacity": {},
            "checkpoints": [], "tail_offset": 0, "tail_instance": 0, "indexed_bytes": 0}


def load(index_dir: Path, log: Path) -> dict:
    """The stored index, or a blank one if it is missing, stale or unreadable.

    A log that is now shorter than what was indexed has been rotated or
    truncated, and every offset held here points somewhere else. That is not
    something to repair; it is something to notice and start again from.
    """
    path = index_path(index_dir, log)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return _blank(log)
    if data.get("schema") != INDEX_SCHEMA or data.get("worker") != log.stem:
        return _blank(log)
    try:
        if log.stat().st_size < data.get("indexed_bytes", 0):
            return _blank(log)
    except OSError:
        return _blank(log)
    return data


def save(index_dir: Path, log: Path, data: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_path(index_dir, log)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def seek_point(data: dict, before: float | None) -> tuple[int, int]:
    """(byte offset, engine instance) to resume from for a window starting at `before`.

    The latest checkpoint at or before the timestamp, so everything the reader
    needs to see is still ahead of it. Without a timestamp -- a whole-attempt
    run -- that is the beginning of the file.
    """
    if before is None:
        return 0, 0
    # One checkpoint further back than strictly necessary. Differencing needs
    # each rank to have been seen before the window opens, and a checkpoint
    # that happens to land on the window boundary would give it nothing to
    # difference against -- silently, as a column of empty deltas rather than
    # an error. A checkpoint is 64 MB of log, which is warm-up to spare.
    recent: list[tuple] = [(0, 0), (0, 0)]
    for point in data.get("checkpoints", []):
        if point["ts"] is None or point["ts"] > before:
            break
        recent = [recent[1], (point["offset"], point["instance"])]
    return recent[0]


def capacity_map(data: dict) -> dict[tuple, float]:
    """The stored capacity, keyed the way the row builder keys it."""
    out = {}
    for key, value in (data.get("capacity") or {}).items():
        instance, _, rank = key.partition("|")
        try:
            out[(int(instance), int(rank))] = float(value)
        except ValueError:
            continue
    return out


def merge_capacity(data: dict, seen: dict[tuple, float]) -> None:
    """Fold freshly observed peaks into the stored ones. A peak only ever rises."""
    stored = data.setdefault("capacity", {})
    for (instance, rank), value in seen.items():
        key = "%d|%d" % (instance, rank)
        if value > stored.get(key, 0.0):
            stored[key] = value


def merge_checkpoints(data: dict, marks: list[tuple]) -> None:
    """Fold newly seen checkpoints into the stored ones, keeping all of them.

    Replacing the list with what this pass happened to see would undo the index
    whenever an older hour is analysed: that run resumes early, records only
    what follows, and every checkpoint past its window disappears -- so the next
    run on a recent hour has nothing to seek to and reads the file from there.
    Offsets are stable for as long as the index is valid at all (a file that
    shrank invalidates the whole thing), so the two sets simply merge.
    """
    by_offset = {int(c["offset"]): c for c in data.get("checkpoints", [])}
    for offset, ts, instance in marks:
        by_offset[int(offset)] = {"offset": int(offset), "ts": ts, "instance": instance}
    data["checkpoints"] = [by_offset[k] for k in sorted(by_offset)]
