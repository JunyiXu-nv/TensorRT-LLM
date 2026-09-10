# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep a directory of hourly rollups up to date, one per instance and hour.

A dashboard cannot run the analysis on demand: it is seven minutes and a
gigabyte and a half per instance-hour, and a browser will not wait. So the
analysis runs here, once per hour that has finished, and leaves 148 KB behind.
Everything downstream reads the directory.

    python3 rollup_store.py --trace-root .../deploy/trace --store .../wb/rollups

Idempotent: an hour already in the store is skipped, so this can be run on a
timer, by hand, or twice at once without doing the work twice. Only hours that
have ended are rolled up -- an hour still being written would be summarised
from a partial trace and then never revisited, which is worse than being late.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report.py"

# An hour is finished once a later hour has appeared in the same attempt. That
# is a stronger claim than "the clock has moved on": it says this instance was
# still serving after the hour ended, so the trace for it is complete rather
# than merely old.
def finished_hours(attempt: Path) -> list[str]:
    root = attempt / "request_trace"
    if not root.is_dir():
        return []
    hours = sorted(p.name for p in root.iterdir() if p.is_dir())
    return hours[:-1]


def attempts(trace_root: Path, days: int = 2) -> list[Path]:
    """Attempt directories from the last `days` day-directories, newest first."""
    out = []
    for month in sorted(trace_root.glob("*"), reverse=True):
        for day in sorted(month.glob("*"), reverse=True)[:days]:
            for run in sorted(day.glob("*")):
                out += sorted(run.glob("attempt-*"))
    return out


def store_name(attempt: Path, hour: str) -> str:
    return f"{attempt.parent.name}__{attempt.name}__{hour}.json"


def already(store: Path, attempt: Path, hour: str) -> bool:
    return (store / store_name(attempt, hour)).exists()


def build_one(attempt: Path, hour: str, store: Path, timeout: int) -> tuple[bool, str]:
    """Run the analysis for one instance-hour and move its rollup into the store."""
    out = store / ".work" / store_name(attempt, hour).removesuffix(".json")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        run = subprocess.run(
            [sys.executable, str(REPORT), str(attempt), "--hour", hour,
             "--json", "--json-only", "--out", str(out)],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timed out after %ds" % timeout
    if run.returncode != 0:
        return False, (run.stderr or run.stdout).strip().splitlines()[-1:] and \
            (run.stderr or run.stdout).strip().splitlines()[-1] or "rc=%d" % run.returncode
    produced = out / "rollup.json"
    if not produced.exists():
        return False, "no rollup.json produced"
    # Written to a temporary name and renamed, so a reader never sees a half
    # written file: on this filesystem rename within a directory is atomic and
    # a partial read is not something the dashboard could detect.
    final = store / store_name(attempt, hour)
    tmp = final.with_suffix(".json.part")
    tmp.write_bytes(produced.read_bytes())
    os.replace(tmp, final)
    produced.unlink(missing_ok=True)
    try:
        out.rmdir()
    except OSError:
        pass
    return True, "%.0fs, %.0f KB" % (time.time() - started, final.stat().st_size / 1024)


def stale_lock(lock: Path, timeout: int) -> bool:
    """Whether a lock left by an earlier run may be taken over.

    Age alone is a poor test in both directions: a builder killed one second in
    holds the store for half an hour, and a builder genuinely wedged is trusted
    for exactly as long. So the holder writes its host and pid, and a lock whose
    process is demonstrably gone is stale at once.

    The pid is only meaningful on the host that wrote it -- this runs from the
    login node and the copier node both -- so a lock from elsewhere falls back
    to the age test rather than being believed or disbelieved on a pid that
    means nothing here.
    """
    try:
        host, pid, _ = lock.read_text().split()
    except (OSError, ValueError):
        return True  # unreadable: it cannot be telling us anything
    if host == socket.gethostname():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return True
        except (PermissionError, ValueError):
            pass  # alive but not ours, or unparseable: fall through to age
        else:
            return False
    try:
        return (time.time() - lock.stat().st_mtime) >= timeout
    except OSError:
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--days", type=int, default=2, help="how many day directories back to look")
    parser.add_argument("--limit", type=int, default=4,
                        help="most hours to build in one invocation; the rest wait for the next")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds for one instance-hour")
    parser.add_argument("--list", action="store_true", help="show what is missing and exit")
    args = parser.parse_args()

    args.store.mkdir(parents=True, exist_ok=True)
    # One builder at a time. Each is a core for seven minutes; several at once
    # on a node other people are using is not worth the wall clock saved.
    lock = args.store / ".lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if not stale_lock(lock, args.timeout):
            # 75, not 0. A caller that asked for twelve rollups and got none
            # because someone else held the lock has not had its request
            # carried out, and a script that reports success either way turns
            # "it did nothing" into something you only find by checking the
            # store afterwards -- which is how this was noticed.
            print("another builder is running: %s" % lock.read_text().strip())
            return 75
        print("taking over a stale lock: %s" % lock.read_text().strip())
        lock.unlink(missing_ok=True)
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, ("%s %d %s\n" % (socket.gethostname(), os.getpid(),
                                      time.strftime("%FT%TZ", time.gmtime()))).encode())
    os.close(handle)

    try:
        missing = []
        for attempt in attempts(args.trace_root, args.days):
            for hour in finished_hours(attempt):
                if not already(args.store, attempt, hour):
                    missing.append((attempt, hour))
        # Newest first: a dashboard is asked about the last hour far more often
        # than about the one before lunch.
        missing.sort(key=lambda pair: pair[1], reverse=True)
        print("%d hour(s) missing from %s" % (len(missing), args.store))
        if args.list:
            for attempt, hour in missing[:40]:
                print("  %s %s" % (hour, attempt.parent.name))
            return 0
        for attempt, hour in missing[:args.limit]:
            ok, note = build_one(attempt, hour, args.store, args.timeout)
            print("  %-7s %s %s  (%s)" % ("built" if ok else "FAILED",
                                          hour, attempt.parent.name, note))
        return 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
