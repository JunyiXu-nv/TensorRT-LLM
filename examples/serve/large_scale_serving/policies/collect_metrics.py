#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Feed fill_first.py with what the engines actually say about their load.

The router is handed conversation and in-flight counts, and neither answers
"is this instance out of room". The engines do answer it, on
`/prometheus/metrics`, but a routing policy runs on the request path and must
not make network calls -- so this polls them out of band and leaves a small
file behind.

Run it anywhere that can reach the workers (the gateway node, or a login node
on the same cluster) and point `--out` somewhere the gateway can read:

    ./collect_metrics.py --gateway http://10.109.43.2:8333 --key junyix \\
        --out /path/on/shared/fs/kf-fleet-metrics.json --interval 5

Two counters carry the signal, both read as deltas between polls rather than
as the since-boot averages the raw values hold. An instance that has been up
for hours has an average dominated by whatever it was doing hours ago; the
delta is what it is doing now.

  trtllm_request_queue_time_seconds   {sum,count} -> mean WAITING time per
      request. A batching scheduler makes a request wait only when it cannot
      fit it, so this is saturation itself rather than a proxy for it.
  trtllm_time_per_output_token_seconds {sum,count} -> mean inter-token time.
      Rises smoothly with batch size, which makes it the better signal for
      "working hard but still fine" -- the regime a fill wants to stay in
      until it ends.

Disaggregated instances are polled per worker and aggregated: queue time is
the worst of them, since a request waits behind whichever stage is busy, while
per-token time comes from the generation workers alone -- context workers
produce no output tokens and report a meaningless zero.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

QUEUE = "trtllm_request_queue_time_seconds"
TPOT = "trtllm_time_per_output_token_seconds"

# A delta over a few requests is not a measurement. Measured on this fleet at
# low load, eight-second windows holding a handful of completions swung the
# mean queue time between 0.1s and 9.6s on one backend and moved another's
# per-token time by 17x -- neither reflected anything the instance was doing
# differently. Below this many completions the window is discarded and the
# previous smoothed value stands.
MIN_SAMPLES = 20

# Exponential smoothing on top of that, because even a well-populated window
# is one window. Low enough to follow a real change within a few polls, high
# enough that one burst does not move the verdict.
ALPHA = 0.3

# The saturation test compares a backend against its own best, so a single
# spuriously fast window becomes a permanently wrong baseline -- every later
# reading looks like a 1.5x degradation and the backend is written off while
# idle. The floor only moves on a window that carried real traffic, and never
# below a fraction of what has already been established.
BEST_FLOOR_RATIO = 0.5

# A field whose last qualifying window is older than this stops being
# published. Without it a quiet backend keeps whatever figure it last managed
# to measure, under a write-time stamp that makes it look current forever.
STALE_FIELD_S = 90.0


def fetch(url, timeout, key=None):
    req = urllib.request.Request(url)
    if key:
        req.add_header("x-api-key", key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def parse_prom(text, name):
    """(sum, count) for one histogram, ignoring its buckets and labels."""
    total = count = None
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        head, _, value = line.rpartition(" ")
        metric = head.split("{", 1)[0]
        if metric == name + "_sum":
            try:
                total = float(value)
            except ValueError:
                pass
        elif metric == name + "_count":
            try:
                count = float(value)
            except ValueError:
                pass
    return total, count


def workers_of(url, timeout):
    """The engines behind one backend URL.

    A disaggregated backend is a proxy and publishes its workers; an
    aggregated one is the engine, and answers 404 here.
    """
    try:
        info = json.loads(fetch(url + "/cluster_info", timeout))
    except (urllib.error.URLError, OSError, ValueError):
        return [(url, True)]
    lists = info.get("server_lists")
    if not isinstance(lists, dict):
        return [(url, True)]
    out = []
    for kind, hosts in lists.items():
        generating = kind != "context"
        for host in hosts or []:
            out.append(("http://" + host, generating))
    return out or [(url, True)]


def poll(backends, prev, timeout):
    """One sweep. `prev` carries the last raw counters and the best-ever TPOT."""
    now = time.time()
    out = {}
    for job, url in sorted(backends.items()):
        worst_queue = None
        tpot_sum = tpot_count = 0.0
        seen = False
        for target, generating in workers_of(url, timeout):
            try:
                text = fetch(target + "/prometheus/metrics", timeout)
            except (urllib.error.URLError, OSError):
                continue
            seen = True
            key = job + "|" + target
            last = prev.get("raw", {}).get(key, {})
            q_sum, q_count = parse_prom(text, QUEUE)
            t_sum, t_count = parse_prom(text, TPOT)
            prev.setdefault("raw", {})[key] = {
                "q_sum": q_sum,
                "q_count": q_count,
                "t_sum": t_sum,
                "t_count": t_count,
            }
            # A counter that went backwards means the worker restarted; its
            # delta is meaningless, so skip this round rather than record a
            # negative or a spike.
            if q_sum is not None and q_count is not None and last.get("q_count") is not None:
                dc = q_count - last["q_count"]
                ds = q_sum - last["q_sum"]
                if dc >= MIN_SAMPLES and ds >= 0:
                    mean = ds / dc
                    worst_queue = mean if worst_queue is None else max(worst_queue, mean)
            if generating and t_sum is not None and last.get("t_count") is not None:
                dc = t_count - last["t_count"]
                ds = t_sum - last["t_sum"]
                if dc >= MIN_SAMPLES and ds >= 0:
                    tpot_sum += ds
                    tpot_count += dc
        if not seen:
            continue
        smooth = prev.setdefault("smooth", {}).setdefault(job, {})
        if worst_queue is not None:
            smooth["queue_s"] = (
                worst_queue
                if "queue_s" not in smooth
                else (1 - ALPHA) * smooth["queue_s"] + ALPHA * worst_queue
            )
            # Stamped when the value moved, not when the file was written. A
            # backend too quiet to clear MIN_SAMPLES keeps its last figure,
            # and a fresh write-time stamp on it would present a reading from
            # ten minutes ago as current -- which is exactly how one 19-second
            # queue measurement stayed authoritative on an idle instance here.
            smooth["queue_ts"] = now
        if tpot_count > 0:
            tpot = tpot_sum / tpot_count
            smooth["tpot_s"] = (
                tpot if "tpot_s" not in smooth else (1 - ALPHA) * smooth["tpot_s"] + ALPHA * tpot
            )
            smooth["tpot_ts"] = now
            # The floor this backend has been seen to reach, which is what
            # "saturated" is measured against. Kept per backend on purpose:
            # a 2-node aggregated instance and an 8-node disaggregated one do
            # not share a baseline, and comparing them to each other declares
            # the small one full while it is idle.
            #
            # Moved on the smoothed value and never more than BEST_FLOOR_RATIO
            # below what is already established, so one fast window cannot
            # define a floor the backend can never reach again.
            best = prev.setdefault("best", {})
            candidate = smooth["tpot_s"]
            if candidate > 0:
                if job not in best:
                    best[job] = candidate
                elif candidate < best[job]:
                    best[job] = max(candidate, best[job] * BEST_FLOOR_RATIO)

        # Nothing measurable this round: leave the previous reading in place
        # rather than publishing a gap the policy would read as "no evidence".
        if not smooth:
            continue
        # Each field carries the age of its own last real measurement, and a
        # field whose measurement has gone stale is simply not published --
        # the policy then falls back rather than acting on a figure nothing
        # has confirmed since.
        stat = {"ts": now}
        for field in ("queue_s", "tpot_s"):
            stamped = smooth.get(field[:-2] + "_ts")
            if field in smooth and stamped is not None and now - stamped <= STALE_FIELD_S:
                stat[field] = round(smooth[field], 6)
                stat[field[:-2] + "_age_s"] = round(now - stamped, 1)
        if job in prev.get("best", {}):
            stat["best_tpot_s"] = round(prev["best"][job], 6)
        out[job] = stat
    return out


def write_atomic(path, payload):
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gateway", required=True)
    ap.add_argument("--key", default=os.environ.get("USER", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=4.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    prev = {}
    while True:
        try:
            fleet = json.loads(fetch(args.gateway + "/_gateway/fleet", args.timeout, args.key))[
                "backends"
            ]
            backends = {j: v["url"] for j, v in fleet.items() if v.get("healthy")}
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            print("fleet unreadable: %s" % exc, file=sys.stderr, flush=True)
            backends = {}
        if backends:
            stats = poll(backends, prev, args.timeout)
            write_atomic(args.out, stats)
            print(
                "%s %s"
                % (
                    time.strftime("%H:%M:%S"),
                    " ".join(
                        "%s q=%.3f t=%.4f%s"
                        % (
                            j,
                            s.get("queue_s", -1),
                            s.get("tpot_s", -1),
                            "" if "best_tpot_s" not in s else "/%.4f" % s["best_tpot_s"],
                        )
                        for j, s in sorted(stats.items())
                    ),
                ),
                flush=True,
            )
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
