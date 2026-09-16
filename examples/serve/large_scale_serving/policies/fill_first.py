# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fill one backend until it is actually saturated, then move to the next.

For benchmarking. The default policy spreads conversations so that no instance
is ever the bottleneck, which is the opposite of what a capacity measurement
needs: to find out what one instance can carry you have to push it until it
stops keeping up, and only then start on the next one.

`select` is handed `(conversations, inflight, remaining_seconds)` per backend
and nothing else, and neither of the first two is a measure of load:

  conversations   an agent spends most of a turn thinking, so a backend
                  holding fifty idle sessions reads exactly like one holding
                  fifty active ones.
  inflight        counts requests the gateway is proxying this instant, which
                  tracks work far better -- but its ceiling is the engine's
                  max batch size, and a backend that has *reached* that
                  ceiling looks identical to one that is merely busy. The
                  queue behind it is where the difference lives, and the
                  gateway cannot see it.

So saturation is read from the engines themselves. A collector polls each
backend's `/prometheus/metrics` and writes the recent per-backend figures to
`--metrics-file`; this policy only reads that file, so the request path costs
one cached JSON load and no network.

Two signals, both rate-of-change over the collector's interval rather than the
since-boot averages the counters hold:

  queue_s   mean time a request spent WAITING before it was scheduled. Zero
            while the scheduler has room, and the first thing to move when it
            does not. This is the definition of saturation for a batching
            engine.
  tpot_s    mean time per output token. Rises with batch size, so it says how
            hard the instance is working rather than whether it is over.

**Each backend is compared against its own best, never against the fleet.**
This fleet runs a 2-node aggregated instance beside 8-node disaggregated ones;
measured cold, their per-token times differ by 5x and their queue times by 20x.
Any single fleet-wide threshold declares the small one saturated while it is
idle, and never declares the large one saturated at all.
"""

import json
import os
import time

# Where the collector writes. The default is on the shared filesystem rather
# than in /tmp on purpose: the gateway and the collector do not run on the same
# machine, so a local path means the policy reads a file nobody is writing and
# silently decides it has no evidence. Overridable so a second experiment can
# run against its own file without editing this one.
METRICS_FILE = os.environ.get(
    "FILL_FIRST_METRICS",
    "/scratch/fsw/portfolios/coreai/projects/coreai_kf_dev/users/junyix/TensorRT-LLM"
    "/examples/serve/large_scale_serving/var/gw/kf-fleet-metrics.json",
)

# A backend counts as saturated once its per-token time is this much worse
# than the best it has been seen to do. 1.5 is deliberately generous: batching
# buys throughput at the cost of latency, so some rise is the system working
# as intended, not the instance running out.
TPOT_KNEE = float(os.environ.get("FILL_FIRST_TPOT_KNEE", "1.5"))

# ...or once requests start waiting this long to be scheduled.
#
# A backstop, not the primary signal, and the number is not a measurement.
# This fleet's disaggregated instances run six context workers to one
# generation worker, so a request waits behind the generation stage as a
# matter of course: measured with the launcher stopped and only residual
# traffic left, queue times were already 0.25s to 2.4s. A ceiling anywhere
# near those declares every instance saturated before the fill starts, which
# is why the knee above carries the decision and this only catches the case
# where an instance has stopped scheduling altogether.
#
# Calibrate it against the run you are about to do rather than trusting this:
# watch collect_metrics.py through one instance's ramp, and set it above what
# that instance does while it is still keeping up.
QUEUE_CEILING_S = float(os.environ.get("FILL_FIRST_QUEUE_CEILING", "8.0"))

# Readings older than this are not evidence. The collector runs every few
# seconds; a file this stale means it has stopped, and stale numbers would
# pin every conversation onto whichever backend was quiet when it died.
MAX_AGE_S = float(os.environ.get("FILL_FIRST_MAX_AGE", "60"))

# How far above a backend's own proven in-flight level a burst may push it
# before placement stops. In-flight is the only signal here that is not
# collected out of band, so it is the only one that can react inside the few
# seconds a burst takes -- queue time and per-token time are both averages
# over completed requests, and a backend buried deep enough to stop completing
# anything reports nothing at all while it is drowning.
BURST_FACTOR = float(os.environ.get("FILL_FIRST_BURST_FACTOR", "1.5"))

# Used only before a backend has proven anything: the ceiling that applies
# while there is no measured level to compare against. Deliberately generous,
# since it is a backstop against a cold-start burst rather than the mechanism
# that decides when an instance is full.
INFLIGHT_CEILING = float(os.environ.get("FILL_FIRST_INFLIGHT_CEILING", "400"))

# How often the metrics file may be stat-ed. The file has to live somewhere
# both the collector and the gateway can see, which on this deployment means
# Lustre -- and an os.stat there is a metadata round trip, not a page-cache
# hit. Checking it on every placement would put one such round trip in front
# of every new conversation. The collector writes every few seconds, so
# looking more often than this cannot learn anything anyway.
RECHECK_S = float(os.environ.get("FILL_FIRST_RECHECK", "2.0"))

_cache = {"mtime": None, "data": {}, "checked": 0.0}

# Highest in-flight each backend has been seen carrying *while healthy*: fresh
# evidence, below the knee, below the queue ceiling. That makes it a level the
# backend has demonstrated it can hold, rather than a number chosen in advance
# -- which matters because the aggregated instance in this fleet sits at 154
# in-flight while the disaggregated ones sit at 43-58, all of them fine.
_proven = {}


def _metrics(now):
    """Last reading per backend, reloaded only when the file changes."""
    if now - _cache["checked"] < RECHECK_S:
        return _cache["data"]
    _cache["checked"] = now
    try:
        mtime = os.path.getmtime(METRICS_FILE)
    except OSError:
        return {}
    if _cache["mtime"] != mtime:
        try:
            with open(METRICS_FILE) as handle:
                _cache["data"] = json.load(handle)
        except (OSError, ValueError):
            # Half-written file, or the collector is mid-replace. The previous
            # reading is better than none; the age check below retires it.
            return _cache["data"]
        _cache["mtime"] = mtime
    return _cache["data"]


def _saturated(job, stat, inflight, now):
    """True when this backend has stopped keeping up with what it already has.

    Ordered fast signal first. A burst arrives in less time than the collector
    takes to notice it, and under a fill-first policy every conversation in
    that burst lands on the same backend -- so the guard that has to hold is
    the one that needs no collector.
    """
    proven = _proven.get(job)
    fresh = isinstance(stat, dict) and now - stat.get("ts", 0) <= MAX_AGE_S

    # Evidence outranks the guard, and the order matters more than it looks.
    # The guard used to run first and return early, which meant the line below
    # that raises `proven` was unreachable the moment in-flight passed
    # 1.5x proven -- so proven froze at whatever level happened to be observed
    # first, and the backend read as saturated from then on no matter what the
    # engines said. A gradual ramp hid it (each step stayed under the factor);
    # production did not, because the launcher starts 25-60 campaigns at once
    # and in-flight arrives as a step, not a slope.
    if fresh:
        queue = stat.get("queue_s")
        tpot, best = stat.get("tpot_s"), stat.get("best_tpot_s")
        over = (queue is not None and queue > QUEUE_CEILING_S) or bool(
            tpot and best and tpot > TPOT_KNEE * best
        )
        if not over and inflight > 0:
            # Measured healthy at this level, so this level is proven --
            # including when it is well above the last one. The engines are a
            # better witness than a factor.
            if inflight > _proven.get(job, 0):
                _proven[job] = inflight
        return over

    # No current evidence. This is where the burst guard belongs: it exists to
    # cover the seconds the collector needs to notice, not to overrule it.
    ceiling = BURST_FACTOR * proven if proven else INFLIGHT_CEILING
    if inflight > ceiling:
        return True
    # Still no evidence, and past the guard. An earlier version called this
    # "not saturated" on the grounds that it degrades to filling in order --
    # which under this policy *is* overloading the first one. A backend nobody
    # can measure is not one to keep loading, unless it is carrying nothing yet.
    return inflight > 0 and proven is not None


def select(accepting):
    now = time.time()
    stats = _metrics(now)

    # Deterministic order, so the fill sequence is the same on every call and
    # across gateway generations -- a benchmark that filled a different
    # instance each run would measure nothing.
    order = sorted(accepting)

    for job in order:
        if not _saturated(job, stats.get(job), accepting[job][1], now):
            return job

    # Everything is saturated: this is no longer a fill, it is a fleet at
    # capacity, and the least-loaded backend is the only sensible answer.
    # inflight rather than conversations, since at this point every backend
    # has plenty of both and only one of them is work.
    return min(order, key=lambda j: (accepting[j][1], accepting[j][0], j))
