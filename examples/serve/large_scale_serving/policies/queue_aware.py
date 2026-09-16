# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spread conversations, but hold back the backends that are queueing now.

`least_conversations` counts sessions, and a session is not load: an agent
spends most of a turn thinking, so fifty idle ones read exactly like fifty
busy ones. On this fleet that let a 2-node aggregated instance accumulate the
same share as the 10-node disaggregated ones, and its queue time reached 63s
against 4.7-13.1s for the others while its session count looked unremarkable.

That queue is what costs campaigns. The agents that failed here did not fail
because tokens came out slowly; they failed because whole responses took a
median of 31.6s against 2.9s for the ones that completed, and the client hung
up -- 635 times, every one of them a client-side close.

So placement still spreads, but a backend whose queue is materially worse than
the least-queued backend *right now* is passed over until it recovers.

**Compared across backends at one instant, never against a backend's own
history.** That distinction is the whole design, and the previous attempt got
it wrong. A per-backend baseline works for per-token time, which has a
physical floor set by the model and the hardware. Queue time has no such
floor: its baseline is set by load, so a chronically congested backend learns
its own congestion as normal. Measured on this fleet, a p10-based floor put
the aggregated instance's trigger at 19.4s -- above its own p90 -- so it was
judged saturated 6.3% of the time while a healthy instance with a faster idle
floor was judged saturated 78% of the time. Exactly backwards: it would have
sent traffic toward the most congested backend.

Comparing peers at one instant has neither problem. It needs no history, it
cannot be poisoned by a backend's own past, and it adapts to architecture
differences for free because the comparison is against whatever else is
serving the same traffic at the same moment.
"""

import os
import time

from fill_first import MAX_AGE_S, _metrics

# A backend is held back when its queue time is this much worse than the
# least-queued backend in the same reading. Generous on purpose: the point is
# to catch an instance that is genuinely behind its peers, not to chase noise
# between two that are both fine.
# 8, from replaying seven hours of this fleet's own queue readings through the
# policy. At 3 the aggregated instance was held back 98.9% of the time, which
# is a permanent drain by another name and would have cost the architecture
# comparison it exists for; at 8 it still takes 10.4% of placements while the
# healthy instances split the rest evenly at ~17.7% each. That it is held at
# all at a factor of eight is the finding, not the tuning: it is genuinely
# that much worse than its peers, not noise.
QUEUE_RATIO = float(os.environ.get("QUEUE_AWARE_RATIO", "8.0"))

# ...and never for a queue shorter than this, whatever the ratio. Without it,
# 0.05s against 0.01s is a factor of five that nobody can feel, and the policy
# would shuffle traffic over measurement noise.
QUEUE_FLOOR_S = float(os.environ.get("QUEUE_AWARE_FLOOR", "1.5"))

# At most this fraction of the fleet may be held back at once. A fleet-wide
# slowdown moves every backend together, and holding back all but one would
# turn a slow fleet into an overloaded single instance -- the opposite of the
# intent, and the failure mode a relative test invites.
MAX_HELD = float(os.environ.get("QUEUE_AWARE_MAX_HELD", "0.5"))


def _queues(accepting, stats, now):
    """Current queue time per candidate, for those that have a fresh one."""
    out = {}
    for job in accepting:
        stat = stats.get(job)
        if not isinstance(stat, dict):
            continue
        if now - stat.get("ts", 0) > MAX_AGE_S:
            continue
        queue = stat.get("queue_s")
        if queue is not None:
            out[job] = queue
    return out


def select(accepting):
    now = time.time()
    queues = _queues(accepting, _metrics(now), now)

    held = set()
    if len(queues) >= 2:
        best = min(queues.values())
        threshold = max(QUEUE_FLOOR_S, QUEUE_RATIO * max(best, 0.01))
        # Worst first, so that when the cap binds it is the worst offenders
        # that are held and not whichever happened to sort first.
        for job, queue in sorted(queues.items(), key=lambda kv: -kv[1]):
            if len(held) >= int(len(accepting) * MAX_HELD):
                break
            if queue > threshold:
                held.add(job)

    # Everything a backend is not being held back for is still decided the way
    # least_conversations decides it: fewest conversations, then fewest
    # in-flight, then job id so the choice is stable.
    candidates = [j for j in accepting if j not in held] or list(accepting)
    return min(candidates, key=lambda j: (accepting[j][0], accepting[j][1], j))
