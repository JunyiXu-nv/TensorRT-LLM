# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM first: fill each GLM-5.3 backend up to a target concurrency, then the rest.

Purpose: measure what one GLM-5.3 NVFP4 4P1D DEP16 instance carries. It is
expected to hold about 300 concurrent requests. New conversations go to GLM
first so its behaviour near that load can be watched. The other pool (K3)
keeps the placement the gateway used before, least_inflight, for whatever GLM
does not take. Conversations stay pinned to the backend that first took them,
as with every policy.

The router asks for a placement only for a conversation it has not pinned:
a new one, or one whose backend went away and is being re-homed. In the
re-homing case, pool affinity has already narrowed the offer to the
conversation's own pool. Placement goes:

  1. If some GLM backend is below both caps, pick the one with the most
     requests in flight. Filling one instance before the next is what brings
     an instance to the target with the traffic the fleet has; spreading would
     leave every GLM instance at a fraction of it.
  2. Otherwise, pick among the other pools' backends by least_inflight. Ties
     are broken by conversations, then by job id.
  3. If only GLM backends are on offer and all are at the caps, pick by
     least_inflight among them.

Caps:
  CAP_INFLIGHT       requests the gateway is proxying to the backend now.
  CAP_CONVERSATIONS  conversations pinned to the backend. For agent traffic,
                     in-flight requests run at roughly a third to a half of
                     the pinned conversations, because tools run between turns.
                     Without this cap a backend could collect far more
                     conversations than its in-flight figure suggests, and
                     overshoot once they all send a turn.

To change a number, edit this file. The gateway reloads it when it changes.
"""

GLM_POOL = "glm5.3"
CAP_INFLIGHT = 300
CAP_CONVERSATIONS = 600


def _stat(loads, index):
    return loads[index] if len(loads) > index else 0


def _least_inflight(jobs, accepting):
    return min(jobs, key=lambda j: (_stat(accepting[j], 1), _stat(accepting[j], 0), j))


def select(accepting, pools):
    glm = [j for j in accepting if pools.get(j) == GLM_POOL]
    under = [
        j
        for j in glm
        if _stat(accepting[j], 1) < CAP_INFLIGHT and _stat(accepting[j], 0) < CAP_CONVERSATIONS
    ]
    if under:
        return max(under, key=lambda j: (_stat(accepting[j], 1), _stat(accepting[j], 0), j))
    others = [j for j in accepting if pools.get(j) != GLM_POOL]
    return _least_inflight(others or glm, accepting)
