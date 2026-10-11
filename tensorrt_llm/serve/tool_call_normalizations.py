# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A per-call record of what serving changed between a generated tool call and the one delivered.

A tool call reaches the client as the parser read it and the Responses layer shaped it, and
both repair or reshape on the way: a call read without its close tags, an argument whose
declared type is not JSON's, a custom tool's extra keys dropped or a sole key taken as its
``input``. Each repair is reasonable for serving -- a near-miss is better run than refused --
but a training export reading the delivered call must be able to tell it from a call the model
wrote correctly. Warnings in the server log are not that: they carry no call identity.

So every delivered call with at least one such event gets one line, keyed by the ``call_id``
the client saw (which is in the request trace's response body), written next to the raw model
output when ``TRTLLM_RAW_OUTPUT_DIR`` is set::

    $TRTLLM_RAW_OUTPUT_DIR/
      2026-10-11T03/
        normalize-<pid>.jsonl    one line per normalized call

A call is reported to the client twice (in the stream and in the final response), so the line
is written once per ``call_id``.
"""

import json
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tensorrt_llm.logger import logger

# The raw output dump's directory (tensorrt_llm.executor.raw_output_hook.RAW_OUTPUT_DIR_ENV):
# these lines belong beside the generations they describe.
_RAW_OUTPUT_DIR_ENV = "TRTLLM_RAW_OUTPUT_DIR"
_HOUR_BUCKET_LEN = 13
# Enough to cover every call a worker can have in flight; older ids are forgotten.
_SEEN_LIMIT = 65536

_lock = threading.Lock()
_seen: "OrderedDict[str, None]" = OrderedDict()
_dropped = 0


def record_tool_call_normalizations(
    call_id: str,
    item_id: Optional[str],
    name: Optional[str],
    events: List[Dict[str, Any]],
) -> None:
    """Write one line for a delivered call that serving changed. Never raises."""
    global _dropped
    if not events:
        return
    directory = os.environ.get(_RAW_OUTPUT_DIR_ENV)
    if not directory:
        return
    with _lock:
        if call_id in _seen:
            return
        _seen[call_id] = None
        if len(_seen) > _SEEN_LIMIT:
            _seen.popitem(last=False)
    recorded_at = datetime.now(timezone.utc).isoformat()
    record = {
        "event": "tool_call_normalization",
        "recorded_at": recorded_at,
        "call_id": call_id,
        "item_id": item_id,
        "name": name,
        "normalizations": events,
    }
    try:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        bucket = os.path.join(directory, recorded_at[:_HOUR_BUCKET_LEN])
        os.makedirs(bucket, exist_ok=True)
        with open(
            os.path.join(bucket, f"normalize-{os.getpid()}.jsonl"), "a", encoding="utf-8"
        ) as output:
            output.write(line)
    except (OSError, TypeError, ValueError) as error:
        _dropped += 1
        if _dropped == 1 or _dropped % 1000 == 0:
            logger.warning(f"Dropped {_dropped} tool call normalization record(s): {error}")
