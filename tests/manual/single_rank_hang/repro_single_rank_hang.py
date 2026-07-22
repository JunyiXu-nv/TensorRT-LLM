#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-rank-failure hang repro DRIVER (needs >=2 GPUs).

Runs a real TP=2 LLM generate(). One rank is wedged mid-decode via the
env-gated fault injection in PyExecutor (TLLM_DEBUG_WEDGE_RANK), so its peers
block in the next NCCL collective. The merged HangDetector should then fire on
every rank and hard-kill (MPI_Abort/SIGKILL) the whole job -- freeing all GPUs
in ~TLLM_DEBUG_HANG_TIMEOUT instead of zombie-holding them until the wall clock.

Launched as a subprocess by run_single_rank_hang.sh. Env used:
  REPRO_MODEL                  HF/local model dir (default: Llama-3.1-8B-Instruct)
  REPRO_TP                     tensor_parallel_size (default 2)
  TLLM_DEBUG_WEDGE_RANK        rank to wedge (default 1)  [read by PyExecutor]
  TLLM_DEBUG_WEDGE_AFTER_STEPS wedge after N loop steps (default 5)
  TLLM_DEBUG_HANG_TIMEOUT      HangDetector timeout in seconds (e.g. 30)
"""

import atexit
import os
import sys
import time

atexit.register(
    lambda: print(
        "[driver] python atexit reached -- any hang after this is C-level (MPI_Finalize)",
        flush=True,
    )
)

from tensorrt_llm import (  # noqa: E402  (atexit marker must register before this import)
    LLM,
    SamplingParams,
)


def main() -> int:
    model = os.environ["REPRO_MODEL"]
    tp = int(os.environ.get("REPRO_TP", "2"))
    print(
        f"[driver] pid={os.getpid()} loading model={model} tp={tp} "
        f"wedge_rank={os.environ.get('TLLM_DEBUG_WEDGE_RANK')} "
        f"hang_timeout={os.environ.get('TLLM_DEBUG_HANG_TIMEOUT')}s",
        flush=True,
    )
    extra = {}
    # Escape hatches for environments where TP init wedges (observed on an
    # H100-PCIe pair: both ranks spin forever inside the tunable_allreduce
    # warmup probe - py-spy: autotuner.__call__ -> torch_custom_ops:2247).
    strat = os.environ.get("REPRO_ALLREDUCE_STRATEGY")
    if strat:
        extra["allreduce_strategy"] = strat
    pp = int(os.environ.get("REPRO_PP", "1"))
    if pp > 1:
        # PP uses NCCL send/recv only - no custom allreduce on the TP path.
        extra["pipeline_parallel_size"] = pp
    llm = LLM(model=model, tensor_parallel_size=tp, max_batch_size=8, **extra)
    print(
        "[driver] model ready; starting generate() -- the wedge fires a few "
        "decode steps in, peers should hang, detector should hard-kill.",
        flush=True,
    )
    t0 = time.time()
    # Long generation so many decode steps run before completion.
    try:
        outs = llm.generate(
            ["Write a very long, detailed story about a dragon and a knight."] * tp,
            SamplingParams(max_tokens=1024),
        )
    # Reaching here means the hard-kill did NOT happen (repro failed to trigger).
    except Exception as e:
        import threading

        print(
            f"[driver] generate() raised {type(e).__name__} at +{time.time() - t0:.1f}s: {e}",
            flush=True,
        )
        for th in threading.enumerate():
            print(f"[driver] live thread: {th.name} daemon={th.daemon}", flush=True)
        try:
            from mpi4py.futures import _lib as _fl

            print("[driver] THREADS_QUEUES size:", len(_fl.THREADS_QUEUES), flush=True)
        except Exception as ie:
            print("[driver] THREADS_QUEUES probe failed:", ie, flush=True)
        _locks = getattr(threading, "_shutdown_locks", None)
        _mgr = [t2 for t2 in threading.enumerate() if "_manager" in t2.name]
        if _mgr and _locks is not None:
            print(
                "[driver] manager tstate_lock in _shutdown_locks:",
                _mgr[0]._tstate_lock in _locks,
                flush=True,
            )
        import faulthandler

        faulthandler.dump_traceback_later(120, exit=False)
        print("[driver] returning 1; watch how long interpreter exit takes...", flush=True)
        return 1
    print(
        f"[driver] UNEXPECTED_COMPLETION in {time.time() - t0:.1f}s (no hard-kill occurred)",
        flush=True,
    )
    for o in outs:
        print("[driver] out:", o.outputs[0].text[:60], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
