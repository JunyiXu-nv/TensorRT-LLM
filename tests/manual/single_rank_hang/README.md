# Single-rank hang e2e — HangDetector hard-kill validation

Manual (GPU-required) end-to-end validation that the in-runtime **HangDetector**
(merged in PR #15612) hard-kills the **whole MPI world** when a single rank
stops making progress, instead of letting peer ranks zombie-hold their GPUs
until the CI wall-clock kill.

## How the hang is simulated

The branch carries one DEBUG-ONLY commit that adds an env-gated fault injection
to `PyExecutor` (`tensorrt_llm/_torch/pyexecutor/py_executor.py`,
`_maybe_debug_wedge_rank()`), called right after `hang_detector.checkpoint()`
in all three executor loops:

- When `TLLM_DEBUG_WEDGE_RANK` matches the rank, that rank enters
  `while True: time.sleep(3600)` after `TLLM_DEBUG_WEDGE_AFTER_STEPS`
  (default 5) executor-loop iterations — a **forever sleep mid-decode**.
- The wedged rank stops issuing collectives, so every peer blocks in its next
  NCCL/MPI op; no rank's loop checkpoints anymore, and each rank's
  HangDetector fires after its timeout → `propagate_hard_kill()` →
  `MPI_Abort` tears down the entire job.
- `TLLM_DEBUG_HANG_TIMEOUT` optionally overrides the detector timeout for
  faster iteration. **Leave it at 300 to validate the production default.**
- All injection is env-gated: with the variables unset the code is inert.

## Running it

Requires ≥2 GPUs and a built TRT-LLM (the injection is pure Python — an
existing container/venv of a recent main works; just make sure this branch's
`py_executor.py`/`hang_detector.py` are the ones imported).

```bash
export LLM_MODELS_ROOT=/path/to/models   # needs llama-3.1-model/Llama-3.1-8B-Instruct (or set REPRO_MODEL)
export TLLM_DEBUG_HANG_TIMEOUT=300       # production default; use 30 for quick iteration
export OUTER_TIMEOUT=1500                # wall-clock cap: model load + wedge + 300s + teardown
bash tests/manual/single_rank_hang/run_single_rank_hang.sh
```

The runner launches a real TP=2 `LLM.generate()` (driver:
`repro_single_rank_hang.py`), wedges rank 1 a few decode steps in, and polls
GPU memory + driver liveness every 3s. **PASS** = summed GPU memory returns to
baseline (all ranks SIGKILLed) roughly `TLLM_DEBUG_HANG_TIMEOUT` seconds after
the wedge, far below the outer cap.

Expected log sequence (driver log at `/tmp/repro_driver.log`):

```
[TLLM_DEBUG_WEDGE] Wedging rank 1 after 5 steps ...
[RANK ...] Hang detected after 300 seconds.        <- every rank, ~300s later
[RANK ...] Thread ... stack trace: ...             <- diagnosis dump
HangDetector: propagating hard-kill to all ranks via MPI_Abort.
MPI_ABORT was invoked on rank ...
```

## Validation record

- 2026-07-02, ipp1-3396 (2×H100): wedge rank 1 → all ranks dead @30s override.
- 2026-07-16, OCI GB200 (4 GPU): detector kill + #16312 teardown → client
  `EngineDeadError` +5s, process exit +22s after the kill.
- 2026-07-21, OCI GB200 (4 GPU): this branch, default **300s** timeout —
  see the PR/branch description for the measured timeline.

## Environment escape hatches (added during 07-22 computelab validation)

- `REPRO_PP=2` (+ `REPRO_TP=1`): run the world as pipeline-parallel instead of
  tensor-parallel. Use when the TP init path wedges on the node (observed on an
  H100-PCIe pair: both ranks spin forever inside the `tunable_allreduce`
  warmup probe — a REAL init-phase hang the detector cannot see).
- `REPRO_ALLREDUCE_STRATEGY=NCCL`: forwarded to `LLM(allreduce_strategy=...)`.
- `NCCL_P2P_DISABLE=1`: required on nodes with broken PCIe P2P (bare 2-rank
  NCCL allreduce fails with P2P on, passes with it off).

## 300s-default validation record (2026-07-22, computelab a4u8g-0120, 2×H100-PCIe)

`REPRO_TP=1 REPRO_PP=2 NCCL_P2P_DISABLE=1 TLLM_DEBUG_HANG_TIMEOUT=300`:
wedge rank 1 at 03:56:50 → `Hang detected after 300 seconds` on BOTH ranks at
04:01:50 (exactly +300s; rank 0 caught via its own stalled loop) → `MPI_Abort`
→ client `EngineDeadError` +305s → GPUs at baseline t=365s → driver exit
t=386s. **PASS (ST-1)** — vs the 2800s wall-clock cap.
