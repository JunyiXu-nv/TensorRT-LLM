# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend-owned post-processing pool: incremental detok, stop truncation, hooks, and metrics.

Replaces rank-0 ``PostprocWorker`` spawning on the ``trtllm.serve_engine.v1`` Proto path so
the engine stays token-only and never needs a tokenizer or text-processing worker pool.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from google.protobuf.json_format import MessageToDict

from tensorrt_llm.contracts.serve_engine.v1 import common_pb2, output_pb2

from .postprocessor_hook import apply_post_processor_hook


@dataclass(slots=True)
class PostprocStepVerdict:
    """Outcome of one frontend post-processing pass over a ``ProtoGenerationStream``."""

    should_cancel: bool = False
    cancel_reason: int = common_pb2.CANCEL_REASON_UNSPECIFIED
    force_done: bool = False


def _iter_text_stop_strings(sampling_params: Any) -> list[str]:
    if sampling_params is None:
        return []
    stop = getattr(sampling_params, "stop", None)
    if not stop:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    return [s for s in stop if isinstance(s, str) and s]


def _apply_detok_and_stop(
    out: Any,
    *,
    tokenizer: Any,
    sampling_params: Any,
    streaming: bool,
    is_final: bool,
    stop_index: int,
    has_hook: bool,
) -> tuple[bool, bool]:
    """Run detokenization and stop-string truncation on one sequence output.

    Returns ``(matched_text_stop, already_finished)``.
    """
    detok = (
        bool(getattr(sampling_params, "detokenize", True)) if sampling_params is not None else True
    )
    if not (detok or has_hook) or tokenizer is None:
        return False, is_final

    skip_special = (
        bool(getattr(sampling_params, "skip_special_tokens", True))
        if sampling_params is not None
        else True
    )
    spaces_between = (
        bool(getattr(sampling_params, "spaces_between_special_tokens", True))
        if sampling_params is not None
        else True
    )
    use_beam = (
        bool(getattr(sampling_params, "use_beam_search", False))
        if sampling_params is not None
        else False
    )
    trimmed = len(out.token_ids) < out._last_token_ids_len
    use_inc = (
        streaming and not use_beam and not trimmed and hasattr(tokenizer, "decode_incrementally")
    )
    if use_inc:
        interval = (
            int(getattr(sampling_params, "_stream_interval", 1) or 1)
            if sampling_params is not None
            else 1
        )
        out.text, out._incremental_states = tokenizer.decode_incrementally(
            out.token_ids_diff,
            prev_text=out.text,
            states=out._incremental_states,
            flush=is_final,
            stream_interval=interval,
            skip_special_tokens=skip_special,
            spaces_between_special_tokens=spaces_between,
        )
    else:
        out.text = tokenizer.decode(
            out.token_ids,
            skip_special_tokens=skip_special,
            spaces_between_special_tokens=spaces_between,
        )
        if trimmed:
            out._incremental_states = None
            out._last_token_ids_len = len(out.token_ids)

    stop_strings = _iter_text_stop_strings(sampling_params)
    include_stop = (
        bool(getattr(sampling_params, "include_stop_str_in_output", False))
        if sampling_params is not None
        else False
    )

    # Engine-reported token-level stop_index maps directly to stop_strings[stop_index] when present.
    if stop_index >= 0 and stop_index < len(stop_strings):
        stop_str = stop_strings[stop_index]
        pos = out.text.find(stop_str)
        if pos != -1:
            out.text = out.text[: pos + (len(stop_str) if include_stop else 0)]
        out.finish_reason = "stop"
        out.stop_reason = stop_str
        return False, True

    # Text-level stop-string match: truncate and signal explicit engine cancellation (fixes H1).
    for stop_str in stop_strings:
        pos = out.text.find(stop_str)
        if pos != -1:
            out.text = out.text[: pos + (len(stop_str) if include_stop else 0)]
            out.finish_reason = "stop"
            out.stop_reason = stop_str
            return True, True

    return False, is_final


def extract_final_metrics(final_info: output_pb2.FinalInfo) -> dict[str, Any]:
    """Extract frontend usage, timing, decoding, and speculative stats from ``FinalInfo``."""
    metrics: dict[str, Any] = {}
    if final_info.HasField("usage"):
        metrics["usage"] = {
            "prompt_tokens": int(final_info.usage.prompt_tokens),
            "completion_tokens": int(final_info.usage.completion_tokens),
            "cached_tokens": int(final_info.usage.cached_tokens),
        }
    if final_info.HasField("timing"):
        metrics["timing"] = MessageToDict(final_info.timing, preserving_proto_field_name=True)
    if final_info.HasField("decoding"):
        metrics["decoding"] = {
            "decoding_iter": int(final_info.decoding.decoding_iter),
            "avg_decoded_tokens_per_iter": float(final_info.decoding.avg_decoded_tokens_per_iter),
        }
    if final_info.HasField("spec"):
        metrics["spec"] = {
            "total_drafted": int(final_info.spec.total_drafted),
            "total_accepted": int(final_info.spec.total_accepted),
            "drafted_per_pos": list(final_info.spec.drafted_per_pos),
            "accepted_per_pos": list(final_info.spec.accepted_per_pos),
        }
    return metrics


def select_best_of_outputs(outputs: list[Any], sampling_params: Any) -> list[Any]:
    """Select top ``n`` sequences by ``cumulative_logprob`` when ``best_of > n``."""
    if sampling_params is None:
        return outputs
    n = int(getattr(sampling_params, "n", len(outputs)) or len(outputs))
    best_of = getattr(sampling_params, "best_of", None)
    if best_of is None or int(best_of) <= n or len(outputs) <= n:
        return outputs
    ranked = sorted(
        outputs,
        key=lambda o: float("-inf")
        if o.cumulative_logprob is None
        else float(o.cumulative_logprob),
        reverse=True,
    )
    selected = ranked[:n]
    for idx, item in enumerate(selected):
        item.index = idx
    return selected


class FrontendPostprocPool:
    """Handle-sharded frontend post-processing executor (detok, stop, hooks, metrics).

    With ``num_workers=0``, runs inline synchronously on the caller's thread. With
    ``num_workers > 0``, dispatches to dedicated single-thread shard executors keyed by
    ``handle % num_workers`` so chunks for the same request always execute strictly in order.
    """

    def __init__(self, *, num_workers: int = 0) -> None:
        self._num_workers = max(0, int(num_workers))
        self._shards: list[ThreadPoolExecutor] = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"frontend_postproc_{i}")
            for i in range(self._num_workers)
        ]

    @property
    def num_workers(self) -> int:
        return self._num_workers

    def process_sequence(
        self,
        stream: Any,
        out: Any,
        *,
        is_final: bool,
        stop_index: int = -1,
    ) -> PostprocStepVerdict:
        def _run() -> PostprocStepVerdict:
            hook = getattr(stream, "_post_processor_hook", None)
            matched_text_stop, _ = _apply_detok_and_stop(
                out,
                tokenizer=stream._tokenizer,
                sampling_params=stream.sampling_params,
                streaming=stream._streaming,
                is_final=is_final,
                stop_index=stop_index,
                has_hook=hook is not None,
            )
            verdict = PostprocStepVerdict()
            if matched_text_stop and not is_final:
                verdict.should_cancel = True
                verdict.cancel_reason = common_pb2.CANCEL_REASON_STOP_STRING
                verdict.force_done = True

            if hook is not None:
                prev_reason = getattr(
                    stream, "_default_abort_reason", common_pb2.CANCEL_REASON_CLIENT_DISCONNECTED
                )
                stream._default_abort_reason = common_pb2.CANCEL_REASON_POSTPROC_TERMINATE
                try:
                    apply_post_processor_hook(hook, stream, streaming=stream._streaming)
                finally:
                    stream._default_abort_reason = prev_reason
                if getattr(stream, "_aborted", False) or (
                    getattr(stream, "_done", False) and not is_final
                ):
                    verdict.should_cancel = True
                    verdict.cancel_reason = common_pb2.CANCEL_REASON_POSTPROC_TERMINATE
                    verdict.force_done = True
            return verdict

        if not self._shards:
            return _run()
        shard = self._shards[int(stream.id) % len(self._shards)]
        return shard.submit(_run).result()

    def close(self) -> None:
        for shard in self._shards:
            shard.shutdown(wait=True)
        self._shards.clear()


__all__ = [
    "FrontendPostprocPool",
    "PostprocStepVerdict",
    "extract_final_metrics",
    "select_best_of_outputs",
]
