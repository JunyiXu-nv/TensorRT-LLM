# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for frontend-owned postproc pool, detok, H1 stop cancellation, hooks, and metrics."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from test_common.engine_client.mock_proto_engine import MockProtoEngine

from tensorrt_llm.contracts.serve_engine.v1 import common_pb2, output_pb2
from tensorrt_llm.engine_client import (
    FrontendPostprocPool,
    ProtoCompletionOutput,
    ProtoEngineFacade,
    ZmqEngineTransport,
    emit,
    suppress,
    terminate,
)
from tensorrt_llm.engine_client.postproc_pool import extract_final_metrics, select_best_of_outputs
from tensorrt_llm.sampling_params import SamplingParams


class _WordTokenizer:
    eos_token_id = 2
    vocab_size = 1000
    _WORDS = {10: "hello", 11: " STOP", 12: " trailing", 20: "world", 21: " secret"}

    def encode(self, text: str, **_: Any) -> list[int]:
        return [1, 10]

    def decode(self, ids: list[int], **_: Any) -> str:
        return "".join(self._WORDS.get(i, f"<{i}>") for i in ids)

    def decode_incrementally(
        self,
        token_ids_diff: list[int],
        *,
        prev_text: str,
        states: Any,
        flush: bool,
        stream_interval: int = 1,
        **_: Any,
    ) -> tuple[str, Any]:
        return prev_text + self.decode(token_ids_diff), states


@pytest.mark.cpu_only
def test_text_stop_string_truncates_and_sends_explicit_cancel_batch() -> None:
    async def _run() -> None:
        with MockProtoEngine(tokens_per_request=[10, 11, 12], step_delay_s=0.02) as engine:
            facade = ProtoEngineFacade(
                ZmqEngineTransport(
                    session_addr=engine.session_addr,
                    control_addr=engine.control_addr,
                    hmac_key=engine.hmac_key,
                ),
                tokenizer=_WordTokenizer(),
                num_postproc_workers=2,
            )
            await facade.start()
            try:
                sp = SamplingParams(max_tokens=3, stop=["STOP"], include_stop_str_in_output=False)
                stream = facade.generate_async([1, 2], sampling_params=sp, streaming=True)
                steps = [s.outputs[0].text async for s in stream]
                assert steps[-1] == "hello "
                assert stream.outputs[0].finish_reason == "stop"
                assert stream.outputs[0].stop_reason == "STOP"
                # Wait briefly for outbound CancelBatch to reach MockProtoEngine (fixes H1).
                for _ in range(30):
                    if engine.received_cancels:
                        break
                    await asyncio.sleep(0.02)
                assert any(
                    reason == common_pb2.CANCEL_REASON_STOP_STRING
                    for _, reason in engine.received_cancels
                ), f"Expected CANCEL_REASON_STOP_STRING, got {engine.received_cancels}"
            finally:
                await facade.aclose()

    asyncio.run(_run())


@pytest.mark.cpu_only
def test_post_processor_hook_runs_with_detokenize_false_and_cancels_on_terminate() -> None:
    seen_chunks: list[str] = []

    def _guard_hook(chunk: Any):
        seen_chunks.append(chunk.text_diff)
        if "secret" in chunk.text:
            return terminate(reason="blocked")
        if "world" in chunk.text_diff:
            return emit("[WORLD]")
        return suppress()

    async def _run() -> None:
        with MockProtoEngine(tokens_per_request=[20, 21, 12], step_delay_s=0.02) as engine:
            facade = ProtoEngineFacade(
                ZmqEngineTransport(
                    session_addr=engine.session_addr,
                    control_addr=engine.control_addr,
                    hmac_key=engine.hmac_key,
                ),
                tokenizer=_WordTokenizer(),
                post_processor_hook=_guard_hook,
            )
            await facade.start()
            try:
                sp = SamplingParams(max_tokens=3, detokenize=False)
                stream = facade.generate_async([1, 2], sampling_params=sp, streaming=True)
                steps = [s.outputs[0].text async for s in stream]
                assert seen_chunks, "Hook must run even when detokenize=False"
                assert any("[WORLD]" in t for t in steps)
                for _ in range(30):
                    if engine.received_cancels:
                        break
                    await asyncio.sleep(0.02)
                assert any(
                    reason == common_pb2.CANCEL_REASON_POSTPROC_TERMINATE
                    for _, reason in engine.received_cancels
                )
            finally:
                await facade.aclose()

    asyncio.run(_run())


@pytest.mark.cpu_only
def test_best_of_selection_and_final_metrics_extraction() -> None:
    outs = [
        ProtoCompletionOutput(index=0, text="low", cumulative_logprob=-5.0),
        ProtoCompletionOutput(index=1, text="high", cumulative_logprob=-0.5),
        ProtoCompletionOutput(index=2, text="mid", cumulative_logprob=-2.0),
    ]
    sp = SamplingParams(n=2, best_of=3, temperature=0.8)
    selected = select_best_of_outputs(outs, sp)
    assert [o.text for o in selected] == ["high", "mid"]
    assert [o.index for o in selected] == [0, 1]

    fi = output_pb2.FinalInfo(
        usage=output_pb2.Usage(prompt_tokens=12, completion_tokens=5, cached_tokens=4),
        decoding=output_pb2.DecodingStats(decoding_iter=3, avg_decoded_tokens_per_iter=1.67),
        spec=output_pb2.SpecDecodingStats(total_drafted=6, total_accepted=4),
    )
    metrics = extract_final_metrics(fi)
    assert metrics["usage"] == {"prompt_tokens": 12, "completion_tokens": 5, "cached_tokens": 4}
    assert metrics["decoding"]["decoding_iter"] == 3
    assert metrics["spec"]["total_accepted"] == 4

    pool = FrontendPostprocPool(num_workers=2)
    assert pool.num_workers == 2
    pool.close()
