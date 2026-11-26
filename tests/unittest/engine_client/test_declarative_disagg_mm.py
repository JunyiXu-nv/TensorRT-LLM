# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for declarative logit_bias/thinking_budget, Disagg SignedHandoff, and zero-copy MM."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from test_common.engine_client.mock_proto_engine import MockProtoEngine

from tensorrt_llm.contracts.serve_engine.v1 import common_pb2, control_pb2, disagg_pb2, request_pb2
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.engine_client import (
    HandoffReleaseTracker,
    ProtoEngineFacade,
    RequestMapper,
    ZmqEngineTransport,
    build_gen_disagg_params,
    decode_rendezvous_ticket_b64,
    encode_rendezvous_ticket_b64,
    extract_handoff_key,
)
from tensorrt_llm.executor.proto_bridge import proto_request_to_generation_request
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openai_server import OpenAIServer


class _Tok:
    eos_token_id = 2
    vocab_size = 1000
    name_or_path = "mock-model"

    def __init__(self) -> None:
        self.tokenizer = self

    def encode(self, text: str, **_: Any) -> list[int]:
        return [1, 10, 11]

    def decode(self, ids: list[int], **_: Any) -> str:
        return " ".join(f"t{i}" for i in ids)


@pytest.mark.cpu_only
def test_declarative_logit_bias_and_thinking_budget_bridge_materialization() -> None:
    mapper = RequestMapper(tokenizer=_Tok())
    # Sparse logit_bias via lightweight carrier (no dense vocab tensor allocated on Frontend)
    sp = SimpleNamespace(
        max_tokens=8,
        end_id=2,
        logit_bias={7: 1.5, 99: -2.0},
        thinking_token_budget=64,
    )
    req, aux = mapper.build_request(
        handle=11,
        request_id="r11",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
    )
    assert aux == []
    assert [(b.token_id, b.bias) for b in req.sampling.logit_bias] == [(7, 1.5), (99, -2.0)]
    assert req.thinking_budget.budget == 64

    # Rank-0 bridge materializes dense tensor & thinking_token_budget on the engine side
    gen_req = proto_request_to_generation_request(req, vocab_size=128)
    assert gen_req.sampling_params.thinking_token_budget == 64
    eb = gen_req.sampling_params.embedding_bias
    assert eb is not None
    assert eb.shape == (128,)
    assert float(eb[7]) == pytest.approx(1.5)
    assert float(eb[99]) == pytest.approx(-2.0)
    assert float(eb[0]) == pytest.approx(0.0)


@pytest.mark.cpu_only
def test_disagg_dg04_handoff_two_leg_roundtrip_and_http_release() -> None:
    ticket = disagg_pb2.RendezvousTicket(
        ctx_instance_id="mock-ctx-0",
        transfer_backend="nixl",
        endpoint="tcp://127.0.0.1:9100",
        ttl_ms=30000,
    )
    ticket_b64 = encode_rendezvous_ticket_b64(ticket)
    assert decode_rendezvous_ticket_b64(ticket_b64).endpoint == "tcp://127.0.0.1:9100"

    async def _run() -> None:
        args_ns = SimpleNamespace(
            model="mock-model",
            backend="pytorch",
            trust_remote_code=False,
            checkpoint_format=None,
            reasoning_parser=None,
            guided_decoding_backend=None,
            post_processor_hook=None,
            return_perf_metrics=False,
            enable_iter_perf_stats=False,
            num_postprocess_workers=0,
            max_beam_width=1,
            gather_generation_logits=False,
            encode_only=False,
        )
        with MockProtoEngine(tokens_per_request=[30, 31]) as engine:
            facade = ProtoEngineFacade(
                ZmqEngineTransport(
                    session_addr=engine.session_addr,
                    control_addr=engine.control_addr,
                    hmac_key=engine.hmac_key,
                ),
                tokenizer=_Tok(),
                args=args_ns,
            )
            server = OpenAIServer(
                generator=facade,
                model="mock-model",
                tool_parser=None,
                server_role=None,
                metadata_server_cfg=None,
                input_processor_workers=1,
            )
            await facade.start()
            try:
                tracker = HandoffReleaseTracker(facade)
                # Leg 1: CONTEXT_ONLY -> emits opaque SignedHandoff
                ctx_out = await facade.generate_async(
                    [1, 2, 3],
                    sampling_params=SamplingParams(max_tokens=1),
                    disaggregated_params=DisaggregatedParams(
                        request_type="context_only",
                        disagg_request_id=9001,
                        schedule_style="context_first",
                    ),
                ).aresult()
                signed_b64 = ctx_out.outputs[0].signed_handoff_b64
                assert signed_b64 is not None
                inst_id, disagg_id = tracker.register(signed_b64)
                assert (inst_id, disagg_id) == ("mock-ctx-0", 9001)
                assert extract_handoff_key(signed_b64) == ("mock-ctx-0", 9001)

                gen_pb = build_gen_disagg_params(signed_b64)
                assert gen_pb.request_id == 9001
                assert gen_pb.handoff.mac

                # Leg 2: GENERATION_ONLY carrying signed_handoff_b64 untouched
                gen_dp = SimpleNamespace(
                    request_type="generation_only",
                    disagg_request_id=9001,
                    schedule_style="context_first",
                    signed_handoff_b64=signed_b64,
                )
                gen_out = await facade.generate_async(
                    [1, 2, 3],
                    sampling_params=SamplingParams(max_tokens=2),
                    disaggregated_params=gen_dp,
                ).aresult()
                assert gen_out.outputs[0].text == "t30 t31"
                assert engine.received_requests[-1].phase == common_pb2.PHASE_GENERATION_ONLY
                assert engine.received_requests[-1].disagg.handoff.mac == gen_pb.handoff.mac

                # HTTP internal release endpoint calls ControlService.ReleaseHandoffs
                transport = httpx.ASGITransport(app=server.app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    resp = await client.post(
                        "/_internal/release_handoffs",
                        json={
                            "items": [{"ctx_instance_id": "mock-ctx-0", "disagg_request_id": 9001}]
                        },
                    )
                    assert resp.status_code == 200, resp.text
                    data = resp.json()
                    assert data["results"][0]["status"] == int(
                        control_pb2.RELEASE_HANDOFF_STATUS_RELEASED
                    )
                assert ("mock-ctx-0", 9001) in engine.released_handoffs
                tracker.mark_consumed(9001)
            finally:
                await facade.aclose()
                for attr in (
                    "_input_proc_executor",
                    "_media_load_executor",
                    "_metrics_stats_executor",
                ):
                    pool = getattr(server, attr, None)
                    if pool is not None:
                        pool.shutdown(wait=True)

    asyncio.run(_run())


@pytest.mark.cpu_only
def test_multimodal_media_and_preprocessed_zero_copy_aux_frames() -> None:
    mapper = RequestMapper(tokenizer=_Tok(), max_inline_tensor_bytes=64)

    # MEDIA mode with v1.1 P-3 image uuid
    req_media, aux_media = mapper.build_request(
        handle=20,
        request_id="mm-media",
        prompt_token_ids=[1, 2],
        multimodal_params={
            "media": [
                {
                    "modality": "image",
                    "url": "https://example.com/img.png",
                    "mime_type": "image/png",
                    "uuid": "img-uuid-123",
                }
            ]
        },
    )
    assert aux_media == []
    assert req_media.mm.mode == request_pb2.MULTIMODAL_MODE_MEDIA
    assert req_media.mm.media[0].uuid == "img-uuid-123"

    # PREPROCESSED mode over live ZMQ carries large pixel_values tensor in aux frame (zero pickle)
    async def _run() -> None:
        with MockProtoEngine(tokens_per_request=[40]) as engine:
            facade = ProtoEngineFacade(
                ZmqEngineTransport(
                    session_addr=engine.session_addr,
                    control_addr=engine.control_addr,
                    hmac_key=engine.hmac_key,
                ),
                tokenizer=_Tok(),
            )
            await facade.start()
            try:
                large_pixels = np.ones((64, 64, 3), dtype=np.float32)  # 48 KB > inline
                # Set low inline threshold via mapper to force aux frame transport
                facade._mapper._max_inline_tensor_bytes = 64
                out = await facade.generate_async(
                    {
                        "prompt_token_ids": [1, 2, 3],
                        "multi_modal_data": {"pixel_values": large_pixels},
                    },
                    sampling_params=SamplingParams(max_tokens=1),
                ).aresult()
                assert out.outputs[0].text == "t40"
                assert engine.received_aux_counts[-1] == 1
                assert (
                    engine.received_requests[-1].mm.mode == request_pb2.MULTIMODAL_MODE_PREPROCESSED
                )
                assert engine.received_requests[-1].mm.model_inputs[0].path == "pixel_values"
            finally:
                await facade.aclose()

    asyncio.run(_run())
