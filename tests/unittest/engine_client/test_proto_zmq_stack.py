# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit, property, live ZMQ, and HTTP E2E tests for ``trtllm.serve_engine.v1`` over ZMQ."""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_common.engine_client.mock_proto_engine import MockProtoEngine, default_mock_capabilities

from tensorrt_llm.contracts.serve_engine.v1 import (
    common_pb2,
    control_pb2,
    output_pb2,
    request_pb2,
    session_pb2,
)
from tensorrt_llm.contracts.serve_engine.v1.zmq_wire import (
    WireFramingError,
    pack_session_frames,
    unpack_session_frames,
)
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.engine_client import (
    CapabilityGate,
    DecodedTokenLogprob,
    EngineCapabilityError,
    EngineTransportError,
    FrameBuilder,
    ProtoEngineFacade,
    RequestMapper,
    ZmqEngineTransport,
    decode_frame,
    decode_tensor_ref,
    encode_tensor_ref,
)
from tensorrt_llm.executor.proto_bridge import Rank0ProtoBridge
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams
from tensorrt_llm.serve.openai_server import OpenAIServer


class _SimpleTokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0
    vocab_size = 1000
    name_or_path = "mock-model"

    def __init__(self) -> None:
        self.tokenizer = self

    def encode(self, text: str, **_: Any) -> list[int]:
        return [1] + [(ord(c) % 900) + 10 for c in text]

    def decode(self, token_ids: list[int], **_: Any) -> str:
        return " ".join(f"tok{t}" for t in token_ids)

    def apply_chat_template(self, conversation: list[dict[str, Any]], **_: Any) -> str:
        return "\n".join(str(m.get("content", "")) for m in conversation)


# ---------------------------------------------------------------------------
# 1. Columnar Frame round-trip + Hypothesis property tests
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@settings(max_examples=40)
@given(
    rows_spec=st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=10_000),
            st.integers(min_value=0, max_value=4),
            st.lists(st.integers(min_value=0, max_value=50_000), min_size=0, max_size=8),
            st.booleans(),
        ),
        min_size=0,
        max_size=12,
    )
)
def test_columnar_frame_roundtrip_property(
    rows_spec: list[tuple[int, int, list[int], bool]],
) -> None:
    fb = FrameBuilder(frame_seq=7, iter_first=10, iter_last=12, emit_time_ns=999)
    for handle, seq, toks, with_lp in rows_spec:
        lps = (
            [DecodedTokenLogprob(sampled=-0.25, rank=1, topk=[(t, -0.25)]) for t in toks]
            if with_lp and toks
            else None
        )
        fb.add_row(
            handle=handle,
            seq=seq,
            tokens=toks,
            finish=common_pb2.FINISH_REASON_NOT_FINISHED,
            row_flags=common_pb2.ROW_FLAG_FIRST_ROW,
            logprobs=lps,
        )
    frame = fb.build()
    decoded = decode_frame(frame)
    assert decoded.frame_seq == 7
    assert len(decoded.rows) == len(rows_spec)
    for got, (handle, seq, toks, with_lp) in zip(decoded.rows, rows_spec):
        assert got.handle == handle
        assert got.seq == seq
        assert got.tokens == toks
        if with_lp and toks:
            assert got.logprobs is not None
            assert len(got.logprobs) == len(toks)
            assert math.isclose(got.logprobs[0].sampled or 0.0, -0.25, abs_tol=1e-5)


@pytest.mark.cpu_only
def test_frame_decoder_rejects_corrupted_column_length() -> None:
    frame = output_pb2.Frame(
        frame_seq=1,
        n_rows=2,
        handle=b"\x01\x00\x00\x00",  # truncated: 4 bytes instead of 16
        token_offsets=np.asarray([0, 0, 0], dtype="<u4").tobytes(),
        finish=bytes([0, 0]),
        row_flags=np.asarray([0, 0], dtype="<u2").tobytes(),
    )
    with pytest.raises(ValueError, match="byte length"):
        decode_frame(frame)


@pytest.mark.cpu_only
def test_tensor_ref_inline_and_zmq_aux_roundtrip() -> None:
    aux: list[bytes] = []
    small = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    large = np.arange(128, dtype=np.float32).reshape(8, 16)
    ref_small = encode_tensor_ref(small, aux_buffers=aux, max_inline_bytes=64)
    ref_large = encode_tensor_ref(large, aux_buffers=aux, max_inline_bytes=64)
    assert ref_small.WhichOneof("location") == "inline_bytes"
    assert ref_large.WhichOneof("location") == "blob"
    assert len(aux) == 1
    np.testing.assert_array_equal(decode_tensor_ref(ref_small, aux), small)
    np.testing.assert_array_equal(decode_tensor_ref(ref_large, aux), large)


@pytest.mark.cpu_only
def test_zmq_wire_hmac_detects_tampering() -> None:
    key = b"k" * 32
    frames = pack_session_frames(key, b"proto-payload", [b"aux0"])
    body, aux = unpack_session_frames(key, frames)
    assert body == b"proto-payload"
    assert aux == [b"aux0"]
    tampered = [frames[0], frames[1], b"proto-payloaX", frames[3]]
    with pytest.raises(WireFramingError, match="HMAC"):
        unpack_session_frames(key, tampered)


# ---------------------------------------------------------------------------
# 2. CapabilityGate & RequestMapper tests
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_capability_gate_enforces_domains_and_disagg_rules() -> None:
    gate = CapabilityGate(default_mock_capabilities())
    mapper = RequestMapper(tokenizer=_SimpleTokenizer())

    req, _ = mapper.build_request(
        handle=1,
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(temperature=3.5),
    )
    with pytest.raises(EngineCapabilityError, match="temperature"):
        gate.validate_request(req)

    # CONTEXT_ONLY without schedule_style must be rejected synchronously
    ctx_req, _ = mapper.build_request(
        handle=2,
        request_id="r2",
        prompt_token_ids=[1, 2, 3],
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
    )
    with pytest.raises(EngineCapabilityError, match="schedule_style"):
        gate.validate_request(ctx_req)

    # Unregistered logits processor name must be rejected
    bad_proc = request_pb2.Request(handle=3, prompt_tokens=[1, 2], logits_processors=["unknown"])
    with pytest.raises(EngineCapabilityError, match="unknown"):
        gate.validate_request(bad_proc)


@pytest.mark.cpu_only
def test_request_mapper_declarative_fields_and_callable_rejection() -> None:
    mapper = RequestMapper(tokenizer=_SimpleTokenizer())
    import torch

    eb = torch.zeros(100, dtype=torch.float32)
    eb[15] = 2.5
    eb[42] = -1.0
    sp = SamplingParams(
        max_tokens=16,
        temperature=0.7,
        guided_decoding=GuidedDecodingParams(json='{"type":"object"}'),
        thinking_token_budget=128,
        logprobs=3,
        embedding_bias=eb,
    )
    req, _ = mapper.build_request(
        handle=5,
        request_id="r5",
        prompt_token_ids=[1, 2, 3],
        sampling_params=sp,
        lora_request=LoRARequest("adapter-a", 7, "/tmp/lora"),
        streaming=True,
    )
    assert [(b.token_id, b.bias) for b in req.sampling.logit_bias] == [(15, 2.5), (42, -1.0)]
    assert req.thinking_budget.budget == 128
    assert req.guided.json_schema == '{"type":"object"}'
    assert req.lora.adapter_id == 7

    sp_bad = SamplingParams(logits_processor=lambda *a, **kw: None)
    with pytest.raises(EngineCapabilityError, match="callables"):
        mapper.build_request(
            handle=6, request_id="r6", prompt_token_ids=[1, 2], sampling_params=sp_bad
        )


# ---------------------------------------------------------------------------
# 3. Live ZMQ end-to-end tests against MockProtoEngine
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_proto_facade_live_zmq_streaming_control_disagg_and_faults() -> None:
    async def _run() -> None:
        with MockProtoEngine(tokens_per_request=[10, 11, 12]) as engine:
            transport = ZmqEngineTransport(
                session_addr=engine.session_addr,
                control_addr=engine.control_addr,
                hmac_key=engine.hmac_key,
                frontend_id=0,
            )
            facade = ProtoEngineFacade(transport, tokenizer=_SimpleTokenizer())
            accepted = await facade.start()
            assert accepted.session_id == 1
            assert facade.check_health()

            # Control RPCs including v1.1 sampled_at_unix_nanos
            info = await facade.get_info()
            assert info.server.host_build == "mock-proto-engine"
            load = await facade.get_load()
            assert load.sampled_at_unix_nanos > 0
            cap = await facade.get_kv_capacity()
            assert cap.max_blocks == 128
            assert await facade.aget_data_transceiver_state() == b"mock-transceiver"
            await facade.lora_load(9, "lora-9", "/tmp/l9")
            loras = await facade.lora_list()
            assert [a.adapter_id for a in loras.adapters] == [9]
            await facade.collective_rpc("sleep", args=(["kv_cache"],))
            assert "sleep" in engine.admin_commands

            # Streaming + logprobs + generation_logits over ZMQ aux zero-copy frame
            sp = SamplingParams(max_tokens=3, logprobs=1, return_generation_logits=True)
            stream = facade.generate_async([1, 2, 3], sampling_params=sp, streaming=True)
            steps = [s.outputs[0].text async for s in stream]
            assert steps[-1] == "tok10 tok11 tok12"
            assert stream.outputs[0].generation_logits is not None
            assert stream.outputs[0].generation_logits.shape == (4, 4)
            assert len(stream.outputs[0].logprobs) == 3

            # Disagg CONTEXT_ONLY -> SignedHandoff -> ReleaseHandoffs
            ctx_stream = facade.generate_async(
                [1, 2, 3],
                sampling_params=SamplingParams(max_tokens=1),
                disaggregated_params=DisaggregatedParams(
                    request_type="context_only",
                    disagg_request_id=777,
                    schedule_style="context_first",
                ),
                streaming=False,
            )
            ctx_out = await ctx_stream.aresult()
            dp = ctx_out.outputs[0].disaggregated_params
            assert dp is not None
            assert ctx_out.outputs[0].signed_handoff_b64
            rel = await facade.release_handoffs([("mock-ctx-0", 777)])
            assert rel.results[0].status == control_pb2.RELEASE_HANDOFF_STATUS_RELEASED

            # ENGINE_DEAD notice fails new / inflight cleanly
            engine.inject_notice(
                session_pb2.SessionNotice(
                    kind=session_pb2.SESSION_NOTICE_KIND_ENGINE_DEAD,
                    message="simulated crash",
                )
            )
            for _ in range(30):
                if not facade.check_health():
                    break
                await asyncio.sleep(0.02)
            assert not facade.check_health()
            with pytest.raises(EngineTransportError):
                facade.generate_async([1, 2, 3])
            await facade.aclose()

    asyncio.run(_run())


@pytest.mark.cpu_only
def test_multi_frontend_identity_routing_and_rank0_proto_bridge() -> None:
    async def _run() -> None:
        class _FakeRsp:
            def __init__(self, cid: int, toks: list[int], final: bool) -> None:
                self.client_id = cid
                self.result = SimpleNamespace(
                    output_token_ids=[toks],
                    is_final=final,
                    sequence_index=0,
                    context_phase_params=None,
                )

            def has_error(self) -> bool:
                return False

        class _FakeWorker:
            def __init__(self) -> None:
                self.llm_args = SimpleNamespace(
                    model="fake",
                    max_seq_len=512,
                    max_batch_size=8,
                    max_num_tokens=1024,
                    reasoning_parser="",
                    max_beam_width=1,
                )
                self._q: list[Any] = []

            def _enqueue_request(self, req: Any) -> None:
                self._q.append(_FakeRsp(req.id, [req.id + 100], True))

            def await_responses(self, timeout: float = 0.0) -> list[Any]:
                out = self._q[:]
                self._q.clear()
                return out

        bridge = Rank0ProtoBridge(_FakeWorker())
        bridge.start()
        try:
            f0 = ProtoEngineFacade(
                ZmqEngineTransport(
                    session_addr=bridge.session_addr,
                    control_addr=bridge.control_addr,
                    hmac_key=bridge.hmac_key,
                    frontend_id=0,
                ),
                tokenizer=_SimpleTokenizer(),
            )
            f1 = ProtoEngineFacade(
                ZmqEngineTransport(
                    session_addr=bridge.session_addr,
                    control_addr=bridge.control_addr,
                    hmac_key=bridge.hmac_key,
                    frontend_id=1,
                ),
                tokenizer=_SimpleTokenizer(),
            )
            await f0.start()
            await f1.start()
            r0, r1 = await asyncio.gather(
                f0.generate_async([1, 2], sampling_params=SamplingParams(max_tokens=2)).aresult(),
                f1.generate_async([3, 4], sampling_params=SamplingParams(max_tokens=2)).aresult(),
            )
            assert r0.outputs[0].token_ids == [r0.id + 100]
            assert r1.outputs[0].token_ids == [r1.id + 100]
            await f0.aclose()
            await f1.aclose()
        finally:
            bridge.stop()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. End-to-end HTTP ASGI test: OpenAIServer -> ProtoEngineFacade -> MockProtoEngine
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_openai_server_http_endpoints_over_zmq_proto() -> None:
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
        with MockProtoEngine(tokens_per_request=[20, 21]) as engine:
            facade = ProtoEngineFacade(
                ZmqEngineTransport(
                    session_addr=engine.session_addr,
                    control_addr=engine.control_addr,
                    hmac_key=engine.hmac_key,
                ),
                tokenizer=_SimpleTokenizer(),
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
                transport = httpx.ASGITransport(app=server.app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                ) as client:
                    r_health = await client.get("/health")
                    assert r_health.status_code == 200

                    r_comp = await client.post(
                        "/v1/completions",
                        json={
                            "model": "mock-model",
                            "prompt": "hello",
                            "max_tokens": 2,
                            "stream": False,
                        },
                    )
                    assert r_comp.status_code == 200, r_comp.text
                    body = r_comp.json()
                    assert body["choices"][0]["text"] == "tok20 tok21"

                    r_stream = await client.post(
                        "/v1/completions",
                        json={
                            "model": "mock-model",
                            "prompt": "hello",
                            "max_tokens": 2,
                            "stream": True,
                        },
                    )
                    assert r_stream.status_code == 200, r_stream.text
                    assert "data: [DONE]" in r_stream.text

                    # Out-of-domain temperature triggers EngineCapabilityError -> HTTP 400
                    r_bad = await client.post(
                        "/v1/completions",
                        json={
                            "model": "mock-model",
                            "prompt": "hello",
                            "temperature": 9.5,
                            "max_tokens": 2,
                        },
                    )
                    assert r_bad.status_code == 400
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
