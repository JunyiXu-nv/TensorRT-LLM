# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .capabilities import CapabilityGate
from .executor import CppExecutorError, GenerationExecutor
from .facade import ProtoCompletionOutput, ProtoEngineFacade, ProtoGenerationStream
from .frame_decoder import (
    DecodedFrame,
    DecodedRow,
    DecodedTokenLogprob,
    FrameBuilder,
    decode_frame,
    decode_tensor_ref,
    encode_tensor_ref,
)
from .handoff import (
    HandoffReleaseTracker,
    build_gen_disagg_params,
    decode_rendezvous_ticket_b64,
    decode_signed_handoff_b64,
    encode_rendezvous_ticket_b64,
    encode_signed_handoff_b64,
    extract_handoff_key,
)
from .interface import (
    EngineCapabilityError,
    EngineClientInterface,
    EngineProtocolError,
    EngineTransportError,
)
from .postproc_pool import (
    FrontendPostprocPool,
    PostprocStepVerdict,
    extract_final_metrics,
    select_best_of_outputs,
)
from .postproc_worker import PostprocWorker
from .postprocessor_hook import (
    PostProcessorHook,
    PostProcessorHookChunk,
    PostProcessorHookVerdict,
    emit,
    suppress,
    terminate,
)
from .prompt_tail import PromptTailPolicy
from .proxy import GenerationExecutorFrontendProxy, GenerationExecutorProxy
from .request_mapper import RequestMapper
from .result import (
    CompletionOutput,
    DetokenizedGenerationResultBase,
    GenerationResult,
    GenerationResultBase,
    IterationResult,
)
from .transport import ControlMethod, EngineTransport
from .transport_zmq import ZmqEngineTransport

__all__ = [
    "CapabilityGate",
    "CompletionOutput",
    "ControlMethod",
    "CppExecutorError",
    "DecodedFrame",
    "DecodedRow",
    "DecodedTokenLogprob",
    "DetokenizedGenerationResultBase",
    "EngineCapabilityError",
    "EngineClientInterface",
    "EngineProtocolError",
    "EngineTransport",
    "EngineTransportError",
    "FrameBuilder",
    "FrontendPostprocPool",
    "GenerationExecutor",
    "GenerationExecutorFrontendProxy",
    "GenerationExecutorProxy",
    "GenerationResult",
    "GenerationResultBase",
    "HandoffReleaseTracker",
    "IterationResult",
    "PostProcessorHook",
    "PostProcessorHookChunk",
    "PostProcessorHookVerdict",
    "PostprocWorker",
    "PromptTailPolicy",
    "ProtoCompletionOutput",
    "ProtoEngineFacade",
    "ProtoGenerationStream",
    "RequestMapper",
    "ZmqEngineTransport",
    "decode_frame",
    "decode_tensor_ref",
    "emit",
    "encode_tensor_ref",
    "suppress",
    "terminate",
]
