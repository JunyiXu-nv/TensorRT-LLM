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
"""Pydantic models for the Anthropic Messages API (``POST /v1/messages``).

Wire-format reference: https://platform.claude.com/docs/en/api/messages

These models cover the subset of the protocol required to serve Anthropic
SDK clients and Claude Code. Request models are permissive (``extra="allow"``)
because Anthropic clients attach evolving auxiliary fields (``metadata``,
``betas``, ``output_config``, ...) that must not fail validation; response
models emit only the fields this server populates.
"""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AnthropicBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class AnthropicTextBlock(AnthropicBaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicImageSource(AnthropicBaseModel):
    type: Literal["base64", "url"]
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None


class AnthropicImageBlock(AnthropicBaseModel):
    type: Literal["image"] = "image"
    source: AnthropicImageSource


class AnthropicToolUseBlock(AnthropicBaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid.uuid4().hex}")
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultBlock(AnthropicBaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[Union[str, List["AnthropicContentBlock"]]] = None
    is_error: Optional[bool] = None


class AnthropicThinkingBlock(AnthropicBaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = None


class AnthropicRedactedThinkingBlock(AnthropicBaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: Optional[str] = None


AnthropicContentBlock = Union[
    AnthropicTextBlock,
    AnthropicImageBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
    AnthropicThinkingBlock,
    AnthropicRedactedThinkingBlock,
]

AnthropicToolResultBlock.model_rebuild()


class AnthropicMessage(AnthropicBaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[AnthropicContentBlock]]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Anthropic-provided tools use a versioned ``type``. Server tools are executed
# by Anthropic's API, while schema client tools are executed by the caller.
SERVER_TOOL_TYPE_PREFIXES = (
    "web_search",
    "web_fetch",
    "code_execution",
    "tool_search_tool_",
    "advisor_",
    "mcp_toolset",
)
SCHEMA_CLIENT_TOOL_TYPE_PREFIXES = (
    "bash_",
    "text_editor_",
    "computer_",
    "memory_",
)


class AnthropicTool(AnthropicBaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    def is_server_tool(self) -> bool:
        if self.type is None or self.type == "custom":
            return False
        return any(self.type.startswith(prefix) for prefix in SERVER_TOOL_TYPE_PREFIXES)

    def is_schema_client_tool(self) -> bool:
        if self.type is None or self.type == "custom":
            return False
        return any(
            self.type.startswith(prefix) for prefix in SCHEMA_CLIENT_TOOL_TYPE_PREFIXES
        )


class AnthropicToolChoice(AnthropicBaseModel):
    type: Literal["auto", "any", "tool", "none"]
    name: Optional[str] = None
    disable_parallel_tool_use: Optional[bool] = None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class AnthropicMessagesRequest(AnthropicBaseModel):
    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    system: Optional[Union[str, List[AnthropicTextBlock]]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None
    # Extended-thinking control; consumed on a best-effort basis.
    thinking: Optional[Dict[str, Any]] = None
    # Claude Code attaches output_config (effort, format) and betas.
    output_config: Optional[Dict[str, Any]] = None
    betas: Optional[List[str]] = None


class AnthropicCountTokensRequest(AnthropicBaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[AnthropicTextBlock]]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    thinking: Optional[Dict[str, Any]] = None
    output_config: Optional[Dict[str, Any]] = None
    betas: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal"]


class AnthropicUsage(AnthropicBaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicMessagesResponse(AnthropicBaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: List[AnthropicContentBlock] = Field(default_factory=list)
    stop_reason: Optional[AnthropicStopReason] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


class AnthropicCountTokensResponse(AnthropicBaseModel):
    input_tokens: int


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

AnthropicErrorType = Literal[
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
    "rate_limit_error",
    "api_error",
    "overloaded_error",
]


class AnthropicError(AnthropicBaseModel):
    type: AnthropicErrorType = "api_error"
    message: str


class AnthropicErrorResponse(AnthropicBaseModel):
    type: Literal["error"] = "error"
    error: AnthropicError
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Streaming events (SSE)
#
# Wire framing is ``event: <type>\ndata: <json>\n\n``; the ``type`` field
# inside the payload must match the event name.
# ---------------------------------------------------------------------------


class AnthropicMessageStartEvent(AnthropicBaseModel):
    type: Literal["message_start"] = "message_start"
    message: AnthropicMessagesResponse


class AnthropicContentBlockStartEvent(AnthropicBaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: AnthropicContentBlock


class AnthropicTextDelta(AnthropicBaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class AnthropicInputJsonDelta(AnthropicBaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class AnthropicThinkingDelta(AnthropicBaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class AnthropicSignatureDelta(AnthropicBaseModel):
    type: Literal["signature_delta"] = "signature_delta"
    signature: str


AnthropicContentDelta = Union[
    AnthropicTextDelta, AnthropicInputJsonDelta, AnthropicThinkingDelta, AnthropicSignatureDelta
]


class AnthropicContentBlockDeltaEvent(AnthropicBaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: AnthropicContentDelta


class AnthropicContentBlockStopEvent(AnthropicBaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class AnthropicMessageDelta(AnthropicBaseModel):
    stop_reason: Optional[AnthropicStopReason] = None
    stop_sequence: Optional[str] = None


class AnthropicMessageDeltaEvent(AnthropicBaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: AnthropicMessageDelta
    usage: Optional[AnthropicUsage] = None


class AnthropicMessageStopEvent(AnthropicBaseModel):
    type: Literal["message_stop"] = "message_stop"


class AnthropicPingEvent(AnthropicBaseModel):
    type: Literal["ping"] = "ping"


class AnthropicErrorEvent(AnthropicBaseModel):
    type: Literal["error"] = "error"
    error: AnthropicError


def anthropic_sse(event: AnthropicBaseModel) -> str:
    """Serialize an event model into one Anthropic SSE frame."""
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"


def current_timestamp() -> int:
    return int(time.time())
