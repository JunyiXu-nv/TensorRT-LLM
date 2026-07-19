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
"""Adapter between the Anthropic Messages API and the OpenAI chat pipeline.

Request direction: :class:`AnthropicMessagesRequest` is translated into a
:class:`ChatCompletionRequest` so the existing ``openai_chat`` path (chat
template, tool parser, reasoning parser, post-processing) is reused verbatim.
Response direction: the resulting :class:`ChatCompletionResponse` is
translated back into an :class:`AnthropicMessagesResponse`.

The adapter is a pure protocol layer: it never touches the tokenizer,
chat templates, or the engine.
"""

import json
import traceback
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi.responses import JSONResponse

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.anthropic_protocol import (
    AnthropicContentBlockDeltaEvent,
    AnthropicContentBlockStartEvent,
    AnthropicContentBlockStopEvent,
    AnthropicError,
    AnthropicErrorEvent,
    AnthropicErrorResponse,
    AnthropicErrorType,
    AnthropicInputJsonDelta,
    AnthropicMessageDelta,
    AnthropicMessageDeltaEvent,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicMessageStartEvent,
    AnthropicMessageStopEvent,
    AnthropicStopReason,
    AnthropicTextBlock,
    AnthropicTextDelta,
    AnthropicThinkingBlock,
    AnthropicThinkingDelta,
    AnthropicToolUseBlock,
    AnthropicUsage,
    anthropic_sse,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionNamedFunction,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
    FunctionDefinition,
    UsageInfo,
)

# OpenAI finish_reason -> Anthropic stop_reason. A string-valued OpenAI
# stop_reason identifies the matched stop sequence and is handled separately.
STOP_REASON_MAP: Dict[str, AnthropicStopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


class AnthropicRequestError(ValueError):
    """Invalid Anthropic request; maps to a 400 with an Anthropic envelope."""


class AnthropicResponseError(ValueError):
    """Invalid upstream model response; maps to an Anthropic API error."""


def anthropic_error_response(
    message: str, error_type: AnthropicErrorType = "api_error", status_code: int = 500
) -> JSONResponse:
    envelope = AnthropicErrorResponse(error=AnthropicError(type=error_type, message=message))
    return JSONResponse(content=envelope.model_dump(exclude_none=True), status_code=status_code)


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> ChatCompletionRequest
# ---------------------------------------------------------------------------


def _system_text_parts(system: Optional[Union[str, List[Any]]]) -> List[str]:
    if system is None:
        return []
    if isinstance(system, str):
        return [system] if system else []
    parts = []
    for block in system:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return parts


def _image_part(block: Any) -> Optional[Dict[str, Any]]:
    source = block.source
    if source.type == "url" and source.url:
        return {"type": "image_url", "image_url": {"url": source.url}}
    if source.type == "base64" and source.data:
        media_type = source.media_type or "image/png"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{source.data}"},
        }
    logger.warning("Anthropic image block with empty source dropped")
    return None


def _tool_result_text(content: Any) -> str:
    """Flatten a tool_result content payload into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(block.text)
        else:
            raise AnthropicRequestError(
                f"Content block type {block_type!r} inside tool_result is not supported"
            )
    return "\n".join(parts)


def _convert_messages(request: AnthropicMessagesRequest) -> List[Dict[str, Any]]:
    system_parts = _system_text_parts(request.system)
    converted: List[Dict[str, Any]] = []

    for message in request.messages:
        if message.role == "system":
            # Anthropic clients put the system prompt in the top-level field;
            # inline system messages are merged into it because most chat
            # templates reject mid-conversation system turns.
            if isinstance(message.content, str):
                system_parts.append(message.content)
            else:
                system_parts.extend(
                    block.text
                    for block in message.content
                    if getattr(block, "type", None) == "text"
                )
            continue

        if isinstance(message.content, str):
            converted.append({"role": message.role, "content": message.content})
            continue

        # Content parts accumulated for the current role; flushed before any
        # role:"tool" message so ordering user(pre) -> tool -> user(post) is
        # preserved.
        parts: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        reasoning_parts: List[str] = []

        def flush_parts():
            if parts:
                converted_message: Dict[str, Any] = {
                    "role": message.role,
                    "content": list(parts),
                }
                if message.role == "assistant" and reasoning_parts:
                    converted_message["reasoning"] = "".join(reasoning_parts)
                    reasoning_parts.clear()
                converted.append(converted_message)
                parts.clear()

        for block in message.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append({"type": "text", "text": block.text})
            elif block_type == "image":
                image_part = _image_part(block)
                if image_part is not None:
                    parts.append(image_part)
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )
            elif block_type == "tool_result":
                flush_parts()
                tool_result_text = _tool_result_text(block.content)
                if block.is_error:
                    tool_result_text = f"Tool execution failed: {tool_result_text}"
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": tool_result_text,
                    }
                )
            elif block_type == "thinking":
                if message.role != "assistant":
                    raise AnthropicRequestError(
                        "thinking content blocks are only valid in assistant messages"
                    )
                reasoning_parts.append(block.thinking)
            elif block_type == "redacted_thinking":
                raise AnthropicRequestError(
                    "redacted_thinking history is not supported by this server"
                )
            else:
                logger.warning("Unsupported Anthropic content block %r skipped", block_type)

        if message.role == "assistant" and tool_calls:
            text_content = "".join(p["text"] for p in parts if p.get("type") == "text")
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": text_content or None,
                "tool_calls": tool_calls,
            }
            if reasoning_parts:
                assistant_message["reasoning"] = "".join(reasoning_parts)
                reasoning_parts.clear()
            converted.append(assistant_message)
            parts.clear()
        else:
            flush_parts()
            if message.role == "assistant" and reasoning_parts:
                converted.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "".join(reasoning_parts),
                    }
                )

    if system_parts:
        converted.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return converted


def _convert_tools(request: AnthropicMessagesRequest) -> Optional[List[ChatCompletionToolsParam]]:
    if not request.tools:
        return None
    if request.tool_choice is not None and request.tool_choice.type == "none":
        return None
    tools = []
    for tool in request.tools:
        if tool.is_server_tool():
            raise AnthropicRequestError(
                f"Anthropic server tool {tool.name!r} (type={tool.type!r}) "
                "is not supported by this server"
            )
        if tool.input_schema is None:
            if tool.is_schema_client_tool():
                raise AnthropicRequestError(
                    f"Anthropic schema client tool {tool.name!r} "
                    f"(type={tool.type!r}) is recognized, but its built-in "
                    "input schema is not implemented by this server"
                )
            raise AnthropicRequestError(
                f"Client tool {tool.name!r} requires input_schema"
            )
        tools.append(
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.input_schema,
                    strict=tool.strict,
                )
            )
        )
    return tools or None


def _convert_tool_choice(
    request: AnthropicMessagesRequest,
    tools: Optional[List[ChatCompletionToolsParam]],
) -> Optional[Union[str, ChatCompletionNamedToolChoiceParam]]:
    choice = request.tool_choice
    if choice is None:
        # check_tool_choice validator defaults to "auto" when tools are set.
        return "auto" if tools else None
    if choice.disable_parallel_tool_use and choice.type != "none":
        raise AnthropicRequestError(
            "tool_choice.disable_parallel_tool_use=true is not supported"
        )
    if choice.type == "none":
        return "none"
    if tools is None:
        raise AnthropicRequestError(
            f"tool_choice type {choice.type!r} requires at least one "
            "client-executable tool; all provided tools were server tools "
            "or the tools list was empty"
        )
    if choice.type == "auto":
        return "auto"
    if choice.type == "any":
        raise AnthropicRequestError(
            "Anthropic tool_choice type 'any' is not supported because the "
            "chat pipeline cannot require an arbitrary tool call"
        )
    if choice.type == "tool":
        if not choice.name:
            raise AnthropicRequestError("tool_choice type 'tool' requires a 'name'")
        tool_names = {t.function.name for t in tools}
        if choice.name not in tool_names:
            raise AnthropicRequestError(f"tool_choice names unknown tool {choice.name!r}")
        return ChatCompletionNamedToolChoiceParam(
            function=ChatCompletionNamedFunction(name=choice.name)
        )
    raise AnthropicRequestError(f"Unsupported tool_choice {choice.type!r}")


def convert_anthropic_request(request: AnthropicMessagesRequest) -> ChatCompletionRequest:
    """Translate an Anthropic Messages request into a chat completion request."""
    messages = _convert_messages(request)
    if not messages:
        raise AnthropicRequestError("messages must not be empty")
    tools = _convert_tools(request)
    tool_choice = _convert_tool_choice(request, tools)

    chat_request: Dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_completion_tokens": request.max_tokens,
        "stream": bool(request.stream),
    }
    if tools is not None:
        chat_request["tools"] = [t.model_dump() for t in tools]
    if tool_choice is not None:
        chat_request["tool_choice"] = (
            tool_choice if isinstance(tool_choice, str) else tool_choice.model_dump()
        )
    if request.temperature is not None:
        chat_request["temperature"] = request.temperature
    if request.top_p is not None:
        chat_request["top_p"] = request.top_p
    if request.top_k is not None:
        chat_request["top_k"] = request.top_k
    if request.stop_sequences:
        chat_request["stop"] = list(request.stop_sequences)

    # Anthropic extended-thinking controls need to reach both the tokenizer
    # template and the reasoning postprocessor.  DeepSeek-V4 selects thinking
    # mode from these template kwargs; without them, configuring the V4
    # reasoning parser alone leaves it in identity mode.
    chat_template_kwargs: Dict[str, Any] = {}
    if request.thinking:
        thinking_type = request.thinking.get("type")
        if thinking_type == "enabled":
            chat_template_kwargs["enable_thinking"] = True
            budget_tokens = request.thinking.get("budget_tokens")
            if (
                isinstance(budget_tokens, bool)
                or not isinstance(budget_tokens, int)
                or budget_tokens < 1024
            ):
                raise AnthropicRequestError(
                    "thinking.type='enabled' requires budget_tokens >= 1024"
                )
            if budget_tokens >= request.max_tokens:
                raise AnthropicRequestError(
                    "thinking budget_tokens must be less than max_tokens"
                )
            chat_request["thinking_token_budget"] = budget_tokens
        elif thinking_type == "adaptive":
            if request.thinking.get("budget_tokens") is not None:
                raise AnthropicRequestError(
                    "thinking.type='adaptive' does not accept budget_tokens"
                )
            chat_template_kwargs["enable_thinking"] = True
        elif thinking_type == "disabled":
            if request.thinking.get("budget_tokens") is not None:
                raise AnthropicRequestError(
                    "thinking.type='disabled' does not accept budget_tokens"
                )
            chat_template_kwargs["enable_thinking"] = False
        else:
            raise AnthropicRequestError(f"Unsupported thinking type {thinking_type!r}")

    if request.output_config:
        reasoning_effort = request.output_config.get("effort")
        if isinstance(reasoning_effort, str) and reasoning_effort:
            chat_template_kwargs["reasoning_effort"] = reasoning_effort
        output_format = request.output_config.get("format")
        if output_format is not None:
            if not isinstance(output_format, dict):
                raise AnthropicRequestError("output_config.format must be an object")
            if output_format.get("type") != "json_schema":
                raise AnthropicRequestError(
                    "Only output_config.format type 'json_schema' is supported"
                )
            schema = output_format.get("schema")
            if not isinstance(schema, dict):
                raise AnthropicRequestError(
                    "output_config.format.schema must be an object"
                )
            chat_request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": schema},
            }

    if chat_template_kwargs:
        chat_request["chat_template_kwargs"] = chat_template_kwargs
    if request.stream:
        chat_request["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": True,
        }
    return ChatCompletionRequest(**chat_request)


# ---------------------------------------------------------------------------
# Response conversion: ChatCompletionResponse -> Anthropic
# ---------------------------------------------------------------------------


def map_stop_reason(finish_reason: Optional[str]) -> AnthropicStopReason:
    if finish_reason is None:
        return "end_turn"
    mapped = STOP_REASON_MAP.get(finish_reason)
    if mapped is None:
        logger.warning("Unmapped finish_reason %r defaulted to 'end_turn'", finish_reason)
        return "end_turn"
    return mapped


def _map_stop_result(
    finish_reason: Optional[str], stop_reason: Optional[Union[int, str]]
) -> tuple[AnthropicStopReason, Optional[str]]:
    if finish_reason == "stop" and isinstance(stop_reason, str):
        return "stop_sequence", stop_reason
    return map_stop_reason(finish_reason), None


def convert_usage(usage: Optional[UsageInfo]) -> AnthropicUsage:
    if usage is None:
        return AnthropicUsage()
    cached = 0
    if usage.prompt_tokens_details is not None:
        cached = usage.prompt_tokens_details.cached_tokens or 0
    input_tokens = max(usage.prompt_tokens - cached, 0)
    anthropic_usage = AnthropicUsage(
        input_tokens=input_tokens,
        output_tokens=usage.completion_tokens or 0,
    )
    if cached > 0:
        anthropic_usage.cache_read_input_tokens = cached
    return anthropic_usage


def convert_chat_response(chat_response: ChatCompletionResponse) -> AnthropicMessagesResponse:
    """Translate a non-streaming chat completion into an Anthropic message."""
    content: List[Any] = []
    stop_reason: AnthropicStopReason = "end_turn"
    stop_sequence: Optional[str] = None

    if chat_response.choices:
        choice = chat_response.choices[0]
        message = choice.message
        reasoning = message.reasoning_content or message.reasoning
        if reasoning:
            content.append(AnthropicThinkingBlock(thinking=reasoning))
        if message.content:
            content.append(AnthropicTextBlock(text=message.content))
        for tool_call in message.tool_calls:
            try:
                tool_input = json.loads(tool_call.function.arguments)
                if not isinstance(tool_input, dict):
                    raise ValueError("arguments is not a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                raise AnthropicResponseError(
                    f"Tool call {tool_call.function.name!r} arguments are not "
                    "a valid JSON object"
                ) from e
            content.append(
                AnthropicToolUseBlock(
                    id=tool_call.id, name=tool_call.function.name, input=tool_input
                )
            )
        stop_reason, stop_sequence = _map_stop_result(
            choice.finish_reason, choice.stop_reason
        )

    if not content:
        # Anthropic responses must carry at least one content block.
        content.append(AnthropicTextBlock(text=""))

    return AnthropicMessagesResponse(
        model=chat_response.model,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        usage=convert_usage(chat_response.usage),
    )


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE chunks -> Anthropic SSE events
# ---------------------------------------------------------------------------


class AnthropicStreamReframer:
    """Stateful reframer from OpenAI chat chunks to Anthropic SSE events.

    Consumes the ``data: <ChatCompletionStreamResponse json>`` lines produced
    by the ``openai_chat`` streaming path and emits the Anthropic event
    sequence::

        message_start
        (content_block_start (content_block_delta)* content_block_stop)*
        message_delta
        message_stop

    Invariants maintained: ``message_start`` is emitted exactly once and
    first; every content block is opened before any delta and closed before a
    block of another type (or another tool call) is opened; block indices are
    monotonically increasing; the delta type always matches the open block
    type.
    """

    def __init__(self, model: str):
        self.model = model
        self.message_id = f"msg_{uuid.uuid4().hex}"
        self.message_started = False
        self.block_index = -1
        self.open_block_type: Optional[str] = None
        self.open_tool_index: Optional[int] = None
        self.stop_reason: AnthropicStopReason = "end_turn"
        self.stop_sequence: Optional[str] = None
        self.final_usage: Optional[AnthropicUsage] = None

    # -- block state machine -------------------------------------------------

    def _close_block(self) -> List[str]:
        if self.open_block_type is None:
            return []
        event = AnthropicContentBlockStopEvent(index=self.block_index)
        self.open_block_type = None
        self.open_tool_index = None
        return [anthropic_sse(event)]

    def _open_block(
        self, block: Any, block_type: str, tool_index: Optional[int] = None
    ) -> List[str]:
        frames = self._close_block()
        self.block_index += 1
        self.open_block_type = block_type
        self.open_tool_index = tool_index
        frames.append(
            anthropic_sse(
                AnthropicContentBlockStartEvent(index=self.block_index, content_block=block)
            )
        )
        return frames

    def _ensure_block(self, block_type: str) -> List[str]:
        if self.open_block_type == block_type:
            return []
        if block_type == "text":
            return self._open_block(AnthropicTextBlock(text=""), "text")
        if block_type == "thinking":
            return self._open_block(AnthropicThinkingBlock(thinking=""), "thinking")
        raise ValueError(f"unexpected block type {block_type}")

    # -- chunk handling -------------------------------------------------------

    def _start_message(self, usage: Optional[AnthropicUsage]) -> List[str]:
        if self.message_started:
            return []
        self.message_started = True
        skeleton = AnthropicMessagesResponse(
            id=self.message_id,
            model=self.model,
            content=[],
            usage=usage or AnthropicUsage(),
        )
        # ``stop_reason``/``stop_sequence`` intentionally stay None here and
        # are delivered by the final message_delta.
        return [anthropic_sse(AnthropicMessageStartEvent(message=skeleton))]

    def process_chunk(self, chunk: ChatCompletionStreamResponse) -> List[str]:
        frames: List[str] = []

        usage = None
        if chunk.usage is not None:
            usage = convert_usage(chunk.usage)
            self.final_usage = usage
        if not self.message_started:
            start_usage = None
            if usage is not None:
                start_usage = AnthropicUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=0,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                )
            frames.extend(self._start_message(start_usage))

        for choice in chunk.choices:
            delta = choice.delta
            reasoning_delta = delta.reasoning_content or delta.reasoning
            if reasoning_delta:
                frames.extend(self._ensure_block("thinking"))
                frames.append(
                    anthropic_sse(
                        AnthropicContentBlockDeltaEvent(
                            index=self.block_index,
                            delta=AnthropicThinkingDelta(thinking=reasoning_delta),
                        )
                    )
                )
            if delta.content:
                frames.extend(self._ensure_block("text"))
                frames.append(
                    anthropic_sse(
                        AnthropicContentBlockDeltaEvent(
                            index=self.block_index, delta=AnthropicTextDelta(text=delta.content)
                        )
                    )
                )
            for tool_call in delta.tool_calls or []:
                function = tool_call.function
                if function is None:
                    continue
                if function.name:
                    # A named fragment starts a new tool call. Force a new
                    # block even if a tool_use block is already open so
                    # argument deltas of parallel calls never merge.
                    block = AnthropicToolUseBlock(
                        id=tool_call.id or f"toolu_{uuid.uuid4().hex}",
                        name=function.name,
                        input={},
                    )
                    frames.extend(self._open_block(block, "tool_use", tool_index=tool_call.index))
                if function.arguments:
                    if self.open_block_type != "tool_use" or (
                        tool_call.index is not None
                        and self.open_tool_index is not None
                        and tool_call.index != self.open_tool_index
                    ):
                        logger.warning(
                            "Dropping tool argument fragment without a "
                            "matching open tool_use block (index=%s)",
                            tool_call.index,
                        )
                        continue
                    frames.append(
                        anthropic_sse(
                            AnthropicContentBlockDeltaEvent(
                                index=self.block_index,
                                delta=AnthropicInputJsonDelta(partial_json=function.arguments),
                            )
                        )
                    )
            if choice.finish_reason:
                self.stop_reason, self.stop_sequence = _map_stop_result(
                    choice.finish_reason, choice.stop_reason
                )

        return frames

    def finish(self) -> List[str]:
        frames = self._close_block()
        frames.extend(self._start_message(None))  # degenerate empty stream
        frames.append(
            anthropic_sse(
                AnthropicMessageDeltaEvent(
                    delta=AnthropicMessageDelta(
                        stop_reason=self.stop_reason,
                        stop_sequence=self.stop_sequence,
                    ),
                    usage=self.final_usage or AnthropicUsage(),
                )
            )
        )
        frames.append(anthropic_sse(AnthropicMessageStopEvent()))
        return frames

    def error(self, message: str) -> List[str]:
        """Close any open block, then surface an error event."""
        frames = self._close_block()
        frames.append(
            anthropic_sse(
                AnthropicErrorEvent(error=AnthropicError(type="api_error", message=message))
            )
        )
        return frames


async def _iter_openai_sse_lines(
    openai_sse: AsyncIterator[Union[str, bytes]],
) -> AsyncIterator[str]:
    """Yield complete lines from string frames or arbitrarily chunked bytes.

    The in-process server produces complete SSE strings, while the
    disaggregated frontend relays raw ``aiohttp`` byte chunks.  Network chunks
    need not align with either UTF-8 characters or SSE line boundaries, so
    retain incomplete input until the next chunk arrives.
    """
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    input_type: Optional[type] = None

    async for payload in openai_sse:
        if not isinstance(payload, (str, bytes)):
            raise TypeError(
                f"OpenAI SSE payload must be str or bytes, got {type(payload)!r}"
            )

        current_type = type(payload)
        if input_type is None:
            input_type = current_type
        elif current_type is not input_type:
            raise TypeError("OpenAI SSE stream cannot mix str and bytes payloads")

        if isinstance(payload, bytes):
            buffer += decoder.decode(payload)
        else:
            buffer += payload

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line.rstrip("\r")

    if input_type is bytes:
        buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer.rstrip("\r")


async def reframe_openai_stream(
    openai_sse: AsyncIterator[Union[str, bytes]], model: str
) -> AsyncIterator[str]:
    """Translate an OpenAI SSE stream into Anthropic SSE frames."""
    reframer = AnthropicStreamReframer(model=model)
    try:
        async for line in _iter_openai_sse_lines(openai_sse):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data:
                continue
            if data == "[DONE]":
                for frame in reframer.finish():
                    yield frame
                return
            try:
                chunk = ChatCompletionStreamResponse(**json.loads(data))
            except (json.JSONDecodeError, ValueError) as e:
                raise AnthropicResponseError("Malformed upstream stream chunk") from e
            for frame in reframer.process_chunk(chunk):
                yield frame
        # Upstream ended without [DONE]; still terminate the message cleanly.
        for frame in reframer.finish():
            yield frame
    except Exception as e:  # noqa: BLE001 - stream must end with an event
        logger.error(
            f"Anthropic stream reframing failed: {e}\n{traceback.format_exc()}"
        )
        for frame in reframer.error("Internal server error"):
            yield frame
