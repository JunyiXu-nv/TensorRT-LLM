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

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Optional, cast

from transformers import PretrainedConfig

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

if TYPE_CHECKING:
    from tensorrt_llm.serve.harmony_adapter import HarmonyAdapter

ToolDict = dict[str, object]


def resolve_model_type_from_config(model_name_or_path: str) -> Optional[str]:
    """Return the checkpoint's declared model type from its config metadata."""
    config_dict, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    model_type = config_dict.get("model_type")
    return model_type if isinstance(model_type, str) else None


def uses_harmony_tokenization(
    use_harmony: Optional[bool] = None,
    model_type: Optional[str] = None,
    model_type_resolver: Optional[Callable[[], Optional[str]]] = None,
) -> bool:
    if os.getenv("DISABLE_HARMONY_ADAPTER", "0") == "1":
        return False
    if use_harmony is not None:
        return use_harmony
    if model_type is None and model_type_resolver is not None:
        model_type = model_type_resolver()
    return model_type == "gpt_oss"


def get_chat_completion_tool_dicts(
    request: ChatCompletionRequest, empty_as_none: bool = False
) -> Optional[list[ToolDict]]:
    if request.tools is None or (empty_as_none and not request.tools):
        return None
    tools: list[ToolDict] = []
    for tool in request.tools:
        if hasattr(tool, "model_dump"):
            tools.append(cast(ToolDict, tool.model_dump()))
        elif isinstance(tool, dict):
            tools.append(cast(ToolDict, tool))
        else:
            raise TypeError(f"Unsupported tool type: {type(tool).__name__}")
    return tools


def tokenize_harmony_chat_request(
    request: ChatCompletionRequest,
    harmony_adapter: Optional["HarmonyAdapter"] = None,
    set_prompt_token_ids: bool = False,
) -> list[int]:
    if request.prompt_token_ids is not None:
        return request.prompt_token_ids

    from tensorrt_llm.serve import harmony_adapter as harmony_adapter_module

    adapter = harmony_adapter or harmony_adapter_module.get_harmony_adapter()
    result = adapter.openai_to_harmony_tokens(
        request.messages,
        get_chat_completion_tool_dicts(request, empty_as_none=True),
        reasoning_effort=harmony_adapter_module.maybe_transform_reasoning_effort(
            request.reasoning_effort
        ),
        tool_choice=request.tool_choice,
    )
    if set_prompt_token_ids:
        request.prompt_token_ids = result
    return result


def apply_reasoning_effort_to_template_kwargs(
    request: ChatCompletionRequest, chat_template_kwargs: dict
) -> dict:
    """Put the request's reasoning_effort where a chat template can read it.

    reasoning_effort reaches a template only if it is placed in the template
    kwargs. Without this the field is validated and then dropped for every model
    that is neither gpt_oss (harmony encodes it instead of rendering a template)
    nor kimi_k3 (its serving extension derives its own kwargs).

    Gated on model_fields_set because the field defaults to LOW: passing it
    unconditionally would state an effort for every request that never asked for
    one. Lowercased because the harmony enum spells its members 'High' while
    templates test for 'high'. An explicit chat_template_kwargs entry wins,
    since that is the caller addressing the template directly.

    Without it GLM-5.3 defaults to its maximum effort on chat completions and
    can spend an entire max_tokens budget thinking without emitting a tool call.
    """
    if (
        "reasoning_effort" in request.model_fields_set
        and request.reasoning_effort is not None
        and "reasoning_effort" not in chat_template_kwargs
    ):
        chat_template_kwargs["reasoning_effort"] = getattr(
            request.reasoning_effort, "value", request.reasoning_effort
        ).lower()
    return chat_template_kwargs


def chat_template_kwargs_for_request(
    request: ChatCompletionRequest,
) -> tuple[dict, Optional[list[str]]]:
    """Template kwargs a chat request renders with, and the server-injected keys.

    Starts from the caller's `chat_template_kwargs` and forwards
    `reasoning_effort` (`apply_reasoning_effort_to_template_kwargs`). The
    second value names the keys the unused-kwargs guard must exempt (see
    `validate_chat_template_kwargs`): those the request already marks as
    server-derived (`injected_chat_template_kwargs`, e.g. set by the Anthropic
    adapter) plus any added here. `reasoning_effort` comes from an API-level
    field, so a template that does not read it ignores it rather than failing
    the request; a `reasoning_effort` the caller put in `chat_template_kwargs`
    itself stays subject to the guard.
    """
    chat_template_kwargs = dict(request.chat_template_kwargs or {})
    caller_keys = set(chat_template_kwargs)
    apply_reasoning_effort_to_template_kwargs(request, chat_template_kwargs)
    injected = set(getattr(request, "injected_chat_template_kwargs", None) or ())
    injected |= set(chat_template_kwargs) - caller_keys
    return chat_template_kwargs, sorted(injected) or None
