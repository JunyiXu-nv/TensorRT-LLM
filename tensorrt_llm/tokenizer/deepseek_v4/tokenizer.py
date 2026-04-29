# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..tokenizer import TransformersTokenizer
from . import _official_encoding as _dsv4_enc

BOS_TOKEN = _dsv4_enc.bos_token
EOS_TOKEN = _dsv4_enc.eos_token
USER_TOKEN = _dsv4_enc.USER_SP_TOKEN
ASSISTANT_TOKEN = _dsv4_enc.ASSISTANT_SP_TOKEN
THINKING_START_TOKEN = _dsv4_enc.thinking_start_token
THINKING_END_TOKEN = _dsv4_enc.thinking_end_token


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n\n".join(parts)
    return str(content)


def _normalize_messages(messages):
    """Deep-copy messages and coerce non-string ``content`` to text.

    The official encoder expects a plain string for system / assistant / tool /
    developer messages. Leaves user-role ``content`` untouched so that
    list-of-block input (e.g. multimodal) can flow through unchanged — the
    official encoder supports ``content_blocks`` on user messages, and
    ``merge_tool_messages`` re-builds ``content_blocks`` from scratch anyway.
    """
    normalized = []
    for msg in messages:
        msg = copy.deepcopy(msg)
        role = msg.get("role")
        if role in ("system", "developer", "assistant", "tool"):
            content = msg.get("content")
            if content is not None and not isinstance(content, str):
                msg["content"] = _message_content_to_text(content)
        normalized.append(msg)
    return normalized


def _resolve_template_kwargs(kwargs):
    """Resolve thinking-mode knobs from kwargs or ``chat_template_kwargs``.

    OpenAI-style front-ends pass template knobs through a nested
    ``chat_template_kwargs`` dict; others pass them as top-level kwargs. Check
    both locations so the shim works in either case.
    """
    template_kwargs = kwargs.get("chat_template_kwargs") or {}

    def _pick(name, default):
        if name in kwargs:
            return kwargs[name]
        if name in template_kwargs:
            return template_kwargs[name]
        return default

    enable_thinking = _pick("enable_thinking", None)
    thinking_mode = _pick("thinking_mode", None)
    if thinking_mode is None:
        thinking_mode = "thinking" if enable_thinking else "chat"
    if thinking_mode not in ("chat", "thinking"):
        raise ValueError(f"Invalid thinking_mode {thinking_mode!r}; expected 'chat' or 'thinking'.")

    reasoning_effort = _pick("reasoning_effort", None)
    drop_thinking = _pick("drop_thinking", True)

    return thinking_mode, reasoning_effort, drop_thinking


def _attach_tools(messages, tools):
    """Attach the caller-provided tool schema to a system message.

    The official encoder reads ``tools`` off individual messages
    (``msg.get("tools")``), not from a separate argument. Attach them to the
    first system message, or prepend a synthetic empty system message if none
    exists.
    """
    if not tools:
        return messages

    for msg in messages:
        if msg.get("role") == "system":
            # Don't clobber a caller-provided per-message tools list.
            if not msg.get("tools"):
                msg["tools"] = tools
            return messages

    return [{"role": "system", "content": "", "tools": tools}, *messages]


def _strip_generation_prompt(rendered, thinking_mode, drop_thinking):
    """Strip the trailing assistant-prompt suffix from ``rendered``.

    The official encoder unconditionally appends
    ``<｜Assistant｜>[<think>|</think>]`` when the last message is ``user`` or
    ``developer``. This helper removes that suffix for callers that asked for
    ``add_generation_prompt=False``.
    """
    # Trailing thinking-start / thinking-end token.
    for suffix in (THINKING_START_TOKEN, THINKING_END_TOKEN):
        if rendered.endswith(ASSISTANT_TOKEN + suffix):
            return rendered[: -len(ASSISTANT_TOKEN + suffix)]
    # Fallback: just the assistant marker (shouldn't normally happen).
    if rendered.endswith(ASSISTANT_TOKEN):
        return rendered[: -len(ASSISTANT_TOKEN)]
    return rendered


class DeepseekV4Tokenizer(TransformersTokenizer):
    """DeepSeek-V4 tokenizer with the checkpoint reference chat format.

    Chat formatting (including tool-calls, tool-result messages and
    thinking mode) is delegated to the vendored official encoder in
    :mod:`_official_encoding`. That encoder is a verbatim copy of
    ``encoding_dsv4.py`` from the upstream DeepSeek-V4 checkpoint, so the
    prompt format stays in lock-step with what the model was trained on.
    """

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "DeepseekV4Tokenizer":
        tokenizer = AutoTokenizer.from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
        return cls(tokenizer)

    def apply_chat_template(self, messages, tools=None, **kwargs):
        add_generation_prompt = kwargs.get("add_generation_prompt", True)
        tokenize = kwargs.get("tokenize", False)

        thinking_mode, reasoning_effort, drop_thinking = _resolve_template_kwargs(kwargs)

        # Deep-copy + coerce content so that downstream mutation inside
        # merge_tool_messages / render_message never aliases caller state.
        messages = _normalize_messages(messages)
        messages = _attach_tools(messages, tools)

        rendered = _dsv4_enc.encode_messages(
            messages,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            add_default_bos_token=True,
            reasoning_effort=reasoning_effort,
        )

        if not add_generation_prompt:
            # The official encoder unconditionally appends the assistant
            # prompt after a trailing user/developer turn. Strip it back off
            # so callers that only want the conversation prefix get what they
            # asked for.
            last_role = messages[-1].get("role") if messages else None
            if last_role in ("user", "developer"):
                rendered = _strip_generation_prompt(rendered, thinking_mode, drop_thinking)

        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered
