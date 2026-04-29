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

from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer


class _DummyTokenizer:
    all_special_tokens = []
    eos_token_id = 1
    pad_token_id = 0
    name_or_path = "dummy"

    def encode(self, text, *args, **kwargs):
        self.last_encoded_text = text
        return [1, 2, 3]


def test_deepseek_v4_chat_template_matches_reference_single_user_prompt():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Question: 1+1?\nAnswer:",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>Question: 1+1?\nAnswer:<｜Assistant｜></think>"
    )


def test_deepseek_v4_chat_template_tokenize_uses_rendered_prompt():
    dummy = _DummyTokenizer()
    tokenizer = DeepseekV4Tokenizer(dummy)

    token_ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )

    assert token_ids == [1, 2, 3]
    assert dummy.last_encoded_text == (
        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
    )


def test_deepseek_v4_custom_tokenizer_reuses_loaded_wrapper():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    args = TorchLlmArgs(model="dummy", tokenizer=tokenizer, custom_tokenizer="deepseek_v4")

    assert args.tokenizer is tokenizer


def test_deepseek_v4_server_chat_template_path_uses_custom_tokenizer():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = apply_chat_template(
        model_type="deepseek_v4",
        tokenizer=tokenizer,
        processor=None,
        conversation=[
            {
                "role": "user",
                "content": "hello",
            }
        ],
        add_generation_prompt=True,
        mm_placeholder_counts=[{}],
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>")


# ---------------------------------------------------------------------------
# Tool-call / tool-result / thinking-mode tests (vendored official encoder)
# ---------------------------------------------------------------------------

_WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


def test_deepseek_v4_tools_kwarg_renders_tool_schema_into_system_block():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Beijing?"},
        ],
        tools=_WEATHER_TOOLS,
        add_generation_prompt=True,
    )

    # Tools block is rendered after the system content in DSML form.
    assert prompt.startswith("<｜begin▁of▁sentence｜>You are a helpful assistant.")
    assert "## Tools" in prompt
    assert "<｜DSML｜tool_calls>" in prompt
    assert "get_weather" in prompt
    assert "<｜User｜>What's the weather in Beijing?" in prompt
    assert prompt.endswith("<｜Assistant｜></think>")


def test_deepseek_v4_tools_kwarg_prepends_synthetic_system_when_missing():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tools=_WEATHER_TOOLS,
        add_generation_prompt=True,
    )

    # Synthetic system message has empty content but still emits the tools block.
    assert "## Tools" in prompt
    assert "get_weather" in prompt


def test_deepseek_v4_assistant_tool_calls_render_as_dsml_invoke_blocks():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "What's the weather in Beijing?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Beijing"}',
                        },
                    }
                ],
            },
        ],
        add_generation_prompt=False,
    )

    assert '<｜DSML｜invoke name="get_weather">' in prompt
    assert '<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>' in prompt
    assert "</｜DSML｜tool_calls>" in prompt


def test_deepseek_v4_tool_role_merges_into_following_user_turn():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Weather in Beijing?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Beijing"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 20C"},
            {"role": "user", "content": "Thanks! How about Shanghai?"},
        ],
        add_generation_prompt=True,
    )

    # The tool message is rendered as a <tool_result> block on a user turn,
    # NOT as a standalone role=tool message (DSv4 has no such role).
    assert "<tool_result>Sunny, 20C</tool_result>" in prompt
    assert "Thanks! How about Shanghai?" in prompt
    assert prompt.endswith("<｜Assistant｜></think>")


def test_deepseek_v4_add_generation_prompt_false_strips_trailing_assistant():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=False
    )

    assert prompt == "<｜begin▁of▁sentence｜><｜User｜>hi"
    assert "<｜Assistant｜>" not in prompt


def test_deepseek_v4_thinking_mode_adds_think_token():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        chat_template_kwargs={"thinking_mode": "thinking"},
    )

    # With thinking_mode=thinking the assistant prompt is followed by the
    # opening <think> tag instead of the closing </think>.
    assert prompt.endswith("<｜Assistant｜><think>")


def test_deepseek_v4_enable_thinking_alias_equivalent_to_thinking_mode():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt_a = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True},
    )
    prompt_b = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        chat_template_kwargs={"thinking_mode": "thinking"},
    )

    assert prompt_a == prompt_b


def test_deepseek_v4_invalid_thinking_mode_raises():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    import pytest

    with pytest.raises(ValueError):
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"thinking_mode": "bogus"},
        )


def test_deepseek_v4_tool_call_path_does_not_mutate_caller_messages():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Beijing"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
    ]
    snapshot = repr(messages)

    tokenizer.apply_chat_template(messages, tools=_WEATHER_TOOLS)

    assert repr(messages) == snapshot
