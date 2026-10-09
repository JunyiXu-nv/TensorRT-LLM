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
"""Kimi-K3 parsers: the streamed and the whole-text view of one generation must agree.

The serving layer reads every generation twice. The stream runs
``KimiK3ReasoningParser.parse_delta`` + ``finish`` and feeds the content it yields to
``KimiK3ToolParser.parse_streaming_increment`` + ``finish``; the final snapshot re-reads the
accumulated text with ``KimiK3ReasoningParser.parse`` and ``KimiK3ToolParser.detect_and_parse``.
Streamed bytes cannot be taken back, so the snapshot has to say exactly what the stream said,
under every way the detokenizer can chunk the text - and neither view may lose anything the
model generated.

The corpus pins one reading per generation shape (``test_reading``); the property tests check
that every chunking of every sample yields that same reading.
"""

import json
import random
from typing import Iterator, List, Optional, Tuple
from unittest.mock import patch

import pytest

from tensorrt_llm.llmapi.reasoning_parser import KimiK3ReasoningParser, ReasoningParserFactory
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam, FunctionDefinition
from tensorrt_llm.serve.tool_parser.kimi_k3_tool_parser import KimiK3ToolParser

pytestmark = pytest.mark.cpu_only

OPEN, CLOSE, SEP, EOM = "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"
BOT = f"{OPEN}tools{SEP}"
EOT = f"{CLOSE}tools{SEP}"
MESSAGE_END = f"{CLOSE}message{SEP}"


def _think(text: str) -> str:
    """The think body; the generation prompt already opened the channel."""
    return f"{text}{CLOSE}think{SEP}"


def _response(text: str) -> str:
    return f"{OPEN}response{SEP}{text}{CLOSE}response{SEP}"


def _argument(key: str, type_: str, value: str) -> str:
    return f'{OPEN}argument key="{key}" type="{type_}"{SEP}{value}{CLOSE}argument{SEP}'


def _call(name: str, index: int, body: str) -> str:
    return f'{OPEN}call tool="{name}" index="{index}"{SEP}{body}{CLOSE}call{SEP}'


def _section(*parts: str) -> str:
    return BOT + "".join(parts) + EOT


def _exec(index: int, command: str) -> str:
    return _call("exec", index, _argument("input", "string", command))


def _exec_args(command: str) -> str:
    return json.dumps({"input": command}, ensure_ascii=False)


TOOLS = [
    ChatCompletionToolsParam(
        type="function",
        function=FunctionDefinition(
            name=name,
            parameters={"type": "object", "properties": {"input": {"type": "string"}}},
        ),
    )
    for name in (
        "exec",
        "functions.exec",
        "functions.update_plan",
        "collaboration.spawn_agent",
    )
]

Calls = List[Tuple[str, str]]

# name -> (generation, thinking, reasoning, visible text, delivered calls)
#
# The generation is everything after the generation prompt (which ends inside
# `<|open|>think<|sep|>`, or `<|open|>response<|sep|>` with thinking off). The reading is
# what a client must get from the two parsers chained the way serving chains them.
CORPUS = {
    "single_call": (
        _think("I will list.")
        + _response("Listing.")
        + _section(_call("functions.exec", 1, _argument("input", "string", "ls -la")))
        + MESSAGE_END,
        True,
        "I will list.",
        "Listing.",
        [("functions.exec", '{"input": "ls -la"}')],
    ),
    "single_call_with_eom": (
        _think("I will list.")
        + _response("Listing.")
        + _section(_exec(1, "ls -la"))
        + MESSAGE_END
        + EOM,
        True,
        "I will list.",
        "Listing.",
        [("exec", '{"input": "ls -la"}')],
    ),
    "multi_call_types": (
        _think("plan")
        + _response("")
        + _section(
            _exec(1, "cat a.py"),
            _call(
                "functions.update_plan",
                2,
                _argument("plan", "array", '[{"step": "x", "done": false}]')
                + _argument("n", "number", "3")
                + _argument("ok", "boolean", "true")
                + _argument("z", "null", "null"),
            ),
            _call(
                "collaboration.spawn_agent",
                3,
                _argument("task", "object", '{"a": {"b": [1, "x\\"y"]}, "u": "日本 ü"}'),
            ),
        )
        + MESSAGE_END,
        True,
        "plan",
        "",
        [
            ("exec", '{"input": "cat a.py"}'),
            (
                "functions.update_plan",
                '{"plan": [{"step": "x", "done": false}], "n": 3, "ok": true, "z": null}',
            ),
            ("collaboration.spawn_agent", '{"task": {"a": {"b": [1, "x\\"y"]}, "u": "日本 ü"}}'),
        ],
    ),
    "prose_only": (
        _think("hmm") + _response("Here is the answer:\n\n```py\nx = 1 < 2\n```\n") + MESSAGE_END,
        True,
        "hmm",
        "Here is the answer:\n\n```py\nx = 1 < 2\n```\n",
        [],
    ),
    "non_thinking_call": (
        f"Sure.{CLOSE}response{SEP}" + _section(_exec(1, "pwd")) + MESSAGE_END + EOM,
        False,
        "",
        "Sure.",
        [("exec", '{"input": "pwd"}')],
    ),
    # Defect 1: text after the section was streamed but missing from the snapshot.
    "prose_after_tools": (
        _think("t")
        + _response("Before.")
        + _section(_exec(1, "pwd"))
        + _response("After the call.")
        + MESSAGE_END,
        True,
        "t",
        "Before.After the call.",
        [("exec", '{"input": "pwd"}')],
    ),
    "junk_after_tools": (
        _think("t") + _response("") + _section(_exec(1, "ls")) + "\n trailing junk " + MESSAGE_END,
        True,
        "t",
        "\n trailing junk ",
        [("exec", '{"input": "ls"}')],
    ),
    # Defect 2: a second section numbered its calls from 0 again.
    "second_tools_section": (
        _think("t")
        + _response("")
        + _section(_exec(1, "ls"))
        + f"{OPEN}think{SEP}"
        + _think("more thinking")
        + _section(_exec(1, "pwd"))
        + MESSAGE_END,
        True,
        "tmore thinking",
        "",
        [("exec", '{"input": "ls"}'), ("exec", '{"input": "pwd"}')],
    ),
    "section_reopened_without_close": (
        _think("t")
        + _response("")
        + BOT
        + _exec(1, "ls")
        + _section(_exec(2, "pwd"))
        + MESSAGE_END,
        True,
        "t",
        "",
        [("exec", '{"input": "ls"}'), ("exec", '{"input": "pwd"}')],
    ),
    # Defect 4: a call cut off mid-block is model output and is released as text.
    "unterminated_call_eos": (
        _think("t")
        + _response("Running:")
        + BOT
        + f'{OPEN}call tool="exec" index="1"{SEP}'
        + f'{OPEN}argument key="input" type="string"{SEP}ls -l',
        True,
        "t",
        "Running:"
        + BOT
        + f'{OPEN}call tool="exec" index="1"{SEP}'
        + f'{OPEN}argument key="input" type="string"{SEP}ls -l',
        [],
    ),
    "unterminated_section_two_calls": (
        _think("t")
        + _response("")
        + BOT
        + _exec(1, "ls")
        + f'{OPEN}call tool="exec" index="2"{SEP}{OPEN}argument key="input"',
        True,
        "t",
        f'{OPEN}call tool="exec" index="2"{SEP}{OPEN}argument key="input"',
        [("exec", '{"input": "ls"}')],
    ),
    # The call's close tag is missing, but its argument is complete and the
    # section closes right after it: there is nothing else the model can have
    # meant, so it is the call.
    "call_missing_close_call": (
        _think("t")
        + _response("")
        + BOT
        + f'{OPEN}call tool="exec" index="1"{SEP}'
        + _argument("input", "string", "ls")
        + EOT
        + MESSAGE_END,
        True,
        "t",
        "",
        [("exec", '{"input": "ls"}')],
    ),
    # The shapes below are close tags the model left out in production (10-09);
    # each call used to reach the client as text, so its action never ran.
    # A value followed straight by the call's close tag; generation stopped
    # there (parallel_tool_calls=false).
    "argument_missing_close_at_the_call_close": (
        _think("t")
        + _response("Checking:")
        + BOT
        + f'{OPEN}call tool="exec" index="1"{SEP}'
        + f'{OPEN}argument key="input" type="string"{SEP}const r = 1;'
        + f"{CLOSE}call{SEP}",
        True,
        "t",
        "Checking:",
        [("exec", '{"input": "const r = 1;"}')],
    ),
    "argument_missing_close_before_the_next_call": (
        _think("t")
        + _response("")
        + BOT
        + f'{OPEN}call tool="exec" index="1"{SEP}'
        + f'{OPEN}argument key="input" type="string"{SEP}pwd{CLOSE}call{SEP}'
        + _exec(2, "ls")
        + EOT
        + MESSAGE_END,
        True,
        "t",
        "",
        [("exec", '{"input": "pwd"}'), ("exec", '{"input": "ls"}')],
    ),
    # The model closed its response and its message with a value still open.
    "message_closed_inside_a_value": (
        _think("t")
        + _response("Checking:")
        + BOT
        + f'{OPEN}call tool="exec" index="1"{SEP}'
        + f'{OPEN}argument key="input" type="string"{SEP}const r = 2;'
        + f"{CLOSE}response{SEP}"
        + MESSAGE_END
        + EOM,
        True,
        "t",
        "Checking:",
        [("exec", '{"input": "const r = 2;"}')],
    ),
    "call_missing_close_before_the_next_call": (
        _think("t")
        + _response("")
        + BOT
        + f'{OPEN}call tool="exec" index="1"{SEP}'
        + _argument("input", "string", "ls")
        + _call("write_stdin", 2, _argument("session_id", "number", "5"))
        + EOT
        + MESSAGE_END,
        True,
        "t",
        "",
        [("exec", '{"input": "ls"}'), ("write_stdin", '{"session_id": 5}')],
    ),
    "call_left_open_by_a_response_close": (
        _think("t")
        + _response("")
        + BOT
        + f'{OPEN}call tool="exec" index="1"{SEP}'
        + _argument("input", "string", "ls")
        + f"{CLOSE}response{SEP}"
        + MESSAGE_END,
        True,
        "t",
        "",
        [("exec", '{"input": "ls"}')],
    ),
    "call_without_tool_attr": (
        _think("t")
        + _response("")
        + _section(
            f'{OPEN}call index="1"{SEP}' + _argument("input", "string", "ls") + f"{CLOSE}call{SEP}"
        )
        + MESSAGE_END,
        True,
        "t",
        _section(
            f'{OPEN}call index="1"{SEP}' + _argument("input", "string", "ls") + f"{CLOSE}call{SEP}"
        ),
        [],
    ),
    "missing_section_close_before_message_close": (
        _think("t") + _response("") + BOT + _exec(1, "ls") + MESSAGE_END,
        True,
        "t",
        "",
        [("exec", '{"input": "ls"}')],
    ),
    # Defect 5: a literal close tag inside a value, and the model's own JSON text.
    "value_contains_close_argument": (
        _think("t")
        + _response("")
        + _section(_exec(1, f"grep -n '{CLOSE}argument{SEP}' parser.py"))
        + MESSAGE_END,
        True,
        "t",
        "",
        [("exec", _exec_args(f"grep -n '{CLOSE}argument{SEP}' parser.py"))],
    ),
    "value_contains_section_close": (
        _think("t") + _response("") + _section(_exec(1, f"echo '{EOT}' done")) + MESSAGE_END,
        True,
        "t",
        "",
        [("exec", _exec_args(f"echo '{EOT}' done"))],
    ),
    "typed_values_raw_json": (
        _think("t")
        + _response("")
        + _section(
            _call(
                "functions.update_plan",
                1,
                _argument("n", "number", "1e3")
                + _argument("big", "number", "123456789012345678901234567890")
                + _argument("f", "number", "1.10")
                + _argument("obj", "object", '{"a" : 1}'),
            )
        )
        + MESSAGE_END,
        True,
        "t",
        "",
        [
            (
                "functions.update_plan",
                '{"n": 1e3, "big": 123456789012345678901234567890, "f": 1.10, "obj": {"a" : 1}}',
            )
        ],
    ),
    "json_block": (
        _think("t")
        + _response("")
        + _section(
            _call(
                "functions.update_plan",
                1,
                f'{OPEN}json type="object"{SEP}{{"plan": "a \\"quoted\\" plan", "n": 1e3}}'
                f"{CLOSE}json{SEP}",
            )
        )
        + MESSAGE_END,
        True,
        "t",
        "",
        [("functions.update_plan", '{"plan": "a \\"quoted\\" plan", "n": 1e3}')],
    ),
    "json_block_invalid": (
        _think("t")
        + _response("")
        + _section(
            _call(
                "functions.update_plan",
                1,
                f'{OPEN}json type="object"{SEP}{{"plan": oops}}{CLOSE}json{SEP}',
            )
        )
        + MESSAGE_END,
        True,
        "t",
        "",
        [("functions.update_plan", '{"plan": oops}')],
    ),
    "escaped_attr_and_unicode": (
        _think("t")
        + _response("")
        + _section(
            _call(
                "exec",
                1,
                f'{OPEN}argument key="in&quot;put" type="string"{SEP}echo "é\U0001f600" && x'
                f"{CLOSE}argument{SEP}",
            )
        )
        + MESSAGE_END,
        True,
        "t",
        "",
        [("exec", '{"in\\"put": "echo \\"é\U0001f600\\" && x"}')],
    ),
    # Defect 6: literal markup in prose is text, not structure.
    "prose_with_tool_tokens": (
        _think("t")
        + _response(f"K3 tool calls start with {BOT} and end with {EOT}; that is the format.")
        + MESSAGE_END,
        True,
        "t",
        f"K3 tool calls start with {BOT} and end with {EOT}; that is the format.",
        [],
    ),
    "prose_with_open_no_sep": (
        _think("t")
        + _response(f"A literal {OPEN} marker, then more prose that matters.")
        + MESSAGE_END,
        True,
        "t",
        f"A literal {OPEN} marker, then more prose that matters.",
        [],
    ),
    "prose_with_eom": (
        _think("t")
        + _response(f"The stop token is {EOM} and after it comes more text.")
        + MESSAGE_END,
        True,
        "t",
        f"The stop token is {EOM} and after it comes more text.",
        [],
    ),
    "prose_with_message_close": (
        _think("t")
        + _response(f"Messages end with {MESSAGE_END} and then more.")
        + MESSAGE_END
        + EOM,
        True,
        "t",
        f"Messages end with {MESSAGE_END} and then more.",
        [],
    ),
    "reasoning_with_markup": (
        _think(f"Recall the format {BOT}...{EOT} - ok.") + _response("Done.") + MESSAGE_END,
        True,
        f"Recall the format {BOT}...{EOT} - ok.",
        "Done.",
        [],
    ),
    "reasoning_quotes_unclosed_call": (
        _think(f'I could emit {BOT}{OPEN}call tool="exec" index="1"{SEP}... but no.')
        + _response("Answer.")
        + MESSAGE_END,
        True,
        f'I could emit {BOT}{OPEN}call tool="exec" index="1"{SEP}... but no.',
        "Answer.",
        [],
    ),
    # Defect 6: channel-less text and reasoning inside a section are not dropped.
    "no_response_channel_after_think": (
        _think("t") + "Answer written without a response tag." + MESSAGE_END,
        True,
        "t",
        "Answer written without a response tag.",
        [],
    ),
    "think_inside_tools_section": (
        _think("t")
        + _response("")
        + _section(
            _exec(1, "ls"), f"{OPEN}think{SEP}reconsidering{CLOSE}think{SEP}", _exec(2, "pwd")
        )
        + MESSAGE_END,
        True,
        "treconsidering",
        "",
        [("exec", '{"input": "ls"}'), ("exec", '{"input": "pwd"}')],
    ),
    "prose_between_calls": (
        _think("t")
        + _response("")
        + _section(_exec(1, "ls"), "\nNow the second call:\n", _exec(2, "pwd"))
        + MESSAGE_END,
        True,
        "t",
        "\nNow the second call:\n",
        [("exec", '{"input": "ls"}'), ("exec", '{"input": "pwd"}')],
    ),
    "section_inside_unclosed_think": (
        "reasoning"
        + _section(_exec(1, "ls"))
        + "more reasoning"
        + f"{CLOSE}think{SEP}"
        + _response("ok")
        + MESSAGE_END,
        True,
        "reasoningmore reasoning",
        "ok",
        [("exec", '{"input": "ls"}')],
    ),
    # A stray close is not a channel transition; it stays in the text.
    "double_think_close": (
        _think("t") + f"{CLOSE}think{SEP}" + _response("Visible.") + MESSAGE_END,
        True,
        "t",
        f"{CLOSE}think{SEP}Visible.",
        [],
    ),
    # Whitespace between two structural tags is formatting, consumed with them.
    "newlines_between_tags": (
        _think("t")
        + "\n"
        + _response("Hi.")
        + "\n"
        + _section("\n" + _call("exec", 1, "\n" + _argument("input", "string", "ls") + "\n") + "\n")
        + "\n"
        + MESSAGE_END
        + "\n"
        + EOM,
        True,
        "t",
        "Hi.",
        [("exec", '{"input": "ls"}')],
    ),
    "content_ends_partial_lt": (
        _think("t") + _response("compare a <") + MESSAGE_END,
        True,
        "t",
        "compare a <",
        [],
    ),
    "content_ends_partial_opener_eos": (
        _think("t") + f"{OPEN}response{SEP}compare a <|op",
        True,
        "t",
        "compare a <|op",
        [],
    ),
}


def _splits(text: str, seed: int) -> Iterator[Tuple[str, List[str]]]:
    """Every way the tests chunk one text.

    Whole, per character, fixed sizes, every 2-way split, and seeded random splits
    interleaved with the empty increments the serving layer's end-of-stream drain sends.
    """
    n = len(text)
    yield "whole", [text]
    yield "char", list(text)
    for size in (2, 3, 5, 7, 11, 16):
        yield f"fixed{size}", [text[i : i + size] for i in range(0, n, size)]
    for cut in range(1, n):
        yield f"2way@{cut}", [text[:cut], text[cut:]]
    rng = random.Random(seed)
    for k in range(20):
        cuts = sorted(rng.sample(range(1, n), min(n - 1, rng.randint(1, 8)))) if n > 1 else []
        bounds = [0, *cuts, n]
        chunks = [text[a:b] for a, b in zip(bounds, bounds[1:])]
        yield f"rand{k}", [piece for chunk in chunks for piece in (chunk, "")]


def _reasoning_parser(thinking: bool) -> KimiK3ReasoningParser:
    kwargs = None if thinking else {"thinking": False}
    return ReasoningParserFactory.create_reasoning_parser("kimi_k3", kwargs)


def _calls(items) -> List[Tuple[int, str, str]]:
    return [(item.tool_index, item.name, item.parameters) for item in items]


def _tool_stream(chunks: List[str]) -> Tuple[str, List[Tuple[int, str, str]]]:
    parser = KimiK3ToolParser()
    text, calls = [], []
    for chunk in chunks:
        result = parser.parse_streaming_increment(chunk, TOOLS)
        text.append(result.normal_text)
        calls.extend(result.calls)
    result = parser.finish(TOOLS)
    text.append(result.normal_text)
    calls.extend(result.calls)
    # Whatever the parser still holds after finish() is lost to both views.
    assert parser._buffer == ""
    return "".join(text), _calls(calls)


def _tool_whole(text: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    result = KimiK3ToolParser().detect_and_parse(text, TOOLS)
    return result.normal_text, _calls(result.calls)


def _reasoning_stream(chunks: List[str], thinking: bool) -> Tuple[str, str]:
    parser = _reasoning_parser(thinking)
    content, reasoning = [], []
    for chunk in chunks:
        result = parser.parse_delta(chunk)
        content.append(result.content)
        reasoning.append(result.reasoning_content)
    result = parser.finish()
    content.append(result.content)
    reasoning.append(result.reasoning_content)
    return "".join(content), "".join(reasoning)


def _pipeline_stream(chunks: List[str], thinking: bool):
    """The streaming path: reasoning deltas feed the tool parser as they are produced."""
    reasoning_parser = _reasoning_parser(thinking)
    tool_parser = KimiK3ToolParser()
    reasoning, text, calls = [], [], []

    def feed(content: Optional[str]) -> None:
        result = tool_parser.parse_streaming_increment(content or "", TOOLS)
        text.append(result.normal_text)
        calls.extend(result.calls)

    for chunk in chunks:
        result = reasoning_parser.parse_delta(chunk)
        reasoning.append(result.reasoning_content)
        feed(result.content)
    result = reasoning_parser.finish()
    reasoning.append(result.reasoning_content)
    feed(result.content)
    result = tool_parser.finish(TOOLS)
    text.append(result.normal_text)
    calls.extend(result.calls)
    return "".join(reasoning), "".join(text), _calls(calls)


def _pipeline_whole(generation: str, thinking: bool):
    """The snapshot path: one-shot reasoning parse, then a whole-text tool parse."""
    parsed = _reasoning_parser(thinking).parse(generation)
    result = KimiK3ToolParser().detect_and_parse(parsed.content, TOOLS)
    return parsed.reasoning_content, result.normal_text, _calls(result.calls)


def _pipeline_events(chunks: List[str], thinking: bool) -> list:
    """The streaming path as a client sees it: ordered reasoning runs, text runs and calls.

    A parser result does not say how its parts interleave; the serving layer places reasoning
    before content and text before calls, so each result must read in that order. Both
    parsers defer whatever would break it (reasoning after content, text after calls) to their
    next increment, and an order-preserving consumer drains that with empty increments before
    ``finish``, as ``responses_utils._flush_tool_parser`` does for the tool parser.
    """
    reasoning_parser = _reasoning_parser(thinking)
    tool_parser = KimiK3ToolParser()
    events: list = []

    def add(kind: str, value) -> None:
        if kind != "call" and not value:
            return
        if kind != "call" and events and events[-1][0] == kind:
            events[-1] = (kind, events[-1][1] + value)
        else:
            events.append((kind, value))

    def feed_tool(result) -> None:
        add("text", result.normal_text)
        for call in result.calls:
            add("call", (call.name, call.parameters))

    def feed_reasoning(result) -> bool:
        add("reasoning", result.reasoning_content)
        feed_tool(tool_parser.parse_streaming_increment(result.content, TOOLS))
        return bool(result.content or result.reasoning_content)

    for chunk in chunks:
        feed_reasoning(reasoning_parser.parse_delta(chunk))
    while feed_reasoning(reasoning_parser.parse_delta("")):
        pass
    feed_reasoning(reasoning_parser.finish())
    while tool_parser.has_tool_call(tool_parser._buffer):
        held = len(tool_parser._buffer)
        feed_tool(tool_parser.parse_streaming_increment("", TOOLS))
        if len(tool_parser._buffer) >= held:
            break
    feed_tool(tool_parser.finish(TOOLS))
    return events


@pytest.mark.parametrize("name", list(CORPUS))
def test_pipeline_event_order_is_chunking_invariant(name):
    """Each result reads in generation order, so the client's item sequence is stable.

    Where a message or reasoning item ends is the same for every chunking, not only the
    concatenated text.
    """
    generation, thinking, *_ = CORPUS[name]
    whole = _pipeline_events([generation], thinking)
    for label, chunks in _splits(generation, seed=3 * len(generation)):
        assert _pipeline_events(chunks, thinking) == whole, f"{name} @ {label}"


@pytest.mark.parametrize(
    ("name", "events"),
    [
        (
            "prose_after_tools",
            [
                ("reasoning", "t"),
                ("text", "Before."),
                ("call", ("exec", '{"input": "pwd"}')),
                ("text", "After the call."),
            ],
        ),
        (
            "second_tools_section",
            [
                ("reasoning", "t"),
                ("call", ("exec", '{"input": "ls"}')),
                ("reasoning", "more thinking"),
                ("call", ("exec", '{"input": "pwd"}')),
            ],
        ),
    ],
)
def test_pipeline_events_follow_the_generation(name, events):
    generation, thinking, *_ = CORPUS[name]
    assert _pipeline_events([generation], thinking) == events


@pytest.mark.parametrize("name", list(CORPUS))
def test_reading(name):
    """The snapshot reading of each generation: nothing dropped, calls numbered in order."""
    generation, thinking, reasoning, text, calls = CORPUS[name]

    got_reasoning, got_text, got_calls = _pipeline_whole(generation, thinking)

    assert got_reasoning == reasoning
    assert got_text == text
    assert [(name, params) for _, name, params in got_calls] == calls
    # Indices number the calls across the whole generation, sections included: the streaming
    # assembly keys call fragments by tool_index, so a repeat merges two calls into one.
    assert [index for index, _, _ in got_calls] == list(range(len(calls)))
    for _, _, params in got_calls:
        if params != '{"plan": oops}':  # the one sample whose model JSON is invalid
            json.loads(params)


@pytest.mark.parametrize("name", list(CORPUS))
def test_tool_parser_stream_matches_detect_and_parse(name):
    """Every chunking of the tool parser's input reproduces ``detect_and_parse``.

    Checked on the content the reasoning parser hands over (the serving chain) and on the raw
    generation (a tool parser deployed without the reasoning parser).
    """
    generation, thinking, *_ = CORPUS[name]
    content = _reasoning_parser(thinking).parse(generation).content
    for source in (content, generation):
        whole = _tool_whole(source)
        for label, chunks in _splits(source, seed=len(source)):
            assert _tool_stream(chunks) == whole, f"{name} @ {label}"


@pytest.mark.parametrize("name", list(CORPUS))
def test_reasoning_parser_stream_matches_parse(name):
    generation, thinking, *_ = CORPUS[name]
    parsed = _reasoning_parser(thinking).parse(generation)
    whole = (parsed.content, parsed.reasoning_content)
    for label, chunks in _splits(generation, seed=len(generation)):
        assert _reasoning_stream(chunks, thinking) == whole, f"{name} @ {label}"


@pytest.mark.parametrize("name", list(CORPUS))
def test_pipeline_stream_matches_snapshot(name):
    """The chained stream, chunked every way, equals the chained snapshot."""
    generation, thinking, *_ = CORPUS[name]
    whole = _pipeline_whole(generation, thinking)
    for label, chunks in _splits(generation, seed=7 * len(generation)):
        assert _pipeline_stream(chunks, thinking) == whole, f"{name} @ {label}"


# Fragments the fuzzed generations are assembled from: every tag of the grammar, broken and
# attribute-less variants, and text that collides with marker prefixes.
_FUZZ_PIECES = [
    OPEN,
    CLOSE,
    SEP,
    EOM,
    f"{OPEN}think{SEP}",
    f"{CLOSE}think{SEP}",
    f"{OPEN}response{SEP}",
    f"{CLOSE}response{SEP}",
    BOT,
    EOT,
    f'{OPEN}call tool="exec" index="1"{SEP}',
    f'{OPEN}call index="2"{SEP}',
    f'{OPEN}argument key="input" type="string"{SEP}',
    f'{OPEN}argument key="n" type="number"{SEP}',
    f'{OPEN}argument type="string"{SEP}',
    f'{OPEN}json type="object"{SEP}',
    f"{CLOSE}argument{SEP}",
    f"{CLOSE}json{SEP}",
    f"{CLOSE}call{SEP}",
    MESSAGE_END,
    "tools",
    "x",
    "ls -la",
    " ",
    "\n",
    "1e3",
    '{"a": 1}',
    "<",
    "<|",
    ">",
    "&quot;",
]
_FUZZ_RNG = random.Random(2026)
FUZZED = [
    (
        "".join(_FUZZ_RNG.choice(_FUZZ_PIECES) for _ in range(_FUZZ_RNG.randint(1, 24))),
        _FUZZ_RNG.random() < 0.8,
    )
    for _ in range(40)
]


@pytest.mark.parametrize("case", range(len(FUZZED)))
def test_fuzzed_generations_read_the_same_under_every_chunking(case):
    """The corpus properties, on random tag soups the corpus does not anticipate."""
    generation, thinking = FUZZED[case]
    parsed = _reasoning_parser(thinking).parse(generation)
    reasoning_whole = (parsed.content, parsed.reasoning_content)
    pipeline_whole = _pipeline_whole(generation, thinking)
    events_whole = _pipeline_events([generation], thinking)
    for label, chunks in _splits(generation, seed=case):
        context = f"{generation!r} @ {label}"
        assert _reasoning_stream(chunks, thinking) == reasoning_whole, context
        assert _pipeline_stream(chunks, thinking) == pipeline_whole, context
        assert _pipeline_events(chunks, thinking) == events_whole, context
    for source in (parsed.content, generation):
        whole = _tool_whole(source)
        for label, chunks in _splits(source, seed=case):
            assert _tool_stream(chunks) == whole, f"{source!r} @ {label}"


# ---------------------------------------------------------------------------
# Tool parser, one defect at a time
# ---------------------------------------------------------------------------


def test_text_after_section_is_kept_by_both_views():
    """Defect 1: ``finish`` released post-section text that ``detect_and_parse`` dropped."""
    content = "Before." + _section(_exec(1, "pwd")) + "After the call."

    assert _tool_whole(content) == ("Before.After the call.", [(0, "exec", '{"input": "pwd"}')])
    assert _tool_stream([content]) == _tool_whole(content)


def test_calls_are_numbered_across_sections():
    """Defect 2: both sections numbered from 0, so the stream assembly merged the calls."""
    content = _section(_exec(1, "ls")) + _section(_exec(1, "pwd"))
    expected = ("", [(0, "exec", '{"input": "ls"}'), (1, "exec", '{"input": "pwd"}')])

    assert _tool_whole(content) == expected
    assert _tool_stream([content]) == expected
    assert _tool_stream(list(content)) == expected


@pytest.mark.parametrize("held", [BOT[:k] for k in range(1, len(BOT))])
def test_every_partial_section_opener_is_held(held):
    """Defect 3: the base class held the *shortest* matching suffix.

    ``Hi<|open|>tools<`` ends with ``<`` (a one-character opener prefix) and with
    ``<|open|>tools<`` (a fourteen-character one); holding only the first released
    ``<|open|>tools`` as text before the opener was complete.
    """
    parser = KimiK3ToolParser()

    result = parser.parse_streaming_increment("Hi" + held, TOOLS)

    assert result.normal_text == "Hi"
    assert parser._buffer == held
    assert parser.finish(TOOLS).normal_text == held


def test_truncated_call_is_released_as_text():
    """Defect 4: the cut-off call and its markup vanished from both views."""
    truncated = (
        BOT
        + f'{OPEN}call tool="exec" index="1"{SEP}{OPEN}argument key="input" type="string"{SEP}ls -l'
    )
    content = "Running:" + truncated

    assert _tool_whole(content) == (content, [])
    assert _tool_stream(list(content)) == (content, [])


def test_truncated_second_call_is_released_complete_first_call_delivered():
    """Defect 4: complete calls are salvaged; the cut-off one is released as text."""
    tail = f'{OPEN}call tool="exec" index="2"{SEP}{OPEN}argument key="input" type="str'
    content = BOT + _exec(1, "ls") + tail

    assert _tool_whole(content) == (tail, [(0, "exec", '{"input": "ls"}')])


def test_single_call_stop_is_the_call_close_tag():
    """parallel_tool_calls=false stops generation on this text, kept in the output."""
    assert KimiK3ToolParser().single_call_stop == f"{CLOSE}call{SEP}"


def test_generation_stopped_after_its_first_call_delivers_that_call_quietly():
    """The single-call stop leaves the tools section open, and that is not an error.

    The kept stop text completes the call, so both views deliver it and nothing
    else. Every parallel_tool_calls=false turn with a call ends this way, so the
    open section is not reported as a warning.
    """
    content = "Running it:" + BOT + _exec(1, "ls -la")
    assert content.endswith(KimiK3ToolParser().single_call_stop)
    expected = ("Running it:", [(0, "exec", _exec_args("ls -la"))])

    with patch("tensorrt_llm.serve.tool_parser.kimi_k3_tool_parser.logger") as log:
        assert _tool_whole(content) == expected
        for label, chunks in _splits(content, seed=11):
            assert _tool_stream(chunks) == expected, label
    log.warning.assert_not_called()


def test_generation_stopped_after_its_first_call_reads_the_same_through_the_pipeline():
    """With the reasoning parser in front, as served: reasoning, then the one call."""
    generation = _think("Plan the listing.") + BOT + _exec(1, "ls -la")
    expected = ("Plan the listing.", "", [(0, "exec", _exec_args("ls -la"))])

    assert _pipeline_whole(generation, thinking=True) == expected
    for label, chunks in _splits(generation, seed=12):
        assert _pipeline_stream(chunks, thinking=True) == expected, label


def test_a_section_left_open_any_other_way_still_warns():
    """Only a section that ends on a complete call is quiet."""
    content = BOT + _exec(1, "ls") + "and then I"
    with patch("tensorrt_llm.serve.tool_parser.kimi_k3_tool_parser.logger") as log:
        text, calls = _tool_whole(content)

    assert calls == [(0, "exec", _exec_args("ls"))]
    assert text == "and then I"
    warnings = [str(call.args[0]) for call in log.warning.call_args_list]
    assert any("never closed" in message for message in warnings)


def test_literal_argument_close_inside_value_does_not_cut_the_value():
    """Defect 5: a value ended at the first ``<|close|>argument<|sep|>`` it contained.

    The truncated command was delivered as runnable.
    """
    command = f"grep -n '{CLOSE}argument{SEP}' parser.py"
    content = _section(_exec(1, command))

    assert _tool_whole(content) == ("", [(0, "exec", _exec_args(command))])


def test_typed_values_keep_the_model_json_text():
    """Defect 5: typed values were re-serialized (``1e3`` became ``1000.0``)."""
    content = _section(
        _call(
            "functions.update_plan",
            1,
            _argument("n", "number", "1e3") + _argument("f", "number", "-0.50"),
        )
    )

    _, calls = _tool_whole(content)

    assert calls == [(0, "functions.update_plan", '{"n": 1e3, "f": -0.50}')]


@pytest.mark.parametrize("body", ["NaN", "Infinity", "-Infinity", "1e3 x", "[1,"])
def test_typed_value_that_is_not_json_is_delivered_as_a_string(body):
    """Arguments must stay JSON: Python's json accepts ``NaN``/``Infinity``, JSON does not."""
    content = _section(_call("functions.update_plan", 1, _argument("n", "number", body)))

    _, calls = _tool_whole(content)

    assert calls == [(0, "functions.update_plan", json.dumps({"n": body}))]


def test_json_block_keeps_the_model_json_text():
    """Defect 5: a valid json-block body passes through verbatim, not re-serialized."""
    body = '{"plan": "x",  "n": 1e3}'
    content = _section(
        _call("functions.update_plan", 1, f'{OPEN}json type="object"{SEP}{body}{CLOSE}json{SEP}')
    )

    assert _tool_whole(content) == ("", [(0, "functions.update_plan", body)])


def test_trailing_framing_is_stripped_by_both_views():
    """Message framing that reaches the tool parser is stripped from both views.

    Without the reasoning parser in front, ``<|close|>message<|sep|>`` reaches the tool parser;
    it is framing only where it ends the generation, and the stream must not emit it either.
    """
    text = f"The answer is 4.{MESSAGE_END}\n{EOM}"
    for _, chunks in _splits(text, seed=1):
        assert _tool_stream(chunks) == ("The answer is 4.", [])


def test_framing_followed_by_text_is_literal():
    text = f"Messages end with {MESSAGE_END} and {EOM} tokens."

    assert _tool_whole(text) == (text, [])
    assert _tool_stream(list(text)) == (text, [])


# ---------------------------------------------------------------------------
# Reasoning parser, one defect at a time
# ---------------------------------------------------------------------------


def test_text_after_think_without_response_tag_is_content():
    """Defect 6: text between ``<|close|>think<|sep|>`` and the message close was dropped."""
    result = _reasoning_parser(True).parse(_think("t") + "Answer." + MESSAGE_END + EOM)

    assert (result.reasoning_content, result.content) == ("t", "Answer.")


def test_reasoning_inside_a_tools_section_is_reasoning():
    """Defect 6: a think block between two calls reached neither view."""
    section = _section(
        _exec(1, "ls"), f"{OPEN}think{SEP}reconsidering{CLOSE}think{SEP}", _exec(2, "pwd")
    )

    result = _reasoning_parser(True).parse(_think("t") + _response("") + section + MESSAGE_END)

    assert result.reasoning_content == "treconsidering"
    assert result.content == _section(_exec(1, "ls"), _exec(2, "pwd"))


@pytest.mark.parametrize(
    "prose",
    [
        f"The stop token is {EOM} and after it comes more text.",
        f"A literal {OPEN} marker, then more prose that matters.",
        f"An unknown tag {OPEN}note{SEP} is just text.",
        f"Messages end with {MESSAGE_END} and then more.",
        f"Sections look like {BOT}...{EOT} in prose.",
    ],
)
def test_literal_markup_in_prose_is_content(prose):
    """Defect 6: literal markup in prose was read as structure and the rest was dropped."""
    result = _reasoning_parser(True).parse(_think("t") + _response(prose) + MESSAGE_END + EOM)

    assert (result.reasoning_content, result.content) == ("t", prose)


def test_whitespace_between_tags_is_formatting_but_channel_text_is_verbatim():
    generation = _think("t") + "\n" + _response("\n Hi.\n") + "\n" + MESSAGE_END + "\n" + EOM

    result = _reasoning_parser(True).parse(generation)

    assert (result.reasoning_content, result.content) == ("t", "\n Hi.\n")
