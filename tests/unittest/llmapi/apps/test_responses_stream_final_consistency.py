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
"""A streamed Responses turn and its final snapshot must agree.

The terminal event's snapshot is built after the stream, so these tests drive
the real streaming processor and compare what it streamed with what its final
event reports.
"""

import json
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai.types.responses.tool import FunctionTool

from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingProcessor,
    StreamedItem,
    _create_output_content,
    _parse_output_message_harmony,
    _whole_text_tool_calls,
    add_stop_kept_in_output,
    create_response_non_store,
    first_tool_call_stop,
)

pytestmark = pytest.mark.cpu_only


def _event_type(frame):
    return frame.split("\n", 1)[0][len("event: ") :]


def _event_data(frame):
    return json.loads(frame.split("data: ", 1)[1])


def _tool(name="exec", properties=("input",)):
    return FunctionTool(
        name=name,
        type="function",
        strict=False,
        parameters={"type": "object", "properties": {p: {"type": "string"} for p in properties}},
    )


def _processor(tools=(), reasoning_parser="glm", tool_parser=None, tool_choice="auto"):
    """A processor built as the server builds one."""
    request = ResponsesRequest(
        model="test-model", input="hi", stream=True, tools=list(tools), tool_choice=tool_choice
    )
    return ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="test-model",
        use_harmony=False,
        reasoning_parser=reasoning_parser,
        tool_parser=tool_parser,
    )


def _generation_chunk(accumulated, diff, done, finish_reason="stop"):
    """One GenerationResult as the streaming loop sees it."""
    output = SimpleNamespace(
        index=0,
        text=accumulated,
        text_diff=diff,
        finish_reason=finish_reason if done else None,
        token_ids=[1, 2, 3],
        disaggregated_params=None,
    )
    return SimpleNamespace(outputs=[output], _done=done, prompt_token_ids=[1, 2], cached_tokens=0)


def _stream(processor, chunks):
    """Drive the processor; returns its frames and the finished result."""
    frames, accumulated, result = [], "", None
    for i, chunk in enumerate(chunks):
        accumulated += chunk
        result = _generation_chunk(accumulated, chunk, i == len(chunks) - 1)
        frames.extend(processor.process_single_output(result))
    return frames, result


def _done_items(frames, item_type=None):
    return [
        data["item"]
        for data in map(_event_data, frames)
        if data.get("type") == "response.output_item.done"
        and item_type in (None, data["item"]["type"])
    ]


def _added_ids(frames):
    return [
        (data["item"]["type"], data["item"]["id"])
        for data in map(_event_data, frames)
        if data.get("type") == "response.output_item.added"
    ]


def _message_texts(items):
    return [item["content"][0]["text"] for item in items if item["type"] == "message"]


def _final_output(processor, result):
    frame = processor.get_final_response_non_store(result)
    assert _event_type(frame) == "response.completed"
    return _event_data(frame)["response"]["output"]


def _snapshot(text, streamed_item_ids, reasoning_parser=None, outputs=None):
    items, messages, _reasoning = _create_output_content(
        SimpleNamespace(outputs=outputs or [SimpleNamespace(index=0, text=text)]),
        reasoning_parser=reasoning_parser,
        streamed_item_ids=streamed_item_ids,
    )
    return items, messages


# ---------------------------------------------------------------------------
# Item ids and texts
# ---------------------------------------------------------------------------


def test_final_items_are_the_streamed_items():
    processor = _processor()
    frames, result = _stream(processor, ["Check the reference.", "</think>", " The answer. "])

    added = _added_ids(frames)
    assert [item_type for item_type, _ in added] == ["reasoning", "message"]
    assert [(i["type"], i["id"]) for i in _done_items(frames)] == added

    final = _final_output(processor, result)
    assert [(item["type"], item["id"]) for item in final] == added
    # Edge whitespace is part of the generation.
    assert _message_texts(final) == [" The answer. "]


def test_streamed_segmentation_is_repeated_verbatim():
    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        items, messages = _snapshot(
            "the answer",
            [
                StreamedItem("message", "msg_first", "the"),
                StreamedItem("message", "msg_2", " answer"),
            ],
        )
    assert [(i.id, i.content[0].text) for i in items] == [
        ("msg_first", "the"),
        ("msg_2", " answer"),
    ]
    assert messages == [{"role": "assistant", "content": "the answer"}]
    assert not mock_logger.warning.called


def test_ids_are_reused_in_emission_order_when_the_record_does_not_cover():
    """Two outputs: streaming reads only outputs[0], so the record is not used."""
    items, _ = _snapshot(
        None,
        [StreamedItem("message", "msg_first"), StreamedItem("message", "msg_second")],
        outputs=[SimpleNamespace(index=0, text="first"), SimpleNamespace(index=1, text="second")],
    )
    assert [(i.id, i.content[0].text) for i in items] == [
        ("msg_first", "first"),
        ("msg_second", "second"),
    ]


def test_a_non_streamed_snapshot_mints_fresh_ids_and_keeps_whitespace():
    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        items, _ = _snapshot("think</think> Backbone ", None, reasoning_parser="glm")
    assert [i.id[:3] for i in items] == ["rs_", "msg"]
    assert items[1].content[0].text == " Backbone "
    assert not mock_logger.warning.called


@pytest.mark.parametrize(
    "streamed",
    [
        [],
        [StreamedItem("message", "msg_first", "answer"), StreamedItem("message", "msg_2", "?")],
    ],
)
def test_a_count_mismatch_is_logged(streamed):
    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        items, _ = _snapshot("answer", streamed)
    assert [i.content[0].text for i in items] == ["answer"]
    if streamed:
        assert items[0].id == "msg_first"
    assert mock_logger.warning.called


# ---------------------------------------------------------------------------
# Sequence numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("postprocess_worker", [False, True])
def test_sequence_numbers_run_contiguously_from_zero(postprocess_worker):
    """A postprocessing worker gets a pickled copy before the opening pair."""
    frontend = _processor()
    producer = pickle.loads(pickle.dumps(frontend)) if postprocess_worker else frontend

    frames = frontend.get_initial_responses()
    body, result = _stream(producer, ["Plan.", "</think>", "Done."])
    frames += body + [producer.get_final_response_non_store(result)]

    assert [_event_data(frame)["sequence_number"] for frame in frames] == list(range(len(frames)))


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------

_EXEC_CALL = "<tool_call>exec<arg_key>input</arg_key><arg_value>ls</arg_value></tool_call>"
_QWEN3_CALL = '<tool_call>\n{"name": "exec", "arguments": {"input": "ls"}}\n</tool_call>'


@pytest.mark.parametrize(
    "parsers, chunks",
    [
        (("glm47", "glm47"), ["Plan it.", "</think>Run it. ", _EXEC_CALL]),
        (("glm", "qwen3"), ["Plan it.", "</think>Run it.\n", _QWEN3_CALL]),
    ],
    ids=["incremental", "whole_text"],
)
def test_final_tool_calls_are_the_streamed_ones(parsers, chunks):
    processor = _processor(tools=[_tool()], reasoning_parser=parsers[0], tool_parser=parsers[1])
    frames, result = _stream(processor, chunks)
    streamed_calls = _done_items(frames, "function_call")
    assert [(c["name"], json.loads(c["arguments"])) for c in streamed_calls] == [
        ("exec", {"input": "ls"})
    ]

    final = _final_output(processor, result)
    assert [i for i in final if i["type"] == "function_call"] == streamed_calls
    assert [i["type"] for i in final] == ["reasoning", "message", "function_call"]
    assert _message_texts(final) == _message_texts(_done_items(frames, "message"))


def test_a_namespace_double_colon_call_is_the_declared_custom_tool_in_both_views():
    """DeepSeek-V4.1 writes "namespace::tool"; the client declared exec under a namespace."""
    functions = {
        "type": "namespace",
        "name": "functions",
        "description": "",
        "tools": [{"type": "custom", "name": "exec", "description": "Run code."}],
    }
    call = (
        "<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="functions::exec">\n'
        '<｜DSML｜ parameter name="input" string="true">ls -la</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>"
    )
    processor = _processor(tools=[functions], reasoning_parser=None, tool_parser="deepseek_v41")
    frames, result = _stream(processor, ["Running it.\n\n", call])
    streamed_calls = _done_items(frames, "custom_tool_call")
    assert [(c["namespace"], c["name"], c["input"]) for c in streamed_calls] == [
        ("functions", "exec", "ls -la")
    ]

    final = _final_output(processor, result)
    assert [i for i in final if i["type"] == "custom_tool_call"] == streamed_calls


def test_an_unterminated_call_is_text_in_both_views():
    unterminated = "<tool_call>exec<arg_key>input</arg_key><arg_value>const cmd = `cat m.json`"
    processor = _processor(tools=[_tool()], reasoning_parser="glm47", tool_parser="glm47")
    frames, result = _stream(processor, ["Plan it.", "</think>Run it.", unterminated])

    assert _done_items(frames, "function_call") == []
    streamed = _done_items(frames)
    assert "".join(_message_texts(streamed)) == "Run it." + unterminated

    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        final = _final_output(processor, result)
    assert [(i["type"], i["id"]) for i in final] == [(i["type"], i["id"]) for i in streamed]
    assert _message_texts(final) == _message_texts(streamed)
    assert not mock_logger.warning.called


def test_tool_choice_none_keeps_the_markup_as_text_in_both_views():
    processor = _processor(
        tools=[_tool()], reasoning_parser="glm47", tool_parser="glm47", tool_choice="none"
    )
    frames, result = _stream(processor, ["Plan it.", "</think>Calling: ", _EXEC_CALL])

    assert _message_texts(_done_items(frames)) == ["Calling: " + _EXEC_CALL]
    final = _final_output(processor, result)
    assert [item["type"] for item in final] == ["reasoning", "message"]
    assert _message_texts(final) == ["Calling: " + _EXEC_CALL]


# ---------------------------------------------------------------------------
# Terminal event and status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "finish_reason, event_type, status, details",
    [
        ("stop", "response.completed", "completed", None),
        ("length", "response.incomplete", "incomplete", {"reason": "max_output_tokens"}),
        ("timeout", "response.failed", "failed", None),
        ("cancelled", "response.completed", "cancelled", None),
    ],
)
def test_the_terminal_event_matches_the_status(finish_reason, event_type, status, details):
    processor = _processor(reasoning_parser=None)
    result = _generation_chunk("Partial answer", "Partial answer", True, finish_reason)
    frames = processor.process_single_output(result)
    assert all(_event_type(f) != "response.completed" for f in frames)

    frame = processor.get_final_response_non_store(result)
    payload = _event_data(frame)
    assert (_event_type(frame), payload["type"]) == (event_type, event_type)
    assert payload["response"]["status"] == status
    assert payload["response"]["incomplete_details"] == details
    assert _message_texts(payload["response"]["output"]) == ["Partial answer"]


@pytest.mark.parametrize(
    "finish_reason, status, details",
    [
        ("stop", "completed", None),
        ("length", "incomplete", {"reason": "max_output_tokens"}),
        # A disaggregated context worker handing off is not a token budget.
        ("not_finished", "incomplete", None),
    ],
)
def test_a_non_streamed_response_explains_an_incomplete_status(finish_reason, status, details):
    request = ResponsesRequest(model="test-model", input="hi", stream=False)
    response = create_response_non_store(
        generation_result=_generation_chunk("Partial", "Partial", True, finish_reason),
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="test-model",
        use_harmony=False,
        num_prompt_tokens=2,
    ).model_dump()
    assert (response["status"], response["incomplete_details"]) == (status, details)


@pytest.mark.parametrize(
    "streamed, whole, owed, warned",
    [
        ("Hello", "Hello world", " world", False),
        # The whole-text parse dropped the leading whitespace the stream sent.
        ("  Hello", "Hello world", " world", False),
        ("Hello ", "Hello", "", False),
        ("Hello", "Goodbye", "", True),
    ],
)
def test_the_stream_releases_what_the_whole_text_parse_reads_beyond_it(
    streamed, whole, owed, warned
):
    helper = SimpleNamespace(emitted_item_ids=[StreamedItem("message", "msg_1", streamed)])
    request = ResponsesRequest(model="test-model", input="hi", stream=True)
    with (
        patch(
            "tensorrt_llm.serve.responses_utils._apply_reasoning_parser",
            return_value=(whole, ""),
        ),
        patch(
            "tensorrt_llm.serve.responses_utils._apply_tool_parser",
            return_value=(whole, []),
        ),
        patch("tensorrt_llm.serve.responses_utils.logger") as logger,
    ):
        released, calls = _whole_text_tool_calls(
            SimpleNamespace(index=0, text=whole), request, helper, None, "qwen3", []
        )
    assert (released, calls, logger.warning.called) == (owed, [], warned)


# ---------------------------------------------------------------------------
# Item status: the snapshot reports what the stream closed each item with
# ---------------------------------------------------------------------------


def test_final_items_report_the_status_the_stream_closed_them_with():
    """Reasoning and function-call items stay `completed` in the snapshot.

    Their done events say `completed`; a snapshot that built them without a
    status read `null` for the same items, so a consumer that filters items by
    status reconstructed a different turn from each view.
    """
    processor = _processor(tools=[_tool()], reasoning_parser="glm47", tool_parser="glm47")
    frames, result = _stream(processor, ["Plan it.", "</think>Calling: ", _EXEC_CALL])

    streamed = {item["type"]: item.get("status") for item in _done_items(frames)}
    final = {item["type"]: item.get("status") for item in _final_output(processor, result)}

    assert streamed == {
        "reasoning": "completed",
        "message": "completed",
        "function_call": "completed",
    }
    assert final == streamed


def test_non_streaming_items_are_completed_too():
    """No stream ran; the body is the only account, and it says completed."""
    items, _messages, _reasoning = _create_output_content(
        SimpleNamespace(outputs=[SimpleNamespace(index=0, text="Plan it.</think>" + _EXEC_CALL)]),
        reasoning_parser="glm47",
        tool_parser="glm47",
        tools=[_tool()],
        streamed_item_ids=None,
    )
    assert [(item.type, item.status) for item in items] == [
        ("reasoning", "completed"),
        ("function_call", "completed"),
    ]


@pytest.mark.parametrize(
    "channel, recipient, item_type",
    [
        ("analysis", None, "reasoning"),
        ("commentary", "functions.get_weather", "function_call"),
        ("commentary", "python", "reasoning"),
    ],
    ids=["analysis", "function_call", "python_commentary"],
)
def test_harmony_reasoning_and_function_call_items_are_completed(channel, recipient, item_type):
    openai_harmony = pytest.importorskip("openai_harmony")
    message = openai_harmony.Message.from_role_and_content(
        openai_harmony.Role.ASSISTANT, '{"city": "Rome"}'
    ).with_channel(channel)
    if recipient is not None:
        message = message.with_recipient(recipient)

    (item,) = _parse_output_message_harmony(message)

    assert item.type == item_type
    assert item.status == "completed"


# ---------------------------------------------------------------------------
# parallel_tool_calls=false: generation ends after the first complete call
# ---------------------------------------------------------------------------


def _responses_request(**fields):
    return ResponsesRequest(model="test-model", input="hi", **fields)


def _calls(items):
    return [item for item in items if item["type"] == "function_call"]


def test_parallel_tool_calls_defaults_to_true_and_adds_no_stop():
    """OpenAI's default; neither an omitted field nor `true` limits the calls."""
    omitted = _responses_request(tools=[_tool()])
    assert omitted.parallel_tool_calls is True
    assert first_tool_call_stop("glm47", omitted) is None

    explicit = _responses_request(tools=[_tool()], parallel_tool_calls=True)
    assert first_tool_call_stop("glm47", explicit) is None


@pytest.mark.parametrize("tool_parser", ["glm47", "glm4"])
def test_parallel_tool_calls_false_stops_on_the_first_glm_call_close(tool_parser):
    """The stop is the text that completes one call, kept in the output.

    Without the kept text the tool parser would see an unterminated call.
    """
    request = _responses_request(tools=[_tool()], parallel_tool_calls=False)
    stop = first_tool_call_stop(tool_parser, request)
    assert stop == "</tool_call>"

    sampling_params = SimpleNamespace(stop=None, include_stop_str_in_output=False)
    add_stop_kept_in_output(sampling_params, stop)
    assert sampling_params.stop == ["</tool_call>"]
    assert sampling_params.include_stop_str_in_output is True


def test_parallel_tool_calls_false_stops_on_the_first_kimi_k3_call_close():
    """K3 calls sit in one tools section; each call block has its own close tag."""
    request = _responses_request(tools=[_tool()], parallel_tool_calls=False)
    assert first_tool_call_stop("kimi_k3", request) == "<|close|>call<|sep|>"


def test_a_kimi_k3_generation_stopped_after_the_first_call_reports_exactly_that_call():
    """What the K3 stop leaves for the serving layer: one call, in both views."""
    call = (
        '<|open|>call tool="exec" index="1"<|sep|>'
        '<|open|>argument key="input" type="string"<|sep|>ls<|close|>argument<|sep|>'
        "<|close|>call<|sep|>"
    )
    processor = _processor(tools=[_tool()], reasoning_parser="kimi_k3", tool_parser="kimi_k3")
    frames, result = _stream(
        processor, ["Plan it.", "<|close|>think<|sep|>", "<|open|>tools<|sep|>", call]
    )

    streamed = _done_items(frames, "function_call")
    assert [(c["name"], json.loads(c["arguments"])) for c in streamed] == [
        ("exec", {"input": "ls"})
    ]
    final_calls = _calls(_final_output(processor, result))
    assert [(c["call_id"], c["arguments"]) for c in final_calls] == [
        (c["call_id"], c["arguments"]) for c in streamed
    ]


def test_existing_stops_are_kept_and_the_call_stop_is_added_once():
    sampling_params = SimpleNamespace(stop="END", include_stop_str_in_output=False)
    add_stop_kept_in_output(sampling_params, "</tool_call>")
    add_stop_kept_in_output(sampling_params, "</tool_call>")
    assert sampling_params.stop == ["END", "</tool_call>"]


@pytest.mark.parametrize(
    "tools, tool_choice, tool_parser",
    [
        ([], "auto", "glm47"),
        ([_tool()], "none", "glm47"),
        ([_tool()], "auto", None),
        ([_tool()], "auto", "deepseek_v3"),
    ],
    ids=["no_tools", "tool_choice_none", "no_tool_parser", "format_without_a_single_call_end"],
)
def test_parallel_tool_calls_false_adds_no_stop_where_it_cannot_apply(
    tools, tool_choice, tool_parser
):
    """No tools, parsing disabled, or calls with no text of their own to stop on."""
    request = _responses_request(tools=tools, tool_choice=tool_choice, parallel_tool_calls=False)
    assert first_tool_call_stop(tool_parser, request) is None


def test_a_generation_stopped_after_the_first_call_reports_exactly_that_call():
    """The generation ends on the kept `</tool_call>` of its first call.

    Both the stream and the snapshot report that call once, with nothing after
    it, and the response completes normally.
    """
    processor = _processor(tools=[_tool()], reasoning_parser="glm47", tool_parser="glm47")
    frames, result = _stream(processor, ["Plan it.", "</think>", _EXEC_CALL])

    assert len(_done_items(frames, "function_call")) == 1

    final = _final_output(processor, result)
    calls = _calls(final)
    assert len(calls) == 1
    assert calls[0]["name"] == "exec"
    assert json.loads(calls[0]["arguments"]) == {"input": "ls"}
    assert all(item["type"] != "message" for item in final)


# ---------------------------------------------------------------------------
# Lone close tags and text around a call: both views publish the same messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    ["</tool_call>", "</tool_call> opens the reply.", "The reply ends on </tool_call>"],
)
def test_a_lone_close_tag_is_the_same_message_on_both_views(content):
    """Every split of the turn after the reasoning keeps the tag on both views."""
    turn = "</think>" + content
    for cut in range(len(turn) + 1):
        chunks = [chunk for chunk in ("Plan it.", turn[:cut], turn[cut:]) if chunk]
        processor = _processor(tools=[_tool()], reasoning_parser="glm47", tool_parser="glm47")
        frames, result = _stream(processor, chunks)
        final = _final_output(processor, result)

        assert _message_texts(_done_items(frames)) == [content], (
            f"the stream lost the tag at cut {cut}"
        )
        assert [(item["type"], item["id"]) for item in final] == _added_ids(frames)
        assert _message_texts(final) == [content], f"the snapshot disagreed at cut {cut}"
        assert not _calls(final)


def test_text_before_and_after_a_call_stays_two_messages_on_both_views():
    """Calls are held until generation ends, so the text on either side is two messages."""
    processor = _processor(tools=[_tool()], reasoning_parser="glm47", tool_parser="glm47")
    frames, result = _stream(
        processor,
        ["Plan it.", "</think>Checking first. ", _EXEC_CALL, " Then I will report back. "],
    )
    streamed = _added_ids(frames)
    assert [item_type for item_type, _ in streamed] == [
        "reasoning",
        "message",
        "message",
        "function_call",
    ]
    streamed_texts = _message_texts(_done_items(frames))
    assert streamed_texts == ["Checking first. ", " Then I will report back. "]

    final = _final_output(processor, result)

    assert [(item["type"], item["id"]) for item in final] == streamed
    assert _message_texts(final) == streamed_texts
    assert [(c["call_id"], c["name"], c["arguments"]) for c in _calls(final)] == [
        (c["call_id"], c["name"], c["arguments"]) for c in _done_items(frames, "function_call")
    ]
