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
"""The raw dump records what the engine produced, not only what it detokenized to.

Token ids accumulated across chunks, the engine's finish reason, the build
that produced the text, and the cheap scans a reader would otherwise run over
terabytes later -- all beside the text, none of it changing what is served.
"""

import glob
import json
import os

import pytest

from tensorrt_llm.executor.postprocessor_hook import (
    PostProcessorHookChunk,
    apply_post_processor_hook,
    emit,
)
from tensorrt_llm.executor.raw_output_hook import (
    BUILD_ID_ENV,
    RAW_OUTPUT_DIR_ENV,
    RawOutputDump,
    text_flags,
)

pytestmark = pytest.mark.cpu_only


class FakeOutput:
    """``CompletionOutput`` as the applier sees it, with both watermarks."""

    def __init__(self, index=0):
        self.index = index
        self.text = ""
        self._last_text_len = 0
        self.token_ids = []
        self._last_token_ids_len = 0
        self.logprobs = None
        self.finish_reason = None
        self.stop_reason = None

    @property
    def text_diff(self):
        return self.text[self._last_text_len :]

    @property
    def token_ids_diff(self):
        return self.token_ids[self._last_token_ids_len :]

    def append(self, text, tokens):
        """Advance both watermarks the way ``_handle_sequence`` does, then add."""
        self._last_text_len = len(self.text)
        self._last_token_ids_len = len(self.token_ids)
        self.text += text
        self.token_ids = self.token_ids + list(tokens)


class FakeResult:
    def __init__(self, request_id=7, outputs=None):
        self.id = request_id
        self.outputs = outputs if outputs is not None else [FakeOutput()]
        self._done = False
        self._aborted = False


def drive(hook, result, chunks, streaming=True, finish_reason="stop"):
    """``chunks`` is a list of (text_delta, token_ids_delta)."""
    for i, (delta, tokens) in enumerate(chunks):
        final = i == len(chunks) - 1
        for output in result.outputs:
            output.append(delta, tokens)
            if final:
                output.finish_reason = finish_reason
        result._done = final
        apply_post_processor_hook(hook, result, streaming=streaming)


def records_in(directory):
    lines = []
    for path in sorted(glob.glob(os.path.join(directory, "*", "raw-*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            lines += [json.loads(line) for line in handle if line.strip()]
    return lines


@pytest.fixture
def dump_dir(tmp_path, monkeypatch):
    directory = tmp_path / "raw"
    monkeypatch.setenv(RAW_OUTPUT_DIR_ENV, str(directory))
    monkeypatch.delenv(BUILD_ID_ENV, raising=False)
    return directory


def test_token_ids_accumulate_across_chunks_and_finish_reason_is_kept(dump_dir):
    hook = RawOutputDump()
    result = FakeResult()
    drive(hook, result, [("<think>", [1, 2]), ("ok</think>", [3]), ("done", [4, 5, 6])], finish_reason="length")

    (record,) = records_in(dump_dir)
    assert record["token_ids"] == [1, 2, 3, 4, 5, 6]
    assert record["num_output_tokens"] == 6
    assert record["finish_reason"] == "length"
    assert record["text"] == "<think>ok</think>done"
    assert "build_id" not in record


def test_a_token_that_detokenizes_to_nothing_is_still_counted(dump_dir):
    # The first half of a multi-byte character: no text yet, one token.
    hook = RawOutputDump()
    result = FakeResult()
    drive(hook, result, [("", [11]), ("é", [12]), (".", [13])])

    (record,) = records_in(dump_dir)
    assert record["frames"] == ["é", "."]  # empty text frames are dropped, as before
    assert record["token_ids"] == [11, 12, 13]  # but their tokens are not


def test_a_non_streaming_request_records_its_tokens_once(dump_dir):
    hook = RawOutputDump()
    result = FakeResult()
    drive(hook, result, [("whole answer", [1, 2, 3])], streaming=False)

    (record,) = records_in(dump_dir)
    assert record["token_ids"] == [1, 2, 3]
    assert record["streaming"] is False


def test_the_hook_still_changes_nothing(dump_dir):
    hook = RawOutputDump()
    result = FakeResult()
    drive(hook, result, [("a", [1]), ("b", [2])])
    assert result.outputs[0].text == "ab"
    assert result.outputs[0].token_ids == [1, 2]


def test_build_id_is_stamped_when_configured(dump_dir, monkeypatch):
    monkeypatch.setenv(BUILD_ID_ENV, "7c1e9a0b2f44")
    hook = RawOutputDump()
    drive(hook, FakeResult(), [("x", [1])])
    (record,) = records_in(dump_dir)
    assert record["build_id"] == "7c1e9a0b2f44"


def test_a_partial_record_carries_what_it_had(dump_dir):
    hook = RawOutputDump()
    result = FakeResult(request_id=3)
    output = result.outputs[0]
    output.append("half", [1, 2])
    apply_post_processor_hook(hook, result, streaming=True)  # not final
    hook.on_request_end(3, "executor_error")

    (record,) = records_in(dump_dir)
    assert record["partial"] is True
    assert record["token_ids"] == [1, 2]
    assert record["finish_reason"] is None
    assert record["end_reason"] == "executor_error"


def test_ids_that_made_no_text_yet_are_still_written_on_failure(dump_dir):
    hook = RawOutputDump()
    result = FakeResult(request_id=4)
    result.outputs[0].append("", [11])  # the first half of a multi-byte character
    apply_post_processor_hook(hook, result, streaming=True)  # not final
    hook.on_request_end(4, "executor_error")

    (record,) = records_in(dump_dir)
    assert record["partial"] is True
    assert record["token_ids"] == [11]
    assert record["text"] == ""


def test_a_clean_short_text_adds_no_flags_key(dump_dir):
    hook = RawOutputDump()
    drive(hook, FakeResult(), [("Sure, ", [1]), ("done.", [2])])
    (record,) = records_in(dump_dir)
    assert "flags" not in record


def test_unserializable_ids_cost_the_ids_and_not_the_record(dump_dir):
    hook = RawOutputDump()
    drive(hook, FakeResult(), [("text", [1, object()])])
    (record,) = records_in(dump_dir)
    assert record["text"] == "text"
    assert record["token_ids"] is None
    assert record["token_ids_error"].startswith("TypeError")


def test_the_chunk_carries_the_output_finish_reason():
    seen = []

    def hook(chunk: PostProcessorHookChunk):
        seen.append(chunk.finish_reason)
        return emit(chunk.text_diff)

    result = FakeResult()
    drive(hook, result, [("a", [1]), ("b", [2])], finish_reason="timeout")
    assert seen == [None, "timeout"]


class TestTextFlags:
    def test_marker_literals_are_counted(self):
        flags = text_flags("x<|im_end|>y<|im_end|> and <|open|>call<|sep|>")
        assert flags["marker_literals"] == {"<|im_end|>": 2, "<|open|>": 1, "<|sep|>": 1}

    def test_no_markers_no_key(self):
        assert "marker_literals" not in text_flags("plain text < | not a marker | >")

    def test_short_text_has_no_repetition_measure(self):
        assert "repetition" not in text_flags("Great. " * 10)

    def test_a_loop_shows_as_a_repeated_suffix(self):
        # A 19-word sentence repeated: the 20-word suffix straddles a sentence
        # boundary, so consecutive matches overlap by a word and str.count
        # (non-overlapping) sees about one match per two sentences. Still an
        # order of magnitude above the honest answer's 1.
        sentence = "I got stuck in a malformed analysis loop and I will now restart the analysis from the top again. "
        text = sentence * 60
        flags = text_flags(text)
        assert flags["repetition"]["suffix_words"] == 20
        assert flags["repetition"]["suffix_repeats"] >= 20

    def test_an_honest_long_answer_repeats_its_suffix_once(self):
        text = " ".join(f"token{i}" for i in range(1000))
        assert text_flags(text)["repetition"]["suffix_repeats"] == 1

    def test_identical_line_runs(self):
        text = "\n".join(["Submitting."] * 40) + "\n" + "x" * 2000
        assert text_flags(text)["repetition"]["max_identical_line_run"] == 40

    def test_dots_only_reasoning_glm_and_k3(self):
        assert text_flags("<think>.......</think>answer")["reasoning_dots_only"] is True
        assert text_flags("<|open|>think<|sep|> . . . <|close|>think<|sep|>answer")["reasoning_dots_only"] is True
        assert "reasoning_dots_only" not in text_flags("<think>real thought</think>answer")
        assert "reasoning_dots_only" not in text_flags("<think></think>answer")

    def test_an_unclosed_think_block_is_no_reasoning_block(self):
        text = "<think>" * 500 + "x" * 100  # the shape of a length-capped loop
        assert "reasoning_dots_only" not in text_flags(text)

    def test_only_the_first_closed_block_is_judged(self):
        assert text_flags("<think>...</think><think>real</think>")["reasoning_dots_only"] is True
        assert "reasoning_dots_only" not in text_flags("<think>real</think><think>...</think>")

    def test_flags_land_on_the_record(self, dump_dir):
        hook = RawOutputDump()
        drive(hook, FakeResult(), [("<think>...</think>", [1]), ("<|im_end|>", [2])])
        (record,) = records_in(dump_dir)
        assert record["flags"]["reasoning_dots_only"] is True
        assert record["flags"]["marker_literals"] == {"<|im_end|>": 1}

    def test_empty_text_has_no_flags(self, dump_dir):
        assert text_flags("") == {}
