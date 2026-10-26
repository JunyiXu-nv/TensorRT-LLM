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
"""Properties of two validator rules, over generated inputs (CPU only, needs `hypothesis`).

- A re-ask (C-03) is a later request that appends blocks to the last user message and changes
  nothing else. Every such request is a re-ask, and no other change is.
- A declared error wording (C-16) matches a message exactly when its parts, split at "...",
  appear in it in order, in any case.

Run with pytest, or as a script, which is how the `trtllm-verify-logic` mutation gate runs it:

    python3 <trtllm-verify-logic>/scripts/mutate.py tests/validator_properties.mutations \
        --python <validator venv python> --runner <abs path of this file> --runner-args ""
"""

import copy
import os
import sys

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SKILL, "scripts", "validate"))

import pipeline  # noqa: E402

SETTINGS = settings(max_examples=300, deadline=None)
TEXT = st.text(alphabet="abcxyz .:'", max_size=12)
BLOCK = st.one_of(
    st.builds(lambda t: {"type": "text", "text": t}, TEXT),
    st.builds(lambda i, t: {"type": "tool_result", "tool_use_id": i, "content": t},
              st.sampled_from(["toolu_1", "toolu_2"]), TEXT),
)  # fmt: skip
OTHER = st.one_of(
    st.builds(
        lambda b: {"role": "assistant", "content": b}, st.lists(BLOCK, min_size=1, max_size=2)
    ),
    st.builds(lambda t: {"role": "system", "content": t}, TEXT),
)


@st.composite
def conversation(draw):
    """A messages body whose last user message carries a list of blocks."""
    before = draw(st.lists(st.one_of(OTHER, st.builds(lambda t: {"role": "user", "content": t}, TEXT)),
                           max_size=4))  # fmt: skip
    user = {"role": "user", "content": draw(st.lists(BLOCK, min_size=1, max_size=3))}
    after = draw(st.lists(OTHER, max_size=2))  # e.g. Claude Code's mid-conversation system message
    return {"model": "m", "max_tokens": 64, "stream": True, "messages": before + [user] + after}


def _last_user(body):
    return max(i for i, m in enumerate(body["messages"]) if m["role"] == "user")


def _reasked(first, later, later_first=False):
    """Whether `later`, answered, re-asks `first` (orders: 1 then 2, or swapped)."""
    a, b = pipeline._reask_key(first), pipeline._reask_key(later)
    order_a, order_b = (2, 1) if later_first else (1, 2)
    turn = {"thread": "th", "order": order_a, "reask": a}
    reasks = {("th", b[0]): [(order_b, tuple(b[1]))]}
    return pipeline.Pipeline._reasked(turn, reasks)


@SETTINGS
@given(conversation(), st.lists(BLOCK, min_size=1, max_size=2))
def test_appending_to_the_last_user_message_is_a_reask(body, appended):
    again = copy.deepcopy(body)
    again["messages"][_last_user(body)]["content"] += appended
    assert _reasked(body, again)


@SETTINGS
@given(conversation(), st.lists(BLOCK, min_size=1, max_size=2))
def test_an_earlier_request_does_not_reask_a_later_one(body, appended):
    again = copy.deepcopy(body)
    again["messages"][_last_user(body)]["content"] += appended
    assert not _reasked(body, again, later_first=True)


@SETTINGS
@given(conversation())
def test_the_same_request_is_not_a_reask(body):
    assert not _reasked(body, copy.deepcopy(body))


@SETTINGS
@given(conversation(), st.data())
def test_changing_any_other_message_is_not_a_reask(body, data):
    last = _last_user(body)
    others = [i for i in range(len(body["messages"])) if i != last]
    assume(others)
    i = data.draw(st.sampled_from(others))
    again = copy.deepcopy(body)
    again["messages"][i] = data.draw(OTHER.filter(lambda m: m != body["messages"][i]))
    again["messages"][last]["content"] += [{"type": "text", "text": "Continue."}]
    assert not _reasked(body, again)


@SETTINGS
@given(conversation(), st.sampled_from([("model", "other"), ("max_tokens", 32), ("stream", False),
                                        ("thinking", {"type": "adaptive"})]))  # fmt: skip
def test_changing_a_request_field_is_not_a_reask(body, change):
    again = copy.deepcopy(body)
    again[change[0]] = change[1]
    again["messages"][_last_user(body)]["content"] += [{"type": "text", "text": "Continue."}]
    assert not _reasked(body, again)


@SETTINGS
@given(conversation(), BLOCK, st.data())
def test_rewriting_the_last_user_message_is_not_a_reask(body, replacement, data):
    last = _last_user(body)
    blocks = body["messages"][last]["content"]
    j = data.draw(st.integers(0, len(blocks) - 1))
    assume(replacement != blocks[j])
    again = copy.deepcopy(body)
    again["messages"][last]["content"][j] = replacement
    again["messages"][last]["content"] += [{"type": "text", "text": "Continue."}]
    assert not _reasked(body, again)


PART = st.text(alphabet="abcdefgh'_:", min_size=1, max_size=6).filter(lambda p: p.strip() == p)


@SETTINGS
@given(
    st.lists(PART, min_size=1, max_size=3), st.lists(TEXT, min_size=4, max_size=4), st.booleans()
)
def test_a_message_holding_the_parts_in_order_matches(parts, fillers, upper):
    message = fillers[0] + "".join(p + f for p, f in zip(parts, fillers[1:] + [""] * 3))
    message = message.upper() if upper else message
    assert pipeline.wording_matches(" ... ".join(parts), message)


@SETTINGS
@given(st.lists(PART, min_size=1, max_size=3), st.data())
def test_a_message_missing_a_part_does_not_match(parts, data):
    k = data.draw(st.integers(0, len(parts) - 1))
    message = " ".join(p for i, p in enumerate(parts) if i != k)
    assume(parts[k].lower() not in message.lower())
    assert not pipeline.wording_matches(" ... ".join(parts), message)


@SETTINGS
@given(PART, PART)
def test_parts_out_of_order_do_not_match(first, second):
    assume(first.lower() not in second.lower() and second.lower() not in first.lower())
    assert not pipeline.wording_matches(f"{first} ... {second}", f"{second} {first}")


@SETTINGS
@given(TEXT)
def test_a_wording_without_parts_matches_nothing(message):
    assert not pipeline.wording_matches(" ... ", message)


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q", "-p", "no:cacheprovider"]))
