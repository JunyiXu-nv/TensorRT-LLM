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
"""Properties of three validator rules, over generated inputs (CPU only, needs `hypothesis`).

- A re-ask (C-03) is a later request that appends blocks to the last user message and changes
  nothing else. Every such request is a re-ask, and no other change is.
- A declared error wording (C-16) matches a message exactly when its parts, split at "...",
  appear in it in order, in any case.
- A reply joins its raw record (C-13) by the first of its ids that names exactly one record.
  Failing that, an engine id that names several joins the one record not ruled out by evidence
  (times outside the reply's window, another output count), and only when that record's times
  are known; a disagg_request_id that names several joins nothing. The window holds every
  instant the recorded stamps can name.

Run with pytest, or as a script, which is how the `trtllm-verify-logic` mutation gate runs it:

    python3 <trtllm-verify-logic>/scripts/mutate.py tests/validator_properties.mutations \
        --python <validator venv python> --runner <abs path of this file> --runner-args ""
"""

import copy
import math
import os
import sys
from datetime import datetime, timedelta, timezone

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


# Raw join (C-13). Times are whole quarter seconds, so every bound is exact in binary.
SKEW = pipeline.RAW_CLOCK_SKEW_S
QUARTERS = st.integers(0, 480).map(lambda q: q / 4)
STAMP = st.one_of(st.none(), st.builds(lambda at, whole: (at, at + (1.0 if whole else 0.0)),
                                       QUARTERS, st.booleans()))  # fmt: skip
COUNT = st.one_of(st.none(), st.integers(0, 3))
READING = st.fixed_dictionaries({"started_at": STAMP, "finished_at": STAMP, "output_tokens": COUNT})
WINDOW = st.one_of(st.none(), st.builds(lambda at, length: (at, at + length), QUARTERS, QUARTERS))
MOMENT = st.datetimes(min_value=datetime(2026, 1, 1), max_value=datetime(2026, 12, 31),
                      timezones=st.just(timezone.utc))  # fmt: skip


def _ruled_out(reading, window, reported):
    """Evidence that a raw record is not this reply's generation.

    A known stamp wholly outside the window widened by the skew, or a known output count other
    than the reply's.
    """
    low, high = window[0] - SKEW, window[1] + SKEW
    if any(s is not None and (s[1] < low or s[0] > high)
           for s in (reading["started_at"], reading["finished_at"])):  # fmt: skip
        return True
    count = reading["output_tokens"]
    return reported is not None and count is not None and count != reported


@SETTINGS
@given(st.lists(READING, min_size=2, max_size=4), WINDOW, COUNT)
def test_a_resolved_record_fits_the_reply(readings, window, reported):
    joined = pipeline.fitting_raw(readings, window, reported)
    if joined is not None:
        assert window is not None and any(joined is r for r in readings)
        assert joined["started_at"] is not None and joined["finished_at"] is not None
        assert not _ruled_out(joined, window, reported)


@SETTINGS
@given(st.lists(READING, min_size=2, max_size=4), WINDOW, COUNT)
def test_a_resolved_record_is_the_only_one_not_ruled_out(readings, window, reported):
    joined = pipeline.fitting_raw(readings, window, reported)
    if joined is not None:
        assert all(_ruled_out(r, window, reported) for r in readings if r is not joined)


@st.composite
def one_left(draw):
    """Records under one engine id of which exactly one, with known times, is not ruled out."""
    window, reported = draw(WINDOW.filter(bool)), draw(COUNT)
    low, high = window[0] - SKEW, window[1] + SKEW

    def inside():  # a stamp that can name an instant in the widened window, edges included
        at = draw(st.integers(math.ceil(low * 4), math.floor(high * 4))) / 4
        return at, at + (1.0 if draw(st.booleans()) else 0.0)

    count = draw(COUNT if reported is None else st.sampled_from([None, reported]))
    fit = {"started_at": inside(), "finished_at": inside(), "output_tokens": count}
    others = draw(st.lists(READING, min_size=1, max_size=3))
    for other in others:
        if not _ruled_out(other, window, reported):  # give it evidence against it
            evidence = draw(
                st.sampled_from(["early", "late"] + (["count"] if reported is not None else []))
            )
            if evidence == "early":
                other["started_at"] = (low - 1.25, low - 0.25)
            elif evidence == "late":
                other["finished_at"] = (high + 0.25, high + 0.25)
            else:
                other["output_tokens"] = reported + 1
    at = draw(st.integers(0, len(others)))
    return others[:at] + [fit] + others[at:], fit, window, reported


@SETTINGS
@given(one_left())
def test_the_one_record_not_ruled_out_is_joined(case):
    readings, fit, window, reported = case
    assert pipeline.fitting_raw(readings, window, reported) is fit


@SETTINGS
@given(st.lists(READING, min_size=2, max_size=4), COUNT)
def test_without_a_window_no_record_is_resolved(readings, reported):
    assert pipeline.fitting_raw(readings, None, reported) is None


KEY = st.sampled_from(["r:5", "r:7", "d:5", "d:9"])
INDEX = st.dictionaries(KEY, st.lists(READING, min_size=1, max_size=3))


@SETTINGS
@given(INDEX, st.lists(KEY, max_size=4), WINDOW, COUNT)
def test_an_id_naming_one_record_joins_it_whatever_its_times(index, keys, window, reported):
    unique = [k for k in keys if len(index.get(k, ())) == 1]
    assume(unique)
    joined, how = pipeline.join_raw(index, keys, window, reported)
    assert how == "unique" and joined is index[unique[0]][0]


@SETTINGS
@given(one_left(), st.sampled_from(["r:5", "r:7"]))
def test_an_engine_id_naming_several_records_joins_the_one_left(case, key):
    readings, fit, window, reported = case
    joined, how = pipeline.join_raw({key: readings}, [key], window, reported)
    assert how == "resolved" and joined is fit


@SETTINGS
@given(one_left(), st.sampled_from(["d:5", "d:9"]))
def test_a_disagg_id_naming_several_records_joins_nothing(case, key):
    readings, _, window, reported = case
    assert pipeline.join_raw({key: readings}, [key], window, reported) == (None, "ambiguous")


@SETTINGS
@given(INDEX, st.lists(KEY, max_size=4), WINDOW, COUNT)
def test_only_an_engine_id_naming_several_records_is_resolved(index, keys, window, reported):
    assume(not [k for k in keys if len(index.get(k, ())) == 1])
    joined, how = pipeline.join_raw(index, keys, window, reported)
    fits = [pipeline.fitting_raw(index[k], window, reported) for k in keys
            if k in index and k.startswith("r:")]  # fmt: skip
    fits = [f for f in fits if f is not None]
    if fits:
        assert how == "resolved" and joined is fits[0]
    else:
        assert joined is None
        assert how == ("ambiguous" if any(k in index for k in keys) else None)


def _recorder_stamp(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


@SETTINGS
@given(MOMENT)
def test_a_stamp_names_the_instant_it_was_written_at(moment):
    at = moment.timestamp()
    for stamp in (_recorder_stamp(moment), moment.isoformat(),
                  moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")):  # fmt: skip
        earliest, latest = pipeline.instant(stamp)
        assert earliest <= at <= latest and latest - earliest <= 1.0


@SETTINGS
@given(MOMENT, QUARTERS, st.booleans())
def test_a_reply_window_holds_the_whole_exchange(arrived, seconds, whole):
    """Arrival to completion, as the recorder (whole seconds) or the request trace stamps them."""
    finished = arrived + timedelta(seconds=seconds)
    write = _recorder_stamp if whole else datetime.isoformat
    for response in ({"finished_at": write(finished)}, {"timing": {"total_s": seconds}}):
        window = pipeline.reply_window(write(arrived), response)
        assert window[0] <= arrived.timestamp() and finished.timestamp() <= window[1]


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q", "-p", "no:cacheprovider"]))
