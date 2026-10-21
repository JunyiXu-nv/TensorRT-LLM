#!/usr/bin/env python3
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
"""The gateway names a conversation to the backend when the client did not.

trtllm-serve keeps a conversation's turns on one context server and one
attention-DP rank only when it can name the conversation: from a header such as
x-session-id, or from a body field such as Codex's prompt_cache_key. On jhb on
2026-10-21 the GLM fill's opencode and n3 traffic sent neither. Every turn then
went to a round-robin rank and prefilled its whole history again, and the
prefill queue grew to 27 s. The gateway already keys such a request by its
prefix; it now passes that key on as X-Session-Id.
"""

import importlib.util
import json
import socket

import pytest
from handover_harness import API_USER, CONVERSATION_HEADER, GATEWAY_PY, Scenario

SPEC = importlib.util.spec_from_file_location("gw_backend_conversation_under_test", GATEWAY_PY)
gw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gw)

PREFIX_KEY = "prefix:0123456789abcdef0123456789abcdef"


# -- the rule ----------------------------------------------------------------
def test_a_prefix_key_is_named_to_the_backend():
    headers = [("Content-Type", "application/json")]
    out = gw.with_backend_conversation(headers, PREFIX_KEY)
    assert out == headers + [("X-Session-Id", PREFIX_KEY)]
    assert headers == [("Content-Type", "application/json")], "the caller's list is not changed"


@pytest.mark.parametrize(
    "key",
    [None, "", "hdr:abc", "prompt_cache_key:019a-session", "client_metadata.thread_id:t1"],
)
def test_keys_the_backend_can_read_itself_are_left_alone(key):
    headers = [("Content-Type", "application/json")]
    assert gw.with_backend_conversation(headers, key) is headers


@pytest.mark.parametrize("name", sorted(gw.BACKEND_CONVERSATION_HEADERS))
def test_a_client_conversation_header_wins(name):
    headers = [(name.upper(), "client-session")]
    assert gw.with_backend_conversation(headers, PREFIX_KEY) is headers


def test_an_empty_client_header_does_not_count():
    headers = [("x-session-id", "  ")]
    assert gw.with_backend_conversation(headers, PREFIX_KEY)[-1] == ("X-Session-Id", PREFIX_KEY)


def test_the_header_set_is_the_backends():
    """Pinned to tensorrt_llm/serve/conversation_id.py:CONVERSATION_ID_HEADERS."""
    assert gw.BACKEND_CONVERSATION_HEADERS == {
        "x-claude-code-session-id",
        "x-claude-session-id",
        "x-session-id",
        "x-correlation-id",
        "x-session-affinity",
        "x-multi-turn-session-id",
    }


# -- through a running gateway ---------------------------------------------------
@pytest.fixture
def make_scenario():
    created = []

    def factory(*args, **kwargs):
        scenario = Scenario(*args, **kwargs)
        created.append(scenario)
        return scenario

    yield factory
    for scenario in reversed(created):
        scenario.close()


def post(port, body, extra=()):
    """POST /v1/responses through the gateway; the fake backend's JSON answer."""
    lines = [
        "POST /v1/responses HTTP/1.1",
        "Host: 127.0.0.1:%d" % port,
        "x-api-key: %s" % API_USER,
        "Content-Length: %d" % len(body),
        "Content-Type: application/json",
        "Connection: close",
    ]
    lines.extend("%s: %s" % header for header in extra)
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
    with socket.create_connection(("127.0.0.1", port), timeout=10.0) as sock:
        sock.sendall(head + body)
        sock.settimeout(15.0)
        received = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            received += chunk
    status_line, _, rest = received.partition(b"\r\n")
    assert b" 200 " in status_line + b" ", received[:300]
    payload = rest.partition(b"\r\n\r\n")[2]
    return json.loads(payload[payload.index(b"{") : payload.rindex(b"}") + 1])


def test_the_backend_sees_the_prefix_key_of_a_client_that_names_nothing(make_scenario):
    scenario = make_scenario("backend-conversation", backends=1).start()
    body = json.dumps(
        {"model": "m", "input": [{"role": "user", "content": "run the kernel test"}]}
    ).encode()
    expected = gw.conversation_key([], body, gw.DEFAULT_KEY_SOURCES)
    assert expected.startswith("prefix:")

    answer = post(scenario.port, body)
    assert answer["session_header"] == expected
    # The same conversation's next turn carries the same name.
    assert post(scenario.port, body)["session_header"] == expected


def test_a_client_that_names_its_conversation_is_forwarded_as_sent(make_scenario):
    scenario = make_scenario("backend-conversation-named", backends=1).start()
    body = json.dumps({"model": "m", "input": [{"role": "user", "content": "hi"}]}).encode()

    # A conversation header the gateway keys on: the backend gets no extra header.
    assert post(scenario.port, body, [(CONVERSATION_HEADER, "c1")])["session_header"] == ""
    # A header only the backend reads: left for the backend, not overridden.
    answer = post(scenario.port, body, [("x-claude-code-session-id", "cc-1")])
    assert answer["session_header"] == ""
    # The client's own x-session-id reaches the backend unchanged.
    assert post(scenario.port, body, [("x-session-id", "s-9")])["session_header"] == "s-9"
