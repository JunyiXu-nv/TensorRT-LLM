"""The gateway names each forwarded request, and signs the terminal it injects.

Pure stdlib, no cluster: ``upstream_head`` is a function of its inputs, and
``SSE_ROTATED`` is a constant. The request trace on the serving side
(tensorrt_llm/serve/request_trace.py) reads both, so what is pinned here is
the contract between the two files.
"""

import importlib.util
import json
import os
from types import SimpleNamespace

import pytest

GATEWAY_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gateway.py")
SPEC = importlib.util.spec_from_file_location("gw_identity_under_test", GATEWAY_PY)
gw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gw)

BACKEND = SimpleNamespace(host="10.0.0.7", port=8200, url="http://10.0.0.7:8200", job_id="job7")


def _head_lines(trace):
    head = gw.Gateway.__new__(gw.Gateway).upstream_head(
        BACKEND, "POST", "/v1/responses", [("content-type", "application/json")], "linj", trace
    )
    return head.decode("latin-1").split("\r\n")


def test_request_id_and_attempt_are_forwarded():
    trace = gw.RequestTrace()
    lines = _head_lines(trace)
    assert "X-Gateway-Request-Id: %s" % trace.rid in lines
    assert "X-Gateway-Attempt: 1" in lines
    assert "X-Gateway-User: linj" in lines
    assert "content-type: application/json" in lines


def test_a_retry_counts_up():
    trace = gw.RequestTrace()
    trace.attempt += 1
    assert "X-Gateway-Attempt: 2" in _head_lines(trace)


def test_without_a_trace_the_head_is_as_before():
    head = gw.Gateway.__new__(gw.Gateway).upstream_head(BACKEND, "GET", "/v1/models", [], "linj")
    assert b"X-Gateway-Request-Id" not in head
    assert b"X-Gateway-Attempt" not in head


def test_the_injected_terminal_says_it_is_the_gateways():
    frame = gw.SSE_ROTATED.decode("utf-8")
    assert frame.startswith("event: error\n")
    data = json.loads(frame.split("\n")[1][len("data: ") :])
    assert data["type"] == "error"
    assert data["error"]["source"] == "gateway"
    assert data["error"]["code"] == "backend_rotated"
    assert data["error"]["type"] == "overloaded_error"


def test_client_copies_of_the_gateway_identity_are_stripped():
    trace = gw.RequestTrace()
    head = gw.Gateway.__new__(gw.Gateway).upstream_head(
        BACKEND,
        "POST",
        "/v1/responses",
        [
            ("X-Gateway-Request-Id", "forged"),
            ("x-gateway-attempt", "9"),
            ("X-Gateway-User", "someone-else"),
            ("content-type", "application/json"),
        ],
        "linj",
        trace,
    )
    lines = head.decode("latin-1").split("\r\n")
    assert [l for l in lines if l.lower().startswith("x-gateway-request-id:")] == ["X-Gateway-Request-Id: %s" % trace.rid]
    assert [l for l in lines if l.lower().startswith("x-gateway-attempt:")] == ["X-Gateway-Attempt: 1"]
    assert [l for l in lines if l.lower().startswith("x-gateway-user:")] == ["X-Gateway-User: linj"]
    assert "content-type: application/json" in lines


def test_the_serving_side_recognizes_the_terminal_by_the_same_marker():
    request_trace = pytest.importorskip("tensorrt_llm.serve.request_trace")
    assert request_trace.gateway_injected_terminal(gw.SSE_ROTATED.decode("utf-8")) is True
    worker_error = 'event: error\ndata: {"type":"error","error":{"type":"server_error","message":"x"}}\n\n'
    assert request_trace.gateway_injected_terminal(worker_error) is False
