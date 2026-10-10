# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-request table: request_trace x perf_metrics x adp_route_trace, joined on the server's request id.

The join key is the proxy's `disagg_request_id` on a disaggregated run. An aggregated run has no
proxy: the response line carries the server's own `client_id`, which is the `request_id` of the
one worker record (unique per server process; the frontend id sits in its high bits), so that is
the key there. `deployment_mode()` tells the two apart from the perf records (only a disaggregated
run has proxy records) or, failing those, from the config snapshot beside the logs.

Sources inside one attempt directory:
  request_trace/<UTC hour>/requests-<pid>.jsonl    one line per request at handler entry (body, status)
  request_trace/<UTC hour>/responses-<pid>.jsonl   one line per request when the response ends (SSE text)
  perf_metrics/perf_metrics-disagg-*.jsonl         proxy record per request: arrival / first-token stamps
  perf_metrics/perf_metrics-server-*.jsonl         worker records (ctx, gen, or the one aggregated server):
                                                   timing, KV blocks, GPU time, draft acceptance
  adp_route_trace-ctx-N.jsonl                      attention-DP routing decision per request

One row per request-trace line. Requests rejected with 400 have no response and no rid; they keep
their row with status=rejected_400 and validation_errors filled.

Timestamps. The trace stamps `recorded_at` when the trace hook runs, which is after the body has been
read and validated: on these 0.3-0.8 MB agent requests that is 0.5-2 s (measured 1.5 s per MB). The
middleware's `server_arrival_time` is taken before any of that, but on a steady clock. The constant
offset between the two clocks is recovered as the minimum of (recorded_at - server_arrival_time) over
the whole attempt, which a near-empty request (health probe) pins to the true offset; `started_at`
is the arrival converted with it, and `parse_ms` is what the handler waited for the body.

Latency columns, both modes: `e2e_ms` is trace `finished_at - started_at`; `prefill_queue_ms` is the
worker's `arrival_time -> first_scheduled_time`; `prefill_ms` its `first_scheduled_time -> first_token_time`;
`gen_decode_ms` its `first_token_time -> last_token_time` (the gen record in a disaggregated run, the
one worker record in an aggregated one); `server_overhead_ms` is `server_arrival_time -> arrival_time`
on the same record (HTTP handler to engine enqueue). `ttft_ms` is the proxy's arrival-to-first-token
when there is a proxy, else the worker's `server_arrival_time -> first_token_time` -- NOT its
`server_first_token_time`, which the aggregated server stamps when the response ends.
`mtp_acceptance` is the record's speculative_decoding.acceptance_rate (accepted / proposed draft tokens).

Token columns come from the usage block the client received (chat: last chunk; responses:
`response.completed`). NOTE: on the /v1/responses route cached_tokens is currently reported as the
whole prompt by the server; the numbers are recorded as-is and must be read with that in mind.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common import parse_iso, to_float

COLUMNS = [
    # identity
    "rid", "trace_id", "session", "route", "started_at", "handler_entry_at", "finished_at", "status", "stream", "model",
    # tokens (from usage)
    "isl_total", "isl_cached", "isl_new", "osl", "cache_hit_ratio",
    # latency (ms)
    "parse_ms", "server_overhead_ms", "proxy_dispatch_ms", "ttft_ms", "e2e_ms", "decode_ms", "prefill_queue_ms", "prefill_ms",
    "kv_transfer_ms", "kv_transfer_bytes", "gen_decode_ms", "gpu_prefill_ms", "mtp_acceptance",
    # engine iterations and KV blocks
    "ctx_first_iter", "ctx_last_iter", "gen_first_iter", "gen_last_iter",
    "ctx_blocks_total", "ctx_blocks_new", "ctx_blocks_reused", "gen_blocks_total",
    # routing
    "ctx_instance", "routed_rank", "route_phase", "route_iter", "route_log_iter",
    # 400s only
    "validation_errors",
]


# ---------------------------------------------------------------- request_trace
def read_jsonl(path: Path):
    """Records of one JSONL file, one at a time.

    A generator rather than a list because every caller only iterates, and the
    request trace is the largest thing here: an agent turn replays its whole
    history, so a single recorded body runs to a hundred kilobytes and an hour
    of one instance is a few gigabytes. Building the list meant holding the
    file and what the caller kept out of it at the same time.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def trace_hour_dirs(attempt: Path, hour: str | None) -> list[Path]:
    root = attempt / "request_trace"
    if hour:
        return [root / hour]
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def next_hour(hour: str) -> str:
    """'2026-09-17T09' -> '2026-09-17T10'."""
    from datetime import datetime, timedelta
    return (datetime.strptime(hour, "%Y-%m-%dT%H") + timedelta(hours=1)).strftime("%Y-%m-%dT%H")


# What request_row() reads off a request line. Everything else on it is the
# recorded body: an agent turn replays its whole history, so the median line
# here is 166 KB and an hour of one instance is 2.1 GB of them -- held, in the
# whole-line version, to read two scalars out of each. Named here rather than
# inlined so the set that is kept and the set that is used are one edit apart.
REQUEST_FIELDS = ("trace_id", "session", "route", "status", "recorded_at", "server_arrival_time",
                  "validation_errors")
REQUEST_BODY_FIELDS = ("stream", "model")


def slim_request(rec: dict) -> dict:
    """A request line reduced to the fields the table reads."""
    body = rec.get("body")
    body = body if isinstance(body, dict) else {}
    kept = {key: rec.get(key) for key in REQUEST_FIELDS}
    kept["body"] = {key: body.get(key) for key in REQUEST_BODY_FIELDS}
    return kept


# The scalars of a request line sit in its first kilobyte, in front of the body. A quoted string, a
# number or a JSON literal, so the value can be handed to json.loads on its own.
_HEAD_FIELDS = {key: re.compile(r'"%s":\s*("(?:[^"\\]|\\.)*"|-?[0-9][0-9.eE+-]*|null|true|false)' % key)
                for key in ("trace_id", "session", "route", "status", "recorded_at", "server_arrival_time")}
_HAS_VALIDATION = re.compile(r'"validation_errors":\s*(?!null)')
_STREAM = re.compile(r'"stream":\s*(true|false)')
_MODEL = re.compile(r'"model":\s*"((?:[^"\\]|\\.)*)"')
LOAD_STATS = {"undecodable": 0, "parsed_fully": 0}   # of the last load_trace(); build_requests notes them


def slim_request_line(line: str) -> dict | None:
    """slim_request() without parsing the body.

    A request line is a few scalars followed by a body of ~170 KB (the agent replays its whole
    history); json.loads of every line was most of the request side's time. The scalars are read
    off the first 2 KB, `model` is the body's first key so its first occurrence is the right one,
    and `stream` sits after the history, so its LAST occurrence is taken -- the history is code and
    conversation and may well contain the text "stream": true. A rejected request keeps its
    validation_errors and is parsed fully; so is anything the regexes cannot place.
    """
    head = line[:2000]
    rec = {}
    for key, pattern in _HEAD_FIELDS.items():
        hit = pattern.search(head)
        rec[key] = json.loads(hit.group(1)) if hit else None
    if rec["trace_id"] is None or _HAS_VALIDATION.search(head):
        LOAD_STATS["parsed_fully"] += 1
        try:
            return slim_request(json.loads(line))
        except ValueError:
            return None
    model = _MODEL.search(line)
    stream = None
    at = line.rfind('"stream":')
    if at >= 0:
        hit = _STREAM.match(line, at)
        stream = None if hit is None else hit.group(1) == "true"
    rec["validation_errors"] = None
    rec["body"] = {"stream": stream, "model": None if model is None else model.group(1)}
    return rec


def load_trace(attempt: Path, hour: str | None) -> tuple[dict[str, dict], dict[str, dict]]:
    """trace_id -> request line (slimmed), trace_id -> response line.

    The trace files are bucketed by the UTC hour of the event: a request that arrives at 09:59 and
    ends at 10:01 has its request line in T09 and its response line in T10. In single-hour mode the
    next hour's response files are therefore read too, keeping only the responses of this hour's
    requests; without them every request finishing across the boundary read as `no_response`.
    (The newest hour of a live instance still shows that artifact until its successor exists.)

    A file cut short (quota hit, crash) ends in a partial line; it is dropped and counted in
    LOAD_STATS rather than failing the hour.
    """
    LOAD_STATS.update(undecodable=0, parsed_fully=0, responses_from_next_hour=0)
    requests: dict[str, dict] = {}
    responses: dict[str, dict] = {}
    hour_dirs = trace_hour_dirs(attempt, hour)
    for hour_dir in hour_dirs:
        for path in sorted(hour_dir.glob("requests-*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rec = slim_request_line(line)
                    if rec is None:
                        LOAD_STATS["undecodable"] += 1
                        continue
                    requests[rec["trace_id"]] = rec
    response_dirs = list(hour_dirs)
    if hour:
        spill = attempt / "request_trace" / next_hour(hour)
        if spill.is_dir():
            response_dirs.append(spill)
    for hour_dir in response_dirs:
        from_next = hour and hour_dir.name != hour
        for path in sorted(hour_dir.glob("responses-*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        LOAD_STATS["undecodable"] += 1
                        continue
                    if hour and rec.get("trace_id") not in requests:
                        continue  # a request of another hour; that hour's report owns it
                    if from_next:
                        LOAD_STATS["responses_from_next_hour"] += 1
                    responses[rec["trace_id"]] = rec
    return requests, responses


_RECORDED_AT = re.compile(r'"recorded_at":\s*"([^"]+)"')
_ARRIVAL = re.compile(r'"server_arrival_time":\s*([0-9][0-9.eE+-]*)')


def clock_offset(attempt: Path, hour: str | None = None) -> float | None:
    """Wall clock minus the steady clock, from the request line least delayed by body parsing.

    Restricted to `hour` when given: the offset is a constant of the frontend processes, an hour of
    lines pins it as well as the whole attempt does, and the whole attempt is tens of GB on a
    long-lived instance -- re-read for every hour report before this took the hour.
    """
    best = None
    for hour_dir in trace_hour_dirs(attempt, hour):
        for path in hour_dir.glob("requests-*.jsonl"):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    head = line[:1000]  # both stamps sit before the body
                    recorded, arrival = _RECORDED_AT.search(head), _ARRIVAL.search(head)
                    if recorded and arrival and (stamp := parse_iso(recorded.group(1))) is not None:
                        offset = stamp - float(arrival.group(1))
                        best = offset if best is None else min(best, offset)
    return best


def sse_events(text: str):
    """Yield (event, data) per SSE block. `data` lines of one block are joined."""
    for block in text.split("\n\n"):
        event, data = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            yield event, "\n".join(data)


def usage_of(route: str, response: dict) -> dict | None:
    """The usage block the client received, whatever the wire format.

    Newer trace lines carry it at the top level; otherwise it is dug out of the response body:
    a JSON body has `usage` directly, a chat stream puts it on the last chunk, a Responses stream
    puts it under `response.usage` of the `response.completed` event.
    """
    if isinstance(response.get("usage"), dict):
        return response["usage"]
    payload = response.get("response") or {}
    body = payload.get("body", payload.get("frames"))
    if isinstance(body, list):
        body = "".join(str(frame) for frame in body)
    if isinstance(body, dict):
        return body.get("usage")
    if not isinstance(body, str):
        return None
    usage = None
    for _event, data in sse_events(body):
        if data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if route == "/v1/responses":
            if chunk.get("type") == "response.completed":
                usage = (chunk.get("response") or {}).get("usage")
        elif chunk.get("usage"):
            usage = chunk["usage"]
    return usage


def tokens_of(usage: dict | None) -> tuple[int | None, int | None, int | None]:
    """(isl_total, isl_cached, osl) from either the chat or the Responses usage shape."""
    if not usage:
        return None, None, None
    if "input_tokens" in usage:  # Responses API
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
        return usage.get("input_tokens"), cached, usage.get("output_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return usage.get("prompt_tokens"), cached, usage.get("completion_tokens")


# ---------------------------------------------------------------- perf_metrics
def perf_role(path: Path, record: dict) -> str:
    """'proxy', 'ctx' or 'gen'. Worker files are named by process, so the hop is read off the content."""
    token = path.stem.split("-")[1] if "-" in path.stem else ""
    if token in ("disagg", "proxy") or "ctx_perf_metrics" in record or "gen_perf_metrics" in record:
        return "proxy"
    if token in ("ctx", "gen"):
        return token
    timing = (record.get("perf_metrics") or {}).get("timing_metrics") or {}
    return "gen" if "kv_cache_transfer_start" in timing else "ctx"


def load_perf(attempt: Path) -> dict[str, dict]:
    """disagg_request_id -> {'proxy': record, 'ctx': record, 'gen': record} (each optional)."""
    perf: dict[str, dict] = {}
    for path in sorted(attempt.glob("perf_metrics/perf_metrics-*.jsonl")) + sorted(attempt.glob("perf_metrics-*.jsonl")):
        for record in read_jsonl(path):
            record = record.get("record", record)  # an older poller wrapped each record
            role = perf_role(path, record)
            rid = record.get("disagg_request_id") or record.get("ctx_request_id") or record.get("request_id")
            if rid is None:
                continue
            perf.setdefault(str(rid), {})[role] = record
    return perf


def worker_timing(record: dict | None) -> dict:
    return ((record or {}).get("perf_metrics") or {}).get("timing_metrics") or {}


def span_ms(timing: dict, start: str, end: str) -> float | None:
    a, b = to_float(timing.get(start)), to_float(timing.get(end))
    return None if a is None or b is None else (b - a) * 1000.0


# ---------------------------------------------------------------- routing
def route_key(decision: dict) -> str | None:
    """The id a routing decision is filed under, matching response_rid() of the same request.

    A decision carries three ids. On a disaggregated ctx worker `req_id` and `disagg_request_id`
    are the proxy's id (the response line's disagg_request_id) and `client_id` is that server's
    own counter, which nothing else records. On an aggregated server `disagg_request_id` is 0,
    `req_id` is the engine's small sequential id and `client_id` is the id the response line has.
    """
    if decision.get("disagg_request_id"):
        return str(decision["disagg_request_id"])
    if decision.get("client_id") is not None:
        return str(decision["client_id"])
    return None if decision.get("req_id") is None else str(decision["req_id"])


def load_routes(attempt: Path) -> dict[str, dict]:
    """request id (see route_key) -> routing decision, tagged with the instance whose trace it came from."""
    routes: dict[str, dict] = {}
    for path in sorted(attempt.glob("adp_route_trace*.jsonl")):
        instance = path.stem.replace("adp_route_trace-", "").replace("adp_route_trace", "server")
        for batch in read_jsonl(path):
            for decision in batch.get("decisions") or []:
                key = route_key(decision)
                if key is None:
                    continue
                routes[key] = {
                    "ctx_instance": instance,
                    "routed_rank": decision.get("best_rank"),
                    "route_phase": decision.get("phase"),
                    "route_iter": batch.get("iter"),
                    "route_log_iter": batch.get("log_iter"),
                }
    return routes


# ---------------------------------------------------------------- rows
def request_row(req: dict, resp: dict | None, perf: dict, route: dict, offset: float | None,
                mode: str = "disagg") -> dict:
    body = req.get("body") if isinstance(req.get("body"), dict) else {}
    handler_entry = parse_iso(req.get("recorded_at"))
    arrival = to_float(req.get("server_arrival_time"))
    started = arrival + offset if arrival is not None and offset is not None else handler_entry
    finished = parse_iso((resp or {}).get("finished_at"))
    isl_total, isl_cached, osl = tokens_of(usage_of(req.get("route", ""), resp) if resp else None)
    isl_new = isl_total - isl_cached if isl_total is not None and isl_cached is not None else None

    proxy, ctx, gen = perf.get("proxy") or {}, perf.get("ctx"), perf.get("gen")
    if mode == "agg" and gen is None:
        # One worker record per request; perf_role() files it as ctx (it has no KV transfer
        # stamps) and it is the decode side too, so the decode columns read off the same record.
        gen = ctx
    ctx_t, gen_t = worker_timing(ctx), worker_timing(gen)
    spec = ((gen or ctx or {}).get("perf_metrics") or {}).get("speculative_decoding") or {}
    ctx_kv = ((ctx or {}).get("perf_metrics") or {}).get("kv_cache_metrics") or {}
    gen_kv = ((gen or {}).get("perf_metrics") or {}).get("kv_cache_metrics") or {}
    ctx_breakdown = (ctx or {}).get("time_breakdown_metrics") or {}

    proxy_arrival, first_token = to_float(proxy.get("disagg_server_arrival_time")), to_float(proxy.get("disagg_server_first_token_time"))
    dispatch = to_float(proxy.get("disagg_ctx_dispatch_time"))
    ttft = (first_token - proxy_arrival) * 1000.0 if proxy_arrival is not None and first_token is not None else None
    if ttft is None:
        # No proxy (aggregated deployment): the worker's own HTTP arrival to its first token.
        # Not server_first_token_time -- that stamp is written when the response ends there,
        # so it would read as E2E.
        ttft = span_ms(ctx_t, "server_arrival_time", "first_token_time")
    e2e = (finished - started) * 1000.0 if started is not None and finished is not None else None

    if req.get("status") == "rejected_400":
        status = "rejected_400"
    else:
        status = (resp or {}).get("status") or "no_response"

    return {
        "rid": response_rid(resp),
        "trace_id": req.get("trace_id"),
        # The conversation key the trace resolved (request_trace.resolve_session_key: headers, then
        # prompt_cache_key / client_metadata in the body); "_no_session" when nothing was found.
        "session": req.get("session") or (resp or {}).get("session"),
        "route": req.get("route"),
        "started_at": started,
        "handler_entry_at": handler_entry,
        "finished_at": finished,
        "status": status,
        "stream": body.get("stream"),
        "model": body.get("model"),
        "isl_total": isl_total,
        "isl_cached": isl_cached,
        "isl_new": isl_new,
        "osl": osl,
        "cache_hit_ratio": (isl_cached / isl_total) if isl_total and isl_cached is not None else None,
        "parse_ms": (handler_entry - started) * 1000.0 if handler_entry is not None and started is not None else None,
        "server_overhead_ms": span_ms(ctx_t, "server_arrival_time", "arrival_time"),
        "proxy_dispatch_ms": (dispatch - proxy_arrival) * 1000.0 if dispatch is not None and proxy_arrival is not None else None,
        "ttft_ms": ttft,
        "e2e_ms": e2e,
        "decode_ms": (e2e - ttft) if e2e is not None and ttft is not None else None,
        "prefill_queue_ms": span_ms(ctx_t, "arrival_time", "first_scheduled_time"),
        "prefill_ms": span_ms(ctx_t, "first_scheduled_time", "first_token_time"),
        "kv_transfer_ms": span_ms(gen_t, "kv_cache_transfer_start", "kv_cache_transfer_end"),
        "kv_transfer_bytes": gen_t.get("kv_cache_size"),
        "gen_decode_ms": span_ms(gen_t, "first_token_time", "last_token_time"),
        "gpu_prefill_ms": to_float(ctx_breakdown.get("ctx_gpu_forward_time")),
        "mtp_acceptance": to_float(spec.get("acceptance_rate")),
        "ctx_first_iter": ((ctx or {}).get("perf_metrics") or {}).get("first_iter"),
        "ctx_last_iter": ((ctx or {}).get("perf_metrics") or {}).get("last_iter"),
        "gen_first_iter": ((gen or {}).get("perf_metrics") or {}).get("first_iter"),
        "gen_last_iter": ((gen or {}).get("perf_metrics") or {}).get("last_iter"),
        "ctx_blocks_total": ctx_kv.get("num_total_allocated_blocks"),
        "ctx_blocks_new": ctx_kv.get("num_new_allocated_blocks"),
        "ctx_blocks_reused": ctx_kv.get("num_reused_blocks"),
        "gen_blocks_total": gen_kv.get("num_total_allocated_blocks"),
        "ctx_instance": route.get("ctx_instance"),
        "routed_rank": route.get("routed_rank"),
        "route_phase": route.get("route_phase"),
        "route_iter": route.get("route_iter"),
        "route_log_iter": route.get("route_log_iter"),
        "validation_errors": _brief(req.get("validation_errors")),
    }


def _brief(errors: Any) -> str | None:
    if not errors:
        return None
    parts = []
    for err in errors if isinstance(errors, list) else [errors]:
        if isinstance(err, dict):
            parts.append(f"{err.get('loc', '')}: {err.get('msg', err.get('message', ''))}".strip(": "))
        else:
            parts.append(str(err))
    return " | ".join(parts)[:300]


def deployment_mode(attempt: Path, perf: dict[str, dict]) -> str:
    """'disagg' when a proxy fronts ctx and gen workers, 'agg' when one server does both phases.

    Only a disaggregated run writes proxy perf records, so those decide when there are any;
    otherwise the config snapshot the launcher left beside the logs does.
    """
    if perf:
        return "disagg" if any("proxy" in roles for roles in perf.values()) else "agg"
    if (attempt / "disagg_config.yaml").exists() or (attempt / "ctx_config.yaml").exists():
        return "disagg"
    return "agg" if (attempt / "server_config.yaml").exists() else "disagg"


def response_rid(resp: dict | None) -> str | None:
    """The id the perf records and routing trace know the request by.

    A disaggregated response carries the proxy's disagg_request_id; an aggregated one carries the
    server's client_id (its request_id) and disagg_request_id is null.
    """
    for key in ("disagg_request_id", "client_id"):
        if resp and resp.get(key) is not None:
            return str(resp[key])
    return None


def build_requests(attempt: Path, hour: str | None) -> tuple[list[dict], list[str]]:
    """All request rows for the attempt (or one UTC hour of it), sorted by start time, plus notes."""
    requests, responses = load_trace(attempt, hour)
    perf, routes, offset = load_perf(attempt), load_routes(attempt), clock_offset(attempt, hour)
    mode = deployment_mode(attempt, perf)
    rows = []
    for tid, req in requests.items():
        resp = responses.get(tid)
        rid = response_rid(resp)
        rows.append(request_row(req, resp, perf.get(rid, {}) if rid else {}, routes.get(rid, {}) if rid else {}, offset, mode))
    rows.sort(key=lambda r: (r["started_at"] or 0, str(r["rid"])))

    with_rid = [r for r in rows if r["rid"] is not None]
    if mode == "agg":
        join_note = (f"Aggregated deployment (one server does prefill and decode). Of {len(with_rid)} requests with a "
                     f"server request id (client_id): {sum(1 for r in with_rid if r['prefill_queue_ms'] is not None)} "
                     f"joined the worker's perf record, {sum(1 for r in with_rid if r['routed_rank'] is not None)} a routing decision.")
    else:
        join_note = (f"Disaggregated deployment. Of {len(with_rid)} requests with a disagg_request_id: "
                     f"{sum(1 for r in with_rid if r['ttft_ms'] is not None)} joined a proxy perf record, "
                     f"{sum(1 for r in with_rid if r['ctx_blocks_total'] is not None)} a ctx worker record, "
                     f"{sum(1 for r in with_rid if r['gen_last_iter'] is not None)} a gen worker record, "
                     f"{sum(1 for r in with_rid if r['routed_rank'] is not None)} a routing decision.")
    notes = [
        f"{len(rows)} request lines, {len(responses)} response lines; "
        f"{sum(1 for r in rows if r['status'] == 'rejected_400')} rejected with 400, "
        f"{sum(1 for r in rows if r['status'] == 'no_response')} accepted but without a response line.",
        join_note,
        f"Usage found on {sum(1 for r in rows if r['isl_total'] is not None)} responses; "
        f"status counts: {dict(Counter(r['status'] for r in rows))}.",
        ("started_at is the middleware arrival (steady clock + recovered offset); the trace's recorded_at lags it by "
         f"the body parse, median {sorted(r['parse_ms'] for r in rows if r['parse_ms'] is not None)[len(rows) // 2]:,.0f} ms here."
         if offset is not None and any(r["parse_ms"] is not None for r in rows) else
         "No server_arrival_time on the request lines: started_at is the trace's recorded_at (handler entry)."),
    ]
    if LOAD_STATS["undecodable"]:
        notes.append(f"{LOAD_STATS['undecodable']} trace line(s) could not be decoded (a file cut short) and were skipped.")
    if LOAD_STATS.get("responses_from_next_hour"):
        notes.append(f"{LOAD_STATS['responses_from_next_hour']} responses were read from the next hour's file (requests that "
                     "finished after the hour ended); the remaining no_response rows have no response line in either hour.")
    return rows, notes


if __name__ == "__main__":  # quick look: python3 requests_table.py <attempt_dir> [hour]
    import sys
    from common import TIME_COLUMNS, write_csv
    attempt = Path(sys.argv[1])
    rows, notes = build_requests(attempt, sys.argv[2] if len(sys.argv) > 2 else None)
    out = write_csv(Path("requests.csv"), rows, COLUMNS, TIME_COLUMNS)
    print(f"{len(rows)} rows -> {out}")
    for note in notes:
        print("  " + note)
