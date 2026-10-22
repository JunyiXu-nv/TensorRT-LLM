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
"""Turn records -> per-check counts, for every protocol (references/validate.md section 4).

Three passes over request-trace shaped input (trace windows, replay output,
recorder captures):

1. requests: one small summary per request (identity, order, tool
   declarations by key, call policy), plus the FIRST appearance per thread of
   every item the harness echoes back and of every tool result;
2. responses: each reply joined to its request and parsed by its protocol
   adapter; per-reply checks run here, with content in memory only;
3. threads: each turn against the next request of its thread (echo, call
   outcomes, resends, over-length handling).

Memory grows with requests and calls, not with history length: a history
replays the same items every turn, and only their first appearance is kept.
"""

import bisect
import json
import zlib
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import profiles as prof
import proto_anthropic
import proto_chat
import proto_responses
from tracelib import expand_shards, header_map, iter_lines, load_json_line
from turnlib import (
    SERVING_SHAPE,
    RequestView,
    classify_call,
    marker_hits,
    payload_digest,
    safe_name,
    sha256_text,
)

ADAPTERS = {
    "/v1/responses": proto_responses,
    "/v1/messages": proto_anthropic,
    "/v1/chat/completions": proto_chat,
}
PROTOCOL_OF = {route: mod.PROTOCOL for route, mod in ADAPTERS.items()}
CLIENT_SIDE = ("client_disconnected", "client_first_event_timeout", "client_timeout")
OVERLENGTH_WORDS = ("context_length_exceeded", "prompt is too long", "context length",
                    "maximum context", "too many tokens", "prompt_too_long", "too long")  # fmt: skip
REPETITIVE_MIN_CHARS = 2000
REPETITIVE_MAX_ZLIB_RATIO = 0.10
EXAMPLES = 5


class Pipeline:
    def __init__(self, harness: prof.Profile, model: prof.Profile, build_id: Optional[str] = None):
        self.harness = harness
        self.model = model
        self.build_id = build_id
        self.outcomes = prof.outcome_rules(harness)
        self.markers = prof.marker_patterns(model)
        self.degenerate = prof.degenerate_patterns(model)
        self.counts: Counter = Counter()
        self.examples: Dict[str, list] = defaultdict(list)
        self.by_rule: Dict[str, Counter] = defaultdict(Counter)
        self.call_classes: Counter = Counter()
        self.outcome_classes: Counter = Counter()
        self.session_events: Dict[str, set] = defaultdict(set)  # check key -> sessions with it
        self.sessions: set = set()
        self.requests: Dict[str, dict] = {}
        self.auxiliary: set = set()
        self.tools: Dict[str, tuple] = {}
        self.echo_first: Dict[tuple, tuple] = {}
        self.result_first: Dict[tuple, tuple] = {}
        self.bodies: Dict[tuple, list] = defaultdict(list)
        self.turns: List[dict] = []
        self.seq = 0
        self.unknown_fields: Counter = Counter()
        self.protocols: Counter = Counter()

    # ------------------------------------------------------------------ helpers

    def note(self, key: str, locator: dict, session: Optional[str] = None, **detail) -> None:
        self.counts[key] += 1
        if session is not None:
            self.session_events[key].add(session)
        if len(self.examples[key]) < EXAMPLES:
            self.examples[key].append({**locator, **detail})

    # ------------------------------------------------------------------ pass 1

    def add_requests(self, paths: List[str]) -> None:
        for path in expand_shards(paths, "requests"):
            for line_no, raw in iter_lines(path):
                rec = load_json_line(raw)
                if rec is None:
                    self.counts["inventory:unparsable_request_lines"] += 1
                    continue
                if rec.get("event", "request") != "request":
                    continue
                self._add_request(rec, {"file": path, "line": line_no,
                                        "trace_id": rec.get("trace_id")})  # fmt: skip

    def _add_request(self, rec: dict, locator: dict) -> None:
        route = str(rec.get("route") or "").split("?", 1)[0]
        body = rec.get("body")
        self.counts["inventory:requests"] += 1
        adapter = ADAPTERS.get(route)
        if adapter is None:
            self.counts["inventory:auxiliary_route:" + route] += 1
            self.auxiliary.add(rec.get("trace_id"))
            return
        if not isinstance(body, dict):
            self.counts["inventory:request_without_body"] += 1
            return
        headers = header_map(rec)
        identity = prof.extract_identity(self.harness, headers, body, rec)
        view = adapter.parse_request(body, headers)
        self.seq += 1
        order = (str(rec.get("recorded_at") or ""), self.seq)
        thread = identity.get("thread")
        if view.tools_key not in self.tools:
            self.tools[view.tools_key] = (view.tools, view.nested)
        for key in view.unknown_fields:
            self.unknown_fields[key] += 1
        self.protocols[view.protocol] += 1
        summary = {
            "locator": locator,
            "protocol": view.protocol,
            "route": route,
            "identity": identity,
            "thread": thread,
            "order": order,
            "stream": view.stream,
            "tools_key": view.tools_key,
            "tool_choice": view.tool_choice,
            "parallel": view.parallel,
            "include_usage": view.include_usage,
            "build_id": rec.get("build_id"),
            "history": view.history_messages,
            "body_sha": rec.get("body_sha256")
            or sha256_text(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str)),
        }
        trace_id = rec.get("trace_id")
        if trace_id in self.requests:
            self.counts["inventory:duplicate_trace_id"] += 1
        self.requests[trace_id] = summary
        if thread:
            self.bodies[(thread, summary["body_sha"])].append((order, trace_id))
            for item_id, echo in view.echo.items():
                key = (thread, item_id)
                digest = (payload_digest(echo.get("payload"), echo.get("call_kind"))
                          if echo["kind"] == "call" else
                          sha256_text(echo.get("text") or "") if echo["kind"] == "text" else
                          echo.get("carrier"))  # fmt: skip
                entry = (order, echo["kind"], echo.get("name"), digest)
                if key not in self.echo_first or order < self.echo_first[key][0]:
                    self.echo_first[key] = entry
            for call_id, text, is_error in view.tool_results:
                key = (thread, call_id)
                if key in self.result_first and self.result_first[key][0] <= order:
                    continue
                cls = self.outcomes.classify(text, is_error) if self.outcomes else None
                self.result_first[key] = (order, cls)

    # ------------------------------------------------------------------ pass 2

    def add_responses(self, paths: List[str]) -> None:
        for path in expand_shards(paths, "responses"):
            for line_no, raw in iter_lines(path):
                rec = load_json_line(raw)
                if rec is None:
                    self.counts["inventory:unparsable_response_lines"] += 1
                    continue
                if rec.get("event", "response") != "response":
                    continue
                self._add_response(rec, {"file": path, "line": line_no,
                                         "trace_id": rec.get("trace_id")})  # fmt: skip

    def _request_view(self, summary: dict) -> RequestView:
        tools, nested = self.tools.get(summary["tools_key"], ({}, {}))
        view = RequestView(protocol=summary["protocol"], route=summary["route"])
        view.stream = summary["stream"]
        view.tools, view.nested = tools, nested
        view.tool_choice = summary["tool_choice"]
        view.parallel = summary["parallel"]
        view.include_usage = summary["include_usage"]
        return view

    def _add_response(self, rec: dict, loc: dict) -> None:
        self.counts["inventory:responses"] += 1
        summary = self.requests.get(rec.get("trace_id"))
        if summary is None:
            if rec.get("trace_id") in self.auxiliary:
                self.counts["inventory:auxiliary_responses"] += 1
            else:
                self.note("I-01:response_without_request", loc)
            return
        adapter = ADAPTERS[summary["route"]]
        request = self._request_view(summary)
        reply = adapter.parse_reply(rec, request)
        session = summary["identity"].get("session") or summary["thread"]
        if session:
            self.sessions.add(session)
        self.counts["den:replies"] += 1
        self.counts["den:replies:" + summary["protocol"]] += 1
        build_id = rec.get("build_id") or summary.get("build_id")
        if not build_id:
            self.counts["I-05:missing_build_id"] += 1
        elif self.build_id and build_id != self.build_id:
            self.note("I-05:other_build_id", loc)
        self._timing(rec, reply)
        client_cut = reply.status in CLIENT_SIDE
        delivered = reply.delivered()
        if client_cut:
            # The client stopped reading: the stream is incomplete by construction, so its
            # wire, call and content rules are not judged; H-03 attributes the cut.
            self.note("H-03:" + str(reply.status), loc, session)
            self.counts["den:client_cut_replies"] += 1
        else:
            self._wire(adapter, reply, loc, session, request)
            self._calls(delivered, request, reply, loc, session)
            self._content(delivered, reply, loc, session)
            self._overlength(reply, loc, session)
        self._runtime(reply, loc, session)
        if summary["thread"] is None:
            self.counts["I-09:turn_without_thread"] += 1
        self.turns.append({
            "trace_id": rec.get("trace_id"),
            "loc": loc,
            "thread": summary["thread"],
            "session": session,
            "agent": summary["identity"].get("agent"),
            "order": summary["order"],
            "terminal": reply.terminal.get("kind"),
            "http_status": reply.http_status,
            "overlength": self._is_overlength(reply),
            "client_error": isinstance(reply.http_status, int) and 400 <= reply.http_status < 500,
            "failed": reply.terminal.get("kind") in ("failed", "error", "none")
            or (isinstance(reply.http_status, int) and reply.http_status >= 400),
            "calls": [(i.call_id, i.name, payload_digest(i.payload, i.call_kind))
                      for i in delivered if i.kind == "call" and i.call_id],
            "texts": [(i.id, sha256_text(i.text or "")) for i in delivered
                      if i.kind == "text" and i.id and summary["protocol"] == "responses"],
            "body_sha": summary["body_sha"],
            "history": summary["history"],
        })  # fmt: skip

    def _timing(self, rec: dict, reply) -> None:
        timing = rec.get("timing") if isinstance(rec.get("timing"), dict) else {}
        first = timing.get("first_event_s", timing.get("first_byte_s"))
        if isinstance(first, (int, float)):
            reply.first_event_s = float(first)
        chunks = timing.get("chunks")
        if isinstance(chunks, list) and chunks:
            times = [
                c[0] for c in chunks if isinstance(c, list) and c and isinstance(c[0], (int, float))
            ]
            start = float(timing.get("headers_s") or 0.0)
            gaps = [b - a for a, b in zip([start] + times, times)]
            reply.max_gap_s = max(gaps) if gaps else None
            reply.timing_known = True

    def _body_rules(self, request, reply) -> List[tuple]:
        """Protocol-independent body rules (protocols.md section 4: WIRE-01 .. WIRE-04)."""
        ok_status = not isinstance(reply.http_status, int) or reply.http_status < 400
        if reply.kind == "text":
            return [("C-01", "WIRE-01")]  # neither SSE nor JSON
        if reply.kind == "none":
            return [("C-04", "WIRE-02")]  # no body at all
        if ok_status and request.stream and reply.kind == "json":
            return [("C-01", "WIRE-03")]  # stream requested, JSON answered
        if not request.stream and reply.kind == "sse":
            return [("C-01", "WIRE-04")]  # JSON requested, stream answered
        return []

    def _wire(self, adapter, reply, loc, session, request=None) -> None:
        mapping = getattr(adapter, "RULE_CHECK", {})
        seen = set()
        if request is not None:
            for check, rule in self._body_rules(request, reply):
                self.by_rule[check][rule] += 1
                self.note(check, loc, session, rule=rule)
        for violation in reply.grammar:
            check = mapping.get(violation.rule, "C-01")
            self.by_rule[check][violation.rule] += 1
            if (check, violation.rule) in seen:
                continue
            seen.add((check, violation.rule))
            self.note(
                check,
                loc,
                session,
                rule=violation.rule,
                **{k: v for k, v in violation.detail.items() if isinstance(v, (int, str, bool))},
            )
        if reply.kind == "sse":
            self.counts["den:streams"] += 1
            if reply.final_items is None:
                self.counts["den:no_streamed_snapshot"] += 1

    def _calls(self, delivered, request, reply, loc, session) -> None:
        calls = [i for i in delivered if i.kind == "call"]
        self.counts["den:calls"] += len(calls)
        ids = [i.call_id for i in calls]
        if any(not i for i in ids) or len(ids) != len(set(ids)):
            self.note("C-05", loc, session)
        if len(calls) > 1 and request.parallel == "single":
            self.note("C-07", loc, session, calls=len(calls))
        mode = request.tool_choice.get("mode")
        completed = reply.terminal.get("kind") == "completed"
        if mode == "none" and calls:
            self.note("C-08", loc, session, mode=mode)
        elif mode == "required" and completed and not calls:
            self.note("C-08", loc, session, mode=mode)
        elif (
            mode == "named"
            and completed
            and (not calls or any(c.name != request.tool_choice.get("name") for c in calls))
        ):
            self.note("C-08", loc, session, mode=mode)
        for call in calls:
            result = classify_call(call, request)
            cls = result["class"]
            self.call_classes[cls] += 1
            if cls in SERVING_SHAPE:
                self.note("C-06", loc, session, cls=cls, **safe_name(call.name))
            elif cls in ("undeclared", "nested_tool_called_directly"):
                self.note("M-01:undeclared", loc, session, cls=cls, **safe_name(call.name))
            elif cls == "payload_invalid":
                self.note("M-01:payload_invalid", loc, session, **safe_name(call.name),
                          **(result.get("detail") or {}))  # fmt: skip
            elif cls == "server_tool_called":
                self.note("M-01:server_tool_called", loc, session, **safe_name(call.name))

    def _content(self, delivered, reply, loc, session) -> None:
        texts = [i.text or "" for i in delivered if i.kind == "text"]
        reasoning = [i.text or "" for i in delivered if i.kind == "reasoning"]
        payloads = [i.payload or "" for i in delivered if i.kind == "call"]
        joined = "\n".join(texts + reasoning)
        if any(rx.search(joined) for rx in self.degenerate):
            self.note("M-02", loc, session)
            self.note("R-04", loc, session)
        bare = [
            m for m, quoted in marker_hits("\n".join(texts + payloads), self.markers) if not quoted
        ]
        if bare:
            self.note("M-03", loc, session, markers=len(bare))
        think = [m for m, quoted in marker_hits("\n".join(texts), self.markers)
                 if not quoted and "think" in m]  # fmt: skip
        if think:
            self.note("C-09", loc, session)
        text = "\n".join(reasoning)
        if len(text) >= REPETITIVE_MIN_CHARS:
            raw = text.encode("utf-8", "surrogatepass")
            if len(zlib.compress(raw, 6)) / len(raw) < REPETITIVE_MAX_ZLIB_RATIO:
                self.note("M-05", loc, session)
        if (reply.terminal.get("kind") == "completed" and not any(t.strip() for t in texts)
                and not payloads):  # fmt: skip
            self.note("M-06", loc, session)

    def _is_overlength(self, reply) -> bool:
        error = reply.terminal.get("error") or {}
        words = " ".join(str(x) for x in (error.get("code"), error.get("type"),
                                          reply.terminal.get("message")) if x).lower()  # fmt: skip
        return any(w in words for w in OVERLENGTH_WORDS)

    def _overlength(self, reply, loc, session) -> None:
        if not self._is_overlength(reply):
            return
        self.counts["den:overlength"] += 1
        expected = self.harness.get_path("errors.overlength")
        if not isinstance(expected, dict) or any(prof.is_todo(v) for v in expected.values()):
            self.note("C-10:profile_unknown", loc, session)
            return
        shape = expected.get("shape")
        error = reply.terminal.get("error") or {}
        if shape == "in_stream":
            ok = (reply.kind == "sse" and reply.terminal.get("kind") == "failed"
                  and error.get("code") == expected.get("code"))  # fmt: skip
        elif shape == "http_400":
            message = (reply.terminal.get("message") or "").lower()
            ok = (reply.http_status == 400
                  and (not expected.get("error_type") or error.get("type") == expected["error_type"])
                  and (not expected.get("wording") or expected["wording"].lower() in message))  # fmt: skip
        else:
            ok = False
        if not ok:
            self.note("C-10", loc, session, http_status=reply.http_status or 0,
                      terminal=reply.terminal.get("kind"))  # fmt: skip

    def _runtime(self, reply, loc, session) -> None:
        first_cut = prof.known(self.harness.get_path("timeouts.first_event_s"))
        idle_cut = prof.known(self.harness.get_path("timeouts.idle_s"))
        if reply.kind != "sse":
            return
        if reply.first_event_s is None:
            self.counts["R-01:no_timing"] += 1
        elif isinstance(first_cut, (int, float)) and reply.first_event_s > first_cut:
            self.note("R-01", loc, session, first_event_s=round(reply.first_event_s, 3))
        if not reply.timing_known:
            self.counts["R-02:no_timing"] += 1
        elif isinstance(idle_cut, (int, float)) and reply.max_gap_s is not None \
                and reply.max_gap_s > idle_cut:  # fmt: skip
            self.note("R-02", loc, session, max_gap_s=round(reply.max_gap_s, 3))

    # ------------------------------------------------------------------ pass 3

    def finish(self) -> None:
        by_thread = defaultdict(list)
        for summary in self.requests.values():
            if summary["thread"]:
                by_thread[summary["thread"]].append(summary)
        for items in by_thread.values():
            items.sort(key=lambda s: s["order"])
        orders = {thread: [s["order"] for s in items] for thread, items in by_thread.items()}
        replies = {t["trace_id"]: t for t in self.turns}
        for turn in self.turns:
            thread = turn["thread"]
            if not thread:
                continue
            nxt = None
            start = bisect.bisect_right(orders[thread], turn["order"])
            for candidate in by_thread[thread][start:]:
                if not turn["agent"] or candidate["identity"].get("agent") in (None, turn["agent"]):
                    nxt = candidate
                    break
            loc = turn["loc"]
            session = turn["session"]
            for call_id, name, digest in turn["calls"]:
                echo = self.echo_first.get((thread, call_id))
                if nxt is None:
                    self.counts["C-03:no_next_turn"] += 1
                elif echo is None:
                    self.counts["C-03:echo_missing"] += 1
                elif echo[0] < turn["order"]:
                    self.counts["C-03:call_id_seen_before"] += 1
                elif echo[2] != name or echo[3] != digest:
                    self.note("C-03", loc, session, **safe_name(name),
                              name_differs=echo[2] != name, payload_differs=echo[3] != digest)  # fmt: skip
                else:
                    self.counts["C-03:echo_equal"] += 1
                result = self.result_first.get((thread, call_id))
                if result is None or result[0] < turn["order"]:
                    self.outcome_classes["pending" if nxt is None else "unanswered"] += 1
                    if nxt is not None:
                        self.counts["I-08:unanswered"] += 1
                elif result[1] is None:
                    self.outcome_classes["unknown_profile"] += 1
                else:
                    self.outcome_classes[result[1]] += 1
                    if result[1] != "result":
                        self.note("H-01:" + result[1], loc, session, **safe_name(name))
            for item_id, digest in turn["texts"]:
                echo = self.echo_first.get((thread, item_id))
                if (
                    nxt is not None
                    and echo is not None
                    and echo[0] > turn["order"]
                    and echo[3] != digest
                ):
                    self.note("C-03:text_differs", loc, session)
            if turn["overlength"] and nxt is not None:
                if nxt["body_sha"] == turn["body_sha"]:
                    self.note("H-04", loc, session, why="identical_resend")
                elif nxt["identity"].get("compaction") or nxt["history"] < turn["history"]:
                    self.counts["H-04:compacted"] += 1
                else:
                    self.counts["H-04:unknown"] += 1
        for (thread, _), occurrences in self.bodies.items():
            if len(occurrences) < 2:
                continue
            occurrences.sort()
            for (_, first_id), (_, again_id) in zip(occurrences, occurrences[1:]):
                summary = self.requests.get(again_id) or {}
                loc = summary.get("locator", {})
                session = (summary.get("identity") or {}).get("session") or thread
                first = replies.get(first_id) or {}
                if first.get("overlength") or first.get("client_error"):
                    # The harness re-sent what we rejected: our error mapping did not lead it to
                    # its recovery path (Codex resent an HTTP 400 instead of compacting).
                    self.note("H-02", loc, session, after="our_client_error")
                elif first.get("failed"):
                    self.note("H-02:retry_after_failure", loc, session)
                else:
                    self.note("H-02:other", loc, session)

    # ------------------------------------------------------------------ output

    def report(self) -> dict:
        return {
            "counts": dict(self.counts),
            "by_rule": {k: dict(v) for k, v in self.by_rule.items()},
            "call_classes": dict(self.call_classes),
            "outcome_classes": dict(self.outcome_classes),
            "sessions": len(self.sessions),
            "sessions_with": {k: len(v) for k, v in self.session_events.items()},
            "protocols": dict(self.protocols),
            "unknown_request_fields": dict(self.unknown_fields.most_common(50)),
            "examples": dict(self.examples),
        }


def run(paths: List[str], harness: prof.Profile, model: prof.Profile,
        request_paths: Optional[List[str]] = None, build_id: Optional[str] = None) -> dict:  # fmt: skip
    pipeline = Pipeline(harness, model, build_id)
    pipeline.add_requests(list(paths) + list(request_paths or []))
    pipeline.add_responses(paths)
    pipeline.finish()
    return pipeline.report()
