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
recorder captures), after an optional raw pass that reads the model's own text
(``raw-*.jsonl``) in its own format (``native.py``, check C-13):

1. requests: one small summary per request (identity, order, tool
   declarations by key, call policy), plus the FIRST appearance per thread of
   every item the harness echoes back and of every tool result;
2. responses: each reply joined to its request and parsed by its protocol
   adapter, then read a second time by the protocol's official SDK
   (``sdk_diff.py``, check C-12); per-reply checks run here, with content in
   memory only;
3. threads: each turn against the next request of its thread (echo, call
   outcomes, resends, over-length handling).

Memory grows with requests and calls, not with history length: a history
replays the same items every turn, and only their first appearance is kept.
"""

import bisect
import json
import multiprocessing
import re
import zlib
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import native as native_mod
import profiles as prof
import proto_anthropic
import proto_chat
import proto_responses
import sdk_diff
import usage as usage_mod
from tracelib import expand_shards, header_map, iter_lines, load_json_line
from turnlib import (
    ECHO_REWRITES,
    SERVING_SHAPE,
    RequestView,
    Violation,
    classify_call,
    marker_hits,
    payload_digest,
    rewritten_digest,
    safe_name,
    sha256_text,
    text_blocks,
)

EMPTY = sha256_text("")


def _echoable(blocks) -> tuple:
    """Text and reasoning blocks a harness can echo: empty text carries nothing to echo."""
    return tuple(b for b in blocks or () if not (b[1] in (EMPTY, None) and not b[2]))


def wording_matches(wording: str, message: str) -> bool:
    """A declared error wording matches a message: its parts, split at "...", appear in order."""
    parts = [p.strip().lower() for p in wording.split("...") if p.strip()]
    text, at = message.lower(), 0
    for part in parts:
        at = text.find(part, at)
        if at < 0:
            return False
        at += len(part)
    return bool(parts)


def _digest(value) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def _reask_key(body: dict) -> Optional[tuple]:
    """A messages request's frame digest, and the block digests of its last user message.

    Two requests with the same frame differ in the last user message only. A later one whose
    blocks extend the earlier one's asked the same thing again with more said: Claude Code does
    that when it drops an empty answer. The earlier answer is then superseded (C-03).
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    users = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"]
    if not users:
        return None
    last = users[-1]
    content = messages[last].get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(blocks, list):
        return None
    frame = dict(body, messages=messages[:last] + [None] + messages[last + 1 :])
    return _digest(frame), tuple(_digest(block) for block in blocks)


ADAPTERS = {
    "/v1/responses": proto_responses,
    "/v1/messages": proto_anthropic,
    "/v1/chat/completions": proto_chat,
}
PROTOCOL_OF = {route: mod.PROTOCOL for route, mod in ADAPTERS.items()}
CLIENT_SIDE = ("client_disconnected", "client_first_event_timeout", "client_timeout")
# The recorder could not reach the server, or lost it before the reply ended
# (record_requests.py writes `upstream_error`): the body is the recorder's own error, or the
# server's bytes up to the loss. Neither is a reply to judge; R-06 counts them.
UPSTREAM_SIDE = ("upstream_error",)
# The recorder stopped before the reply ended: the request is pending at the capture's end, the
# way a request at a window's cutoff is. Nothing is judged.
CAPTURE_SIDE = ("recorder_shutdown",)
# HTTP statuses a server may answer a well-formed request with (C-16 does not judge them).
CAPACITY_STATUSES = (429,)
# Protocols whose requests carry the conversation as `messages` (re-asks, C-03).
REASK_PROTOCOLS = ("anthropic-messages", "chat-completions")
OVERLENGTH_WORDS = ("context_length_exceeded", "prompt is too long", "context length",
                    "maximum context", "too many tokens", "prompt_too_long", "too long")  # fmt: skip
COUNT_TOKENS = "/v1/messages/count_tokens"
# The engine request id inside a reply id: Chat's chatcmpl-<id>, Completions' cmpl-<id>, and an
# Anthropic message named msg_<id>_<hex>.
ENGINE_ID = re.compile(r'"id"\s*:\s*"(?:(?:chatcmpl|cmpl)-(\d+)|msg_(\d+)_[0-9a-f]+)"')
REPETITIVE_MIN_CHARS = 2000
REPETITIVE_MAX_ZLIB_RATIO = 0.10
EXAMPLES = 5


class Pipeline:
    def __init__(self, harness: prof.Profile, model: prof.Profile, build_id: Optional[str] = None):
        self.harness = harness
        self.model = model
        self.build_id = build_id
        self.outcomes = prof.outcome_rules(harness)
        # What the harness changes in our calls when it sends them back (C-03); `exact` or a
        # list of turnlib.ECHO_REWRITES. An unknown name fails here, not as a silent pass.
        rewrites = harness.get_path("echo.tool_input", "exact") if harness else "exact"
        rewrites = (
            [] if rewrites == "exact" else [rewrites] if isinstance(rewrites, str) else rewrites
        )
        unknown = [r for r in rewrites if r not in ECHO_REWRITES]
        if not isinstance(rewrites, list) or unknown:
            raise ValueError(
                f"harness profile echo.tool_input: unknown rewrite {unknown or rewrites}"
            )
        self.echo_rewrites = tuple(rewrites)
        # Rejections the harness recovers from by dropping a capability (C-16).
        retries = harness.get_path("errors.capability_retries") if harness else None
        self.capability_retries = tuple(
            (str(e.get("field") or "?"), e["wording"]) for e in retries or ()
            if isinstance(e, dict) and isinstance(e.get("wording"), str) and not prof.is_todo(e["wording"])
        )  # fmt: skip
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
        self.unknown_fields: Counter = Counter()
        self.protocols: Counter = Counter()
        self.native = prof.native_card(model)
        self.prompt_offset = prof.prompt_offset(model)
        self.count_requests: Dict[str, str] = {}  # count_tokens trace id -> counted-body digest
        self.count_results: Dict[str, list] = defaultdict(list)  # digest -> counts answered
        self.raw: Dict[str, dict] = {}
        self.raw_given = False

    # ------------------------------------------------------------------ helpers

    def note(self, key: str, locator: dict, session: Optional[str] = None, **detail) -> None:
        self.counts[key] += 1
        if session is not None:
            self.session_events[key].add(session)
        if len(self.examples[key]) < EXAMPLES:
            self.examples[key].append({**locator, **detail})

    # ------------------------------------------------------------------ raw pass

    def add_raw(self, paths: List[str], workers: int = 1) -> None:
        """Read every raw record in the model's own format, keyed by its engine ids."""
        shards = expand_shards(paths, "raw")
        self.raw_given = bool(paths)
        self.counts["den:raw_shards"] += len(shards)
        if not shards or self.native is None:
            return
        tasks = [(path, self.native) for path in shards]
        if workers > 1 and len(shards) > 1:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                parts = list(pool.imap_unordered(_raw_task, tasks))
        else:
            parts = [_raw_task(task) for task in tasks]
        # An engine id names one generation only within its server process: the same id from
        # two records is ambiguous, and an ambiguous id joins nothing (fail closed).
        ambiguous = set()
        for part, counts in parts:
            self.counts.update(counts)
            for key, value in part.items():
                if value is None or key in self.raw or key in ambiguous:
                    ambiguous.add(key)
                    continue
                self.raw[key] = value
        for key in ambiguous:
            self.raw.pop(key, None)
            self.counts["I-01:raw_ambiguous_id:" + key.split(":", 1)[0]] += 1

    # ------------------------------------------------------------------ pass 1

    def add_requests(self, paths: List[str], workers: int = 1) -> None:
        shards = expand_shards(paths, "requests")
        if workers > 1 and len(shards) > 1:
            tasks = [(i, s, self.harness, self.model, self.build_id) for i, s in enumerate(shards)]
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                for part in pool.imap_unordered(_requests_task, tasks):
                    self._merge_requests(part)
        else:
            for index, path in enumerate(shards):
                self._scan_request_shard(index, path)

    def _scan_request_shard(self, index: int, path: str) -> None:
        for line_no, raw in iter_lines(path):
            rec = load_json_line(raw)
            if rec is None:
                self.counts["inventory:unparsable_request_lines"] += 1
                continue
            if rec.get("event", "request") != "request":
                continue
            self._add_request(rec, {"file": path, "line": line_no, "trace_id": rec.get("trace_id")},
                              (index, line_no))  # fmt: skip

    def _request_state(self) -> dict:
        return {"counts": self.counts, "requests": self.requests, "auxiliary": self.auxiliary,
                "tools": self.tools, "echo_first": self.echo_first, "result_first": self.result_first,
                "bodies": dict(self.bodies), "unknown_fields": self.unknown_fields,
                "protocols": self.protocols, "count_requests": self.count_requests}  # fmt: skip

    def _merge_requests(self, part: dict) -> None:
        self.counts.update(part["counts"])
        for trace_id, summary in part["requests"].items():
            if trace_id in self.requests:
                self.counts["inventory:duplicate_trace_id"] += 1
            self.requests[trace_id] = summary
        self.auxiliary |= part["auxiliary"]
        for key, value in part["tools"].items():
            self.tools.setdefault(key, value)
        for mine, theirs in ((self.echo_first, part["echo_first"]),
                             (self.result_first, part["result_first"])):  # fmt: skip
            for key, value in theirs.items():
                if key not in mine or value[0] < mine[key][0]:
                    mine[key] = value
        for key, value in part["bodies"].items():
            self.bodies[key].extend(value)
        self.unknown_fields.update(part["unknown_fields"])
        self.protocols.update(part["protocols"])
        self.count_requests.update(part["count_requests"])

    def _add_request(self, rec: dict, locator: dict, position: tuple = (0, 0)) -> None:
        route = str(rec.get("route") or "").split("?", 1)[0]
        body = rec.get("body")
        self.counts["inventory:requests"] += 1
        if route == COUNT_TOKENS and isinstance(body, dict):
            # Not a turn: its answer is held against the input a turn with the same body billed.
            self.count_requests[rec.get("trace_id")] = usage_mod.counted_digest(body)
            self.counts["inventory:count_tokens_requests"] += 1
            return
        adapter = ADAPTERS.get(route)
        if adapter is None:
            self.counts["inventory:auxiliary_route:" + route] += 1
            self.auxiliary.add(rec.get("trace_id"))
            return
        if not isinstance(body, dict):
            self.counts["inventory:request_without_body"] += 1
            return
        cell_protocol = prof.known(self.harness.get("protocol")) if self.harness else None
        if cell_protocol and adapter.PROTOCOL != cell_protocol:
            # Traffic of another protocol in the window (probes, other clients) is not this
            # cell's: it is counted and excluded, never judged against this harness.
            self.counts["inventory:foreign_protocol:" + adapter.PROTOCOL] += 1
            self.auxiliary.add(rec.get("trace_id"))
            return
        headers = header_map(rec)
        identity = prof.extract_identity(self.harness, headers, body, rec)
        view = adapter.parse_request(body, headers)
        # Arrival time first; the shard index and line make the order total and independent of
        # which worker read the shard.
        order = (str(rec.get("recorded_at") or ""),) + tuple(position)
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
            "choices": view.choices,
            "counted": usage_mod.counted_digest(body)
            if adapter.PROTOCOL == "anthropic-messages"
            else None,
            # Server-side clearing the client asked for: the rendered prompt may shrink (USG-08).
            "context_management": bool(body.get("context_management")),
            "build_id": rec.get("build_id"),
            "history": view.history_messages,
            "last_digest": view.last_message_digest,
            "body_sha": rec.get("body_sha256") or _digest(body),
            "reask": _reask_key(body) if adapter.PROTOCOL in REASK_PROTOCOLS else None,
        }
        trace_id = rec.get("trace_id")
        if trace_id in self.requests:
            self.counts["inventory:duplicate_trace_id"] += 1
        self.requests[trace_id] = summary
        if thread:
            self.bodies[(thread, summary["body_sha"])].append((order, trace_id))
            for item_id, echo in view.echo.items():
                key = (thread, item_id)
                known = self.echo_first.get(key)
                if known is not None and known[0] <= order:
                    continue  # histories replay every item each turn: digest the first only
                digest = (payload_digest(echo.get("payload"), echo.get("call_kind"))
                          if echo["kind"] == "call" else
                          sha256_text(echo.get("text") or "") if echo["kind"] == "text" else
                          echo.get("carrier"))  # fmt: skip
                # Also digested under the harness's declared rewrites (echo.tool_input).
                loose = (self._loose_digest(echo.get("payload"), echo.get("call_kind"),
                                            view.tools.get(echo.get("name") or ""), False)
                         if echo["kind"] == "call" else None)  # fmt: skip
                self.echo_first[key] = (order, echo["kind"], echo.get("name"), digest, loose)
            for position, previous, blocks in view.assistant_turns:
                # Keyed by the message it answered as well: a history that was rewritten does not
                # match, and only the first appearance is kept, as for every echoed item.
                key = (thread, "pos", position, previous)
                known = self.echo_first.get(key)
                if known is None or order < known[0]:
                    self.echo_first[key] = (order, "texts", None, _echoable(blocks))
            for call_id, text, is_error in view.tool_results:
                key = (thread, call_id)
                if key in self.result_first and self.result_first[key][0] <= order:
                    continue
                cls = self.outcomes.classify(text, is_error) if self.outcomes else None
                self.result_first[key] = (order, cls)

    # ------------------------------------------------------------------ pass 2

    def add_responses(self, paths: List[str], workers: int = 1) -> None:
        global _SHARED
        shards = expand_shards(paths, "responses")
        if workers > 1 and len(shards) > 1:
            _SHARED = self  # read by forked workers; copy-on-write, never sent through a pipe
            try:
                with multiprocessing.get_context("fork").Pool(workers) as pool:
                    for part in pool.imap_unordered(_responses_task, shards):
                        self._merge_replies(part)
            finally:
                _SHARED = None
        else:
            for path in shards:
                self._scan_response_shard(path)

    def _scan_response_shard(self, path: str) -> None:
        for line_no, raw in iter_lines(path):
            rec = load_json_line(raw)
            if rec is None:
                self.counts["inventory:unparsable_response_lines"] += 1
                continue
            if rec.get("event", "response") != "response":
                continue
            self._add_response(
                rec, {"file": path, "line": line_no, "trace_id": rec.get("trace_id")}
            )

    def _reset_replies(self) -> None:
        self.counts = Counter()
        self.examples = defaultdict(list)
        self.by_rule = defaultdict(Counter)
        self.call_classes = Counter()
        self.session_events = defaultdict(set)
        self.sessions = set()
        self.turns = []
        self.count_results = defaultdict(list)

    def _reply_state(self) -> dict:
        return {"counts": self.counts, "examples": dict(self.examples),
                "by_rule": {k: dict(v) for k, v in self.by_rule.items()},
                "call_classes": self.call_classes, "session_events": dict(self.session_events),
                "sessions": self.sessions, "turns": self.turns,
                "count_results": dict(self.count_results)}  # fmt: skip

    def _merge_replies(self, part: dict) -> None:
        self.counts.update(part["counts"])
        for key, examples in part["examples"].items():
            room = EXAMPLES - len(self.examples[key])
            if room > 0:
                self.examples[key].extend(examples[:room])
        for check, rules in part["by_rule"].items():
            self.by_rule[check].update(rules)
        self.call_classes.update(part["call_classes"])
        for key, sessions in part["session_events"].items():
            self.session_events[key] |= sessions
        self.sessions |= part["sessions"]
        self.turns.extend(part["turns"])
        for digest, values in part["count_results"].items():
            self.count_results[digest].extend(values)

    def _request_view(self, summary: dict) -> RequestView:
        tools, nested = self.tools.get(summary["tools_key"], ({}, {}))
        view = RequestView(protocol=summary["protocol"], route=summary["route"])
        view.stream = summary["stream"]
        view.tools, view.nested = tools, nested
        view.tool_choice = summary["tool_choice"]
        view.parallel = summary["parallel"]
        view.include_usage = summary["include_usage"]
        view.choices = summary.get("choices", 1)
        return view

    def _add_response(self, rec: dict, loc: dict) -> None:
        self.counts["inventory:responses"] += 1
        summary = self.requests.get(rec.get("trace_id"))
        if summary is None and rec.get("trace_id") in self.count_requests:
            body = (rec.get("response") or {}).get("body")
            value = body.get("input_tokens") if isinstance(body, dict) else None
            if rec.get("http_status") in (None, 200) and isinstance(value, int):
                self.count_results[self.count_requests[rec["trace_id"]]].append(value)
            else:
                self.counts["C-14:count_tokens_unanswered"] += 1
            return
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
        # A client that stopped reading before the terminal event left a stream that is
        # incomplete by construction: its wire, call and content rules are not judged, and H-03
        # attributes the cut. A client that closed after the terminal event cut nothing.
        client_cut = reply.status in CLIENT_SIDE and reply.terminal.get("kind") == "none"
        delivered = reply.delivered()
        if reply.status in CLIENT_SIDE and not client_cut:
            self.counts["H-03:closed_after_terminal"] += 1
        if client_cut:
            self.note("H-03:" + str(reply.status), loc, session)
            self.counts["den:client_cut_replies"] += 1
        elif reply.status in UPSTREAM_SIDE:
            # Not the server's reply, or not all of it: R-06 counts it and nothing else judges it.
            self.note("R-06", loc, session, http_status=reply.http_status or 0)
            self.counts["den:upstream_cut_replies"] += 1
        elif reply.status in CAPTURE_SIDE:
            self.counts["inventory:pending_at_capture_end"] += 1
        else:
            self._wire(adapter, reply, loc, session, request)
            self._sdk(summary["protocol"], rec, reply, loc, session)
            self._raw_view(rec, reply, delivered, request, loc, session)
            self._usage(summary, rec, reply, delivered, request, loc, session)
            self._calls(delivered, request, reply, loc, session)
            self._content(delivered, reply, loc, session)
            self._overlength(reply, loc, session)
            self._rejected(reply, loc, session)
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
            "calls": [(i.call_id, i.name, payload_digest(i.payload, i.call_kind),
                       self._loose_digest(i.payload, i.call_kind, request.tools.get(i.name or ""), True))
                      for i in delivered if i.kind == "call" and i.call_id],
            "texts": [(i.id, sha256_text(i.text or "")) for i in delivered
                      if i.kind == "text" and i.id and summary["protocol"] == "responses"],
            "text_blocks": (_echoable(text_blocks(delivered))
                            if summary["protocol"] != "responses" else ()),
            "body_sha": summary["body_sha"],
            "reask": summary.get("reask"),
            "history": summary["history"],
            "last_digest": summary.get("last_digest"),
            "counted": summary.get("counted"),
            "billed_input": usage_mod.input_tokens(summary["protocol"], reply.usage),
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
        unchecked = sorted(reply.extras.get("unchecked") or ())
        if unchecked:
            self.by_rule["C-11"]["WIRE-05"] += 1
            self.note("C-11", loc, session, elements=",".join(unchecked)[:200])
            for element in unchecked:
                self.counts["C-11:element:" + element] += 1
        if reply.kind == "sse":
            self.counts["den:streams"] += 1
            if reply.final_items is None:
                self.counts["den:no_streamed_snapshot"] += 1

    def _sdk(self, protocol, rec, reply, loc, session) -> None:
        """C-12: the official SDK's reading of the same bytes agrees with the adapter's."""
        if sdk_diff.unavailable(protocol):
            self.counts["C-12:sdk_unavailable:" + protocol] += 1
            return
        found = sdk_diff.compare(protocol, rec, reply)
        if found is None:
            self.counts["C-12:not_read"] += 1
            return
        self.counts["den:sdk_read"] += 1
        for violation in found:
            self.by_rule["C-12"][violation.rule] += 1
            what = violation.detail.get("fields") or violation.detail.get("error")
            self.counts[f"C-12:{violation.rule}:{what}"] += 1
        if found:
            first = found[0]
            self.note("C-12", loc, session, rules=",".join(v.rule for v in found),
                      **{k: v for k, v in first.detail.items() if isinstance(v, (int, str, bool))})  # fmt: skip

    def _raw_view(self, rec, reply, delivered, request, loc, session) -> None:
        """C-13: what was delivered is what the model wrote (its raw record, read natively)."""
        if not self.raw_given or (isinstance(reply.http_status, int) and reply.http_status >= 400):
            return
        if self.native is None:
            self.counts["C-13:no_native_reader"] += 1
            return
        keys = []
        if rec.get("disagg_request_id") not in (None, ""):
            keys.append("d:" + str(rec["disagg_request_id"]))
        body = (rec.get("response") or {}).get("body")
        match = ENGINE_ID.search(body if isinstance(body, str) else json.dumps(body or {})[:400])
        if match:
            keys.append("r:" + (match.group(1) or match.group(2)))
        raw = next((self.raw[k] for k in keys if k in self.raw), None)
        self.counts["den:raw_candidates"] += 1
        if raw is None:
            self.counts["C-13:unjoined"] += 1
            return
        self.counts["den:raw_joined"] += 1
        found, notes = native_mod.compare(self.native, raw, delivered,
                                          reply.terminal.get("kind") == "completed",
                                          request.parallel == "single")  # fmt: skip
        reported = usage_mod.output_tokens(request.protocol, reply.usage)
        if raw.get("output_tokens") is not None and reported is not None \
                and reply.terminal.get("kind") == "completed":  # fmt: skip
            self.counts["den:usage_vs_engine_output"] += 1
            if reported != raw["output_tokens"]:
                self._usage_found(
                    [Violation("USG-06", {"why": "output_vs_engine_count"})], loc, session
                )
        for item in notes:
            self.counts["C-13:" + item] += 1
        anomalies = [n.split(":", 1)[1] for n in notes if n.startswith("anomaly:")]
        if anomalies:
            self.note("M-07", loc, session, anomalies=",".join(anomalies)[:200])
        for violation in found:
            self.by_rule["C-13"][violation.rule] += 1
            self.counts[f"C-13:{violation.rule}:{violation.detail.get('why')}"] += 1
        if found:
            self.note("C-13", loc, session, rules=",".join(v.rule for v in found),
                      why=str(found[0].detail.get("why")))  # fmt: skip

    def _usage(self, summary, rec, reply, delivered, request, loc, session) -> None:
        """C-14: usage present, typed and consistent (USG-01 .. 04), and true to the trace (USG-07)."""
        protocol = summary["protocol"]
        found = usage_mod.rules(protocol, reply, delivered, request.include_usage)
        prompt = rec.get("prompt_tokens") if isinstance(rec.get("prompt_tokens"), dict) else {}
        billed = usage_mod.input_tokens(protocol, reply.usage)
        if isinstance(prompt.get("count"), int) and billed is not None:
            self.counts["den:usage_vs_trace_prompt"] += 1
            if billed != prompt["count"] - self.prompt_offset:
                found.append(Violation("USG-07", {"why": "input_vs_server_prompt_count"}))
        self._usage_found(found, loc, session)

    def _usage_found(self, found, loc, session) -> None:
        for violation in found:
            self.by_rule["C-14"][violation.rule] += 1
            self.counts[f"C-14:{violation.rule}:{violation.detail.get('why')}"] += 1
        if found:
            self.note("C-14", loc, session, rules=",".join(v.rule for v in found),
                      why=str(found[0].detail.get("why")))  # fmt: skip

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

    def _rejected(self, reply, loc, session) -> None:
        """C-16: the server accepted what the harness sent.

        A 4xx other than over-length (C-10) or capacity fails, unless the harness profile
        declares it a capability the harness drops and retries without (`errors.capability_retries`,
        matched by wording); a declared one is flagged, so the round trip it costs stays visible.
        """
        status = reply.http_status
        if not isinstance(status, int) or not 400 <= status < 500 or status in CAPACITY_STATUSES:
            return
        if self._is_overlength(reply):
            return
        message = str(reply.terminal.get("message") or "")
        for field, wording in self.capability_retries:
            if wording_matches(wording, message):
                self.note("C-16:declared", loc, session, field=field[:80], http_status=status)
                self.counts["C-16:declared_field:" + field[:80]] += 1
                return
        error = reply.terminal.get("error") or {}
        self.note("C-16", loc, session, http_status=status, error_type=str(error.get("type") or ""))

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
        # I-01: a request with no response record at all. The capture lost it, or its answer lies
        # outside the window; either way the evidence is partial (inventory is BLOCKED).
        answered = {t["trace_id"] for t in self.turns}
        for trace_id, summary in self.requests.items():
            if trace_id not in answered:
                session = (summary.get("identity") or {}).get("session") or summary["thread"]
                self.note("I-01:request_without_response", summary["locator"], session)
        for turn in self.turns:
            # USG-05: count_tokens over the same counted body answers the input this turn billed.
            counted = self.count_results.get(turn.get("counted")) if turn.get("counted") else None
            if counted and turn.get("billed_input") is not None:
                self.counts["den:count_tokens_pairs"] += 1
                if any(value != turn["billed_input"] for value in counted):
                    self._usage_found([Violation("USG-05", {"why": "count_tokens_vs_billed_input"})],
                                      turn["loc"], turn["session"])  # fmt: skip
        by_thread = defaultdict(list)
        for summary in self.requests.values():
            if summary["thread"]:
                by_thread[summary["thread"]].append(summary)
        for items in by_thread.values():
            items.sort(key=lambda s: s["order"])
        orders = {thread: [s["order"] for s in items] for thread, items in by_thread.items()}
        replies = {t["trace_id"]: t for t in self.turns}
        # A request answered more than once (the harness re-sent it after a cut, or a proxy
        # retried it after the network failed), or asked again with more appended to its last
        # user message: only the last completed answer can be the one the client went on with.
        # The earlier answers are superseded, not echoed.
        completed = defaultdict(list)
        reasks = defaultdict(list)
        for t in self.turns:
            if t["thread"] and t["terminal"] == "completed":
                completed[(t["thread"], t["body_sha"])].append(t["order"])
                if t.get("reask"):
                    reasks[(t["thread"], t["reask"][0])].append((t["order"], tuple(t["reask"][1])))
        for turn in self.turns:
            thread = turn["thread"]
            if not thread:
                continue
            again = any(o > turn["order"] for o in completed.get((thread, turn["body_sha"]), ()))
            reasked = not again and self._reasked(turn, reasks)
            if again or reasked:
                self.counts["C-03:superseded_answers"] += 1
                self.counts["C-03:echo_superseded"] += len(turn["calls"])
                if reasked:
                    self.note("C-03:reasked_answers", turn["loc"], turn["session"])
                continue
            nxt = None
            start = bisect.bisect_right(orders[thread], turn["order"])
            for candidate in by_thread[thread][start:]:
                if not turn["agent"] or candidate["identity"].get("agent") in (None, turn["agent"]):
                    nxt = candidate
                    break
            loc = turn["loc"]
            session = turn["session"]
            echoed = bool(turn["calls"]) and nxt is not None and all(
                (e := self.echo_first.get((thread, call[0]))) is not None and e[0] > turn["order"]
                and self._call_echo(e, call) is not None for call in turn["calls"])  # fmt: skip
            following = (
                replies.get((nxt or {}).get("locator", {}).get("trace_id")) if echoed else None
            )
            if following and following.get("billed_input") is not None \
                    and turn.get("billed_input") is not None:  # fmt: skip
                # USG-08: the next prompt holds this turn's calls, so it can only have grown.
                self.counts["den:next_input_pairs"] += 1
                if following["billed_input"] < turn["billed_input"]:
                    if nxt.get("context_management"):
                        self.counts["C-14:next_input_shrank_by_context_management"] += 1
                    else:
                        self._usage_found(
                            [Violation("USG-08", {"why": "next_input_shrank"})], loc, session
                        )
            for call in turn["calls"]:
                call_id, name, digest = call[:3]
                echo = self.echo_first.get((thread, call_id))
                if nxt is None:
                    self.counts["C-03:no_next_turn"] += 1
                elif echo is None:
                    self.counts["C-03:echo_missing"] += 1
                elif echo[0] < turn["order"]:
                    self.counts["C-03:call_id_seen_before"] += 1
                elif (how := self._call_echo(echo, call)) is None:
                    self.note("C-03", loc, session, **safe_name(name),
                              name_differs=echo[2] != name, payload_differs=echo[3] != digest)  # fmt: skip
                else:
                    self.counts["C-03:" + how] += 1
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
            if nxt is not None:
                self._text_echo(turn, thread, loc, session)
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

    @staticmethod
    def _reasked(turn: dict, reasks: dict) -> bool:
        """A later answered request of the thread asked this one again with more appended."""
        if not turn.get("reask"):
            return False
        frame, blocks = turn["reask"][0], tuple(turn["reask"][1])
        return any(order > turn["order"] and len(later) > len(blocks)
                   and later[:len(blocks)] == blocks
                   for order, later in reasks.get((turn["thread"], frame), ()))  # fmt: skip

    def _loose_digest(self, payload, call_kind, tool, delivered: bool) -> Optional[str]:
        """A function call's digest under the harness's declared echo rewrites, else None.

        Default-valued arguments are left out on both sides. A cut ``cd <dir> &&`` is cut
        from what we delivered only, so an echo that changed the directory still differs.
        """
        if not self.echo_rewrites or call_kind != "function":
            return None
        rewrites = [r for r in self.echo_rewrites if delivered or r != "cd_cwd_prefix"]
        return rewritten_digest(payload, tool.schema if tool is not None else None, rewrites)

    @staticmethod
    def _call_echo(echo: tuple, call: tuple) -> Optional[str]:
        """How the harness echoed one of our calls, or None when it differs.

        ``echo_equal``, or ``echo_rewritten``: equal under the rewrites its profile
        declares in ``echo.tool_input``.
        """
        name, digest = call[1], call[2]
        if echo[2] != name:
            return None
        if echo[3] == digest:
            return "echo_equal"
        loose = call[3] if len(call) > 3 else None
        if loose is not None and len(echo) > 4 and echo[4] == loose:
            return "echo_rewritten"
        return None

    def _text_echo(self, turn, thread, loc, session) -> None:
        """C-03 for texts: what the harness sent back is what we delivered (harness ``echo``)."""
        policy = self.harness.get_path("echo") if self.harness else None
        policy = policy if isinstance(policy, dict) else {}
        kinds = [k for k, key in (("text", "text"), ("reasoning", "reasoning"))
                 if policy.get(key) == "exact"]  # fmt: skip
        unknown = [k for k in ("text", "reasoning") if prof.is_todo(policy.get(k))]
        if turn["texts"]:  # Responses: the echo carries our item ids
            for item_id, digest in turn["texts"]:
                echo = self.echo_first.get((thread, item_id))
                if echo is None or echo[0] <= turn["order"]:
                    self.counts["C-03:text_echo_missing"] += 1
                elif "text" not in kinds:
                    self.counts["C-03:text_echo_unchecked"] += "text" in unknown
                elif echo[3] != digest:
                    self.note("C-03", loc, session, field="text")
                else:
                    self.counts["C-03:text_echo_equal"] += 1
            return
        delivered = turn["text_blocks"]
        if not delivered:
            return
        echo = self.echo_first.get((thread, "pos", turn["history"], turn.get("last_digest")))
        if echo is None or echo[0] <= turn["order"]:
            self.counts["C-03:text_echo_missing"] += 1
            return
        echoed = echo[3]
        if any(k in unknown for k, _, _ in delivered):
            self.counts["C-03:text_echo_unchecked"] += 1
        differs = [k for k in kinds
                   if [b for b in delivered if b[0] == k] != [b for b in echoed if b[0] == k]]  # fmt: skip
        if differs:
            self.note("C-03", loc, session, field=",".join(differs))
        elif kinds:
            self.counts["C-03:text_echo_equal"] += 1

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


_SHARED: Optional["Pipeline"] = None


def _requests_task(args) -> dict:
    index, path, harness, model, build_id = args
    pipeline = Pipeline(harness, model, build_id)
    pipeline._scan_request_shard(index, path)
    return pipeline._request_state()


def _raw_task(args):
    path, card = args
    part, counts = {}, Counter()
    for _, raw in iter_lines(path):
        rec = load_json_line(raw)
        if rec is None:
            counts["inventory:unparsable_raw_lines"] += 1
            continue
        counts["den:raw_records"] += 1
        keys = []
        if rec.get("disagg_request_id") not in (None, ""):
            keys.append("d:" + str(rec["disagg_request_id"]))
        if rec.get("request_id") not in (None, ""):
            keys.append("r:" + str(rec["request_id"]))
        if not keys:
            counts["I-01:raw_without_id"] += 1
            continue
        reading = native_mod.summary(card, rec)
        count = rec.get("num_output_tokens")
        reading["output_tokens"] = (
            count
            if isinstance(count, int)
            else (len(rec["token_ids"]) if isinstance(rec.get("token_ids"), list) else None)
        )
        for key in keys:
            # A second record under one id within a shard is just as ambiguous as across shards.
            part[key] = None if key in part else reading
    return part, counts


def _responses_task(path: str) -> dict:
    pipeline = _SHARED
    pipeline._reset_replies()
    pipeline._scan_response_shard(path)
    return pipeline._reply_state()


def run(paths: List[str], harness: prof.Profile, model: prof.Profile,
        request_paths: Optional[List[str]] = None, build_id: Optional[str] = None,
        workers: int = 1, raw_paths: Optional[List[str]] = None) -> dict:  # fmt: skip
    pipeline = Pipeline(harness, model, build_id)
    pipeline.add_raw(list(raw_paths or []), workers)
    pipeline.add_requests(list(paths) + list(request_paths or []), workers)
    pipeline.add_responses(paths, workers)
    pipeline.finish()
    return pipeline.report()
