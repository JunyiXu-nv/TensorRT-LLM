"""Tests for conversation routing in the gateway.

The routing decision has no network in it, so it is tested directly rather than
through a live proxy. Every payload here is shaped after traffic captured from
Codex and n3 against this deployment rather than invented -- the nested
`client_metadata.session_id` in particular is the reason a scan of top-level
body keys concludes that chat completions carries no conversation identity.
"""

import argparse
import json
import time
import unittest

import gateway


def headers(**kwargs):
    return [(name.replace("_", "-"), value) for name, value in kwargs.items()]


def body(payload):
    return json.dumps(payload).encode()


def tally(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


class ConversationKey(unittest.TestCase):
    def test_responses_uses_prompt_cache_key(self):
        key = gateway.conversation_key(
            headers(content_type="application/json"),
            body({
                "model": "glm",
                "prompt_cache_key": "01a074ca-1d86-7870-ba89-416001960f00",
                "input": "hello",
            }),
        )
        self.assertEqual(key, "prompt_cache_key:01a074ca-1d86-7870-ba89-416001960f00")

    def test_chat_completions_uses_nested_session_id(self):
        key = gateway.conversation_key(
            headers(content_type="application/json"),
            body({
                "model": "glm",
                "messages": [{"role": "user", "content": "hi"}],
                "client_metadata": {
                    "session_id": "01a06b54-b709-70e1-9139-7b14562f9792",
                    "thread_id": "01a06b54-b709-70e1-9139-7b14562f9792",
                    "turn_id": "01a06b54-b8ee-7132-a476-d2a6e2000000",
                },
            }),
        )
        self.assertEqual(
            key, "client_metadata.session_id:01a06b54-b709-70e1-9139-7b14562f9792")

    def test_turn_id_is_never_the_key(self):
        """turn_id changes every turn; pinning on it would defeat the point."""
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "client_metadata": {"session_id": "S", "turn_id": "T1"},
        }
        first = gateway.conversation_key(headers(), body(payload))
        payload["client_metadata"]["turn_id"] = "T2"
        payload["messages"].append({"role": "assistant", "content": "there"})
        self.assertEqual(first, gateway.conversation_key(headers(), body(payload)))

    def test_header_wins_over_body(self):
        key = gateway.conversation_key(
            headers(x_conversation_id="explicit"),
            body({"prompt_cache_key": "ignored"}),
        )
        self.assertEqual(key, "hdr:explicit")

    def test_prefix_hash_is_stable_as_the_conversation_grows(self):
        payload = {
            "messages": [
                {"role": "system", "content": "You are a kernel engineer."},
                {"role": "user", "content": "Optimise problem 90794."},
            ]
        }
        first = gateway.conversation_key(headers(), body(payload))
        self.assertTrue(first.startswith("prefix:"))
        for turn in range(5):
            payload["messages"].append({"role": "assistant", "content": "turn %d" % turn})
            payload["messages"].append({"role": "user", "content": "next %d" % turn})
            self.assertEqual(first, gateway.conversation_key(headers(), body(payload)))

    def test_prefix_hash_separates_different_openings(self):
        def key_for(opening):
            return gateway.conversation_key(headers(), body({
                "messages": [
                    {"role": "system", "content": "same prompt"},
                    {"role": "user", "content": opening},
                ]}))

        self.assertNotEqual(key_for("problem 90794"), key_for("problem 90795"))

    def test_instructions_participate_in_the_prefix(self):
        """Responses puts the system prompt in `instructions`, not in the input."""
        def key_for(instructions):
            return gateway.conversation_key(headers(), body({
                "instructions": instructions, "input": "same opening"}))

        self.assertNotEqual(key_for("be terse"), key_for("be verbose"))

    def test_malformed_bodies_do_not_raise(self):
        for payload in (b"not json", b"", None, b"[1,2,3]", b'"a string"',
                        body({"model": "glm"}), body({"messages": []})):
            self.assertIsNone(gateway.conversation_key(headers(), payload))

    def test_blank_header_falls_through_to_the_body(self):
        key = gateway.conversation_key(
            headers(x_conversation_id="   "), body({"prompt_cache_key": "real"}))
        self.assertEqual(key, "prompt_cache_key:real")


class LogRendering(unittest.TestCase):
    """The log line has to tell two conversations apart, which is the whole job."""

    def test_two_keys_from_one_client_do_not_render_identically(self):
        prefix = "client_metadata.session_id:"
        first = gateway.short_convo(prefix + "01a06b54-b709-70e1-9139-7b14562f9792")
        second = gateway.short_convo(prefix + "01a06b54-b709-70e1-9139-000000000000")
        self.assertNotEqual(first, second)

    def test_short_keys_are_left_alone(self):
        self.assertEqual("hdr:abc", gateway.short_convo("hdr:abc"))

    def test_none_renders_as_a_dash(self):
        self.assertEqual("-", gateway.short_convo(None))

    def test_the_source_survives_shortening(self):
        rendered = gateway.short_convo("client_metadata.session_id:" + "x" * 40)
        self.assertTrue(rendered.startswith("client_metadata."))


class Routing(unittest.TestCase):
    def setUp(self):
        self.router = gateway.Router(ttl=1800, capacity=100)

    def test_same_conversation_returns_to_the_same_backend(self):
        loads = {"a": (0, 0), "b": (0, 0)}
        first = self.router.route("convo:1", loads, {"a", "b"})
        for _ in range(10):
            self.assertEqual(first, self.router.route("convo:1", loads, {"a", "b"}))
        self.assertEqual(self.router.hits, 10)

    def test_new_conversations_spread_evenly(self):
        placed = []
        for index in range(6):
            counts = self.router.counts({"a": None, "b": None, "c": None})
            loads = {job: (counts[job], 0) for job in ("a", "b", "c")}
            placed.append(self.router.route("convo:%d" % index, loads, {"a", "b", "c"}))
        self.assertEqual(sorted(tally(placed).values()), [2, 2, 2])

    def test_a_dead_backend_rehomes_rather_than_failing(self):
        first = self.router.route("convo:1", {"a": (0, 0), "b": (0, 0)}, {"a", "b"})
        survivor = "b" if first == "a" else "a"
        self.assertEqual(survivor,
                         self.router.route("convo:1", {survivor: (0, 0)}, {survivor}))
        self.assertEqual(self.router.rehomed, 1)

    def test_a_pin_survives_its_backend_leaving_the_accepting_set(self):
        """Draining must not evict conversations that are already running."""
        self.assertEqual("a", self.router.route("convo:1", {"a": (0, 0)}, {"a"}))
        # `a` now drains: still serving, no longer accepting.
        self.assertEqual("a", self.router.route("convo:1", {"b": (0, 0)}, {"a", "b"}))
        # A new conversation goes to the one that is accepting.
        self.assertEqual("b", self.router.route("convo:2", {"b": (0, 0)}, {"a", "b"}))

    def test_no_backend_returns_none(self):
        self.assertIsNone(self.router.route("convo:1", {}, set()))
        self.assertIsNone(self.router.route(None, {}, set()))

    def test_pinned_conversation_with_nothing_accepting_still_gets_its_backend(self):
        """Every backend draining: an in-flight conversation must not 503."""
        self.assertEqual("a", self.router.route("convo:1", {"a": (0, 0)}, {"a"}))
        self.assertEqual("a", self.router.route("convo:1", {}, {"a"}))

    def test_unidentified_requests_balance_without_pinning(self):
        self.assertEqual("b", self.router.route(None, {"a": (0, 5), "b": (0, 0)}, {"a", "b"}))
        self.assertEqual(0, len(self.router.pins))

    def test_conversation_count_outranks_inflight(self):
        """An idle agent still occupies a backend, and must still count."""
        self.assertEqual("b", self.router.route("new", {"a": (5, 0), "b": (1, 9)}, {"a", "b"}))

    def test_pins_expire(self):
        self.router.route("convo:1", {"a": (0, 0)}, {"a"}, now=1000)
        self.router.route("convo:2", {"a": (0, 0)}, {"a"}, now=1000 + 1801)
        self.assertNotIn("convo:1", self.router.pins)

    def test_activity_postpones_expiry(self):
        self.router.route("convo:1", {"a": (0, 0)}, {"a"}, now=1000)
        for step in range(1, 6):
            self.router.route("convo:1", {"a": (0, 0)}, {"a"}, now=1000 + step * 1000)
        self.assertIn("convo:1", self.router.pins)

    def test_capacity_drops_the_least_recently_used(self):
        router = gateway.Router(ttl=1800, capacity=3)
        for index in range(5):
            router.route("convo:%d" % index, {"a": (0, 0)}, {"a"})
        self.assertEqual(3, len(router.pins))
        self.assertNotIn("convo:0", router.pins)
        self.assertIn("convo:4", router.pins)


class AcceptingSet(unittest.TestCase):
    """Fleet.accepting() is what replaces the single-active election."""

    def fleet(self, margin=1800, **backends):
        args = argparse.Namespace(
            new_conversation_margin=margin, sticky_ttl=1800, sticky_capacity=100,
            users="/nonexistent", fleet_dir="/nonexistent")
        fleet = gateway.Fleet(args)
        now = time.time()
        for job_id, (healthy, remaining) in backends.items():
            backend = gateway.Backend({
                "job_id": job_id, "url": "http://host:8000",
                "end_time": now + remaining, "heartbeat": now})
            backend.healthy = healthy
            fleet.backends[job_id] = backend
        return fleet

    def test_healthy_backends_all_accept(self):
        fleet = self.fleet(a=(True, 7200), b=(True, 7200))
        self.assertEqual({"a", "b"}, set(fleet.accepting()))
        self.assertEqual({"a", "b"}, fleet.serving())

    def test_unhealthy_backends_neither_serve_nor_accept(self):
        fleet = self.fleet(a=(True, 7200), b=(False, 7200))
        self.assertEqual({"a"}, set(fleet.accepting()))
        self.assertEqual({"a"}, fleet.serving())

    def test_a_backend_near_its_end_stops_taking_new_conversations(self):
        fleet = self.fleet(a=(True, 7200), b=(True, 600))
        self.assertEqual({"a"}, set(fleet.accepting()))
        # ...but it keeps serving the ones it already has.
        self.assertEqual({"a", "b"}, fleet.serving())

    def test_draining_and_superseded_are_excluded(self):
        fleet = self.fleet(a=(True, 7200), b=(True, 7200), c=(True, 7200))
        fleet.draining["b"] = 0
        fleet.superseded.add("c")
        self.assertEqual({"a"}, set(fleet.accepting()))
        self.assertEqual({"a", "b", "c"}, fleet.serving())

    def test_all_backends_ageing_out_still_offers_the_longest_lived(self):
        """Refusing every new conversation would be worse than a short one."""
        fleet = self.fleet(a=(True, 300), b=(True, 900))
        self.assertEqual({"b"}, set(fleet.accepting()))

    def test_no_healthy_backend_accepts_nothing(self):
        fleet = self.fleet(a=(False, 7200))
        self.assertEqual({}, fleet.accepting())
        self.assertEqual(set(), fleet.serving())


if __name__ == "__main__":
    unittest.main(verbosity=2)
