"""Tests for conversation routing in the gateway.

The routing decision has no network in it, so it is tested directly rather than
through a live proxy. Every payload here is shaped after traffic captured from
Codex and n3 against this deployment rather than invented -- the nested
`client_metadata.session_id` in particular is the reason a scan of top-level
body keys concludes that chat completions carries no conversation identity.
"""

import asyncio
import argparse
import json
import os
import shutil
import tempfile
import textwrap
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
            users="/nonexistent", fleet_dir="/nonexistent",
            route_policy="least_conversations", router_state=None, key_sources=None)
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

    def test_draining_is_excluded(self):
        fleet = self.fleet(a=(True, 7200), b=(True, 7200))
        fleet.draining["b"] = 0
        self.assertEqual({"a"}, set(fleet.accepting()))
        self.assertEqual({"a", "b"}, fleet.serving())

    def test_superseded_backends_still_accept(self):
        """Every instance but the longest-lived one is superseded by definition.

        Excluding them collapsed a fleet of N healthy instances into one
        accepting instance the moment a successor was elected -- so the routing
        was multi-instance in name only. Reproduced against a live gateway:
        `accepting` went from ['OLD'] to ['NEW'] the instant NEW won the
        election, with OLD still healthy and still serving.
        """
        fleet = self.fleet(old=(True, 7200), new=(True, 93600))
        fleet.superseded.add("old")
        self.assertEqual({"old", "new"}, set(fleet.accepting()))

    def test_a_whole_fleet_keeps_accepting_after_an_election(self):
        fleet = self.fleet(a=(True, 7200), b=(True, 7200), c=(True, 90000))
        # c has the longest life, so an election supersedes both others.
        fleet.superseded.update({"a", "b"})
        self.assertEqual({"a", "b", "c"}, set(fleet.accepting()))

    def test_all_backends_ageing_out_still_offers_the_longest_lived(self):
        """Refusing every new conversation would be worse than a short one."""
        fleet = self.fleet(a=(True, 300), b=(True, 900))
        self.assertEqual({"b"}, set(fleet.accepting()))

    def test_no_healthy_backend_accepts_nothing(self):
        fleet = self.fleet(a=(False, 7200))
        self.assertEqual({}, fleet.accepting())
        self.assertEqual(set(), fleet.serving())


class Policies(unittest.TestCase):
    """Placement policy is selectable, and each one means what it says."""

    @staticmethod
    def router(policy):
        return gateway.Router(ttl=1800, capacity=100, policy=policy)

    # (conversations, inflight, remaining_seconds)
    LOAD = {"a": (5, 0, 7200), "b": (1, 9, 600)}

    def test_least_conversations_ignores_inflight(self):
        self.assertEqual("b", self.router("least_conversations").route("k", self.LOAD, {"a", "b"}))

    def test_least_inflight_ignores_conversations(self):
        self.assertEqual("a", self.router("least_inflight").route("k", self.LOAD, {"a", "b"}))

    def test_longest_lived_picks_the_most_remaining_time(self):
        self.assertEqual("a", self.router("longest_lived").route("k", self.LOAD, {"a", "b"}))

    def test_round_robin_cycles(self):
        router = self.router("round_robin")
        placed = [router.route("k%d" % i, {"a": (0, 0, 0), "b": (0, 0, 0)}, {"a", "b"})
                  for i in range(4)]
        self.assertEqual(["a", "b", "a", "b"], placed)

    def test_policy_can_change_between_calls(self):
        router = self.router("least_conversations")
        self.assertEqual("b", router.route("k1", self.LOAD, {"a", "b"}))
        router.policy = "least_inflight"
        self.assertEqual("a", router.route("k2", self.LOAD, {"a", "b"}))

    def test_two_tuple_loads_still_work(self):
        """Callers that do not supply a lifetime must not crash the policy."""
        router = self.router("longest_lived")
        self.assertIn(router.route("k", {"a": (0, 0), "b": (0, 0)}, {"a", "b"}), ("a", "b"))


class ManualControl(unittest.TestCase):
    def setUp(self):
        self.router = gateway.Router(ttl=1800, capacity=100)

    def test_a_manual_pin_overrides_the_load_picture(self):
        self.router.pin("convo:1", "a")
        for _ in range(5):
            self.assertEqual("a", self.router.route("convo:1", {"b": (0, 0, 0)}, {"a", "b"}))

    def test_a_manual_pin_does_not_expire(self):
        self.router.pin("convo:1", "a")
        self.assertEqual("a", self.router.route("convo:1", {"b": (0, 0, 0)}, {"a", "b"},
                                                now=time.time() + 99999))

    def test_a_manual_pin_yields_to_a_backend_that_is_gone(self):
        """Being emphatic about a dead backend would just mean refusing to serve."""
        self.router.pin("convo:1", "a")
        self.assertEqual("b", self.router.route("convo:1", {"b": (0, 0, 0)}, {"b"}))

    def test_unpin_restores_normal_placement(self):
        self.router.pin("convo:1", "a")
        self.assertEqual("a", self.router.unpin("convo:1"))
        self.assertEqual("b", self.router.route("convo:1", {"b": (0, 0, 0)}, {"a", "b"}))

    def test_unpinning_something_unpinned_is_harmless(self):
        self.assertIsNone(self.router.unpin("never-seen"))


class Persistence(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mktemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(self.path) and os.unlink(self.path))

    def make(self):
        return gateway.Router(ttl=1800, capacity=100, state_path=self.path)

    def test_pins_survive_a_restart(self):
        first = self.make()
        first.route("convo:1", {"a": (0, 0, 0)}, {"a"})
        first.pin("convo:2", "b")
        first.paused.add("c")
        first.policy = "round_robin"
        first.dirty = True
        first.save()

        second = self.make()
        second.load()
        self.assertEqual("a", second.route("convo:1", {"b": (0, 0, 0)}, {"a", "b"}))
        self.assertEqual({"convo:2": "b"}, second.manual)
        self.assertEqual({"c"}, second.paused)
        self.assertEqual("round_robin", second.policy)

    def test_expired_pins_are_not_restored(self):
        first = self.make()
        first.pins["old"] = ("a", time.time() - 9999)
        first.dirty = True
        first.save()
        second = self.make()
        second.load()
        self.assertNotIn("old", second.pins)

    def test_a_corrupt_state_file_is_ignored_not_fatal(self):
        with open(self.path, "w") as handle:
            handle.write("{ not json")
        router = self.make()
        router.load()  # must not raise
        self.assertEqual(0, len(router.pins))

    def test_a_state_file_that_is_not_an_object_is_ignored(self):
        with open(self.path, "w") as handle:
            handle.write("[1, 2, 3]")
        router = self.make()
        router.load()
        self.assertEqual(0, len(router.pins))

    def test_saving_is_skipped_when_nothing_changed(self):
        router = self.make()
        router.save()
        self.assertFalse(os.path.exists(self.path))


class KeySourceConfiguration(unittest.TestCase):
    def test_a_custom_chain_changes_precedence(self):
        payload = body({"prompt_cache_key": "P", "client_metadata": {"session_id": "S"}})
        default = gateway.conversation_key(headers(), payload)
        self.assertEqual("prompt_cache_key:P", default)
        reordered = gateway.conversation_key(
            headers(), payload, ["body:client_metadata.session_id", "body:prompt_cache_key"])
        self.assertEqual("client_metadata.session_id:S", reordered)

    def test_a_chain_can_exclude_the_prefix_fallback(self):
        payload = body({"messages": [{"role": "user", "content": "hi"}]})
        self.assertIsNone(gateway.conversation_key(headers(), payload, ["body:prompt_cache_key"]))

    def test_dotted_paths_reach_arbitrary_depth(self):
        payload = body({"a": {"b": {"c": "deep"}}})
        self.assertEqual("a.b.c:deep",
                         gateway.conversation_key(headers(), payload, ["body:a.b.c"]))

    def test_a_path_through_a_non_object_does_not_raise(self):
        payload = body({"a": "not an object"})
        self.assertIsNone(gateway.conversation_key(headers(), payload, ["body:a.b.c"]))


class StaleHeartbeats(unittest.TestCase):
    """A shared filesystem stall must not empty the fleet.

    Every serving job writes its registration to the same directory, so one
    stall there makes every heartbeat look stale in the same sweep. Under load
    all four backends went `gone: no heartbeat for 30s` together and came back
    five seconds later, having answered /health the whole time -- 43 requests
    got 503 from a fleet that was healthy.
    """

    def fleet(self, stale_after=30):
        args = argparse.Namespace(
            new_conversation_margin=1800, sticky_ttl=1800, sticky_capacity=100,
            users="/nonexistent", fleet_dir=tempfile.mkdtemp(),
            stale_after=stale_after, route_policy="least_conversations",
            router_state=None, key_sources=None)
        return gateway.Fleet(args)

    def register(self, fleet, job_id, heartbeat_age, healthy):
        now = time.time()
        record = {"job_id": job_id, "url": "http://host:8000",
                  "end_time": now + 7200, "heartbeat": now - heartbeat_age}
        path = os.path.join(fleet.args.fleet_dir, "%s.json" % job_id)
        with open(path, "w") as handle:
            json.dump(record, handle)
        return path

    def test_a_healthy_backend_survives_a_stale_heartbeat(self):
        fleet = self.fleet()
        self.register(fleet, "a", 0, True)
        fleet.discover()
        fleet.backends["a"].healthy = True
        # The heartbeat now looks 300s old, as it would after a filesystem stall.
        self.register(fleet, "a", 300, True)
        fleet.discover()
        self.assertIn("a", fleet.backends)

    def test_an_unhealthy_backend_with_a_stale_heartbeat_is_retired(self):
        fleet = self.fleet()
        self.register(fleet, "a", 0, True)
        fleet.discover()
        fleet.backends["a"].healthy = False
        self.register(fleet, "a", 300, False)
        fleet.discover()
        self.assertNotIn("a", fleet.backends)

    def test_a_deregistered_backend_is_retired_even_while_healthy(self):
        """Removing the file is how a job says it is going away."""
        fleet = self.fleet()
        path = self.register(fleet, "a", 0, True)
        fleet.discover()
        fleet.backends["a"].healthy = True
        os.unlink(path)
        fleet.discover()
        self.assertNotIn("a", fleet.backends)

    def test_a_whole_fleet_survives_one_stalled_sweep(self):
        fleet = self.fleet()
        for job in ("a", "b", "c", "d"):
            self.register(fleet, job, 0, True)
        fleet.discover()
        for job in fleet.backends.values():
            job.healthy = True
        for job in ("a", "b", "c", "d"):
            self.register(fleet, job, 300, True)
        fleet.discover()
        self.assertEqual({"a", "b", "c", "d"}, set(fleet.backends))



class ReviveDeadBackends(unittest.TestCase):
    """A deployment that exited but kept its allocation must be restarted.

    Twice in one night an in-service instance had its server killed, wrote
    "attempt 1 exited with status 143; allocation retained" and then sat dead
    for tens of minutes holding eight idle nodes. The gateway already knew how
    to restart such a job -- but only for a successor it had just submitted,
    never for one that had been serving for hours.
    """

    def fleet(self, state, healthy=False, heartbeat_age=0.0,
              revive_limit=3, revive_cooldown=180):
        args = argparse.Namespace(
            new_conversation_margin=1800, sticky_ttl=1800, sticky_capacity=100,
            users="/nonexistent", fleet_dir=tempfile.mkdtemp(),
            stale_after=30, route_policy="least_conversations",
            router_state=None, key_sources=None,
            revive_limit=revive_limit, revive_cooldown=revive_cooldown)
        fleet = gateway.Fleet(args)
        now = time.time()
        backend = gateway.Backend({
            "job_id": "500", "url": "http://node-a:8400", "run_dir": "/run/500",
            "state": state, "end_time": now + 3600,
            "heartbeat": now - heartbeat_age,
        })
        backend.healthy = healthy
        fleet.backends["500"] = backend
        return fleet

    def revive(self, fleet):
        """One supervision sweep, capturing what it asked serve.sh to do."""
        calls = []

        async def fake_run(_fleet, *args):
            calls.append(args)
            return 0, ""

        original = gateway.run_serve_sh
        gateway.run_serve_sh = fake_run
        try:
            asyncio.run(gateway.revive_dead_backends(fleet, time.time()))
        finally:
            gateway.run_serve_sh = original
        return calls

    def test_an_exited_backend_is_restarted_in_place(self):
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        self.assertEqual([("restart", "/run/500")], self.revive(fleet))

    def test_a_stopped_backend_is_restarted(self):
        fleet = self.fleet("stopped; allocation retained")
        self.assertEqual([("restart", "/run/500")], self.revive(fleet))

    def test_a_serving_backend_is_left_alone(self):
        fleet = self.fleet("running attempt 1", healthy=True)
        self.assertEqual([], self.revive(fleet))

    def test_an_unhealthy_backend_that_did_not_exit_is_left_alone(self):
        """A failed probe can be a blip; only the deployment's own state is
        evidence that its server is gone."""
        fleet = self.fleet("running attempt 1", healthy=False)
        self.assertEqual([], self.revive(fleet))

    def test_a_draining_backend_is_left_alone(self):
        """A roll is in flight -- restarting would fight fleetctl for the job."""
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        fleet.draining["500"] = time.time() + 600
        self.assertEqual([], self.revive(fleet))

    def test_a_superseded_backend_is_left_alone(self):
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        fleet.superseded.add("500")
        self.assertEqual([], self.revive(fleet))

    def test_a_stale_controller_is_not_asked(self):
        """Nobody is left to read the control file, so writing it is noise."""
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained",
                           heartbeat_age=3600)
        self.assertEqual([], self.revive(fleet))

    def test_restarts_are_capped(self):
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained",
                           revive_cooldown=0)
        attempts = 0
        for _ in range(fleet.args.revive_limit + 3):
            attempts += len(self.revive(fleet))
        self.assertEqual(fleet.args.revive_limit, attempts)

    def test_cooldown_spaces_attempts(self):
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        self.assertEqual(1, len(self.revive(fleet)))
        self.assertEqual([], self.revive(fleet), "second attempt inside the cooldown")

    def test_disabled_by_zero_limit(self):
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained",
                           revive_limit=0)
        self.assertEqual([], self.revive(fleet))

    def test_the_pending_successor_is_left_to_supervise_pending(self):
        """Two paths restarting one job would double its attempt budget."""
        fleet = self.fleet("attempt 1 exited with status 143; allocation retained")
        fleet.pending = ("500", time.time())
        self.assertEqual([], self.revive(fleet))


class RegistrationPath(unittest.TestCase):
    """The job id becomes a filename, so it decides where the gateway writes."""

    def path(self, job_id, root="/fleet"):
        return gateway.registration_path(root, job_id)

    def test_an_ordinary_job_id_lands_in_the_fleet_directory(self):
        self.assertEqual("/fleet/279495.json", self.path("279495"))
        self.assertEqual("/fleet/inst-a_2.json", self.path("inst-a_2"))

    def test_traversal_is_refused(self):
        for bad in ("../evil", "../../etc/passwd", "a/b", "..", ".",
                    "/etc/passwd", "a\\b"):
            self.assertIsNone(self.path(bad), bad)

    def test_a_leading_dot_is_refused(self):
        """.gitignore.json is a file the directory owner did not ask for."""
        self.assertIsNone(self.path(".hidden"))

    def test_empty_absurd_and_non_string_ids_are_refused(self):
        for bad in ("", "x" * 65, None, 279495, [], {"job_id": "a"}):
            self.assertIsNone(self.path(bad), repr(bad))

    def test_a_long_but_legal_id_is_allowed(self):
        self.assertIsNotNone(self.path("j" * 64))


def _policy_dir(tmp, **files):
    for name, body in files.items():
        with open(os.path.join(tmp, name + ".py"), "w") as handle:
            handle.write(textwrap.dedent(body))
    registry = gateway.PolicyDir(tmp)
    registry.reload()
    return registry


GOOD = """
    def select(accepting):
        return sorted(accepting)[-1]
"""
RAISES = """
    def select(accepting):
        raise RuntimeError("nope")
"""
LIES = """
    def select(accepting):
        return "not-a-backend"
"""

ACCEPTING = {"a": (1, 1, 7200), "b": (5, 5, 7200)}


class PolicyLoading(unittest.TestCase):
    """Custom policies come from files, and a bad file costs only itself."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_policy_file_supplies_a_policy_named_after_it(self):
        registry = _policy_dir(self.tmp, pick_last=GOOD)
        self.assertEqual(("pick_last",), registry.names())
        self.assertEqual("b", registry.run("pick_last", ACCEPTING))

    def test_a_file_without_select_is_not_a_policy(self):
        registry = _policy_dir(self.tmp, empty="x = 1\n")
        self.assertEqual((), registry.names())

    def test_a_file_that_fails_to_import_costs_only_itself(self):
        registry = _policy_dir(self.tmp, broken="def select(  # unclosed\n",
                               fine=GOOD)
        self.assertEqual(("fine",), registry.names())

    def test_a_policy_may_not_shadow_a_built_in(self):
        registry = _policy_dir(self.tmp, round_robin=GOOD)
        self.assertEqual((), registry.names())

    def test_underscore_files_are_helpers_not_policies(self):
        registry = _policy_dir(self.tmp, _shared=GOOD, real=GOOD)
        self.assertEqual(("real",), registry.names())

    def test_deleting_the_file_withdraws_the_policy(self):
        registry = _policy_dir(self.tmp, gone=GOOD)
        os.remove(os.path.join(self.tmp, "gone.py"))
        registry.reload()
        self.assertEqual((), registry.names())
        self.assertIsNone(registry.run("gone", ACCEPTING))


class PolicyFailure(unittest.TestCase):
    """A policy runs on the request path, so it is contained, not trusted."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_raising_policy_returns_none_rather_than_propagating(self):
        registry = _policy_dir(self.tmp, bad=RAISES)
        self.assertIsNone(registry.run("bad", ACCEPTING))

    def test_a_policy_naming_a_backend_that_is_not_on_offer_is_refused(self):
        registry = _policy_dir(self.tmp, liar=LIES)
        self.assertIsNone(registry.run("liar", ACCEPTING))

    def test_repeated_failure_disables_the_policy(self):
        registry = _policy_dir(self.tmp, bad=RAISES)
        for _ in range(gateway.PolicyDir.strikes):
            registry.run("bad", ACCEPTING)
        self.assertIn("bad", registry.disabled)
        self.assertEqual((), registry.names())

    def test_a_working_policy_is_never_struck(self):
        registry = _policy_dir(self.tmp, fine=GOOD)
        for _ in range(10):
            self.assertEqual("b", registry.run("fine", ACCEPTING))
        self.assertEqual(set(), registry.disabled)

    def test_editing_the_file_clears_the_strikes(self):
        """Fixing a policy is how you re-enable it; restarting is not."""
        registry = _policy_dir(self.tmp, bad=RAISES)
        for _ in range(gateway.PolicyDir.strikes):
            registry.run("bad", ACCEPTING)
        self.assertIn("bad", registry.disabled)
        path = os.path.join(self.tmp, "bad.py")
        with open(path, "w") as handle:
            handle.write(textwrap.dedent(GOOD))
        os.utime(path, (0, 0))          # any change, not a later one
        registry.reload()
        self.assertEqual(("bad",), registry.names())
        self.assertEqual("b", registry.run("bad", ACCEPTING))


class PolicyInRouter(unittest.TestCase):
    """Placement uses a custom policy, and survives one that misbehaves."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def router(self, policy, **files):
        router = gateway.Router(1800, 100, policy=policy, state_path=None,
                                key_sources=list(gateway.DEFAULT_KEY_SOURCES))
        router.policies = _policy_dir(self.tmp, **files)
        return router

    def test_a_custom_policy_places_the_conversation(self):
        router = self.router("pick_last", pick_last=GOOD)
        self.assertEqual("b", router.select(ACCEPTING))

    def test_known_policies_lists_built_ins_and_custom_together(self):
        router = self.router("pick_last", pick_last=GOOD)
        known = gateway.known_policies(router)
        self.assertIn("least_conversations", known)
        self.assertIn("pick_last", known)

    def test_a_failing_custom_policy_falls_back_to_the_default(self):
        """The conversation that arrives during a bad policy still gets placed."""
        router = self.router("bad", bad=RAISES)
        self.assertEqual("a", router.select(ACCEPTING))   # least_conversations

    def test_a_policy_that_is_not_loaded_falls_back_rather_than_raising(self):
        router = self.router("never_written")
        self.assertEqual("a", router.select(ACCEPTING))

    def test_built_ins_still_work_with_a_policy_dir_present(self):
        router = self.router("least_inflight", pick_last=GOOD)
        self.assertEqual("a", router.select(ACCEPTING))


if __name__ == "__main__":
    unittest.main(verbosity=2)
