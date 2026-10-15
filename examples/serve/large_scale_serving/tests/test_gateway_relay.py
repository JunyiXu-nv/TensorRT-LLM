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
"""Rolling one instance before its wall clock, on a fleet of several.

These are unit tests against the supervisor's bookkeeping rather than scenario
tests through a live gateway. What they cover is a decision made from the
backend table -- which job replaces which -- and driving that through a real
fleet would mean staging jobs with real wall clocks hours apart.

The bug they pin down cost the whole feature. `elect` ranks backends by end
time and marked the loser superseded, which is a replacement on a
single-instance deployment and nothing of the kind on this one: i02 submitted
an hour after i00 outlives it while both serve. Relay drained every instance
but the longest-lived, so it was switched off fleet-wide with --no-relay and
stayed off.
"""

import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "gw_relay_under_test", os.path.join(HERE, os.pardir, "gateway.py")
)
gw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gw)


def make_backend(job_id, label, end_time, healthy=True):
    b = gw.Backend(
        {
            "job_id": job_id,
            "url": "http://127.0.0.1:1",
            "run_dir": "/var/2026-09/17/junyix_0917_%s_kffleet_glm5.2_%s" % (job_id, label),
            "end_time": end_time,
            "heartbeat": 0,
        }
    )
    b.healthy = healthy
    b.healthy_since = 1.0 if healthy else 0.0
    return b


class FakeArgs:
    no_relay = False
    relay_per_instance = True
    lead_time = 2700
    min_submit_interval = 300
    promote_after = 60
    fleetctl = "/bin/true"
    fleet_config = "/dev/null"
    # relay_jobs.json goes beside this file; tests that save point it at a
    # temporary directory.
    users = "/nonexistent/users.txt"


def make_fleet(backends):
    fleet = gw.Fleet.__new__(gw.Fleet)
    fleet.backends = {b.job_id: b for b in backends}
    fleet.superseded = set()
    fleet.draining = {}
    fleet.relaying = {}
    fleet.relay_jobs = {}
    fleet.inflight = {}
    fleet.pending = None
    fleet.active = None
    fleet.ever_active = False
    fleet.args = FakeArgs()
    return fleet


def test_a_longer_lived_instance_does_not_supersede_a_different_one():
    """The bug that forced --no-relay, stated directly.

    i01 is submitted later than i00 and therefore outlives it, which is what
    every fleet looks like a few hours in. Neither is replacing the other.
    """
    fleet = make_fleet([make_backend("j00", "i00", 1000), make_backend("j01", "i01", 9000)])
    fleet.active = "j00"
    fleet.elect()
    assert fleet.active == "j01", "the election still picks the longest-lived backend"
    assert fleet.superseded == set(), (
        "i00 was marked superseded by i01, which would drain an instance that is "
        "serving and replacing nothing: %s" % fleet.superseded
    )


def test_a_newer_job_of_the_same_instance_does_supersede_it():
    """The case the feature exists for has to keep working.

    Same label, later end time: this is a replacement, and the predecessor is
    supposed to be retired once the successor holds up.
    """
    fleet = make_fleet([make_backend("old", "i00", 1000), make_backend("new", "i00", 9000)])
    fleet.active = "old"
    fleet.elect()
    assert fleet.active == "new"
    assert fleet.superseded == {"old"}, fleet.superseded


@pytest.mark.asyncio
async def test_each_instance_is_rolled_against_its_own_clock():
    """Only the instance that is actually expiring gets a successor.

    The single-lineage relay asked whether `active` was expiring, and `active`
    is the longest-lived backend -- so it asked about the one instance that is
    furthest from needing anything.
    """
    calls = []

    async def fake_fleetctl(fleet, *args):
        calls.append(args)
        return 0, ""

    gw.run_fleetctl = fake_fleetctl
    now = 10_000.0
    fleet = make_fleet(
        [
            make_backend("j00", "i00", now + 600),  # 10 min left: due
            make_backend("j02", "i02", now + 36_000),  # 10 h left: not due
        ]
    )
    await gw.relay_per_instance(fleet, now)

    assert calls == [("up", "--only", "i00", "--force")], calls
    assert "i00" in fleet.relaying and "i02" not in fleet.relaying


@pytest.mark.asyncio
async def test_a_second_sweep_does_not_submit_twice():
    """Idempotence, which is the whole risk of a timer that fires every sweep.

    A submitted job takes minutes to register, so the "does a successor exist"
    test stays false meanwhile; without the per-label cooldown the supervisor
    would submit one on every pass and pile up jobs for one instance.
    """
    calls = []

    async def fake_fleetctl(fleet, *args):
        calls.append(args)
        return 0, ""

    gw.run_fleetctl = fake_fleetctl
    now = 10_000.0
    fleet = make_fleet([make_backend("j00", "i00", now + 600)])
    await gw.relay_per_instance(fleet, now)
    await gw.relay_per_instance(fleet, now + 30)
    assert len(calls) == 1, calls

    # ...and once the cooldown lapses it is allowed to try again, because the
    # first submit may simply have failed.
    await gw.relay_per_instance(fleet, now + FakeArgs.min_submit_interval + 1)
    assert len(calls) == 2, calls


@pytest.mark.asyncio
async def test_an_instance_that_already_has_a_successor_is_left_alone():
    """Two jobs of one label means the roll already happened."""
    calls = []

    async def fake_fleetctl(fleet, *args):
        calls.append(args)
        return 0, ""

    gw.run_fleetctl = fake_fleetctl
    now = 10_000.0
    fleet = make_fleet(
        [make_backend("old", "i00", now + 600), make_backend("new", "i00", now + 36_000)]
    )
    await gw.relay_per_instance(fleet, now)
    assert calls == [], calls
    assert fleet.superseded == {"old"}, (
        "the predecessor was not handed to the drain, so its allocation would "
        "be held until the wall clock: %s" % fleet.superseded
    )


def submitting(calls, job_id):
    """A fleetctl stand-in that reports one submitted job, as the real one prints it."""

    async def fake_fleetctl(fleet, *args):
        calls.append(args)
        return 0, "  %s: 12 nodes -> %s.yaml\n    submitted job %s" % (args[2], args[2], job_id)

    return fake_fleetctl


def scheduler_says(state):
    async def fake_status(job_id):
        return state

    return fake_status


@pytest.mark.asyncio
async def test_a_queued_successor_is_not_submitted_again(tmp_path, monkeypatch):
    """The duplicate the dedupe pass kept cancelling.

    A successor can wait in the queue far longer than the cooldown -- the
    short QoS holds it until nodes free up -- and the cooldown alone
    resubmitted every five minutes meanwhile.
    """
    calls = []
    monkeypatch.setattr(gw, "run_fleetctl", submitting(calls, "555"))
    monkeypatch.setattr(gw, "slurm_job_status", scheduler_says(("PENDING", "Priority")))
    now = 10_000.0
    fleet = make_fleet([make_backend("j00", "i00", now + 600)])
    fleet.args.users = str(tmp_path / "users.txt")
    await gw.relay_per_instance(fleet, now)
    assert fleet.relay_jobs == {"i00": ("555", now)}, fleet.relay_jobs
    for later in (1, 2, 5):
        await gw.relay_per_instance(fleet, now + later * FakeArgs.min_submit_interval + 1)
    assert len(calls) == 1, calls


@pytest.mark.asyncio
async def test_a_successor_the_scheduler_no_longer_has_is_replaced(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gw, "run_fleetctl", submitting(calls, "555"))
    monkeypatch.setattr(gw, "slurm_job_status", scheduler_says(("GONE", "")))
    now = 10_000.0
    fleet = make_fleet([make_backend("j00", "i00", now + 600)])
    fleet.args.users = str(tmp_path / "users.txt")
    await gw.relay_per_instance(fleet, now)
    await gw.relay_per_instance(fleet, now + FakeArgs.min_submit_interval + 1)
    assert len(calls) == 2, calls


@pytest.mark.asyncio
async def test_no_answer_from_the_scheduler_submits_nothing(tmp_path, monkeypatch):
    """A blind resubmission is how duplicates start."""
    calls = []
    monkeypatch.setattr(gw, "run_fleetctl", submitting(calls, "555"))
    monkeypatch.setattr(gw, "slurm_job_status", scheduler_says(None))
    now = 10_000.0
    fleet = make_fleet([make_backend("j00", "i00", now + 600)])
    fleet.args.users = str(tmp_path / "users.txt")
    await gw.relay_per_instance(fleet, now)
    await gw.relay_per_instance(fleet, now + FakeArgs.min_submit_interval + 1)
    assert len(calls) == 1, calls


@pytest.mark.asyncio
async def test_a_registered_successor_is_relayed_in_its_turn(tmp_path, monkeypatch):
    """The record must not outlive the registration.

    The successor becomes the label's next job to relay; a record left behind
    would read as "already submitted" when that job nears its own end.
    """
    calls = []
    monkeypatch.setattr(gw, "run_fleetctl", submitting(calls, "555"))
    monkeypatch.setattr(gw, "slurm_job_status", scheduler_says(("RUNNING", "None")))
    now = 10_000.0
    fleet = make_fleet([make_backend("j00", "i00", now + 600)])
    fleet.args.users = str(tmp_path / "users.txt")
    await gw.relay_per_instance(fleet, now)
    # 555 registers, a full lifetime ahead of it.
    fleet.backends["555"] = make_backend("555", "i00", now + 7200)
    await gw.relay_per_instance(fleet, now + 60)
    assert fleet.relay_jobs == {}, fleet.relay_jobs
    # Its predecessor is gone, and 555 itself nears its end.
    del fleet.backends["j00"]
    monkeypatch.setattr(gw, "run_fleetctl", submitting(calls, "556"))
    await gw.relay_per_instance(fleet, now + 7200 - 600)
    assert len(calls) == 2, calls
    assert fleet.relay_jobs == {"i00": ("556", now + 7200 - 600)}, fleet.relay_jobs


@pytest.mark.asyncio
async def test_a_loading_successor_that_drops_out_is_not_submitted_again(tmp_path, monkeypatch):
    """A successor flapping in and out of the table while it loads (jhb, 10-15).

    Its registration exists from the controller's start, but its heartbeat can
    lapse past --stale-after while the server loads, so it appears and goes.
    The record used to be dropped on the first appearance; the next lapse then
    left the label with one member and no record, and a duplicate went out.
    """
    calls = []
    monkeypatch.setattr(gw, "run_fleetctl", submitting(calls, "555"))
    monkeypatch.setattr(gw, "slurm_job_status", scheduler_says(("RUNNING", "None")))
    now = 10_000.0
    fleet = make_fleet([make_backend("j00", "i00", now + 600)])
    fleet.args.users = str(tmp_path / "users.txt")
    await gw.relay_per_instance(fleet, now)
    assert fleet.relay_jobs == {"i00": ("555", now)}, fleet.relay_jobs

    # 555 registers while it loads: present, not yet healthy.
    fleet.backends["555"] = make_backend("555", "i00", now + 7200, healthy=False)
    await gw.relay_per_instance(fleet, now + 30)
    assert fleet.relay_jobs == {"i00": ("555", now)}, "dropped before 555 served"

    # Its heartbeat lapses and it drops out, well past the cooldown.
    del fleet.backends["555"]
    await gw.relay_per_instance(fleet, now + FakeArgs.min_submit_interval + 1)
    await gw.relay_per_instance(fleet, now + 2 * FakeArgs.min_submit_interval + 1)
    assert len(calls) == 1, "a duplicate was submitted while 555 was still coming: %s" % calls

    # 555 comes back and serves: now the record is done.
    fleet.backends["555"] = make_backend("555", "i00", now + 7200)
    await gw.relay_per_instance(fleet, now + 2 * FakeArgs.min_submit_interval + 30)
    assert fleet.relay_jobs == {}, fleet.relay_jobs
    assert len(calls) == 1, calls


@pytest.mark.asyncio
async def test_the_record_survives_a_handover(tmp_path, monkeypatch):
    """10-14 13:21Z: the new process resubmitted hs1 within a second of taking over."""
    calls = []
    monkeypatch.setattr(gw, "run_fleetctl", submitting(calls, "555"))
    monkeypatch.setattr(gw, "slurm_job_status", scheduler_says(("PENDING", "Priority")))
    now = 10_000.0
    users = str(tmp_path / "users.txt")
    first = make_fleet([make_backend("j00", "i00", now + 600)])
    first.args.users = users
    await gw.relay_per_instance(first, now)

    second = make_fleet([make_backend("j00", "i00", now + 600)])
    second.args.users = users
    second.pools = {}
    gw.load_relay_jobs(second, now=now + 1)
    assert second.relay_jobs == {"i00": ("555", now)}, second.relay_jobs
    await gw.relay_per_instance(second, now + 1)
    assert len(calls) == 1, "the successor resubmitted %s at once" % (calls,)


def test_an_old_record_is_not_trusted(tmp_path):
    users = str(tmp_path / "users.txt")
    with open(os.path.join(str(tmp_path), "relay_jobs.json"), "w") as handle:
        handle.write(
            '{"version": 1, "pools": {"default": {"i00": ["555", 1000.0], "i01": ["556", 9000.0]}}}'
        )
    fleet = make_fleet([])
    fleet.args.users = users
    fleet.pools = {}
    gw.load_relay_jobs(fleet, now=1000.0 + gw.RELAY_JOB_MAX_AGE + 1)
    assert fleet.relay_jobs == {"i01": ("556", 9000.0)}, fleet.relay_jobs
