"""Unit tests for the bounded scan job store (LR2 §8.6).

The store must stay bounded by construction (capacity, per-bucket sample count,
per-sample bytes, whole-record bytes, distinct counter keys), enforce CAS +
owner + lease/TTL, be idempotent on re-create, and reap a dead owner's RUNNING
job to ABANDONED so a late completion cannot resurrect it.
"""

from __future__ import annotations

import pytest

from lightrag.kg.scan_job_store import (
    AsyncioScanJobStore,
    ScanJobCreateOutcome,
    ScanJobStatus,
    ScanJobUpdateConflict,
    _ScanJobRecord,
)

pytestmark = pytest.mark.offline


class _Clock:
    """Manual clock for deterministic lease/TTL tests."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def _store(**kw) -> AsyncioScanJobStore:
    clock = kw.pop("clock", _Clock())
    return AsyncioScanJobStore("ws", clock=clock, **kw)


def test_create_accepted_then_already_exists_idempotent():
    store = _store()
    r1 = store.create("t1", "owner")
    assert r1.outcome is ScanJobCreateOutcome.ACCEPTED
    assert r1.record["status"] == ScanJobStatus.RUNNING.value
    assert r1.record["version"] == 1
    assert "owner_token" not in r1.record  # public snapshot never leaks the token

    r2 = store.create("t1", "owner2")
    assert r2.outcome is ScanJobCreateOutcome.ALREADY_EXISTS
    assert r2.record["version"] == 1  # existing record returned, not clobbered


def test_update_applies_deltas_and_bumps_version_under_cas():
    store = _store()
    store.create("t1", "owner")
    res = store.update(
        "t1", "owner", count_deltas={"claimed_new": 2}, expected_version=1
    )
    assert res.ok and res.record["version"] == 2
    assert res.record["counts"]["claimed_new"] == 2

    # Stale version → rejected, record echoed for re-sync, no mutation.
    stale = store.update(
        "t1", "owner", count_deltas={"claimed_new": 5}, expected_version=1
    )
    assert not stale.ok and stale.conflict is ScanJobUpdateConflict.VERSION
    assert stale.record["counts"]["claimed_new"] == 2  # unchanged


def test_update_owner_and_not_found_rejections():
    store = _store()
    store.create("t1", "owner")
    wrong = store.update("t1", "intruder", count_deltas={"x": 1}, expected_version=1)
    assert not wrong.ok and wrong.conflict is ScanJobUpdateConflict.OWNER
    missing = store.update("nope", "owner", count_deltas={"x": 1}, expected_version=1)
    assert not missing.ok and missing.conflict is ScanJobUpdateConflict.NOT_FOUND


def test_sample_byte_cap_and_count_cap():
    store = _store(sample_limit=2, sample_max_bytes=8)
    store.create("t1", "owner")
    v = 1
    # A too-long sample is byte-truncated and flagged.
    r = store.update("t1", "owner", sample=("error", "x" * 100), expected_version=v)
    assert r.ok
    bucket = r.record["samples"]["error"]
    assert len(bucket["items"][0].encode()) <= 8 and bucket["truncated"] is True
    v = r.record["version"]
    # Fill to the count cap, then overflow is dropped (not retained).
    for _ in range(5):
        r = store.update("t1", "owner", sample=("error", "y"), expected_version=v)
        v = r.record["version"]
    bucket = store.get("t1")["samples"]["error"]
    assert len(bucket["items"]) == 2  # capped at sample_limit
    assert bucket["dropped"] >= 1


def test_record_byte_ceiling_drops_further_samples():
    # A tiny record ceiling: after a couple of samples, further ones are dropped
    # even below the per-bucket count cap.
    store = _store(sample_limit=100, sample_max_bytes=64, record_max_bytes=400)
    store.create("t1", "owner")
    v = 1
    dropped_seen = False
    for _ in range(50):
        r = store.update(
            "t1", "owner", sample=("processed", "z" * 60), expected_version=v
        )
        v = r.record["version"]
        if r.record["samples"]["processed"]["dropped"] > 0:
            dropped_seen = True
    assert dropped_seen
    # The record stays bounded well under an O(N) blowup.
    total = sum(
        len(s.encode()) for s in store.get("t1")["samples"]["processed"]["items"]
    )
    assert total <= 400


def test_distinct_counter_key_cap():
    store = _store(max_counter_keys=3)
    store.create("t1", "owner")
    v = 1
    for i in range(10):
        r = store.update("t1", "owner", count_deltas={f"k{i}": 1}, expected_version=v)
        v = r.record["version"]
    record = store.get("t1")
    assert len(record["counts"]) == 3  # capped
    assert record["counters_dropped"] == 7  # and the drops are not silent


def test_an_over_long_counter_key_cannot_breach_the_record_ceiling():
    """Fix-proof: the distinct-key cap bounds HOW MANY keys, not how big one is,
    and the record ceiling was only re-checked on the sample path. A single
    100 KB key therefore landed in a store whose whole-record ceiling was
    1 KB."""
    store = _store(record_max_bytes=1024, counter_key_max_bytes=64)
    store.create("t1", "owner")
    huge = "k" * 100_000
    r = store.update("t1", "owner", count_deltas={huge: 1}, expected_version=1)
    assert r.ok  # the update itself still succeeds; only the delta is refused
    record = store.get("t1")
    assert record["counts"] == {}
    assert record["counters_dropped"] == 1


def test_counter_keys_are_measured_in_utf8_bytes_and_refused_not_truncated():
    """A CJK key is 3 bytes per character, so a char-length check would let a
    3x-over-cap key through. And an over-long key must be REFUSED, not clipped:
    truncating would fuse two distinct labels into one counter."""
    store = _store(counter_key_max_bytes=12)
    store.create("t1", "owner")
    # 5 CJK chars = 15 UTF-8 bytes > 12, but only 5 by len().
    r = store.update(
        "t1", "owner", count_deltas={"文档解析失败": 1}, expected_version=1
    )
    assert r.ok and r.record["counts"] == {} and r.record["counters_dropped"] == 1

    long_a, long_b = "same_prefix_a", "same_prefix_b"  # 13 bytes each, over cap
    v = r.record["version"]
    r = store.update(
        "t1", "owner", count_deltas={long_a: 1, long_b: 1}, expected_version=v
    )
    # Neither was clipped to a shared 12-byte "same_prefix_" counter.
    assert r.record["counts"] == {}
    assert r.record["counters_dropped"] == 3


def test_a_bounded_key_still_counts_and_repeats_add_no_bytes():
    """The cap must not get in the way of the real taxonomy: every in-tree key is
    a short label, and repeated deltas on an existing key are pure integer
    updates that no byte bound can refuse."""
    store = _store(record_max_bytes=400, counter_key_max_bytes=64)
    store.create("t1", "owner")
    v = 1
    for _ in range(200):
        r = store.update(
            "t1",
            "owner",
            count_deltas={"resume_same_physical_source": 1},
            expected_version=v,
        )
        v = r.record["version"]
    record = store.get("t1")
    assert record["counts"]["resume_same_physical_source"] == 200
    assert record["counters_dropped"] == 0


def test_the_record_ceiling_covers_the_scalars_not_just_the_payload():
    """Fix-proof: ``approx_bytes`` allowed a flat 256 bytes for "the scalar
    fields", so whatever the caller put in an identifier or a message sailed past
    ``record_max_bytes`` — the reviewer's repro accepted a ~100 KB record in a
    512-byte store."""
    store = _store(record_max_bytes=512, identifier_max_bytes=128)
    huge = "t" * 100_000
    refused = store.create(huge, "owner")
    assert refused.outcome is ScanJobCreateOutcome.INVALID_IDENTIFIER
    assert refused.record is None
    assert store.get(huge) is None
    # Same for the owner token.
    assert store.create("t1", huge).outcome is ScanJobCreateOutcome.INVALID_IDENTIFIER
    assert store.snapshot() == []

    # A bounded identifier is accepted and now COUNTS toward the ceiling.
    store.create("t" * 100, "o" * 100)
    record_bytes = store._jobs["t" * 100].approx_bytes()
    assert record_bytes >= 200


def test_a_record_that_is_born_over_the_ceiling_is_never_created():
    """Fix-proof: the per-identifier cap covers ``track_id`` / ``owner_token``, but
    ``workspace`` is a scalar ``approx_bytes`` counts and no caller passes to
    ``create`` — a 100 KB namespace seated a ~100 KB record in a 512-byte store,
    and every later mutation then (correctly, and confusingly) refused to grow it.
    The invariant belongs to the record, so the built record is what gets
    measured."""
    store = AsyncioScanJobStore("W" * 100_000, clock=_Clock(), record_max_bytes=512)
    refused = store.create("t1", "owner")
    assert refused.outcome is ScanJobCreateOutcome.INVALID_IDENTIFIER
    assert "record ceiling" in refused.message
    assert store.get("t1") is None and store.snapshot() == []


def test_the_refusal_does_not_evict_a_terminal_record_on_its_way_out():
    """Ordering: an over-ceiling create must not spend the capacity slot it was
    never going to use."""
    store = AsyncioScanJobStore(
        "W" * 1000, clock=_Clock(), capacity=1, record_max_bytes=512
    )
    # Seat one terminal record the eviction step would happily take.
    store._jobs["survivor"] = _ScanJobRecord(
        track_id="survivor",
        workspace="",
        owner_token="o",
        status=ScanJobStatus.COMPLETED.value,
        counts={},
        samples={},
        created_at=1000.0,
        updated_at=1000.0,
        lease_expires_at=1000.0,
        version=1,
    )
    assert (
        store.create("t1", "owner").outcome is ScanJobCreateOutcome.INVALID_IDENTIFIER
    )
    assert store.get("survivor") is not None


def test_a_terminal_message_cannot_push_the_record_past_the_ceiling():
    """Fix-proof: ``set_status`` / ``cancel`` / the lease reaper capped the message
    at ``sample_max_bytes`` only, with no regard for what the record had already
    spent — so a record filled to the ceiling by samples still accepted one more
    message's worth of bytes on top of it. Every mutation means every mutation."""
    for terminate in (
        lambda store, version: store.set_status(
            "t1",
            "owner",
            ScanJobStatus.FAILED,
            expected_version=version,
            message="m" * 256,
        ),
        lambda store, version: store.cancel("t1", "owner", message="m" * 256),
    ):
        store = _store(sample_limit=100, sample_max_bytes=256, record_max_bytes=400)
        store.create("t1", "owner")
        record = store._jobs["t1"]
        version = record.version
        while record.approx_bytes() + 32 <= 400:
            version = store.update(
                "t1", "owner", sample=("processed", "z" * 32), expected_version=version
            ).record["version"]

        assert terminate(store, version).ok
        assert record.approx_bytes() <= 400
        # The message is truncated, not silently discarded whole.
        assert record.message and record.message[0] == "m"


def test_the_lease_reaper_message_respects_the_ceiling_too():
    clock = _Clock(1000.0)
    store = _store(
        clock=clock,
        lease_seconds=10.0,
        sample_limit=100,
        sample_max_bytes=256,
        record_max_bytes=400,
    )
    store.create("t1", "owner")
    record = store._jobs["t1"]
    version = record.version
    while record.approx_bytes() + 32 <= 400:
        version = store.update(
            "t1", "owner", sample=("processed", "z" * 32), expected_version=version
        ).record["version"]

    clock.t = 1011.0  # lease expired → reaped to ABANDONED with its message
    assert store.get("t1")["status"] == ScanJobStatus.ABANDONED.value
    assert record.approx_bytes() <= 400


def test_an_arbitrary_precision_counter_is_refused_not_stored_as_eight_bytes():
    """Fix-proof: Python ints are unbounded, ``approx_bytes`` counts every
    counter value as 8 bytes, and a 1001-digit delta was accepted — ~450 bytes of
    payload the ceiling never saw. 2**53 is also the last value a JSON client can
    read back exactly, and this record is serialized to JSON."""
    store = _store()
    store.create("t1", "owner")
    absurd = 10**1000

    r = store.update(
        "t1", "owner", count_deltas={"discovered": absurd}, expected_version=1
    )
    assert r.ok
    assert r.record["counts"] == {}
    assert r.record["counters_dropped"] == 1

    # And an accumulation that would cross the bound is refused too, so the
    # value cannot be walked past it one legal delta at a time.
    v = r.record["version"]
    r = store.update(
        "t1", "owner", count_deltas={"discovered": 2**53 - 1}, expected_version=v
    )
    v = r.record["version"]
    r = store.update("t1", "owner", count_deltas={"discovered": 99}, expected_version=v)
    assert r.record["counts"]["discovered"] == 2**53 - 1
    assert r.record["counters_dropped"] == 2


def test_a_bool_is_not_a_counter_delta():
    """``isinstance(True, int)`` is True, so a bool would land as 1/0 and read
    back as a count nobody wrote."""
    store = _store()
    store.create("t1", "owner")
    r = store.update(
        "t1", "owner", count_deltas={"discovered": True}, expected_version=1
    )
    assert r.ok and r.record["counts"] == {}


def test_capacity_evicts_terminal_then_rejects_all_running():
    store = _store(capacity=2)
    store.create("a", "o")
    store.create("b", "o")
    # Full of RUNNING jobs → refuse a third.
    full = store.create("c", "o")
    assert full.outcome is ScanJobCreateOutcome.CAPACITY_EXCEEDED

    # Terminate one; the oldest terminal is now evictable.
    store.set_status("a", "o", ScanJobStatus.COMPLETED, expected_version=1)
    ok = store.create("c", "o")
    assert ok.outcome is ScanJobCreateOutcome.ACCEPTED
    assert store.get("a") is None  # evicted


def test_set_status_terminal_transition_and_late_completion_loses():
    store = _store()
    store.create("t1", "owner")
    done = store.set_status("t1", "owner", ScanJobStatus.COMPLETED, expected_version=1)
    assert done.ok and done.record["status"] == ScanJobStatus.COMPLETED.value
    # A second (late) completion at the stale version loses — already terminal.
    late = store.set_status("t1", "owner", ScanJobStatus.FAILED, expected_version=1)
    assert not late.ok and late.conflict is ScanJobUpdateConflict.TERMINAL
    assert store.get("t1")["status"] == ScanJobStatus.COMPLETED.value


def test_set_status_rejects_non_terminal_target():
    store = _store()
    store.create("t1", "owner")
    r = store.set_status("t1", "owner", ScanJobStatus.RUNNING, expected_version=1)
    assert not r.ok and r.conflict is ScanJobUpdateConflict.INVALID_STATUS


def test_lease_expiry_abandons_running_job():
    clock = _Clock(1000.0)
    store = _store(clock=clock, lease_seconds=10.0)
    store.create("t1", "owner")
    clock.t = 1011.0  # lease (1000+10) expired
    snap = store.get("t1")
    assert snap["status"] == ScanJobStatus.ABANDONED.value
    # A late owner completion now loses (job already ABANDONED).
    late = store.set_status("t1", "owner", ScanJobStatus.COMPLETED, expected_version=1)
    assert not late.ok and late.conflict is ScanJobUpdateConflict.TERMINAL


def test_ttl_evicts_old_terminal_on_reap():
    clock = _Clock(1000.0)
    store = _store(clock=clock, ttl_seconds=100.0)
    store.create("t1", "owner")
    store.set_status("t1", "owner", ScanJobStatus.COMPLETED, expected_version=1)
    clock.t = 1101.0  # terminal older than TTL
    # A create triggers a reap that evicts the expired terminal.
    store.create("t2", "owner")
    assert store.get("t1") is None


def test_cancel_is_owner_checked_and_idempotent():
    store = _store()
    store.create("t1", "owner")
    wrong = store.cancel("t1", "intruder")
    assert not wrong.ok and wrong.conflict is ScanJobUpdateConflict.OWNER
    ok = store.cancel("t1", "owner")
    assert ok.ok and ok.record["status"] == ScanJobStatus.CANCELLED.value
    # Cancelling an already-terminal job is a no-op success (idempotent teardown).
    again = store.cancel("t1", "owner")
    assert again.ok and again.record["status"] == ScanJobStatus.CANCELLED.value


def test_remove_terminal_never_removes_running():
    store = _store()
    store.create("t1", "owner")
    assert store.remove_terminal("t1") is False  # RUNNING is protected
    store.set_status("t1", "owner", ScanJobStatus.CANCELLED, expected_version=1)
    assert store.remove_terminal("t1") is True
    assert store.get("t1") is None


def test_resolution_layer_single_process_workspace_isolated():
    # get_scan_job_store resolves a per-workspace store single-process; distinct
    # workspaces are isolated and the same workspace reaches the same store.
    import lightrag.kg.shared_storage as shared_storage

    shared_storage.initialize_share_data()
    try:
        s_a = shared_storage.get_scan_job_store("wsA")
        s_a2 = shared_storage.get_scan_job_store("wsA")
        s_b = shared_storage.get_scan_job_store("wsB")
        assert s_a is s_a2  # same workspace → cached same instance
        assert s_a is not s_b

        r = s_a.create("t1", "owner")
        assert r.outcome is ScanJobCreateOutcome.ACCEPTED
        assert s_a.get("t1") is not None
        assert s_b.get("t1") is None  # workspace-isolated
    finally:
        shared_storage.finalize_share_data()
