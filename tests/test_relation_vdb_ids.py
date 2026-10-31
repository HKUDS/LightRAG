"""Regression tests for versioned relation VDB IDs.

Two distinct undirected edges can concatenate to the same string
("A", "BC") and ("AB", "C") both give "ABC", so hashing the plain
concatenation assigns both edges one VDB id and the last writer wins.
The versioned id length-prefixes the canonical endpoint pair, which
keeps those edges distinct while keeping reversed endpoints on one id.
"""

import pytest

from lightrag.utils import (
    compute_mdhash_id,
    filter_deletable_relation_vdb_ids,
    make_relation_vdb_id,
    make_relation_vdb_ids,
)

pytestmark = pytest.mark.offline


def test_colliding_endpoint_pairs_get_distinct_versioned_ids():
    first = make_relation_vdb_ids("A", "BC")
    second = make_relation_vdb_ids("AB", "C")

    assert first[0] != second[0]
    assert first[0].startswith("rel-v2-")
    assert second[0].startswith("rel-v2-")
    # The legacy fallback still collides; only the versioned id fixes it.
    assert first[1] == second[1] == compute_mdhash_id("ABC", prefix="rel-")


def test_reversed_endpoints_share_one_versioned_id():
    assert make_relation_vdb_ids("A", "BC") == make_relation_vdb_ids("BC", "A")
    assert make_relation_vdb_id("A", "BC") == make_relation_vdb_id("BC", "A")


def test_versioned_id_length_prefixes_the_canonical_pair():
    import hashlib

    material = b"1:A:BC"
    assert make_relation_vdb_id("A", "BC") == (
        "rel-v2-" + hashlib.md5(material).hexdigest()
    )
    # A separator inside a name must not recreate the collision.
    assert make_relation_vdb_id("A:B", "C") != make_relation_vdb_id("A", "B:C")


def test_non_ascii_endpoint_pairs_stay_distinct():
    assert make_relation_vdb_id("aé", "b") != make_relation_vdb_id("a", "éb")


def test_self_loop_deduplicates_candidates():
    ids = make_relation_vdb_ids("A", "A")

    assert len(ids) == 2
    assert ids[0].startswith("rel-v2-")
    assert ids[1] == compute_mdhash_id("AA", prefix="rel-")


def test_legacy_aliases_follow_the_versioned_id():
    ids = make_relation_vdb_ids("Bob", "Alice")

    assert ids[0] == make_relation_vdb_id("Alice", "Bob")
    assert ids[1] == compute_mdhash_id("AliceBob", prefix="rel-")
    assert ids[2] == compute_mdhash_id("BobAlice", prefix="rel-")


class _FakeRelationVDB:
    """Dict-backed relation vector store with aligned get_by_ids."""

    def __init__(self, records):
        self.records = dict(records)

    async def get_by_ids(self, ids):
        return [self.records.get(i) for i in ids]

    async def delete(self, ids):
        for i in ids:
            self.records.pop(i, None)


def _relation_record(src, tgt):
    return {"src_id": src, "tgt_id": tgt, "content": f"{src} {tgt}"}


async def test_deleting_one_colliding_edge_keeps_the_other_edge_vector():
    # Mixed state: both edges share one legacy alias, whose record names
    # the AB-C edge. Deleting A-BC must keep that alias (and AB-C's
    # versioned record) while removing A-BC's own versioned record.
    shared = compute_mdhash_id("ABC", prefix="rel-")
    versioned_a_bc = make_relation_vdb_id("A", "BC")
    versioned_ab_c = make_relation_vdb_id("AB", "C")
    vdb = _FakeRelationVDB(
        {
            shared: _relation_record("AB", "C"),
            versioned_a_bc: _relation_record("A", "BC"),
            versioned_ab_c: _relation_record("AB", "C"),
        }
    )

    ids = await filter_deletable_relation_vdb_ids(vdb, [("A", "BC")])
    assert versioned_a_bc in ids
    assert shared not in ids
    assert versioned_ab_c not in ids
    await vdb.delete(ids)
    assert shared in vdb.records
    assert versioned_ab_c in vdb.records
    assert versioned_a_bc not in vdb.records

    # Deleting the owning edge removes the shared alias with it.
    ids = await filter_deletable_relation_vdb_ids(vdb, [("AB", "C")])
    assert shared in ids
    assert versioned_ab_c in ids
    await vdb.delete(ids)
    assert shared not in vdb.records
    assert versioned_ab_c not in vdb.records


async def test_legacy_alias_without_endpoint_payload_is_retained():
    # A record that cannot be attributed to any edge must never be
    # deleted for an edge: losing data is worse than retaining an orphan
    # the rebuild rewrites or drops.
    shared = compute_mdhash_id("ABC", prefix="rel-")
    versioned_a_bc = make_relation_vdb_id("A", "BC")
    vdb = _FakeRelationVDB(
        {
            shared: {"content": "no endpoint fields"},
            versioned_a_bc: _relation_record("A", "BC"),
        }
    )

    ids = await filter_deletable_relation_vdb_ids(vdb, [("A", "BC")])
    assert versioned_a_bc in ids
    assert shared not in ids


async def test_missing_legacy_record_and_batch_claims():
    # A missing alias has nothing to delete; a batch deleting both
    # colliding edges removes the shared alias when its record names
    # one of the batch's edges.
    shared = compute_mdhash_id("ABC", prefix="rel-")
    versioned_a_bc = make_relation_vdb_id("A", "BC")
    versioned_ab_c = make_relation_vdb_id("AB", "C")

    vdb = _FakeRelationVDB({versioned_a_bc: _relation_record("A", "BC")})
    ids = await filter_deletable_relation_vdb_ids(vdb, [("A", "BC")])
    assert ids == [versioned_a_bc]

    vdb = _FakeRelationVDB(
        {
            shared: _relation_record("AB", "C"),
            versioned_a_bc: _relation_record("A", "BC"),
            versioned_ab_c: _relation_record("AB", "C"),
        }
    )
    ids = await filter_deletable_relation_vdb_ids(vdb, [("A", "BC"), ("AB", "C")])
    assert shared in ids
    assert versioned_a_bc in ids
    assert versioned_ab_c in ids


async def test_legacy_alias_matches_reversed_endpoints():
    # Edges are undirected, so a record naming the reversed pair still
    # names the edge being deleted.
    shared = compute_mdhash_id("ABC", prefix="rel-")
    versioned_a_bc = make_relation_vdb_id("A", "BC")
    vdb = _FakeRelationVDB(
        {
            shared: _relation_record("BC", "A"),
            versioned_a_bc: _relation_record("A", "BC"),
        }
    )

    ids = await filter_deletable_relation_vdb_ids(vdb, [("A", "BC")])
    assert shared in ids
    assert versioned_a_bc in ids
