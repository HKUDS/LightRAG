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
