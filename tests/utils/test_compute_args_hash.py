"""Regression tests for ``compute_args_hash`` cache-key collision.

Verifies the length-prefixed encoding closes the delimiter-less join bug
reported in issue #3392, where adjacent field boundaries could shift a
character and produce identical MD5 cache keys for semantically distinct
argument tuples.

Coverage:
- The two collision cases from the issue now produce distinct hashes.
- Inputs containing the ``"\\x1e"`` record-separator (or any delimiter a
  sentinel-based fix might have chosen) still do not collide.
- Identical argument tuples still hash identically (cache HIT preserved).
- Single-argument calls preserve the legacy hash value verbatim, so
  ``compute_mdhash_id`` document IDs remain stable across upgrades.
- Surrogate / invalid Unicode handling does not regress.
"""

from __future__ import annotations

import hashlib

import pytest

from lightrag.utils import compute_args_hash, compute_mdhash_id


pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Collision cases from issue #3392 — must now produce DISTINCT hashes.
# ---------------------------------------------------------------------------


def test_adjacent_text_field_boundary_does_not_collide():
    """Reproduces issue case 1: (query, response_type) boundary shift."""
    a = compute_args_hash("hybrid", "abc", "x", 10, 20, 6000, 8000, 30000)
    b = compute_args_hash("hybrid", "ab", "cx", 10, 20, 6000, 8000, 30000)
    assert a != b, (
        f"Adjacent free-text boundary shift must not collide, both hashed to {a}"
    )


def test_adjacent_integer_field_boundary_does_not_collide():
    """Reproduces issue case 2: (top_k, chunk_top_k) boundary shift."""
    c = compute_args_hash("hybrid", "q", "text", 1, 20, 6000, 8000, 30000)
    d = compute_args_hash("hybrid", "q", "text", 12, 0, 6000, 8000, 30000)
    assert c != d, (
        f"Adjacent integer boundary shift must not collide, both hashed to {c}"
    )


# ---------------------------------------------------------------------------
# Sentinel robustness: even inputs containing a would-be delimiter character
# must not collide, because the encoding is length-based, not delimiter-based.
# ---------------------------------------------------------------------------


def test_inputs_containing_record_separator_do_not_collide():
    """A sentinel-based fix would still be vulnerable to inputs that contain
    the sentinel character. Length-prefixing must be immune."""
    e = compute_args_hash("a\x1eb", "c")
    f = compute_args_hash("a", "b\x1ec")
    assert e != f, "Inputs containing the record-separator character must not collide"


# ---------------------------------------------------------------------------
# Cache HIT preserved: identical tuples still hash identically.
# ---------------------------------------------------------------------------


def test_identical_arguments_hash_identically():
    g = compute_args_hash("hybrid", "abc", "x", 10, 20, 6000, 8000, 30000)
    h = compute_args_hash("hybrid", "abc", "x", 10, 20, 6000, 8000, 30000)
    assert g == h, "Identical argument tuples must hash identically (cache HIT)"


# ---------------------------------------------------------------------------
# Single-argument stability: document IDs (compute_mdhash_id) must not change,
# otherwise persisted storage records break on upgrade.
# ---------------------------------------------------------------------------


def test_single_argument_preserves_legacy_hash():
    """``compute_mdhash_id`` builds document IDs via single-arg
    ``compute_args_hash(content)``. Those IDs are persisted in KV/vector/graph
    storage, so the single-arg hash value must remain the raw MD5 of the
    content string — unchanged across this fix."""
    content = "hello world test content"
    expected_raw_md5 = hashlib.md5(content.encode("utf-8")).hexdigest()

    # No prefix
    assert compute_mdhash_id(content) == expected_raw_md5, (
        "Single-arg compute_args_hash must equal raw MD5 of the content"
    )
    # With prefix (document / entity / relation IDs)
    assert compute_mdhash_id(content, prefix="doc-") == "doc-" + expected_raw_md5


def test_single_argument_empty_string_preserves_legacy_hash():
    """Empty content must still hash to the well-known empty-MD5 value."""
    empty_md5 = hashlib.md5(b"").hexdigest()
    assert compute_args_hash("") == empty_md5
    assert compute_mdhash_id("") == empty_md5


# ---------------------------------------------------------------------------
# Unicode safety: surrogate / invalid characters must not raise and must still
# produce a stable hash (regression guard for the 'replace' error handler).
# ---------------------------------------------------------------------------


def test_surrogate_unicode_does_not_raise_and_is_stable():
    # lone surrogate that cannot encode as UTF-8 directly
    bad = "before\udcffafter"
    h1 = compute_args_hash(bad, "field2")
    h2 = compute_args_hash(bad, "field2")
    assert isinstance(h1, str) and len(h1) == 32, "must return a 32-char MD5 hex"
    assert h1 == h2, "same surrogate input must produce a stable hash"


def test_two_args_with_surrogate_still_distinguishable():
    """The 'replace' error handler must not erase field distinctions."""
    a = compute_args_hash("ab\udcff", "xy")
    b = compute_args_hash("ab", "\udcffxy")
    assert a != b, "Surrogate replacement must not collapse distinct fields"
