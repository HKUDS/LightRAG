"""Tests for ``subtract_source_ids`` removal semantics.

The helper decides what a graph object still has attribution from after a purge
removes chunks: ``lightrag/lightrag.py`` computes ``remaining_sources`` from it
in both the entity and the relation branch of ``_purge_kg_contributions``, and
an object whose remaining sources are empty is *deleted* while a non-empty
result keeps it. Two properties are therefore load-bearing rather than cosmetic:

* **Falsy ids are dropped.** A stray ``""`` surviving into the result would make
  ``remaining_sources`` truthy and strand an object that owns no real chunk —
  precisely the unattributable orphan the purge contract exists to prevent.
* **Order is preserved.** The surviving list is written back as the object's
  chunk attribution, and ``apply_source_ids_limit`` truncates it positionally,
  so reordering would silently change which ids survive a later cap.

It had no direct tests; this file pins both, alongside the sibling
``test_merge_source_ids.py`` for the collection half of the same contract.
"""

from __future__ import annotations

import pytest

from lightrag.utils import subtract_source_ids

pytestmark = pytest.mark.offline


class TestRemoval:
    def test_listed_ids_are_removed(self):
        assert subtract_source_ids(["a", "b", "c"], ["b"]) == ["a", "c"]

    def test_removing_every_id_yields_an_empty_list(self):
        # The caller reads this as "no attribution left" and deletes the object.
        assert subtract_source_ids(["a", "b"], ["a", "b"]) == []

    def test_ids_absent_from_the_source_are_ignored(self):
        assert subtract_source_ids(["a"], ["b", "c"]) == ["a"]

    def test_an_empty_removal_collection_keeps_every_real_id(self):
        assert subtract_source_ids(["a", "b"], []) == ["a", "b"]

    def test_an_empty_source_yields_an_empty_list(self):
        assert subtract_source_ids([], ["a"]) == []


class TestOrderIsPreserved:
    def test_surviving_ids_keep_their_original_order(self):
        source = ["d", "a", "c", "b"]

        assert subtract_source_ids(source, ["c"]) == ["d", "a", "b"]

    def test_order_is_not_derived_from_the_removal_collection(self):
        # The removal argument is a set internally; its iteration order must not
        # leak into the result.
        assert subtract_source_ids(["a", "b", "c"], {"c", "a"}) == ["b"]

    def test_duplicate_survivors_are_kept_as_is(self):
        """Deduplication is not this helper's job.

        ``merge_source_ids`` is the normalization point; collapsing duplicates
        here would change the count that ``apply_source_ids_limit`` sees.
        """
        assert subtract_source_ids(["a", "a", "b"], ["b"]) == ["a", "a"]


class TestFalsyIdsAreDropped:
    @pytest.mark.parametrize(
        "removal",
        [
            pytest.param([], id="empty-removal-fast-path"),
            pytest.param(["x"], id="non-empty-removal-path"),
        ],
    )
    def test_empty_strings_are_dropped_on_both_branches(self, removal):
        """The no-removal fast path filters too.

        The implementation returns early when the removal set is empty, so the
        two paths filter falsy ids independently — an empty id leaking through
        either one would keep an object alive with no real attribution.
        """
        assert subtract_source_ids(["a", "", "b"], removal) == ["a", "b"]

    def test_an_all_empty_source_reduces_to_an_empty_list(self):
        # Not "has attribution" — the object must still be deletable.
        assert subtract_source_ids(["", ""], []) == []

    def test_none_entries_are_dropped(self):
        assert subtract_source_ids(["a", None, "b"], []) == ["a", "b"]


class TestInputTypes:
    def test_a_generator_source_is_accepted(self):
        assert subtract_source_ids((sid for sid in ["a", "b"]), ["b"]) == ["a"]

    def test_a_set_removal_collection_is_accepted(self):
        assert subtract_source_ids(["a", "b"], {"a"}) == ["b"]

    def test_a_tuple_removal_collection_is_accepted(self):
        assert subtract_source_ids(["a", "b"], ("a",)) == ["b"]

    def test_the_source_iterable_is_not_mutated(self):
        source = ["a", "b", "c"]

        subtract_source_ids(source, ["b"])

        assert source == ["a", "b", "c"]
