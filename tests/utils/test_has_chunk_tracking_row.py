"""Truth table for ``has_chunk_tracking_row`` (#3609).

Row presence must be decided by schema — a dict carrying a ``chunk_ids``
list — never by list truthiness. A present-but-empty row is authoritative
("this object tracks no chunks"); an absent row or a legacy/partial shape
means UNKNOWN and is the only case that may fall back to the graph's
possibly-stale ``source_id``. This predicate is shared by every call site
(entity/relation edit, rename migration, entity merge, ingestion merge), so
its truth table is pinned once here.
"""

import pytest

from lightrag.utils import has_chunk_tracking_row

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "stored",
    [
        {"chunk_ids": [], "count": 0},  # curated empty row — authoritative
        {"chunk_ids": ["c1"], "count": 1},
        {"chunk_ids": ["c1", "c2"]},  # count missing is irrelevant
    ],
)
def test_present_rows(stored):
    assert has_chunk_tracking_row(stored) is True


@pytest.mark.parametrize(
    "stored",
    [
        None,  # absent row
        {},  # legacy/partial: no chunk_ids key
        {"count": 0},  # legacy/partial: no chunk_ids key
        {"chunk_ids": None},  # malformed: null instead of list
        {"chunk_ids": "c1"},  # malformed: scalar instead of list
        {"chunk_ids": {"c1": 1}},  # malformed: wrong container
        [],  # malformed: not a dict
        "row",  # malformed: not a dict
    ],
)
def test_absent_or_malformed_rows(stored):
    assert has_chunk_tracking_row(stored) is False
