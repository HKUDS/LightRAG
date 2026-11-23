"""Regression tests for ``join_unique`` ordering in ``_merge_attributes``.

``_merge_attributes`` is the strategy dispatcher behind ``amerge_entities`` /
``amerge_relations``. Both ``source_id`` and ``file_path`` are merged with the
``join_unique`` strategy, which used to collect items into a ``set`` before
joining them -- so the merged order was whatever ``str.__hash__`` produced, and
``PYTHONHASHSEED`` randomization made it differ from one process to the next.

That order is load-bearing, differently for each of the two fields:

``source_id``
    ``apply_source_ids_limit`` truncates positionally (FIFO keeps the tail,
    IGNORE_NEW keeps the head). Chunk tracking outranks the graph's
    ``source_id``, so when a tracking row exists it supplies the order and the
    graph string is rewritten from it. The damage lands on the fallback path,
    where no tracking row exists -- legacy graphs, or a merge whose inputs had
    no usable row and which therefore writes no target row either. There the
    graph string is the only authority, and both limit strategies dropped
    arbitrary chunks instead of the ones they are named after.

``file_path``
    Has no tracking side-channel at all: the graph string is the only store,
    and ``_merge_nodes_then_upsert`` (``_merge_edges_then_upsert`` for
    relations) caps it at ``max_file_paths`` by the same positional rule. This
    field is affected unconditionally.

Preserving first-seen order also lines the merged ``source_id`` up with the
chunk tracking row the same merge writes, which already deduplicates in order
(source entities first, then the target).

**Why the fixtures use twenty chunk IDs.** A ``set`` of six short, similar
strings reproduces insertion order outright under roughly a fifth of the
possible hash seeds, so a small fixture lets the set-based implementation pass
by coincidence -- including under ``PYTHONHASHSEED=0``, the value CI systems
pin for reproducibility. At twenty IDs no checked seed (0-5 and random ones)
reproduces insertion order, either in whole or in either ten-element half, so
every assertion below fails against the old implementation under whatever
seed the test process runs with. That is also why no test starts interpreters
under several seeds: each of these already fails under any one of them.
"""

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import apply_source_ids_limit
from lightrag.utils_graph import _merge_attributes


pytestmark = pytest.mark.offline

# Set iteration order is sensitive to the exact strings, and this spelling is
# the one the seed check covered: do not renumber or re-pad these.
FIRST_HALF = [f"chunk-{i:02d}" for i in range(1, 11)]
SECOND_HALF = [f"chunk-{i:02d}" for i in range(11, 21)]
ALL_CHUNKS = FIRST_HALF + SECOND_HALF
HALF = len(FIRST_HALF)


@pytest.mark.parametrize("key", ["source_id", "file_path"])
def test_join_unique_preserves_first_seen_order(key):
    # Both provenance fields use join_unique, and file_path has no tracking
    # side-channel -- the order stored here is the only order it ever has.
    first = {key: GRAPH_FIELD_SEP.join(FIRST_HALF)}
    second = {key: GRAPH_FIELD_SEP.join(SECOND_HALF)}

    merged = _merge_attributes([first, second], {key: "join_unique"})

    assert merged[key].split(GRAPH_FIELD_SEP) == ALL_CHUNKS


def test_join_unique_dedups_without_reordering():
    # Overlapping provenance: the shared IDs keep their first position instead
    # of moving to where the second entity mentions them again.
    overlap = FIRST_HALF[-3:]
    first = {"source_id": GRAPH_FIELD_SEP.join(FIRST_HALF)}
    second = {"source_id": GRAPH_FIELD_SEP.join(overlap + SECOND_HALF)}

    merged = _merge_attributes([first, second], {"source_id": "join_unique"})

    assert merged["source_id"].split(GRAPH_FIELD_SEP) == ALL_CHUNKS


def test_join_unique_order_lets_fifo_keep_the_newest_chunks():
    # A merged entity over max_source_ids_per_entity on the fallback path (no
    # chunk tracking row). FIFO keeps the tail, so it must keep the chunks the
    # entity merged last contributed.
    merged = _merge_attributes(
        [
            {"source_id": GRAPH_FIELD_SEP.join(FIRST_HALF)},
            {"source_id": GRAPH_FIELD_SEP.join(SECOND_HALF)},
        ],
        {"source_id": "join_unique"},
    )

    kept = apply_source_ids_limit(
        merged["source_id"].split(GRAPH_FIELD_SEP), HALF, "FIFO"
    )

    assert kept == SECOND_HALF


def test_join_unique_order_lets_ignore_new_keep_the_oldest_chunks():
    # The counterpart, so the fix cannot pass by reversing the list instead:
    # IGNORE_NEW keeps the head.
    merged = _merge_attributes(
        [
            {"source_id": GRAPH_FIELD_SEP.join(FIRST_HALF)},
            {"source_id": GRAPH_FIELD_SEP.join(SECOND_HALF)},
        ],
        {"source_id": "join_unique"},
    )

    kept = apply_source_ids_limit(
        merged["source_id"].split(GRAPH_FIELD_SEP), HALF, "IGNORE_NEW"
    )

    assert kept == FIRST_HALF
