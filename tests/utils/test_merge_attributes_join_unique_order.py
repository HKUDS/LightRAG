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
pin for reproducibility. At twenty IDs no probed seed reproduces insertion
order, either in whole or in either ten-element half, so every assertion below
fails deterministically against the old implementation rather than only for
most seeds.
"""

import contextlib
import os
import subprocess
import sys
import textwrap

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import apply_source_ids_limit
from lightrag.utils_graph import _merge_attributes

# Set iteration order is sensitive to the exact strings, and this spelling is
# the one the seed probe covered: do not renumber or re-pad these.
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


_PROBE = textwrap.dedent(
    """
    from lightrag.constants import GRAPH_FIELD_SEP
    from lightrag.utils_graph import _merge_attributes

    first = [f"chunk-{i:02d}" for i in range(1, 11)]
    second = [f"chunk-{i:02d}" for i in range(11, 21)]
    merged = _merge_attributes(
        [
            {"source_id": GRAPH_FIELD_SEP.join(first)},
            {"source_id": GRAPH_FIELD_SEP.join(second)},
        ],
        {"source_id": "join_unique"},
    )
    print(merged["source_id"].replace(GRAPH_FIELD_SEP, ","))
    """
)

# PYTHONHASHSEED is fixed once per interpreter, so the seed has to be chosen
# before the process starts -- which is why this runs out of process at all.
# Inheriting the caller's seed is not enough: a CI runner that pins
# PYTHONHASHSEED=0 for reproducibility would hand every probe the same seed and
# the test would silently lose its power to detect variance. The seeds are
# passed explicitly instead, and 0 is deliberately one of them because it is
# the value that turns hash randomization off.
_PROBE_SEEDS = ("0", "1", "2")


def test_join_unique_order_is_identical_under_every_hash_seed():
    """The merged order must not depend on how the IDs happen to hash."""
    # Launched together rather than in sequence: the probes are independent and
    # each pays a fresh interpreter start plus a lightrag import.
    merged_orders = {}
    with contextlib.ExitStack() as stack:
        # Entered as context managers so a failing assertion below still closes
        # and reaps the probes that have not been read yet.
        processes = [
            stack.enter_context(
                subprocess.Popen(
                    [sys.executable, "-c", _PROBE],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    # Inherit the environment so the probe can import lightrag
                    # at all; override only the seed.
                    env={**os.environ, "PYTHONHASHSEED": seed},
                )
            )
            for seed in _PROBE_SEEDS
        ]

        for seed, process in zip(_PROBE_SEEDS, processes):
            stdout, stderr = process.communicate()
            assert process.returncode == 0, (
                f"probe failed (PYTHONHASHSEED={seed}): {stderr}"
            )
            merged_orders[seed] = stdout.strip().splitlines()[-1]

    assert set(merged_orders.values()) == {",".join(ALL_CHUNKS)}, merged_orders
