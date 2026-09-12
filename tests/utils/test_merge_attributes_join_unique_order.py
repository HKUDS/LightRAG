"""Regression tests for ``join_unique`` ordering in ``_merge_attributes``.

``_merge_attributes`` is the strategy dispatcher behind ``amerge_entities`` /
``amerge_relations``. ``source_id`` and ``file_path`` are merged with the
``join_unique`` strategy, which used to collect items into a ``set`` before
joining them.

Set iteration order for strings is not stable across processes (PYTHONHASHSEED
randomization), and the order of ``source_id`` is load-bearing downstream:
``apply_source_ids_limit`` truncates positionally, keeping the tail under
``FIFO`` and the head under ``IGNORE_NEW``. A randomly ordered ``source_id``
therefore made both limit strategies discard arbitrary chunks instead of the
ones they name.

The sibling branch in the same dispatcher, ``join_unique_comma``, already
sorted its output, and the codebase uses ``dict.fromkeys`` for order-preserving
dedup in ~18 other places -- including three in ``utils_graph`` itself.
"""

import subprocess
import sys
import textwrap

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import apply_source_ids_limit
from lightrag.utils_graph import _merge_attributes


def test_join_unique_preserves_first_seen_order():
    # Scenario: two entities merged, each carrying its own chunk provenance.
    first = {"source_id": GRAPH_FIELD_SEP.join(["chunk-01", "chunk-02"])}
    second = {"source_id": GRAPH_FIELD_SEP.join(["chunk-03", "chunk-04"])}

    merged = _merge_attributes([first, second], {"source_id": "join_unique"})

    assert merged["source_id"].split(GRAPH_FIELD_SEP) == [
        "chunk-01",
        "chunk-02",
        "chunk-03",
        "chunk-04",
    ]


def test_join_unique_dedups_without_reordering():
    # Scenario: overlapping provenance -- the duplicate keeps its first position.
    first = {"source_id": GRAPH_FIELD_SEP.join(["chunk-01", "chunk-02"])}
    second = {"source_id": GRAPH_FIELD_SEP.join(["chunk-02", "chunk-03"])}

    merged = _merge_attributes([first, second], {"source_id": "join_unique"})

    assert merged["source_id"].split(GRAPH_FIELD_SEP) == [
        "chunk-01",
        "chunk-02",
        "chunk-03",
    ]


def test_join_unique_order_lets_fifo_keep_the_newest_chunks():
    # Scenario: the merged entity exceeds max_source_ids_per_entity.
    # FIFO keeps the tail, so it must keep the most recently merged chunks.
    first = {"source_id": GRAPH_FIELD_SEP.join(["chunk-01", "chunk-02", "chunk-03"])}
    second = {"source_id": GRAPH_FIELD_SEP.join(["chunk-04", "chunk-05", "chunk-06"])}

    merged = _merge_attributes([first, second], {"source_id": "join_unique"})
    kept = apply_source_ids_limit(merged["source_id"].split(GRAPH_FIELD_SEP), 3, "FIFO")

    assert kept == ["chunk-04", "chunk-05", "chunk-06"]


def test_join_unique_order_lets_ignore_new_keep_the_oldest_chunks():
    # Scenario: the counterpart -- IGNORE_NEW keeps the head.
    first = {"source_id": GRAPH_FIELD_SEP.join(["chunk-01", "chunk-02", "chunk-03"])}
    second = {"source_id": GRAPH_FIELD_SEP.join(["chunk-04", "chunk-05", "chunk-06"])}

    merged = _merge_attributes([first, second], {"source_id": "join_unique"})
    kept = apply_source_ids_limit(
        merged["source_id"].split(GRAPH_FIELD_SEP), 3, "IGNORE_NEW"
    )

    assert kept == ["chunk-01", "chunk-02", "chunk-03"]


def test_join_unique_is_stable_across_processes():
    """Same input, fresh interpreters: the merged order must not change.

    An in-process assertion cannot catch this -- ``PYTHONHASHSEED`` is fixed
    once per interpreter, so a set-based implementation looks deterministic
    when tested inside a single process. Only separate processes expose it.
    """
    probe = textwrap.dedent(
        """
        from lightrag.constants import GRAPH_FIELD_SEP
        from lightrag.utils_graph import _merge_attributes

        first = {"source_id": GRAPH_FIELD_SEP.join(["chunk-01", "chunk-02", "chunk-03"])}
        second = {"source_id": GRAPH_FIELD_SEP.join(["chunk-04", "chunk-05", "chunk-06"])}
        merged = _merge_attributes([first, second], {"source_id": "join_unique"})
        print(merged["source_id"].replace(GRAPH_FIELD_SEP, ","))
        """
    )

    results = set()
    for _ in range(5):
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        results.add(completed.stdout.strip().splitlines()[-1])

    assert results == {"chunk-01,chunk-02,chunk-03,chunk-04,chunk-05,chunk-06"}
