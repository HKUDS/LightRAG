"""Offline operator repair for chunk tracking (#3838, R4).

WHY it cannot be the startup migration: ``_migrate_chunk_tracking_storage`` is
gated on ``is_empty()``, so a single leftover row silently suppresses it for the
whole install, and it seeds from the graph ``source_id`` — a KEEP-truncated view
that chunk tracking is supposed to OUTRANK. Repairing a polluted row through it
therefore costs a provenance downgrade across every object.

The repair pins the issue's acceptance properties plus the conservative safety
boundary required by rename, merge, and explicit creation:

* it runs regardless of ``is_empty()``;
* it writes no invented row for an object with neither an existing authoritative
  row nor cached attribution;
* it retains current-object rows the cache cannot reproduce;
* nothing it writes originates from the graph ``source_id``;
* the attribution is chunk-granular (from the cache), never document-granular —
  attributing an object to every chunk of every document naming it is the same
  phantom evidence the issue exists to remove.
"""

import argparse
import json

import pytest

from lightrag.base import DocStatus
from lightrag.tools import chunk_tracking_repair
from lightrag.tools.chunk_tracking_repair import (
    apply_chunk_tracking_repair_plan,
    build_chunk_tracking_repair_plan,
)
from lightrag.utils import make_relation_chunk_key

pytestmark = pytest.mark.offline


def _extraction_payload(entities=(), relationships=()):
    """A JSON-format cached extraction result, the shape the rebuild path reads."""
    return json.dumps(
        {
            "entities": [
                {"name": name, "type": "person", "description": f"{name} description"}
                for name in entities
            ],
            "relationships": [
                {
                    "source": src,
                    "target": tgt,
                    "description": f"{src} relates to {tgt}",
                    "keywords": "relates",
                    "strength": 1,
                }
                for src, tgt in relationships
            ],
        }
    )


class _Doc:
    def __init__(self, chunks_list):
        self.chunks_list = chunks_list


class _DocStatus:
    def __init__(self, docs, *, boom: bool = False):
        self.docs = docs
        self.boom = boom
        self.strict_calls: list[bool] = []

    async def get_docs_by_statuses(self, statuses, strict: bool = False):
        self.strict_calls.append(strict)
        if self.boom:
            raise KeyError("unparseable doc_status row")
        return dict(self.docs)


class _KV:
    """Enough of ``BaseKVStorage`` for the repair, with drop/upsert bookkeeping."""

    def __init__(self, data=None):
        self.data = dict(data or {})
        self.drops = 0
        self.index_done_calls = 0
        self.is_empty_calls = 0

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, ids):
        return [self.data.get(key) for key in ids]

    async def filter_keys(self, keys):
        return {key for key in keys if key not in self.data}

    async def upsert(self, payload):
        self.data.update(payload)

    async def delete(self, ids):
        for key in ids:
            self.data.pop(key, None)

    async def is_empty(self):
        self.is_empty_calls += 1
        return not self.data

    async def drop(self):
        self.drops += 1
        self.data.clear()
        return {"status": "success", "message": "data dropped"}

    async def index_done_callback(self):
        self.index_done_calls += 1


class _Graph:
    """Existence oracle. ``source_id`` is present exactly so the repair can be
    caught reading it: every value here is absent from the cached attribution."""

    def __init__(self, labels, edges):
        self._labels = list(labels)
        self._edges = list(edges)

    async def get_all_labels(self):
        return list(self._labels)

    async def get_all_edges(self):
        return [
            {
                "source": src,
                "target": tgt,
                "source_id": "chunk-truncated-by-KEEP",
            }
            for src, tgt in self._edges
        ]


class _Repairer:
    def __init__(
        self,
        *,
        docs,
        chunks,
        cache,
        graph,
        entity_chunks=None,
        relation_chunks=None,
        doc_status=None,
    ):
        self.doc_status = doc_status or _DocStatus(docs)
        self.text_chunks = _KV(chunks)
        self.llm_response_cache = _KV(cache)
        self.chunk_entity_relation_graph = graph
        self.entity_chunks = entity_chunks if entity_chunks is not None else _KV()
        self.relation_chunks = relation_chunks if relation_chunks is not None else _KV()
        self.working_dir = "/tmp/test-rag"
        self.workspace = "test-workspace"
        self.initialized = False
        self.finalized = False

    async def initialize_storages(self):
        self.initialized = True

    async def finalize_storages(self):
        self.finalized = True


async def _repair(repairer, *, allow_missing_rows: bool = False):
    plan = await build_chunk_tracking_repair_plan(repairer)
    return await apply_chunk_tracking_repair_plan(
        repairer, plan, allow_missing_rows=allow_missing_rows
    )


def _two_chunk_corpus():
    """ALICE in both chunks, BOB only in c2, ALICE--BOB only in c2."""
    chunks = {
        "c1": {"content": "one", "llm_cache_list": ["cache-1"]},
        "c2": {"content": "two", "llm_cache_list": ["cache-2"]},
    }
    cache = {
        "cache-1": {
            "cache_type": "extract",
            "chunk_id": "c1",
            "return": _extraction_payload(entities=["ALICE"]),
            "create_time": 1,
        },
        "cache-2": {
            "cache_type": "extract",
            "chunk_id": "c2",
            "return": _extraction_payload(
                entities=["ALICE", "BOB"], relationships=[("ALICE", "BOB")]
            ),
            "create_time": 2,
        },
    }
    docs = {"doc-1": _Doc(["c1", "c2"])}
    graph = _Graph(["ALICE", "BOB"], [("ALICE", "BOB")])
    return docs, chunks, cache, graph


async def test_repair_runs_on_a_non_empty_store_and_evicts_the_orphan_row():
    """Not gated on ``is_empty()``.

    The store carries an orphan row (an object no longer in the graph, the
    residue a crash between a graph commit and its tracking delete leaves) and
    a polluted row. The empty-store migration would skip the whole namespace;
    the repair rewrites it.
    """
    docs, chunks, cache, graph = _two_chunk_corpus()
    entity_chunks = _KV(
        {
            "GHOST": {"chunk_ids": ["c-gone"], "count": 1},
            "ALICE": {"chunk_ids": ["c1", "c2"], "count": 2},
        }
    )
    relation_chunks = _KV(
        {
            make_relation_chunk_key("GHOST", "ALICE"): {
                "chunk_ids": ["c-gone"],
                "count": 1,
            }
        }
    )
    repairer = _Repairer(
        docs=docs,
        chunks=chunks,
        cache=cache,
        graph=graph,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
    )

    report = await _repair(repairer)

    assert entity_chunks.drops == 1
    assert relation_chunks.drops == 1
    assert "GHOST" not in entity_chunks.data
    assert entity_chunks.data["ALICE"]["chunk_ids"] == ["c1", "c2"]
    assert entity_chunks.data["BOB"]["chunk_ids"] == ["c2"]
    assert list(relation_chunks.data) == [make_relation_chunk_key("ALICE", "BOB")]
    assert relation_chunks.data[make_relation_chunk_key("ALICE", "BOB")] == {
        "chunk_ids": ["c2"],
        "count": 1,
    }
    # The gate the migration consults is never even asked.
    assert entity_chunks.is_empty_calls == 0
    assert relation_chunks.is_empty_calls == 0
    assert report.entity_rows_written == 2
    assert report.relation_rows_written == 1
    assert report.chunks_with_cache == 2
    assert report.chunks_without_cache == 0


async def test_plan_is_complete_and_read_only_before_apply():
    docs, chunks, cache, graph = _two_chunk_corpus()
    entity_chunks = _KV({"GHOST": {"chunk_ids": ["old"], "count": 1}})
    relation_chunks = _KV({"OLD": {"chunk_ids": ["old"], "count": 1}})
    repairer = _Repairer(
        docs=docs,
        chunks=chunks,
        cache=cache,
        graph=graph,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
    )

    plan = await build_chunk_tracking_repair_plan(repairer)

    assert plan.report.entity_rows_planned == 2
    assert plan.report.relation_rows_planned == 1
    assert entity_chunks.data == {"GHOST": {"chunk_ids": ["old"], "count": 1}}
    assert relation_chunks.data == {"OLD": {"chunk_ids": ["old"], "count": 1}}
    assert entity_chunks.drops == 0
    assert relation_chunks.drops == 0


async def test_no_row_is_written_for_an_object_without_cached_extraction():
    """Cache-less objects get NO row — not a row seeded from somewhere else.

    Their absence restores the purge classifier's ``source_id`` fallback, a
    known bounded degradation; a fabricated row would be phantom evidence.
    """
    docs, chunks, cache, _ = _two_chunk_corpus()
    # CAROL and CAROL--ALICE exist in the graph but no cached chunk names them.
    graph = _Graph(["ALICE", "BOB", "CAROL"], [("ALICE", "BOB"), ("CAROL", "ALICE")])
    # A third chunk whose extraction cache was cleared.
    chunks["c3"] = {"content": "three", "llm_cache_list": ["cache-3"]}
    docs["doc-1"].chunks_list.append("c3")

    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)
    plan = await build_chunk_tracking_repair_plan(repairer)
    assert any("graph entity" in reason for reason in plan.blockers({"entity_chunks"}))
    report = await apply_chunk_tracking_repair_plan(
        repairer, plan, allow_missing_rows=True
    )

    assert "CAROL" not in repairer.entity_chunks.data
    assert (
        make_relation_chunk_key("CAROL", "ALICE") not in repairer.relation_chunks.data
    )
    assert report.entities_without_tracking_row == 1
    assert report.relations_without_tracking_row == 1
    assert report.chunks_without_cache == 1
    assert any("no usable cached extraction" in w for w in report.warnings)


async def test_nothing_written_originates_from_the_graph_source_id():
    """The graph is an existence oracle only.

    Every node/edge here carries a ``source_id`` naming chunks that the cache
    never attributes. Seeding from it — what the startup migration does — would
    put those ids into the rebuilt rows and downgrade provenance install-wide.
    """
    docs, chunks, cache, _ = _two_chunk_corpus()

    class _SourceIdGraph(_Graph):
        async def get_all_nodes(self):  # pragma: no cover - must never be called
            raise AssertionError("the repair must not walk graph nodes for source_id")

    graph = _SourceIdGraph(["ALICE", "BOB"], [("ALICE", "BOB")])
    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)

    await _repair(repairer)

    written = [
        chunk_id
        for row in list(repairer.entity_chunks.data.values())
        + list(repairer.relation_chunks.data.values())
        for chunk_id in row["chunk_ids"]
    ]
    assert written, "the repair wrote nothing, so the assertion below is vacuous"
    assert "chunk-truncated-by-KEEP" not in written
    assert set(written) <= {"c1", "c2"}


async def test_existing_current_key_rows_survive_cache_name_mismatch():
    """Rename/merge/manual-create attribution cannot be reconstructed from the
    extraction cache because the cache still carries the original object name."""
    docs, chunks, cache, _ = _two_chunk_corpus()
    graph = _Graph(["ALICIA", "BOB"], [("ALICIA", "BOB")])
    entity_chunks = _KV(
        {
            "ALICIA": {"chunk_ids": ["c1", "c2"], "count": 2},
            "BOB": {"chunk_ids": ["c2"], "count": 1},
        }
    )
    relation_key = make_relation_chunk_key("ALICIA", "BOB")
    relation_chunks = _KV({relation_key: {"chunk_ids": ["c2"], "count": 1}})
    repairer = _Repairer(
        docs=docs,
        chunks=chunks,
        cache=cache,
        graph=graph,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
    )

    report = await _repair(repairer)

    assert entity_chunks.data["ALICIA"]["chunk_ids"] == ["c1", "c2"]
    assert relation_chunks.data[relation_key]["chunk_ids"] == ["c2"]
    assert report.existing_entity_rows == 2
    assert report.existing_relation_rows == 1


async def test_authoritative_empty_rows_from_manual_creation_survive():
    graph = _Graph(["MANUAL_A", "MANUAL_B"], [("MANUAL_A", "MANUAL_B")])
    relation_key = make_relation_chunk_key("MANUAL_A", "MANUAL_B")
    entity_chunks = _KV(
        {
            "MANUAL_A": {"chunk_ids": [], "count": 0},
            "MANUAL_B": {"chunk_ids": [], "count": 0},
        }
    )
    relation_chunks = _KV({relation_key: {"chunk_ids": [], "count": 0}})
    repairer = _Repairer(
        docs={},
        chunks={},
        cache={},
        graph=graph,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
    )

    await _repair(repairer)

    assert entity_chunks.data["MANUAL_A"] == {"chunk_ids": [], "count": 0}
    assert relation_chunks.data[relation_key] == {"chunk_ids": [], "count": 0}


async def test_attribution_is_chunk_granular_not_document_granular():
    """BOB is named only by c2, so its row must not inherit the doc's other chunks.

    Document-granularity seeding is explicitly forbidden: those rows feed
    ``existing_full_source_ids`` in ``_merge_edges_then_upsert``, breaking the
    relation weight contract and misclassifying chunk-subset purges.
    """
    docs, chunks, cache, graph = _two_chunk_corpus()
    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)

    await _repair(repairer)

    assert repairer.entity_chunks.data["BOB"]["chunk_ids"] == ["c2"]
    assert repairer.relation_chunks.data[make_relation_chunk_key("ALICE", "BOB")][
        "chunk_ids"
    ] == ["c2"]


async def test_objects_absent_from_the_graph_are_not_resurrected():
    """A cached chunk still naming a manually deleted entity must not re-create
    its row — that is the very orphan the repair exists to remove."""
    docs, chunks, cache, _ = _two_chunk_corpus()
    graph = _Graph(["ALICE"], [])  # BOB and ALICE--BOB were deleted by an admin call

    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)
    report = await _repair(repairer)

    assert list(repairer.entity_chunks.data) == ["ALICE"]
    assert repairer.relation_chunks.data == {}
    assert report.relation_rows_written == 0


async def test_doc_status_is_read_complete_or_raise_and_nothing_is_dropped():
    """The chunk universe decides which rows survive, so a partial read must
    abort BEFORE the drop rather than silently shrink the rebuild."""
    docs, chunks, cache, graph = _two_chunk_corpus()
    entity_chunks = _KV({"ALICE": {"chunk_ids": ["c1"], "count": 1}})
    relation_chunks = _KV()
    repairer = _Repairer(
        docs=docs,
        chunks=chunks,
        cache=cache,
        graph=graph,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
        doc_status=_DocStatus(docs, boom=True),
    )

    with pytest.raises(KeyError):
        await _repair(repairer)

    assert repairer.doc_status.strict_calls == [True]
    assert entity_chunks.drops == 0
    assert relation_chunks.drops == 0
    assert entity_chunks.data == {"ALICE": {"chunk_ids": ["c1"], "count": 1}}


async def test_every_doc_status_state_contributes_its_chunks():
    """A FAILED or in-flight document still owns real chunks whose cached
    extraction is real evidence; restricting to PROCESSED would drop rows."""
    docs, chunks, cache, graph = _two_chunk_corpus()
    doc_status = _DocStatus(docs)
    repairer = _Repairer(
        docs=docs, chunks=chunks, cache=cache, graph=graph, doc_status=doc_status
    )

    captured: list[list] = []
    original = doc_status.get_docs_by_statuses

    async def _spy(statuses, strict: bool = False):
        captured.append(list(statuses))
        return await original(statuses, strict=strict)

    doc_status.get_docs_by_statuses = _spy
    await _repair(repairer)

    assert set(captured[0]) == set(DocStatus)


async def test_report_distinguishes_total_documents_and_documents_with_chunks():
    docs, chunks, cache, graph = _two_chunk_corpus()
    docs["doc-without-chunks"] = _Doc([])
    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)

    plan = await build_chunk_tracking_repair_plan(repairer)

    assert plan.report.scanned_documents == 2
    assert plan.report.documents_with_chunks == 1


async def test_report_distinguishes_cached_results_from_extracted_attribution():
    docs, chunks, cache, graph = _two_chunk_corpus()
    cache["cache-1"]["return"] = _extraction_payload()
    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)

    plan = await build_chunk_tracking_repair_plan(repairer)

    assert plan.report.chunks_with_cache == 2
    assert plan.report.chunks_with_attribution == 1
    assert plan.report.chunks_without_cache == 0


async def test_chunks_missing_from_text_chunks_are_skipped():
    """``chunks_list`` can name chunks a failed ingestion never wrote."""
    docs, chunks, cache, graph = _two_chunk_corpus()
    docs["doc-2"] = _Doc(["c-never-written"])

    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)
    report = await _repair(repairer)

    assert report.scanned_chunks == 3
    assert report.chunks_with_cache == 2
    assert report.chunks_without_cache == 1
    assert repairer.entity_chunks.data["ALICE"]["chunk_ids"] == ["c1", "c2"]


async def test_empty_result_is_refused_before_the_first_drop():
    """An empty replacement for a non-empty graph would be re-seeded from
    ``source_id`` at startup, so apply must fail closed before mutation."""
    docs, chunks, _, graph = _two_chunk_corpus()
    entity_chunks = _KV({"GHOST": {"chunk_ids": ["c1"], "count": 1}})
    relation_chunks = _KV({"OLD": {"chunk_ids": ["c2"], "count": 1}})
    repairer = _Repairer(
        docs=docs,
        chunks=chunks,
        cache={},
        graph=graph,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
    )

    plan = await build_chunk_tracking_repair_plan(repairer)

    assert plan.unsafe_empty_namespaces == ["entity_chunks", "relation_chunks"]
    with pytest.raises(ValueError, match="Refusing to drop tracking"):
        await apply_chunk_tracking_repair_plan(repairer, plan)
    assert entity_chunks.drops == 0
    assert relation_chunks.drops == 0
    assert entity_chunks.data["GHOST"]["chunk_ids"] == ["c1"]


async def test_a_failed_drop_aborts_instead_of_reporting_success():
    """``drop()`` reports failure in its return value; a silent no-op would leave
    the stale rows in place under a "repaired" answer."""
    docs, chunks, cache, graph = _two_chunk_corpus()

    class _RefusingKV(_KV):
        async def drop(self):
            self.drops += 1
            return {"status": "error", "message": "backend refused"}

    entity_chunks = _RefusingKV({"GHOST": {"chunk_ids": ["c-gone"], "count": 1}})
    repairer = _Repairer(
        docs=docs,
        chunks=chunks,
        cache=cache,
        graph=graph,
        entity_chunks=entity_chunks,
    )

    with pytest.raises(RuntimeError, match="Failed to drop entity_chunks"):
        await _repair(repairer)
    assert entity_chunks.data == {"GHOST": {"chunk_ids": ["c-gone"], "count": 1}}


async def test_entity_write_failure_leaves_relation_namespace_untouched():
    docs, chunks, cache, graph = _two_chunk_corpus()

    class _FailingUpsertKV(_KV):
        async def upsert(self, payload):
            raise RuntimeError("write failed")

    entity_chunks = _FailingUpsertKV()
    relation_chunks = _KV({"OLD": {"chunk_ids": ["old"], "count": 1}})
    repairer = _Repairer(
        docs=docs,
        chunks=chunks,
        cache=cache,
        graph=graph,
        entity_chunks=entity_chunks,
        relation_chunks=relation_chunks,
    )

    with pytest.raises(RuntimeError, match="write failed"):
        await _repair(repairer)

    assert entity_chunks.drops == 1
    assert relation_chunks.drops == 0
    assert relation_chunks.data == {"OLD": {"chunk_ids": ["old"], "count": 1}}


async def test_repair_requires_chunk_tracking_to_be_configured():
    docs, chunks, cache, graph = _two_chunk_corpus()
    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)
    repairer.entity_chunks = None

    with pytest.raises(ValueError, match="requires configured storages"):
        await _repair(repairer)


async def test_a_legitimately_relation_free_corpus_does_not_warn_about_a_reseed():
    """The re-seed warning is about a namespace that SHOULD have rows and has
    none. A graph with no edges leaves relation_chunks empty by construction,
    and the startup migration would find nothing to re-seed from either."""
    docs, chunks, cache, _ = _two_chunk_corpus()
    graph = _Graph(["ALICE", "BOB"], [])

    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)
    report = await _repair(repairer)

    assert report.entity_rows_written == 2
    assert report.relation_rows_written == 0
    assert not any("re-seed" in w for w in report.warnings)


async def test_cli_requires_offline_confirmation_before_storage_initialization(
    monkeypatch,
):
    async def _must_not_build():  # pragma: no cover - assertion is the test
        raise AssertionError("storage must not initialize before confirmation")

    monkeypatch.setattr(chunk_tracking_repair, "_confirm_offline", lambda _yes: False)
    monkeypatch.setattr(chunk_tracking_repair, "_build_rag", _must_not_build)

    result = await chunk_tracking_repair.run(
        argparse.Namespace(
            apply=False,
            yes=False,
            namespace="both",
            allow_empty_graph=False,
            allow_missing_rows=False,
        )
    )

    assert result is True


async def test_cli_defaults_to_a_read_only_plan(monkeypatch):
    docs, chunks, cache, graph = _two_chunk_corpus()
    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)

    async def _build():
        return repairer

    monkeypatch.setattr(chunk_tracking_repair, "_build_rag", _build)

    result = await chunk_tracking_repair.run(
        argparse.Namespace(
            apply=False,
            yes=True,
            namespace="both",
            allow_empty_graph=False,
            allow_missing_rows=False,
        )
    )

    assert result is True
    assert repairer.initialized is True
    assert repairer.finalized is True
    assert repairer.entity_chunks.drops == 0
    assert repairer.relation_chunks.drops == 0


async def test_cli_returns_failure_for_an_unsafe_apply_without_dropping(monkeypatch):
    docs, chunks, _, graph = _two_chunk_corpus()
    repairer = _Repairer(docs=docs, chunks=chunks, cache={}, graph=graph)

    async def _build():
        return repairer

    monkeypatch.setattr(chunk_tracking_repair, "_build_rag", _build)

    result = await chunk_tracking_repair.run(
        argparse.Namespace(
            apply=True,
            yes=True,
            namespace="both",
            allow_empty_graph=False,
            allow_missing_rows=False,
        )
    )

    assert result is False
    assert repairer.entity_chunks.drops == 0
    assert repairer.relation_chunks.drops == 0
    assert repairer.finalized is True


async def test_cli_unsafe_dry_run_returns_failure(monkeypatch):
    docs, chunks, _, graph = _two_chunk_corpus()
    repairer = _Repairer(docs=docs, chunks=chunks, cache={}, graph=graph)

    async def _build():
        return repairer

    monkeypatch.setattr(chunk_tracking_repair, "_build_rag", _build)

    result = await chunk_tracking_repair.run(
        argparse.Namespace(
            apply=False,
            yes=True,
            namespace="both",
            allow_empty_graph=False,
            allow_missing_rows=False,
        )
    )

    assert result is False
    assert repairer.entity_chunks.drops == 0
    assert repairer.relation_chunks.drops == 0


async def test_empty_graph_requires_explicit_override_before_apply():
    entity_chunks = _KV({"ORPHAN": {"chunk_ids": ["c1"], "count": 1}})
    repairer = _Repairer(
        docs={},
        chunks={},
        cache={},
        graph=_Graph([], []),
        entity_chunks=entity_chunks,
    )
    plan = await build_chunk_tracking_repair_plan(repairer)

    assert plan.blockers({"entity_chunks"}) == ["graph is empty or unavailable"]
    with pytest.raises(ValueError, match="graph is empty or unavailable"):
        await apply_chunk_tracking_repair_plan(
            repairer, plan, namespaces={"entity_chunks"}
        )
    assert entity_chunks.drops == 0

    await apply_chunk_tracking_repair_plan(
        repairer,
        plan,
        namespaces={"entity_chunks"},
        allow_empty_graph=True,
    )
    assert entity_chunks.drops == 1
    assert entity_chunks.data == {}


async def test_namespace_scope_does_not_let_relation_block_entity_repair():
    docs, chunks, cache, _ = _two_chunk_corpus()
    graph = _Graph(["ALICE", "BOB"], [("CAROL", "DAVE")])
    repairer = _Repairer(docs=docs, chunks=chunks, cache=cache, graph=graph)
    plan = await build_chunk_tracking_repair_plan(repairer)

    assert plan.unsafe_empty_namespaces == ["relation_chunks"]
    await apply_chunk_tracking_repair_plan(repairer, plan, namespaces={"entity_chunks"})

    assert repairer.entity_chunks.drops == 1
    assert repairer.relation_chunks.drops == 0
