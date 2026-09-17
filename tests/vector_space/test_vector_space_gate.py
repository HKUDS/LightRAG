"""The two cross-storage startup checks (issue #3978, PR 9).

One invariant governs nearly every case here:

    Only a probe that ran and returned a negative verdict may refuse. Every
    form of "it could not run" falls back to the pre-upgrade behaviour.

So the tests come in pairs: one that the check fires on real evidence, and one
that it stays out of the way when the evidence is missing, unreadable or
ambiguous. A safety feature that turns a provider outage into a startup failure
is worse than the gap it closes.

See docs/design/VectorSpaceProvenance.md.
"""

import asyncio

import numpy as np
import pytest

from lightrag.base import DocStatus
from lightrag.exceptions import (
    StorageCapabilityError,
    VectorSpaceMismatchError,
    VectorStorageEmptyError,
)
from lightrag.utils import compute_mdhash_id
from lightrag.vector_space_gate import (
    ADOPT_COSINE,
    REFUSE_COSINE,
    _cosine,
    check_vector_space_at_startup,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeGraph:
    """A graph storage that can answer, be empty, or be broken."""

    def __init__(self, labels=(), error=None, edges=None, edge_error=None):
        self._labels = list(labels)
        self._error = error
        # Default: one edge per pair of labels, so a graph with entities also
        # has relations unless a test says otherwise.
        self._edges = (
            list(edges)
            if edges is not None
            else [
                {"source": a, "target": b}
                for a, b in zip(self._labels, self._labels[1:])
            ]
        )
        self._edge_error = edge_error
        self.calls = 0

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._labels[:limit]

    async def iter_edges(self, batch_size: int):
        if self._edge_error is not None:
            raise self._edge_error
        for start in range(0, len(self._edges), batch_size):
            yield self._edges[start : start + batch_size]


class FakeDocStatus:
    """Doc-status with just the one question the gate asks."""

    def __init__(self, processed=1, error=None, unfinished=0):
        self._processed = processed
        self._error = error
        self._unfinished = unfinished

    async def count_docs_by_statuses(self, statuses, *, strict=True):
        if self._error is not None:
            raise self._error
        return sum(
            self._processed if status is DocStatus.PROCESSED else 0
            for status in statuses
        ) + sum(
            self._unfinished if status is DocStatus.PROCESSING else 0
            for status in statuses
        )


class FakeVectorStorage:
    """A vector storage with just the surface the gate touches."""

    final_namespace = "vdb_entities_probe"
    namespace = "entities"

    persists_vectors = True

    def __init__(
        self,
        rows=None,
        vectors=None,
        *,
        pending=False,
        adopt_result=True,
        adopt_error=None,
        read_error=None,
        vector_read_error=None,
        empty_error=None,
    ):
        self.reads = 0
        self.empty_reads = 0
        self._empty_error = empty_error
        self._rows = list(rows or [])
        self._vectors = dict(vectors or {})
        self._pending = pending
        self._adopt_result = adopt_result
        self._adopt_error = adopt_error
        self._read_error = read_error
        self._vector_read_error = vector_read_error
        self.adopted = 0

    async def is_empty(self) -> bool:
        self.empty_reads += 1
        if self._empty_error is not None:
            raise self._empty_error
        return not self._rows

    async def get_by_ids(self, ids):
        self.reads += 1
        if self._read_error is not None:
            raise self._read_error
        wanted = set(ids)
        return [row for row in self._rows if row.get("id") in wanted]

    async def get_vectors_by_ids(self, ids):
        if self._vector_read_error is not None:
            raise self._vector_read_error
        return {k: v for k, v in self._vectors.items() if k in set(ids)}

    async def vector_space_adoption_pending(self) -> bool:
        return self._pending

    async def adopt_vector_space(self) -> bool:
        self.adopted += 1
        if self._adopt_error is not None:
            raise self._adopt_error
        return self._adopt_result


class FakeEmbedding:
    """Returns a fixed vector, or fails, or hangs."""

    model_name = "probe-model"
    embedding_dim = 4

    def __init__(self, vector=None, *, error=None, hang=False):
        self._vector = vector
        self._error = error
        self._hang = hang
        self.calls = 0

    async def __call__(self, texts, **kwargs):
        self.calls += 1
        if self._hang:
            await asyncio.sleep(3600)
        if self._error is not None:
            raise self._error
        return np.array([list(self._vector) for _ in texts], dtype=np.float32)


class FakeKVStorage:
    """``text_chunks`` -- the source side of the chunk pairing.

    ``is_empty`` mirrors the REAL ``BaseKVStorage`` contract, which catches its
    backend errors and answers ``True``. The gate depends on that: an
    unreadable source lands on the same branch as an empty one (skip), which is
    the safe direction. The vector side is the opposite contract, and
    ``FakeVectorStorage.is_empty`` raises to match it.
    """

    def __init__(self, *, rows=0, error=None):
        self._rows = rows
        self._error = error
        self.empty_reads = 0

    async def is_empty(self) -> bool:
        self.empty_reads += 1
        if self._error is not None:
            return True  # what every real KV backend does on failure
        return self._rows == 0


def _entity_row(name="Alice", content="Alice is an engineer."):
    return {"id": compute_mdhash_id(name, prefix="ent-"), "content": content}


async def _run(
    graph,
    vdb,
    embedding,
    doc_status=None,
    rebuilding=False,
    *,
    relationships_vdb=None,
    chunks_vdb=None,
    text_chunks=None,
):
    """Drive the gate.

    ``relationships_vdb`` and ``chunks_vdb`` default to populated stores so a
    test that is about the ENTITY pairing is not answered by one of its
    siblings. A test that wants those pairings passes them explicitly.
    """
    await check_vector_space_at_startup(
        graph=graph,
        entities_vdb=vdb,
        relationships_vdb=(
            FakeVectorStorage(rows=[{"id": "rel-1"}])
            if relationships_vdb is None
            else relationships_vdb
        ),
        chunks_vdb=(
            FakeVectorStorage(rows=[{"id": "chunk-1"}])
            if chunks_vdb is None
            else chunks_vdb
        ),
        text_chunks=FakeKVStorage(rows=1) if text_chunks is None else text_chunks,
        doc_status=FakeDocStatus() if doc_status is None else doc_status,
        embedding_func=embedding,
        expect_empty_vector_storage=rebuilding,
    )


# ---------------------------------------------------------------------------
# The empty-container gate
# ---------------------------------------------------------------------------


class TestEmptyContainerGate:
    async def test_refuses_when_the_graph_has_entities_and_the_vdb_has_none(self):
        """The shape a model change leaves on Milvus / Qdrant / PostgreSQL: a
        new, correctly named, empty container that no marker can ever flag."""
        graph = FakeGraph(labels=["Alice", "Bob", "Carol"])
        vdb = FakeVectorStorage(rows=[])

        with pytest.raises(VectorStorageEmptyError) as excinfo:
            await _run(graph, vdb, FakeEmbedding())

        error = excinfo.value
        assert error.vdb_name == "entities"
        assert error.source == "the knowledge graph"
        assert "lightrag-rebuild-vdb" in str(error)

    async def test_is_not_a_space_mismatch(self):
        """`lightrag-rebuild-vdb` answers VectorSpaceMismatchError by DROPPING
        the container. There is nothing to drop here, and a tool that conflated
        the two would report a destruction it never performed."""
        graph = FakeGraph(labels=["Alice"])

        with pytest.raises(VectorStorageEmptyError) as excinfo:
            await _run(graph, FakeVectorStorage(rows=[]), FakeEmbedding())

        assert not isinstance(excinfo.value, VectorSpaceMismatchError)

    async def test_a_declared_rebuild_is_not_refused(self):
        """A caller that owns the repopulation starts from exactly this state:
        an in-process rebuild, or the switch from NoopVectorDBStorage
        (graph-only ingestion) to a real vector backend."""
        graph = FakeGraph(labels=["Alice", "Bob"])

        await _run(graph, FakeVectorStorage(rows=[]), FakeEmbedding(), rebuilding=True)

    async def test_an_unfinished_ingest_is_not_refused(self):
        """A graph ahead of the vector store, while a document is still in
        flight, is a residue that HEALS -- an interrupted ingest, a batch that
        will be retried. Refusing here would block the very process whose next
        run repairs it (AGENTS.md, *Consistency without transactions*)."""
        graph = FakeGraph(labels=["Alice", "Bob"])

        await _run(
            graph,
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=FakeDocStatus(processed=0, unfinished=1),
        )

    async def test_unfinished_work_anywhere_does_not_refuse(self):
        """A PROCESSED row says nothing about THESE entities. The graph is
        ranked by degree, so an ingest that crashed after writing a batch of
        well-connected nodes but before their vector upserts fills the whole
        sample with rows that never had vectors -- while an older, unrelated
        PROCESSED document supplies the "evidence" to refuse on. That refuses
        the retry that would have healed it."""
        graph = FakeGraph(labels=["Alice", "Bob"])

        await _run(
            graph,
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=FakeDocStatus(processed=5, unfinished=1),
        )

    async def test_an_admin_only_workspace_still_refuses(self):
        """``acreate_entity`` / ``ainsert_custom_kg`` write graph entities AND
        their vectors, and write no doc-status row. Requiring a PROCESSED
        document would exempt such a workspace forever -- and it is the one
        place where "a later pipeline run repairs it" is false, because no
        pipeline run will ever recreate objects an operator made by hand."""
        graph = FakeGraph(labels=["Alice"])

        with pytest.raises(VectorStorageEmptyError):
            await _run(
                graph,
                FakeVectorStorage(rows=[]),
                FakeEmbedding(),
                doc_status=FakeDocStatus(processed=0, unfinished=0),
            )

    async def test_an_unreadable_doc_status_does_not_refuse(self):
        """The last question before a refusal. An answer nobody could read is
        not evidence that vectors are missing -- and a strict count RAISES
        rather than reporting what it managed to collect, which is why the
        gate asks for one."""
        graph = FakeGraph(labels=["Alice"])

        await _run(
            graph,
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=FakeDocStatus(error=RuntimeError("doc status down")),
        )

    async def test_a_non_persistent_backend_is_never_judged(self):
        """NoopVectorDBStorage keeps no vectors, so its reads are misses BY
        DESIGN. Graph-only ingestion is supported, and after its first document
        the graph holds entities and doc-status holds a PROCESSED row -- the
        gate would otherwise refuse every restart of it."""

        class NoopLike(FakeVectorStorage):
            persists_vectors = False

        vdb = NoopLike(rows=[], pending=True)
        embedding = FakeEmbedding()

        await _run(FakeGraph(labels=["Alice", "Bob"]), vdb, embedding)

        # Nothing was even asked of it.
        assert (vdb.reads, embedding.calls, vdb.adopted) == (0, 0, 0)

    async def test_emptiness_is_asked_not_inferred(self):
        """The gate reads ``is_empty()`` and nothing else.

        It used to infer emptiness from a sample of ``get_by_ids`` misses,
        which could not distinguish a miss from a swallowed transport error --
        every server-backed reader catches and returns ``[]``. ``is_empty`` is
        specified to RAISE instead, so the two are different values again."""
        vdb = FakeVectorStorage(rows=[])

        with pytest.raises(VectorStorageEmptyError):
            await _run(FakeGraph(labels=["Alice"]), vdb, FakeEmbedding())

        assert (vdb.empty_reads, vdb.reads) == (1, 0)

    async def test_a_raising_is_empty_does_not_refuse(self):
        """The whole point of the fail-loud contract: a backend that could not
        read the container says so, and a raise is never evidence of
        emptiness."""
        vdb = FakeVectorStorage(rows=[], empty_error=RuntimeError("cluster down"))

        await _run(FakeGraph(labels=["Alice"]), vdb, FakeEmbedding())

    async def test_a_backend_that_cannot_answer_is_skipped(self):
        """``StorageCapabilityError`` is the fail-closed default on the base
        class. A backend that never implemented ``is_empty`` is one the gate
        cannot question -- which is where every backend stood before it
        existed."""
        vdb = FakeVectorStorage(
            rows=[], empty_error=StorageCapabilityError("not supported")
        )

        await _run(FakeGraph(labels=["Alice"]), vdb, FakeEmbedding())

    async def test_a_populated_vdb_passes(self):
        graph = FakeGraph(labels=["Alice"])
        vdb = FakeVectorStorage(rows=[_entity_row()])

        await _run(graph, vdb, FakeEmbedding())

    async def test_an_empty_graph_exempts_only_the_graph_pairings(self):
        """An empty graph means the entity and relation pairings have no source
        to compare against -- it does NOT mean the whole check is over.

        This is the bug the pairing model fixes: the old early return ended the
        entire function, so a corpus producing text chunks but no extracted
        entities left ``chunks_vdb`` unchecked."""
        entities = FakeVectorStorage(rows=[], pending=True)
        chunks = FakeVectorStorage(rows=[])
        embedding = FakeEmbedding()

        with pytest.raises(VectorStorageEmptyError) as excinfo:
            await _run(
                FakeGraph(labels=[]),
                entities,
                embedding,
                chunks_vdb=chunks,
                text_chunks=FakeKVStorage(rows=7),
            )

        assert excinfo.value.vdb_name == "chunks"
        # The entity pairing asked nothing of its own store.
        assert entities.empty_reads == 0

    async def test_a_fresh_install_is_not_refused(self):
        """Everything empty: no graph, no chunks, no vectors. Every pairing has
        an empty source, so none of them has a question to ask."""
        entities = FakeVectorStorage(rows=[], pending=True)
        embedding = FakeEmbedding()

        await _run(
            FakeGraph(labels=[]),
            entities,
            embedding,
            relationships_vdb=FakeVectorStorage(rows=[]),
            chunks_vdb=FakeVectorStorage(rows=[]),
            text_chunks=FakeKVStorage(rows=0),
        )

        assert embedding.calls == 0
        assert entities.adopted == 0

    async def test_a_broken_graph_backend_does_not_fail_startup(self):
        """A transient graph error is not evidence about the embedding space,
        and it was not a startup failure before this check existed."""
        graph = FakeGraph(error=RuntimeError("neo4j unreachable"))

        await _run(graph, FakeVectorStorage(rows=[]), FakeEmbedding())

    async def test_an_unreadable_vdb_does_not_fail_startup(self):
        """Specifically NOT read as 'empty': a read that errored says nothing
        about what the container holds."""
        graph = FakeGraph(labels=["Alice"])
        vdb = FakeVectorStorage(
            rows=[_entity_row()], read_error=RuntimeError("cluster down")
        )

        await _run(graph, vdb, FakeEmbedding())


class TestPairings:
    """A vector storage is an INDEX; whether being empty is a defect depends on
    the data it indexes. There are three such pairings and none substitutes for
    another."""

    async def test_chunks_are_checked_against_text_chunks(self):
        """The pairing the graph cannot supply: a corpus can produce text
        chunks and extract no entities at all."""
        chunks = FakeVectorStorage(rows=[])

        with pytest.raises(VectorStorageEmptyError) as excinfo:
            await _run(
                FakeGraph(labels=["Alice"]),
                FakeVectorStorage(rows=[_entity_row()]),
                FakeEmbedding(),
                chunks_vdb=chunks,
                text_chunks=FakeKVStorage(rows=3),
            )

        assert excinfo.value.vdb_name == "chunks"

    async def test_no_text_chunks_asks_nothing_of_the_chunk_store(self):
        """A workspace built entirely through ``acreate_entity`` has entities
        and no chunks. Its empty chunk store is correct, not a defect."""
        chunks = FakeVectorStorage(rows=[])

        await _run(
            FakeGraph(labels=["Alice"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            chunks_vdb=chunks,
            text_chunks=FakeKVStorage(rows=0),
        )

        assert chunks.empty_reads == 0

    async def test_an_unreadable_text_chunks_source_does_not_refuse(self):
        """``BaseKVStorage.is_empty`` catches its errors and answers True, so an
        unreadable source arrives as 'no chunks'. That lands on the skip branch,
        which is the direction this module wants everywhere."""
        chunks = FakeVectorStorage(rows=[])

        await _run(
            FakeGraph(labels=["Alice"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            chunks_vdb=chunks,
            text_chunks=FakeKVStorage(rows=3, error=RuntimeError("redis down")),
        )

        assert chunks.empty_reads == 0

    async def test_relations_are_checked_against_graph_edges(self):
        """An interrupted rebuild can leave entities populated and relations
        not: `lightrag-rebuild-vdb` rebuilds them as separate steps."""
        relationships = FakeVectorStorage(rows=[])

        with pytest.raises(VectorStorageEmptyError) as excinfo:
            await _run(
                FakeGraph(labels=["Alice", "Bob"]),
                FakeVectorStorage(rows=[_entity_row()]),
                FakeEmbedding(),
                relationships_vdb=relationships,
            )

        assert excinfo.value.vdb_name == "relationships"

    async def test_a_graph_with_no_edges_asks_nothing_of_the_relation_store(self):
        """Entities without relations is an ordinary corpus, not a defect."""
        relationships = FakeVectorStorage(rows=[])

        await _run(
            FakeGraph(labels=["Alice"], edges=[]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            relationships_vdb=relationships,
        )

        assert relationships.empty_reads == 0

    async def test_a_backend_without_edge_iteration_is_skipped(self):
        """``iter_edges`` is fail-closed on the base class. A backend that never
        implemented it cannot answer the relation pairing, and 'cannot answer'
        is never evidence."""
        relationships = FakeVectorStorage(rows=[])

        await _run(
            FakeGraph(
                labels=["Alice", "Bob"],
                edge_error=StorageCapabilityError("no bounded edge iteration"),
            ),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            relationships_vdb=relationships,
        )

        assert relationships.empty_reads == 0

    async def test_doc_status_is_consulted_once_for_all_pairings(self):
        """The refusal branch is the only one that costs a doc-status read, and
        three pairings must not turn it into three reads."""

        class CountingDocStatus(FakeDocStatus):
            def __init__(self):
                super().__init__(processed=0, unfinished=1)
                self.counts = 0

            async def count_docs_by_statuses(self, statuses, *, strict=True):
                self.counts += 1
                return await super().count_docs_by_statuses(statuses, strict=strict)

        doc_status = CountingDocStatus()

        await _run(
            FakeGraph(labels=["Alice", "Bob"]),
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=doc_status,
            relationships_vdb=FakeVectorStorage(rows=[]),
            chunks_vdb=FakeVectorStorage(rows=[]),
            text_chunks=FakeKVStorage(rows=3),
        )

        assert doc_status.counts == 1

    async def test_a_healthy_start_never_touches_doc_status(self):
        """Every pairing satisfied: the gate must not pay for the question it
        only asks before refusing."""

        class ExplodingDocStatus:
            async def count_docs_by_statuses(self, statuses, *, strict=True):
                raise AssertionError("doc status must not be consulted")

        await _run(
            FakeGraph(labels=["Alice", "Bob"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            doc_status=ExplodingDocStatus(),
        )


# ---------------------------------------------------------------------------
# The adoption probe
# ---------------------------------------------------------------------------


class TestAdoptionProbe:
    @staticmethod
    def _setup(stored_vector, fresh_vector, **kwargs):
        row = _entity_row()
        vdb = FakeVectorStorage(
            rows=[row],
            vectors={row["id"]: stored_vector},
            pending=True,
            **kwargs,
        )
        return FakeGraph(labels=["Alice"]), vdb, FakeEmbedding(fresh_vector)

    async def test_a_reproduced_vector_adopts_the_container(self):
        graph, vdb, embedding = self._setup([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])

        await _run(graph, vdb, embedding)

        assert embedding.calls == 1
        assert vdb.adopted == 1

    async def test_a_foreign_vector_refuses(self):
        """The failure this whole feature exists to catch: a same-dimension
        model swap, invisible to every dimension check."""
        graph, vdb, embedding = self._setup([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            await _run(graph, vdb, embedding)

        assert "lightrag-rebuild-vdb" in str(excinfo.value)
        assert vdb.adopted == 0

    async def test_an_ambiguous_cosine_settles_nothing(self):
        """Between the two thresholds the probe has not returned a negative
        verdict, so it may not refuse -- and it has not earned adoption either."""
        midpoint = (ADOPT_COSINE + REFUSE_COSINE) / 2
        stored = [1.0, 0.0]
        fresh = [midpoint, float(np.sqrt(1 - midpoint**2))]
        graph, vdb, embedding = self._setup(stored, fresh)

        await _run(graph, vdb, embedding)

        assert vdb.adopted == 0

    async def test_an_embedding_failure_never_prevents_startup(self):
        row = _entity_row()
        vdb = FakeVectorStorage(
            rows=[row], vectors={row["id"]: [1.0, 0.0]}, pending=True
        )
        embedding = FakeEmbedding(error=RuntimeError("provider 503"))

        await _run(FakeGraph(labels=["Alice"]), vdb, embedding)

        assert vdb.adopted == 0

    async def test_a_hanging_provider_is_bounded_and_inconclusive(self, monkeypatch):
        """Providers retry with exponential backoff, so an unreachable endpoint
        can burn 30-60s per call. A safety feature must not become an
        availability cost."""
        monkeypatch.setenv("LIGHTRAG_VECTOR_SPACE_PROBE_TIMEOUT", "0.05")
        row = _entity_row()
        vdb = FakeVectorStorage(
            rows=[row], vectors={row["id"]: [1.0, 0.0]}, pending=True
        )

        await _run(FakeGraph(labels=["Alice"]), vdb, FakeEmbedding(hang=True))

        assert vdb.adopted == 0

    async def test_a_sample_without_vectors_is_inconclusive(self):
        row = _entity_row()
        vdb = FakeVectorStorage(rows=[row], vectors={}, pending=True)
        embedding = FakeEmbedding([1.0, 0.0])

        await _run(FakeGraph(labels=["Alice"]), vdb, embedding)

        assert embedding.calls == 0
        assert vdb.adopted == 0

    async def test_a_sample_without_content_is_inconclusive(self):
        row = _entity_row()
        row["content"] = "   "
        vdb = FakeVectorStorage(
            rows=[row], vectors={row["id"]: [1.0, 0.0]}, pending=True
        )
        embedding = FakeEmbedding([1.0, 0.0])

        await _run(FakeGraph(labels=["Alice"]), vdb, embedding)

        assert embedding.calls == 0
        assert vdb.adopted == 0

    async def test_unreadable_vectors_are_inconclusive(self):
        row = _entity_row()
        vdb = FakeVectorStorage(
            rows=[row], pending=True, vector_read_error=RuntimeError("timeout")
        )
        embedding = FakeEmbedding([1.0, 0.0])

        await _run(FakeGraph(labels=["Alice"]), vdb, embedding)

        assert embedding.calls == 0
        assert vdb.adopted == 0

    async def test_a_marked_container_costs_no_embedding_call(self):
        """After a marker is recorded the cost is zero forever -- the probe is
        the only expensive thing here and it must not run on every startup."""
        row = _entity_row()
        vdb = FakeVectorStorage(
            rows=[row], vectors={row["id"]: [1.0, 0.0]}, pending=False
        )
        embedding = FakeEmbedding([1.0, 0.0])

        await _run(FakeGraph(labels=["Alice"]), vdb, embedding)

        assert embedding.calls == 0
        assert vdb.adopted == 0

    async def test_a_denied_marker_write_does_not_fail_startup(self):
        graph, vdb, embedding = self._setup([1.0, 0.0], [1.0, 0.0], adopt_result=False)

        await _run(graph, vdb, embedding)

        assert vdb.adopted == 1

    async def test_a_raising_adopt_does_not_fail_startup(self):
        """``adopt_vector_space`` is contracted never to raise; the caller
        still refuses to let a backend that breaks that contract take the
        process down."""
        graph, vdb, embedding = self._setup(
            [1.0, 0.0], [1.0, 0.0], adopt_error=RuntimeError("disk full")
        )

        await _run(graph, vdb, embedding)

    async def test_only_the_probed_store_is_adopted(self):
        """The three vector targets share an embedding_func but NOT a history.

        ``lightrag-rebuild-vdb`` rebuilds entities, relationships and chunks
        separately, so an interrupted rebuild after a same-dimension model
        change can leave entities in the current space while the others still
        hold the previous model's vectors. Stamping this model onto those on
        the entity verdict would record a lie permanently -- the exact failure
        this feature exists to prevent -- so a store nobody probed stays
        unmarked.
        """
        row = _entity_row()
        entities = FakeVectorStorage(
            rows=[row], vectors={row["id"]: [1.0, 0.0]}, pending=True
        )
        relationships = FakeVectorStorage(pending=True)
        embedding = FakeEmbedding([1.0, 0.0])

        await _run(FakeGraph(labels=["Alice"]), entities, embedding)

        assert embedding.calls == 1
        assert entities.adopted == 1
        assert relationships.adopted == 0


# ---------------------------------------------------------------------------
# The comparison itself
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors_score_one(self):
        assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_scale_does_not_matter(self):
        """A backend that stores normalized vectors must not read as a
        different model."""
        assert _cosine([1.0, 2.0], [10.0, 20.0]) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "left,right",
        [
            ([1.0, 2.0], [1.0, 2.0, 3.0]),  # length mismatch
            ([], []),  # nothing to compare
            ([0.0, 0.0], [1.0, 2.0]),  # zero magnitude
            ([float("nan"), 1.0], [1.0, 2.0]),  # unusable values
        ],
    )
    def test_an_undefined_comparison_is_none_not_zero(self, left, right):
        """0.0 is the strongest possible evidence of a changed model. A
        comparison that says nothing must not be spelled that way."""
        assert _cosine(left, right) is None
