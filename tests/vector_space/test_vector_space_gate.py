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
from types import SimpleNamespace

import numpy as np
import pytest

from lightrag.base import DocStatus
from lightrag.exceptions import (
    StorageCapabilityError,
    VectorSpaceMismatchError,
    VectorStorageEmptyError,
)
from lightrag.utils import compute_mdhash_id, make_relation_vdb_ids
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

    def __init__(
        self,
        labels=(),
        error=None,
        edges=None,
        edge_error=None,
        source_id="chunk-1",
        node_error=None,
    ):
        self._labels = list(labels)
        self._error = error
        # Default: every object names a real chunk, i.e. a document produced
        # it. Tests for the admin-authored case pass source_id="manual_creation"
        # (what acreate_entity stamps) or "".
        self._source_id = source_id
        self._node_error = node_error
        # Default: one edge per pair of labels, so a graph with entities also
        # has relations unless a test says otherwise.
        self._edges = (
            list(edges)
            if edges is not None
            else [
                {"source": a, "target": b, "source_id": source_id}
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

    async def get_nodes_batch(self, node_ids: list[str]) -> dict:
        if self._node_error is not None:
            raise self._node_error
        return {
            name: {"entity_id": name, "source_id": self._source_id} for name in node_ids
        }

    async def iter_edges(self, batch_size: int):
        if self._edge_error is not None:
            raise self._edge_error
        for start in range(0, len(self._edges), batch_size):
            yield self._edges[start : start + batch_size]


class FakeDocStatus:
    """Doc-status with just the one question the gate asks."""

    def __init__(
        self,
        processed=1,
        error=None,
        unfinished=0,
        doc_statuses=None,
        chunks_by_doc=None,
    ):
        self._processed = processed
        self._error = error
        self._unfinished = unfinished
        # Per-document status for the healability trail. Default: the document
        # that owns the sampled chunks is itself unfinished, i.e. a retry
        # WOULD rewrite those objects.
        self._doc_statuses = (
            dict(doc_statuses)
            if doc_statuses is not None
            else {"doc-1": DocStatus.PROCESSING}
        )
        # chunks_list per document, for the chunk pairing's probe.
        self._chunks_by_doc = dict(chunks_by_doc or {})

    async def get_docs_by_statuses_page(self, statuses, *, limit, strict=True):
        if self._error is not None:
            raise self._error
        wanted = set(statuses)
        docs = {
            doc_id: SimpleNamespace(id=doc_id, status=status)
            for doc_id, status in self._doc_statuses.items()
            if status in wanted
        }
        return SimpleNamespace(docs=docs, next_position=None)

    async def get_full_docs_by_ids(self, doc_ids, *, strict=True):
        if self._error is not None:
            raise self._error
        return {
            doc_id: SimpleNamespace(
                id=doc_id,
                status=self._doc_statuses[doc_id],
                chunks_list=list(self._chunks_by_doc.get(doc_id, [])),
            )
            for doc_id in doc_ids
            if doc_id in self._doc_statuses
        }

    async def get_docs_by_ids(self, doc_ids, *, strict=True):
        if self._error is not None:
            raise self._error
        return {
            doc_id: SimpleNamespace(id=doc_id, status=status)
            for doc_id in doc_ids
            if (status := self._doc_statuses.get(doc_id)) is not None
        }

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

    def __init__(self, vector=None, *, error=None, hang=False, by_text=None):
        self._vector = vector
        self._error = error
        self._hang = hang
        # Per-text vectors, for containers that hold more than one embedding
        # space: each stored content re-embeds to its OWN fresh vector.
        self._by_text = dict(by_text or {})
        self.calls = 0
        self.batch_sizes: list[int] = []

    async def __call__(self, texts, **kwargs):
        self.calls += 1
        self.batch_sizes.append(len(texts))
        if self._hang:
            await asyncio.sleep(3600)
        if self._error is not None:
            raise self._error
        return np.array(
            [list(self._by_text.get(text, self._vector)) for text in texts],
            dtype=np.float32,
        )


class FakeKVStorage:
    """``text_chunks`` -- the source side of the chunk pairing.

    ``is_empty`` mirrors the REAL ``BaseKVStorage`` contract, which catches its
    backend errors and answers ``True``; ``iter_rows`` mirrors ITS contract and
    raises. The gate reads the source through ``iter_rows`` on the starts that
    have a chunk BASELINE to establish, for exactly that reason: that verdict
    backs a durable ``origin=empty`` record, and an unreadable source must land
    on "no evidence", never on "empty". Every other start keeps ``is_empty()``,
    whose "empty" only skips a check. The vector side is the opposite contract,
    and ``FakeVectorStorage.is_empty`` raises to match it.
    """

    def __init__(self, *, rows=0, error=None, chunk_owner="doc-1"):
        self._rows = rows
        self._error = error
        # Which document each chunk row names. None models a chunk row that is
        # gone, or one whose owning document was never recorded -- what
        # ainsert_custom_kg leaves behind, since it writes no doc-status row.
        self._chunk_owner = chunk_owner
        self.empty_reads = 0
        self.enumerations = 0

    async def is_empty(self) -> bool:
        self.empty_reads += 1
        if self._error is not None:
            return True  # what every real KV backend does on failure
        return self._rows == 0

    async def get_by_ids(self, ids):
        if self._chunk_owner is None:
            return []
        return [{"id": cid, "full_doc_id": self._chunk_owner} for cid in ids]

    def iter_rows(self, *, page_size=200):
        """The chunk probe's sample: the first page of rows, ``_id`` included."""
        self.enumerations += 1

        async def _gen():
            if self._error is not None:
                raise self._error  # what every real KV backend does on failure
            for i in range(min(self._rows, page_size)):
                yield {"_id": f"chunk-{i + 1}", "content": f"chunk {i + 1}"}

        return _gen()


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
    baseline_targets=(),
):
    """Drive the gate.

    ``relationships_vdb`` and ``chunks_vdb`` default to populated stores so a
    test that is about the ENTITY pairing is not answered by one of its
    siblings. A test that wants those pairings passes them explicitly.

    ``baseline_targets`` defaults to none, i.e. a workspace whose three
    baselines are already recorded: the coverage gate alone, on the cheap
    reads it has always used. A test about the BASELINE evidence -- the strict
    source read, the container's ``is_empty()``, a forced probe -- names the
    targets whose record is absent.
    """
    return await check_vector_space_at_startup(
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
        baseline_targets=baseline_targets,
    )


# ---------------------------------------------------------------------------
# The coverage gate
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


class TestUnfinishedWorkExemption:
    """The doc-status exemption is workspace-wide; the evidence it excuses is
    per-container. An unfinished document explains the missing vectors for the
    objects THAT document produces, and nothing else."""

    async def test_unfinished_work_still_exempts_pipeline_built_data(self):
        """The exemption that four review rounds put here must survive: an
        ingest interrupted before its vector flush heals by being retried."""
        await _run(
            FakeGraph(labels=["Alice", "Bob"], source_id="chunk-7"),
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=FakeDocStatus(processed=5, unfinished=1),
        )

    async def test_unfinished_work_does_not_exempt_admin_built_entities(self):
        """``acreate_entity`` stamps ``source_id="manual_creation"`` and writes
        no doc-status row. No pipeline run will ever recreate such an object, so
        an unrelated PENDING document must not excuse its empty container --
        otherwise those entities are silently unretrievable forever."""
        with pytest.raises(VectorStorageEmptyError) as excinfo:
            await _run(
                FakeGraph(labels=["Manual"], source_id="manual_creation"),
                FakeVectorStorage(rows=[]),
                FakeEmbedding(),
                doc_status=FakeDocStatus(processed=5, unfinished=1),
            )

        assert excinfo.value.vdb_name == "entities"

    async def test_an_empty_source_id_counts_as_documentless(self):
        """The other shape the admin writers leave behind."""
        with pytest.raises(VectorStorageEmptyError):
            await _run(
                FakeGraph(labels=["Manual"], source_id=""),
                FakeVectorStorage(rows=[]),
                FakeEmbedding(),
                doc_status=FakeDocStatus(processed=5, unfinished=1),
            )

    async def test_unfinished_work_does_not_exempt_admin_built_relations(self):
        """Edges carry ``source_id`` too, and ``iter_edges`` yields the whole
        payload, so the relation pairing gets the same treatment."""
        with pytest.raises(VectorStorageEmptyError) as excinfo:
            await _run(
                FakeGraph(labels=["A", "B"], source_id="manual_creation"),
                FakeVectorStorage(rows=[_entity_row()]),
                FakeEmbedding(),
                doc_status=FakeDocStatus(processed=5, unfinished=1),
                relationships_vdb=FakeVectorStorage(rows=[]),
            )

        assert excinfo.value.vdb_name == "relationships"

    async def test_a_custom_kg_object_sourced_to_a_real_chunk_still_refuses(self):
        """``ainsert_custom_kg`` maps its entities onto the call's OWN chunks, so
        their ``source_id`` is a real ``chunk-*`` id -- not a placeholder. But it
        writes no doc-status row, so the chunk's ``full_doc_id`` names nothing a
        retry will ever reprocess. A placeholder test read that as
        document-produced and handed it the exemption."""
        with pytest.raises(VectorStorageEmptyError) as excinfo:
            await _run(
                FakeGraph(labels=["Custom"], source_id="chunk-abc123"),
                FakeVectorStorage(rows=[]),
                FakeEmbedding(),
                doc_status=FakeDocStatus(
                    processed=5,
                    unfinished=1,
                    # The chunk names this document; doc_status has never heard
                    # of it, which is exactly what ainsert_custom_kg leaves.
                    doc_statuses={"doc-unrelated": DocStatus.PENDING},
                ),
                text_chunks=FakeKVStorage(rows=3, chunk_owner="custom-kg-doc"),
            )

        assert excinfo.value.vdb_name == "entities"

    async def test_a_finished_document_is_not_healed_by_an_unrelated_pending_one(self):
        """The workspace-wide count cannot see this: the sampled objects belong
        to a document that has FINISHED, so the PENDING document's retry will
        not rewrite them."""
        with pytest.raises(VectorStorageEmptyError):
            await _run(
                FakeGraph(labels=["Alice"], source_id="chunk-1"),
                FakeVectorStorage(rows=[]),
                FakeEmbedding(),
                doc_status=FakeDocStatus(
                    processed=5,
                    unfinished=1,
                    doc_statuses={
                        "doc-1": DocStatus.PROCESSED,
                        "doc-other": DocStatus.PENDING,
                    },
                ),
                text_chunks=FakeKVStorage(rows=3, chunk_owner="doc-1"),
            )

    async def test_an_object_owned_by_the_unfinished_document_is_exempt(self):
        """The case the whole exemption exists for, now resolved by the trail
        rather than by a workspace-wide count."""
        await _run(
            FakeGraph(labels=["Alice"], source_id="chunk-1"),
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=FakeDocStatus(
                processed=0,
                unfinished=1,
                doc_statuses={"doc-1": DocStatus.PROCESSING},
            ),
            text_chunks=FakeKVStorage(rows=3, chunk_owner="doc-1"),
        )

    async def test_chunks_of_a_finished_document_are_not_healed_by_a_pending_one(
        self,
    ):
        """The chunk pairing had no healability probe at all, so it still used
        the raw workspace-wide count: any unrelated PENDING row excused an empty
        chunks_vdb even when every existing chunk belonged to a document that
        had finished. `naive` / `mix` then serve nothing, permanently once the
        pending document writes one chunk vector."""
        with pytest.raises(VectorStorageEmptyError) as excinfo:
            await _run(
                FakeGraph(labels=[]),
                FakeVectorStorage(rows=[]),
                FakeEmbedding(),
                doc_status=FakeDocStatus(
                    processed=1,
                    unfinished=1,
                    doc_statuses={
                        "doc-done": DocStatus.PROCESSED,
                        "doc-new": DocStatus.PENDING,
                    },
                    chunks_by_doc={"doc-done": ["chunk-done-1"]},
                ),
                chunks_vdb=FakeVectorStorage(rows=[]),
                text_chunks=FakeKVStorage(rows=4, chunk_owner="doc-done"),
            )

        assert excinfo.value.vdb_name == "chunks"

    async def test_chunks_of_only_unfinished_documents_stay_exempt(self):
        """The case the exemption exists for, on the chunk pairing: every chunk
        belongs to a document still in flight, so its retry rewrites them."""
        await _run(
            FakeGraph(labels=[]),
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=FakeDocStatus(
                processed=0,
                unfinished=1,
                doc_statuses={"doc-new": DocStatus.PROCESSING},
                chunks_by_doc={"doc-new": ["chunk-new-1"]},
            ),
            chunks_vdb=FakeVectorStorage(rows=[]),
            text_chunks=FakeKVStorage(rows=4, chunk_owner="doc-new"),
        )

    async def test_a_finished_document_whose_chunks_are_gone_stays_exempt(self):
        """Same cannot-tell rule as the graph trail: an empty KV read is as
        often a swallowed transport error as a real absence."""
        await _run(
            FakeGraph(labels=[]),
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=FakeDocStatus(
                processed=1,
                unfinished=1,
                doc_statuses={
                    "doc-done": DocStatus.PROCESSED,
                    "doc-new": DocStatus.PENDING,
                },
                chunks_by_doc={"doc-done": ["chunk-done-1"]},
            ),
            chunks_vdb=FakeVectorStorage(rows=[]),
            text_chunks=FakeKVStorage(rows=4, chunk_owner=None),
        )

    async def test_an_unreadable_chunk_trail_keeps_the_exemption(self):
        """A KV read that comes back empty is a swallowed transport error as
        often as a real absence, so it must not be read as an orphan."""
        await _run(
            FakeGraph(labels=["Alice"], source_id="chunk-1"),
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=FakeDocStatus(processed=5, unfinished=1),
            text_chunks=FakeKVStorage(rows=3, chunk_owner=None),
        )

    async def test_an_unreadable_node_payload_keeps_the_exemption(self):
        """Sampling is sound here ONLY because a miss declines to refuse. A
        probe that cannot run must land on the same side."""
        await _run(
            FakeGraph(
                labels=["Manual"],
                source_id="manual_creation",
                node_error=RuntimeError("graph read failed"),
            ),
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=FakeDocStatus(processed=5, unfinished=1),
        )

    async def test_the_probe_is_not_paid_on_a_healthy_start(self):
        """It reads node PAYLOADS, not names, so it must stay on the branch
        that was already about to exempt."""

        class ExplodingNodeRead(FakeGraph):
            async def get_nodes_batch(self, node_ids):
                raise AssertionError("must not be consulted on a healthy start")

        await _run(
            ExplodingNodeRead(labels=["Alice"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
        )


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
            baseline_targets=("chunks",),
        )

        # The coverage GATE asks nothing and refuses nothing. The one
        # ``is_empty()`` read is the baseline evidence: an empty source
        # records ``origin=empty`` only if the container is empty too.
        assert (chunks.empty_reads, chunks.reads) == (1, 0)

    async def test_an_unreadable_text_chunks_source_does_not_refuse(self):
        """An unreadable source supplies no evidence: skip, do not refuse --
        and, since the same verdict backs the chunk baseline, it is ``None``,
        never ``False``. ``BaseKVStorage.is_empty`` would have said "empty"
        (it catches its errors), which is why the gate reads ``iter_rows``."""
        chunks = FakeVectorStorage(rows=[])

        evidence = await _run(
            FakeGraph(labels=["Alice"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            chunks_vdb=chunks,
            text_chunks=FakeKVStorage(rows=3, error=RuntimeError("redis down")),
            baseline_targets=("chunks",),
        )

        assert chunks.empty_reads == 0
        assert evidence.source_populated["chunks"] is None

    async def test_an_empty_text_chunks_source_is_confirmed_by_a_fail_loud_read(
        self,
    ):
        """``False`` -- the answer that records ``origin=empty`` -- comes only
        from a read that would have raised had it failed."""
        text_chunks = FakeKVStorage(rows=0)

        evidence = await _run(
            FakeGraph(labels=["Alice"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            chunks_vdb=FakeVectorStorage(rows=[]),
            text_chunks=text_chunks,
            baseline_targets=("chunks",),
        )

        assert evidence.source_populated["chunks"] is False
        assert text_chunks.empty_reads == 0, "is_empty() was not consulted"

    async def test_an_empty_source_asks_its_container_before_an_empty_baseline(
        self,
    ):
        """``origin=empty`` is a durable claim about the CONTAINER, so an empty
        source is not enough: the evidence also carries the container's
        fail-loud ``is_empty()`` -- ``True`` (record), ``False`` (survivors
        nobody can vouch for: record nothing), ``None`` (unreadable: record
        nothing). A populated source never asks."""
        graph = FakeGraph(labels=["Alice"])
        entities = FakeVectorStorage(rows=[_entity_row()])

        both_empty = await _run(
            graph,
            entities,
            FakeEmbedding(),
            chunks_vdb=FakeVectorStorage(rows=[]),
            text_chunks=FakeKVStorage(rows=0),
            baseline_targets=("chunks",),
        )
        assert both_empty.index_empty["chunks"] is True

        survivors = await _run(
            graph,
            entities,
            FakeEmbedding(),
            chunks_vdb=FakeVectorStorage(rows=[{"id": "chunk-1"}]),
            text_chunks=FakeKVStorage(rows=0),
            baseline_targets=("chunks",),
        )
        assert survivors.index_empty["chunks"] is False

        unreadable = await _run(
            graph,
            entities,
            FakeEmbedding(),
            chunks_vdb=FakeVectorStorage(rows=[], empty_error=RuntimeError("down")),
            text_chunks=FakeKVStorage(rows=0),
            baseline_targets=("chunks",),
        )
        assert unreadable.index_empty["chunks"] is None

        populated = await _run(
            graph,
            entities,
            FakeEmbedding(),
            chunks_vdb=FakeVectorStorage(rows=[{"id": "chunk-1"}]),
            text_chunks=FakeKVStorage(rows=1),
            baseline_targets=("chunks",),
        )
        assert "chunks" not in populated.index_empty

    async def test_a_kv_store_without_enumeration_answers_populated_only(self):
        """The ``is_empty()`` fallback for a backend that cannot page its rows:
        "populated" is trustworthy and keeps the coverage check, "empty" is
        indistinguishable from an outage and becomes "unknown"."""

        class NoEnumeration(FakeKVStorage):
            def iter_rows(self, *, page_size=200):
                raise StorageCapabilityError("no enumeration")

        populated = await _run(
            FakeGraph(labels=["Alice"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            chunks_vdb=FakeVectorStorage(rows=[{"id": "chunk-1"}]),
            text_chunks=NoEnumeration(rows=1),
            baseline_targets=("chunks",),
        )
        assert populated.source_populated["chunks"] is True

        chunks = FakeVectorStorage(rows=[])
        with pytest.raises(VectorStorageEmptyError):
            await _run(
                FakeGraph(labels=["Alice"]),
                FakeVectorStorage(rows=[_entity_row()]),
                FakeEmbedding(),
                chunks_vdb=chunks,
                text_chunks=NoEnumeration(rows=1),
                baseline_targets=("chunks",),
            )

        unknown = await _run(
            FakeGraph(labels=["Alice"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            chunks_vdb=FakeVectorStorage(rows=[]),
            text_chunks=NoEnumeration(rows=0),
            baseline_targets=("chunks",),
        )
        assert unknown.source_populated["chunks"] is None

    async def test_a_recorded_chunk_baseline_enumerates_nothing(self):
        """The strict read is the price of CLAIMING a baseline, not of starting.

        Every later start has the record already, so the chunk source is read
        the way the coverage gate has always read it. It matters beyond a
        round trip: ``BaseKVStorage.iter_rows`` forbids a startup path from
        scanning a namespace, and ``JsonKVStorage`` -- rows in a
        ``Manager().dict()`` -- snapshots its whole key list before it can
        yield a first page.
        """
        text_chunks = FakeKVStorage(rows=3)

        evidence = await _run(
            FakeGraph(labels=["Alice"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            chunks_vdb=FakeVectorStorage(rows=[{"id": "chunk-1"}]),
            text_chunks=text_chunks,
        )

        assert text_chunks.enumerations == 0
        assert text_chunks.empty_reads == 1
        assert evidence.source_populated["chunks"] is True

    async def test_a_recorded_baseline_asks_no_container_whether_it_is_empty(self):
        """``index_empty`` is evidence for a claim, so a target that claims
        nothing does not gather it -- and an empty source still refuses
        nothing, exactly as before baselines existed."""
        chunks = FakeVectorStorage(rows=[])

        evidence = await _run(
            FakeGraph(labels=["Alice"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding(),
            chunks_vdb=chunks,
            text_chunks=FakeKVStorage(rows=0),
        )

        assert chunks.empty_reads == 0
        assert evidence.index_empty == {}

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
            baseline_targets=("relationships",),
        )

        # The coverage GATE asks nothing and refuses nothing. The one
        # ``is_empty()`` read is the baseline evidence: an empty source
        # records ``origin=empty`` only if the container is empty too.
        assert (relationships.empty_reads, relationships.reads) == (1, 0)

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

    async def test_a_mixed_container_refuses_instead_of_adopting(self):
        """The container holds vectors from TWO models at once.

        Reachable today: an upgrade that also switches to a same-dimension
        model starts unmarked when the probe cannot run (absent evidence never
        refuses), keeps serving, and then takes new writes -- so the legacy
        rows and this model's rows share one container. Adopting on whichever
        row the sample happened to reach first would stamp this model over the
        foreign half and hide it permanently."""
        old_row = _entity_row("Legacy", content="written by the previous model")
        new_row = _entity_row("Fresh", content="written by this model")
        vdb = FakeVectorStorage(
            rows=[new_row, old_row],
            vectors={
                new_row["id"]: [1.0, 0.0, 0.0, 0.0],
                old_row["id"]: [1.0, 0.0, 0.0, 0.0],
            },
            pending=True,
        )
        embedding = FakeEmbedding(
            by_text={
                # reproduces its stored vector
                new_row["content"]: [1.0, 0.0, 0.0, 0.0],
                # does not: a different embedding space
                old_row["content"]: [0.0, 1.0, 0.0, 0.0],
            }
        )

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            await _run(FakeGraph(labels=["Fresh", "Legacy"]), vdb, embedding)

        assert "more than one embedding space" in str(excinfo.value)
        assert vdb.adopted == 0

    async def test_every_compared_record_must_reproduce_before_adopting(self):
        """The homogeneity rule, from the other side: all of them agreeing is
        what adoption needs, and it is one batched embedding call."""
        rows = [_entity_row(f"E{i}", content=f"content {i}") for i in range(3)]
        vdb = FakeVectorStorage(
            rows=rows,
            vectors={row["id"]: [1.0, 0.0, 0.0, 0.0] for row in rows},
            pending=True,
        )
        embedding = FakeEmbedding([1.0, 0.0, 0.0, 0.0])

        await _run(FakeGraph(labels=["E0", "E1", "E2"]), vdb, embedding)

        assert vdb.adopted == 1
        assert embedding.calls == 1, "the probe must not pay a round trip per row"
        assert embedding.batch_sizes == [3]

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

    async def test_each_store_is_adopted_only_on_its_own_evidence(self):
        """The three vector targets share an embedding_func but NOT a history.

        ``lightrag-rebuild-vdb`` rebuilds entities, relationships and chunks
        separately, so an interrupted rebuild after a same-dimension model
        change can leave entities in the current space while the others still
        hold the previous model's vectors. Stamping this model onto those on
        the ENTITY verdict would record a lie permanently -- so each container
        is probed on a sample from its own source, and one whose sample yields
        nothing stays unmarked whatever its siblings proved.
        """
        row = _entity_row()
        entities = FakeVectorStorage(
            rows=[row], vectors={row["id"]: [1.0, 0.0]}, pending=True
        )
        # Pending, but the graph has no edges: nothing to sample, no verdict.
        relationships = FakeVectorStorage(pending=True)
        embedding = FakeEmbedding([1.0, 0.0])

        await _run(
            FakeGraph(labels=["Alice"], edges=[]),
            entities,
            embedding,
            relationships_vdb=relationships,
        )

        assert embedding.calls == 1
        assert entities.adopted == 1
        assert relationships.adopted == 0

    async def test_a_legacy_reverse_id_relation_store_is_probed_too(self):
        """A historical custom-KG import may have hashed its relation vectors
        under the reverse-order id (``make_relation_vdb_ids(...)[1]``). The
        sample carries both candidate ids, so an all-legacy store is examined
        and adopts on its own reproduced vectors instead of staying without a
        baseline forever."""
        legacy_id = make_relation_vdb_ids("Alice", "Bob")[1]
        graph = FakeGraph(labels=["Alice", "Bob"])  # one Alice-Bob edge
        entities = FakeVectorStorage(rows=[_entity_row()])
        legacy = FakeVectorStorage(
            rows=[{"id": legacy_id, "content": "Alice works with Bob"}],
            vectors={legacy_id: [1.0, 0.0]},
            pending=True,
        )

        await _run(graph, entities, FakeEmbedding([1.0, 0.0]), relationships_vdb=legacy)

        assert legacy.adopted == 1

    async def test_a_mixed_relation_store_is_not_adopted_on_canonical_rows_alone(
        self,
    ):
        """Canonical rows reproduce, the legacy reverse-id row does not: the
        verdict must cover both, so the container is not adopted."""
        canonical_id, legacy_id = make_relation_vdb_ids("Alice", "Bob")
        graph = FakeGraph(labels=["Alice", "Bob"])
        entities = FakeVectorStorage(rows=[_entity_row()])
        mixed = FakeVectorStorage(
            rows=[
                {"id": canonical_id, "content": "canonical"},
                {"id": legacy_id, "content": "legacy"},
            ],
            vectors={canonical_id: [1.0, 0.0], legacy_id: [1.0, 0.0]},
            pending=True,
        )
        embedding = FakeEmbedding([1.0, 0.0], by_text={"legacy": [0.0, 1.0]})

        try:
            await _run(graph, entities, embedding, relationships_vdb=mixed)
        except VectorSpaceMismatchError:
            pass

        assert mixed.adopted == 0

    async def test_relations_are_probed_from_the_graphs_edges(self):
        """Relations have their own sample -- the first batch of iter_edges
        mapped to the canonical relation id -- so their container adopts on
        its own reproduced vectors, and refuses on its own foreign ones."""
        rel_id = make_relation_vdb_ids("Alice", "Bob")[0]
        graph = FakeGraph(labels=["Alice", "Bob"])  # one Alice-Bob edge
        entities = FakeVectorStorage(rows=[_entity_row()])  # marked already

        reproduced = FakeVectorStorage(
            rows=[{"id": rel_id, "content": "Alice works with Bob"}],
            vectors={rel_id: [1.0, 0.0]},
            pending=True,
        )
        await _run(
            graph, entities, FakeEmbedding([1.0, 0.0]), relationships_vdb=reproduced
        )
        assert reproduced.adopted == 1

        foreign = FakeVectorStorage(
            rows=[{"id": rel_id, "content": "Alice works with Bob"}],
            vectors={rel_id: [1.0, 0.0]},
            pending=True,
        )
        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            await _run(
                graph, entities, FakeEmbedding([0.0, 1.0]), relationships_vdb=foreign
            )
        assert "relationships" in str(excinfo.value)
        assert foreign.adopted == 0

    async def test_chunks_are_probed_from_the_first_page_of_text_chunks(self):
        """Chunks have their own sample too -- the first page of the KV
        store's ``iter_rows`` -- which is what the enumeration surface made
        possible. The chunk container adopts on its own vectors."""
        entities = FakeVectorStorage(rows=[_entity_row()])
        chunks = FakeVectorStorage(
            rows=[{"id": "chunk-1", "content": "chunk 1"}],
            vectors={"chunk-1": [1.0, 0.0]},
            pending=True,
        )
        embedding = FakeEmbedding([1.0, 0.0])

        await _run(
            FakeGraph(labels=["Alice"]),
            entities,
            embedding,
            chunks_vdb=chunks,
            text_chunks=FakeKVStorage(rows=1),
        )

        assert chunks.adopted == 1
        assert embedding.calls == 1, "entities were marked; only chunks probed"

    async def test_a_kv_store_without_enumeration_leaves_chunks_unmarked(self):
        """A backend that cannot page its rows cannot supply a chunk sample.
        That is 'could not run', never a refusal -- and never an adoption."""

        class NoEnumeration(FakeKVStorage):
            def iter_rows(self, *, page_size=200):
                raise StorageCapabilityError("no enumeration")

        chunks = FakeVectorStorage(
            rows=[{"id": "chunk-1", "content": "chunk 1"}],
            vectors={"chunk-1": [1.0, 0.0]},
            pending=True,
        )
        await _run(
            FakeGraph(labels=["Alice"]),
            FakeVectorStorage(rows=[_entity_row()]),
            FakeEmbedding([1.0, 0.0]),
            chunks_vdb=chunks,
            text_chunks=NoEnumeration(rows=1),
        )
        assert chunks.adopted == 0


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
