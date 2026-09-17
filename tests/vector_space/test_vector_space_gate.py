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

from lightrag.exceptions import VectorSpaceMismatchError, VectorStorageEmptyError
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

    def __init__(self, labels=(), error=None):
        self._labels = list(labels)
        self._error = error
        self.calls = 0

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._labels[:limit]


class FakeDocStatus:
    """Doc-status with just the one question the gate asks."""

    def __init__(self, processed=1, error=None):
        self._processed = processed
        self._error = error

    async def get_status_counts(self):
        if self._error is not None:
            raise self._error
        return {"processed": self._processed, "pending": 0}


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
    ):
        self.reads = 0
        self._rows = list(rows or [])
        self._vectors = dict(vectors or {})
        self._pending = pending
        self._adopt_result = adopt_result
        self._adopt_error = adopt_error
        self._read_error = read_error
        self._vector_read_error = vector_read_error
        self.adopted = 0

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


def _entity_row(name="Alice", content="Alice is an engineer."):
    return {"id": compute_mdhash_id(name, prefix="ent-"), "content": content}


async def _run(
    graph, vdb, embedding, adoptable=None, doc_status=None, rebuilding=False
):
    await check_vector_space_at_startup(
        graph=graph,
        entities_vdb=vdb,
        doc_status=FakeDocStatus() if doc_status is None else doc_status,
        embedding_func=embedding,
        adoptable=[vdb] if adoptable is None else adoptable,
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
        assert error.sampled == 3
        assert "lightrag-rebuild-vdb" in str(error)

    async def test_is_not_a_space_mismatch(self):
        """`lightrag-rebuild-vdb` answers VectorSpaceMismatchError by DROPPING
        the container. There is nothing to drop here, and a tool that conflated
        the two would report a destruction it never performed."""
        graph = FakeGraph(labels=["Alice"])

        with pytest.raises(VectorStorageEmptyError) as excinfo:
            await _run(graph, FakeVectorStorage(rows=[]), FakeEmbedding())

        assert not isinstance(excinfo.value, VectorSpaceMismatchError)

    async def test_a_positional_miss_is_not_read_as_a_row(self):
        """Nano returns one entry per requested id, with None where the row is
        absent -- most backends return a compacted list. Counting the list
        instead of the rows would start the server on top of a vanished vector
        store."""

        class PositionalMissStorage(FakeVectorStorage):
            async def get_by_ids(self, ids):
                return [None for _ in ids]

        graph = FakeGraph(labels=["Alice", "Bob"])

        with pytest.raises(VectorStorageEmptyError):
            await _run(graph, PositionalMissStorage(), FakeEmbedding())

    async def test_a_declared_rebuild_is_not_refused(self):
        """A caller that owns the repopulation starts from exactly this state:
        an in-process rebuild, or the switch from NoopVectorDBStorage
        (graph-only ingestion) to a real vector backend."""
        graph = FakeGraph(labels=["Alice", "Bob"])

        await _run(graph, FakeVectorStorage(rows=[]), FakeEmbedding(), rebuilding=True)

    async def test_an_unfinished_ingest_is_not_refused(self):
        """A graph ahead of the vector store, with nothing PROCESSED, is a
        residue that HEALS -- an interrupted ingest, a failed batch, a graph
        built through the admin API. Refusing here would block the very
        process whose next run repairs it (AGENTS.md, *Consistency without
        transactions*)."""
        graph = FakeGraph(labels=["Alice", "Bob"])

        await _run(
            graph,
            FakeVectorStorage(rows=[]),
            FakeEmbedding(),
            doc_status=FakeDocStatus(processed=0),
        )

    async def test_an_unreadable_doc_status_does_not_refuse(self):
        """The last question before a refusal. An answer nobody could read is
        not evidence that vectors are missing."""
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

    async def test_an_empty_read_is_confirmed_before_refusing(self):
        """Every server-backed get_by_ids CATCHES its transport errors and
        returns an empty list, so 'empty' and 'the cluster blinked' arrive as
        the same value. A blip on the first read must not refuse."""

        class BlinkingStorage(FakeVectorStorage):
            async def get_by_ids(self, ids):
                self.reads += 1
                if self.reads == 1:
                    return []  # the swallowed failure
                return [_entity_row()]

        vdb = BlinkingStorage()

        await _run(FakeGraph(labels=["Alice"]), vdb, FakeEmbedding())

        assert vdb.reads == 2

    async def test_a_genuinely_empty_store_still_refuses_after_confirming(self):
        vdb = FakeVectorStorage(rows=[])

        with pytest.raises(VectorStorageEmptyError):
            await _run(FakeGraph(labels=["Alice"]), vdb, FakeEmbedding())

        assert vdb.reads == 2

    async def test_a_raising_confirmation_read_does_not_refuse(self):
        """An exception is the one failure the backends DO surface, and it is
        unambiguous: the read did not run, so it is not evidence."""

        class FailsOnConfirmation(FakeVectorStorage):
            async def get_by_ids(self, ids):
                self.reads += 1
                if self.reads == 1:
                    return []
                raise RuntimeError("cluster down")

        await _run(FakeGraph(labels=["Alice"]), FailsOnConfirmation(), FakeEmbedding())

    async def test_a_populated_vdb_passes(self):
        graph = FakeGraph(labels=["Alice"])
        vdb = FakeVectorStorage(rows=[_entity_row()])

        await _run(graph, vdb, FakeEmbedding())

    async def test_an_empty_graph_asks_nothing(self):
        """Nothing SHOULD have a vector, so neither check has a question. A
        fresh install must not be refused for having no data yet."""
        vdb = FakeVectorStorage(rows=[], pending=True)
        embedding = FakeEmbedding()

        await _run(FakeGraph(labels=[]), vdb, embedding)

        assert embedding.calls == 0
        assert vdb.adopted == 0

    async def test_a_broken_graph_backend_does_not_fail_startup(self):
        """A transient graph error is not evidence about the embedding space,
        and it was not a startup failure before this check existed."""
        graph = FakeGraph(error=RuntimeError("neo4j unreachable"))

        await _run(graph, FakeVectorStorage(rows=[]), FakeEmbedding())

    async def test_an_unreadable_vdb_does_not_fail_startup(self):
        """Specifically NOT read as 'empty': a read that errored says nothing
        about what the container holds."""
        graph = FakeGraph(labels=["Alice"])
        vdb = FakeVectorStorage(read_error=RuntimeError("cluster down"))

        await _run(graph, vdb, FakeEmbedding())


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

    async def test_every_pending_storage_is_adopted_from_the_one_probe(self):
        """The three vector storages share one embedding_func and were written
        by the same deployment, so probing the entity store settles it."""
        row = _entity_row()
        entities = FakeVectorStorage(
            rows=[row], vectors={row["id"]: [1.0, 0.0]}, pending=True
        )
        relationships = FakeVectorStorage(pending=True)
        chunks = FakeVectorStorage(pending=False)
        embedding = FakeEmbedding([1.0, 0.0])

        await _run(
            FakeGraph(labels=["Alice"]),
            entities,
            embedding,
            adoptable=[entities, relationships, chunks],
        )

        assert embedding.calls == 1
        assert (entities.adopted, relationships.adopted, chunks.adopted) == (1, 1, 0)


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
