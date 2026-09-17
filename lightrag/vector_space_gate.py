"""The two startup checks that no single storage can make on its own.

Both ask a question that spans the graph and a vector storage, which is why
they live one layer up in ``LightRAG.initialize_storages()`` rather than inside
a backend's ``initialize()``. A vector storage cannot tell on its own whether
being empty is a defect: that depends on the data it INDEXES, which lives in
another storage entirely.

**The empty-container gate.** Milvus, Qdrant and PostgreSQL encode the
embedding model in the container NAME, so changing the model does not reuse the
old container -- it provisions a new, empty, correctly named one. No marker can
ever fire there, and nothing else notices: every vector query simply returns
nothing, which reads as "no relevant context" rather than as a broken
deployment. The detectable shape is cross-storage: *the graph holds entities
and the vector store holds none*. It also catches what no marker could -- a
deleted vector file, a container emptied out of band, a rebuild that was
interrupted -- and it self-clears, because a rebuild answers the question by
populating the container.

The check runs over three pairings, one per vector target, because each is a
separate container that a model change replaces separately::

    text_chunks (KV)  ->  chunks_vdb
    graph entities    ->  entities_vdb
    graph relations   ->  relationships_vdb

None of them substitutes for another: a workspace built through
``acreate_entity`` has entities and no chunks, and a corpus that extracts
nothing has chunks and no entities. See ``_PairingGate``.

**The adoption probe.** Every container that predates the provenance marker
records no model, and *absent evidence never refuses*, so that silence never
ends by itself. It has to be ended by adopting the container -- but not
blindly: an operator who upgrades AND switches to a same-dimension model in one
step would otherwise get the new model's name stamped onto the old model's
vectors, permanently, after which the gate can never fire again. So adoption
carries evidence: re-embed one stored ``content`` with the current model and
compare it against the vector stored beside it.

The governing invariant, which every branch here is written to preserve:

    Only a probe that ran and returned a negative verdict may refuse. Every
    form of "it could not run" falls back to the pre-upgrade behaviour.

An embedding provider that is down, a timeout, a source storage that cannot be
read, an index that cannot answer, a marker write denied by permissions -- all
of these
leave the container unmarked and the instance serving, which is exactly the
detection capability LightRAG had before this feature existed.

Read ``docs/design/VectorSpaceProvenance.md`` before changing the verdicts.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from lightrag.base import DocStatus
from lightrag.exceptions import (
    StorageCapabilityError,
    VectorSpaceMismatchError,
    VectorStorageEmptyError,
)
from lightrag.kg.vector_space import declared_model_name
from lightrag.utils import compute_mdhash_id, logger

# How many graph entities the ADOPTION PROBE looks up to find one row carrying
# both a ``content`` field and a stored vector. Not a confidence knob -- the
# probe needs exactly one usable row, and the margin covers rows whose content
# is missing or whose vector cannot be fetched. The empty-container gate does
# not sample at all; it asks ``is_empty()``.
SAMPLE_SIZE = 32

# Adoption needs near-identity. The same model re-embedding the same text lands
# at ~1.0; the gap to a different model is wide (typically 0.0-0.5), so neither
# halfvec quantization nor a normalization difference moves a same-model result
# below this.
ADOPT_COSINE = 0.95

# ... and refusal needs the other end of that gap, not merely "not adoption".
# The band between the two is where a verdict cannot be trusted either way --
# an unusually short content field, a provider that silently truncated, a
# fine-tune of the same base model. The invariant above makes that band
# inconclusive rather than a refusal: failing a live deployment closed on
# ambiguous evidence is worse than the unmarked container it already had.
REFUSE_COSINE = 0.70

_PROBE_TIMEOUT_ENV = "LIGHTRAG_VECTOR_SPACE_PROBE_TIMEOUT"
DEFAULT_PROBE_TIMEOUT = 30.0


def _probe_timeout() -> float:
    """Seconds to wait for the probe's single embedding call.

    Bounded because the providers retry: ``@retry(stop_after_attempt(3),
    wait_exponential(min=4, max=10))`` means an unreachable endpoint can burn
    30-60s, and a safety feature that delays every startup by a minute is an
    availability cost of its own. A timeout is just another way the probe could
    not run, so it lands inconclusive like the rest.
    """
    raw = os.environ.get(_PROBE_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_PROBE_TIMEOUT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"{_PROBE_TIMEOUT_ENV}={raw!r} is not a number; "
            f"using {DEFAULT_PROBE_TIMEOUT}s"
        )
        return DEFAULT_PROBE_TIMEOUT
    if value <= 0:
        logger.warning(
            f"{_PROBE_TIMEOUT_ENV}={raw!r} is not positive; "
            f"using {DEFAULT_PROBE_TIMEOUT}s"
        )
        return DEFAULT_PROBE_TIMEOUT
    return value


def _cosine(a: Any, b: Any) -> float | None:
    """Cosine similarity of two vectors, or ``None`` if it is undefined.

    ``None`` for a length mismatch or a zero-magnitude vector: both mean the
    comparison says nothing, and a silent 0.0 there would read as the strongest
    possible evidence of a changed model.
    """
    import numpy as np

    try:
        left = np.asarray(a, dtype=np.float64).reshape(-1)
        right = np.asarray(b, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if left.size == 0 or left.size != right.size:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not denominator or not np.isfinite(denominator):
        return None
    similarity = float(np.dot(left, right) / denominator)
    if not np.isfinite(similarity):
        return None
    return similarity


async def _sample_entity_ids(graph, limit: int) -> list[str]:
    """Vector ids for up to ``limit`` entities the graph actually holds.

    ``get_popular_labels`` rather than ``iter_labels``: it is on every graph
    backend (``iter_labels`` fails closed on backends that never implemented
    it), it is bounded by construction, and ranking by degree biases the sample
    towards entities that are certainly real -- the base contract warns that a
    backend MAY surface a label with edges but no node document, and such an
    artifact would never have been embedded.
    """
    labels = await graph.get_popular_labels(limit=limit)
    return [compute_mdhash_id(label, prefix="ent-") for label in labels]


async def _probe_same_embedding_space(
    rows: list[dict[str, Any]],
    vectors: dict[str, list[float]],
    embedding_func,
) -> tuple[bool | None, str]:
    """Re-embed one stored content and compare it with its stored vector.

    Returns ``(verdict, detail)`` where the verdict is ``True`` (adopt),
    ``False`` (refuse) or ``None`` (inconclusive -- do neither).
    """
    candidate = None
    for row in rows:
        row_id = row.get("id")
        content = row.get("content")
        if not row_id or not isinstance(content, str) or not content.strip():
            continue
        stored = vectors.get(row_id)
        if stored is None:
            continue
        candidate = (row_id, content, stored)
        break

    if candidate is None:
        return None, "no sampled record carried both a content field and a vector"

    row_id, content, stored = candidate
    try:
        # context="document" matches how every backend embeds on the way in;
        # a provider that prefixes queries differently would otherwise be
        # compared against a vector it never would have written.
        fresh = await asyncio.wait_for(
            embedding_func([content], context="document"),
            timeout=_probe_timeout(),
        )
    except asyncio.TimeoutError:
        return None, f"the embedding call timed out after {_probe_timeout()}s"
    except Exception as e:
        return None, f"the embedding call failed ({type(e).__name__}: {e})"

    try:
        fresh_vector = fresh[0]
    except (IndexError, TypeError, KeyError):
        return None, "the embedding function returned no vector"

    similarity = _cosine(fresh_vector, stored)
    if similarity is None:
        return None, "the stored and freshly embedded vectors are not comparable"
    if similarity >= ADOPT_COSINE:
        return True, f"cosine {similarity:.3f} against record '{row_id}'"
    if similarity <= REFUSE_COSINE:
        return False, f"cosine {similarity:.3f} against record '{row_id}'"
    return (
        None,
        f"cosine {similarity:.3f} against record '{row_id}' falls between "
        f"{REFUSE_COSINE} and {ADOPT_COSINE}, which settles nothing",
    )


async def _vectors_are_expected(doc_status) -> bool:
    """Whether this workspace SHOULD have vectors for its indexed data by now.

    The caller has already established that some indexed data is not empty
    while its vector storage is. This decides whether that gap is a defect or
    work in flight, so the question is only whether anything is UNFINISHED.
    Nothing is, in two quite different situations, and both mean the same thing
    here:

    * **Every document finished.** PROCESSED is the pipeline's claim that it
      wrote everything those documents produce, vectors included.
    * **There are no documents at all.** ``acreate_entity`` and
      ``ainsert_custom_kg`` write graph entities AND their vectors, and write
      no doc-status row. Requiring a PROCESSED document would exempt such a
      workspace forever -- and it is the one place where "a later pipeline run
      repairs it" is simply false, because no pipeline run will ever recreate
      objects an operator created by hand.

    Anything PENDING, PARSING, ANALYZING, PROCESSING or FAILED is the opposite:
    the residue has an owner and heals by being retried. Counting only the
    unfinished states is also what keeps a PROCESSED row from being read as
    evidence about THESE entities -- the graph sample is ranked by degree, so an
    ingest that crashed after writing a batch of well-connected nodes but before
    their vector upserts fills the whole sample with rows that never had
    vectors, and an older unrelated document must not supply the "evidence" to
    refuse on.

    Counted with ``count_docs_by_statuses(strict=True)``, never
    ``get_status_counts()``: the latter is documented to swallow its errors and
    report what it managed to collect, so a doc-status read that failed halfway
    could show a PROCESSED row while missing the PENDING one that explains
    everything -- and refuse the restart that would have healed it. Strict
    counting raises instead, and a raise answers "do not refuse", like every
    other unreadable thing this module consults.
    """
    if doc_status is None:
        return False
    unfinished_statuses = [
        status for status in DocStatus if status is not DocStatus.PROCESSED
    ]
    try:
        unfinished = await doc_status.count_docs_by_statuses(
            unfinished_statuses, strict=True
        )
    except Exception as e:
        logger.warning(
            f"Not refusing an empty vector storage: the document status could "
            f"not be counted ({type(e).__name__}: {e})"
        )
        return False

    if unfinished:
        logger.warning(
            f"Not refusing an empty vector storage: {unfinished} document(s) in "
            f"this workspace have not finished processing, so a graph ahead of "
            f"the vector store is work in flight rather than a defect."
        )
        return False
    return True


async def _source_is_populated(name: str, probe) -> bool | None:
    """Run one source-side probe: ``True``/``False``, or ``None`` if unanswerable.

    ``None`` and ``False`` are handled identically by the caller, but they are
    kept apart here so the log says which one happened. Neither can refuse: a
    source that is empty poses no question, and a source that cannot be read
    supplies no evidence.

    Worth stating because it looks like an omission: ``BaseKVStorage.is_empty``
    CATCHES its backend errors and answers ``True``, so a ``text_chunks`` read
    that fails arrives here as "no chunks" rather than as a raise. That lands on
    the same branch as a genuinely empty store -- skip, do not refuse -- which
    is the direction this module wants everywhere, so the imprecision is
    accepted rather than worked around. It is the mirror image of why the
    INDEX side needed a new method: there, "I could not read it" answered as
    "empty" would refuse a healthy deployment.
    """
    try:
        return await probe()
    except Exception as e:
        logger.warning(
            f"Skipping the empty-container check for {name}: the source data "
            f"could not be read ({type(e).__name__}: {e})"
        )
        return None


async def _graph_has_nodes(graph) -> bool:
    """Whether the graph holds at least one entity.

    ``get_popular_labels`` rather than ``iter_labels``: it is abstract on
    ``BaseGraphStorage`` so every backend implements it, while ``iter_labels``
    fails closed on backends that never did.
    """
    return bool(await graph.get_popular_labels(limit=1))


async def _graph_has_edges(graph) -> bool:
    """Whether the graph holds at least one relation.

    ``iter_edges`` is the only bounded edge reader on the base class
    (``get_all_edges`` materializes the whole graph). It is fail-closed by
    default -- a backend that never implemented it raises
    ``StorageCapabilityError``, which the caller reads as "unanswerable" and
    skips. All seven in-tree graph backends implement it.
    """
    iterator = graph.iter_edges(batch_size=1)
    try:
        async for batch in iterator:
            return bool(batch)
        return False
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


async def _index_is_empty(name: str, vdb) -> bool | None:
    """Ask the vector storage whether it holds anything. ``None`` = no answer.

    Unlike every other read on these backends, ``is_empty`` is specified to
    RAISE on a backend failure instead of reporting a miss, which is the whole
    reason it exists: ``get_by_ids`` catches its transport errors and returns an
    empty list, so an outage and an empty container arrive as one value. Here
    they do not, and only a positive ``True`` can refuse.
    """
    try:
        return await vdb.is_empty()
    except StorageCapabilityError as e:
        logger.debug(
            f"Skipping the empty-container check for {name}: "
            f"{type(vdb).__name__} cannot answer it ({e})"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Skipping the empty-container check for {name}: the vector "
            f"storage could not be read ({type(e).__name__}: {e})"
        )
        return None


class _PairingGate:
    """The empty-container gate over one (source data, vector index) pairing.

    A vector storage is an INDEX over data held elsewhere, and there are three
    of them::

        text_chunks (KV)    ->  chunks_vdb
        graph entities      ->  entities_vdb
        graph relations     ->  relationships_vdb

    The rule is per-pairing, because each index is a separate container that a
    model change replaces separately: **if the indexed data is not empty, its
    index must not be empty either.** Checking only the entity pairing (which
    is what this gate did first) misses a corpus that produces text chunks but
    no extracted entities -- the graph is empty, so the check has nothing to
    ask, while ``chunks_vdb`` is silently serving nothing.

    The pairings do not substitute for one another. A workspace built entirely
    through ``acreate_entity`` has entities and no chunks; a corpus that
    extracts nothing has chunks and no entities. Only the pairing whose source
    is populated has a question to ask.

    ``doc_status`` is consulted at most once no matter how many pairings reach
    that branch, and never on the healthy path.
    """

    def __init__(self, *, doc_status, expect_empty_vector_storage: bool) -> None:
        self._doc_status = doc_status
        self._expect_empty = expect_empty_vector_storage
        self._vectors_expected: bool | None = None

    async def _are_vectors_expected(self) -> bool:
        if self._vectors_expected is None:
            self._vectors_expected = await _vectors_are_expected(self._doc_status)
        return self._vectors_expected

    async def check(self, *, name: str, source: str, vdb, source_probe) -> None:
        """Refuse iff the source is populated and the index is provably empty."""
        if vdb is None:
            return

        if not getattr(vdb, "persists_vectors", True):
            # NoopVectorDBStorage and anything else that declares it keeps no
            # vectors: emptiness is its design, not a defect. Graph-only
            # ingestion is a supported configuration, and without this the gate
            # would refuse every restart after its first document.
            # `rebuilding_vector_storage` is not the answer there: such a
            # deployment is not rebuilding anything. Same capability
            # `lightrag-rebuild-vdb` already reads.
            return

        if await _source_is_populated(name, source_probe) is not True:
            return

        if await _index_is_empty(name, vdb) is not True:
            return

        if self._expect_empty:
            # The caller owns the repopulation that follows -- an in-process
            # rebuild, or the supported switch from NoopVectorDBStorage
            # (graph-only ingestion) to a real vector backend. Refusing here
            # would block the only thing that clears the condition.
            logger.info(
                f"The {name} vector storage holds no vectors while {source} is "
                f"not empty. Serving anyway: this instance declared it is "
                f"rebuilding them."
            )
            return

        if not await self._are_vectors_expected():
            # An index behind its source, while a document is still in flight,
            # is a residue that HEALS: an ingest interrupted before its vector
            # flush, a batch that failed and will be retried. ``AGENTS.md``
            # *Consistency without transactions* is explicit that such a state
            # must not be escalated -- and escalating it here would refuse to
            # start the very process whose next run repairs it.
            logger.warning(
                f"The {name} vector storage holds no vectors while {source} is "
                f"not empty, but this workspace has unfinished documents. "
                f"Serving anyway: an ingest that has not finished writing its "
                f"vectors is repaired by the next pipeline run, not by a "
                f"refusal."
            )
            return

        raise VectorStorageEmptyError(
            vdb_name=name,
            container=getattr(vdb, "final_namespace", None),
            source=source,
        )


async def _run_adoption_probe(graph, entities_vdb, embedding_func) -> None:
    """Re-embed one stored entity and adopt the container if it matches.

    Separate from the gate above, and running after it, because the two ask
    different questions of different things. The gate asks whether a container
    is empty; this asks whether a NON-empty container's vectors came from this
    model. It samples on its own -- the gate no longer produces a sample, and
    this needs rows with both a ``content`` field and a stored vector, which
    emptiness alone never yields.

    ONLY the entity store is adoptable here, because it is the only one this
    function gathers evidence about. The three vector targets share an
    ``embedding_func`` but NOT a history: `lightrag-rebuild-vdb` rebuilds
    entities, relationships and chunks separately (and offers
    "entities_vdb + relationships_vdb" as a partial target), so an interrupted
    rebuild after a same-dimension model change can leave entities in the
    current space while relationships or chunks still hold the previous model's
    vectors. Adopting those on the entity verdict would stamp this model's name
    onto foreign vectors -- permanently, and invisibly, which is the exact lie
    this whole feature exists to prevent. They stay unmarked until they can be
    probed with their own sample.
    """
    try:
        if not await entities_vdb.vector_space_adoption_pending():
            return
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            f"Could not ask {type(entities_vdb).__name__} whether it needs "
            f"embedding-space adoption: {e}"
        )
        return

    if declared_model_name(embedding_func) is None:
        # Nothing to record, so nothing to prove. Reached only if a backend
        # reported pending without checking; the storages answer False here.
        return

    try:
        sample_ids = await _sample_entity_ids(graph, SAMPLE_SIZE)
    except Exception as e:
        logger.warning(
            f"Embedding-space adoption deferred: the graph storage could not "
            f"supply a sample ({type(e).__name__}: {e})"
        )
        return
    if not sample_ids:
        return

    try:
        found = await entities_vdb.get_by_ids(sample_ids)
    except Exception as e:
        logger.warning(
            f"Embedding-space adoption deferred: the entity vector storage "
            f"could not be read ({type(e).__name__}: {e})"
        )
        return

    # The backends disagree on the shape of a MISS. ``BaseVectorStorage``
    # documents "the objects that were found", and most return a compacted
    # list; Nano returns one entry per requested id, positionally, with None
    # where a row is absent. Counting the list rather than the rows would feed
    # ``None`` placeholders into the probe below.
    rows = [row for row in (found or []) if isinstance(row, dict)]
    if not rows:
        return

    try:
        vectors = await entities_vdb.get_vectors_by_ids(
            [row["id"] for row in rows if row.get("id")]
        )
    except Exception as e:
        logger.warning(
            f"Embedding-space adoption deferred: the stored vectors could not "
            f"be read ({type(e).__name__}: {e})"
        )
        return

    verdict, detail = await _probe_same_embedding_space(rows, vectors, embedding_func)

    if verdict is False:
        raise VectorSpaceMismatchError(
            backend=type(entities_vdb).__name__,
            container=getattr(entities_vdb, "final_namespace", "entities"),
            expected_model=declared_model_name(embedding_func),
            stored_model=None,
            detail=(
                f"Re-embedding a stored record with the configured model did "
                f"not reproduce its stored vector ({detail}), so these vectors "
                f"were written by a different model."
            ),
        )

    if verdict is None:
        logger.warning(
            f"Embedding-space adoption deferred: {detail}. The vector storages "
            f"stay unmarked, so a later swap to a different model of the same "
            f"dimension cannot be detected yet; this is retried on the next "
            f"start."
        )
        return

    try:
        if await entities_vdb.adopt_vector_space():
            logger.info(
                f"Recorded the embedding model on {type(entities_vdb).__name__} "
                f"'{getattr(entities_vdb, 'namespace', '?')}' ({detail})"
            )
    except VectorSpaceMismatchError:
        raise
    except Exception as e:  # pragma: no cover - adopt must not raise
        logger.warning(
            f"Could not record the embedding model on "
            f"{type(entities_vdb).__name__}: {e}"
        )


async def check_vector_space_at_startup(
    *,
    graph,
    entities_vdb,
    relationships_vdb=None,
    chunks_vdb=None,
    text_chunks=None,
    doc_status=None,
    embedding_func,
    expect_empty_vector_storage: bool = False,
) -> None:
    """Run the empty-container gate over all three pairings, then the probe.

    Args:
        graph: the graph storage, already initialized. It is the source side of
            two pairings: its entities and its relations.
        entities_vdb: the entity vector storage, already initialized.
        relationships_vdb: the relation vector storage. Skipped when ``None``.
        chunks_vdb: the chunk vector storage. Skipped when ``None``.
        text_chunks: the chunk KV storage -- the source side of the chunk
            pairing, and the only one that does not come from the graph.
        doc_status: the doc-status storage. Consulted at most once, and only
            when a pairing is about to refuse, so the healthy path pays nothing.
        embedding_func: this instance's embedding function.
        expect_empty_vector_storage: this caller is about to repopulate the
            vector storages, so an empty one is the expected starting state
            rather than a defect. See ``LightRAG.rebuilding_vector_storage``.

    Raises:
        VectorStorageEmptyError: some indexed data is not empty while its vector
            storage holds nothing, no document is unfinished, and the caller did
            not declare that it is about to rebuild.
        VectorSpaceMismatchError: the probe ran and the stored vectors did not
            come from this embedding model.

    Nothing else escapes. Every other failure -- an unreachable graph backend, a
    vector store that errors on a read, a marker that cannot be written -- is
    logged and swallowed, because none of them is evidence about the embedding
    space and none of them was a startup failure before this check existed.
    """
    gate = _PairingGate(
        doc_status=doc_status,
        expect_empty_vector_storage=expect_empty_vector_storage,
    )

    if text_chunks is not None:
        await gate.check(
            name="chunks",
            source="the text chunk storage",
            vdb=chunks_vdb,
            source_probe=lambda: _source_is_empty_inverted(text_chunks),
        )

    await gate.check(
        name="entities",
        source="the knowledge graph",
        vdb=entities_vdb,
        source_probe=lambda: _graph_has_nodes(graph),
    )

    await gate.check(
        name="relationships",
        source="the knowledge graph",
        vdb=relationships_vdb,
        source_probe=lambda: _graph_has_edges(graph),
    )

    if entities_vdb is not None and getattr(entities_vdb, "persists_vectors", True):
        await _run_adoption_probe(graph, entities_vdb, embedding_func)


async def _source_is_empty_inverted(kv_storage) -> bool:
    """``True`` when the KV storage holds rows. Adapts the KV sense of empty."""
    return not await kv_storage.is_empty()
