"""The two startup checks that no single storage can make on its own.

Both ask a question that spans the graph and a vector storage, which is why
they live one layer up in ``LightRAG.initialize_storages()`` rather than inside
a backend's ``initialize()``. A vector storage cannot tell on its own whether
being empty is a defect: that depends on the data it INDEXES, which lives in
another storage entirely.

**The coverage gate.** A vector storage is an INDEX over a source dataset, and
an index that covers nothing while its source is not empty is a broken
deployment that reports itself as an empty answer. Nothing else notices: every
vector query simply returns nothing, which reads as "no relevant context". The
detectable shape is cross-storage: *the graph holds entities and the vector
store holds none*.

What reaches that shape is not one cause but a family: a container dropped or a
volume not mounted, a rebuild that stopped halfway, the vector backend switched
without migrating the data, a workspace prefix that does not match the one the
vectors were written under -- and, on the backends that encode the embedding
model in the container NAME (Milvus, Qdrant, PostgreSQL), a model change, which
provisions a new, empty, correctly named container that no marker can ever
fire on. That last one is an *identity* question that this gate happens to
catch through its consequence; the recorded per-workspace embedding space
answers it directly and earlier. This gate's own question stays coverage, and
neither substitutes for the other. See ``docs/design/VectorSpaceProvenance.md``.

The condition self-clears: a rebuild answers the question by populating the
container, which is why the gate asks it fresh at every startup rather than
recording anything.

The check runs over three pairings, one per vector target, because each is a
separate container, filled by a separate write path and replaced separately::

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
carries evidence: re-embed a few stored ``content`` fields with the current
model and compare them against the vectors stored beside them.

The probe runs PER TARGET, each on a sample drawn from its own source --
entities from the graph's most-connected labels, relations from the first
batch of ``iter_edges``, chunks from the first page of
``text_chunks.iter_rows`` -- because the three targets share an
``embedding_func`` but not a history: `lightrag-rebuild-vdb` rebuilds them
separately, so an interrupted rebuild after a same-dimension model change can
leave entities in the current space while relationships or chunks still hold
the previous model's vectors. A verdict about one container is evidence about
that container only; nothing is ever adopted, or recorded as a baseline, on a
sibling's verdict.

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
from dataclasses import dataclass, field
from typing import Any, Collection

from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP, RELATION_NO_EVIDENCE_SOURCE_IDS
from lightrag.exceptions import (
    StorageCapabilityError,
    VectorSpaceMismatchError,
    VectorStorageEmptyError,
)
from lightrag.kg.vector_space import declared_model_name
from lightrag.utils import compute_mdhash_id, logger, make_relation_vdb_ids


@dataclass
class StartupEvidence:
    """What the startup checks established, for the caller that records baselines.

    ``source_populated`` answers, per vector target, whether the data that
    target INDEXES holds anything: ``True`` / ``False``, or ``None`` when the
    source could not be read or the pairing was not examined. ``probes`` is
    the adoption probe's verdict per target: ``True`` when the stored vectors
    reproduced under the configured model, ``None`` when the probe did not run
    or could not answer (``probe_details`` says why). A negative verdict is
    never stored here -- it raises ``VectorSpaceMismatchError`` instead,
    because a refusal is not evidence to record anything on.

    Only a positive verdict may establish an ``origin=probe`` baseline for
    THAT target, and only a ``False`` source whose container is confirmed
    empty too (``index_empty``) may establish an ``origin=empty`` one.
    ``None`` anywhere means "no evidence", and no evidence records nothing.
    """

    source_populated: dict[str, bool | None] = field(default_factory=dict)
    # Per target whose SOURCE is empty: whether its vector container is empty
    # too, from the fail-loud ``is_empty()``; ``None`` when it could not be
    # read. An ``origin=empty`` baseline needs BOTH to be empty -- a populated
    # container behind an empty source holds vectors nobody can vouch for and
    # nothing to sample, so it records nothing.
    index_empty: dict[str, bool | None] = field(default_factory=dict)
    probes: dict[str, bool | None] = field(default_factory=dict)
    probe_details: dict[str, str] = field(default_factory=dict)


# How many graph entities the ADOPTION PROBE looks up to find one row carrying
# both a ``content`` field and a stored vector. Not a confidence knob -- the
# probe needs exactly one usable row, and the margin covers rows whose content
# is missing or whose vector cannot be fetched. The coverage gate does not
# sample at all; it asks ``is_empty()``.
SAMPLE_SIZE = 32

# How many graph objects to trace back to their owning document. Bounded
# because each one costs payload reads rather than names, and sampling is sound
# here: a miss only declines to refuse (see ``_nothing_will_heal``). Paid only
# on the branch that was already about to exempt, never on a healthy start.
DOCUMENTLESS_SAMPLE_SIZE = 32

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

# How many stored records the probe compares before it will adopt. ONE is not
# enough, and the case that proves it is reachable today: an upgrade that also
# switches to a same-dimension model starts unmarked when the embedder is
# unreachable (inconclusive), keeps serving, and then accepts new writes -- so
# the container ends up holding the previous model's vectors AND this model's.
# A single row picked out of that mixture adopts on whichever one it happened
# to land on, and a marker recorded over a mixed container hides the foreign
# half forever.
#
# Comparing several rows turns that mixture from something adoption conceals
# into something the probe REPORTS: a row that fails while others pass is a
# positive verdict about the container, not an ambiguity. One batched embedding
# call, so the cost is the same round trip as before.
PROBE_ROWS = 8

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


async def _sample_relation_ids(graph, limit: int) -> list[str]:
    """Vector ids for up to ``limit`` relations the graph actually holds.

    The first batch of ``iter_edges`` -- the only bounded edge reader on the
    base class -- mapped to BOTH candidate relation ids: the canonical one the
    write path uses (``make_relation_vdb_ids`` puts it first) and the legacy
    reverse-order one a historical custom-KG import may have hashed under.
    Sampling only the canonical id would never examine an all-legacy store
    (no baseline, ever) and would judge a mixed one on its canonical rows
    alone. A backend that never implemented ``iter_edges`` raises
    ``StorageCapabilityError``, which the caller reads as "could not sample".
    """
    iterator = graph.iter_edges(batch_size=limit)
    try:
        async for batch in iterator:
            ids: list[str] = []
            for edge in batch:
                if not isinstance(edge, dict):
                    continue
                src, tgt = edge.get("source"), edge.get("target")
                if src is None or tgt is None:
                    continue
                src, tgt = str(src), str(tgt)
                if not src.strip() or not tgt.strip():
                    continue
                for rel_id in make_relation_vdb_ids(src, tgt):
                    if rel_id not in ids:
                        ids.append(rel_id)
            return ids[: 2 * limit]
        return []
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


async def _sample_chunk_ids(text_chunks, limit: int) -> list[str]:
    """Vector ids for up to ``limit`` DISTINCT chunks the KV store holds.

    The first page of ``BaseKVStorage.iter_rows``, closed as soon as the
    budget is spent. Chunk vectors are keyed by the chunk id, so the row id is
    the vector id.

    **Distinct, and the sample may come back short.** ``iter_rows`` is a
    best-effort snapshot: ``RedisKVStorage`` says in its own docstring that
    ``SCAN`` may return a key twice while the keyspace is rehashing, and puts
    the de-duplication on the caller. This is that caller, and it is the one
    that must care -- ``_probe_same_embedding_space`` treats each row as an
    independent record, so a repeated row would occupy comparison slots
    without adding evidence, and a container holding two embedding spaces
    would be judged on fewer distinct chunks than the sample size promises.
    Half a sample of genuinely distinct rows is worth more than a full one of
    copies, so the budget below counts rows EXAMINED, not ids kept: the read
    stays exactly the size it was, and duplicates shrink the sample instead of
    padding it.

    What this cannot do is make a sample representative. A container whose
    foreign vectors all sit outside the first page is adopted on the rows that
    were seen -- the residue inherent to sampling, stated in
    ``_probe_same_embedding_space``. De-duplication keeps the sample as wide
    as the read allows; it does not widen the read.
    """
    iterator = text_chunks.iter_rows(page_size=limit)
    ids: list[str] = []
    seen: set[str] = set()
    examined = 0
    try:
        async for row in iterator:
            examined += 1
            if isinstance(row, dict):
                row_id = row.get("_id") or row.get("id")
                if isinstance(row_id, str) and row_id and row_id not in seen:
                    seen.add(row_id)
                    ids.append(row_id)
            if len(ids) >= limit or examined >= limit:
                break
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()
    return ids


async def _probe_same_embedding_space(
    rows: list[dict[str, Any]],
    vectors: dict[str, list[float]],
    embedding_func,
) -> tuple[bool | None, str]:
    """Re-embed several stored contents and compare them with their vectors.

    Returns ``(verdict, detail)`` where the verdict is ``True`` (adopt),
    ``False`` (refuse) or ``None`` (inconclusive -- do neither).

    **Adoption needs every compared record to agree.** A container can hold
    vectors from two models at once: an upgrade that also switched to a
    same-dimension model starts unmarked when the probe cannot run, keeps
    serving, and then takes new writes. Adopting on one row out of that mixture
    stamps this model over the foreign half and hides it permanently -- the
    same lie as adopting a store nobody probed, one level down.

    So a row that fails while others pass is treated as what it is: positive
    evidence that the container is NOT homogeneous, which refuses. The rule
    stays "only a negative verdict may refuse" -- a reproduced cosine of 0.3 is
    a negative verdict, whoever else in the container agrees.
    """
    candidates: list[tuple[str, str, Any]] = []
    for row in rows:
        row_id = row.get("id")
        content = row.get("content")
        if not row_id or not isinstance(content, str) or not content.strip():
            continue
        stored = vectors.get(row_id)
        if stored is None:
            continue
        candidates.append((row_id, content, stored))
        if len(candidates) >= PROBE_ROWS:
            break

    if not candidates:
        return None, "no sampled record carried both a content field and a vector"

    try:
        # One call for all of them: the cost is the round trip, not the texts.
        # context="document" matches how every backend embeds on the way in; a
        # provider that prefixes queries differently would otherwise be
        # compared against vectors it never would have written.
        fresh = await asyncio.wait_for(
            embedding_func(
                [content for _, content, _ in candidates], context="document"
            ),
            timeout=_probe_timeout(),
        )
    except asyncio.TimeoutError:
        return None, f"the embedding call timed out after {_probe_timeout()}s"
    except Exception as e:
        return None, f"the embedding call failed ({type(e).__name__}: {e})"

    adopted: list[str] = []
    inconclusive: list[str] = []
    for index, (row_id, _, stored) in enumerate(candidates):
        try:
            fresh_vector = fresh[index]
        except (IndexError, TypeError, KeyError):
            inconclusive.append(
                f"the embedding function returned no vector for record '{row_id}'"
            )
            continue

        similarity = _cosine(fresh_vector, stored)
        if similarity is None:
            inconclusive.append(
                f"the stored and freshly embedded vectors for record "
                f"'{row_id}' are not comparable"
            )
            continue
        if similarity <= REFUSE_COSINE:
            if adopted:
                return False, (
                    f"cosine {similarity:.3f} against record '{row_id}' while "
                    f"{len(adopted)} other sampled record(s) reproduced their "
                    f"vectors, so this container holds vectors from more than "
                    f"one embedding space"
                )
            return False, f"cosine {similarity:.3f} against record '{row_id}'"
        if similarity < ADOPT_COSINE:
            inconclusive.append(
                f"cosine {similarity:.3f} against record '{row_id}' falls "
                f"between {REFUSE_COSINE} and {ADOPT_COSINE}, which settles "
                f"nothing"
            )
            continue
        adopted.append(row_id)

    # Negative evidence anywhere in the batch outranks an inconclusive row.
    if inconclusive:
        return None, "; ".join(inconclusive)
    return True, (
        f"{len(adopted)} sampled record(s) reproduced their stored vectors, "
        f"including '{adopted[0]}'"
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

    Every other state -- PENDING, PARSING, ANALYZING, PREPROCESSED, PROCESSING,
    FAILED -- is the opposite: the residue has an owner and heals by being
    retried. The list is DERIVED from ``DocStatus`` rather than written out,
    which is the safe direction for a state added later: a new member counts as
    unfinished and can only ever suppress a refusal, never enable one. Counting
    only the unfinished states is also what keeps a PROCESSED row from being
    read as
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

    The distinction matters beyond the log since the verdict also feeds the
    embedding baselines: ``False`` records the configured model as that
    target's ``origin=empty`` baseline, ``None`` records nothing. So a probe
    must answer ``False`` only on a read that would have raised had it failed.
    ``BaseKVStorage.is_empty`` does not qualify -- it catches its errors and
    answers ``True`` -- which is why the chunk source is read through
    ``_chunk_source_is_populated``, in its strict mode, on the starts that
    have a chunk baseline to establish. The graph readers behind the other two
    probes propagate their failures.
    """
    try:
        return await probe()
    except Exception as e:
        logger.warning(
            f"Skipping the coverage check for {name}: the source data "
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


def _chunk_ids_of(source_id: Any) -> list[str]:
    """The real chunk ids a graph object's ``source_id`` names, if any.

    Placeholders are dropped: ``RELATION_NO_EVIDENCE_SOURCE_IDS`` is the repo's
    existing name for what the admin writers stamp instead of a chunk id
    (``acreate_entity`` defaults to ``"manual_creation"``), and an empty field
    means the same thing. Reusing that set keeps one definition rather than a
    second that can drift.
    """
    if not isinstance(source_id, str):
        return []
    return [
        part.strip()
        for part in source_id.split(GRAPH_FIELD_SEP)
        if part.strip() and part.strip() not in RELATION_NO_EVIDENCE_SOURCE_IDS
    ]


async def _node_source_ids(graph, limit: int) -> list[Any]:
    """``source_id`` of up to ``limit`` graph entities."""
    labels = await graph.get_popular_labels(limit=limit)
    if not labels:
        return []
    nodes = await graph.get_nodes_batch(list(labels))
    return [
        node.get("source_id")
        for node in (nodes or {}).values()
        if isinstance(node, dict)
    ]


async def _edge_source_ids(graph, limit: int) -> list[Any]:
    """``source_id`` of the first batch of graph relations."""
    iterator = graph.iter_edges(batch_size=limit)
    try:
        async for batch in iterator:
            return [edge.get("source_id") for edge in batch if isinstance(edge, dict)]
        return []
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


async def _finished_doc_chunk_ids(doc_status, limit: int) -> list[str]:
    """Chunk ids belonging to documents that have FINISHED.

    The chunk pairing's equivalent of the graph probes, and it has to come from
    doc-status rather than from the source: ``BaseKVStorage`` has no enumeration
    API, so ``text_chunks`` can say it is non-empty but not which rows it holds.
    Inverting the question -- ask doc-status for chunks a retry will NOT rewrite,
    then confirm they are really in ``text_chunks`` -- needs only bounded reads.

    One page of PROCESSED documents, hydrated for their ``chunks_list``: the
    pattern ``get_full_docs_by_ids`` documents for itself. Both reads are
    strict, so an incomplete answer raises rather than under-reporting into a
    refusal.
    """
    if doc_status is None:
        return []
    page = await doc_status.get_docs_by_statuses_page(
        [DocStatus.PROCESSED], limit=limit, strict=True
    )
    doc_ids = list((page.docs or {}).keys())
    if not doc_ids:
        return []
    records = await doc_status.get_full_docs_by_ids(doc_ids, strict=True)
    chunk_ids: list[str] = []
    for record in (records or {}).values():
        for chunk_id in getattr(record, "chunks_list", None) or []:
            if isinstance(chunk_id, str) and chunk_id.strip():
                chunk_ids.append(chunk_id.strip())
    return chunk_ids


async def _is_owned_by_unfinished_doc(chunk_ids, text_chunks, doc_status) -> bool:
    """Whether a retry of an unfinished document would rewrite these chunks.

    The actual question behind the doc-status exemption. Resolved by following
    the object's own trail -- ``source_id`` -> ``text_chunks`` -> ``full_doc_id``
    -> ``doc_status`` -- rather than by guessing from the shape of the id, which
    is what an earlier version did and got wrong twice:

    * ``acreate_entity`` stamps the placeholder ``"manual_creation"``, so a
      placeholder test caught it;
    * but ``ainsert_custom_kg`` maps its entities onto the call's OWN chunks, so
      their ``source_id`` is a real ``chunk-*`` id while the chunk's
      ``full_doc_id`` names no doc-status row at all. A placeholder test reads
      that as document-produced and hands it the exemption it must not get.

    Following the trail settles both, and a third case the placeholder test
    never reached: an object produced by a document that has FINISHED, in a
    workspace where some OTHER document is unfinished. That document's retry
    will not rewrite this object either.

    Every read is bounded by the sample. Unreadable answers ``True`` -- "assume
    a retry covers it" -- because only positive evidence may refuse.
    """
    if not chunk_ids:
        # Placeholders only: no chunk, so no document, so no retry.
        return False
    if text_chunks is None or doc_status is None:
        return True

    rows = await text_chunks.get_by_ids(sorted(set(chunk_ids)))
    doc_ids = {
        row.get("full_doc_id")
        for row in (rows or [])
        if isinstance(row, dict) and row.get("full_doc_id")
    }
    if not doc_ids:
        # Nothing came back. That is NOT evidence of an orphan: like every
        # other read on these backends, ``BaseKVStorage.get_by_ids`` catches
        # its transport errors and returns an empty list, so a cluster blip and
        # a genuinely missing chunk arrive as the same value -- and reading it
        # as "no document" would refuse a healthy deployment, the exact defect
        # ``is_empty()`` exists to avoid on the vector side.
        #
        # This costs a miss, not a false refusal: an object whose chunk really
        # is gone gets the exemption it should not have. The custom-KG case is
        # unaffected, because those chunks EXIST and name a ``full_doc_id`` that
        # has no doc-status row -- resolved below, not here.
        return True

    records = await doc_status.get_docs_by_ids(sorted(doc_ids), strict=True)
    return any(
        getattr(record, "status", None) is not DocStatus.PROCESSED
        for record in (records or {}).values()
    )


async def _nothing_will_heal(name: str, probe, text_chunks, doc_status) -> bool:
    """Whether a retry provably cannot restore what is missing here.

    Answers the one thing the doc-status COUNT cannot: that count is
    workspace-wide, while an unfinished document only ever rewrites the objects
    it produces. This resolves each sampled object to its owning document and
    asks whether THAT document is unfinished.

    **Sampling is sound here, unlike for emptiness.** A sample that misses the
    unhealable object answers ``False``, which only declines to refuse -- the
    behaviour without this check at all. It can add refusals for what it finds,
    never remove one. ``is_empty()`` had the opposite exposure, which is why
    that one had to be asked rather than sampled.

    A probe that cannot run answers ``False`` for the same reason.
    """
    if probe is None:
        return False
    try:
        source_ids = await probe()
        for source_id in source_ids:
            owned = await _is_owned_by_unfinished_doc(
                _chunk_ids_of(source_id), text_chunks, doc_status
            )
            if not owned:
                return True
        return False
    except Exception as e:
        logger.warning(
            f"Could not tell whether a retry would restore {name} "
            f"({type(e).__name__}: {e}); treating the missing vectors as work "
            f"a retry will heal"
        )
        return False


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
            f"Skipping the coverage check for {name}: "
            f"{type(vdb).__name__} cannot answer it ({e})"
        )
        return None
    except Exception as e:
        logger.warning(
            f"Skipping the coverage check for {name}: the vector "
            f"storage could not be read ({type(e).__name__}: {e})"
        )
        return None


class _PairingGate:
    """The coverage gate over one (source data, vector index) pairing.

    A vector storage is an INDEX over data held elsewhere, and there are three
    of them::

        text_chunks (KV)    ->  chunks_vdb
        graph entities      ->  entities_vdb
        graph relations     ->  relationships_vdb

    The rule is per-pairing, because each index is a separate container with
    its own write path and its own ways of being lost: **if the indexed data is
    not empty, its index must not be empty either.** Checking only the entity
    pairing (which
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

    def __init__(
        self, *, doc_status, text_chunks=None, expect_empty_vector_storage: bool
    ) -> None:
        self._doc_status = doc_status
        # Held for the healability trail (source_id -> chunk -> document), not
        # for the chunk pairing, which reads it through its own source probe.
        self._text_chunks = text_chunks
        self._expect_empty = expect_empty_vector_storage
        self._vectors_expected: bool | None = None

    async def _are_vectors_expected(self) -> bool:
        if self._vectors_expected is None:
            self._vectors_expected = await _vectors_are_expected(self._doc_status)
        return self._vectors_expected

    async def check(
        self, *, name: str, source: str, vdb, source_probe, no_healing_probe=None
    ) -> bool | None:
        """Refuse iff the source is populated and the index is provably empty.

        Returns the source-side verdict (``True`` populated, ``False`` empty,
        ``None`` unreadable or not examined) so the caller can record a
        baseline on it. ``no_healing_probe`` is consulted only on the branch
        the unfinished-document exemption would otherwise take. See
        ``_nothing_will_heal``.
        """
        if vdb is None:
            return None

        if not getattr(vdb, "persists_vectors", True):
            # NoopVectorDBStorage and anything else that declares it keeps no
            # vectors: emptiness is its design, not a defect. Graph-only
            # ingestion is a supported configuration, and without this the gate
            # would refuse every restart after its first document.
            # `rebuilding_vector_storage` is not the answer there: such a
            # deployment is not rebuilding anything. Same capability
            # `lightrag-rebuild-vdb` already reads.
            return None

        populated = await _source_is_populated(name, source_probe)
        if populated is not True:
            return populated

        if await _index_is_empty(name, vdb) is not True:
            return True

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
            return True

        if not await self._are_vectors_expected():
            # An index behind its source, while a document is still in flight,
            # is a residue that HEALS: an ingest interrupted before its vector
            # flush, a batch that failed and will be retried. ``AGENTS.md``
            # *Consistency without transactions* is explicit that such a state
            # must not be escalated -- and escalating it here would refuse to
            # start the very process whose next run repairs it.
            #
            # But that exemption is workspace-wide while the evidence it
            # excuses is per-container: an unfinished document explains the
            # missing vectors for the objects THAT document produces, and
            # nothing else. When the source also holds objects no document
            # produced, no retry will recreate them, so the exemption does not
            # reach them and the refusal stands.
            if not await _nothing_will_heal(
                source, no_healing_probe, self._text_chunks, self._doc_status
            ):
                logger.warning(
                    f"The {name} vector storage holds no vectors while {source} "
                    f"is not empty, but this workspace has unfinished "
                    f"documents. Serving anyway: an ingest that has not "
                    f"finished writing its vectors is repaired by the next "
                    f"pipeline run, not by a refusal."
                )
                return True
            logger.warning(
                f"The {name} vector storage holds no vectors while {source} is "
                f"not empty. This workspace has unfinished documents, but it "
                f"also holds objects no unfinished document would rewrite, so "
                f"no pipeline run can recreate their vectors. Refusing."
            )

        raise VectorStorageEmptyError(
            vdb_name=name,
            container=getattr(vdb, "final_namespace", None),
            source=source,
            workspace=_effective_workspace(vdb),
        )


def _effective_workspace(vdb) -> str | None:
    """The workspace ``vdb`` actually opened, after any backend override.

    Qdrant keeps the override in ``effective_workspace`` and leaves
    ``workspace`` as configured; Milvus, PostgreSQL, MongoDB and OpenSearch
    overwrite ``workspace`` with it during setup. ``None`` when neither is
    there.
    """
    effective = getattr(vdb, "effective_workspace", None)
    if effective is not None:
        return effective
    return getattr(vdb, "workspace", None)


async def _probe_target(
    name: str, vdb, sampler, embedding_func, *, force: bool = False
) -> tuple[bool | None, str]:
    """Re-embed stored records of ONE target; adopt its marker if they match.

    Separate from the gate above, and running after it, because the two ask
    different questions of different things. The gate asks whether a container
    is empty; this asks whether a NON-empty container's vectors came from this
    model. ``sampler`` draws the candidate ids from the target's own source
    (see the module docstring); the probe needs rows with both a ``content``
    field and a stored vector, which emptiness alone never yields.

    Runs when the container's marker is pending, or when ``force`` is set --
    the caller has an absent BASELINE for this target to establish and needs
    the same evidence, whatever the container marker says (Milvus, Qdrant and
    PostgreSQL carry none). Returns ``(verdict, detail)``: ``True``
    reproduced, ``None`` did not run or could not answer. A negative verdict
    raises. The verdict is about THIS container only.
    """
    try:
        pending = bool(await vdb.vector_space_adoption_pending())
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            f"Could not ask {type(vdb).__name__} whether {name} needs "
            f"embedding-space adoption: {e}"
        )
        pending = False
    if not pending and not force:
        return None, "the container already records its embedding model"

    if declared_model_name(embedding_func) is None:
        # Nothing to record, so nothing to prove. Reached only if a backend
        # reported pending without checking; the storages answer False here.
        return None, "this process declares no embedding model name"

    try:
        sample_ids = await sampler()
    except Exception as e:
        detail = (
            f"the {name} source could not supply a sample ({type(e).__name__}: {e})"
        )
        logger.warning(f"Embedding-space adoption of {name} deferred: {detail}")
        return None, detail
    if not sample_ids:
        return None, f"the {name} source holds nothing to sample"

    try:
        found = await vdb.get_by_ids(sample_ids)
    except Exception as e:
        detail = (
            f"the {name} vector storage could not be read ({type(e).__name__}: {e})"
        )
        logger.warning(f"Embedding-space adoption of {name} deferred: {detail}")
        return None, detail

    # The backends disagree on the shape of a MISS. ``BaseVectorStorage``
    # documents "the objects that were found", and most return a compacted
    # list; Nano returns one entry per requested id, positionally, with None
    # where a row is absent. Counting the list rather than the rows would feed
    # ``None`` placeholders into the probe below.
    rows = [row for row in (found or []) if isinstance(row, dict)]
    if not rows:
        return None, f"none of the sampled {name} records has a vector record"

    try:
        vectors = await vdb.get_vectors_by_ids(
            [row["id"] for row in rows if row.get("id")]
        )
    except Exception as e:
        detail = f"the stored vectors could not be read ({type(e).__name__}: {e})"
        logger.warning(f"Embedding-space adoption of {name} deferred: {detail}")
        return None, detail

    verdict, detail = await _probe_same_embedding_space(rows, vectors, embedding_func)

    if verdict is False:
        raise VectorSpaceMismatchError(
            backend=type(vdb).__name__,
            container=getattr(vdb, "final_namespace", name),
            expected_model=declared_model_name(embedding_func),
            stored_model=None,
            detail=(
                f"Re-embedding a stored {name} record with the configured model "
                f"did not reproduce its stored vector ({detail}), so these "
                f"vectors were written by a different model."
            ),
        )

    if verdict is None:
        logger.warning(
            f"Embedding-space adoption of {name} deferred: {detail}. The "
            f"container stays unmarked, so a later swap to a different model of "
            f"the same dimension cannot be detected yet; this is retried on the "
            f"next start."
        )
        return None, detail

    if pending:
        try:
            if await vdb.adopt_vector_space():
                logger.info(
                    f"Recorded the embedding model on {type(vdb).__name__} "
                    f"'{getattr(vdb, 'namespace', name)}' ({detail})"
                )
        except VectorSpaceMismatchError:
            raise
        except Exception as e:  # pragma: no cover - adopt must not raise
            logger.warning(
                f"Could not record the embedding model on {type(vdb).__name__} "
                f"'{name}': {e}"
            )
    return True, detail


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
    baseline_targets: Collection[str] = (),
) -> StartupEvidence:
    """Run the coverage gate over all three pairings, then the probes.

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
        baseline_targets: targets (``"entities"``, ``"relationships"``,
            ``"chunks"``) whose baseline is absent, so the caller is about to
            establish one from what this call returns
            (``docs/design/ConfigurationStorage.md``). Three things key off
            it, all for the same reason -- a durable claim needs evidence a
            coverage check does not: the target's probe runs even when the
            container's own marker needs no adoption (on the backends without
            a marker this is the only way a probe ever runs); its source is
            read strictly, so "empty" cannot come from a failed read; and its
            container is asked whether it is empty. A target that is not
            listed pays none of that, which is what keeps every later start
            on the read the coverage gate has always used.

    Returns:
        The evidence the checks gathered -- see ``StartupEvidence``.

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
        text_chunks=text_chunks,
        expect_empty_vector_storage=expect_empty_vector_storage,
    )
    evidence = StartupEvidence()

    if text_chunks is not None:
        evidence.source_populated["chunks"] = await gate.check(
            name="chunks",
            source="the text chunk storage",
            vdb=chunks_vdb,
            source_probe=lambda: _chunk_source_is_populated(
                text_chunks, strict="chunks" in baseline_targets
            ),
            no_healing_probe=lambda: _finished_doc_chunk_ids(
                doc_status, DOCUMENTLESS_SAMPLE_SIZE
            ),
        )

    evidence.source_populated["entities"] = await gate.check(
        name="entities",
        source="the knowledge graph",
        vdb=entities_vdb,
        source_probe=lambda: _graph_has_nodes(graph),
        no_healing_probe=lambda: _node_source_ids(graph, DOCUMENTLESS_SAMPLE_SIZE),
    )

    evidence.source_populated["relationships"] = await gate.check(
        name="relationships",
        source="the knowledge graph",
        vdb=relationships_vdb,
        source_probe=lambda: _graph_has_edges(graph),
        no_healing_probe=lambda: _edge_source_ids(graph, DOCUMENTLESS_SAMPLE_SIZE),
    )

    # For a target whose source is EMPTY, ask its container too: an
    # ``origin=empty`` baseline is a durable claim that the configured model
    # is the one this container's vectors are in, and it holds only when there
    # are none. A source lost or restored empty while its container survived
    # must leave the baseline absent, not stamp the model over the survivors.
    for name, vdb in (
        ("chunks", chunks_vdb),
        ("entities", entities_vdb),
        ("relationships", relationships_vdb),
    ):
        if name not in baseline_targets:
            # Nobody consumes the answer: only ``_establish_embedding_baselines``
            # reads ``index_empty``, and only for the targets it claims.
            continue
        if evidence.source_populated.get(name) is not False or vdb is None:
            continue
        if not getattr(vdb, "persists_vectors", True):
            continue
        evidence.index_empty[name] = await _index_is_empty_for_baseline(name, vdb)

    # One probe per target, each on its own sample. The chunk probe needs the
    # chunk source; without it there is nothing to sample from.
    probes = (
        ("entities", entities_vdb, lambda: _sample_entity_ids(graph, SAMPLE_SIZE)),
        (
            "relationships",
            relationships_vdb,
            lambda: _sample_relation_ids(graph, SAMPLE_SIZE),
        ),
        (
            "chunks",
            chunks_vdb,
            (lambda: _sample_chunk_ids(text_chunks, SAMPLE_SIZE))
            if text_chunks is not None
            else None,
        ),
    )
    for name, vdb, sampler in probes:
        if vdb is None or sampler is None:
            continue
        if not getattr(vdb, "persists_vectors", True):
            continue
        evidence.probes[name], evidence.probe_details[name] = await _probe_target(
            name, vdb, sampler, embedding_func, force=name in baseline_targets
        )
    return evidence


async def _index_is_empty_for_baseline(name: str, vdb) -> bool | None:
    """``vdb.is_empty()`` for the baseline decision: ``True`` / ``False``, or
    ``None`` when the container could not be read or cannot answer. Same
    fail-loud read as ``_index_is_empty``; only the log differs, because here
    the consequence is an unrecorded baseline, not a skipped coverage check.
    """
    try:
        return bool(await vdb.is_empty())
    except StorageCapabilityError as e:
        logger.info(
            f"Whether the {name} vector storage is empty cannot be established: "
            f"{type(vdb).__name__} cannot answer it ({e}). Its embedding "
            f"baseline stays unrecorded."
        )
        return None
    except Exception as e:
        logger.warning(
            f"Whether the {name} vector storage is empty could not be established "
            f"({type(e).__name__}: {e}). Its embedding baseline stays unrecorded."
        )
        return None


async def _chunk_source_is_populated(
    text_chunks, *, strict: bool = True
) -> bool | None:
    """Whether ``text_chunks`` holds a row: ``True``, ``False``, or ``None``
    when empty and unreadable cannot be told apart.

    ``strict=False`` is ``BaseKVStorage.is_empty()``, the read the coverage
    check has always used. It catches its backend errors and answers ``True``
    (empty), which the check acts on by skipping -- harmless, because the
    caller has a baseline recorded for chunks already and will claim nothing
    from this verdict.

    ``strict=True`` is for the start that must CLAIM that baseline. A durable
    ``origin=empty`` record says the configured model is the one this
    container's vectors are in, so a transient outage read as "empty" would
    stamp that model over vectors nobody probed. The answer then comes from
    the first page of ``iter_rows``, which the base contract requires to RAISE
    on a backend failure: a row means populated, a clean end means empty, a
    raise propagates to the caller as "no evidence". It is the same bounded
    read the chunk probe samples from -- and it costs whatever enumeration
    costs on that backend, which is why it is spent only on the starts that
    need it (``JsonKVStorage`` snapshots its key list before the first page).

    **"Bounded" bounds the rows, not the round trips.** On a backend that
    finds its namespace by scanning a key prefix, proving the namespace EMPTY
    means reaching the end of the keyspace however many batches that takes:
    ``RedisKVStorage.iter_rows`` walks ``SCAN`` to a zero cursor, because
    Redis applies ``MATCH`` after each batch and a batch that matches nothing
    is indistinguishable from the end of the namespace. That is not a cost
    this read introduces -- ``is_empty()``, the read it replaces and the one
    every other start still uses, is ``scan_iter(match=..., count=1)`` over
    the same keyspace, so the empty case has always walked it. What the
    strict read changes is the ANSWER on failure, which is the whole point:
    ``is_empty()`` reports an outage as "empty", and here that would be a
    durable ``origin=empty`` record.

    A backend without enumeration falls back to ``is_empty()``, whose
    "populated" is trustworthy and whose "empty" is not: the first answers
    ``True``, the second ``None``. The coverage check only acts on ``True``,
    so that backend keeps exactly the check it had, and never a baseline it
    did not earn.
    """
    if not strict:
        return not await text_chunks.is_empty()
    try:
        iterator = text_chunks.iter_rows(page_size=1)
    except StorageCapabilityError:
        iterator = None
    if iterator is not None:
        try:
            async for _row in iterator:
                return True
            return False
        except StorageCapabilityError:
            # The base default raises on first iteration, not at the call.
            pass
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()
    if not await text_chunks.is_empty():
        return True
    logger.info(
        f"{type(text_chunks).__name__} cannot enumerate its rows and its "
        f"is_empty() answers True on a failed read too, so whether the text "
        f"chunk storage is empty cannot be established; the chunk baseline "
        f"stays unrecorded until it can."
    )
    return None
