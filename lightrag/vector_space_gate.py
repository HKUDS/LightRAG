"""The two startup checks that no single storage can make on its own.

Both ask a question that spans the graph and a vector storage, which is why
they live one layer up in ``LightRAG.initialize_storages()`` rather than inside
a backend's ``initialize()``. ``BaseVectorStorage`` has no enumeration API --
only ``get_by_ids(ids)`` -- so a backend cannot obtain a sample of its own rows
without a pile of new methods; up here the graph supplies the ids and the
existing readers supply the content and the vector.

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

An embedding provider that is down, a timeout, an empty graph, a sample absent
from the vector store, a marker write denied by permissions -- all of these
leave the container unmarked and the instance serving, which is exactly the
detection capability LightRAG had before this feature existed.

Read ``docs/design/VectorSpaceProvenance.md`` before changing the verdicts.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from lightrag.base import DocStatus
from lightrag.exceptions import VectorSpaceMismatchError, VectorStorageEmptyError
from lightrag.kg.vector_space import declared_model_name
from lightrag.utils import compute_mdhash_id, logger

# How many graph entities to look up. The gate refuses only when NONE of them
# has a vector, so this is the confidence knob: a store that still holds half
# its rows survives 32 lookups with probability 2**-32, while a store that has
# lost 99% of them is caught about half the time -- and being caught is the
# right answer there too. Kept small because it is one round trip on every
# startup.
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


async def _has_processed_documents(doc_status) -> bool:
    """Whether the pipeline has ever finished a document in this workspace.

    A PROCESSED document is the pipeline's own claim that it wrote everything
    that document produces, vectors included. That claim is what makes a
    missing vector a defect rather than work still in flight, so it is the
    difference between the gate and a false alarm.

    Anything that goes wrong here answers ``False``: a doc-status backend that
    cannot be read is not evidence that vectors are missing, and this is the
    last question asked before a refusal.
    """
    if doc_status is None:
        return False
    try:
        counts = await doc_status.get_status_counts()
    except Exception as e:
        logger.warning(
            f"Not refusing an empty vector storage: the document status could "
            f"not be read ({type(e).__name__}: {e})"
        )
        return False
    if not isinstance(counts, dict):
        return False
    return int(counts.get(DocStatus.PROCESSED.value, 0) or 0) > 0


async def check_vector_space_at_startup(
    *,
    graph,
    entities_vdb,
    doc_status=None,
    embedding_func,
    adoptable=(),
    expect_empty_vector_storage: bool = False,
) -> None:
    """Run the empty-container gate, then the adoption probe if one is needed.

    Both read the SAME sample, because both questions are about the same
    evidence: which of the graph's entities have vectors, and do those vectors
    come from this model. The gate answers first -- an empty container has
    nothing to probe.

    Args:
        graph: the graph storage, already initialized.
        entities_vdb: the entity vector storage, already initialized. It is the
            one sampled because the graph's own ids address it directly.
        doc_status: the doc-status storage. Consulted only when the sample comes
            back empty, so the healthy path pays nothing for it.
        embedding_func: this instance's embedding function.
        adoptable: the vector storages that may be adopted on a positive
            verdict. They share one ``embedding_func`` and were written by the
            same deployment, so the evidence gathered from ``entities_vdb``
            settles the question for all of them.
        expect_empty_vector_storage: this caller is about to repopulate the
            vector storages, so an empty one is the expected starting state
            rather than a defect. See ``LightRAG.rebuilding_vector_storage``.

    Raises:
        VectorStorageEmptyError: the graph holds entities, at least one document
            has been PROCESSED, not one of the sampled entities has a vector,
            and the caller did not declare that it is about to rebuild them.
        VectorSpaceMismatchError: the probe ran and the stored vectors did not
            come from this embedding model.

    Nothing else escapes. Every other failure -- an unreachable graph backend,
    a vector store that errors on a read, a marker that cannot be written -- is
    logged and swallowed, because none of them is evidence about the embedding
    space and none of them was a startup failure before this check existed.
    """
    pending = []
    for vdb in adoptable:
        try:
            if await vdb.vector_space_adoption_pending():
                pending.append(vdb)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                f"Could not ask {type(vdb).__name__} whether it needs "
                f"embedding-space adoption: {e}"
            )

    try:
        sample_ids = await _sample_entity_ids(graph, SAMPLE_SIZE)
    except Exception as e:
        logger.warning(
            f"Skipping the embedding-space startup check: the graph storage "
            f"could not supply a sample ({type(e).__name__}: {e})"
        )
        return

    if not sample_ids:
        # An empty graph is not evidence of anything: there is nothing that
        # SHOULD have a vector, so neither check has a question to ask.
        return

    try:
        found = await entities_vdb.get_by_ids(sample_ids)
    except Exception as e:
        logger.warning(
            f"Skipping the embedding-space startup check: the entity vector "
            f"storage could not be read ({type(e).__name__}: {e})"
        )
        return

    # The backends disagree on the shape of a MISS. ``BaseVectorStorage``
    # documents "the objects that were found", and most return a compacted
    # list; Nano returns one entry per requested id, positionally, with None
    # where a row is absent. Reading that as "the container has rows" is the
    # difference between catching a vanished vector store and starting on top
    # of one, so the rows are counted, not the list.
    rows = [row for row in (found or []) if isinstance(row, dict)]

    if not rows:
        if expect_empty_vector_storage:
            # The caller owns the repopulation that follows -- an in-process
            # rebuild, or the supported switch from NoopVectorDBStorage
            # (graph-only ingestion) to a real vector backend. Refusing here
            # would block the only thing that clears the condition.
            logger.info(
                f"The entity vector storage holds no vectors for any of "
                f"{len(sample_ids)} entities sampled from the knowledge graph. "
                f"Serving anyway: this instance declared it is rebuilding them."
            )
            return
        if await _has_processed_documents(doc_status):
            raise VectorStorageEmptyError(
                vdb_name="entities",
                container=getattr(entities_vdb, "final_namespace", None),
                sampled=len(sample_ids),
            )
        # A graph ahead of the vector store, with no document the pipeline ever
        # called PROCESSED, is a residue that HEALS: an ingest interrupted
        # before its vector flush, a batch that failed and will be retried, a
        # graph built through the admin API. ``AGENTS.md`` *Consistency without
        # transactions* is explicit that such a state must not be escalated --
        # and escalating it here would refuse to start the very process whose
        # next run repairs it.
        logger.warning(
            f"The entity vector storage holds no vectors for any of "
            f"{len(sample_ids)} entities sampled from the knowledge graph, but "
            f"no document has been processed yet. Serving anyway: an ingest "
            f"that has not finished writing its vectors is repaired by the "
            f"next pipeline run, not by a refusal."
        )
        return

    if not pending:
        return

    if declared_model_name(embedding_func) is None:
        # Nothing to record, so nothing to prove. Reached only if a backend
        # reported pending without checking; the storages answer False here.
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

    for vdb in pending:
        try:
            if await vdb.adopt_vector_space():
                logger.info(
                    f"Recorded the embedding model on {type(vdb).__name__} "
                    f"'{getattr(vdb, 'namespace', '?')}' ({detail})"
                )
        except VectorSpaceMismatchError:
            raise
        except Exception as e:  # pragma: no cover - adopt must not raise
            logger.warning(
                f"Could not record the embedding model on {type(vdb).__name__}: {e}"
            )
