"""Embedding-space provenance shared by every vector storage backend.

A vector container (index / collection / table / file) is only usable by the
process that wrote it if both sides agree on the embedding space: the
dimension AND the model. A dimension is not an identity -- two models can both
emit 1024 floats and have unrelated vector spaces -- so the model name has to
be *recorded* next to the vectors, never inferred from the dimension.

Only the backends whose container NAME carries no model information need this:
Nano, FAISS, MongoDB and OpenSearch. Milvus, Qdrant and PostgreSQL encode
``{folded_model}_{dim}d`` in the collection or table name, so a model change
already lands in a different container and a second copy of that fact would
only drift.

This module owns the three pieces those four need and nothing else: what "this
instance's model" means, what gets written into the container's marker, and the
comparison that decides whether to refuse. Where the marker lives is each
backend's choice among its own METADATA (index ``_meta``, a collection
validator, a file's own header) -- never a vector record, which would enter the
ANN index and be returned by a search. The *contents* and the *verdict* are
here so the four cannot drift apart on them.

Read ``docs/design/VectorSpaceProvenance.md`` before changing the verdict
rules -- in particular before making absent evidence refuse.
"""

from __future__ import annotations

from typing import Any

from lightrag.exceptions import VectorSpaceMismatchError

# Marker keys. Identical across backends so an operator reading a raw index
# mapping, a Mongo document or a table comment sees the same two names.
VECTOR_SPACE_MODEL_KEY = "lightrag_embedding_model"
VECTOR_SPACE_DIM_KEY = "lightrag_embedding_dim"


def declared_model_name(embedding_func: Any) -> str | None:
    """The embedding model name this instance is configured with, or ``None``.

    The single place that decides what "this instance's model" means, so the
    value compared against a marker is the same value that was written into
    one. ``None`` means "this process cannot say" -- an ``embedding_func``
    with no ``model_name``, an empty or whitespace-only name, or a non-string.

    The name is returned **unfolded**: exactly as configured, not lowercased
    and not character-folded the way ``_generate_collection_suffix`` folds it
    for a collection name. That folding is what makes the suffix unable to
    tell ``text-embedding-3-large`` from ``text_embedding_3_large``; recording
    the folded form here would import the same blind spot.
    """
    model_name = getattr(embedding_func, "model_name", None)
    if not isinstance(model_name, str):
        return None
    model_name = model_name.strip()
    return model_name or None


def declared_dimension(embedding_func: Any) -> int | None:
    """The embedding dimension this instance is configured with, or ``None``."""
    dim = getattr(embedding_func, "embedding_dim", None)
    if isinstance(dim, bool) or not isinstance(dim, int):
        return None
    return dim


def vector_space_marker(embedding_func: Any) -> dict[str, Any]:
    """The provenance payload to record when provisioning a container.

    A key is written only when this process can actually say what it holds:
    an unknown model contributes no key rather than a ``None``, because a
    recorded ``None`` is indistinguishable from "an older LightRAG wrote this"
    and would make the never-refuse rule below permanent for that container.
    """
    marker: dict[str, Any] = {}
    model = declared_model_name(embedding_func)
    if model is not None:
        marker[VECTOR_SPACE_MODEL_KEY] = model
    dim = declared_dimension(embedding_func)
    if dim is not None:
        marker[VECTOR_SPACE_DIM_KEY] = dim
    return marker


def read_vector_space_marker(payload: Any) -> tuple[str | None, int | None]:
    """Extract ``(model, dim)`` from a stored marker payload.

    Tolerant on purpose: a payload this code cannot parse, a missing key, a
    blank string or a non-integer dimension all read as ``None`` -- "not
    recorded". A marker we cannot read is not evidence of a mismatch, and
    turning an unparseable payload into a refusal would wedge every container
    written by a version that stored it differently.
    """
    if not isinstance(payload, dict):
        return None, None

    model = payload.get(VECTOR_SPACE_MODEL_KEY)
    if isinstance(model, str):
        model = model.strip() or None
    else:
        model = None

    dim = payload.get(VECTOR_SPACE_DIM_KEY)
    if isinstance(dim, bool):
        dim = None
    elif isinstance(dim, int):
        pass
    elif isinstance(dim, str):
        try:
            dim = int(dim.strip())
        except (TypeError, ValueError):
            dim = None
    else:
        dim = None

    return model, dim


def assert_vector_space_matches(
    *,
    backend: str,
    container: str,
    embedding_func: Any,
    stored_model: str | None,
    stored_dim: int | None,
    detail: str | None = None,
) -> None:
    """Refuse to attach to a container written in a different embedding space.

    Two independent comparisons, either of which refuses:

    * **Dimension.** A recorded dimension that differs from this instance's.
    * **Model.** A recorded model name that differs from this instance's.
      Compared verbatim -- the recorded name is unfolded (see
      ``declared_model_name``), so this is the check the collection-name
      suffix cannot make.

    **Absent evidence never refuses.** A container that records no model, or
    no dimension, predates this marker; reading silence as a mismatch would
    refuse every index written before the provenance existed. The rule is
    symmetric: a process whose ``embedding_func`` declares no ``model_name``
    cannot contradict a recorded one either, because it does not know what it
    is. Both silences are the reason each backend needs a documented one-time
    decision about what to do with pre-existing unmarked containers -- and why
    backfilling a marker matters: without one, the silence never ends.

    Raises:
        VectorSpaceMismatchError: on either mismatch. Nothing is mutated.
    """
    expected_model = declared_model_name(embedding_func)
    expected_dim = declared_dimension(embedding_func)

    dim_conflict = (
        stored_dim is not None
        and expected_dim is not None
        and stored_dim != expected_dim
    )
    model_conflict = (
        stored_model is not None
        and expected_model is not None
        and stored_model != expected_model
    )
    if not dim_conflict and not model_conflict:
        return

    raise VectorSpaceMismatchError(
        backend=backend,
        container=container,
        expected_model=expected_model,
        expected_dim=expected_dim,
        stored_model=stored_model,
        stored_dim=stored_dim,
        detail=detail,
    )
