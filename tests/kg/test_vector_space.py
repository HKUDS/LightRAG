"""Verdict rules for the shared embedding-space provenance helpers.

Pins the three rules that ``docs/design/VectorSpaceProvenance.md`` states and
that seven backends depend on being identical: what "this instance's model"
means, what gets recorded, and — the one with teeth — that absent evidence
never refuses.
"""

import pytest

from lightrag.exceptions import VectorSpaceMismatchError
from lightrag.kg.vector_space import (
    VECTOR_SPACE_DIM_KEY,
    VECTOR_SPACE_MODEL_KEY,
    assert_vector_space_matches,
    declared_model_name,
    read_vector_space_marker,
    vector_space_marker,
)

pytestmark = pytest.mark.offline


class _Embedder:
    def __init__(self, model_name=None, embedding_dim=1024):
        self.model_name = model_name
        self.embedding_dim = embedding_dim


def _assert(stored_model, stored_dim, embedder):
    assert_vector_space_matches(
        backend="FakeVectorStorage",
        container="fake_index",
        embedding_func=embedder,
        stored_model=stored_model,
        stored_dim=stored_dim,
    )


# ---------------------------------------------------------------------------
# declared_model_name / vector_space_marker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "configured, expected",
    [
        ("bge-m3", "bge-m3"),
        ("  bge-m3  ", "bge-m3"),
        ("", None),
        ("   ", None),
        (None, None),
        (123, None),
    ],
)
def test_declared_model_name_normalizes_one_way_for_every_backend(configured, expected):
    assert declared_model_name(_Embedder(model_name=configured)) == expected


def test_marker_records_the_unfolded_model_name():
    # The collection-name suffix folds this to text_embedding_3_large_3072d;
    # recording the folded form would lose the distinction from
    # "text_embedding_3_large", which is the blind spot the marker exists to
    # close on the suffixed backends.
    marker = vector_space_marker(
        _Embedder(model_name="text-embedding-3-large", embedding_dim=3072)
    )
    assert marker == {
        VECTOR_SPACE_MODEL_KEY: "text-embedding-3-large",
        VECTOR_SPACE_DIM_KEY: 3072,
    }


def test_marker_omits_an_unknown_model_rather_than_recording_none():
    # A recorded None would be indistinguishable from "written before the
    # marker existed", making the never-refuse rule permanent for it.
    marker = vector_space_marker(_Embedder(model_name=None, embedding_dim=768))
    assert marker == {VECTOR_SPACE_DIM_KEY: 768}


# ---------------------------------------------------------------------------
# read_vector_space_marker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({VECTOR_SPACE_MODEL_KEY: "m", VECTOR_SPACE_DIM_KEY: 8}, ("m", 8)),
        ({VECTOR_SPACE_MODEL_KEY: " m ", VECTOR_SPACE_DIM_KEY: "8"}, ("m", 8)),
        ({VECTOR_SPACE_MODEL_KEY: "", VECTOR_SPACE_DIM_KEY: "eight"}, (None, None)),
        ({VECTOR_SPACE_DIM_KEY: True}, (None, None)),
        ({}, (None, None)),
        (None, (None, None)),
        ("not a mapping", (None, None)),
    ],
)
def test_unreadable_marker_fields_read_as_absent(payload, expected):
    assert read_vector_space_marker(payload) == expected


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def test_matching_space_does_not_refuse():
    _assert("bge-m3", 1024, _Embedder("bge-m3", 1024))


def test_dimension_change_refuses():
    with pytest.raises(VectorSpaceMismatchError) as excinfo:
        _assert("bge-m3", 1024, _Embedder("bge-m3", 768))
    assert excinfo.value.stored_dim == 1024
    assert excinfo.value.expected_dim == 768
    assert "1024 -> 768" in str(excinfo.value)


def test_same_dimension_model_change_refuses():
    # The defect this whole marker exists for: two models, one dimension.
    with pytest.raises(VectorSpaceMismatchError) as excinfo:
        _assert("bge-m3", 1024, _Embedder("other-model", 1024))
    message = str(excinfo.value)
    assert "'bge-m3' -> 'other-model'" in message
    assert "lightrag-rebuild-vdb" in message


def test_case_and_punctuation_differences_are_a_mismatch():
    # These two fold to the same collection-name suffix, so the suffix cannot
    # separate them; the unfolded marker can.
    with pytest.raises(VectorSpaceMismatchError):
        _assert(
            "text-embedding-3-large", 3072, _Embedder("text_embedding_3_large", 3072)
        )


def test_absent_stored_model_never_refuses():
    # An index written before the marker existed must still be servable.
    _assert(None, 1024, _Embedder("bge-m3", 1024))


def test_absent_stored_dimension_never_refuses():
    _assert("bge-m3", None, _Embedder("bge-m3", 1024))


def test_absent_declared_model_never_refuses():
    # Symmetric: a process that cannot say what model it runs cannot
    # contradict a recorded one.
    _assert("bge-m3", 1024, _Embedder(None, 1024))


def test_a_container_recording_nothing_never_refuses():
    _assert(None, None, _Embedder("bge-m3", 1024))


def test_dimension_mismatch_refuses_even_when_the_model_matches_and_vice_versa():
    with pytest.raises(VectorSpaceMismatchError):
        _assert("bge-m3", 1024, _Embedder(None, 768))
    with pytest.raises(VectorSpaceMismatchError):
        _assert("bge-m3", None, _Embedder("other", 768))
