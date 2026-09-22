"""Collection suffixes must isolate raw embedding-model names.

Folding to ``[a-z0-9_]`` maps ``vendor/model:v1`` and ``vendor_model/v1`` onto
one token. With a shared dimension those models used to share a Milvus
collection, a Qdrant collection, and a PostgreSQL table. The digested suffix
keeps a readable folded prefix and appends the first 8 hex characters of
SHA-256 over the stripped model name.

Pre-digest containers keep the old name. Adopting one is allowed only when
it is unowned or already owned by this model; a different owner is left
untouched so the upgrade does not hand one model's vectors to another.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from lightrag.base import (
    BaseVectorStorage,
    _MAX_COLLECTION_SUFFIX_LENGTH,
    _legacy_folded_suffix,
    _resolve_pre_digest_container_adoption,
)

pytestmark = pytest.mark.offline


def _digest(model_name: str) -> str:
    return hashlib.sha256(model_name.strip().encode("utf-8")).hexdigest()[:8]


def _suffix(model_name, dim=1024):
    embedding = SimpleNamespace(model_name=model_name, embedding_dim=dim)
    return BaseVectorStorage._generate_collection_suffix(
        SimpleNamespace(embedding_func=embedding)
    )


def _legacy(model_name, dim=1024):
    embedding = SimpleNamespace(model_name=model_name, embedding_dim=dim)
    return BaseVectorStorage._legacy_folded_collection_suffix(
        SimpleNamespace(embedding_func=embedding)
    )


def test_fold_equivalent_names_get_distinct_suffixes():
    names = ["vendor/model:v1", "vendor_model/v1", "Vendor/Model:V1"]
    suffixes = [_suffix(name) for name in names]

    assert len(set(suffixes)) == 3
    for name, suffix in zip(names, suffixes):
        assert suffix.startswith("vendor_model_v1_1024d_")
        assert suffix.endswith(_digest(name))
        assert suffix == f"vendor_model_v1_1024d_{_digest(name)}"


def test_case_and_punctuation_stay_distinct_when_the_folded_prefix_matches():
    slash = _suffix("BAAI/bge-m3", 1024)
    underscore = _suffix("baai_bge-m3", 1024)

    assert slash.startswith("baai_bge_m3_1024d_")
    assert underscore.startswith("baai_bge_m3_1024d_")
    assert slash != underscore


def test_surrounding_whitespace_does_not_create_a_second_model():
    assert _suffix("  vendor/model:v1  ") == _suffix("vendor/model:v1")
    assert _legacy("  vendor/model:v1\n") == "vendor_model_v1_1024d"


def test_internal_differences_survive_folding():
    assert _suffix("foo bar") != _suffix("foo_bar")
    assert _suffix("foo bar").startswith("foo_bar_1024d_")
    assert _suffix("foo_bar").startswith("foo_bar_1024d_")


def test_dimension_stays_in_the_suffix_and_the_digest_is_of_the_name_only():
    short = _suffix("nomic-embed-text:v1.5", 768)
    wide = _suffix("nomic-embed-text:v1.5", 1024)
    digest = _digest("nomic-embed-text:v1.5")

    assert short == f"nomic_embed_text_v1_5_768d_{digest}"
    assert wide == f"nomic_embed_text_v1_5_1024d_{digest}"
    assert short != wide


@pytest.mark.parametrize("model_name", ["", "   ", None, 123])
def test_missing_model_name_has_no_suffix(model_name):
    assert _suffix(model_name) is None
    assert _legacy(model_name) is None


def test_legacy_suffix_is_the_pre_digest_folded_name():
    assert _legacy("text-embedding-3-large", 3072) == (
        "text_embedding_3_large_3072d"
    )
    digested = _suffix("text-embedding-3-large", 3072)
    assert digested == "text_embedding_3_large_3072d_" + _digest(
        "text-embedding-3-large"
    )


def test_legacy_suffix_is_not_truncated_but_the_digested_suffix_fits_postgres():
    model = ("A" * 120) + "/tail-a"
    other = ("A" * 120) + "/tail-b"
    legacy = _legacy_folded_suffix(model, 1024)
    digested = _suffix(model, 1024)
    other_digested = _suffix(other, 1024)

    assert legacy == ("a" * 120) + "_tail_a_1024d"
    assert len(legacy) > _MAX_COLLECTION_SUFFIX_LENGTH
    assert len(digested) <= _MAX_COLLECTION_SUFFIX_LENGTH
    assert len(other_digested) <= _MAX_COLLECTION_SUFFIX_LENGTH
    assert digested != other_digested
    assert digested.endswith(_digest(model))
    assert "1024d" in digested
    # Worst vector base table plus the joining underscore.
    assert len("LIGHTRAG_VDB_RELATION_" + digested) <= 63


@pytest.mark.parametrize(
    ("digested_exists", "legacy_exists", "owner", "expected"),
    [
        (True, True, None, False),
        (True, False, None, False),
        (False, False, None, False),
        (False, True, None, True),
        (False, True, "vendor/model:v1", True),
        (False, True, "vendor_model/v1", False),
    ],
)
def test_pre_digest_container_is_adopted_only_when_unowned_or_ours(
    digested_exists, legacy_exists, owner, expected
):
    assert (
        _resolve_pre_digest_container_adoption(
            digested_exists=digested_exists,
            legacy_exists=legacy_exists,
            legacy_owner=owner,
            model_name="vendor/model:v1",
        )
        is expected
    )
