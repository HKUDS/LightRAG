import pytest

from lightrag.api.config import (
    resolve_postgres_enable_vector_setting,
    validate_postgres_enable_vector_setting,
)


def test_auto_derives_vector_enabled_for_pgvector_storage() -> None:
    env: dict[str, str] = {}

    enabled, source = resolve_postgres_enable_vector_setting(
        "PGVectorStorage", environ=env
    )

    assert enabled is True
    assert source == "auto-derived"
    assert env["POSTGRES_ENABLE_VECTOR"] == "true"


def test_auto_derives_vector_disabled_for_non_pgvector_storage() -> None:
    env: dict[str, str] = {}

    enabled, source = resolve_postgres_enable_vector_setting(
        "NanoVectorDBStorage", environ=env
    )

    assert enabled is False
    assert source == "auto-derived"
    assert env["POSTGRES_ENABLE_VECTOR"] == "false"


def test_explicit_env_true_overrides_vector_storage_choice() -> None:
    env = {"POSTGRES_ENABLE_VECTOR": "true"}

    enabled, source = resolve_postgres_enable_vector_setting(
        "NanoVectorDBStorage", environ=env
    )

    assert enabled is True
    assert source == "explicit"
    assert env["POSTGRES_ENABLE_VECTOR"] == "true"


def test_explicit_env_false_overrides_pgvector_storage_choice() -> None:
    env = {"POSTGRES_ENABLE_VECTOR": "false"}

    enabled, source = resolve_postgres_enable_vector_setting(
        "PGVectorStorage", environ=env
    )

    assert enabled is False
    assert source == "explicit"
    assert env["POSTGRES_ENABLE_VECTOR"] == "false"


def test_explicit_true_conflicting_with_non_pgvector_storage_fails_fast() -> None:
    with pytest.raises(
        ValueError,
        match="conflicts with LIGHTRAG_VECTOR_STORAGE.*Remove or comment out POSTGRES_ENABLE_VECTOR",
    ):
        validate_postgres_enable_vector_setting("NanoVectorDBStorage", True, "explicit")


def test_explicit_false_conflicting_with_pgvector_storage_fails_fast() -> None:
    with pytest.raises(
        ValueError,
        match="conflicts with LIGHTRAG_VECTOR_STORAGE.*Remove or comment out POSTGRES_ENABLE_VECTOR",
    ):
        validate_postgres_enable_vector_setting("PGVectorStorage", False, "explicit")
