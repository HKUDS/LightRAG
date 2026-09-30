"""Startup deprecation warning for storage backends scheduled for removal.

Redis (``RedisKVStorage`` / ``RedisDocStatusStorage``) is being retired. It
keeps working; constructing a ``LightRAG`` that selects it only logs a warning
naming the PostgreSQL replacement.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag.kg import (
    DEPRECATED_STORAGES,
    STORAGE_IMPLEMENTATIONS,
    deprecated_storage_message,
)
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


async def _embed(texts, **kwargs):
    return np.zeros((len(texts), 8), dtype=np.float32)


_EMBEDDING = EmbeddingFunc(embedding_dim=8, max_token_size=512, func=_embed)


def test_redis_backends_are_deprecated_in_favor_of_postgres():
    kv = deprecated_storage_message("RedisKVStorage")
    doc_status = deprecated_storage_message("RedisDocStatusStorage")

    assert kv is not None and "PGKVStorage" in kv
    assert doc_status is not None and "PGDocStatusStorage" in doc_status


@pytest.mark.parametrize("name", ["PGKVStorage", "JsonKVStorage", "Unknown"])
def test_other_backends_are_not_deprecated(name):
    assert deprecated_storage_message(name) is None


def test_every_deprecated_backend_is_still_registered():
    """A misspelled key would make the warning silently never fire."""
    registered = {
        name
        for info in STORAGE_IMPLEMENTATIONS.values()
        for name in info["implementations"]
    }
    for name, replacement in DEPRECATED_STORAGES.items():
        assert name in registered
        assert replacement in registered


def _deprecation_records(caplog, monkeypatch, tmp_path, **storages) -> list[str]:
    from lightrag import LightRAG

    monkeypatch.setenv("REDIS_URI", "redis://localhost:6379")
    logger = logging.getLogger("lightrag")
    previous_propagate = logger.propagate
    logger.propagate = True  # lightrag's logger does not propagate by default
    try:
        with caplog.at_level(logging.WARNING, logger="lightrag"):
            # __post_init__ only builds the storage objects; nothing connects
            # to Redis until initialize_storages().
            LightRAG(
                working_dir=str(tmp_path),
                llm_model_func=AsyncMock(return_value=""),
                embedding_func=_EMBEDDING,
                **storages,
            )
    finally:
        logger.propagate = previous_propagate
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "deprecated" in record.getMessage()
    ]


def test_constructing_with_redis_logs_one_warning_per_backend(
    caplog, monkeypatch, tmp_path
):
    messages = _deprecation_records(
        caplog,
        monkeypatch,
        tmp_path,
        kv_storage="RedisKVStorage",
        doc_status_storage="RedisDocStatusStorage",
    )

    assert len(messages) == 2
    assert any("RedisKVStorage" in m and "PGKVStorage" in m for m in messages)
    assert any(
        "RedisDocStatusStorage" in m and "PGDocStatusStorage" in m for m in messages
    )


def test_default_storages_log_no_deprecation(caplog, monkeypatch, tmp_path):
    assert _deprecation_records(caplog, monkeypatch, tmp_path) == []
