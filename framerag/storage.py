"""LightRAG storage factory utilities for FrameRAG.

FrameRAG inherits LightRAG's production storage backends:
  - JsonKVStorage  : JSON-persisted async key-value store with shared-memory locking
  - NanoVectorDBStorage : NanoVectorDB-backed cosine similarity search

Factory helpers create correctly-configured instances for each namespace.
"""
from __future__ import annotations

from typing import Callable, Awaitable
import numpy as np

from lightrag.utils import EmbeddingFunc
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage


def wrap_embed(
    raw_func: Callable[[list[str]], Awaitable[np.ndarray]],
    embedding_dim: int,
) -> EmbeddingFunc:
    """Wrap a raw async embed callable in LightRAG's EmbeddingFunc."""
    if isinstance(raw_func, EmbeddingFunc):
        return raw_func
    return EmbeddingFunc(embedding_dim=embedding_dim, func=raw_func)


def make_kv(namespace: str, working_dir: str) -> JsonKVStorage:
    """Create a JsonKVStorage for the given namespace under working_dir."""
    return JsonKVStorage(
        namespace=namespace,
        workspace="",
        global_config={"working_dir": working_dir},
        embedding_func=None,
    )


def make_vdb(
    namespace: str,
    working_dir: str,
    embedding_func: EmbeddingFunc,
    meta_fields: set[str],
) -> NanoVectorDBStorage:
    """Create a NanoVectorDBStorage for the given namespace under working_dir."""
    return NanoVectorDBStorage(
        namespace=namespace,
        workspace="",
        global_config={
            "working_dir": working_dir,
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=embedding_func,
        meta_fields=meta_fields,
    )
