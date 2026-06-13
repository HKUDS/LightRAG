"""Simple self-contained storage backends for FrameRAG.

SimpleKVStore  : JSON-backed key-value store.
SimpleVectorStore: numpy-backed vector store with cosine search, persisted as JSON.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np


class SimpleKVStore:
    """Async-friendly JSON-backed key-value store."""

    def __init__(self, file_path: str):
        self._path = file_path
        self._data: dict[str, Any] = {}

    async def initialize(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    async def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    async def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def all_values(self) -> list[Any]:
        return list(self._data.values())

    async def save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self._data)


class SimpleVectorStore:
    """numpy-backed cosine similarity vector store, persisted as JSON.

    Each entry has:
      id        : str
      embedding : list[float]
      metadata  : dict  (arbitrary payload for retrieval)
    """

    def __init__(self, file_path: str, embedding_dim: int):
        self._path = file_path
        self._dim = embedding_dim
        self._ids: list[str] = []
        self._embeddings: Optional[np.ndarray] = None   # shape [n, dim]
        self._metadata: dict[str, dict] = {}

    async def initialize(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._ids = data.get("ids", [])
            embs = data.get("embeddings", [])
            self._embeddings = np.array(embs, dtype=np.float32) if embs else None
            self._metadata = data.get("metadata", {})

    async def upsert(self, id: str, embedding: list[float], metadata: dict) -> None:
        vec = np.array(embedding, dtype=np.float32)
        if id in self._ids:
            idx = self._ids.index(id)
            self._embeddings[idx] = vec
            self._metadata[id] = metadata
        else:
            self._ids.append(id)
            if self._embeddings is None:
                self._embeddings = vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, vec.reshape(1, -1)])
            self._metadata[id] = metadata

    async def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Return top-k results sorted by cosine similarity descending."""
        if self._embeddings is None or len(self._ids) == 0:
            return []

        q = query_vec.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        normed = self._embeddings / norms
        scores = normed @ q_norm                   # [n]

        k = min(top_k, len(self._ids))
        idxs = np.argpartition(scores, -k)[-k:]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]

        results = []
        for i in idxs:
            score = float(scores[i])
            if score >= min_score:
                results.append({"id": self._ids[i], "score": score, **self._metadata[self._ids[i]]})
        return results

    async def get_by_id(self, id: str) -> Optional[dict]:
        if id not in self._ids:
            return None
        return {"id": id, **self._metadata[id]}

    async def get_all_ids(self) -> list[str]:
        return list(self._ids)

    async def get_embedding(self, id: str) -> Optional[np.ndarray]:
        if id not in self._ids:
            return None
        idx = self._ids.index(id)
        return self._embeddings[idx]

    async def get_all_embeddings(self) -> tuple[list[str], Optional[np.ndarray]]:
        return self._ids, self._embeddings

    async def save(self) -> None:
        data = {
            "ids": self._ids,
            "embeddings": self._embeddings.tolist() if self._embeddings is not None else [],
            "metadata": self._metadata,
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self._ids)
