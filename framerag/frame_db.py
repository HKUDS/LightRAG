"""Frame Database: stores and retrieves FrameNet-style frame definitions."""
from __future__ import annotations

import os
import numpy as np
from typing import Optional, Callable, Awaitable

from .storage import SimpleKVStore, SimpleVectorStore
from .types import FrameDefinitionSchema, CoreFESchema, NonCoreFESchema


class FrameDatabase:
    """Persistent store for frame definitions with embedding-based lookup.

    Two storage layers:
      - _kv  : SimpleKVStore keyed by frame_name → full definition dict
      - _vdb : SimpleVectorStore keyed by frame_name → embedding for similarity search
    """

    SIMILARITY_THRESHOLD = 0.88    # reuse existing frame if top match >= this
    HINT_TOP_K = 5                 # how many frame hints to pass to LLM

    def __init__(
        self,
        working_dir: str,
        embedding_func: Callable[[list[str]], Awaitable[np.ndarray]],
        embedding_dim: int = 1536,
    ):
        db_dir = os.path.join(working_dir, "frame_db")
        os.makedirs(db_dir, exist_ok=True)
        self._kv = SimpleKVStore(os.path.join(db_dir, "frames.json"))
        self._vdb = SimpleVectorStore(
            os.path.join(db_dir, "frames_vec.json"), embedding_dim
        )
        self._embed = embedding_func

    async def initialize(self) -> None:
        await self._kv.initialize()
        await self._vdb.initialize()

    # ──────────────────────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────────────────────

    async def upsert_frame(self, frame: FrameDefinitionSchema) -> None:
        """Insert or update a frame definition and its embedding."""
        text = f"{frame.frame_name} [SEP] {frame.frame_definition}"
        emb = await self._embed([text])
        vec = emb[0].tolist()
        frame.embedding = vec

        payload = {
            "frame_name": frame.frame_name,
            "lexical_units": frame.lexical_units,
            "frame_definition": frame.frame_definition,
            "core_fes": [
                {"fe_name": fe.fe_name, "fe_definition": fe.fe_definition,
                 "semantic_type": fe.semantic_type}
                for fe in frame.core_fes
            ],
            "noncore_fes": [
                {"fe_name": fe.fe_name, "fe_definition": fe.fe_definition,
                 "semantic_type": fe.semantic_type}
                for fe in frame.noncore_fes
            ],
            "is_from_framenet": frame.is_from_framenet,
            "usage_count": frame.usage_count,
        }
        await self._kv.set(frame.frame_name, payload)
        await self._vdb.upsert(frame.frame_name, vec, metadata=payload)

    async def increment_usage(self, frame_name: str) -> None:
        data = await self._kv.get(frame_name)
        if data:
            data["usage_count"] = data.get("usage_count", 0) + 1
            await self._kv.set(frame_name, data)
            meta = await self._vdb.get_by_id(frame_name)
            if meta:
                meta["usage_count"] = data["usage_count"]
                emb = await self._vdb.get_embedding(frame_name)
                if emb is not None:
                    await self._vdb.upsert(frame_name, emb.tolist(), metadata=meta)

    # ──────────────────────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────────────────────

    async def get(self, frame_name: str) -> Optional[FrameDefinitionSchema]:
        data = await self._kv.get(frame_name)
        return self._dict_to_schema(data) if data else None

    async def exists(self, frame_name: str) -> bool:
        return (await self._kv.get(frame_name)) is not None

    async def get_hints_for_chunk(self, chunk_text: str) -> list[dict]:
        """Embed chunk_text and return top-K similar frames as hint dicts."""
        emb = await self._embed([chunk_text])
        results = await self._vdb.search(emb[0], top_k=self.HINT_TOP_K)
        return [
            {
                "frame_name": r["frame_name"],
                "lexical_units": r["lexical_units"],
                "frame_definition": r["frame_definition"],
                "core_fes": [fe["fe_name"] for fe in r["core_fes"]],
                "score": round(r["score"], 3),
            }
            for r in results
        ]

    async def find_similar(
        self,
        trigger_lemma: str,
        event_description: str,
        threshold: Optional[float] = None,
    ) -> Optional[FrameDefinitionSchema]:
        """Return existing frame if similarity >= threshold, else None."""
        thr = threshold or self.SIMILARITY_THRESHOLD
        query = f"{trigger_lemma}: {event_description}"
        emb = await self._embed([query])
        results = await self._vdb.search(emb[0], top_k=1, min_score=thr)
        if results:
            return await self.get(results[0]["frame_name"])
        return None

    async def all_frames(self) -> list[FrameDefinitionSchema]:
        items = await self._kv.all_values()
        return [self._dict_to_schema(d) for d in items]

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    async def save(self) -> None:
        await self._kv.save()
        await self._vdb.save()

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dict_to_schema(data: dict) -> FrameDefinitionSchema:
        return FrameDefinitionSchema(
            frame_name=data["frame_name"],
            lexical_units=data.get("lexical_units", []),
            frame_definition=data["frame_definition"],
            core_fes=[
                CoreFESchema(**fe) for fe in data.get("core_fes", [])
            ],
            noncore_fes=[
                NonCoreFESchema(**fe) for fe in data.get("noncore_fes", [])
            ],
            is_from_framenet=data.get("is_from_framenet", False),
            usage_count=data.get("usage_count", 0),
        )

    def format_hints_for_prompt(self, hints: list[dict]) -> str:
        if not hints:
            return "(no existing frames yet - create new frames as needed)"
        lines = []
        for h in hints:
            lus = ", ".join(h["lexical_units"])
            fes = ", ".join(h["core_fes"])
            lines.append(
                f'  - {h["frame_name"]} (LUs: {lus}) | Core FEs: {fes}\n'
                f'    Definition: {h["frame_definition"]}'
            )
        return "\n".join(lines)
