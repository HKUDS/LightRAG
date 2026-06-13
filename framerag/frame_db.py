"""Frame Database: stores and retrieves FrameNet-style frame definitions.

Backed by LightRAG's JsonKVStorage (definitions) and NanoVectorDBStorage
(embedding-based similarity lookup). A local frame-name set is persisted
in frame_names.json for all_frames() support.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from lightrag.utils import EmbeddingFunc

from .storage import make_kv, make_vdb
from .types import FrameDefinitionSchema, CoreFESchema, NonCoreFESchema


class FrameDatabase:
    """Persistent store for frame definitions with embedding-based lookup."""

    SIMILARITY_THRESHOLD = 0.88
    HINT_TOP_K = 5

    def __init__(self, working_dir: str, embedding_func: EmbeddingFunc):
        db_dir = os.path.join(working_dir, "frame_db")
        os.makedirs(db_dir, exist_ok=True)
        self._dir = db_dir

        self._kv  = make_kv("frame_defs", db_dir)
        self._vdb = make_vdb(
            "frame_vec", db_dir, embedding_func,
            {"frame_name", "frame_definition"},
        )
        self._frame_names: set[str] = set()

    async def initialize(self) -> None:
        await self._kv.initialize()
        await self._vdb.initialize()
        self._load_names()

    async def index_done_callback(self) -> None:
        await self._kv.index_done_callback()
        await self._vdb.index_done_callback()
        self._save_names()

    def _names_path(self) -> str:
        return os.path.join(self._dir, "frame_names.json")

    def _load_names(self) -> None:
        p = self._names_path()
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                self._frame_names = set(json.load(f))

    def _save_names(self) -> None:
        with open(self._names_path(), "w", encoding="utf-8") as f:
            json.dump(list(self._frame_names), f, ensure_ascii=False)

    # ──────────────────────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────────────────────

    async def upsert_frame(self, frame: FrameDefinitionSchema) -> None:
        payload = {
            "frame_name":       frame.frame_name,
            "lexical_units":    frame.lexical_units,
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
            "usage_count":      frame.usage_count,
        }
        await self._kv.upsert({frame.frame_name: payload})
        content = f"{frame.frame_name} [SEP] {frame.frame_definition}"
        await self._vdb.upsert({
            frame.frame_name: {
                "content":          content,
                "frame_name":       frame.frame_name,
                "frame_definition": frame.frame_definition,
            }
        })
        self._frame_names.add(frame.frame_name)

    async def increment_usage(self, frame_name: str) -> None:
        data = await self._kv.get_by_id(frame_name)
        if data:
            data["usage_count"] = data.get("usage_count", 0) + 1
            await self._kv.upsert({frame_name: data})

    # ──────────────────────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────────────────────

    async def get(self, frame_name: str) -> Optional[FrameDefinitionSchema]:
        data = await self._kv.get_by_id(frame_name)
        return self._dict_to_schema(data) if data else None

    async def exists(self, frame_name: str) -> bool:
        return frame_name in self._frame_names

    async def get_hints_for_chunk(self, chunk_text: str) -> list[dict]:
        """Return top-K similar frames as hint dicts (embedding from content)."""
        results = await self._vdb.query(chunk_text, top_k=self.HINT_TOP_K)
        hints = []
        for r in results:
            fname = r.get("frame_name", "")
            if not fname:
                continue
            data = await self._kv.get_by_id(fname)
            if not data:
                continue
            hints.append({
                "frame_name":       data["frame_name"],
                "lexical_units":    data.get("lexical_units", []),
                "frame_definition": data.get("frame_definition", ""),
                "core_fes":         [fe["fe_name"] for fe in data.get("core_fes", [])],
                "score":            round(r.get("distance", 0.0), 3),
            })
        return hints

    async def find_similar(
        self,
        trigger_lemma: str,
        event_description: str,
        threshold: Optional[float] = None,
        query_embedding: Optional[list[float]] = None,
    ) -> Optional[FrameDefinitionSchema]:
        """Return best-matching frame if similarity >= threshold."""
        thr = threshold or self.SIMILARITY_THRESHOLD
        query_str = f"{trigger_lemma}: {event_description}"
        results = await self._vdb.query(
            query_str, top_k=1, query_embedding=query_embedding
        )
        for r in results:
            if r.get("distance", 0.0) >= thr:
                return await self.get(r.get("frame_name", ""))
        return None

    async def all_frames(self) -> list[FrameDefinitionSchema]:
        frames = []
        for name in self._frame_names:
            data = await self._kv.get_by_id(name)
            if data:
                frames.append(self._dict_to_schema(data))
        return frames

    async def get_frame_embedding(self, frame_name: str) -> Optional[list[float]]:
        """Return the stored vector for a frame (for expanded-frame seeding)."""
        results = await self._vdb.query(
            frame_name, top_k=1, query_embedding=None
        )
        # No direct get-by-id for vectors; search and filter by exact name
        for r in results:
            if r.get("frame_name") == frame_name:
                return None  # NanoVDB query only returns meta, not raw vector
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dict_to_schema(data: dict) -> FrameDefinitionSchema:
        return FrameDefinitionSchema(
            frame_name=data["frame_name"],
            lexical_units=data.get("lexical_units", []),
            frame_definition=data["frame_definition"],
            core_fes=[CoreFESchema(**fe) for fe in data.get("core_fes", [])],
            noncore_fes=[NonCoreFESchema(**fe) for fe in data.get("noncore_fes", [])],
            is_from_framenet=data.get("is_from_framenet", False),
            usage_count=data.get("usage_count", 0),
        )

    def format_hints_for_prompt(self, hints: list[dict]) -> str:
        if not hints:
            return "(no existing frames yet — create new frames as needed)"
        lines = []
        for h in hints:
            lus = ", ".join(h["lexical_units"])
            fes = ", ".join(h["core_fes"])
            lines.append(
                f'  - {h["frame_name"]} (LUs: {lus}) | Core FEs: {fes}\n'
                f'    Definition: {h["frame_definition"]}'
            )
        return "\n".join(lines)
