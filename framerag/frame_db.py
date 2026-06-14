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
        # Frame-level VDB: embeds "frame_name [SEP] frame_definition"
        self._vdb = make_vdb(
            "frame_vec", db_dir, embedding_func,
            {"frame_name", "frame_definition"},
        )
        # FE-level VDB: embeds "fe_name: fe_definition [FRAME: frame_name]"
        self._fe_vdb = make_vdb(
            "fe_vec", db_dir, embedding_func,
            {"fe_name", "frame_name", "fe_definition"},
        )
        self._frame_names: set[str] = set()

    async def initialize(self) -> None:
        await self._kv.initialize()
        await self._vdb.initialize()
        await self._fe_vdb.initialize()
        self._load_names()

    async def index_done_callback(self) -> None:
        await self._kv.index_done_callback()
        await self._vdb.index_done_callback()
        await self._fe_vdb.index_done_callback()
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
        # Index each FE definition separately for semantic role search
        fe_batch: dict[str, dict] = {}
        for fe in frame.core_fes + frame.noncore_fes:
            if not fe.fe_definition:
                continue
            fe_key = f"{frame.frame_name}::{fe.fe_name}"
            fe_batch[fe_key] = {
                "content":      f"{fe.fe_name}: {fe.fe_definition} [FRAME: {frame.frame_name}]",
                "fe_name":      fe.fe_name,
                "frame_name":   frame.frame_name,
                "fe_definition": fe.fe_definition,
            }
        if fe_batch:
            await self._fe_vdb.upsert(fe_batch)
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
        """Return top-K similar frames as hint dicts (embedding from content).

        core_fes is a list of dicts with fe_name and fe_definition so
        format_hints_for_prompt() can surface full FE semantics to the LLM.
        """
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
                "core_fes": [
                    {
                        "fe_name":       fe["fe_name"],
                        "fe_definition": fe.get("fe_definition", ""),
                    }
                    for fe in data.get("core_fes", [])
                ],
                "score":            round(r.get("distance", 0.0), 3),
            })
        return hints

    async def search_related_frames(
        self,
        query: str,
        top_k: int = 8,
        threshold: float = 0.5,
        exclude: Optional[set[str]] = None,
    ) -> list[str]:
        """Return frame names semantically related to query via embedding search.

        Used for embedding-based frame expansion at query time (no LLM needed).
        """
        results = await self._vdb.query(query, top_k=top_k)
        frames: list[str] = []
        seen: set[str] = set(exclude or [])
        for r in results:
            fname = r.get("frame_name", "")
            score = r.get("distance", 0.0)
            if fname and fname not in seen and score >= threshold:
                frames.append(fname)
                seen.add(fname)
        return frames

    async def search_by_fe(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Return FE-level hits most semantically similar to query.

        Useful for finding which semantic roles are relevant to a question
        (e.g., "who caused it" → Agent FE, "how much" → Value FE).
        Returns list of {fe_name, frame_name, fe_definition, score}.
        """
        results = await self._fe_vdb.query(query, top_k=top_k)
        hits = []
        for r in results:
            fe_name = r.get("fe_name", "")
            if not fe_name:
                continue
            hits.append({
                "fe_name":       fe_name,
                "frame_name":    r.get("frame_name", ""),
                "fe_definition": r.get("fe_definition", ""),
                "score":         round(r.get("distance", 0.0), 3),
            })
        return hits

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
            core_fes = h.get("core_fes", [])
            if core_fes and isinstance(core_fes[0], dict):
                # Full dicts with definitions: show name + brief definition
                fe_parts = []
                for fe in core_fes:
                    defn = fe.get("fe_definition", "")
                    fe_parts.append(
                        f'{fe["fe_name"]} ({defn})' if defn else fe["fe_name"]
                    )
                fes_text = "; ".join(fe_parts)
            else:
                # Legacy: plain name strings
                fes_text = ", ".join(str(f) for f in core_fes)
            lines.append(
                f'  - {h["frame_name"]} (LUs: {lus})\n'
                f'    Definition: {h["frame_definition"]}\n'
                f'    Core FEs: {fes_text}'
            )
        return "\n".join(lines)
