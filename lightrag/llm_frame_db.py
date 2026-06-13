"""
Dynamic Frame Database — FSRAG-inspired LLM-driven frame semantics.

Manages a persistent database of semantic frames with Frame Elements (FEs)
and Lexical Units (LUs).  Frames are generated dynamically by an LLM during
indexing and cached for reuse across sessions.

Database file: <working_dir>/frame_db.json
Schema:
{
  "frames": {
    "FrameName": {
      "name": str,
      "definition": str,
      "frame_elements": {
        "RoleName": {"definition": str, "type": "core"|"peripheral"}
      },
      "lexical_units": ["word.pos", ...],
      "relations": ["RelatedFrame", ...]
    }
  },
  "lu_index": {"word": ["FrameName", ...]}
}
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from lightrag.utils import logger

_FRAME_DB_FILENAME = "frame_db.json"
_WRITE_LOCK = asyncio.Lock()

# Module-level singletons keyed by working_dir (one DB per LightRAG instance)
_INSTANCES: dict[str, "DynamicFrameDatabase"] = {}


class DynamicFrameDatabase:
    """Persistent, async-safe store of dynamically generated semantic frames."""

    def __init__(self, working_dir: str):
        self._path = Path(working_dir) / _FRAME_DB_FILENAME
        self.frames: dict[str, dict] = {}
        self.lu_index: dict[str, list[str]] = {}
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.frames = data.get("frames", {})
            self.lu_index = data.get("lu_index", {})
            logger.info(
                "[frame_db] Loaded %d frames, %d LU entries from %s",
                len(self.frames),
                len(self.lu_index),
                self._path,
            )
        except Exception as exc:
            logger.warning("[frame_db] Load failed (%s) — starting fresh.", exc)

    async def save(self) -> None:
        async with _WRITE_LOCK:
            data = {"frames": self.frames, "lu_index": self.lu_index}
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(self._path)
        logger.debug(
            "[frame_db] Saved %d frames to %s", len(self.frames), self._path
        )

    # ── Read helpers ───────────────────────────────────────────────────────────

    def has_frame(self, name: str) -> bool:
        return name in self.frames

    def get_frame(self, name: str) -> Optional[dict]:
        return self.frames.get(name)

    def get_all_names(self) -> list[str]:
        return list(self.frames.keys())

    def frames_for_lu(self, word: str) -> list[str]:
        """Return frame names associated with a lexical unit word."""
        return self.lu_index.get(word.lower(), [])

    def summary_for_llm(self, max_frames: int = 40) -> str:
        """Compact text listing existing frames — fed to LLM for deduplication."""
        lines: list[str] = []
        for name, f in list(self.frames.items())[:max_frames]:
            fe_names = ", ".join(f.get("frame_elements", {}).keys())
            defn = f.get("definition", "")[:90]
            lines.append(f"- {name}: {defn}  [FEs: {fe_names}]")
        extra = len(self.frames) - max_frames
        if extra > 0:
            lines.append(f"  … and {extra} more frames.")
        return "\n".join(lines) if lines else "(empty — no frames defined yet)"

    def find_candidate(self, name: str) -> Optional[str]:
        """Exact or case-insensitive name match against existing frames."""
        if name in self.frames:
            return name
        nl = name.lower()
        for existing in self.frames:
            if existing.lower() == nl:
                return existing
        return None

    # ── Write helpers ──────────────────────────────────────────────────────────

    def add_frame(self, frame_dict: dict) -> None:
        """Register a frame and update the LU index."""
        name = frame_dict["name"]
        self.frames[name] = frame_dict
        for lu in frame_dict.get("lexical_units", []):
            word = lu.split(".")[0].lower()
            bucket = self.lu_index.setdefault(word, [])
            if name not in bucket:
                bucket.append(name)
        logger.debug("[frame_db] Added frame '%s'", name)


# ── Module-level factory ───────────────────────────────────────────────────────

def get_frame_db(working_dir: str) -> DynamicFrameDatabase:
    """Return (and cache) the DynamicFrameDatabase for *working_dir*."""
    if working_dir not in _INSTANCES:
        _INSTANCES[working_dir] = DynamicFrameDatabase(working_dir)
    return _INSTANCES[working_dir]
