"""Lightweight document status tracking for FrameRAG."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

from lightrag.utils import logger, compute_mdhash_id

from .storage import make_kv


class DocStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    PROCESSED  = "processed"
    FAILED     = "failed"


@dataclass
class DocRecord:
    doc_id:       str
    source_doc:   str
    status:       DocStatus
    chunk_ids:    list[str]     = field(default_factory=list)
    chunks_count: int           = 0
    error_msg:    Optional[str] = None
    created_at:   float         = field(default_factory=time.time)
    updated_at:   float         = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @staticmethod
    def from_dict(data: dict) -> "DocRecord":
        data = dict(data)
        data["status"] = DocStatus(data.get("status", DocStatus.PROCESSED))
        return DocRecord(**{k: v for k, v in data.items()
                            if k in DocRecord.__dataclass_fields__})


class DocStore:
    """Persistent document status tracker backed by JsonKVStorage."""

    def __init__(self, working_dir: str):
        self._kv = make_kv("doc_status", working_dir)

    async def initialize(self) -> None:
        await self._kv.initialize()

    async def index_done_callback(self) -> None:
        await self._kv.index_done_callback()

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def make_doc_id(source_doc: str) -> str:
        return compute_mdhash_id(source_doc, prefix="doc-")

    async def upsert(self, record: DocRecord) -> None:
        record.updated_at = time.time()
        await self._kv.upsert({record.doc_id: record.to_dict()})

    async def get(self, doc_id: str) -> Optional[DocRecord]:
        data = await self._kv.get_by_id(doc_id)
        return DocRecord.from_dict(data) if data else None

    async def list_all(self) -> list[DocRecord]:
        """Return all records sorted newest-first."""
        records: list[DocRecord] = []
        if not hasattr(self._kv, "_data") or self._kv._data is None:
            return records
        for doc_id in list(self._kv._data.keys()):
            rec = await self.get(doc_id)
            if rec:
                records.append(rec)
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    async def list_by_status(self, status: DocStatus) -> list[DocRecord]:
        return [r for r in await self.list_all() if r.status == status]

    async def get_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in DocStatus}
        for rec in await self.list_all():
            counts[rec.status.value] += 1
        return counts

    async def delete(self, doc_id: str) -> None:
        await self._kv.delete([doc_id])
