"""A synthetic clause corpus for the core evidence-policy tests.

The documents model the same clause-version scenario as the demo: a rule
whose deadline changes from 180 to 270 days, a later amendment that changes it
again and then expires, a commentary that must never decide a rule, and a
same-scope statement from another source that contradicts the rule.

Everything here is fictional. The chunk IDs are fixture IDs, not LightRAG
storage IDs: a real workspace binds them after ingestion.
"""

from __future__ import annotations

import hashlib
from typing import Any

SCOPE = {
    "jurisdiction": "SYNTH-A",
    "instrument_id": "SYNTH-RULE",
    "article_id": "8",
    "subject_scope": "standard_members",
}

DOC_OLD = "d-old"
DOC_NEW = "d-new"
DOC_THIRD = "d-third"
DOC_COMMENT = "d-comment"
DOC_CONFLICT = "d-conflict"
DOC_UNDATED = "d-undated"

TEXT_OLD = "虚构规则 R，第 8 条：标准成员办理申请的期限为 180 天，自 2025-01-01 起生效。"
TEXT_NEW = (
    "虚构修订 R-1：自 2026-07-01 起，将虚构规则 R 第 8 条的办理期限从 180 天调整为 270 天。"
)
TEXT_THIRD = (
    "虚构修订 R-2：自 2026-10-01 起，将 R-1 所规定的 270 天调整为 300 天；"
    "本修订的有效期截至 2026-11-01，不恢复此前已被取代的条款。"
)
TEXT_COMMENT = "非官方评论：我认为标准成员办理申请只需 90 天。这是评论意见，不是规则修订。"
TEXT_CONFLICT = (
    "同级来源的另一份虚构规则声明：SYNTH-A 标准成员办理期限为 240 天，"
    "自 2026-08-01 起适用；本文件未指明替代任何现有版本。"
)
TEXT_UNDATED = "一份日期不详的虚构通知称：标准成员办理期限调整为 300 天，未说明适用时间。"

DOCUMENTS = {
    DOC_OLD: TEXT_OLD,
    DOC_NEW: TEXT_NEW,
    DOC_THIRD: TEXT_THIRD,
    DOC_COMMENT: TEXT_COMMENT,
    DOC_CONFLICT: TEXT_CONFLICT,
    DOC_UNDATED: TEXT_UNDATED,
}

CHUNK_OF_DOC = {
    DOC_OLD: "c-old",
    DOC_NEW: "c-new",
    DOC_THIRD: "c-third",
    DOC_COMMENT: "c-comment",
    DOC_CONFLICT: "c-conflict",
    DOC_UNDATED: "c-undated",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_ref(doc_id: str) -> dict[str, Any]:
    text = DOCUMENTS[doc_id]
    return {
        "chunk_id": CHUNK_OF_DOC[doc_id],
        "doc_id": doc_id,
        "span": [0, len(text)],
        "content_sha256": sha256_text(text),
        "file_path": f"{doc_id}.txt",
    }


def chunk_rows() -> list[dict[str, Any]]:
    """Rows shaped like ``text_chunks.get_by_ids`` output (id under ``_id``)."""
    return [
        {
            "_id": CHUNK_OF_DOC[doc_id],
            "content": text,
            "full_doc_id": doc_id,
            "file_path": f"{doc_id}.txt",
            "chunk_order_index": 0,
        }
        for doc_id, text in DOCUMENTS.items()
    ]


def catalog_mapping(
    *, include: list[str] | None = None, conflict_rank: str = "official"
) -> dict[str, Any]:
    versions: list[dict[str, Any]] = [
        {
            "version_id": "V-old",
            "document_id": DOC_OLD,
            "scope_key": SCOPE,
            "kind": "rule",
            "source_id": "synthetic-regulator",
            "authority_class": "official",
            "value": 180,
            "valid_from": "2025-01-01",
            "valid_to": None,
            "supersedes": [],
            "requires_evidence": [],
            "chunk_refs": [_chunk_ref(DOC_OLD)],
            "source_locator": f"{DOC_OLD}: 第 8 条",
        },
        {
            "version_id": "V-new",
            "document_id": DOC_NEW,
            "scope_key": SCOPE,
            "kind": "amendment",
            "source_id": "synthetic-regulator",
            "authority_class": "official",
            "value": None,
            "valid_from": "2026-07-01",
            "valid_to": None,
            "supersedes": ["V-old"],
            "requires_evidence": ["V-old"],
            "chunk_refs": [_chunk_ref(DOC_NEW)],
            "source_locator": f"{DOC_NEW}: 全文",
        },
        {
            "version_id": "V-third",
            "document_id": DOC_THIRD,
            "scope_key": SCOPE,
            "kind": "amendment",
            "source_id": "synthetic-regulator",
            "authority_class": "official",
            "value": None,
            "valid_from": "2026-10-01",
            "valid_to": "2026-11-01",
            "supersedes": ["V-new"],
            "requires_evidence": ["V-old", "V-new"],
            "chunk_refs": [_chunk_ref(DOC_THIRD)],
            "source_locator": f"{DOC_THIRD}: 全文",
        },
        {
            "version_id": "V-comment",
            "document_id": DOC_COMMENT,
            "scope_key": SCOPE,
            "kind": "commentary",
            "source_id": "synthetic-blog",
            "authority_class": "commentary",
            "value": 90,
            "valid_from": "2026-08-01",
            "valid_to": None,
            "supersedes": [],
            "requires_evidence": [],
            "chunk_refs": [_chunk_ref(DOC_COMMENT)],
            "source_locator": f"{DOC_COMMENT}: 全文",
        },
        {
            "version_id": "V-conflict",
            "document_id": DOC_CONFLICT,
            "scope_key": SCOPE,
            "kind": "rule",
            "source_id": "synthetic-peer-agency",
            "authority_class": conflict_rank,
            "value": 240,
            "valid_from": "2026-08-01",
            "valid_to": None,
            "supersedes": [],
            "requires_evidence": [],
            "chunk_refs": [_chunk_ref(DOC_CONFLICT)],
            "source_locator": f"{DOC_CONFLICT}: 全文",
        },
        {
            "version_id": "V-undated",
            "document_id": DOC_UNDATED,
            "scope_key": SCOPE,
            "kind": "amendment",
            "source_id": "synthetic-regulator",
            "authority_class": "official",
            "value": 300,
            "valid_from": None,
            "valid_to": None,
            "supersedes": [],
            "requires_evidence": [],
            "chunk_refs": [_chunk_ref(DOC_UNDATED)],
            "source_locator": f"{DOC_UNDATED}: 全文",
        },
    ]
    if include is not None:
        versions = [v for v in versions if v["version_id"] in set(include)]
    documents = [
        {"id": doc_id, "published_at": None, "content": text}
        for doc_id, text in DOCUMENTS.items()
    ]
    return {
        "schema_version": 1,
        "catalog_revision": "test-2026-10-26",
        "policy": {"version": "test-authority-v1", "authority_ranking": ["statutory", "official"]},
        "documents": documents,
        "versions": versions,
    }


class FakeChunkStore:
    """Minimal ``text_chunks`` stand-in: ``get_by_ids`` with None placeholders."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = {row["_id"]: row for row in (rows or chunk_rows())}
        self.calls = 0

    async def get_by_ids(self, ids: list[str], fields: list[str] | None = None):
        self.calls += 1
        return [self._rows.get(chunk_id) for chunk_id in ids]


class FailingChunkStore:
    async def get_by_ids(self, ids: list[str], fields: list[str] | None = None):
        raise RuntimeError("text_chunks backend unavailable")
