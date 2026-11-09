"""The one 180 -> 270 material chain F01-F04 runs on.

Two fictional documents and nothing else: the original clause (180 days) and a
revision that raises it to 270 days from 2026-07-01. The revision is *published*
on 2026-05-20, so "announced but not yet effective" (F04) is a real date
question, not a fixture trick.

The catalog is built from what the workspace actually stored (see
``run_live.build_catalog``): chunk IDs, document IDs and span hashes come from
``text_chunks`` after ingestion. Nothing here invents an ID.
"""

from __future__ import annotations

from typing import Any

SCOPE: dict[str, str] = {
    "jurisdiction": "DEMO-A",
    "instrument_id": "DEMO-RULE-R",
    "article_id": "8",
    "subject_scope": "standard_members",
}

DOC_OLD = "doc-3037-old"
DOC_NEW = "doc-3037-new"

DOCUMENTS: list[dict[str, Any]] = [
    {
        "doc_id": DOC_OLD,
        "file_path": "3037/demo/rule-R-art8-2025.txt",
        "published_at": "2025-01-01",
        "version_id": "V-old",
        "content": (
            "虚构规则 R 第 8 条\n\n"
            "标准成员办理申请的期限为 180 天。\n"
            "本条自 2025-01-01 起施行。"
        ),
    },
    {
        "doc_id": DOC_NEW,
        "file_path": "3037/demo/rule-R-art8-amendment-2026.txt",
        "published_at": "2026-05-20",
        "version_id": "V-new",
        "content": (
            "虚构修订 R-1（2026-05-20 公布）\n\n"
            "自 2026-07-01 起，将虚构规则 R 第 8 条规定的标准成员办理申请期限"
            "由 180 天调整为 270 天。\n"
            "本修订自 2026-07-01 生效，生效前原 180 天继续适用。"
        ),
    },
]

#: The version relations for the limited corpus. ``V-new`` is an *edition*: it
#: carries no ``value`` of its own, so it never competes as a second answer --
#: it replaces the wording of V-old inside the supersession chain.
VERSION_SPECS: list[dict[str, Any]] = [
    {
        "version_id": "V-old",
        "document_id": DOC_OLD,
        "kind": "rule",
        "source_id": "demo-regulator",
        "authority_class": "official",
        "value": 180,
        "valid_from": "2025-01-01",
        "valid_to": None,
        "supersedes": [],
        "requires_evidence": [],
        "source_locator": "rule-R-art8-2025.txt: 第 8 条",
    },
    {
        "version_id": "V-new",
        "document_id": DOC_NEW,
        "kind": "amendment",
        "source_id": "demo-regulator",
        "authority_class": "official",
        "value": None,
        "valid_from": "2026-07-01",
        "valid_to": None,
        "supersedes": ["V-old"],
        "requires_evidence": ["V-old"],
        "source_locator": "rule-R-art8-amendment-2026.txt: 全文",
    },
]

QUERY = "标准成员办理申请的期限是多少天？"


def catalog_mapping(
    *,
    document_ids: dict[str, str],
    chunk_refs: dict[str, list[dict[str, Any]]],
    revision: str,
) -> dict[str, Any]:
    """Assemble a catalog document from real workspace identifiers.

    ``document_ids`` maps a spec ``doc_id`` to the ID the pipeline actually
    assigned; ``chunk_refs`` maps a version ID to its bound spans.
    """
    documents = [
        {
            "id": document_ids[spec["doc_id"]],
            "published_at": spec["published_at"],
            "content": spec["content"],
        }
        for spec in DOCUMENTS
    ]
    versions = []
    for spec in VERSION_SPECS:
        entry = dict(spec)
        entry["document_id"] = document_ids[spec["document_id"]]
        entry["scope_key"] = dict(SCOPE)
        entry["chunk_refs"] = chunk_refs[spec["version_id"]]
        versions.append(entry)
    return {
        "schema_version": 1,
        "catalog_revision": revision,
        "policy": {
            "version": "demo-authority-v1",
            "authority_ranking": ["statutory", "official"],
        },
        "notes": {
            "corpus": "fictional; one clause, one revision, one date question",
            "purpose": "issue 3037 F01-F04 on a real workspace",
        },
        "documents": documents,
        "versions": versions,
    }


def spec_of(version_id: str) -> dict[str, Any]:
    for spec in VERSION_SPECS:
        if spec["version_id"] == version_id:
            return spec
    raise KeyError(version_id)


def document_spec(doc_id: str) -> dict[str, Any]:
    for spec in DOCUMENTS:
        if spec["doc_id"] == doc_id:
            return spec
    raise KeyError(doc_id)
