"""Binding catalog versions to real chunks of a workspace, fail-loud.

A catalog version is only evidence once its ``chunk_refs`` resolve to a real
``text_chunks`` row whose content still matches the recorded span hash. The
contract (README section 4):

- never fabricate a chunk ID; a chunk that cannot be read back is
  ``missing_evidence``, not an empty span;
- ``get_by_ids`` returning ``None`` for a requested ID (confirmed absent) and
  the call raising (backend error) are different outcomes and stay different;
- a chunk carrying spans of two or more versions is *not* ambiguous -- each
  version keeps its own span -- but a version whose spans cannot be told apart
  from another version's is ``mapping_ambiguous`` and stops the answer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import Catalog, CatalogError, CatalogVersion, ChunkRef


class ChunkReader(Protocol):
    """The slice of ``BaseKVStorage`` (``text_chunks``) the binder needs."""

    async def get_by_ids(
        self, ids: list[str], fields: list[str] | None = None
    ) -> list[dict[str, Any]]: ...


class StorageBackendError(RuntimeError):
    """A storage call raised. Distinct from "the row is absent"."""


@dataclass(frozen=True)
class BoundSpan:
    """One version's contribution to a chunk: the text that carries it."""

    version_id: str
    chunk_id: str
    doc_id: str
    file_path: str
    span: tuple[int, int]
    text: str
    content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "span": list(self.span),
            "text": self.text,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class BindingProblem:
    version_id: str
    reason: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Binding:
    spans: dict[str, tuple[BoundSpan, ...]]
    problems: tuple[BindingProblem, ...]
    chunk_versions: dict[str, tuple[str, ...]]

    @property
    def ok(self) -> bool:
        return not self.problems

    def problems_for(self, version_ids) -> tuple[BindingProblem, ...]:
        wanted = set(version_ids)
        return tuple(p for p in self.problems if p.version_id in wanted)


async def bind_versions(
    catalog: Catalog,
    versions: list[str],
    reader: ChunkReader,
    *,
    require_doc_match: bool = True,
) -> Binding:
    """Resolve ``versions`` to spans of real chunks, verifying every hash."""
    wanted = [vid for vid in dict.fromkeys(versions) if vid in catalog.versions]
    unknown = [vid for vid in versions if vid not in catalog.versions]
    problems: list[BindingProblem] = [
        BindingProblem(vid, "missing_evidence", "version is not in the catalog")
        for vid in unknown
    ]
    chunk_ids = sorted(
        {ref.chunk_id for vid in wanted for ref in catalog.versions[vid].chunk_refs}
    )

    rows: dict[str, dict[str, Any] | None] = {}
    if chunk_ids:
        try:
            fetched = await reader.get_by_ids(chunk_ids)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed error
            raise StorageBackendError(
                f"text_chunks.get_by_ids failed for {len(chunk_ids)} chunk(s): {exc}"
            ) from exc
        for index, row in enumerate(fetched or []):
            # ``get_by_ids`` returns a ``None`` placeholder for a confirmed
            # missing key and the key itself under ``_id`` (not ``id``); a
            # backend may also omit the id entirely, in which case the
            # documented request order is what pairs a row with its ID.
            if not isinstance(row, dict):
                continue
            row_id = (
                row.get("_id")
                or row.get("id")
                or row.get("chunk_id")
                or row.get("__id__")
            )
            if row_id is None and index < len(chunk_ids):
                row_id = chunk_ids[index]
            if row_id is not None:
                rows[str(row_id)] = row
        for chunk_id in chunk_ids:
            rows.setdefault(chunk_id, None)

    spans: dict[str, list[BoundSpan]] = {}
    chunk_versions: dict[str, list[str]] = {}
    seen_spans: dict[tuple[str, tuple[int, int]], str] = {}

    for version_id in wanted:
        version = catalog.versions[version_id]
        if not version.chunk_refs:
            problems.append(
                BindingProblem(
                    version_id, "missing_evidence", "catalog version has no chunk_refs"
                )
            )
            continue
        for ref in version.chunk_refs:
            problem = _check_row(
                version, ref, rows.get(ref.chunk_id), require_doc_match
            )
            if problem is not None:
                problems.append(problem)
                continue
            content = rows[ref.chunk_id]["content"]
            start, end = ref.span
            text = content[start:end]
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest != ref.content_sha256:
                problems.append(
                    BindingProblem(
                        version_id,
                        "missing_evidence",
                        f"chunk {ref.chunk_id} span {list(ref.span)} no longer matches "
                        f"recorded sha256 (chunk changed or catalog is stale)",
                    )
                )
                continue
            key = (ref.chunk_id, ref.span)
            owner = seen_spans.get(key)
            if owner is not None and owner != version_id:
                for vid in (owner, version_id):
                    problems.append(
                        BindingProblem(
                            vid,
                            "mapping_ambiguous",
                            f"span {list(ref.span)} of chunk {ref.chunk_id} is attributed "
                            f"to both {owner} and {version_id}",
                        )
                    )
                continue
            seen_spans[key] = version_id
            spans.setdefault(version_id, []).append(
                BoundSpan(
                    version_id=version_id,
                    chunk_id=ref.chunk_id,
                    doc_id=ref.doc_id,
                    file_path=ref.file_path,
                    span=ref.span,
                    text=text,
                    content_sha256=digest,
                )
            )
            bucket = chunk_versions.setdefault(ref.chunk_id, [])
            if version_id not in bucket:
                bucket.append(version_id)

    return Binding(
        spans={vid: tuple(items) for vid, items in spans.items()},
        problems=tuple(problems),
        chunk_versions={cid: tuple(items) for cid, items in chunk_versions.items()},
    )


def _check_row(
    version: CatalogVersion,
    ref: ChunkRef,
    row: dict[str, Any] | None,
    require_doc_match: bool,
) -> BindingProblem | None:
    if row is None:
        return BindingProblem(
            version.version_id,
            "missing_evidence",
            f"chunk {ref.chunk_id} is absent from text_chunks (deleted or never "
            "ingested); the catalog will not invent a replacement",
        )
    content = row.get("content")
    if not isinstance(content, str):
        return BindingProblem(
            version.version_id,
            "missing_evidence",
            f"chunk {ref.chunk_id} has no text content",
        )
    start, end = ref.span
    if end > len(content):
        return BindingProblem(
            version.version_id,
            "missing_evidence",
            f"chunk {ref.chunk_id} span {list(ref.span)} exceeds content length "
            f"{len(content)}",
        )
    if require_doc_match:
        doc_ids = _doc_ids_of(row.get("full_doc_id"))
        if ref.doc_id not in doc_ids:
            return BindingProblem(
                version.version_id,
                "mapping_ambiguous",
                f"chunk {ref.chunk_id} belongs to document(s) {sorted(doc_ids)}, "
                f"not {ref.doc_id!r}",
            )
    return None


def _doc_ids_of(value: Any) -> list[str]:
    import json as _json

    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = _json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item]
        return [text]
    return [str(value)]


def problem_to_catalog_error(problems: tuple[BindingProblem, ...]) -> CatalogError:
    first = problems[0]
    return CatalogError(
        first.reason, first.detail, problems=[p.as_dict() for p in problems]
    )
