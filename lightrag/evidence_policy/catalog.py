"""Read-only, validated catalog loader.

The loader is the only place that reads a catalog file. A catalog is either
valid or it raises :class:`CatalogError`; there is no partially loaded state,
because a partially loaded catalog would silently answer from a subset of the
versions. The returned :class:`Catalog` is immutable and can be frozen into a
request snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import (
    RESOLUTION_STATUSES,  # noqa: F401  (re-exported for callers' convenience)
    VERSION_KINDS,
    AuthorityPolicy,
    Catalog,
    CatalogDocument,
    CatalogError,
    CatalogVersion,
    ChunkRef,
    ScopeKey,
    canonical_json,
    parse_full_date,
    sha256_text,
    validate_source_id,
)

SUPPORTED_SCHEMA_VERSION = 1


def load_catalog(path: str | Path) -> Catalog:
    """Load and validate a catalog document from ``path``."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError("missing_metadata", f"catalog not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError("missing_metadata", f"catalog is not JSON: {path}: {exc}") from exc
    return catalog_from_mapping(raw, source_path=str(path))


def catalog_from_mapping(raw: Any, *, source_path: str = "") -> Catalog:
    if not isinstance(raw, dict):
        raise CatalogError("missing_metadata", "catalog must be a JSON object")
    schema_version = raw.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise CatalogError(
            "missing_metadata",
            f"unsupported catalog schema_version {schema_version!r}",
            expected=SUPPORTED_SCHEMA_VERSION,
        )
    revision = raw.get("catalog_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise CatalogError("missing_metadata", "catalog_revision must be a non-empty string")
    policy = _load_policy(raw.get("policy"))

    documents: dict[str, CatalogDocument] = {}
    for entry in _require_list(raw, "documents"):
        if not isinstance(entry, dict):
            raise CatalogError("missing_metadata", "documents entries must be objects")
        doc_id = entry.get("id")
        if not isinstance(doc_id, str) or not doc_id:
            raise CatalogError("missing_metadata", "documents entry needs a string id")
        if doc_id in documents:
            raise CatalogError("missing_metadata", f"duplicate document id {doc_id!r}")
        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            raise CatalogError(
                "missing_metadata", f"document {doc_id!r} needs non-empty content"
            )
        documents[doc_id] = CatalogDocument(
            id=doc_id,
            published_at=_optional_date(entry.get("published_at"), doc_id),
            content=content,
        )

    versions: dict[str, CatalogVersion] = {}
    for entry in _require_list(raw, "versions"):
        version = _load_version(entry, documents)
        if version.version_id in versions:
            raise CatalogError(
                "missing_metadata", f"duplicate version_id {version.version_id!r}"
            )
        versions[version.version_id] = version

    by_scope: dict[ScopeKey, list[str]] = {}
    for version in versions.values():
        by_scope.setdefault(version.scope_key, []).append(version.version_id)

    superseded_by: dict[str, list[str]] = {}
    for version in versions.values():
        for previous in version.supersedes:
            superseded_by.setdefault(previous, []).append(version.version_id)

    chunk_index: dict[str, list[str]] = {}
    for version in versions.values():
        for ref in version.chunk_refs:
            bucket = chunk_index.setdefault(ref.chunk_id, [])
            if version.version_id not in bucket:
                bucket.append(version.version_id)

    catalog = Catalog(
        revision=revision.strip(),
        policy=policy,
        documents=documents,
        versions=versions,
        by_scope={scope: tuple(sorted(ids)) for scope, ids in by_scope.items()},
        superseded_by={vid: tuple(sorted(ids)) for vid, ids in superseded_by.items()},
        source_path=source_path,
        notes=raw.get("notes") if isinstance(raw.get("notes"), dict) else {},
        chunk_index={cid: tuple(sorted(ids)) for cid, ids in chunk_index.items()},
    )
    _validate_links(catalog)
    _validate_acyclic(catalog)
    fingerprint = sha256_text(canonical_json(_normalized(catalog)))
    return Catalog(
        revision=catalog.revision,
        policy=catalog.policy,
        documents=catalog.documents,
        versions=catalog.versions,
        by_scope=catalog.by_scope,
        superseded_by=catalog.superseded_by,
        fingerprint=fingerprint,
        source_path=catalog.source_path,
        notes=catalog.notes,
        chunk_index=catalog.chunk_index,
    )


def _require_list(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise CatalogError("missing_metadata", f"catalog.{key} must be a list")
    return value


def _load_policy(raw: Any) -> AuthorityPolicy:
    if not isinstance(raw, dict):
        raise CatalogError("missing_metadata", "catalog.policy must be an object")
    version = raw.get("version")
    ranking = raw.get("authority_ranking")
    if not isinstance(version, str) or not version.strip():
        raise CatalogError("missing_metadata", "catalog.policy.version must be a string")
    if (
        not isinstance(ranking, list)
        or not ranking
        or not all(isinstance(item, str) and item for item in ranking)
    ):
        raise CatalogError(
            "missing_metadata",
            "catalog.policy.authority_ranking must be a non-empty list of strings",
        )
    if len(set(ranking)) != len(ranking):
        raise CatalogError(
            "missing_metadata", "catalog.policy.authority_ranking contains duplicates"
        )
    reserved = {
        "commentary": "the version kind 'commentary' can never be authoritative",
        "untrusted": "the authority class 'untrusted' is reserved",
    }
    for name, why in reserved.items():
        if name in ranking:
            raise CatalogError(
                "missing_metadata",
                f"catalog.policy.authority_ranking must not list {name!r}: {why}",
            )
    return AuthorityPolicy(version=version.strip(), ranking=tuple(ranking))


def _optional_date(value: Any, where: str):
    if value is None or value == "":
        return None
    try:
        return parse_full_date(value, field_name=f"{where}.published_at")
    except CatalogError as exc:
        # Only full dates participate; a coarse published date is display-only
        # and therefore allowed to stay unknown instead of failing the catalog.
        if "full YYYY-MM-DD" in exc.message or "calendar date" in exc.message:
            return None
        raise


def _load_version(entry: Any, documents: dict[str, CatalogDocument]) -> CatalogVersion:
    if not isinstance(entry, dict):
        raise CatalogError("missing_metadata", "versions entries must be objects")
    version_id = entry.get("version_id")
    if not isinstance(version_id, str) or not version_id:
        raise CatalogError("missing_metadata", "versions entry needs a string version_id")
    document_id = entry.get("document_id")
    if document_id not in documents:
        raise CatalogError(
            "missing_metadata",
            f"versions[{version_id}].document_id {document_id!r} is not in documents",
            version_id=version_id,
        )
    kind = entry.get("kind")
    if kind not in VERSION_KINDS:
        raise CatalogError(
            "missing_metadata",
            f"versions[{version_id}].kind must be one of {sorted(VERSION_KINDS)}",
            version_id=version_id,
        )
    authority_class = entry.get("authority_class")
    if not isinstance(authority_class, str) or not authority_class:
        raise CatalogError(
            "missing_metadata",
            f"versions[{version_id}].authority_class must be a string",
            version_id=version_id,
        )
    if kind == "commentary" and authority_class not in ("commentary",):
        raise CatalogError(
            "missing_metadata",
            f"versions[{version_id}] is a commentary but claims authority_class "
            f"{authority_class!r}; a commentary cannot carry official authority",
            version_id=version_id,
        )
    source_id = validate_source_id(entry.get("source_id"))

    valid_from = _required_precision_date(entry.get("valid_from"), version_id, "valid_from")
    valid_to = _required_precision_date(entry.get("valid_to"), version_id, "valid_to")
    if valid_from is not None and valid_to is not None and valid_from >= valid_to:
        raise CatalogError(
            "unknown_time",
            f"versions[{version_id}] has an inverted validity interval "
            f"[{valid_from}, {valid_to})",
            version_id=version_id,
        )

    supersedes = _id_tuple(entry.get("supersedes", []), version_id, "supersedes")
    requires = _id_tuple(entry.get("requires_evidence", []), version_id, "requires_evidence")
    for name, ids in (("supersedes", supersedes), ("requires_evidence", requires)):
        if version_id in ids:
            raise CatalogError(
                "catalog_cycle",
                f"versions[{version_id}].{name} references itself",
                version_id=version_id,
            )
    if kind == "commentary" and supersedes:
        raise CatalogError(
            "missing_metadata",
            f"versions[{version_id}] is a commentary and cannot supersede a rule",
            version_id=version_id,
        )
    if kind == "repeal" and not supersedes:
        raise CatalogError(
            "missing_metadata",
            f"versions[{version_id}] is a repeal but names nothing in supersedes",
            version_id=version_id,
        )

    refs_raw = entry.get("chunk_refs", [])
    if not isinstance(refs_raw, list):
        raise CatalogError(
            "missing_metadata", f"versions[{version_id}].chunk_refs must be a list"
        )
    refs = tuple(ChunkRef.from_mapping(ref, version_id=version_id) for ref in refs_raw)

    locator = entry.get("source_locator")
    if locator is not None and not isinstance(locator, str):
        raise CatalogError(
            "missing_metadata", f"versions[{version_id}].source_locator must be a string"
        )
    label = entry.get("label") or ""
    if not isinstance(label, str):
        raise CatalogError("missing_metadata", f"versions[{version_id}].label must be a string")

    document = documents[document_id]
    try:
        published_at = _required_precision_date(
            entry.get("published_at"), version_id, "published_at"
        )
    except CatalogError:
        published_at = None
    if published_at is None:
        published_at = document.published_at

    return CatalogVersion(
        version_id=version_id,
        document_id=document_id,
        scope_key=ScopeKey.from_mapping(entry.get("scope_key"), field_name=f"versions[{version_id}].scope_key"),
        kind=kind,
        source_id=source_id,
        authority_class=authority_class,
        valid_from=valid_from,
        valid_to=valid_to,
        supersedes=supersedes,
        requires_evidence=requires,
        chunk_refs=refs,
        value=entry.get("value"),
        label=label,
        source_locator=locator or "",
        published_at=published_at,
    )


def _required_precision_date(value: Any, version_id: str, field_name: str):
    try:
        return parse_full_date(value, field_name=f"versions[{version_id}].{field_name}")
    except CatalogError as exc:
        if "full YYYY-MM-DD" in exc.message or "calendar date" in exc.message:
            raise CatalogError(
                "unknown_time",
                f"versions[{version_id}].{field_name} must be a full date or null: "
                f"{value!r}; coarse values are not padded",
                version_id=version_id,
                field=field_name,
            ) from exc
        raise


def _id_tuple(raw: Any, version_id: str, field_name: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        raise CatalogError(
            "missing_metadata",
            f"versions[{version_id}].{field_name} must be a list of version ids",
            version_id=version_id,
        )
    if len(set(raw)) != len(raw):
        raise CatalogError(
            "missing_metadata",
            f"versions[{version_id}].{field_name} contains duplicates",
            version_id=version_id,
        )
    return tuple(raw)


def _validate_links(catalog: Catalog) -> None:
    for version in catalog.versions.values():
        for name, ids in (
            ("supersedes", version.supersedes),
            ("requires_evidence", version.requires_evidence),
        ):
            for target_id in ids:
                target = catalog.versions.get(target_id)
                if target is None:
                    raise CatalogError(
                        "missing_evidence",
                        f"versions[{version.version_id}].{name} points at unknown "
                        f"version {target_id!r}",
                        version_id=version.version_id,
                        target=target_id,
                    )
                if target.scope_key != version.scope_key:
                    raise CatalogError(
                        "missing_metadata",
                        f"versions[{version.version_id}].{name} crosses scope into "
                        f"{target.scope_key.as_dict()}",
                        version_id=version.version_id,
                        target=target_id,
                    )


def _validate_acyclic(catalog: Catalog) -> None:
    state: dict[str, int] = {}  # 1 = visiting, 2 = done

    def visit(version_id: str, path: tuple[str, ...]) -> None:
        if state.get(version_id) == 2:
            return
        if state.get(version_id) == 1:
            cycle = path[path.index(version_id) :] + (version_id,)
            raise CatalogError(
                "catalog_cycle",
                "supersedes cycle: " + " -> ".join(cycle),
                cycle=list(cycle),
            )
        state[version_id] = 1
        version = catalog.versions[version_id]
        for target in (*version.supersedes, *version.requires_evidence):
            visit(target, path + (version_id,))
        state[version_id] = 2

    for version_id in catalog.versions:
        visit(version_id, ())


def _normalized(catalog: Catalog) -> dict:
    return {
        "catalog_revision": catalog.revision,
        "policy": {"version": catalog.policy.version, "authority_ranking": list(catalog.policy.ranking)},
        "documents": {
            doc_id: {"published_at": _iso(doc.published_at), "content": doc.content}
            for doc_id, doc in sorted(catalog.documents.items())
        },
        "versions": {
            vid: {
                "document_id": v.document_id,
                "scope_key": v.scope_key.as_dict(),
                "kind": v.kind,
                "source_id": v.source_id,
                "authority_class": v.authority_class,
                "valid_from": _iso(v.valid_from),
                "valid_to": _iso(v.valid_to),
                "supersedes": list(v.supersedes),
                "requires_evidence": list(v.requires_evidence),
                "chunk_refs": [ref.as_dict() for ref in v.chunk_refs],
                "value": v.value,
                "label": v.label,
                "source_locator": v.source_locator,
                "published_at": _iso(v.published_at),
            }
            for vid, v in sorted(catalog.versions.items())
        },
    }


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None
