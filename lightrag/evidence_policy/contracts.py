"""Data contracts for authority- and effective-time-aware retrieval.

Every rule here is a *rule*, not an implementation detail: callers that violate
them get an explicit :class:`CatalogError` instead of a silently degraded
result. The rationale and the accepted limits are in
``docs/design/EvidencePolicyContract.md``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Reason codes (see docs/design/EvidencePolicyContract.md).
# ---------------------------------------------------------------------------

ReasonCode = Literal[
    "active",
    "superseded",
    "not_yet_effective",
    "expired",
    "repealed",
    "different_scope",
    "untrusted_source",
    "unknown_time",
    "missing_metadata",
    "mapping_ambiguous",
    "unresolved_conflict",
    "catalog_cycle",
    "missing_evidence",
    "expansion_limit",
    "group_budget_exceeded",
    "backend_error",
    "rerank_degraded",
    "cache_mismatch",
    "unknown",
]

REASON_CODES: frozenset[str] = frozenset(
    [
        "active",
        "superseded",
        "not_yet_effective",
        "expired",
        "repealed",
        "different_scope",
        "untrusted_source",
        "unknown_time",
        "missing_metadata",
        "mapping_ambiguous",
        "unresolved_conflict",
        "catalog_cycle",
        "missing_evidence",
        "expansion_limit",
        "group_budget_exceeded",
        "backend_error",
        "rerank_degraded",
        "cache_mismatch",
        "unknown",
    ]
)

# Result status returned to the caller. Only ``resolved`` may carry a
# evidence context; every other status has to say why. Model output is not
# automatically verified by the query path.
ResolutionStatus = Literal[
    "resolved",
    "needs_clarification",
    "insufficient_evidence",
    "no_applicable_rule",
    "unresolved_conflict",
    "unknown_time",
    "mapping_ambiguous",
    "catalog_invalid",
    "expansion_limit",
    "insufficient_context_budget",
    "backend_error",
    "unsupported",
    "invalid_request",
]

RESOLUTION_STATUSES: frozenset[str] = frozenset(
    [
        "resolved",
        "needs_clarification",
        "insufficient_evidence",
        "no_applicable_rule",
        "unresolved_conflict",
        "unknown_time",
        "mapping_ambiguous",
        "catalog_invalid",
        "expansion_limit",
        "insufficient_context_budget",
        "backend_error",
        "unsupported",
        "invalid_request",
    ]
)

VersionKind = Literal["rule", "amendment", "repeal", "commentary"]
VERSION_KINDS: frozenset[str] = frozenset(["rule", "amendment", "repeal", "commentary"])


class CatalogError(ValueError):
    """The catalog is invalid. Never downgrade this to "nothing was found"."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class RequestError(ValueError):
    """The request itself is invalid (bad date, unsupported mode, ...)."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


# ---------------------------------------------------------------------------
# Dates: full precision only. "2026" or "2026-10" is NOT 2026-01-01.
# ---------------------------------------------------------------------------

_FULL_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_full_date(value: Any, *, field_name: str) -> dt.date | None:
    """Parse an ISO full date, or ``None`` for an explicit "unknown".

    An empty string and ``None`` both mean unknown. Anything that is not a
   ``YYYY-MM-DD`` calendar date is rejected -- including decade-, year- or
    month-precision values, which keep their own precision by being rejected
    into the unknown bucket rather than being padded to the first of January.
    """
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not _FULL_DATE.match(value):
        raise CatalogError(
            "unknown_time",
            f"{field_name} must be a full YYYY-MM-DD date, got {value!r}",
            field=field_name,
            value=value,
        )
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise CatalogError(
            "unknown_time",
            f"{field_name} is not a calendar date: {value!r} ({exc})",
            field=field_name,
            value=value,
        ) from exc


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

SCOPE_FIELDS: tuple[str, ...] = (
    "jurisdiction",
    "instrument_id",
    "article_id",
    "subject_scope",
)


@dataclass(frozen=True, order=True)
class ScopeKey:
    """A clause identity.

    Compared as a tuple of four normalized fields, never as a joined string:
    ``("A", "BC", "1", "x")`` and ``("AB", "C", "1", "x")`` are different
    clauses and a delimiter could be constructed to confuse both. Two versions
    with different scope keys never supersede one another.
    """

    jurisdiction: str
    instrument_id: str
    article_id: str
    subject_scope: str

    @classmethod
    def from_mapping(cls, mapping: Any, *, field_name: str = "scope_key") -> "ScopeKey":
        if not isinstance(mapping, dict):
            raise CatalogError(
                "missing_metadata",
                f"{field_name} must be an object with {list(SCOPE_FIELDS)}",
                field=field_name,
            )
        missing = [name for name in SCOPE_FIELDS if not mapping.get(name)]
        if missing:
            raise CatalogError(
                "missing_metadata",
                f"{field_name} is missing {missing}",
                field=field_name,
                missing=missing,
            )
        extra = set(mapping) - set(SCOPE_FIELDS)
        if extra:
            # Unknown identity components would silently widen a scope match.
            raise CatalogError(
                "missing_metadata",
                f"{field_name} has unknown components {sorted(extra)}",
                field=field_name,
                unknown=sorted(extra),
            )
        values = [mapping[name] for name in SCOPE_FIELDS]
        for name, value in zip(SCOPE_FIELDS, values):
            if not isinstance(value, str):
                raise CatalogError(
                    "missing_metadata",
                    f"{field_name}.{name} must be a string",
                    field=f"{field_name}.{name}",
                )
        return cls(*(value.strip() for value in values))

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in SCOPE_FIELDS}

    def describe(self) -> str:
        return "|".join(getattr(self, name) for name in SCOPE_FIELDS)


# ---------------------------------------------------------------------------
# Chunk binding
# ---------------------------------------------------------------------------

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ChunkRef:
    """A span of a real LightRAG chunk that carries (part of) a clause version.

    ``chunk_id`` must be a chunk ID read back from the workspace
    ``text_chunks`` store; ``span`` is a half-open ``[start, end)`` character
    range *within the chunk content*; ``content_sha256`` is the hash of
    ``content[span]``. The loader only checks the shape; binding checks the
    content.
    """

    chunk_id: str
    doc_id: str
    span: tuple[int, int]
    content_sha256: str
    file_path: str = ""

    @classmethod
    def from_mapping(cls, mapping: Any, *, version_id: str) -> "ChunkRef":
        where = f"versions[{version_id}].chunk_refs"
        if not isinstance(mapping, dict):
            raise CatalogError("missing_metadata", f"{where} entries must be objects")
        chunk_id = mapping.get("chunk_id")
        doc_id = mapping.get("doc_id")
        span = mapping.get("span")
        digest = mapping.get("content_sha256")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise CatalogError("missing_metadata", f"{where} needs chunk_id", version_id=version_id)
        if not isinstance(doc_id, str) or not doc_id:
            raise CatalogError("missing_metadata", f"{where} needs doc_id", version_id=version_id)
        if (
            not isinstance(span, (list, tuple))
            or len(span) != 2
            or not all(type(bound) is int for bound in span)
            or span[0] < 0
            or span[1] <= span[0]
        ):
            raise CatalogError(
                "missing_metadata",
                f"{where} span must be a half-open [start, end) with 0 <= start < end",
                version_id=version_id,
                span=span,
            )
        if not isinstance(digest, str) or not _SHA256.match(digest):
            raise CatalogError(
                "missing_metadata",
                f"{where} needs a lowercase hex sha256 content_sha256",
                version_id=version_id,
            )
        file_path = mapping.get("file_path") or ""
        if not isinstance(file_path, str):
            raise CatalogError("missing_metadata", f"{where} file_path must be a string")
        return cls(
            chunk_id=chunk_id,
            doc_id=doc_id,
            span=(span[0], span[1]),
            content_sha256=digest,
            file_path=file_path,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "span": list(self.span),
            "content_sha256": self.content_sha256,
            "file_path": self.file_path,
        }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Catalog documents and versions
# ---------------------------------------------------------------------------

_SOURCE_ID = re.compile(r"^[A-Za-z0-9._:-]+$")


@dataclass(frozen=True)
class CatalogDocument:
    id: str
    published_at: dt.date | None
    content: str


@dataclass(frozen=True)
class CatalogVersion:
    """One clause version. Immutable once loaded, so a request snapshot is
    safe to share between coroutines without copying."""

    version_id: str
    document_id: str
    scope_key: ScopeKey
    kind: VersionKind
    source_id: str
    authority_class: str
    valid_from: dt.date | None
    valid_to: dt.date | None
    supersedes: tuple[str, ...]
    requires_evidence: tuple[str, ...]
    chunk_refs: tuple[ChunkRef, ...]
    value: Any = None
    label: str = ""
    source_locator: str = ""
    published_at: dt.date | None = None

    def is_effective_at(self, as_of: dt.date) -> bool:
        """``[valid_from, valid_to)`` with unknown bounds meaning unbounded.

        An unknown ``valid_from`` can never be shown to be effective; the
        resolver refuses to invent a start date.
        """
        if self.valid_from is None:
            return False
        if as_of < self.valid_from:
            return False
        return self.valid_to is None or as_of < self.valid_to


@dataclass(frozen=True)
class AuthorityPolicy:
    """A caller-configured partial order over ``authority_class`` values.

    Only classes listed in ``ranking`` are trusted; anything else is
    ``untrusted_source``. ``commentary`` never determines a rule even if the
    ranking places it high, because the version ``kind`` is checked separately.
    """

    version: str
    ranking: tuple[str, ...]

    def rank(self, authority_class: str) -> int | None:
        try:
            return self.ranking.index(authority_class)
        except ValueError:
            return None


@dataclass(frozen=True)
class Catalog:
    """An immutable, validated catalog snapshot.

    ``fingerprint`` is a hash of the normalized content, so a cache key or a
    run record can name the exact catalog state that produced an answer.
    ``chunk_index`` maps a chunk ID to every version whose ``chunk_refs`` name
    it; it is derived, so it is excluded from ``fingerprint``.
    """

    revision: str
    policy: AuthorityPolicy
    documents: dict[str, CatalogDocument]
    versions: dict[str, CatalogVersion]
    by_scope: dict[ScopeKey, tuple[str, ...]]
    superseded_by: dict[str, tuple[str, ...]]
    fingerprint: str = ""
    source_path: str = ""
    notes: dict[str, Any] = field(default_factory=dict)
    chunk_index: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def versions_for_scope(self, scope: ScopeKey) -> tuple[CatalogVersion, ...]:
        return tuple(self.versions[vid] for vid in self.by_scope.get(scope, ()))

    def versions_for_chunk(self, chunk_id: str) -> tuple[CatalogVersion, ...]:
        return tuple(self.versions[vid] for vid in self.chunk_index.get(chunk_id, ()))

    def chain_to(self, version_id: str) -> tuple[str, ...]:
        """Every catalogue version this one supersedes, directly or transitively.

        Used by expansion: a seed amendment must drag in the rule it amends,
        however many links deep. Cycles are rejected at load time, so this
        terminates.
        """
        seen: list[str] = []
        stack = list(self.versions[version_id].supersedes)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.append(current)
            stack.extend(self.versions[current].supersedes)
        return tuple(seen)

    def superseding_versions(self, version_id: str) -> tuple[str, ...]:
        return self.superseded_by.get(version_id, ())


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scope_sort_key(scope_map: dict[str, str] | ScopeKey) -> tuple[str, ...]:
    key = scope_map if isinstance(scope_map, ScopeKey) else ScopeKey.from_mapping(scope_map)
    return tuple(getattr(key, name) for name in SCOPE_FIELDS)


def validate_source_id(value: Any) -> str:
    if not isinstance(value, str) or not _SOURCE_ID.match(value):
        raise CatalogError(
            "missing_metadata",
            "source_id must be a short identifier (letters, digits, . _ : -)",
            value=value,
        )
    return value
