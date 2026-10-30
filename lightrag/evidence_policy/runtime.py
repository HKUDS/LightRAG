"""Request-scoped orchestration of the evidence policy.

This module is the only place that turns a ``QueryParam`` into a decision. It
is deliberately independent of the query pipeline: it takes seed chunk IDs and
a chunk reader, and returns a typed outcome. ``operate.py`` calls it from both
retrieval entry points so ``naive`` and ``mix`` cannot drift apart.

Order of operations, and why (see "Resolution Order" in
``docs/design/EvidencePolicyContract.md``):

1. Freeze the date once, at request start.
2. Identify clause scopes from explicit request fields *or* from the chunks the
   retriever actually returned.
3. Resolve which version applies -- never which chunk ranked highest.
4. Bind the whole evidence closure to real chunks, fail loud on a missing or
   changed span.
5. Build groups, rank within the resolved set, and apply an all-or-nothing
   budget measured on rendered group text, before final prompt overhead.

Anything that is not ``resolved`` returns its own status. It never falls back
to ordinary semantic retrieval, because that is the behaviour being fixed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .binding import Binding, ChunkReader, StorageBackendError, bind_versions
from .contracts import (
    Catalog,
    RequestError,
    ScopeKey,
    parse_full_date,
    sha256_text,
)
from .groups import (
    BudgetDecision,
    EvidenceGroup,
    GroupError,
    apply_budget,
    build_groups,
    citations_of,
    render_prompt,
)
from .resolver import Resolution, resolve

#: Policy versions this build implements. A request may only name one of these.
POLICY_VERSIONS: tuple[str, ...] = ("authority-v1",)

#: Modes the first version covers. ``local``/``global``/``hybrid`` are refused
#: explicitly rather than silently answered by the old semantics.
SUPPORTED_MODES: tuple[str, ...] = ("naive", "mix")

# Initial limits. They are configuration, not a claim of universal optimality;
# every request that hits one is reported instead of truncated.
MAX_SCOPES = 8
MAX_VERSIONS_PER_SCOPE = 16
MAX_EXTRA_CHUNKS = 64


@dataclass(frozen=True)
class EvidencePolicyConfig:
    """A validated, request-scoped policy request.

    Frozen at request start so a long-running query cannot observe a different
    date, a different catalog, or a caller mutating ``QueryParam`` midway.
    """

    version: str
    mode: str
    as_of: dt.date
    as_of_source: str
    scopes: tuple[ScopeKey, ...] = ()
    available_tokens: int | None = None
    max_scopes: int = MAX_SCOPES
    max_versions_per_scope: int = MAX_VERSIONS_PER_SCOPE
    max_extra_chunks: int = MAX_EXTRA_CHUNKS
    require_doc_match: bool = True


@dataclass
class EvidenceOutcome:
    """What the pipeline needs to answer, or to explain why it will not."""

    status: str
    detail: str = ""
    context_text: str = ""
    citations: tuple[dict[str, Any], ...] = ()
    groups: tuple[EvidenceGroup, ...] = ()
    resolution: Resolution | None = None
    binding: Binding | None = None
    budget: BudgetDecision | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    def raw_data(
        self, *, mode: str, keywords: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Project the outcome into the public ``data``/``metadata`` shape.

        ``entities`` and ``relationships`` are empty on purpose: under this
        policy the answer context is the evidence groups, so exposing a merged
        graph description would advertise knowledge the answer never used.
        """
        chunks: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        reference_ids: dict[str, str] = {}
        for group in self.groups:
            for span in group.spans:
                file_path = span.file_path or "unknown_source"
                if file_path not in reference_ids:
                    reference_ids[file_path] = str(len(reference_ids) + 1)
                    references.append(
                        {
                            "reference_id": reference_ids[file_path],
                            "file_path": file_path,
                        }
                    )
                chunks.append(
                    {
                        "reference_id": reference_ids[file_path],
                        "content": span.text,
                        "file_path": file_path,
                        "chunk_id": span.chunk_id,
                        "citation_key": span.citation_key,
                        "version_id": span.version_id,
                        "role": span.role,
                        "span": list(span.span),
                        "group_id": group.group_id,
                    }
                )
        return {
            "status": "success" if self.resolved else "failure",
            "message": "Query processed successfully"
            if self.resolved
            else self.detail or self.status,
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": chunks,
                "references": references,
            },
            "metadata": {
                "query_mode": mode,
                "keywords": keywords or {"high_level": [], "low_level": []},
                "evidence_policy": self.diagnostics,
            },
        }


def build_policy_config(
    param: Any,
    *,
    catalog: Catalog | None,
    timezone: str = "UTC",
    available_tokens: int | None = None,
    now: dt.datetime | None = None,
) -> EvidencePolicyConfig | None:
    """Validate ``param`` and freeze it. ``None`` means the policy is off.

    Every refusal raises :class:`RequestError` with a status code, so the caller
    can return a typed failure instead of a generic 500 or a silent fallback.
    """
    version = getattr(param, "evidence_policy", None)
    as_of_raw = getattr(param, "as_of", None)
    scope_keys = getattr(param, "scope_keys", None) or []

    if not version:
        if as_of_raw or scope_keys:
            raise RequestError(
                "invalid_request",
                "as_of and scope_keys only apply to evidence_policy requests; "
                "set evidence_policy or drop them",
            )
        return None
    if version not in POLICY_VERSIONS:
        raise RequestError(
            "invalid_request",
            f"unsupported evidence_policy {version!r}; expected one of "
            f"{list(POLICY_VERSIONS)}",
        )
    if catalog is None:
        raise RequestError(
            "invalid_request",
            "evidence_policy requires an evidence catalog on the LightRAG "
            "instance (evidence_catalog=...)",
        )
    mode = getattr(param, "mode", "mix")
    if mode not in SUPPORTED_MODES:
        raise RequestError(
            "unsupported",
            f"evidence_policy {version!r} does not cover mode {mode!r}; "
            f"supported modes are {list(SUPPORTED_MODES)}",
        )
    if getattr(param, "stream", False):
        raise RequestError(
            "unsupported",
            "evidence_policy answers are published only after citation checks, "
            "so this version does not stream",
        )

    if as_of_raw:
        as_of = parse_full_date(as_of_raw, field_name="as_of")
        if as_of is None:
            raise RequestError("invalid_request", "as_of is empty")
        source = "explicit"
    else:
        as_of, source = _current_date(timezone, now)

    scopes = tuple(
        ScopeKey.from_mapping(scope, field_name=f"scope_keys[{index}]")
        for index, scope in enumerate(scope_keys)
    )
    if len(scopes) > MAX_SCOPES:
        raise RequestError(
            "expansion_limit",
            f"{len(scopes)} explicit scopes exceed max_scopes={MAX_SCOPES}",
        )
    if available_tokens is not None and available_tokens < 0:
        raise RequestError("invalid_request", "available_tokens must be >= 0")

    return EvidencePolicyConfig(
        version=version,
        mode=mode,
        as_of=as_of,
        as_of_source=source,
        scopes=scopes,
        available_tokens=available_tokens,
    )


def _current_date(timezone: str, now: dt.datetime | None) -> tuple[dt.date, str]:
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise RequestError(
            "invalid_request", f"unknown timezone {timezone!r} for evidence_policy"
        ) from exc
    moment = now or dt.datetime.now(tz=dt.timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(zone).date(), "resolved_once_at_request_start"


def scopes_from_seeds(catalog: Catalog, seed_chunk_ids: Any) -> tuple[ScopeKey, ...]:
    """Clause identities the retrieved chunks belong to, in retrieval order.

    A chunk may carry spans of several versions of the *same* clause (an
    article that was amended in place) or of different clauses. Each distinct
    scope is returned once; a chunk that maps to no catalog entry contributes
    nothing, which is why an unannotated corpus yields
    ``insufficient_evidence`` rather than a guess.
    """
    scopes: list[ScopeKey] = []
    for chunk_id in seed_chunk_ids or ():
        if not chunk_id:
            continue
        for version in catalog.versions_for_chunk(str(chunk_id)):
            if version.scope_key not in scopes:
                scopes.append(version.scope_key)
    return tuple(scopes)


async def run_evidence_policy(
    config: EvidencePolicyConfig,
    catalog: Catalog,
    *,
    seed_chunk_ids: Any = (),
    reader: ChunkReader,
    instructions: str = "",
    count_tokens: Callable[[str], int] | None = None,
) -> EvidenceOutcome:
    """Resolve, bind, group, budget and render one request."""
    diagnostics: dict[str, Any] = {
        "status": "pending",
        "policy_version": config.version,
        "resolved_as_of": config.as_of.isoformat(),
        "as_of_source": config.as_of_source,
        "catalog_revision": catalog.revision,
        "catalog_fingerprint": catalog.fingerprint,
        "requested_scopes": [scope.as_dict() for scope in config.scopes],
        "seed_chunk_ids": sorted({str(cid) for cid in seed_chunk_ids or () if cid}),
    }
    outcome = EvidenceOutcome(status="pending", diagnostics=diagnostics)

    scopes = list(config.scopes) or list(scopes_from_seeds(catalog, seed_chunk_ids))
    # Which versions the retriever actually surfaced, not just which clauses.
    # The resolver needs this: an undated version that no query ever returned
    # must not block every answer in the workspace, while one that WAS returned
    # means the question may be about a rule whose date is unknown.
    seeded_versions = {
        version.version_id
        for chunk_id in seed_chunk_ids or ()
        if chunk_id
        for version in catalog.versions_for_chunk(str(chunk_id))
    }
    diagnostics["seed_version_ids"] = sorted(seeded_versions)
    diagnostics["resolved_scopes"] = [scope.as_dict() for scope in scopes]
    if not scopes:
        return _fail(
            outcome,
            "insufficient_evidence",
            "no clause scope was named and no retrieved chunk belongs to a "
            "catalog version",
        )
    if len(scopes) > config.max_scopes:
        return _fail(
            outcome,
            "expansion_limit",
            f"{len(scopes)} scopes exceed max_scopes={config.max_scopes}; "
            "the request is not answered from a truncated scope set",
        )
    for scope in scopes:
        versions = catalog.versions_for_scope(scope)
        if not versions:
            return _fail(
                outcome,
                "insufficient_evidence",
                f"{scope.describe()}: the catalog holds no version for this scope",
            )
        if len(versions) > config.max_versions_per_scope:
            return _fail(
                outcome,
                "expansion_limit",
                f"{scope.describe()}: {len(versions)} versions exceed "
                f"max_versions_per_scope={config.max_versions_per_scope}",
            )

    resolution = resolve(
        catalog, scopes, config.as_of, seed_version_ids=seeded_versions
    )
    outcome.resolution = resolution
    diagnostics["resolution"] = resolution.as_dict()
    if not resolution.resolved:
        return _fail(
            outcome,
            resolution.status,
            resolution.detail
            or "; ".join(
                f"{scope.scope_key.describe()}: {scope.status}"
                for scope in resolution.scopes
            ),
        )

    evidence_ids = resolution.required_evidence()
    extra = _non_seed_chunks(catalog, evidence_ids, seed_chunk_ids)
    diagnostics["expanded_chunk_ids"] = extra
    if len(extra) > config.max_extra_chunks:
        return _fail(
            outcome,
            "expansion_limit",
            f"expanding the evidence closure needs {len(extra)} chunks beyond the "
            f"seed set, above max_extra_chunks={config.max_extra_chunks}",
        )

    try:
        binding = await bind_versions(
            catalog, evidence_ids, reader, require_doc_match=config.require_doc_match
        )
    except StorageBackendError as exc:
        outcome.binding = None
        return _fail(outcome, "backend_error", str(exc))
    outcome.binding = binding
    if not binding.ok:
        return _fail(
            outcome,
            binding.problems[0].reason,
            "; ".join(f"{p.version_id}: {p.detail}" for p in binding.problems),
        )

    try:
        groups = build_groups(catalog, resolution, binding)
    except GroupError as exc:
        return _fail(outcome, "missing_evidence", str(exc))
    diagnostics["groups"] = [group.group_id for group in groups]

    counter = count_tokens or (lambda text: max(1, len(text.split())))
    if config.available_tokens is not None:
        budget = apply_budget(
            groups, available_tokens=config.available_tokens, count_tokens=counter
        )
        outcome.budget = budget
        diagnostics["budget"] = budget.as_dict()
        if budget.status != "fits":
            return _fail(outcome, "insufficient_context_budget", budget.detail)
        groups = budget.included

    context_text = (
        f"requested_as_of: {config.as_of.isoformat()}\n"
        "Answer which rule applies on this requested date. Distinguish it from "
        "a source's publication or effective date.\n\n"
        + render_prompt(groups, instructions=instructions)
    )
    outcome.status = "resolved"
    outcome.groups = groups
    outcome.citations = citations_of(groups)
    outcome.context_text = context_text
    diagnostics["status"] = "resolved"
    diagnostics["selected_versions"] = resolution.selected_versions()
    diagnostics["required_evidence"] = evidence_ids
    diagnostics["citation_keys"] = [item["citation_key"] for item in outcome.citations]
    diagnostics["context_tokens"] = counter(context_text)
    diagnostics["context_sha256"] = sha256_text(context_text)
    return outcome


def _fail(outcome: EvidenceOutcome, status: str, detail: str) -> EvidenceOutcome:
    outcome.status = status
    outcome.detail = detail
    outcome.diagnostics["status"] = status
    outcome.diagnostics["detail"] = detail
    return outcome


def _non_seed_chunks(
    catalog: Catalog, evidence_ids: list[str], seed_chunk_ids: Any
) -> list[str]:
    seeded = {str(cid) for cid in seed_chunk_ids or () if cid}
    chunks = {
        ref.chunk_id
        for version_id in evidence_ids
        for ref in catalog.versions[version_id].chunk_refs
    }
    return sorted(chunk for chunk in chunks if chunk not in seeded)


def failure_content(status: str, detail: str) -> str:
    """User-visible refusal. Never a confident answer from partial evidence."""
    reason = detail.strip() or status
    return (
        "Unable to answer under the requested evidence policy "
        f"(status: {status}).\n{reason}"
    )


__all__ = [
    "EvidenceOutcome",
    "EvidencePolicyConfig",
    "MAX_EXTRA_CHUNKS",
    "MAX_SCOPES",
    "MAX_VERSIONS_PER_SCOPE",
    "POLICY_VERSIONS",
    "SUPPORTED_MODES",
    "build_policy_config",
    "failure_content",
    "run_evidence_policy",
    "scopes_from_seeds",
]
