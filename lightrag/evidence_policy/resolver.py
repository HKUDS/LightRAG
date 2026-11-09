"""Pure resolution of "which clause version applies at a given date".

The resolver is a pure function of ``(catalog, scope, as_of)``: no storage, no
network, no clocks. Rules, in the order they are applied (README section 5):

1. **Time first, authority second.** A version enters the decision only when
   ``as_of`` falls inside its own ``[valid_from, valid_to)``. A revision that
   takes effect later never displaces the rule in force now.
2. **Supersession is an approved event, not a date.** A previous version stops
   applying at the EARLIER of its approved successor's effective date and its
   own ``valid_to``. If the successor later expires or is repealed, the
   previous version does not come back: the scope can end up with no
   applicable rule.
3. **An edition edits another edition.** An amendment that supersedes another
   version and carries no ``value`` of its own is an *edition*: it stays inside
   the supersession chain and never competes as a second answer. An amendment
   with its own value (resolution CASE-03's fixture) is a stand-alone version
   and competes only once it is effective.
4. **Authority only decides between contradictions with no approved relation.**
   Two stand-alone versions of the same scope, both effective, from different
   sources, with neither in the other's supersession chain and holding
   different answers, are an ``unresolved_conflict`` when they share an
   authority rank. The resolver never breaks that tie on publication date and
   never compares across scopes.

Every version of every evaluated scope receives exactly one reason code, and
the result names every scope it was asked about -- including the ones with
nothing to say.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .contracts import Catalog, CatalogVersion, ReasonCode, ResolutionStatus, ScopeKey

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VersionVerdict:
    version_id: str
    reason: ReasonCode
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"version_id": self.version_id, "reason": self.reason}
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass(frozen=True)
class ScopeResolution:
    scope_key: ScopeKey
    status: ResolutionStatus
    selected: tuple[str, ...]
    required_evidence: tuple[str, ...]
    conflict_versions: tuple[str, ...]
    verdicts: tuple[VersionVerdict, ...]
    detail: str = ""
    candidates: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    def reason_map(self) -> dict[str, str]:
        return {verdict.version_id: verdict.reason for verdict in self.verdicts}

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_key": self.scope_key.as_dict(),
            "status": self.status,
            "selected": list(self.selected),
            "required_evidence": list(self.required_evidence),
            "conflict_versions": list(self.conflict_versions),
            "verdicts": [verdict.as_dict() for verdict in self.verdicts],
            "detail": self.detail,
            "candidates": list(self.candidates),
        }


@dataclass(frozen=True)
class Resolution:
    as_of: dt.date
    status: ResolutionStatus
    scopes: tuple[ScopeResolution, ...]
    detail: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"

    def selected_versions(self) -> list[str]:
        """Versions that answer the request. Empty unless the request resolved.

        A request where one scope failed is not ``resolved``, and a partially
        verified request must not hand back the scopes that did work: the
        caller would answer from a mix of verified and unverified clauses.
        """
        if not self.resolved:
            return []
        return [vid for scope in self.scopes for vid in scope.selected]

    def required_evidence(self) -> list[str]:
        if not self.resolved:
            return []
        ordered: list[str] = []
        for scope in self.scopes:
            if not scope.resolved:
                continue
            for vid in scope.required_evidence:
                if vid not in ordered:
                    ordered.append(vid)
        return ordered

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "status": self.status,
            "detail": self.detail,
            "scopes": [scope.as_dict() for scope in self.scopes],
            "selected_versions": self.selected_versions(),
            "required_evidence": self.required_evidence(),
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def resolve(
    catalog: Catalog,
    scopes: list[ScopeKey],
    as_of: dt.date,
    *,
    seed_version_ids: set[str] | None = None,
) -> Resolution:
    """Resolve every clause scope attached to a request.

    A scope the caller names explicitly is resolved even when nothing recalled
    a chunk for it -- that is what makes a revision which was never retrieved
    still reachable. Only ``resolved`` scopes contribute evidence; a request
    where any scope fails to resolve is not ``resolved`` as a whole, because a
    blended answer over a partially verified request is the failure this
    policy exists to prevent.
    """
    seeds = set(seed_version_ids or set())
    if not scopes:
        return Resolution(
            as_of=as_of,
            status="insufficient_evidence",
            scopes=(),
            detail="no clause scope was identified in the request",
        )
    unique_scopes = list(dict.fromkeys(scopes))
    scope_results = [
        resolve_scope(catalog, scope, as_of, seeds=seeds) for scope in unique_scopes
    ]
    statuses = {scope.status for scope in scope_results}
    if len(statuses) == 1:
        overall: ResolutionStatus = scope_results[0].status
    else:
        for candidate in (
            "unresolved_conflict",
            "unknown_time",
            "mapping_ambiguous",
            "insufficient_evidence",
            "no_applicable_rule",
            "resolved",
        ):
            if candidate in statuses:
                overall = candidate  # type: ignore[assignment]
                break
        else:  # pragma: no cover - every status is listed above
            overall = "resolved"
    return Resolution(
        as_of=as_of,
        status=overall,
        scopes=tuple(scope_results),
        diagnostics={
            "seed_versions": sorted(seeds),
            "seed_versions_outside_requested_scopes": sorted(
                vid
                for vid in seeds
                if vid not in catalog.versions
                or catalog.versions[vid].scope_key not in set(unique_scopes)
            ),
        },
    )


def resolve_scope(
    catalog: Catalog,
    scope: ScopeKey,
    as_of: dt.date,
    *,
    seeds: set[str] | None = None,
) -> ScopeResolution:
    """Decide which versions of one scope apply at ``as_of``."""
    seeds = seeds or set()
    candidate_ids = catalog.by_scope.get(scope, ())
    if not candidate_ids:
        return ScopeResolution(
            scope_key=scope,
            status="insufficient_evidence",
            selected=(),
            required_evidence=(),
            conflict_versions=(),
            verdicts=(),
            detail="the catalog holds no version for this scope",
        )

    versions = [catalog.versions[vid] for vid in candidate_ids]
    verdicts: dict[str, VersionVerdict] = {}

    ranked = {v.version_id: catalog.policy.rank(v.authority_class) for v in versions}
    if all(rank is None for rank in ranked.values()):
        for v in versions:
            verdicts[v.version_id] = VersionVerdict(
                v.version_id,
                "untrusted_source",
                f"authority_class {v.authority_class!r} is not in the policy "
                f"ranking {list(catalog.policy.ranking)}",
            )
        return _finish(scope, versions, verdicts, "insufficient_evidence")

    # -- 0. sources of truth for this scope -------------------------------
    # An *edition* (an amendment that supersedes something and carries no value
    # of its own) is not a second answer, but it IS the head of the chain it
    # edits: "article 8 changes 180 to 270" is what is in force, and the rule it
    # replaced travels with it as required evidence. Excluding editions from the
    # roots made every amended clause resolve to "no applicable rule", which is
    # the case this policy exists to answer.
    if not any(v.kind != "commentary" for v in versions):
        for v in versions:
            verdicts[v.version_id] = VersionVerdict(
                v.version_id,
                "untrusted_source",
                "only commentary is catalogued for this scope; commentary never "
                "determines a rule",
            )
        return _finish(scope, versions, verdicts, "insufficient_evidence")

    # -- 1. time -----------------------------------------------------------
    effective: list[CatalogVersion] = []
    for version in versions:
        if catalog.policy.rank(version.authority_class) is None:
            # A source the policy does not rank cannot establish this scope,
            # however confidently its own text claims authority. It stays out
            # of candidacy instead of becoming a second "answer" that authority
            # then has to argue with.
            verdicts[version.version_id] = VersionVerdict(
                version.version_id,
                "untrusted_source",
                f"authority_class {version.authority_class!r} is not in the "
                f"policy ranking {list(catalog.policy.ranking)}",
            )
            continue
        # Always recorded: a version that is merely ahead of its window must
        # not resurface as the answer if its evaluator is not effective either.
        verdict = _time_verdict(version, as_of)
        verdicts[version.version_id] = verdict
        if verdict.reason == "active" and version.kind != "repeal":
            effective.append(version)

    # -- 2. an unknown start date blocks a definitive answer ---------------
    # A version of this scope whose effective date is unknown may or may not be
    # in force, so no other version can be asserted as *the* rule -- but only
    # when the request is actually about it. "About it" means the retriever
    # surfaced it (``seeds``) or nothing dated applies at all: a catalog-wide
    # undated notice that no query ever returns would otherwise make every
    # clause in the workspace unanswerable. The verdict is still recorded, so
    # the reason stays visible instead of disappearing.
    blocking_unknown = [
        v
        for v in versions
        if verdicts[v.version_id].reason == "unknown_time"
        and v.kind != "commentary"
        and (v.version_id in seeds or not effective)
        and not _touch_only_known_dead(catalog, v, verdicts)
    ]
    if blocking_unknown:
        return _finish(
            scope,
            versions,
            verdicts,
            "unknown_time",
            detail="unknown effective date for "
            + ", ".join(sorted(v.version_id for v in blocking_unknown)),
        )

    # -- 3. supersession and repeal among the effective versions ----------
    chain_roots = _chain_roots(catalog, effective, verdicts, as_of)
    repealed_by: dict[str, CatalogVersion] = {}
    for root in chain_roots:
        family = [root] + [
            catalog.versions[vid] for vid in catalog.chain_to(root.version_id)
        ]
        for member in family:
            repeal = _repeal_for(catalog, member, as_of)
            if repeal is not None and member.version_id not in repealed_by:
                repealed_by[member.version_id] = repeal

    for version in versions:
        if version.kind == "repeal" and version.version_id not in verdicts:
            verdicts[version.version_id] = VersionVerdict(
                version.version_id,
                "not_yet_effective"
                if version.valid_from is None or version.valid_from > as_of
                else "active",
            )

    for member_id, repeal in repealed_by.items():
        if member_id in verdicts and verdicts[member_id].reason == "active":
            verdicts[member_id] = VersionVerdict(
                member_id,
                "repealed",
                f"repealed by {repeal.version_id} from {repeal.valid_from}",
            )

    if not chain_roots:
        # Every version of the scope is out of its window (expired, or ahead of
        # it) or was displaced by a successor. The scope is closed: no rule
        # applies. An unknown date already returned above, so nothing here can
        # be revived by a version whose validity is merely undated.
        return _finish(scope, versions, verdicts, "no_applicable_rule")

    # -- 4. authority and conflict among surviving chain roots ------------
    conflict_ids = _apply_authority(catalog, scope, chain_roots, verdicts)
    if conflict_ids:
        return _finish(
            scope,
            versions,
            verdicts,
            "unresolved_conflict",
            conflict_versions=tuple(conflict_ids),
        )

    active_roots = [r for r in chain_roots if verdicts[r.version_id].reason == "active"]
    if not active_roots:
        # Every root was repealed: an explicit termination, not missing
        # evidence. The scope is closed and the previous version is not
        # restored (a repeal of the successor never revives its predecessor).
        return _finish(
            scope,
            versions,
            verdicts,
            "no_applicable_rule",
            detail="every applicable version of this scope has been repealed",
        )

    selected = _editorial_winner(active_roots, catalog)
    required = [selected.version_id]
    for dependency in reversed(_closure(catalog, selected.version_id)):
        if dependency not in required:
            required.append(dependency)
    # Seeds that sit on the selected chain are named for the record, in
    # chain order, but they do not change the answer.
    for vid in sorted(seeds & set(required), key=required.index):
        if vid not in required:  # pragma: no cover - kept for clarity
            required.append(vid)

    if len(active_roots) > 1:
        # Same authority class, same effective clause, no approved relation and
        # no different answer: the catalog cannot tell which one establishes
        # the scope, and guessing would be a silent authority decision.
        for root in active_roots:
            if root.version_id == selected.version_id:
                continue
            verdicts[root.version_id] = VersionVerdict(
                root.version_id,
                "unresolved_conflict",
                f"equally authoritative as {selected.version_id} with no "
                "approved supersession",
            )
        return _finish(
            scope,
            versions,
            verdicts,
            "unresolved_conflict",
            conflict_versions=tuple(sorted(r.version_id for r in active_roots)),
        )
    verdicts[selected.version_id] = VersionVerdict(selected.version_id, "active")

    return ScopeResolution(
        scope_key=scope,
        status="resolved",
        selected=(selected.version_id,),
        required_evidence=tuple(required),
        conflict_versions=(),
        verdicts=_ordered_verdicts(candidate_ids, verdicts),
        candidates=tuple(candidate_ids),
    )


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def _time_verdict(version: CatalogVersion, as_of: dt.date) -> VersionVerdict:
    """Whether this version's own window contains ``as_of``.

    Commentary never determines a rule, and an unknown start date is never
    inferred from a publication date: the version is reported as unknown rather
    than treated as "not yet effective" or "already effective".
    """
    if version.kind == "commentary":
        return VersionVerdict(
            version.version_id, "untrusted_source", "commentary never determines a rule"
        )
    if version.valid_from is None:
        return VersionVerdict(
            version.version_id,
            "unknown_time",
            "valid_from is unknown; a start date is never inferred",
        )
    if as_of < version.valid_from:
        return VersionVerdict(version.version_id, "not_yet_effective")
    if version.valid_to is not None and as_of >= version.valid_to:
        return VersionVerdict(
            version.version_id,
            "expired",
            f"valid_to {version.valid_to.isoformat()} has already been reached",
        )
    return VersionVerdict(version.version_id, "active")


def _chain_roots(
    catalog: Catalog,
    effective: list[CatalogVersion],
    verdicts: dict[str, VersionVerdict],
    as_of: dt.date,
) -> list[CatalogVersion]:
    """Drop effective versions that an effective successor has displaced."""
    cut: dict[str, tuple[dt.date, CatalogVersion]] = {}
    for successor in catalog.versions.values():
        if successor.valid_from is None:
            continue
        for previous in successor.supersedes:
            previous_version = catalog.versions[previous]
            previous_end = previous_version.valid_to
            when = (
                previous_end
                if previous_end is not None and previous_end < successor.valid_from
                else successor.valid_from
            )
            current = cut.get(previous)
            if current is None or (when, successor.version_id) < (
                current[0],
                current[1].version_id,
            ):
                cut[previous] = (when, successor)

    roots: list[CatalogVersion] = []
    for version in effective:
        edge = cut.get(version.version_id)
        if edge is not None and as_of >= edge[0]:
            when, successor = edge
            verdicts[version.version_id] = VersionVerdict(
                version.version_id,
                "superseded",
                f"superseded by {successor.version_id} from {when.isoformat()}",
            )
            continue
        if version.kind == "repeal":
            # A repeal decides only through the versions it names.
            continue
        roots.append(version)
    return roots


def _repeal_for(
    catalog: Catalog, member: CatalogVersion, as_of: dt.date
) -> CatalogVersion | None:
    """The earliest repeal of ``member`` that has taken effect at ``as_of``."""
    candidates: list[CatalogVersion] = []
    for successor_id in catalog.superseding_versions(member.version_id):
        successor = catalog.versions[successor_id]
        if successor.kind != "repeal" or successor.valid_from is None:
            continue
        if catalog.policy.rank(successor.authority_class) is None:
            continue
        if successor.valid_from > as_of:
            continue
        candidates.append(successor)
    if not candidates:
        return None
    return min(candidates, key=lambda v: (v.valid_from, v.version_id))


# ---------------------------------------------------------------------------
# Authority
# ---------------------------------------------------------------------------


def _apply_authority(
    catalog: Catalog,
    scope: ScopeKey,
    roots: list[CatalogVersion],
    verdicts: dict[str, VersionVerdict],
) -> list[str]:
    """Clear contradictions that authority can settle; report the rest."""
    conflicts: list[str] = []
    for i, first in enumerate(roots):
        for second in roots[i + 1 :]:
            first_rank = catalog.policy.rank(first.authority_class)
            second_rank = catalog.policy.rank(second.authority_class)
            if (
                first_rank is not None
                and second_rank is not None
                and first_rank != second_rank
            ):
                # A higher authority class is not a contradiction: the lower
                # one simply does not establish this scope.
                loser, winner = (
                    (second, first) if first_rank < second_rank else (first, second)
                )
                if verdicts[loser.version_id].reason == "active":
                    verdicts[loser.version_id] = VersionVerdict(
                        loser.version_id,
                        "superseded",
                        f"same-scope {winner.version_id} holds a higher authority "
                        f"class ({winner.authority_class} over {loser.authority_class}); "
                        "the catalog records no explicit supersession",
                    )
                continue
            if _contradicts(catalog, first, second):
                for version in (first, second):
                    verdicts[version.version_id] = VersionVerdict(
                        version.version_id,
                        "unresolved_conflict",
                        f"contradicts {second.version_id if version is first else first.version_id} "
                        "with no approved supersession and no authority difference",
                    )
                conflicts.extend([first.version_id, second.version_id])
    return list(dict.fromkeys(conflicts))


def _contradicts(
    catalog: Catalog, first: CatalogVersion, second: CatalogVersion
) -> bool:
    if first.source_id == second.source_id:
        return False
    if first.value is None or second.value is None:
        return False
    if catalog.policy.rank(first.authority_class) is None:
        return False
    if catalog.policy.rank(second.authority_class) is None:
        return False
    return first.value != second.value


def _editorial_winner(roots: list[CatalogVersion], catalog: Catalog) -> CatalogVersion:
    """Highest authority class; then the most recent ``valid_from``.

    ``valid_from`` breaks ties between editions of the *same* instrument that
    do not supersede each other. Publication date and retrieval score are never
    consulted.
    """

    def rank_of(version: CatalogVersion) -> int:
        rank = catalog.policy.rank(version.authority_class)
        return rank if rank is not None else len(catalog.policy.ranking)

    best = min(rank_of(version) for version in roots)
    pool = [version for version in roots if rank_of(version) == best]
    return max(pool, key=lambda v: (v.valid_from, v.version_id))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Chain:
    """Supersession-chain membership, computed once per request."""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog
        self._editions: dict[str, bool] = {}

    def is_edition(self, version_id: str) -> bool:
        if version_id not in self._editions:
            version = self._catalog.versions[version_id]
            self._editions[version_id] = (
                version.kind == "amendment"
                and bool(version.supersedes)
                and version.value is None
            )
        return self._editions[version_id]


def _touch_only_known_dead(
    catalog: Catalog, version: CatalogVersion, verdicts: dict[str, VersionVerdict]
) -> bool:
    """True when an unknown-dated version only edits versions already known to
    be out of force -- it cannot revive them, so it is not worth a veto."""
    if not version.supersedes:
        return False
    return all(
        verdicts[vid].reason in ("expired", "superseded", "repealed")
        for vid in version.supersedes
        if vid in verdicts
    )


def _closure(catalog: Catalog, version_id: str) -> list[str]:
    """Everything needed to interpret ``version_id``: the editions it replaces,
    transitively, plus explicit ``requires_evidence``; oldest first."""
    ordered: list[str] = []
    seen: set[str] = set()
    queue = [version_id]
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if current != version_id:
            ordered.append(current)
        version = catalog.versions[current]
        queue.extend(version.supersedes)
        queue.extend(version.requires_evidence)
    ordered.sort(key=lambda vid: (catalog.versions[vid].valid_from or dt.date.min, vid))
    return ordered


def _ordered_verdicts(
    candidate_ids: tuple[str, ...], verdicts: dict[str, VersionVerdict]
) -> tuple[VersionVerdict, ...]:
    return tuple(
        verdicts.get(vid) or VersionVerdict(vid, "unknown", "no verdict recorded")
        for vid in candidate_ids
    )


def _finish(
    scope: ScopeKey,
    versions: list[CatalogVersion],
    verdicts: dict[str, VersionVerdict],
    status: ResolutionStatus,
    *,
    conflict_versions: tuple[str, ...] = (),
    detail: str = "",
) -> ScopeResolution:
    candidate_ids = tuple(v.version_id for v in versions)
    return ScopeResolution(
        scope_key=scope,
        status=status,
        selected=(),
        required_evidence=(),
        conflict_versions=conflict_versions,
        verdicts=_ordered_verdicts(candidate_ids, verdicts),
        detail=detail,
        candidates=candidate_ids,
    )
