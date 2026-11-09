"""Evidence groups, group ranking, and the all-or-nothing group budget.

An evidence group is the set of spans needed to state one scope's answer
*correctly*: the selected version plus everything that interprets it. It is
never split (see "Evidence Groups" in
``docs/design/EvidencePolicyContract.md``):

- the amendment alone is not enough ("article 8 changes 180 to 270" does not
  say which rule it changes), so the rule it supersedes travels with it;
- the old sentence travels with it, explicitly labelled as replaced, instead
  of being dropped -- dropping it would leave the reader unable to tell that
  the number came from a change;
- spans, not whole chunks: a chunk that also holds an untouched article keeps
  that article available for its own scope.

The budget is measured on rendered group text; later prompt overhead is not
included (see the contract's known limitations). A group either fits whole or
the answer is
``insufficient_context_budget``. Trimming a group to fit means answering from
a partial chain, which is the failure being fixed -- so it is not offered.

Between groups, ranking is relevance *within* the resolved status. The
resolver's decision is never revisited here: authority is not a score and is
not added to a relevance number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .binding import Binding
from .contracts import Catalog
from .resolver import Resolution, ScopeResolution

OLD_SPAN_LABEL = "superseded_for_current_answer"
SELECTED_SPAN_LABEL = "applies_at_as_of"


@dataclass(frozen=True)
class EvidenceSpan:
    version_id: str
    role: str  # SELECTED_SPAN_LABEL | OLD_SPAN_LABEL
    chunk_id: str
    doc_id: str
    file_path: str
    span: tuple[int, int]
    text: str
    content_sha256: str
    source_locator: str = ""

    @property
    def citation_key(self) -> str:
        return f"{self.chunk_id}#{self.span[0]}-{self.span[1]}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "role": self.role,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "span": list(self.span),
            "content_sha256": self.content_sha256,
            "source_locator": self.source_locator,
            "citation_key": self.citation_key,
        }


@dataclass(frozen=True)
class EvidenceGroup:
    scope_key: dict[str, str]
    selected: str
    versions: tuple[str, ...]
    spans: tuple[EvidenceSpan, ...]
    relevance: float | None = None
    score_source: str = "not_observed"

    @property
    def group_id(self) -> str:
        return f"G::{self.selected}"

    def render(self) -> str:
        lines = [
            f"### Evidence group {self.group_id}",
            f"scope: {json.dumps(self.scope_key, ensure_ascii=False, sort_keys=True)}",
            f"selected_version: {self.selected}",
        ]
        for version_id in self.versions:
            version_spans = [
                span for span in self.spans if span.version_id == version_id
            ]
            lines.append(f"[{version_id}]")
            for span in version_spans:
                lines.append(f"  ({span.role}) {span.citation_key}: {span.text}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "scope_key": self.scope_key,
            "selected": self.selected,
            "versions": list(self.versions),
            "relevance": self.relevance,
            "score_source": self.score_source,
            "spans": [span.as_dict() for span in self.spans],
        }


class GroupError(RuntimeError):
    """A group cannot be built. Never silently shrink the group."""


@dataclass(frozen=True)
class BudgetDecision:
    included: tuple[EvidenceGroup, ...]
    excluded: tuple[EvidenceGroup, ...]
    tokens: dict[str, int]
    available_tokens: int
    status: str  # "fits" | "insufficient_context_budget"
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "included": [g.group_id for g in self.included],
            "excluded": [g.group_id for g in self.excluded],
            "tokens": self.tokens,
            "available_tokens": self.available_tokens,
            "status": self.status,
            "detail": self.detail,
        }


def build_groups(
    catalog: Catalog,
    resolution: Resolution,
    binding: Binding,
) -> tuple[EvidenceGroup, ...]:
    """One group per resolved scope, or raise if a group is incomplete."""
    groups: list[EvidenceGroup] = []
    for scope in resolution.scopes:
        if not scope.resolved:
            continue
        groups.append(_build_group(catalog, scope, binding))
    return tuple(groups)


def _build_group(
    catalog: Catalog, scope: ScopeResolution, binding: Binding
) -> EvidenceGroup:
    selected = scope.selected[0]
    problems = binding.problems_for(scope.required_evidence)
    if problems:
        raise GroupError(
            "; ".join(f"{p.version_id}: {p.reason}: {p.detail}" for p in problems)
        )
    spans: list[EvidenceSpan] = []
    for version_id in scope.required_evidence:
        version = catalog.versions[version_id]
        bound = binding.spans.get(version_id, ())
        if not bound:
            raise GroupError(f"{version_id} has no bound span")
        role = SELECTED_SPAN_LABEL if version_id == selected else OLD_SPAN_LABEL
        for span in bound:
            spans.append(
                EvidenceSpan(
                    version_id=version_id,
                    role=role,
                    chunk_id=span.chunk_id,
                    doc_id=span.doc_id,
                    file_path=span.file_path,
                    span=span.span,
                    text=span.text,
                    content_sha256=span.content_sha256,
                    source_locator=version.source_locator,
                )
            )
    spans.sort(key=_span_order)
    return EvidenceGroup(
        scope_key=scope.scope_key.as_dict(),
        selected=selected,
        versions=tuple(scope.required_evidence),
        spans=tuple(spans),
    )


def _span_order(span: EvidenceSpan) -> tuple[int, str, int]:
    """Selected spans first, then the replaced wording they modified."""
    role_rank = 0 if span.role == SELECTED_SPAN_LABEL else 1
    return (role_rank, span.chunk_id, span.span[0])


def rank_groups(
    groups: tuple[EvidenceGroup, ...],
    *,
    relevance_of: Callable[[EvidenceGroup], tuple[float, str]] | None = None,
) -> tuple[EvidenceGroup, ...]:
    """Order groups by relevance *within* their resolved scope.

    ``relevance_of`` returns ``(score, score_source)``. A group with no score
    keeps its catalog order and is marked ``not_observed`` -- it is not given a
    made-up zero that would look like a measurement.
    """
    scored: list[tuple[int, float, int, EvidenceGroup, str]] = []
    for index, group in enumerate(groups):
        if relevance_of is None:
            scored.append((1, 0.0, index, group, "not_observed"))
            continue
        score, source = relevance_of(group)
        scored.append((0, -float(score), index, group, source))
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(
        EvidenceGroup(
            scope_key=group.scope_key,
            selected=group.selected,
            versions=group.versions,
            spans=group.spans,
            relevance=None if source == "not_observed" else -score,
            score_source=source,
        )
        for _, score, _, group, source in scored
    )


def apply_budget(
    groups: tuple[EvidenceGroup, ...],
    *,
    available_tokens: int,
    count_tokens: Callable[[str], int],
) -> BudgetDecision:
    """Keep whole groups in rank order until the next one no longer fits.

    A group that does not fit is excluded, and if any excluded group belongs to
    a scope whose decision is `resolved`, the outcome is
    ``insufficient_context_budget`` -- the request does not silently answer
    from the groups that happened to fit.
    """
    if available_tokens < 0:
        raise ValueError("available_tokens must be >= 0")
    included: list[EvidenceGroup] = []
    excluded: list[EvidenceGroup] = []
    tokens: dict[str, int] = {}
    used = 0
    for group in groups:
        group_tokens = count_tokens(group.render())
        tokens[group.group_id] = group_tokens
        if used + group_tokens <= available_tokens:
            included.append(group)
            used += group_tokens
        else:
            excluded.append(group)
    if excluded:
        status = "insufficient_context_budget"
        detail = (
            f"{len(excluded)} evidence group(s) do not fit in {available_tokens} "
            f"tokens; a partial group would answer from an incomplete chain"
        )
    else:
        status = "fits"
        detail = ""
    return BudgetDecision(
        included=tuple(included),
        excluded=tuple(excluded),
        tokens=tokens,
        available_tokens=available_tokens,
        status=status,
        detail=detail,
    )


@dataclass(frozen=True)
class PromptBundle:
    text: str
    citations: tuple[dict[str, Any], ...]
    tokens: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


def render_prompt(groups: tuple[EvidenceGroup, ...], *, instructions: str = "") -> str:
    """Render the evidence block exactly as the model will receive it.

    Entities and relationships are deliberately absent: a KG description
    merges what every version said and cannot prove which version a sentence
    came from, so under this policy it is not sent.

    ``instructions`` is the operator-supplied block that explains what a
    version, an applicable date, a supersession and a conflict mean. It is
    injected here rather than into the system prompt so that a caller-supplied
    ``system_prompt`` cannot drop it and leave the labels unexplained.
    """
    header = (
        "The following evidence groups are the only source for the answer.\n"
        "Each group states the rule that applies at the requested date first, "
        "then the replaced wording it modified.\n"
        f"Spans labelled {OLD_SPAN_LABEL} are historical: never present them as "
        "the current rule.\n"
        "Cite every claim with the citation key in parentheses.\n"
    )
    if instructions.strip():
        header = f"{header}\n{instructions.strip()}\n"
    body = "\n\n".join(group.render() for group in groups)
    return f"{header}\n{body}"


def citations_of(groups: tuple[EvidenceGroup, ...]) -> tuple[dict[str, Any], ...]:
    seen: dict[str, dict[str, Any]] = {}
    for group in groups:
        for span in group.spans:
            if span.citation_key not in seen:
                seen[span.citation_key] = {
                    "citation_key": span.citation_key,
                    "version_id": span.version_id,
                    "role": span.role,
                    "chunk_id": span.chunk_id,
                    "doc_id": span.doc_id,
                    "file_path": span.file_path,
                    "span": list(span.span),
                    "source_locator": span.source_locator,
                }
    return tuple(seen.values())


def verify_citations(
    cited: list[str], groups: tuple[EvidenceGroup, ...]
) -> dict[str, Any]:
    """Citation support: every cited key must exist in the rendered groups.

    This proves the *pointer* resolves and the span carries the claim's source
    text; it does not prove the model's reasoning is correct.
    """
    known = {span.citation_key for group in groups for span in group.spans}
    missing = [key for key in cited if key not in known]
    return {
        "supported": not missing and bool(cited),
        "unknown_citations": missing,
        "cited": cited,
    }
