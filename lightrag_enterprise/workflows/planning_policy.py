from __future__ import annotations

from dataclasses import dataclass

from .plan_scorecard import PlanCandidate, score_plan


@dataclass(frozen=True)
class PlanningPolicy:
    max_candidates: int = 3
    expose_private_reasoning: bool = False
    require_audit_record: bool = True


def select_plan(
    candidates: list[PlanCandidate], policy: PlanningPolicy | None = None
) -> PlanCandidate:
    """Select a plan using a bounded scorecard, not exposed chain-of-thought."""

    effective_policy = policy or PlanningPolicy()
    if not candidates:
        raise ValueError("At least one plan candidate is required")
    bounded = candidates[: effective_policy.max_candidates]
    return max(bounded, key=lambda candidate: score_plan(candidate).total)
