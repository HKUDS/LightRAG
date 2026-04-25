from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanCandidate:
    name: str
    implementation_cost: int
    latency_risk: int
    security_risk: int
    repo_alignment: int
    extensibility: int
    validation_strength: int


@dataclass(frozen=True)
class PlanScorecard:
    total: int
    rationale: str


def score_plan(candidate: PlanCandidate) -> PlanScorecard:
    positive = (
        candidate.repo_alignment
        + candidate.extensibility
        + candidate.validation_strength
    )
    negative = (
        candidate.implementation_cost + candidate.latency_risk + candidate.security_risk
    )
    total = positive - negative
    return PlanScorecard(
        total=total,
        rationale=(
            "Score = repo alignment + extensibility + validation "
            "minus implementation, latency, and security risk."
        ),
    )
