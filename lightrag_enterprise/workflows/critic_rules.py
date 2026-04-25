from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CriticFinding:
    rule: str
    severity: str
    message: str


def evaluate_critic_rules(plan_summary: str) -> list[CriticFinding]:
    findings: list[CriticFinding] = []
    lowered = plan_summary.lower()
    if "rewrite lightrag" in lowered or "replace lightrag" in lowered:
        findings.append(
            CriticFinding(
                rule="preserve_core",
                severity="high",
                message="Plan appears to replace the LightRAG core instead of wrapping it.",
            )
        )
    if "hardcode" in lowered and "model" in lowered:
        findings.append(
            CriticFinding(
                rule="dynamic_catalog",
                severity="medium",
                message="Model selection must come from runtime catalog sync, not fixed lists.",
            )
        )
    if "external send" in lowered and "approval" not in lowered:
        findings.append(
            CriticFinding(
                rule="human_approval",
                severity="medium",
                message="External or destructive actions require an approval gate.",
            )
        )
    return findings
