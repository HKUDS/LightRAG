from __future__ import annotations

from .base import (
    CandidateGenerationResult,
    GenerationContext,
    candidate_predicates,
    combine_results,
    replace_relation_candidate,
    string_value,
    type_is,
    generate_with_builders,
)


def generate(context: GenerationContext) -> CandidateGenerationResult:
    candidates = [
        candidate
        for issue in context.issues
        for candidate in (
            _manifestation_candidate(issue),
            _causative_agent_candidate(issue),
        )
        if candidate is not None
    ]
    return combine_results(
        generate_with_builders(context),
        CandidateGenerationResult(candidates=candidates),
    )


def _manifestation_candidate(issue: dict) -> dict | None:
    if "has_manifestation" not in candidate_predicates(issue):
        return None
    source = string_value(issue.get("source"))
    target = string_value(issue.get("target"))
    if _is_finding_type(issue.get("source_type")) and _is_condition_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="has_manifestation",
            new_source=target,
            new_target=source,
        )
    if _is_condition_type(issue.get("source_type")) and _is_finding_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(issue, new_keywords="has_manifestation")
    return None


def _causative_agent_candidate(issue: dict) -> dict | None:
    if "causative_agent" not in candidate_predicates(issue):
        return None
    source = string_value(issue.get("source"))
    target = string_value(issue.get("target"))
    if type_is(issue.get("source_type"), "Pathogen") and _is_condition_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="causative_agent",
            new_source=target,
            new_target=source,
        )
    if _is_condition_type(issue.get("source_type")) and type_is(
        issue.get("target_type"), "Pathogen"
    ):
        return replace_relation_candidate(issue, new_keywords="causative_agent")
    return None


def _is_condition_type(value: object) -> bool:
    return type_is(value, "Disease", "ClinicalCondition", "Syndrome")


def _is_finding_type(value: object) -> bool:
    return type_is(
        value,
        "Symptom",
        "Sign",
        "ClinicalFinding",
        "sign_or_symptom",
    )
