from __future__ import annotations

from .base import (
    CandidateGenerationResult,
    GenerationContext,
    candidate_predicates,
    combine_results,
    merged_qualifier_repair,
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
            _targets_disease_candidate(issue),
            _reduces_risk_candidate(issue),
            _recommended_for_candidate(issue),
        )
        if candidate is not None
    ]
    return combine_results(
        generate_with_builders(context),
        CandidateGenerationResult(candidates=candidates),
    )


def _targets_disease_candidate(issue: dict) -> dict | None:
    if "targets_disease" not in candidate_predicates(issue):
        return None
    if _is_prevention_source_type(issue.get("source_type")) and _is_disease_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(issue, new_keywords="targets_disease")
    return None


def _reduces_risk_candidate(issue: dict) -> dict | None:
    if "reduces_risk_of" not in candidate_predicates(issue):
        return None
    if "targets_disease" in candidate_predicates(issue) and _is_disease_type(
        issue.get("target_type")
    ):
        return None
    if _is_prevention_source_type(issue.get("source_type")) and _is_risk_target_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="reduces_risk_of",
            qualifiers=_merged_qualifier_repair(issue),
        )
    return None


def _recommended_for_candidate(issue: dict) -> dict | None:
    if "recommended_for" not in candidate_predicates(issue):
        return None
    if not (
        _is_prevention_source_type(issue.get("source_type"))
        and type_is(issue.get("target_type"), "Population")
    ):
        return None
    qualifiers = _merged_qualifier_repair(issue)
    qualifiers.setdefault("purpose", "prevention")
    if not _has_scope(qualifiers):
        return None
    return replace_relation_candidate(
        issue,
        new_keywords="recommended_for",
        qualifiers=qualifiers,
    )


def _merged_qualifier_repair(issue: dict) -> dict:
    return merged_qualifier_repair(issue)


def _has_scope(qualifiers: dict) -> bool:
    return any(
        string_value(qualifiers.get(key))
        for key in (
            "condition",
            "age",
            "age_min",
            "age_max",
            "population",
            "route",
            "timing",
            "time_window",
        )
    )


def _is_prevention_source_type(value: object) -> bool:
    return type_is(value, "Vaccine", "PublicHealthMeasure")


def _is_disease_type(value: object) -> bool:
    return type_is(value, "Disease", "ClinicalCondition")


def _is_risk_target_type(value: object) -> bool:
    return type_is(value, "Disease", "ClinicalCondition", "Outcome", "Complication")
