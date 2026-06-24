from __future__ import annotations

from .base import (
    CandidateGenerationResult,
    GenerationContext,
    candidate_rejection,
    candidate_predicates,
    combine_results,
    is_conflict_review_issue,
    merged_qualifier_repair,
    replace_relation_candidate,
    string_value,
    type_is,
    generate_with_builders,
)

_SAFETY_PREDICATE_PRIORITY = (
    "contraindicated_for",
    "not_recommended_for",
    "precaution_for",
    "temporarily_deferred_for",
)


def generate(context: GenerationContext) -> CandidateGenerationResult:
    if context.issue_family == "multi_predicate_split":
        return generate_with_builders(context)

    candidates: list[dict] = []
    rejections: list[dict] = []
    for issue in context.issues:
        if is_conflict_review_issue(issue):
            rejections.append(
                candidate_rejection(
                    issue,
                    error_code="REVIEW_CONFLICT_NOT_MUTATED",
                    error="Conflict review issues are not deterministic treatment mutations.",
                )
            )
            continue
        for candidate in (
            _has_indication_candidate(issue),
            _recommends_candidate(issue),
            _recommended_for_candidate(issue),
        ):
            if candidate is not None:
                candidates.append(candidate)
        safety_candidate, safety_rejection = _safety_candidate(issue)
        if safety_candidate is not None:
            candidates.append(safety_candidate)
        if safety_rejection is not None:
            rejections.append(safety_rejection)
    return combine_results(
        generate_with_builders(context),
        CandidateGenerationResult(candidates=candidates, rejections=rejections),
    )


def _has_indication_candidate(issue: dict) -> dict | None:
    if "has_indication" not in candidate_predicates(issue):
        return None
    source = string_value(issue.get("source"))
    target = string_value(issue.get("target"))
    if _is_intervention_type(issue.get("source_type")) and _is_indication_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(issue, new_keywords="has_indication")
    if _is_indication_type(issue.get("source_type")) and _is_intervention_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="has_indication",
            new_source=target,
            new_target=source,
        )
    return None


def _recommends_candidate(issue: dict) -> dict | None:
    if "recommends" not in candidate_predicates(issue):
        return None
    source = string_value(issue.get("source"))
    target = string_value(issue.get("target"))
    if _is_recommendation_type(issue.get("source_type")) and _is_intervention_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(issue, new_keywords="recommends")
    if _is_intervention_type(issue.get("source_type")) and _is_recommendation_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="recommends",
            new_source=target,
            new_target=source,
        )
    return None


def _recommended_for_candidate(issue: dict) -> dict | None:
    if "recommended_for" not in candidate_predicates(issue):
        return None
    if not (
        _is_intervention_type(issue.get("source_type"))
        and type_is(issue.get("target_type"), "Population")
    ):
        return None
    qualifiers = _merged_qualifier_repair(issue)
    qualifiers.setdefault(
        "purpose",
        "prevention"
        if type_is(issue.get("source_type"), "Vaccine", "PublicHealthMeasure")
        else "treatment",
    )
    if not _has_recommended_for_scope(qualifiers):
        return None
    return replace_relation_candidate(
        issue,
        new_keywords="recommended_for",
        qualifiers=qualifiers,
    )


def _safety_candidate(issue: dict) -> tuple[dict | None, dict | None]:
    predicates = candidate_predicates(issue)
    matched = [
        candidate for candidate in _SAFETY_PREDICATE_PRIORITY if candidate in predicates
    ]
    if not matched:
        return None, None
    if len(matched) > 1:
        return None, candidate_rejection(
            issue,
            error_code="AMBIGUOUS_SAFETY_PREDICATE",
            error="Multiple safety predicates are possible; deterministic generator will not choose one.",
        )
    predicate = matched[0]
    if not (
        _is_intervention_type(issue.get("source_type"))
        and _is_safety_target_type(issue.get("target_type"))
    ):
        return None, None
    qualifiers = _merged_qualifier_repair(issue)
    if not string_value(qualifiers.get("reason")):
        return None, candidate_rejection(
            issue,
            error_code="SAFETY_REASON_QUALIFIER_REQUIRED",
            error="Safety-limiting treatment predicates require an explicit reason qualifier.",
        )
    return replace_relation_candidate(
        issue,
        new_keywords=predicate,
        qualifiers=qualifiers,
    ), None


def _merged_qualifier_repair(issue: dict) -> dict:
    return merged_qualifier_repair(issue)


def _has_recommended_for_scope(qualifiers: dict) -> bool:
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


def _is_intervention_type(value: object) -> bool:
    return type_is(
        value,
        "Drug",
        "DrugIngredient",
        "Treatment",
        "Procedure",
        "Vaccine",
        "PublicHealthMeasure",
    )


def _is_indication_type(value: object) -> bool:
    return type_is(value, "Disease", "ClinicalCondition")


def _is_recommendation_type(value: object) -> bool:
    return type_is(value, "Guideline", "Recommendation", "ClinicalPathway")


def _is_safety_target_type(value: object) -> bool:
    return type_is(value, "Population", "ClinicalCondition", "RiskFactor")
