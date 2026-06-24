from __future__ import annotations

from collections.abc import Callable

from . import (
    clinical_modeling,
    diagnosis,
    entity_cleanup,
    prevention,
    risk_safety,
    treatment,
)
from .base import (
    CandidateGenerationResult,
    GenerationContext,
    candidate_rejection,
    combine_results,
    is_conflict_review_issue,
)

Generator = Callable[[GenerationContext], CandidateGenerationResult]

_GENERATORS: dict[str, Generator] = {
    "clinical_modeling": clinical_modeling.generate,
    "direction": clinical_modeling.generate,
    "diagnosis": diagnosis.generate,
    "treatment": treatment.generate,
    "multi_predicate_split": treatment.generate,
    "risk_safety": risk_safety.generate,
    "prevention": prevention.generate,
    "entity_cleanup": entity_cleanup.generate,
    "alias_role_conflict": entity_cleanup.generate,
}

_CLINICAL_PREDICATES = {"has_manifestation", "causative_agent", "is_a"}
_DIAGNOSIS_PREDICATES = {
    "has_diagnostic_criterion",
    "criterion_requires",
    "has_evidence",
    "supports_or_refutes",
    "orders_test",
    "has_result",
    "uses_specimen",
    "performed_by_method",
}
_TREATMENT_PREDICATES = {
    "has_indication",
    "recommends",
    "recommended_for",
    "not_recommended_for",
    "contraindicated_for",
    "precaution_for",
    "temporarily_deferred_for",
    "has_dosing_regimen",
    "may_cause_adverse_reaction",
}
_RISK_PREDICATES = {
    "risk_factor_for",
    "high_risk_for",
    "increases_risk_of",
    "acute_exacerbation_of",
    "has_complication",
    "has_risk_factor",
}
_PREVENTION_PREDICATES = {
    "targets_disease",
    "reduces_risk_of",
    "has_dose_schedule",
}


def generate_candidates(context: GenerationContext) -> CandidateGenerationResult:
    conflict_rejections = [
        candidate_rejection(
            issue,
            error_code="REVIEW_CONFLICT_NOT_MUTATED",
            error="Conflict review issues require review and do not get deterministic mutation candidates.",
        )
        for issue in context.issues
        if is_conflict_review_issue(issue)
    ]
    filtered_issues = [
        issue for issue in context.issues if not is_conflict_review_issue(issue)
    ]
    if not filtered_issues:
        return CandidateGenerationResult(rejections=conflict_rejections)

    if context.issue_family == "legacy_schema":
        return combine_results(
            CandidateGenerationResult(rejections=conflict_rejections),
            _generate_legacy_schema(
                GenerationContext(
                    issue_family=context.issue_family,
                    issues=filtered_issues,
                    builders=context.builders,
                    edges_by_id=context.edges_by_id,
                    nodes_by_id=context.nodes_by_id,
                    all_edges=context.all_edges,
                )
            ),
        )

    generator = _GENERATORS.get(context.issue_family)
    if generator is None:
        return CandidateGenerationResult(
            rejections=[
                candidate_rejection(
                    issue,
                    error_code="NO_GENERATOR_FOR_FAMILY",
                    error=(
                        "No deterministic generator registered for "
                        f"{context.issue_family}"
                    ),
                )
                for issue in filtered_issues
            ]
        )
    filtered_context = GenerationContext(
        issue_family=context.issue_family,
        issues=filtered_issues,
        builders=context.builders,
        edges_by_id=context.edges_by_id,
        nodes_by_id=context.nodes_by_id,
        all_edges=context.all_edges,
    )
    return combine_results(
        CandidateGenerationResult(rejections=conflict_rejections),
        generator(filtered_context),
    )


def _generate_legacy_schema(context: GenerationContext) -> CandidateGenerationResult:
    builder_result = clinical_modeling.generate(
        GenerationContext(
            issue_family=context.issue_family,
            issues=context.issues,
            builders=context.builders,
            edges_by_id=context.edges_by_id,
            nodes_by_id=context.nodes_by_id,
            all_edges=context.all_edges,
        )
    )
    family_results: list[CandidateGenerationResult] = []
    for family, predicates, generator in (
        ("clinical_modeling", _CLINICAL_PREDICATES, clinical_modeling.generate),
        ("diagnosis", _DIAGNOSIS_PREDICATES, diagnosis.generate),
        ("treatment", _TREATMENT_PREDICATES, treatment.generate),
        ("risk_safety", _RISK_PREDICATES, risk_safety.generate),
        ("prevention", _PREVENTION_PREDICATES, prevention.generate),
    ):
        issues = _issues_with_any_predicate(context.issues, predicates)
        if not issues:
            continue
        family_results.append(
            generator(
                GenerationContext(
                    issue_family=family,
                    issues=issues,
                    builders=(),
                    edges_by_id=context.edges_by_id,
                    nodes_by_id=context.nodes_by_id,
                    all_edges=context.all_edges,
                )
            )
        )
    return combine_results(builder_result, *family_results)


def _issues_with_any_predicate(
    issues: tuple[dict] | list[dict],
    predicates: set[str],
) -> list[dict]:
    return [
        issue
        for issue in issues
        if {
            str(predicate).strip()
            for predicate in issue.get("candidate_predicates", [])
            if isinstance(predicate, str) and predicate.strip()
        }
        & predicates
    ]
