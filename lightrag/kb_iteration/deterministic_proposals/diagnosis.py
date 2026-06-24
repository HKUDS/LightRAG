from __future__ import annotations

from .base import (
    CandidateGenerationResult,
    GenerationContext,
    candidate_rejection,
    candidate_predicates,
    combine_results,
    merged_qualifier_repair,
    replace_relation_candidate,
    string_value,
    type_is,
    generate_with_builders,
)


def generate(context: GenerationContext) -> CandidateGenerationResult:
    candidates: list[dict] = []
    rejections: list[dict] = []
    for issue in context.issues:
        support_candidate, support_rejection, support_handled = (
            _supports_or_refutes_candidate(issue)
        )
        if support_candidate is not None:
            candidates.append(support_candidate)
        if support_rejection is not None:
            rejections.append(support_rejection)
        if support_handled:
            continue
        for candidate in (
            _diagnostic_criterion_candidate(issue),
            _orders_test_candidate(issue),
        ):
            if candidate is not None:
                candidates.append(candidate)
    return combine_results(
        generate_with_builders(context),
        CandidateGenerationResult(candidates=candidates, rejections=rejections),
    )


def _diagnostic_criterion_candidate(issue: dict) -> dict | None:
    if "has_diagnostic_criterion" not in candidate_predicates(issue):
        return None
    source = string_value(issue.get("source"))
    target = string_value(issue.get("target"))
    if _is_finding_type(issue.get("source_type")) and _is_condition_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="has_diagnostic_criterion",
            new_source=target,
            new_target=source,
        )
    if _is_condition_type(issue.get("source_type")) and _is_finding_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="has_diagnostic_criterion",
        )
    return None


def _orders_test_candidate(issue: dict) -> dict | None:
    if "orders_test" not in candidate_predicates(issue):
        return None
    source = string_value(issue.get("source"))
    target = string_value(issue.get("target"))
    if type_is(issue.get("source_type"), "Test", "DiagnosticTest") and (
        _is_condition_type(issue.get("target_type"))
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="orders_test",
            new_source=target,
            new_target=source,
            qualifiers={"indication": "diagnosis_or_severity_assessment"},
        )
    if _is_condition_type(issue.get("source_type")) and type_is(
        issue.get("target_type"), "Test", "DiagnosticTest"
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="orders_test",
            qualifiers={"indication": "diagnosis_or_severity_assessment"},
        )
    return None


def _supports_or_refutes_candidate(
    issue: dict,
) -> tuple[dict | None, dict | None, bool]:
    if "supports_or_refutes" not in candidate_predicates(issue):
        return None, None, False
    source_is_evidence = _is_diagnostic_evidence_type(issue.get("source_type"))
    target_is_evidence = _is_diagnostic_evidence_type(issue.get("target_type"))
    source_is_condition = _is_condition_type(issue.get("source_type"))
    target_is_condition = _is_condition_type(issue.get("target_type"))
    if not (
        source_is_evidence
        and target_is_condition
        or source_is_condition
        and target_is_evidence
    ):
        return None, None, False

    qualifiers = merged_qualifier_repair(issue)
    polarity = string_value(qualifiers.get("polarity"))
    if polarity not in {"supports", "refutes"}:
        return None, candidate_rejection(
            issue,
            error_code="SUPPORTS_OR_REFUTES_POLARITY_REQUIRED",
            error="supports_or_refutes requires explicit polarity supports or refutes.",
        ), True

    if source_is_condition:
        return replace_relation_candidate(
            issue,
            new_keywords="supports_or_refutes",
            new_source=string_value(issue.get("target")),
            new_target=string_value(issue.get("source")),
            qualifiers=qualifiers,
        ), None, True
    return replace_relation_candidate(
        issue,
        new_keywords="supports_or_refutes",
        qualifiers=qualifiers,
    ), None, True


def _is_condition_type(value: object) -> bool:
    return type_is(value, "Disease", "ClinicalCondition")


def _is_finding_type(value: object) -> bool:
    return type_is(
        value,
        "DiagnosticCriterion",
        "ClinicalFinding",
        "Symptom",
        "Sign",
    )


def _is_diagnostic_evidence_type(value: object) -> bool:
    return type_is(
        value,
        "Evidence",
        "TestResult",
        "TestResultPattern",
        "ClinicalFinding",
    )
