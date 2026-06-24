from __future__ import annotations

from .base import (
    CandidateGenerationResult,
    GenerationContext,
    candidate_rejection,
    candidate_predicates,
    combine_results,
    issue_text,
    is_conflict_review_issue,
    replace_relation_candidate,
    string_value,
    type_is,
    generate_with_builders,
)


def generate(context: GenerationContext) -> CandidateGenerationResult:
    candidates: list[dict] = []
    rejections: list[dict] = []
    for issue in context.issues:
        if is_conflict_review_issue(issue):
            rejections.append(
                candidate_rejection(
                    issue,
                    error_code="REVIEW_CONFLICT_NOT_MUTATED",
                    error="Conflict review issues are not deterministic risk mutations.",
                )
            )
            continue
        risk_candidate, risk_rejection = _risk_relation_candidate(issue)
        if risk_candidate is not None:
            candidates.append(risk_candidate)
        if risk_rejection is not None:
            rejections.append(risk_rejection)
        adverse_candidate = _adverse_reaction_candidate(issue)
        if adverse_candidate is not None:
            candidates.append(adverse_candidate)
    return combine_results(
        generate_with_builders(context),
        CandidateGenerationResult(candidates=candidates, rejections=rejections),
    )


def _risk_relation_candidate(issue: dict) -> tuple[dict | None, dict | None]:
    predicates = candidate_predicates(issue)
    matched = predicates & {
        "risk_factor_for",
        "high_risk_for",
        "increases_risk_of",
    }
    if matched and _outcome_or_severity_endpoint_requires_review(issue, matched):
        return None, candidate_rejection(
            issue,
            error_code="OUTCOME_OR_SEVERITY_NOT_RISK_FACTOR",
            error=(
                "Outcome or severity endpoints require explicit risk-increase "
                "semantics; deterministic generator will not model them as risk factors."
            ),
        )
    if len(matched) > 1:
        selected = _risk_predicate_from_issue(issue, matched)
        if not selected:
            return None, candidate_rejection(
                issue,
                error_code="AMBIGUOUS_RISK_PREDICATE",
                error="Multiple risk predicates are possible; deterministic generator will not choose one.",
            )
        return _candidate_for_selected_risk_predicate(issue, selected), None
    if "risk_factor_for" in matched:
        return _risk_factor_candidate(issue), None
    if "high_risk_for" in matched:
        return _high_risk_candidate(issue), None
    if "increases_risk_of" in matched:
        return _increases_risk_candidate(issue), None
    return None, None


def _risk_predicate_from_issue(issue: dict, matched: set[str]) -> str:
    text = issue_text(issue)
    if "increases_risk_of" in matched and _explicit_risk_increase_text(issue):
        return "increases_risk_of"
    if "risk_factor_for" in matched and type_is(issue.get("source_type"), "RiskFactor"):
        return "risk_factor_for"
    if "risk_factor_for" in matched and (
        "risk_factor" in text or "risk factor" in text or "高危因素" in text
    ):
        return "risk_factor_for"
    if "increases_risk_of" in matched and (
        "increases_risk" in text or "increase" in text or "风险升高" in text
    ):
        return "increases_risk_of"
    if "high_risk_for" in matched and type_is(issue.get("source_type"), "Population"):
        return "high_risk_for"
    return ""


def _candidate_for_selected_risk_predicate(
    issue: dict,
    predicate: str,
) -> dict | None:
    if predicate == "risk_factor_for":
        return _risk_factor_candidate(issue)
    if predicate == "high_risk_for":
        return _high_risk_candidate(issue)
    if predicate == "increases_risk_of":
        return _increases_risk_candidate(issue)
    return None


def _risk_factor_candidate(issue: dict) -> dict | None:
    if "risk_factor_for" not in candidate_predicates(issue):
        return None
    source = string_value(issue.get("source"))
    target = string_value(issue.get("target"))
    if _is_risk_factor_type(issue.get("source_type")) and _is_risk_target_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(issue, new_keywords="risk_factor_for")
    if _is_risk_target_type(issue.get("source_type")) and _is_risk_factor_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="risk_factor_for",
            new_source=target,
            new_target=source,
        )
    return None


def _high_risk_candidate(issue: dict) -> dict | None:
    if "high_risk_for" not in candidate_predicates(issue):
        return None
    source = string_value(issue.get("source"))
    target = string_value(issue.get("target"))
    if type_is(issue.get("source_type"), "Population") and _is_risk_target_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(issue, new_keywords="high_risk_for")
    if _is_risk_target_type(issue.get("source_type")) and type_is(
        issue.get("target_type"), "Population"
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="high_risk_for",
            new_source=target,
            new_target=source,
        )
    return None


def _increases_risk_candidate(issue: dict) -> dict | None:
    if "increases_risk_of" not in candidate_predicates(issue):
        return None
    if _is_risk_factor_type(issue.get("source_type")) and _is_risk_target_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(issue, new_keywords="increases_risk_of")
    return None


def _outcome_or_severity_endpoint_requires_review(
    issue: dict,
    matched: set[str],
) -> bool:
    if _explicit_risk_increase_text(issue):
        return False
    if _is_outcome_or_severity_type(issue.get("source_type")):
        return True
    if not _is_outcome_or_severity_type(issue.get("target_type")):
        return False
    return not (
        "risk_factor_for" in matched
        and type_is(issue.get("source_type"), "RiskFactor")
    )


def _explicit_risk_increase_text(issue: dict) -> bool:
    text = " ".join(
        string_value(issue.get(key)).casefold()
        for key in (
            "issue_kind",
            "keywords",
            "medical_subcase",
            "evidence_quote",
        )
    )
    if "风险" in text and any(
        marker in text for marker in ("增加", "提高", "升高", "增高")
    ):
        return True
    return any(
        marker in text
        for marker in (
            "increases_risk",
            "increase risk",
            "increased risk",
            "raises risk",
            "增加风险",
            "提高风险",
            "升高风险",
            "风险增加",
            "风险升高",
            "风险增高",
            "椋庨櫓鍗囬珮",
        )
    )


def _adverse_reaction_candidate(issue: dict) -> dict | None:
    if "may_cause_adverse_reaction" not in candidate_predicates(issue):
        return None
    if _is_intervention_type(issue.get("source_type")) and _is_adverse_target_type(
        issue.get("target_type")
    ):
        return replace_relation_candidate(
            issue,
            new_keywords="may_cause_adverse_reaction",
        )
    return None


def _is_risk_factor_type(value: object) -> bool:
    return type_is(value, "RiskFactor", "ClinicalCondition", "Disease")


def _is_risk_target_type(value: object) -> bool:
    return type_is(value, "Disease", "ClinicalCondition", "Outcome", "Complication")


def _is_intervention_type(value: object) -> bool:
    return type_is(value, "Drug", "Treatment", "Vaccine", "Procedure")


def _is_adverse_target_type(value: object) -> bool:
    return type_is(value, "AdverseReaction", "Symptom", "sign_or_symptom")


def _is_outcome_or_severity_type(value: object) -> bool:
    return type_is(value, "Outcome", "Severity", "SeverityGrade")
