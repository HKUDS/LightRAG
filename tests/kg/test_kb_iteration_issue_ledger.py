import json
from collections import Counter
from pathlib import Path

import pytest

from lightrag.kb_iteration.agent_pipeline import action_candidate_proposals_from_scan
from lightrag.kb_iteration.deterministic_proposals.base import CandidateGenerationResult
from lightrag.kb_iteration.models import KGSnapshot, SnapshotEdge, SnapshotNode
from lightrag.kb_iteration.issue_ledger import (
    _rejected_action_fingerprints,
    normalize_raw_issues,
    route_state_counts,
    scan_deterministic_candidates,
)
from lightrag.kb_iteration.proposal_fingerprints import candidate_fingerprints
from lightrag.kb_iteration.proposals import validate_proposal
from lightrag.kb_iteration.quality import evaluate_snapshot_quality


RAW_ISSUE_CONTRACT_KEYS = {
    "issue_ref",
    "issue_source",
    "issue_family",
    "issue_kind",
    "issue_order",
    "edge_id",
    "source",
    "source_type",
    "target",
    "target_type",
    "keywords",
    "qualifiers",
    "candidate_predicates",
    "repair_options",
    "suggested_qualifiers",
    "source_id",
    "file_path",
    "evidence_quote",
    "issue_fingerprint",
    "evidence_spans",
    "auto_fixable",
    "blocked_reason",
}


def test_issue_ledger_routes_every_raw_issue_once(tmp_path: Path) -> None:
    package = _make_ledger_package(
        tmp_path,
        issues=[
            _ledger_issue(
                "edge-deterministic",
                issue_kind="reverse_clinical_manifestation",
                keywords="clinical_manifestation",
                candidate_predicates=["has_manifestation"],
            ),
            _ledger_issue(
                "edge-residual",
                issue_kind="conflicting_recommendation_safety_scope",
                suggested_action="review_conflict",
                candidate_predicates=["recommended_for", "contraindicated_for"],
            ),
            _ledger_issue(
                "edge-missing-evidence",
                issue_kind="treatment_domain_range_mismatch",
                source_id="",
                file_path="",
                candidate_predicates=["has_indication"],
            ),
            _ledger_issue(
                "edge-stale-keywords",
                issue_kind="treatment_domain_range_mismatch",
                keywords="old_recommended_treatment",
                candidate_predicates=["has_indication"],
            ),
        ],
        edges=[
            _ledger_edge(
                "edge-deterministic",
                source="cough",
                target="flu",
                keywords="clinical_manifestation",
            ),
            _ledger_edge("edge-residual"),
            _ledger_edge("edge-missing-evidence"),
            _ledger_edge("edge-stale-keywords", keywords="recommended_treatment"),
        ],
    )

    result = scan_deterministic_candidates(package)
    counts = route_state_counts(result.issue_routes)
    issue_refs = [str(issue["issue_ref"]) for issue in result.issues]
    route_refs = [route.issue_ref for route in result.issue_routes]

    expected_states = {
        "deterministic_covered",
        "llm_residual",
        "blocked_safety",
        "blocked_apply",
        "blocked_evidence",
        "deferred_budget",
        "duplicate",
        "stale",
    }
    assert set(counts) <= expected_states
    assert set(route_refs) == set(issue_refs)
    assert Counter(route_refs) == Counter({issue_ref: 1 for issue_ref in issue_refs})
    assert len(result.issues) == sum(counts.get(state, 0) for state in expected_states)


def test_candidate_gate_rejections_route_to_terminal_blocked_dispositions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _make_ledger_package(
        tmp_path,
        issues=[
            _ledger_issue(
                "edge-known-bad",
                issue_kind="treatment_domain_range_mismatch",
                candidate_predicates=["has_indication"],
            ),
            _ledger_issue(
                "edge-evidence-object",
                issue_kind="evidence_contract_mismatch",
                candidate_predicates=["has_indication"],
            ),
            _ledger_issue(
                "edge-missing-types",
                issue_kind="candidate_expansion_contract_mismatch",
                candidate_predicates=["has_manifestation"],
            ),
        ],
        edges=[
            _ledger_edge("edge-known-bad"),
            _ledger_edge("edge-evidence-object"),
            _ledger_edge("edge-missing-types"),
        ],
    )

    def fake_candidates(
        chunk: list[dict[str, object]],
        **_kwargs: object,
    ) -> CandidateGenerationResult:
        rejections = []
        for issue in chunk:
            edge_id = str(issue["edge_id"])
            error_code = {
                "edge-known-bad": "KNOWN_BAD_MEDICAL_PATTERN",
                "edge-evidence-object": "EVIDENCE_MUST_BE_STRING",
                "edge-missing-types": "CANDIDATE_EDGE_TYPES_REQUIRED",
            }[edge_id]
            rejections.append(
                {
                    "candidate_id": f"candidate-{edge_id}",
                    "issue_ref": issue["issue_ref"],
                    "issue_family": issue["issue_family"],
                    "stage": "deterministic_candidate",
                    "error_code": error_code,
                    "error": f"{error_code} rejection",
                }
            )
        return CandidateGenerationResult(candidates=[], rejections=rejections)

    monkeypatch.setattr(
        "lightrag.kb_iteration.proposal_orchestrator._action_candidates_for_issues",
        fake_candidates,
    )

    result = scan_deterministic_candidates(package)
    routes = {route.issue_ref: route for route in result.issue_routes}
    routes_by_edge = {
        str(issue["edge_id"]): routes[str(issue["issue_ref"])]
        for issue in result.issues
    }

    assert routes_by_edge["edge-known-bad"].generation_disposition == "blocked_safety"
    assert routes_by_edge["edge-known-bad"].route_state == "blocked_safety"
    assert (
        routes_by_edge["edge-evidence-object"].generation_disposition
        == "blocked_evidence"
    )
    assert routes_by_edge["edge-evidence-object"].route_state == "blocked_evidence"
    assert (
        routes_by_edge["edge-missing-types"].generation_disposition
        == "blocked_schema"
    )
    assert routes_by_edge["edge-missing-types"].route_state == "blocked_schema"
    assert all(route.route_state != "llm_residual" for route in routes_by_edge.values())


@pytest.mark.parametrize("error_code", ["MISSING_EVIDENCE", "INSUFFICIENT_EVIDENCE"])
def test_content_evidence_rejections_remain_llm_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
) -> None:
    package = _make_ledger_package(
        tmp_path,
        issues=[
            _ledger_issue(
                "edge-content-evidence",
                issue_kind="treatment_domain_range_mismatch",
                candidate_predicates=["has_indication"],
            ),
            _ledger_issue(
                "edge-evidence-contract",
                issue_kind="evidence_contract_mismatch",
                candidate_predicates=["has_indication"],
            ),
        ],
        edges=[
            _ledger_edge("edge-content-evidence"),
            _ledger_edge("edge-evidence-contract"),
        ],
    )

    def fake_candidates(
        chunk: list[dict[str, object]],
        **_kwargs: object,
    ) -> CandidateGenerationResult:
        rejections = []
        for issue in chunk:
            edge_id = str(issue["edge_id"])
            rejection_code = (
                error_code
                if edge_id == "edge-content-evidence"
                else "EVIDENCE_MUST_BE_STRING"
            )
            rejections.append(
                {
                    "candidate_id": f"candidate-{edge_id}",
                    "issue_ref": issue["issue_ref"],
                    "issue_family": issue["issue_family"],
                    "stage": "deterministic_candidate",
                    "error_code": rejection_code,
                    "error": f"{rejection_code} rejection",
                }
            )
        return CandidateGenerationResult(candidates=[], rejections=rejections)

    monkeypatch.setattr(
        "lightrag.kb_iteration.proposal_orchestrator._action_candidates_for_issues",
        fake_candidates,
    )

    result = scan_deterministic_candidates(package)
    routes = {route.issue_ref: route for route in result.issue_routes}
    routes_by_edge = {
        str(issue["edge_id"]): routes[str(issue["issue_ref"])]
        for issue in result.issues
    }

    content_route = routes_by_edge["edge-content-evidence"]
    assert content_route.route_state == "llm_residual"
    assert content_route.generation_disposition == "llm_residual"
    assert content_route.reason_code == error_code

    contract_route = routes_by_edge["edge-evidence-contract"]
    assert contract_route.route_state == "blocked_evidence"
    assert contract_route.generation_disposition == "blocked_evidence"
    assert contract_route.reason_code == "EVIDENCE_MUST_BE_STRING"


def test_normalize_raw_issues_includes_schema_and_cleanup_issues() -> None:
    quality = {
        "details": {
            "medical_schema_issues": [
                {
                    "issue_kind": "missing_required_relation_qualifier",
                    "edge_id": "edge-evidence",
                    "source": "positive-pcr",
                    "source_type": "TestResultPattern",
                    "target": "flu",
                    "target_type": "Disease",
                    "keywords": "supports_or_refutes",
                    "qualifiers": {},
                    "candidate_predicates": ["supports_or_refutes"],
                    "repair_options": [{"predicate": "supports_or_refutes"}],
                    "suggested_qualifiers": {"polarity": "supports"},
                    "source_id": "chunk-1",
                    "file_path": "guide.md",
                    "evidence_quote": "Positive PCR supports influenza diagnosis.",
                    "auto_fixable": False,
                    "blocked_reason": "missing polarity",
                }
            ],
            "entity_cleanup_issues": [
                {
                    "issue_kind": "value_node_to_qualifier",
                    "node_id": "dose-75mg",
                    "candidate_predicates": ["has_dosing_regimen"],
                    "repair_options": [{"qualifier": "dose"}],
                    "source_id": "chunk-2",
                    "file_path": "guide.md",
                    "evidence_quote": "Oseltamivir 75 mg twice daily.",
                }
            ],
        }
    }

    issues = normalize_raw_issues(quality)

    assert [issue["issue_source"] for issue in issues] == [
        "medical_schema_issues",
        "entity_cleanup_issues",
    ]
    assert [issue["issue_family"] for issue in issues] == [
        "diagnosis",
        "entity_cleanup",
    ]
    for issue in issues:
        assert RAW_ISSUE_CONTRACT_KEYS <= set(issue)


def test_normalize_raw_issues_merges_all_structured_sources_before_findings() -> None:
    quality = {
        "details": {
            "medical_schema_issues": [
                {
                    "issue_kind": "diagnostic_evidence_direction_mismatch",
                    "edge_id": "edge-diagnosis",
                    "source_id": "chunk-diagnosis",
                    "file_path": "diagnosis.md",
                    "evidence_quote": "A PCR result supports influenza diagnosis.",
                }
            ],
            "entity_cleanup_issues": [
                {
                    "issue_kind": "synonym_duplicate",
                    "canonical_label": "influenza",
                    "node_ids": ["flu", "influenza"],
                }
            ],
            "evidence_issues": [
                {
                    "issue_kind": "missing_evidence_span",
                    "edge_id": "edge-evidence",
                    "evidence_spans": [
                        {
                            "source_id": "chunk-evidence",
                            "file_path": "evidence.md",
                            "evidence_quote": "Evidence text.",
                        }
                    ],
                }
            ],
            "generic_relation_issues": [
                {
                    "issue_kind": "generic_relation_keyword",
                    "edge_id": "edge-generic",
                    "keywords": "related_to",
                }
            ],
            "hierarchy_issues": [
                {
                    "issue_kind": "hierarchy_parent_mismatch",
                    "edge_id": "edge-hierarchy",
                }
            ],
        },
        "findings": [
            {
                "severity": "high",
                "category": "fallback",
                "message": "Fallback should not be used when structured issues exist.",
            }
        ],
    }

    issues = normalize_raw_issues(quality)

    assert [issue["issue_source"] for issue in issues] == [
        "medical_schema_issues",
        "entity_cleanup_issues",
        "evidence_issues",
        "generic_relation_issues",
        "hierarchy_issues",
    ]
    assert all(issue["issue_source"] != "findings" for issue in issues)
    assert issues[0]["evidence_spans"] == [
        {
            "source_id": "chunk-diagnosis",
            "file_path": "diagnosis.md",
            "evidence_quote": "A PCR result supports influenza diagnosis.",
        }
    ]
    assert issues[1]["evidence_spans"] == []
    assert issues[2]["evidence_spans"] == [
        {
            "source_id": "chunk-evidence",
            "file_path": "evidence.md",
            "evidence_quote": "Evidence text.",
        }
    ]
    for issue in issues:
        assert issue["issue_fingerprint"]
        assert issue["issue_fingerprint"][:12] in issue["issue_ref"]
        assert issue["issue_ref"].startswith(
            f"{issue['issue_source']}:{issue['issue_family']}:{issue['issue_kind']}:"
        )


def test_normalize_raw_issues_preserves_distinct_same_edge_schema_issues() -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode(
                "flu",
                "Influenza",
                "Disease",
                source_id="chunk-1",
                file_path="guide.md",
            ),
            SnapshotNode(
                "cough",
                "Cough",
                "Symptom",
                source_id="chunk-1",
                file_path="guide.md",
            ),
        ],
        edges=[
            SnapshotEdge(
                "edge-overloaded",
                "flu",
                "cough",
                "causative_agent, has_result",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )
    score = evaluate_snapshot_quality(snapshot)
    structured_issues = [
        issue
        for issue in score.details["medical_schema_issues"]
        if issue["edge_id"] == "edge-overloaded"
        and issue["issue_kind"] == "canonical_relation_domain_range_mismatch"
    ]

    issues = normalize_raw_issues(score.to_dict())
    normalized_issues = [
        issue
        for issue in issues
        if issue["edge_id"] == "edge-overloaded"
        and issue["issue_kind"] == "canonical_relation_domain_range_mismatch"
    ]

    assert [issue["candidate_predicates"] for issue in structured_issues] == [
        ["causative_agent"],
        ["has_result"],
    ]
    assert [issue["candidate_predicates"] for issue in normalized_issues] == [
        ["causative_agent"],
        ["has_result"],
    ]
    assert len({issue["issue_ref"] for issue in normalized_issues}) == 2


def test_normalize_raw_issues_preserves_issues_differing_only_by_suggested_qualifiers() -> None:
    quality = {
        "details": {
            "medical_schema_issues": [
                {
                    "issue_kind": "missing_required_relation_qualifier",
                    "edge_id": "edge-evidence",
                    "source": "positive-pcr",
                    "source_type": "TestResultPattern",
                    "target": "flu",
                    "target_type": "Disease",
                    "keywords": "supports_or_refutes",
                    "candidate_predicates": ["supports_or_refutes"],
                    "suggested_qualifiers": {"polarity": "supports"},
                },
                {
                    "issue_kind": "missing_required_relation_qualifier",
                    "edge_id": "edge-evidence",
                    "source": "positive-pcr",
                    "source_type": "TestResultPattern",
                    "target": "flu",
                    "target_type": "Disease",
                    "keywords": "supports_or_refutes",
                    "candidate_predicates": ["supports_or_refutes"],
                    "suggested_qualifiers": {"polarity": "refutes"},
                },
            ]
        }
    }

    issues = normalize_raw_issues(quality)

    assert [issue["suggested_qualifiers"] for issue in issues] == [
        {"polarity": "supports"},
        {"polarity": "refutes"},
    ]
    assert len({issue["issue_ref"] for issue in issues}) == 2


def test_normalize_raw_issues_uses_unique_refs_for_same_predicate_collisions() -> None:
    quality = {
        "details": {
            "medical_schema_issues": [
                {
                    "issue_kind": "missing_required_relation_qualifier",
                    "edge_id": "edge-evidence",
                    "candidate_predicates": ["supports_or_refutes"],
                    "validation_errors": [
                        "supports_or_refutes requires qualifier polarity"
                    ],
                },
                {
                    "issue_kind": "missing_required_relation_qualifier",
                    "edge_id": "edge-evidence",
                    "candidate_predicates": ["supports_or_refutes"],
                    "validation_errors": [
                        "supports_or_refutes requires qualifier evidence_direction"
                    ],
                },
                {
                    "issue_kind": "missing_required_relation_qualifier",
                    "edge_id": "edge-evidence",
                    "candidate_predicates": ["supports_or_refutes"],
                    "validation_errors": [
                        "supports_or_refutes requires one qualifier from strength|certainty"
                    ],
                },
            ]
        }
    }

    first = normalize_raw_issues(quality)
    second = normalize_raw_issues(quality)

    assert len(first) == 3
    assert [issue["issue_ref"] for issue in first] == [
        issue["issue_ref"] for issue in second
    ]
    assert len({issue["issue_ref"] for issue in first}) == 3


def test_normalize_raw_issues_uses_unique_refs_for_candidate_only_collisions() -> None:
    quality = {
        "details": {
            "medical_schema_issues": [
                {
                    "issue_kind": "canonical_relation_domain_range_mismatch",
                    "edge_id": "edge-overloaded",
                    "candidate_predicates": ["causative_agent"],
                },
                {
                    "issue_kind": "canonical_relation_domain_range_mismatch",
                    "edge_id": "edge-overloaded",
                    "candidate_predicates": ["has_result"],
                },
            ]
        }
    }

    first = normalize_raw_issues(quality)
    second = normalize_raw_issues(quality)

    assert [issue["candidate_predicates"] for issue in first] == [
        ["causative_agent"],
        ["has_result"],
    ]
    assert [issue["issue_ref"] for issue in first] == [
        issue["issue_ref"] for issue in second
    ]
    assert len({issue["issue_ref"] for issue in first}) == 2


def test_normalize_raw_issues_collision_ref_is_stable_when_sibling_removed() -> None:
    kept_issue = {
        "issue_kind": "missing_required_relation_qualifier",
        "edge_id": "edge-evidence",
        "candidate_predicates": ["supports_or_refutes"],
        "suggested_qualifiers": {"polarity": "refutes"},
    }
    quality_with_sibling = {
        "details": {
            "medical_schema_issues": [
                {
                    "issue_kind": "missing_required_relation_qualifier",
                    "edge_id": "edge-evidence",
                    "candidate_predicates": ["supports_or_refutes"],
                    "suggested_qualifiers": {"polarity": "supports"},
                },
                kept_issue,
            ]
        }
    }
    quality_without_sibling = {
        "details": {"medical_schema_issues": [kept_issue]}
    }

    ref_with_sibling = normalize_raw_issues(quality_with_sibling)[1]["issue_ref"]
    ref_without_sibling = normalize_raw_issues(quality_without_sibling)[0][
        "issue_ref"
    ]

    assert ref_with_sibling == ref_without_sibling


def test_normalize_raw_issues_collision_refs_are_stable_when_order_reversed() -> None:
    first_issue = {
        "issue_kind": "missing_required_relation_qualifier",
        "edge_id": "edge-evidence",
        "candidate_predicates": ["supports_or_refutes"],
        "validation_errors": ["supports_or_refutes requires qualifier polarity"],
    }
    second_issue = {
        "issue_kind": "missing_required_relation_qualifier",
        "edge_id": "edge-evidence",
        "candidate_predicates": ["supports_or_refutes"],
        "validation_errors": [
            "supports_or_refutes requires qualifier evidence_direction"
        ],
    }
    forward_quality = {
        "details": {"medical_schema_issues": [first_issue, second_issue]}
    }
    reverse_quality = {
        "details": {"medical_schema_issues": [second_issue, first_issue]}
    }

    forward_refs = {
        tuple(issue["validation_errors"]): issue["issue_ref"]
        for issue in normalize_raw_issues(forward_quality)
    }
    reverse_refs = {
        tuple(issue["validation_errors"]): issue["issue_ref"]
        for issue in normalize_raw_issues(reverse_quality)
    }

    assert forward_refs == reverse_refs


def test_normalize_raw_issues_fallback_ref_is_stable_when_sibling_removed() -> None:
    kept_issue = {
        "issue_kind": "unanchored_schema_issue",
        "keywords": "supports_or_refutes",
        "candidate_predicates": ["supports_or_refutes"],
        "suggested_qualifiers": {"polarity": "supports"},
    }
    quality_with_sibling = {
        "details": {
            "medical_schema_issues": [
                {
                    "issue_kind": "unanchored_schema_issue",
                    "keywords": "has_result",
                    "candidate_predicates": ["has_result"],
                },
                kept_issue,
            ]
        }
    }
    quality_without_sibling = {
        "details": {"medical_schema_issues": [kept_issue]}
    }

    ref_with_sibling = normalize_raw_issues(quality_with_sibling)[1]["issue_ref"]
    ref_without_sibling = normalize_raw_issues(quality_without_sibling)[0][
        "issue_ref"
    ]

    assert ref_with_sibling == ref_without_sibling


def test_normalize_raw_issues_fallback_refs_are_stable_when_order_reversed() -> None:
    first_issue = {
        "issue_kind": "unanchored_schema_issue",
        "keywords": "supports_or_refutes",
        "candidate_predicates": ["supports_or_refutes"],
    }
    second_issue = {
        "issue_kind": "unanchored_schema_issue",
        "keywords": "has_result",
        "candidate_predicates": ["has_result"],
    }
    forward_quality = {
        "details": {"medical_schema_issues": [first_issue, second_issue]}
    }
    reverse_quality = {
        "details": {"medical_schema_issues": [second_issue, first_issue]}
    }

    forward_refs = {
        tuple(issue["candidate_predicates"]): issue["issue_ref"]
        for issue in normalize_raw_issues(forward_quality)
    }
    reverse_refs = {
        tuple(issue["candidate_predicates"]): issue["issue_ref"]
        for issue in normalize_raw_issues(reverse_quality)
    }

    assert forward_refs == reverse_refs


def test_normalize_raw_issues_deep_copies_nested_contract_payloads() -> None:
    quality = {
        "details": {
            "medical_schema_issues": [
                {
                    "issue_kind": "missing_required_relation_qualifier",
                    "edge_id": "edge-evidence",
                    "qualifiers": {"scope": {"age_min": 7}},
                    "candidate_predicates": ["supports_or_refutes"],
                    "suggested_qualifiers": {"polarity": {"value": "supports"}},
                    "repair_options": [
                        {
                            "predicate": "supports_or_refutes",
                            "qualifiers": {"polarity": {"value": "supports"}},
                        }
                    ],
                }
            ]
        }
    }

    issue = normalize_raw_issues(quality)[0]
    issue["qualifiers"]["scope"]["age_min"] = 18
    issue["suggested_qualifiers"]["polarity"]["value"] = "refutes"
    issue["repair_options"][0]["qualifiers"]["polarity"]["value"] = "refutes"

    source_issue = quality["details"]["medical_schema_issues"][0]
    assert source_issue["qualifiers"] == {"scope": {"age_min": 7}}
    assert source_issue["suggested_qualifiers"] == {
        "polarity": {"value": "supports"}
    }
    assert source_issue["repair_options"] == [
        {
            "predicate": "supports_or_refutes",
            "qualifiers": {"polarity": {"value": "supports"}},
        }
    ]


def test_normalize_raw_issues_uses_stable_issue_refs() -> None:
    quality = {
        "details": {
            "medical_schema_issues": [
                {
                    "issue_kind": "legacy_overloaded_relation",
                    "edge_id": "edge-treatment",
                    "candidate_predicates": ["has_indication", "recommends"],
                }
            ],
            "entity_cleanup_issues": [
                {
                    "issue_kind": "synonym_duplicate",
                    "canonical_label": "influenza",
                    "node_ids": ["flu-short", "flu-full"],
                }
            ],
        }
    }

    first = normalize_raw_issues(quality)
    second = normalize_raw_issues(quality)

    assert [issue["issue_ref"] for issue in first] == [
        issue["issue_ref"] for issue in second
    ]
    assert first[0]["issue_ref"].startswith(
        "medical_schema_issues:treatment:legacy_overloaded_relation:"
    )
    assert first[1]["issue_ref"].startswith(
        "entity_cleanup_issues:entity_cleanup:synonym_duplicate:"
    )


def test_normalize_raw_issues_ignores_volatile_fields_in_fingerprint() -> None:
    base_issue = {
        "issue_kind": "legacy_overloaded_relation",
        "edge_id": "edge-treatment",
        "candidate_predicates": ["has_indication", "recommends"],
        "source": "flu",
        "target": "oseltamivir",
        "keywords": "recommended_treatment",
    }
    volatile_issue = {
        **base_issue,
        "generated_at": "2026-06-23T10:00:00Z",
        "run_id": "run-volatile",
        "transient_report_metadata": {"row": 17, "score": 0.42},
    }

    first = normalize_raw_issues(
        {"details": {"medical_schema_issues": [base_issue]}}
    )[0]
    second = normalize_raw_issues(
        {"details": {"medical_schema_issues": [volatile_issue]}}
    )[0]

    assert second["generated_at"] == "2026-06-23T10:00:00Z"
    assert first["issue_fingerprint"] == second["issue_fingerprint"]
    assert first["issue_ref"] == second["issue_ref"]


def test_normalize_raw_issues_falls_back_to_findings() -> None:
    quality = {
        "findings": [
            {
                "severity": "high",
                "category": "relation_semantics",
                "message": "Generic relation keywords should be replaced.",
                "evidence": ["edge:edge-related"],
                "suggested_fix_type": "replace_relation_keyword",
            }
        ]
    }

    issues = normalize_raw_issues(quality)

    assert len(issues) == 1
    issue = issues[0]
    assert RAW_ISSUE_CONTRACT_KEYS <= set(issue)
    assert issue["issue_source"] == "findings"
    assert issue["issue_kind"] == "replace_relation_keyword"
    assert issue["edge_id"] == "edge-related"
    assert issue["blocked_reason"] == "fallback_quality_finding"


def test_scan_blocks_unsupported_apply_candidate_before_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _make_ledger_package(
        tmp_path,
        issues=[
            _ledger_issue(
                "edge-alias",
                issue_kind="synonym_duplicate",
                suggested_action="entity_alias_merge",
            )
        ],
        edges=[_ledger_edge("edge-alias")],
    )

    issue_ref = normalize_raw_issues(
        json.loads(
            (package / "snapshots" / "quality_score.json").read_text(encoding="utf-8")
        )
    )[0]["issue_ref"]

    def candidates(*args: object, **kwargs: object) -> CandidateGenerationResult:
        return CandidateGenerationResult(
            candidates=[
                {
                    "candidate_id": "candidate-alias-merge",
                    "proposal_type": "entity_alias_merge",
                    "target": "node:flu-short",
                    "issue_ref": issue_ref,
                    "issue_kind": "synonym_duplicate",
                    "action_payload": {"action": "entity_alias_merge"},
                }
            ],
            rejections=[],
        )

    monkeypatch.setattr(
        "lightrag.kb_iteration.proposal_orchestrator._action_candidates_for_issues",
        candidates,
    )

    result = scan_deterministic_candidates(package, prevalidate_action_candidates=False)

    assert result.candidates == []
    assert result.issue_routes[0].route_state == "blocked_apply"
    assert result.issue_routes[0].reason_code == "APPLY_UNSUPPORTED"
    assert result.rejections[0]["error_code"] == "APPLY_UNSUPPORTED"
    assert result.rejections[0]["candidate"]["candidate_id"] == "candidate-alias-merge"
    _assert_candidate_rejection_contract(result.rejections[0])


def test_scan_valid_candidate_overrides_unsupported_candidate_for_same_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _make_ledger_package(
        tmp_path,
        issues=[_ledger_issue("edge-mixed")],
        edges=[_ledger_edge("edge-mixed")],
    )
    issue_ref = normalize_raw_issues(
        json.loads(
            (package / "snapshots" / "quality_score.json").read_text(encoding="utf-8")
        )
    )[0]["issue_ref"]

    def candidates(*args: object, **kwargs: object) -> CandidateGenerationResult:
        return CandidateGenerationResult(
            candidates=[
                {
                    "candidate_id": "candidate-unsupported",
                    "proposal_type": "entity_alias_merge",
                    "target": "node:flu-short",
                    "issue_ref": issue_ref,
                    "issue_kind": "synonym_duplicate",
                    "action_payload": {"action": "entity_alias_merge"},
                },
                {
                    "candidate_id": "candidate-supported",
                    "proposal_type": "medical_relation_schema_migration",
                    "target": "edge:edge-mixed",
                    "issue_ref": issue_ref,
                    "issue_kind": "legacy_overloaded_relation",
                    "action_payload": {
                        "action": "replace_relation",
                        "edge_id": "edge-mixed",
                        "expected_source": "flu",
                        "expected_target": "oseltamivir",
                        "current_keywords": "recommended_treatment",
                        "new_source": "oseltamivir",
                        "new_target": "flu",
                        "new_keywords": "has_indication",
                        "qualifiers": {},
                    },
                },
            ],
            rejections=[],
        )

    monkeypatch.setattr(
        "lightrag.kb_iteration.proposal_orchestrator._action_candidates_for_issues",
        candidates,
    )

    result = scan_deterministic_candidates(package, prevalidate_action_candidates=False)

    assert [candidate["candidate_id"] for candidate in result.candidates] == [
        "candidate-supported"
    ]
    assert result.issue_routes[0].route_state == "deterministic_covered"
    assert result.issue_routes[0].candidate_ids == ["candidate-supported"]
    assert result.issue_routes[0].reason_code == "DETERMINISTIC_CANDIDATE_VALID"
    assert [rejection["error_code"] for rejection in result.rejections] == [
        "APPLY_UNSUPPORTED"
    ]
    _assert_candidate_rejection_contract(result.rejections[0])


def test_scan_dedupes_exact_candidate_fingerprints_after_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _make_ledger_package(
        tmp_path,
        issues=[
            _ledger_issue("edge-duplicate-a"),
            _ledger_issue("edge-duplicate-b"),
        ],
        edges=[
            _ledger_edge("edge-duplicate-a"),
            _ledger_edge("edge-duplicate-b"),
        ],
    )
    issue_refs = [
        issue["issue_ref"]
        for issue in normalize_raw_issues(
            json.loads(
                (package / "snapshots" / "quality_score.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    ]
    action_payload = {
        "action": "replace_relation",
        "edge_id": "edge-duplicate-a",
        "expected_source": "flu",
        "expected_target": "oseltamivir",
        "current_keywords": "recommended_treatment",
        "new_source": "oseltamivir",
        "new_target": "flu",
        "new_keywords": "has_indication",
        "qualifiers": {},
    }

    def candidates(*args: object, **kwargs: object) -> CandidateGenerationResult:
        return CandidateGenerationResult(
            candidates=[
                {
                    "candidate_id": "candidate-duplicate-a",
                    "proposal_type": "medical_relation_schema_migration",
                    "target": "edge:edge-duplicate-a",
                    "issue_ref": issue_refs[0],
                    "issue_kind": "legacy_overloaded_relation",
                    "action_payload": dict(action_payload),
                },
                {
                    "candidate_id": "candidate-duplicate-b",
                    "proposal_type": "medical_relation_schema_migration",
                    "target": "edge:edge-duplicate-a",
                    "issue_ref": issue_refs[1],
                    "issue_kind": "legacy_overloaded_relation",
                    "action_payload": dict(action_payload),
                },
            ],
            rejections=[],
        )

    monkeypatch.setattr(
        "lightrag.kb_iteration.proposal_orchestrator._action_candidates_for_issues",
        candidates,
    )

    result = scan_deterministic_candidates(package, prevalidate_action_candidates=False)

    assert [candidate["candidate_id"] for candidate in result.candidates] == [
        "candidate-duplicate-a"
    ]
    assert [route.route_state for route in result.issue_routes] == [
        "deterministic_covered",
        "duplicate",
    ]
    assert result.issue_routes[1].reason_code == "DUPLICATE_ACTION_FINGERPRINT"
    assert result.rejections[0]["error_code"] == "DUPLICATE_ACTION_FINGERPRINT"
    _assert_candidate_rejection_contract(result.rejections[0])


def test_scan_blocks_candidate_from_scoped_decision_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = _make_ledger_package(
        tmp_path,
        issues=[_ledger_issue("edge-memory")],
        edges=[_ledger_edge("edge-memory")],
    )
    issue_ref = normalize_raw_issues(
        json.loads(
            (package / "snapshots" / "quality_score.json").read_text(
                encoding="utf-8"
            )
        )
    )[0]["issue_ref"]
    candidate = {
        "candidate_id": "candidate-memory-v2",
        "proposal_type": "medical_relation_schema_migration",
        "target": "edge:edge-memory",
        "issue_ref": issue_ref,
        "issue_kind": "legacy_overloaded_relation",
        "action_payload": {
            "action": "replace_relation",
            "edge_id": "edge-memory",
            "expected_source": "flu",
            "expected_target": "oseltamivir",
            "current_keywords": "recommended_treatment",
            "new_source": "oseltamivir",
            "new_target": "flu",
            "new_keywords": "has_indication",
            "qualifiers": {},
        },
    }
    fingerprints = candidate_fingerprints(candidate)
    (package / "rejected_changes.md").write_text(
        "# Rejected Changes\n\n"
        "## prop-action-candidate-memory-v1\n\n"
        "```json\n"
        + json.dumps(
            {
                "proposal_id": "prop-action-candidate-memory-v1",
                "proposal_type": "medical_relation_schema_migration",
                "proposal_target": "edge:edge-memory",
                "decision": "reject",
                "rejection_scope": "exact_action",
                "schema_version": "medical_relation_schema_v1",
                "semantic_fingerprint": fingerprints.semantic,
                "execution_fingerprint": fingerprints.execution,
                "evidence_fingerprint": fingerprints.evidence,
                "action_payload": candidate["action_payload"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n```\n",
        encoding="utf-8",
    )

    def candidates(*args: object, **kwargs: object) -> CandidateGenerationResult:
        return CandidateGenerationResult(candidates=[candidate], rejections=[])

    monkeypatch.setattr(
        "lightrag.kb_iteration.proposal_orchestrator._action_candidates_for_issues",
        candidates,
    )

    result = scan_deterministic_candidates(package, prevalidate_action_candidates=False)

    assert result.candidates == []
    assert result.rejections[0]["error_code"] == "REJECTED_DECISION_MEMORY"
    assert result.rejections[0]["generation_disposition"] == "blocked_decision_memory"
    _assert_candidate_rejection_contract(result.rejections[0])
    assert result.issue_routes[0].route_state == "duplicate"
    assert result.issue_routes[0].generation_disposition == "blocked_decision_memory"


def test_partial_modern_decision_record_does_not_enter_legacy_rejected_memory(
    tmp_path: Path,
) -> None:
    package = _make_ledger_package(
        tmp_path,
        issues=[_ledger_issue("edge-partial-modern")],
        edges=[_ledger_edge("edge-partial-modern")],
    )
    (package / "rejected_changes.md").write_text(
        "# Rejected Changes\n\n"
        "## prop-action-candidate-partial-modern\n\n"
        "```json\n"
        + json.dumps(
            {
                "proposal_id": "prop-action-candidate-partial-modern",
                "proposal_type": "medical_relation_schema_migration",
                "proposal_target": "edge:edge-partial-modern",
                "decision": "reject",
                "rejection_scope": "exact_action",
                "schema_version": "medical_relation_schema_v1",
                "semantic_fingerprint": "abc",
                "action_payload": {
                    "action": "replace_relation",
                    "edge_id": "edge-partial-modern",
                    "expected_source": "flu",
                    "expected_target": "oseltamivir",
                    "current_keywords": "recommended_treatment",
                    "new_source": "oseltamivir",
                    "new_target": "flu",
                    "new_keywords": "has_indication",
                    "qualifiers": {},
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n```\n",
        encoding="utf-8",
    )

    assert _rejected_action_fingerprints(package) == set()


def test_deterministic_family_caps_defer_overflow_after_dedupe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issues = [
        _ledger_issue(
            f"edge-diagnosis-{index}",
            issue_kind="diagnostic_evidence_direction_mismatch",
            candidate_predicates=["supports_or_refutes"],
        )
        | {"issue_family": "diagnosis"}
        for index in range(2)
    ]
    package = _make_ledger_package(tmp_path, issues=issues)

    def candidates(
        chunk: list[dict[str, object]],
        **_kwargs: object,
    ) -> CandidateGenerationResult:
        return CandidateGenerationResult(
            candidates=[
                {
                    "proposal_type": "medical_relation_schema_migration",
                    "target": f"edge:{issue['edge_id']}",
                    "issue_ref": issue["issue_ref"],
                    "issue_kind": issue["issue_kind"],
                    "action_payload": {
                        "action": "replace_relation",
                        "edge_id": issue["edge_id"],
                        "expected_source": issue["source"],
                        "expected_target": issue["target"],
                        "current_keywords": issue["keywords"],
                        "new_source": issue["source"],
                        "new_target": issue["target"],
                        "new_keywords": "supports_or_refutes",
                        "qualifiers": {"polarity": "supports"},
                    },
                }
                for issue in chunk
            ],
            rejections=[],
        )

    monkeypatch.setattr(
        "lightrag.kb_iteration.proposal_orchestrator._action_candidates_for_issues",
        candidates,
    )

    result = scan_deterministic_candidates(
        package,
        prevalidate_action_candidates=False,
        deterministic_family_caps={"diagnosis": 1},
    )

    assert len(result.candidates) == 1
    assert [route.route_state for route in result.issue_routes] == [
        "deterministic_covered",
        "deferred_budget",
    ]
    assert result.issue_routes[1].reason_code == "FAMILY_CAP_REACHED"
    assert result.rejections[0]["error_code"] == "FAMILY_CAP_REACHED"
    _assert_candidate_rejection_contract(
        result.rejections[0],
        issue_family="diagnosis",
    )


def test_deterministic_candidates_are_sorted_by_family_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issues = [
        _ledger_issue(
            "edge-prevention",
            issue_kind="prevention_population_direction_mismatch",
            candidate_predicates=["recommended_for"],
        )
        | {"issue_family": "prevention"},
        _ledger_issue(
            "edge-diagnosis",
            issue_kind="diagnostic_evidence_direction_mismatch",
            candidate_predicates=["supports_or_refutes"],
        )
        | {"issue_family": "diagnosis"},
    ]
    package = _make_ledger_package(tmp_path, issues=issues)

    def candidates(
        chunk: list[dict[str, object]],
        **_kwargs: object,
    ) -> CandidateGenerationResult:
        return CandidateGenerationResult(
            candidates=[
                {
                    "proposal_type": "medical_relation_schema_migration",
                    "target": f"edge:{issue['edge_id']}",
                    "issue_ref": issue["issue_ref"],
                    "issue_kind": issue["issue_kind"],
                    "action_payload": {
                        "action": "replace_relation",
                        "edge_id": issue["edge_id"],
                        "expected_source": issue["source"],
                        "expected_target": issue["target"],
                        "current_keywords": issue["keywords"],
                        "new_source": issue["source"],
                        "new_target": issue["target"],
                        "new_keywords": issue["candidate_predicates"][0],
                        "qualifiers": {"polarity": "supports"},
                    },
                }
                for issue in chunk
            ],
            rejections=[],
        )

    monkeypatch.setattr(
        "lightrag.kb_iteration.proposal_orchestrator._action_candidates_for_issues",
        candidates,
    )

    result = scan_deterministic_candidates(
        package,
        prevalidate_action_candidates=False,
    )

    assert [candidate["target"] for candidate in result.candidates] == [
        "edge:edge-diagnosis",
        "edge:edge-prevention",
    ]


def test_diagnosis_generator_builds_supports_or_refutes_from_quality_polarity(
    tmp_path: Path,
) -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode("positive-pcr", "Positive PCR", "TestResultPattern"),
            SnapshotNode("flu", "Influenza", "Disease"),
        ],
        edges=[
            SnapshotEdge(
                "edge-positive-pcr-flu",
                "positive-pcr",
                "flu",
                "diagnostic_basis",
                source_id="chunk-1",
                file_path="guide.md",
                properties={"qualifiers": {"polarity": "supports"}},
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )
    package = _make_quality_package(tmp_path, snapshot)

    scan = scan_deterministic_candidates(package)
    proposal_build = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )

    assert proposal_build.rejected == []
    assert len(proposal_build.proposals) == 1
    proposal = proposal_build.proposals[0]
    validate_proposal(proposal)
    assert proposal.action_payload["new_source"] == "positive-pcr"
    assert proposal.action_payload["new_target"] == "flu"
    assert proposal.action_payload["new_keywords"] == "supports_or_refutes"
    assert proposal.action_payload["qualifiers"] == {"polarity": "supports"}
    assert scan.issue_routes[0].route_state == "deterministic_covered"
    assert scan.issue_routes[0].proposal_ids == [proposal.id]


def test_diagnosis_generator_leaves_test_result_without_polarity_residual(
    tmp_path: Path,
) -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode("positive-pcr", "Positive PCR", "TestResultPattern"),
            SnapshotNode("flu", "Influenza", "Disease"),
        ],
        edges=[
            SnapshotEdge(
                "edge-positive-pcr-flu",
                "positive-pcr",
                "flu",
                "diagnostic_basis",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )
    package = _make_quality_package(tmp_path, snapshot)

    scan = scan_deterministic_candidates(package)

    assert scan.candidates == []
    assert scan.issue_routes[0].route_state == "llm_residual"
    assert scan.issue_routes[0].reason_code == "SUPPORTS_OR_REFUTES_POLARITY_REQUIRED"
    assert scan.rejections[0]["error_code"] == "SUPPORTS_OR_REFUTES_POLARITY_REQUIRED"
    _assert_candidate_rejection_contract(
        scan.rejections[0],
        issue_family="diagnosis",
    )


def test_treatment_generator_rejects_safety_candidate_without_reason(
    tmp_path: Path,
) -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode("live-vaccine", "Live attenuated vaccine", "Vaccine"),
            SnapshotNode("pregnancy", "Pregnancy", "Population"),
        ],
        edges=[
            SnapshotEdge(
                "edge-vaccine-pregnancy",
                "live-vaccine",
                "pregnancy",
                "temporarily_deferred_for",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )
    package = _make_quality_package(tmp_path, snapshot)

    scan = scan_deterministic_candidates(package)

    assert scan.candidates == []
    assert scan.issue_routes[0].route_state == "llm_residual"
    assert scan.issue_routes[0].reason_code == "SAFETY_REASON_QUALIFIER_REQUIRED"
    assert scan.rejections[0]["error_code"] == "SAFETY_REASON_QUALIFIER_REQUIRED"


def test_prevention_recommended_for_sets_prevention_purpose_from_quality_output(
    tmp_path: Path,
) -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode("flu-vaccine", "Influenza vaccine", "Vaccine"),
            SnapshotNode("older-adults", "Older adults", "Population"),
        ],
        edges=[
            SnapshotEdge(
                "edge-vaccine-older-adults",
                "flu-vaccine",
                "older-adults",
                "applies_to",
                source_id="chunk-1",
                file_path="guide.md",
                properties={"qualifiers": {"age_min": "65", "age_unit": "year"}},
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )
    package = _make_quality_package(tmp_path, snapshot)

    scan = scan_deterministic_candidates(package)

    assert len(scan.candidates) == 1
    candidate = scan.candidates[0]
    assert candidate["action_payload"]["new_keywords"] == "recommended_for"
    assert candidate["action_payload"]["qualifiers"] == {
        "age_min": "65",
        "age_unit": "year",
        "purpose": "prevention",
    }


def test_risk_safety_generator_rejects_outcome_as_risk_factor_candidate(
    tmp_path: Path,
) -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode("flu", "Influenza", "Disease"),
            SnapshotNode("death", "Death", "Outcome"),
        ],
        edges=[
            SnapshotEdge(
                "edge-flu-death",
                "flu",
                "death",
                "complication_risk",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )
    package = _make_quality_package(tmp_path, snapshot)

    scan = scan_deterministic_candidates(package)

    assert scan.candidates == []
    assert scan.issue_routes[0].route_state == "llm_residual"
    assert scan.issue_routes[0].reason_code == "OUTCOME_OR_SEVERITY_NOT_RISK_FACTOR"
    assert scan.rejections[0]["error_code"] == "OUTCOME_OR_SEVERITY_NOT_RISK_FACTOR"


def test_risk_safety_generator_allows_risk_factor_to_outcome(
    tmp_path: Path,
) -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode("chronic-disease", "Chronic cardiopulmonary disease", "RiskFactor"),
            SnapshotNode("hospitalization", "Hospitalization", "Outcome"),
        ],
        edges=[
            SnapshotEdge(
                "edge-risk-factor-hospitalization",
                "chronic-disease",
                "hospitalization",
                "complication_risk",
                source_id="chunk-1",
                file_path="guide.md",
            )
        ],
        metadata={"profile": "clinical_guideline_zh"},
    )
    package = _make_quality_package(tmp_path, snapshot)

    scan = scan_deterministic_candidates(package)

    assert len(scan.candidates) == 1
    candidate = scan.candidates[0]
    assert candidate["action_payload"]["new_keywords"] == "risk_factor_for"
    assert scan.issue_routes[0].route_state == "deterministic_covered"

    proposal_build = action_candidate_proposals_from_scan(
        scan,
        output_dir=package,
        previous_outputs={},
    )
    assert proposal_build.rejected == []
    assert len(proposal_build.proposals) == 1
    proposal = proposal_build.proposals[0]
    validate_proposal(proposal)
    assert proposal.action_payload["new_keywords"] == "risk_factor_for"
    assert scan.issue_routes[0].proposal_ids == [proposal.id]


def test_risk_safety_generator_uses_chinese_risk_increase_text(
    tmp_path: Path,
) -> None:
    package = _make_ledger_package(
        tmp_path,
        issues=[
            {
                "issue_kind": "legacy_overloaded_relation",
                "edge_id": "edge-risk-increase",
                "source": "chronic-disease",
                "source_type": "ClinicalCondition",
                "target": "hospitalization",
                "target_type": "Outcome",
                "keywords": "complication_risk",
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "evidence_quote": "慢性基础疾病会增加住院风险。",
                "suggested_action": "replace_relation",
                "candidate_predicates": ["risk_factor_for", "increases_risk_of"],
            }
        ],
        edges=[
            {
                **_ledger_edge(
                    "edge-risk-increase",
                    source="chronic-disease",
                    target="hospitalization",
                    keywords="complication_risk",
                ),
                "source_type": "ClinicalCondition",
                "target_type": "Outcome",
            }
        ],
    )

    scan = scan_deterministic_candidates(package, prevalidate_action_candidates=False)

    assert len(scan.candidates) == 1
    candidate = scan.candidates[0]
    assert candidate["action_payload"]["new_keywords"] == "increases_risk_of"


def test_entity_merge_quality_issue_routes_blocked_apply_with_precise_reason(
    tmp_path: Path,
) -> None:
    snapshot = KGSnapshot(
        workspace="medical_demo",
        generated_at="2026-06-23T00:00:00+08:00",
        source_files=["guide.md"],
        nodes=[
            SnapshotNode("flu", "Influenza", "Disease"),
            SnapshotNode("influenza", "Influenza", "Disease"),
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )
    package = _make_quality_package(tmp_path, snapshot)

    scan = scan_deterministic_candidates(package)

    assert scan.candidates == []
    assert scan.issue_routes[0].route_state == "blocked_apply"
    assert scan.issue_routes[0].reason_code == "ENTITY_MERGE_APPLY_NOT_SUPPORTED"
    assert scan.rejections[0]["error_code"] == "ENTITY_MERGE_APPLY_NOT_SUPPORTED"


def _make_ledger_package(
    tmp_path: Path,
    *,
    issues: list[dict[str, object]],
    edges: list[dict[str, object]] | None = None,
) -> Path:
    package = tmp_path / "kb-iteration"
    snapshot_edges = edges or [_ledger_edge(str(issue["edge_id"])) for issue in issues]
    _write_json(
        package / "snapshots" / "kg_snapshot.json",
        {
            "workspace": "medical-demo",
            "generated_at": "2026-06-23T00:00:00+08:00",
            "source_files": ["guide.md"],
            "nodes": [],
            "edges": snapshot_edges,
            "metadata": {"profile": "clinical_guideline_zh"},
        },
    )
    _write_json(
        package / "snapshots" / "quality_score.json",
        {
            "overall": 70,
            "subscores": {},
            "metrics": {"medical_schema_issue_count": len(issues)},
            "details": {"medical_schema_issues": issues},
            "findings": [],
            "critical_blockers": [],
        },
    )
    (package / "accepted_changes.md").write_text("# Accepted Changes\n", encoding="utf-8")
    (package / "rejected_changes.md").write_text("# Rejected Changes\n", encoding="utf-8")
    return package


def _make_quality_package(tmp_path: Path, snapshot: KGSnapshot) -> Path:
    package = tmp_path / "kb-iteration"
    score = evaluate_snapshot_quality(snapshot)
    _write_json(package / "snapshots" / "kg_snapshot.json", snapshot.to_dict())
    _write_json(package / "snapshots" / "quality_score.json", score.to_dict())
    (package / "accepted_changes.md").write_text("# Accepted Changes\n", encoding="utf-8")
    (package / "rejected_changes.md").write_text("# Rejected Changes\n", encoding="utf-8")
    return package


def _ledger_issue(
    edge_id: str,
    *,
    issue_kind: str = "legacy_overloaded_relation",
    source: str = "flu",
    target: str = "oseltamivir",
    keywords: str = "recommended_treatment",
    source_id: str = "chunk-1",
    file_path: str = "guide.md",
    suggested_action: str = "replace_relation",
    candidate_predicates: list[str] | None = None,
) -> dict[str, object]:
    return {
        "issue_kind": issue_kind,
        "edge_id": edge_id,
        "source": source,
        "target": target,
        "keywords": keywords,
        "source_id": source_id,
        "file_path": file_path,
        "suggested_action": suggested_action,
        "candidate_predicates": candidate_predicates or ["has_indication"],
    }


def _ledger_edge(
    edge_id: str,
    *,
    source: str = "flu",
    target: str = "oseltamivir",
    keywords: str = "recommended_treatment",
) -> dict[str, object]:
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "keywords": keywords,
        "description": "",
        "source_id": "chunk-1",
        "file_path": "guide.md",
        "weight": None,
        "properties": {},
    }


def _candidate_gate_payload(edge_id: str) -> dict[str, object]:
    if edge_id == "edge-missing-types":
        return {
            "action": "candidate_kg_expansion",
            "candidate_nodes": [],
            "candidate_edges": [
                {
                    "source": "flu",
                    "target": "fever",
                    "keywords": "has_manifestation",
                }
            ],
            "source_id": "chunk-1",
            "file_path": "guide.md",
            "evidence_quote": "Influenza may present with fever.",
        }
    return {
        "action": "replace_relation",
        "edge_id": edge_id,
        "expected_source": "flu",
        "expected_target": "oseltamivir",
        "current_keywords": "recommended_treatment",
        "new_source": "flu-vaccine" if edge_id == "edge-known-bad" else "flu",
        "new_target": "older adults" if edge_id == "edge-known-bad" else "oseltamivir",
        "new_keywords": "has_indication",
        "qualifiers": {},
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _assert_candidate_rejection_contract(
    rejection: dict[str, object],
    *,
    issue_family: str | None = None,
) -> None:
    for field_name in (
        "candidate_id",
        "issue_ref",
        "issue_family",
        "stage",
        "error_code",
        "error",
    ):
        assert str(rejection.get(field_name) or "").strip(), field_name
    if issue_family is not None:
        assert rejection["issue_family"] == issue_family
