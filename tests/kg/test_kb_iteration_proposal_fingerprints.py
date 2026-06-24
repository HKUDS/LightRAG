from __future__ import annotations

from typing import Any

from lightrag.kb_iteration.proposal_fingerprints import (
    candidate_fingerprints,
    decision_fingerprints,
    decision_memory_suppresses_candidate,
    proposal_fingerprints,
)


def test_replace_relation_fingerprints_ignore_candidate_id_suffixes() -> None:
    first = _candidate(candidate_id="candidate-treatment-v1")
    rerun = _candidate(candidate_id="candidate-treatment-v1-rerun-003")

    assert candidate_fingerprints(first) == candidate_fingerprints(rerun)


def test_replace_relation_fingerprints_split_semantic_execution_and_evidence() -> None:
    original = candidate_fingerprints(_candidate())
    evidence_changed = candidate_fingerprints(
        _candidate(evidence_quote="Oseltamivir works best when started early.")
    )
    current_keywords_changed = candidate_fingerprints(
        _candidate(current_keywords="legacy_treatment_relation")
    )
    qualifier_changed = candidate_fingerprints(
        _candidate(qualifiers={"purpose": "post_exposure_prophylaxis"})
    )

    assert evidence_changed.semantic == original.semantic
    assert evidence_changed.execution == original.execution
    assert evidence_changed.evidence != original.evidence
    assert current_keywords_changed.semantic == original.semantic
    assert current_keywords_changed.execution != original.execution
    assert current_keywords_changed.evidence == original.evidence
    assert qualifier_changed.semantic != original.semantic


def test_parseable_evidence_strings_ignore_field_order() -> None:
    original = proposal_fingerprints(
        _proposal(
            evidence=[
                (
                    "source_id: chunk-1; file_path: guide.md; "
                    "relation_id: edge-oseltamivir-flu; "
                    "evidence_quote: Oseltamivir is indicated for confirmed influenza."
                )
            ]
        )
    )
    reordered = proposal_fingerprints(
        _proposal(
            evidence=[
                (
                    "file_path: guide.md; evidence_quote: Oseltamivir is indicated "
                    "for confirmed influenza.; relation_id: edge-oseltamivir-flu; "
                    "source_id: chunk-1"
                )
            ]
        )
    )

    assert reordered.evidence == original.evidence


def test_parseable_evidence_strings_change_when_grounded_fields_change() -> None:
    original = proposal_fingerprints(
        _proposal(
            evidence=[
                (
                    "source_id: chunk-1; file_path: guide.md; "
                    "evidence_quote: Oseltamivir is indicated for confirmed influenza."
                )
            ]
        )
    )
    changed_quote = proposal_fingerprints(
        _proposal(
            evidence=[
                (
                    "source_id: chunk-1; file_path: guide.md; "
                    "evidence_quote: Oseltamivir should be started early."
                )
            ]
        )
    )
    changed_source = proposal_fingerprints(
        _proposal(
            evidence=[
                (
                    "source_id: chunk-2; file_path: guide.md; "
                    "evidence_quote: Oseltamivir is indicated for confirmed influenza."
                )
            ]
        )
    )
    changed_file = proposal_fingerprints(
        _proposal(
            evidence=[
                (
                    "source_id: chunk-1; file_path: appendix.md; "
                    "evidence_quote: Oseltamivir is indicated for confirmed influenza."
                )
            ]
        )
    )

    assert changed_quote.evidence != original.evidence
    assert changed_source.evidence != original.evidence
    assert changed_file.evidence != original.evidence


def test_partially_parseable_evidence_strings_preserve_unrecognized_text() -> None:
    first = proposal_fingerprints(
        _proposal(evidence=["source_id: chunk-1; clinical note says: A"])
    )
    second = proposal_fingerprints(
        _proposal(evidence=["source_id: chunk-1; clinical note says: B"])
    )

    assert second.evidence != first.evidence


def test_decision_fingerprints_returns_none_for_legacy_payload_without_fingerprints() -> None:
    assert decision_fingerprints({"decision": "reject", "action_payload": {}}) is None


def test_scoped_decision_memory_respects_schema_and_evidence_scope() -> None:
    candidate = _candidate()
    fingerprints = candidate_fingerprints(candidate, schema_version="schema-v1")
    decision = {
        "decision": "reject",
        "rejection_scope": "semantic_until_schema_change",
        "schema_version": "schema-v1",
        "semantic_fingerprint": fingerprints.semantic,
        "execution_fingerprint": fingerprints.execution,
        "evidence_fingerprint": fingerprints.evidence,
    }

    assert (
        decision_memory_suppresses_candidate(
            candidate,
            [decision],
            schema_version="schema-v1",
        )
        is True
    )
    assert (
        decision_memory_suppresses_candidate(
            candidate,
            [decision],
            schema_version="schema-v2",
        )
        is False
    )


def _candidate(
    *,
    candidate_id: str = "candidate-treatment-v1",
    current_keywords: str = "recommended_treatment",
    evidence_quote: str = "Oseltamivir is indicated for confirmed influenza.",
    qualifiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "proposal_type": "medical_relation_schema_migration",
        "target": "edge:edge-oseltamivir-flu",
        "schema_version": "schema-v1",
        "evidence_spans": [
            {
                "source_id": "chunk-1",
                "file_path": "guide.md",
                "evidence_quote": evidence_quote,
            }
        ],
        "action_payload": {
            "action": "replace_relation",
            "edge_id": "edge-oseltamivir-flu",
            "expected_source": "oseltamivir",
            "expected_target": "influenza",
            "current_keywords": current_keywords,
            "new_source": "oseltamivir",
            "new_target": "influenza",
            "new_keywords": "has_indication",
            "qualifiers": qualifiers or {"purpose": "treatment"},
        },
    }


def _proposal(*, evidence: list[str]) -> dict[str, Any]:
    return {
        "id": "proposal-supported-replace-relation",
        "type": "medical_relation_schema_migration",
        "target": "edge:edge-oseltamivir-flu",
        "evidence": evidence,
        "action_payload": {
            "action": "replace_relation",
            "edge_id": "edge-oseltamivir-flu",
            "expected_source": "oseltamivir",
            "expected_target": "influenza",
            "current_keywords": "recommended_treatment",
            "new_source": "oseltamivir",
            "new_target": "influenza",
            "new_keywords": "has_indication",
            "qualifiers": {"purpose": "treatment"},
        },
    }
