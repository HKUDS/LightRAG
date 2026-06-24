from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kb_iteration.apply import (
    APPLY_SOURCE,
    APPLY_RESULT_JSON,
    APPLY_RESULT_MARKDOWN,
    ApplyChangeStatus,
    AcceptedApplyResult,
    ApplyChange,
    PROPOSAL_APPLY_HANDLER_REGISTRY,
    apply_accepted_changes_to_graph,
    branch_key_from_proposal,
    category_for_branch_key,
    load_proposals_by_id,
    proposal_apply_capability,
    render_apply_result_markdown,
    write_apply_result_artifacts,
)
from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.models import KGSnapshot, SnapshotNode
from lightrag.kb_iteration.proposals import (
    validate_proposal,
    write_approval_queue,
    write_improvement_backlog,
)
from lightrag.kb_iteration.quality import evaluate_snapshot_quality
from lightrag.medical_kg.ontology import TOP_LEVEL_MEDICAL_CATEGORIES


class FakeGraph:
    def __init__(self, existing_nodes: set[str] | None = None) -> None:
        self.nodes = {node_id: {} for node_id in existing_nodes or set()}
        self.edges: dict[tuple[str, str], dict] = {}
        self.upserted_nodes: list[tuple[str, dict]] = []
        self.upserted_edges: list[tuple[str, str, dict]] = []
        self.removed_edges: list[tuple[str, str]] = []
        self.removed_nodes: list[str] = []
        self.upsert_edge_failures: set[tuple[str, str]] = set()
        self.index_done_calls = 0

    async def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return (source_node_id, target_node_id) in self.edges

    async def get_node(self, node_id: str) -> dict | None:
        return self.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        if node_id not in self.nodes:
            return 0
        return sum(1 for source, target in self.edges if node_id in {source, target})

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict | None:
        edge = self.edges.get((source_node_id, target_node_id))
        return dict(edge) if edge is not None else None

    async def get_node_edges(self, node_id: str) -> list[tuple[str, str]] | None:
        if node_id not in self.nodes:
            return None
        return [
            (source_node_id, target_node_id)
            for source_node_id, target_node_id in self.edges
            if node_id in {source_node_id, target_node_id}
        ]

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict]]) -> None:
        self.upserted_nodes.extend(nodes)
        for node_id, node_data in nodes:
            self.nodes[node_id] = node_data

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict
    ) -> None:
        if (source_node_id, target_node_id) in self.upsert_edge_failures:
            raise RuntimeError("upsert failed")
        self.upserted_edges.append((source_node_id, target_node_id, edge_data))
        self.edges[(source_node_id, target_node_id)] = edge_data

    async def upsert_edges_batch(self, edges: list[tuple[str, str, dict]]) -> None:
        self.upserted_edges.extend(edges)
        for source_node_id, target_node_id, edge_data in edges:
            self.edges[(source_node_id, target_node_id)] = edge_data

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        self.removed_edges.extend(edges)
        for source_node_id, target_node_id in edges:
            self.edges.pop((source_node_id, target_node_id), None)

    async def remove_nodes(self, nodes: list[str]) -> None:
        self.removed_nodes.extend(nodes)
        for node_id in nodes:
            self.nodes.pop(node_id, None)
        for edge_key in list(self.edges):
            if any(node_id in edge_key for node_id in nodes):
                self.edges.pop(edge_key, None)

    async def index_done_callback(self) -> None:
        self.index_done_calls += 1


class UndirectedFakeGraph(FakeGraph):
    def _edge_key(self, source_node_id: str, target_node_id: str) -> tuple[str, str]:
        if (source_node_id, target_node_id) in self.edges:
            return (source_node_id, target_node_id)
        reverse_key = (target_node_id, source_node_id)
        if reverse_key in self.edges:
            return reverse_key
        return (source_node_id, target_node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._edge_key(source_node_id, target_node_id) in self.edges

    async def get_edge(self, source_node_id: str, target_node_id: str) -> dict | None:
        edge = self.edges.get(self._edge_key(source_node_id, target_node_id))
        return dict(edge) if edge is not None else None


class FakeRAG:
    def __init__(self, graph: FakeGraph | None = None) -> None:
        if graph is not None:
            self.chunk_entity_relation_graph = graph


class RecordingLock:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls

    async def __aenter__(self) -> None:
        self.calls[-1]["entered"] = True

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.calls[-1]["exited"] = True


def install_recording_keyed_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    lock_calls: list[dict[str, Any]] = []

    def fake_keyed_lock(
        keys: str | list[str],
        namespace: str = "default",
        enable_logging: bool = False,
    ) -> RecordingLock:
        lock_calls.append(
            {
                "keys": keys,
                "namespace": namespace,
                "enable_logging": enable_logging,
                "entered": False,
                "exited": False,
            }
        )
        return RecordingLock(lock_calls)

    monkeypatch.setattr(
        "lightrag.kb_iteration.apply.get_storage_keyed_lock",
        fake_keyed_lock,
    )
    return lock_calls


def _replace_relation_proposal(
    proposal_id: str = "proposal-replace-dry-cough-flu",
    **payload_overrides: object,
) -> dict[str, Any]:
    payload: dict[str, object] = {
        "action": "replace_relation",
        "edge_id": "edge-dry-cough-flu",
        "expected_source": "dry-cough",
        "expected_target": "flu",
        "new_source": "flu",
        "new_target": "dry-cough",
        "new_keywords": "has_manifestation",
        "current_keywords": "\u4e34\u5e8a\u8868\u73b0",
    }
    payload.update(payload_overrides)
    return {
        "id": proposal_id,
        "type": "medical_relation_schema_migration",
        "target": "edge:edge-dry-cough-flu",
        "proposed_change": "Reverse dry cough manifestation relation.",
        "reason": "Clinical manifestation edges should point disease to symptom.",
        "evidence": ["relation_id:edge-dry-cough-flu; source_id:chunk-1"],
        "action_payload": payload,
    }


def _retire_relation_proposal(
    proposal_id: str = "proposal-retire-category-edge",
    **payload_overrides: object,
) -> dict[str, Any]:
    payload: dict[str, object] = {
        "action": "retire_relation",
        "edge_id": "clinical-manifestations->flu",
        "expected_source": "clinical-manifestations",
        "expected_target": "flu",
        "current_keywords": "clinical_manifestation",
        "retirement_reason": "category node is not an atomic clinical manifestation",
    }
    payload.update(payload_overrides)
    return {
        "id": proposal_id,
        "type": "medical_relation_schema_migration",
        "target": "edge:clinical-manifestations->flu",
        "proposed_change": "Retire invalid category manifestation relation.",
        "reason": "The edge cannot be normalized without creating a false fact.",
        "evidence": ["relation_id:clinical-manifestations->flu; source_id:chunk-1"],
        "action_payload": payload,
    }


def _split_relation_proposal(
    proposal_id: str = "proposal-split-drug-relation",
    **payload_overrides: object,
) -> dict[str, Any]:
    payload: dict[str, object] = {
        "action": "split_relation",
        "edge_id": "edge-zanamivir-mixed",
        "expected_source": "zanamivir",
        "expected_target": "mixed-target",
        "current_keywords": "recommended_treatment,applies_to",
        "retire_original": True,
        "new_edges": [
            {
                "source": "zanamivir",
                "target": "influenza",
                "predicate": "has_indication",
                "qualifiers": {"purpose": "treatment"},
            },
            {
                "source": "zanamivir",
                "target": "children",
                "predicate": "recommended_for",
                "qualifiers": {
                    "purpose": "treatment",
                    "age_min": 7,
                    "age_unit": "year",
                    "route": "inhalation",
                },
            },
        ],
    }
    payload.update(payload_overrides)
    return {
        "id": proposal_id,
        "type": "medical_fact_role_split",
        "target": "edge:edge-zanamivir-mixed",
        "proposed_change": "Split overloaded zanamivir relation.",
        "reason": "The relation mixes indication and population recommendation semantics.",
        "evidence": ["relation_id:edge-zanamivir-mixed; source_id:chunk-1"],
        "action_payload": payload,
    }


def test_proposal_apply_capability_reports_supported_handlers() -> None:
    expected_supported = {
        ("add_hierarchy_branch", "add_hierarchy_branch"),
        ("medical_relation_schema_migration", "replace_relation"),
        ("medical_relation_schema_migration", "retire_relation"),
        ("medical_fact_role_split", "split_relation"),
        ("value_node_to_qualifier", "value_node_to_qualifier"),
        ("candidate_kg_expansion", "candidate_kg_expansion"),
    }

    assert set(PROPOSAL_APPLY_HANDLER_REGISTRY) == expected_supported
    for proposal_type, action in PROPOSAL_APPLY_HANDLER_REGISTRY:
        proposal = {
            "type": proposal_type,
            "action_payload": {"action": action} if action != proposal_type else {},
        }

        capability = proposal_apply_capability(proposal)

        assert capability.supported is True
        assert capability.action == action
        assert capability.reason == ""


def test_proposal_apply_capability_blocks_entity_alias_merge() -> None:
    proposal = {
        "type": "entity_alias_merge",
        "action_payload": {"action": "entity_alias_merge"},
    }

    capability = proposal_apply_capability(proposal)

    assert capability.supported is False
    assert capability.action == "entity_alias_merge"
    assert "Unsupported" in capability.reason


def test_add_hierarchy_branch_is_a_known_mutation_type() -> None:
    proposal = ImprovementProposal(
        id="prop-add-branch-diagnosis",
        type="add_hierarchy_branch",
        target="hierarchy",
        proposed_change="Create branch diagnosis_testing.",
        reason="Missing required branch diagnosis_testing.",
        evidence=["item_id: influenza; source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={"hierarchy_missing_branch_count": -1},
    )

    validate_proposal(proposal)


def test_add_hierarchy_branch_without_approval_requires_known_mutation_type() -> None:
    proposal = ImprovementProposal(
        id="prop-add-branch-diagnosis",
        type="add_hierarchy_branch",
        target="hierarchy",
        proposed_change="Create branch diagnosis_testing.",
        reason="Missing required branch diagnosis_testing.",
        evidence=["item_id: influenza; source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=False,
        expected_metric_change={"hierarchy_missing_branch_count": -1},
    )

    with pytest.raises(
        ValueError, match="^proposal type add_hierarchy_branch requires approval$"
    ):
        validate_proposal(proposal)


def test_medical_schema_migration_proposal_requires_approval_and_action_payload(
    tmp_path: Path,
) -> None:
    proposal = ImprovementProposal(
        id="prop-normalize-dry-cough-flu",
        type="medical_relation_schema_migration",
        target="edge:edge-dry-cough-flu",
        proposed_change=(
            "Reverse manifestation edge and set canonical predicate has_manifestation."
        ),
        reason="Symptom-to-disease clinical manifestation direction is invalid.",
        evidence=[
            "relation_id:edge-dry-cough-flu; item_id:dry-cough,flu; source_id:chunk-1"
        ],
        confidence=0.86,
        risk="medium",
        requires_approval=True,
        expected_metric_change={"relation_semantic_issue_count": -1},
        action_payload={
            "action": "replace_relation",
            "edge_id": "edge-dry-cough-flu",
            "expected_source": "dry-cough",
            "expected_target": "flu",
            "new_source": "flu",
            "new_target": "dry-cough",
            "new_keywords": "has_manifestation",
            "current_keywords": "临床表现",
        },
    )

    validate_proposal(proposal)
    write_approval_queue([proposal], tmp_path)

    text = (tmp_path / "approval_queue.md").read_text(encoding="utf-8")
    assert "action_payload:" in text
    assert "replace_relation" in text


def test_medical_schema_migration_without_human_approval_is_rejected() -> None:
    proposal = ImprovementProposal(
        id="prop-normalize-unsafe",
        type="medical_relation_schema_migration",
        target="edge:e1",
        proposed_change="Normalize relation.",
        reason="Mutation must be approved.",
        evidence=["relation_id:e1"],
        confidence=0.5,
        risk="medium",
        requires_approval=False,
        expected_metric_change={},
        action_payload={"action": "replace_relation", "edge_id": "e1"},
    )

    with pytest.raises(ValueError, match="requires approval"):
        validate_proposal(proposal)


def test_load_proposals_by_id_reads_queue_and_backlog_yaml_sections(
    tmp_path: Path,
) -> None:
    queued = ImprovementProposal(
        id="proposal-queue-diagnosis",
        type="add_hierarchy_branch",
        target="hierarchy",
        proposed_change="Create branch diagnosis_testing.",
        reason="Missing required branch diagnosis_testing.",
        evidence=["item_id: influenza; source_id: chunk-1"],
        confidence=0.9,
        risk="low",
        requires_approval=True,
        expected_metric_change={"hierarchy_missing_branch_count": -1},
    )
    backlog = ImprovementProposal(
        id="proposal-backlog-note",
        type="quality_report_note",
        target="quality_report.md",
        proposed_change="Record a quality observation.",
        reason="Reviewer context should be retained.",
        evidence=[],
        confidence=0.4,
        risk="low",
        requires_approval=False,
        expected_metric_change={},
    )
    write_approval_queue([queued, backlog], tmp_path)
    write_improvement_backlog([queued, backlog], tmp_path)

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-queue-diagnosis", "proposal-backlog-note"}
    assert isinstance(proposals["proposal-queue-diagnosis"], dict)
    assert proposals["proposal-queue-diagnosis"]["proposed_change"] == (
        "Create branch diagnosis_testing."
    )
    assert proposals["proposal-backlog-note"]["target"] == "quality_report.md"


def test_load_proposals_by_id_preserves_unknown_future_keys(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-future-key
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record future metadata.
    reason: Future queue records may include extra fields.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
    future_reviewer: retained
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert proposals["proposal-future-key"]["future_reviewer"] == "retained"


def test_load_proposals_by_id_skips_invalid_malformed_optional_field(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-invalid-evidence
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record invalid evidence shape.
    reason: Enough fields are present for validation.
    evidence: not-a-list
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
  - id: proposal-valid-note
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record valid evidence shape.
    reason: Valid proposals should still load.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-valid-note"}


def test_load_proposals_by_id_skips_incomplete_records_without_id(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - type: quality_report_note
    target: quality_report.md
    proposed_change: Record missing id.
    reason: Missing id records are incomplete.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
  - id: proposal-with-id
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record present id.
    reason: Complete records should still load.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-with-id"}


def test_load_proposals_by_id_skips_records_missing_required_fields(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-incomplete
    type: quality_report_note
  - id: proposal-complete
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record complete proposal.
    reason: Complete records should load.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-complete"}


def test_load_proposals_by_id_skips_records_missing_risk(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-missing-risk
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record missing risk.
    reason: Records missing renderer fields are incomplete.
    evidence: []
    confidence: 0.4
    requires_approval: false
    expected_metric_change: {}
  - id: proposal-complete-risk
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record complete risk.
    reason: Complete records should still load.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-complete-risk"}


def test_load_proposals_by_id_keeps_queue_payload_for_duplicate_ids(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-duplicate
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record queue version.
    reason: Queue should win.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )
    (tmp_path / "improvement_backlog.md").write_text(
        """# Improvement Backlog

proposals:
  - id: proposal-duplicate
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record backlog version.
    reason: Backlog should only fill missing ids.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert proposals["proposal-duplicate"]["proposed_change"] == "Record queue version."


def test_load_proposals_by_id_finds_proposals_after_introductory_markdown(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

Review these proposals before applying accepted changes.

proposals:
  - id: proposal-after-intro
    type: quality_report_note
    target: quality_report.md
    proposed_change: Record proposal after intro.
    reason: Markdown prose may precede YAML.
    evidence: []
    confidence: 0.4
    risk: low
    requires_approval: false
    expected_metric_change: {}
""",
        encoding="utf-8",
    )

    proposals = load_proposals_by_id(tmp_path)

    assert set(proposals) == {"proposal-after-intro"}


def test_load_proposals_by_id_returns_empty_for_malformed_yaml(
    tmp_path: Path,
) -> None:
    (tmp_path / "approval_queue.md").write_text(
        """# Approval Queue

proposals:
  - id: proposal-bad-yaml
    type: quality_report_note
    target: quality_report.md
    proposed_change: [unterminated
""",
        encoding="utf-8",
    )

    assert load_proposals_by_id(tmp_path) == {}


def test_branch_key_from_proposal_extracts_known_ontology_key() -> None:
    proposal = {
        "id": "proposal-diagnosis-branch",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": "Create branch diagnosis_testing.",
        "reason": "Missing required branch.",
    }

    assert branch_key_from_proposal(proposal) == "diagnosis_testing"


def test_branch_key_from_proposal_prefers_explicit_key_over_later_ontology_keyword() -> None:
    proposal = {
        "id": "prop-add-branch-high-risk",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": (
            "Create a new first-level branch with key 'high_risk_population' "
            "and label '高危人群'."
        ),
        "reason": (
            "The graph is missing high risk groups, a key component of clinical "
            "prevention and treatment guidance."
        ),
        "evidence": ["item_id: 6月龄以上人群; source_id: chunk-1"],
    }

    assert branch_key_from_proposal(proposal) == "high_risk_population"


@pytest.mark.parametrize(
    "branch_reference",
    ["diagnosis_testing.child", "diagnosis_testing-v2"],
)
def test_branch_key_from_proposal_rejects_identifier_suffixes(
    branch_reference: str,
) -> None:
    proposal = {
        "id": "proposal-prefixed-branch",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": f"Create branch {branch_reference}.",
        "reason": "This is not the standalone branch key.",
    }

    assert branch_key_from_proposal(proposal) == ""


def test_branch_key_from_proposal_returns_empty_string_for_unknown_branch() -> None:
    proposal = {
        "id": "proposal-unknown-branch",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": "Create branch administrative_notes.",
        "reason": "Missing required branch.",
    }

    assert branch_key_from_proposal(proposal) == ""


def test_apply_change_status_blocked_value_is_stable() -> None:
    assert ApplyChangeStatus.BLOCKED.value == "blocked"


def test_category_for_branch_key_returns_only_top_level_category() -> None:
    assert category_for_branch_key("diagnosis_testing").key == "diagnosis_testing"
    assert category_for_branch_key("diagnosis_testing.child") is None


def test_apply_change_to_dict_matches_spec_shape() -> None:
    change = ApplyChange(
        proposal_id="proposal-diagnosis-branch",
        proposal_type="add_hierarchy_branch",
        target="hierarchy",
        status=ApplyChangeStatus.BLOCKED,
        action="add_hierarchy_branch",
        branch_key="diagnosis_testing",
        branch_label="Diagnosis Testing",
        evidence=["item_id: influenza; source_id: chunk-1"],
        reason="Requires graph mutation and Task 2 is artifact-only.",
    )

    assert change.to_dict() == {
        "proposal_id": "proposal-diagnosis-branch",
        "proposal_type": "add_hierarchy_branch",
        "target": "hierarchy",
        "status": "blocked",
        "action": "add_hierarchy_branch",
        "branch_key": "diagnosis_testing",
        "branch_label": "Diagnosis Testing",
        "evidence": ["item_id: influenza; source_id: chunk-1"],
        "reason": "Requires graph mutation and Task 2 is artifact-only.",
    }


def test_apply_result_to_dict_matches_spec_shape_and_computed_counts() -> None:
    result = AcceptedApplyResult(
        workspace="influenza_medical_v1",
        applied_at="2026-06-19T10:00:00+08:00",
        source_artifact=APPLY_SOURCE,
        proposal_ids=["proposal-applied", "proposal-blocked"],
        quality_before={"metrics": {"hierarchy_missing_branch_count": 2}},
        quality_after={"metrics": {"hierarchy_missing_branch_count": 1}},
        changes=[
            ApplyChange(
                proposal_id="proposal-applied",
                proposal_type="quality_report_note",
                target="quality_report.md",
                status=ApplyChangeStatus.APPLIED,
                action="record_note",
            ),
            ApplyChange(
                proposal_id="proposal-blocked",
                proposal_type="add_hierarchy_branch",
                target="hierarchy",
                status=ApplyChangeStatus.BLOCKED,
                action="add_hierarchy_branch",
                branch_key="diagnosis_testing",
            ),
        ],
    )

    payload = result.to_dict()

    assert result.applied_count == 1
    assert result.blocked_count == 1
    assert payload["workspace"] == "influenza_medical_v1"
    assert payload["applied_at"] == "2026-06-19T10:00:00+08:00"
    assert payload["source_artifact"] == "kb_iteration_apply"
    assert payload["proposal_ids"] == ["proposal-applied", "proposal-blocked"]
    assert payload["applied_count"] == 1
    assert payload["blocked_count"] == 1
    assert payload["quality_before"] == {
        "metrics": {"hierarchy_missing_branch_count": 2}
    }
    assert payload["quality_after"] == {
        "metrics": {"hierarchy_missing_branch_count": 1}
    }
    assert payload["changes"][0]["status"] == "applied"
    assert "quality_delta" not in payload


def test_apply_result_artifacts_use_spec_filenames_and_metric_transition(
    tmp_path: Path,
) -> None:
    result = AcceptedApplyResult(
        workspace="influenza_medical_v1",
        applied_at="2026-06-19T10:00:00+08:00",
        source_artifact=APPLY_SOURCE,
        proposal_ids=["proposal-diagnosis-branch"],
        quality_before={"metrics": {"hierarchy_missing_branch_count": 2}},
        quality_after={"metrics": {"hierarchy_missing_branch_count": 1}},
        changes=[
            ApplyChange(
                proposal_id="proposal-diagnosis-branch",
                proposal_type="add_hierarchy_branch",
                target="hierarchy",
                status=ApplyChangeStatus.BLOCKED,
                action="add_hierarchy_branch",
                branch_key="diagnosis_testing",
                branch_label="Diagnosis Testing",
                reason="Graph mutation is out of scope for Task 2.",
            )
        ],
    )

    json_path, markdown_path = write_apply_result_artifacts(result, tmp_path)
    markdown = render_apply_result_markdown(result)

    assert APPLY_SOURCE == "kb_iteration_apply"
    assert APPLY_RESULT_JSON == "accepted_changes_apply_result.json"
    assert APPLY_RESULT_MARKDOWN == "accepted_changes_apply_result.md"
    assert json_path == tmp_path / APPLY_RESULT_JSON
    assert markdown_path == tmp_path / APPLY_RESULT_MARKDOWN
    assert not (tmp_path / "accepted_changes_execution.json").exists()
    assert not (tmp_path / "accepted_changes_execution.md").exists()
    assert "hierarchy_missing_branch_count: 2 -> 1" in markdown
    assert "## Changes" in markdown
    assert "proposal-diagnosis-branch: blocked (diagnosis_testing)" in markdown


@pytest.mark.asyncio
async def test_apply_accepted_changes_upserts_branch_without_evidence_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"influenza", "skipped->edge"})
    lock_calls = install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-diagnosis-branch",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": "Create branch diagnosis_testing.",
        "reason": "Missing required branch diagnosis_testing.",
        "evidence": [
            "item_id: influenza; source_id: chunk-1",
            "item_id: missing; source_id: chunk-2",
            "item_id: skipped->edge; source_id: chunk-3",
        ],
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-diagnosis-branch"}],
        proposals_by_id={"proposal-diagnosis-branch": proposal},
    )

    category = next(
        category
        for category in TOP_LEVEL_MEDICAL_CATEGORIES
        if category.key == "diagnosis_testing"
    )
    assert result.workspace == "influenza_medical_v1"
    assert result.source_artifact == "accepted_changes.md"
    assert result.proposal_ids == ["proposal-diagnosis-branch"]
    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert result.changes[0].branch_key == "diagnosis_testing"
    assert result.changes[0].branch_label == category.label
    assert result.changes[0].evidence == proposal["evidence"]
    assert result.applied_at.endswith("+00:00")
    assert graph.index_done_calls == 1
    assert graph.upserted_nodes == [
        (
            "diagnosis_testing",
            {
                "entity_id": "diagnosis_testing",
                "label": category.label,
                "entity_type": "MedicalGroup",
                "description": f"Top-level medical hierarchy branch: {category.label}",
                "source_id": APPLY_SOURCE,
                "file_path": "accepted_changes.md",
                "medical_group": "diagnosis_testing",
                "aliases": GRAPH_FIELD_SEP.join(category.aliases),
                "generated_by": APPLY_SOURCE,
                "accepted_proposal_ids": "proposal-diagnosis-branch",
            },
        )
    ]
    assert graph.upserted_edges == []
    assert all(
        not isinstance(value, (list, dict))
        for _, node_data in graph.upserted_nodes
        for value in node_data.values()
    )
    assert lock_calls == [
        {
            "keys": "diagnosis_testing",
            "namespace": "influenza_medical_v1:GraphDB",
            "enable_logging": False,
            "entered": True,
            "exited": True,
        }
    ]


@pytest.mark.asyncio
async def test_apply_accepted_changes_is_idempotent_for_existing_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"influenza"})
    install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-diagnosis-branch",
        "type": "add_hierarchy_branch",
        "target": "hierarchy",
        "proposed_change": "Create branch diagnosis_testing.",
        "reason": "Missing required branch diagnosis_testing.",
        "evidence": ["item_id: influenza; source_id: chunk-1"],
    }
    first = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-diagnosis-branch"}],
        proposals_by_id={"proposal-diagnosis-branch": proposal},
    )

    second = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-diagnosis-branch"}],
        proposals_by_id={"proposal-diagnosis-branch": proposal},
    )

    assert first.changes[0].status == ApplyChangeStatus.APPLIED
    assert second.changes[0].status == ApplyChangeStatus.ALREADY_PRESENT
    assert len(graph.upserted_nodes) == 1
    assert len(graph.upserted_edges) == 0
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
async def test_apply_candidate_kg_expansion_creates_nodes_edges_and_retires_old_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"流感病毒", "甲型流感"})
    graph.edges[("流感病毒", "甲型流感")] = {
        "id": "流感病毒->甲型流感",
        "source_node_id": "流感病毒",
        "target_node_id": "甲型流感",
        "keywords": "病原分型",
        "source_id": "chunk-020",
        "file_path": "儿童流感指南.pdf",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "prop-action-candidate-typed-flu-a",
        "type": "candidate_kg_expansion",
        "target": "kg:candidate:typed-influenza-pathogen:甲型流感",
        "proposed_change": "Create a precise typed influenza pathogen candidate.",
        "reason": "Typed influenza diseases should not use generic influenza virus as the causative agent.",
        "evidence": [
            "source_id: chunk-020; file_path: 儿童流感指南.pdf; relation_id: 流感病毒->甲型流感"
        ],
        "action_payload": {
            "candidate_nodes": [
                {
                    "id": "甲型流感病毒",
                    "label": "甲型流感病毒",
                    "entity_type": "pathogen",
                    "description": "甲型流感的精确病原体，属于流感病毒。",
                }
            ],
            "candidate_edges": [
                {
                    "source": "甲型流感病毒",
                    "target": "流感病毒",
                    "keywords": "is_a",
                    "description": "甲型流感病毒是流感病毒的一个分型病原体。",
                },
                {
                    "source": "甲型流感",
                    "target": "甲型流感病毒",
                    "keywords": "causative_agent",
                    "description": "甲型流感应指向精确的甲型流感病毒病原体。",
                },
            ],
            "retire_edges": [
                {
                    "source": "流感病毒",
                    "target": "甲型流感",
                    "keywords": "病原分型",
                    "reason": "用精确病原体节点和疾病 causative_agent 关系替代泛化分型边。",
                }
            ],
            "source_id": "chunk-020",
            "file_path": "儿童流感指南.pdf",
            "evidence_quote": "流感病毒->甲型流感",
            "why_not_existing": "甲型流感需要精确病原体节点，不能直接指向泛化流感病毒。",
        },
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "prop-action-candidate-typed-flu-a"}],
        proposals_by_id={"prop-action-candidate-typed-flu-a": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert result.changes[0].action == "candidate_kg_expansion"
    assert graph.nodes["甲型流感病毒"]["entity_type"] == "pathogen"
    assert graph.edges[("甲型流感病毒", "流感病毒")]["keywords"] == "is_a"
    assert (
        graph.edges[("甲型流感", "甲型流感病毒")]["keywords"]
        == "causative_agent"
    )
    assert graph.edges[("甲型流感", "甲型流感病毒")]["accepted_proposal_ids"] == (
        "prop-action-candidate-typed-flu-a"
    )
    assert ("流感病毒", "甲型流感") in graph.removed_edges
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
async def test_apply_accepted_changes_replaces_medical_relation_preserving_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {
        "keywords": "\u4e34\u5e8a\u8868\u73b0",
        "source_id": "chunk-1",
        "file_path": "guideline.md",
        "description": "Dry cough is a clinical manifestation of flu.",
        "accepted_proposal_ids": "proposal-existing",
    }
    lock_calls = install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal(
        qualifiers={"certainty": "guideline_supported"}
    )

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert result.changes[0].action == "replace_relation"
    assert ("dry-cough", "flu") not in graph.edges
    assert graph.removed_edges == [("dry-cough", "flu")]
    assert graph.upserted_edges == [
        (
            "flu",
            "dry-cough",
            {
                "keywords": "has_manifestation",
                "source_id": "chunk-1",
                "file_path": "guideline.md",
                "description": "Dry cough is a clinical manifestation of flu.",
                "accepted_proposal_ids": (
                    f"proposal-existing{GRAPH_FIELD_SEP}"
                    "proposal-replace-dry-cough-flu"
                ),
                "id": "flu->dry-cough",
                "source": "flu",
                "target": "dry-cough",
                "source_node_id": "flu",
                "target_node_id": "dry-cough",
                "normalized_by": APPLY_SOURCE,
                "qualifiers": '{"certainty":"guideline_supported"}',
            },
        )
    ]
    assert graph.index_done_calls == 1
    assert lock_calls == [
        {
            "keys": ["dry-cough", "flu"],
            "namespace": "influenza_medical_v1:GraphDB",
            "enable_logging": False,
            "entered": True,
            "exited": True,
        }
    ]


@pytest.mark.asyncio
async def test_apply_accepted_changes_splits_overloaded_medical_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(
        existing_nodes={"zanamivir", "mixed-target", "influenza", "children"}
    )
    graph.edges[("zanamivir", "mixed-target")] = {
        "id": "edge-zanamivir-mixed",
        "source": "zanamivir",
        "target": "mixed-target",
        "source_node_id": "zanamivir",
        "target_node_id": "mixed-target",
        "keywords": "recommended_treatment,applies_to",
        "source_id": "chunk-1",
        "file_path": "guideline.md",
        "description": "Zanamivir relation mixes indication and population.",
    }
    lock_calls = install_recording_keyed_lock(monkeypatch)
    proposal = _split_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-split-drug-relation"}],
        proposals_by_id={"proposal-split-drug-relation": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert result.changes[0].action == "split_relation"
    assert ("zanamivir", "mixed-target") not in graph.edges
    assert graph.removed_edges == [("zanamivir", "mixed-target")]
    assert graph.edges[("zanamivir", "influenza")]["keywords"] == "has_indication"
    assert graph.edges[("zanamivir", "children")]["keywords"] == "recommended_for"
    assert graph.edges[("zanamivir", "children")]["qualifiers"] == (
        '{"age_min":7,"age_unit":"year","purpose":"treatment","route":"inhalation"}'
    )
    assert graph.edges[("zanamivir", "influenza")]["accepted_proposal_ids"] == (
        "proposal-split-drug-relation"
    )
    assert graph.index_done_calls == 1
    assert lock_calls == [
        {
            "keys": ["children", "influenza", "mixed-target", "zanamivir"],
            "namespace": "influenza_medical_v1:GraphDB",
            "enable_logging": False,
            "entered": True,
            "exited": True,
        }
    ]


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_medical_relation_when_expected_edge_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "Expected edge was not found" in result.changes[0].reason
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_retires_invalid_medical_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"clinical-manifestations", "flu"})
    graph.edges[("clinical-manifestations", "flu")] = {
        "id": "clinical-manifestations->flu",
        "keywords": "clinical_manifestation",
        "source_id": "chunk-1",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _retire_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-retire-category-edge"}],
        proposals_by_id={"proposal-retire-category-edge": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert result.changes[0].action == "retire_relation"
    assert ("clinical-manifestations", "flu") not in graph.edges
    assert graph.removed_edges == [("clinical-manifestations", "flu")]
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_medical_relation_keyword_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {"keywords": "related_to"}
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "keyword" in result.changes[0].reason
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_medical_relation_noncanonical_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {"keywords": "\u4e34\u5e8a\u8868\u73b0"}
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal(new_keywords="manifestation")

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "canonical" in result.changes[0].reason
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_medical_relation_incomplete_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {"keywords": "\u4e34\u5e8a\u8868\u73b0"}
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal(new_target="")

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "Incomplete" in result.changes[0].reason
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_node",
    ["flu", "dry-cough"],
)
async def test_apply_accepted_changes_blocks_medical_relation_missing_new_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    missing_node: str,
) -> None:
    existing_nodes = {"dry-cough", "flu"} - {missing_node}
    graph = FakeGraph(existing_nodes=existing_nodes)
    graph.edges[("dry-cough", "flu")] = {
        "keywords": "\u4e34\u5e8a\u8868\u73b0",
        "source_id": "chunk-1",
        "file_path": "guideline.md",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "endpoint" in result.changes[0].reason
    assert ("dry-cough", "flu") in graph.edges
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_field", ["id", "edge_id", "relation_id"])
async def test_apply_accepted_changes_blocks_medical_relation_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {
        "keywords": "\u4e34\u5e8a\u8868\u73b0",
        identity_field: "different-edge",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "identity" in result.changes[0].reason
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_repairs_legacy_source_target_attribute_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"flu-vaccine", "seasonal-flu"})
    graph.edges[("flu-vaccine", "seasonal-flu")] = {
        "keywords": "\u9884\u9632\u63aa\u65bd",
        "source": "seasonal-flu",
        "target": "flu-vaccine",
        "source_id": "chunk-1",
        "file_path": "guideline.md",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal(
        proposal_id="proposal-replace-vaccine-seasonal-flu",
        edge_id="flu-vaccine->seasonal-flu",
        expected_source="flu-vaccine",
        expected_target="seasonal-flu",
        new_source="flu-vaccine",
        new_target="seasonal-flu",
        new_keywords="targets_disease",
        current_keywords="\u9884\u9632\u63aa\u65bd",
    )

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-vaccine-seasonal-flu"}],
        proposals_by_id={"proposal-replace-vaccine-seasonal-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert graph.edges[("flu-vaccine", "seasonal-flu")]["keywords"] == (
        "targets_disease"
    )
    assert graph.edges[("flu-vaccine", "seasonal-flu")]["id"] == (
        "flu-vaccine->seasonal-flu"
    )
    assert graph.edges[("flu-vaccine", "seasonal-flu")]["source"] == "flu-vaccine"
    assert graph.edges[("flu-vaccine", "seasonal-flu")]["target"] == "seasonal-flu"
    assert graph.edges[("flu-vaccine", "seasonal-flu")]["source_node_id"] == (
        "flu-vaccine"
    )
    assert graph.edges[("flu-vaccine", "seasonal-flu")]["target_node_id"] == (
        "seasonal-flu"
    )
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "orientation_fields",
    [
        {"source_node_id": "flu", "target_node_id": "dry-cough"},
    ],
)
async def test_apply_accepted_changes_blocks_medical_relation_orientation_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    orientation_fields: dict[str, str],
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {
        "keywords": "\u4e34\u5e8a\u8868\u73b0",
        **orientation_fields,
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "orientation" in result.changes[0].reason
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_relation_replacement_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {
        "keywords": "\u4e34\u5e8a\u8868\u73b0",
        "source_id": "chunk-1",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    first = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )
    second = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert first.changes[0].status == ApplyChangeStatus.APPLIED
    assert second.changes[0].status == ApplyChangeStatus.ALREADY_PRESENT
    assert graph.edges[("flu", "dry-cough")]["keywords"] == "has_manifestation"
    assert (
        graph.edges[("flu", "dry-cough")]["accepted_proposal_ids"]
        == "proposal-replace-dry-cough-flu"
    )
    assert graph.removed_edges == [("dry-cough", "flu")]
    assert len(graph.upserted_edges) == 1
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
async def test_apply_accepted_changes_cleans_old_relation_when_target_already_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {
        "keywords": "\u4e34\u5e8a\u8868\u73b0",
        "source_id": "chunk-1",
    }
    graph.edges[("flu", "dry-cough")] = {
        "keywords": "has_manifestation",
        "accepted_proposal_ids": "proposal-replace-dry-cough-flu",
        "source_id": "chunk-1",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert graph.removed_edges == [("dry-cough", "flu")]
    assert ("dry-cough", "flu") not in graph.edges
    assert graph.edges[("flu", "dry-cough")]["keywords"] == "has_manifestation"
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
async def test_apply_accepted_changes_normalizes_existing_target_edge_and_removes_old_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"antiviral-drug", "flu", "flu-virus"})
    graph.edges[("flu-virus", "antiviral-drug")] = {
        "id": "flu-virus->antiviral-drug",
        "keywords": "recommended_treatment",
        "source_id": "chunk-old",
    }
    graph.edges[("antiviral-drug", "flu")] = {
        "id": "antiviral-drug->flu",
        "keywords": "recommended_treatment",
        "source_id": "chunk-existing",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal(
        proposal_id="proposal-normalize-existing-treatment-target",
        edge_id="flu-virus->antiviral-drug",
        expected_source="flu-virus",
        expected_target="antiviral-drug",
        current_keywords="recommended_treatment",
        new_source="antiviral-drug",
        new_target="flu",
        new_keywords="has_indication",
    )

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-normalize-existing-treatment-target"}],
        proposals_by_id={"proposal-normalize-existing-treatment-target": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert graph.removed_edges == [("flu-virus", "antiviral-drug")]
    assert ("flu-virus", "antiviral-drug") not in graph.edges
    assert graph.edges[("antiviral-drug", "flu")]["keywords"] == "has_indication"
    assert graph.edges[("antiviral-drug", "flu")]["source_node_id"] == "antiviral-drug"
    assert graph.edges[("antiviral-drug", "flu")]["target_node_id"] == "flu"
    assert graph.upserted_edges == [
        (
            "antiviral-drug",
            "flu",
            graph.edges[("antiviral-drug", "flu")],
        )
    ]
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
async def test_apply_accepted_changes_keeps_old_relation_when_replacement_upsert_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {
        "keywords": "\u4e34\u5e8a\u8868\u73b0",
        "source_id": "chunk-1",
    }
    graph.upsert_edge_failures.add(("flu", "dry-cough"))
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    with pytest.raises(RuntimeError, match="upsert failed"):
        await apply_accepted_changes_to_graph(
            rag=FakeRAG(graph),
            workspace="influenza_medical_v1",
            records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
            proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
        )

    assert ("dry-cough", "flu") in graph.edges
    assert graph.edges[("dry-cough", "flu")]["keywords"] == "\u4e34\u5e8a\u8868\u73b0"
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_relation_replacement_is_idempotent_for_undirected_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = UndirectedFakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("flu", "dry-cough")] = {
        "keywords": "has_manifestation",
        "accepted_proposal_ids": "proposal-replace-dry-cough-flu",
        "source_id": "chunk-1",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert graph.removed_edges == []
    assert graph.upserted_edges == [
        (
            "flu",
            "dry-cough",
            {
                "keywords": "has_manifestation",
                "accepted_proposal_ids": "proposal-replace-dry-cough-flu",
                "source_id": "chunk-1",
                "id": "flu->dry-cough",
                "source": "flu",
                "target": "dry-cough",
                "source_node_id": "flu",
                "target_node_id": "dry-cough",
                "normalized_by": APPLY_SOURCE,
            },
        )
    ]
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
async def test_apply_accepted_changes_normalizes_undirected_edge_with_new_direction_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = UndirectedFakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("flu", "dry-cough")] = {
        "id": "edge-dry-cough-flu",
        "source_node_id": "flu",
        "target_node_id": "dry-cough",
        "keywords": "\u4e34\u5e8a\u8868\u73b0",
        "source_id": "chunk-1",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert graph.removed_edges == []
    assert graph.upserted_edges == [
        (
            "flu",
            "dry-cough",
            {
                "id": "flu->dry-cough",
                "source_node_id": "flu",
                "target_node_id": "dry-cough",
                "keywords": "has_manifestation",
                "source_id": "chunk-1",
                "source": "flu",
                "target": "dry-cough",
                "normalized_by": APPLY_SOURCE,
                "accepted_proposal_ids": "proposal-replace-dry-cough-flu",
            },
        )
    ]
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
async def test_apply_accepted_changes_relation_replacement_already_present_with_direction_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = UndirectedFakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("flu", "dry-cough")] = {
        "id": "flu->dry-cough",
        "source_node_id": "flu",
        "target_node_id": "dry-cough",
        "keywords": "has_manifestation",
        "accepted_proposal_ids": "proposal-replace-dry-cough-flu",
        "source_id": "chunk-1",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.ALREADY_PRESENT
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_merges_duplicate_replacement_qualifiers_after_prior_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = UndirectedFakeGraph(existing_nodes={"children", "oseltamivir"})
    graph.edges[("oseltamivir", "children")] = {
        "id": "oseltamivir->children",
        "source_node_id": "oseltamivir",
        "target_node_id": "children",
        "keywords": "recommended_for",
        "accepted_proposal_ids": "proposal-first",
        "source_id": "chunk-1",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal(
        proposal_id="proposal-second",
        edge_id="children->oseltamivir",
        expected_source="children",
        expected_target="oseltamivir",
        current_keywords="recommended_for",
        new_source="oseltamivir",
        new_target="children",
        new_keywords="recommended_for",
        qualifiers={"condition": "influenza", "purpose": "treatment"},
    )

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-second"}],
        proposals_by_id={"proposal-second": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert graph.removed_edges == []
    assert graph.edges[("oseltamivir", "children")]["accepted_proposal_ids"] == (
        f"proposal-first{GRAPH_FIELD_SEP}proposal-second"
    )
    assert graph.edges[("oseltamivir", "children")]["qualifiers"] == (
        '{"condition":"influenza","purpose":"treatment"}'
    )
    assert graph.upserted_edges == [
        (
            "oseltamivir",
            "children",
            graph.edges[("oseltamivir", "children")],
        )
    ]
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
async def test_apply_accepted_changes_merges_proposal_id_when_relation_already_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"flu", "pathogen-test"})
    graph.edges[("flu", "pathogen-test")] = {
        "id": "flu->pathogen-test",
        "source_node_id": "flu",
        "target_node_id": "pathogen-test",
        "keywords": "has_diagnostic_criterion",
        "accepted_proposal_ids": "proposal-first",
        "source_id": "chunk-1",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal(
        proposal_id="proposal-second",
        edge_id="flu->pathogen-test",
        expected_source="flu",
        expected_target="pathogen-test",
        current_keywords="诊断依据",
        new_source="flu",
        new_target="pathogen-test",
        new_keywords="has_diagnostic_criterion",
    )

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-second"}],
        proposals_by_id={"proposal-second": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert graph.edges[("flu", "pathogen-test")]["accepted_proposal_ids"] == (
        f"proposal-first{GRAPH_FIELD_SEP}proposal-second"
    )
    assert graph.removed_edges == []
    assert graph.upserted_edges == [
        (
            "flu",
            "pathogen-test",
            graph.edges[("flu", "pathogen-test")],
        )
    ]
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload_overrides",
    [
        {"current_keywords": None},
        {"current_keywords": ""},
    ],
)
async def test_apply_accepted_changes_blocks_medical_relation_without_current_keywords(
    monkeypatch: pytest.MonkeyPatch,
    payload_overrides: dict[str, object],
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {"keywords": "\u4e34\u5e8a\u8868\u73b0"}
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal(**payload_overrides)

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "Incomplete" in result.changes[0].reason
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_edge_id", [True, False])
async def test_apply_accepted_changes_blocks_medical_relation_without_edge_id(
    monkeypatch: pytest.MonkeyPatch,
    missing_edge_id: bool,
) -> None:
    graph = FakeGraph(existing_nodes={"dry-cough", "flu"})
    graph.edges[("dry-cough", "flu")] = {"keywords": "\u4e34\u5e8a\u8868\u73b0"}
    install_recording_keyed_lock(monkeypatch)
    proposal = _replace_relation_proposal(edge_id="")
    if missing_edge_id:
        proposal["action_payload"].pop("edge_id")

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-replace-dry-cough-flu"}],
        proposals_by_id={"proposal-replace-dry-cough-flu": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "Incomplete" in result.changes[0].reason
    assert graph.removed_edges == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_moves_single_value_node_to_carrier_qualifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"oseltamivir", "flu", "dose-75mg"})
    graph.edges[("oseltamivir", "dose-75mg")] = {"keywords": "has_value"}
    graph.edges[("oseltamivir", "flu")] = {
        "keywords": "has_indication",
        "source_id": "chunk-2",
    }
    lock_calls = install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-dose-qualifier",
        "type": "value_node_to_qualifier",
        "target": "node:dose-75mg",
        "proposed_change": "Move dose value node to treatment edge qualifier.",
        "reason": "Dose values should qualify treatment facts.",
        "evidence": ["node:dose-75mg"],
        "action_payload": {
            "value_node_id": "dose-75mg",
            "incident_edge_id": "oseltamivir->dose-75mg",
            "expected_incident_keywords": "has_value",
            "carrier_edge_id": "oseltamivir->flu",
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "flu",
            "expected_carrier_keywords": "has_indication",
            "carrier_source_type": "Drug",
            "carrier_target_type": "Disease",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        },
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-dose-qualifier"}],
        proposals_by_id={"proposal-dose-qualifier": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert result.changes[0].action == "value_node_to_qualifier"
    assert graph.removed_nodes == ["dose-75mg"]
    assert json.loads(graph.edges[("oseltamivir", "flu")]["qualifiers"]) == {
        "dose": "75 mg"
    }
    assert (
        graph.edges[("oseltamivir", "flu")]["accepted_proposal_ids"]
        == "proposal-dose-qualifier"
    )
    assert graph.index_done_calls == 1
    assert lock_calls == [
        {
            "keys": ["dose-75mg", "flu", "oseltamivir"],
            "namespace": "influenza_medical_v1:GraphDB",
            "enable_logging": False,
            "entered": True,
            "exited": True,
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("value_node_id", ["oseltamivir", "flu"])
async def test_apply_accepted_changes_blocks_value_node_carrier_endpoint_overlap(
    monkeypatch: pytest.MonkeyPatch,
    value_node_id: str,
) -> None:
    graph = FakeGraph(existing_nodes={"oseltamivir", "flu"})
    graph.edges[("oseltamivir", "flu")] = {"keywords": "has_indication"}
    lock_calls = install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-overlap-qualifier",
        "type": "value_node_to_qualifier",
        "target": f"node:{value_node_id}",
        "action_payload": {
            "value_node_id": value_node_id,
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "flu",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        },
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-overlap-qualifier"}],
        proposals_by_id={"proposal-overlap-qualifier": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "overlap" in result.changes[0].reason
    assert graph.removed_nodes == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0
    assert lock_calls == []


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_value_node_unrelated_sole_incident_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"oseltamivir", "flu", "dose-75mg", "adult"})
    graph.edges[("adult", "dose-75mg")] = {"keywords": "has_value"}
    graph.edges[("oseltamivir", "flu")] = {"keywords": "has_indication"}
    install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-dose-qualifier",
        "type": "value_node_to_qualifier",
        "target": "node:dose-75mg",
        "action_payload": {
            "value_node_id": "dose-75mg",
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "flu",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        },
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-dose-qualifier"}],
        proposals_by_id={"proposal-dose-qualifier": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "incident edge" in result.changes[0].reason
    assert graph.removed_nodes == []
    assert "qualifier_dose" not in graph.edges[("oseltamivir", "flu")]
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_value_node_non_value_like_incident_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"oseltamivir", "flu", "dose-75mg"})
    graph.edges[("oseltamivir", "dose-75mg")] = {"keywords": "contraindicated_for"}
    graph.edges[("oseltamivir", "flu")] = {"keywords": "has_indication"}
    install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-dose-qualifier",
        "type": "value_node_to_qualifier",
        "target": "node:dose-75mg",
        "action_payload": {
            "value_node_id": "dose-75mg",
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "flu",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        },
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-dose-qualifier"}],
        proposals_by_id={"proposal-dose-qualifier": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "value-like" in result.changes[0].reason
    assert graph.removed_nodes == []
    assert "qualifier_dose" not in graph.edges[("oseltamivir", "flu")]
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_value_node_when_incident_edges_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"oseltamivir", "flu", "dose-75mg"})
    graph.edges[("oseltamivir", "dose-75mg")] = {"keywords": "has_value"}
    graph.edges[("oseltamivir", "flu")] = {"keywords": "has_indication"}

    async def get_node_edges_returns_none(node_id: str) -> None:
        return None

    graph.get_node_edges = get_node_edges_returns_none
    install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-dose-qualifier",
        "type": "value_node_to_qualifier",
        "target": "node:dose-75mg",
        "action_payload": {
            "value_node_id": "dose-75mg",
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "flu",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        },
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-dose-qualifier"}],
        proposals_by_id={"proposal-dose-qualifier": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "incident edge" in result.changes[0].reason
    assert graph.removed_nodes == []
    assert "qualifier_dose" not in graph.edges[("oseltamivir", "flu")]
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_cleans_partial_value_node_qualifier_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"oseltamivir", "flu", "dose-75mg"})
    graph.edges[("oseltamivir", "dose-75mg")] = {"keywords": "has_value"}
    graph.edges[("oseltamivir", "flu")] = {
        "keywords": "has_indication",
        "qualifier_dose": "75 mg",
        "accepted_proposal_ids": "proposal-dose-qualifier",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-dose-qualifier",
        "type": "value_node_to_qualifier",
        "target": "node:dose-75mg",
        "action_payload": {
            "value_node_id": "dose-75mg",
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "flu",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        },
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-dose-qualifier"}],
        proposals_by_id={"proposal-dose-qualifier": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.APPLIED
    assert graph.removed_nodes == ["dose-75mg"]
    assert graph.edges[("oseltamivir", "flu")]["qualifier_dose"] == "75 mg"
    assert (
        graph.edges[("oseltamivir", "flu")]["accepted_proposal_ids"]
        == "proposal-dose-qualifier"
    )
    assert graph.index_done_calls == 1


@pytest.mark.asyncio
async def test_apply_accepted_changes_value_node_to_qualifier_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"oseltamivir", "flu"})
    graph.edges[("oseltamivir", "flu")] = {
        "keywords": "has_indication",
        "qualifier_dose": "75 mg",
        "accepted_proposal_ids": "proposal-dose-qualifier",
    }
    install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-dose-qualifier",
        "type": "value_node_to_qualifier",
        "target": "node:dose-75mg",
        "action_payload": {
            "value_node_id": "dose-75mg",
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "flu",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        },
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-dose-qualifier"}],
        proposals_by_id={"proposal-dose-qualifier": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.ALREADY_PRESENT
    assert graph.removed_nodes == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_value_node_when_degree_is_not_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"oseltamivir", "flu", "dose-75mg", "adult"})
    graph.edges[("oseltamivir", "dose-75mg")] = {"keywords": "has_value"}
    graph.edges[("adult", "dose-75mg")] = {"keywords": "has_value"}
    graph.edges[("oseltamivir", "flu")] = {"keywords": "has_indication"}
    install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-dose-qualifier",
        "type": "value_node_to_qualifier",
        "target": "node:dose-75mg",
        "action_payload": {
            "value_node_id": "dose-75mg",
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "flu",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        },
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-dose-qualifier"}],
        proposals_by_id={"proposal-dose-qualifier": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "degree" in result.changes[0].reason
    assert graph.removed_nodes == []
    assert "qualifier_dose" not in graph.edges[("oseltamivir", "flu")]
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_value_node_when_carrier_edge_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph(existing_nodes={"oseltamivir", "flu", "dose-75mg"})
    graph.edges[("oseltamivir", "dose-75mg")] = {"keywords": "has_value"}
    install_recording_keyed_lock(monkeypatch)
    proposal = {
        "id": "proposal-dose-qualifier",
        "type": "value_node_to_qualifier",
        "target": "node:dose-75mg",
        "action_payload": {
            "value_node_id": "dose-75mg",
            "carrier_edge_source": "oseltamivir",
            "carrier_edge_target": "flu",
            "qualifier_key": "dose",
            "qualifier_value": "75 mg",
        },
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-dose-qualifier"}],
        proposals_by_id={"proposal-dose-qualifier": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert "Carrier edge was not found" in result.changes[0].reason
    assert graph.removed_nodes == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_unsupported_type_writes_nothing() -> None:
    graph = FakeGraph()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-note", "proposal_type": "quality_report_note"}],
        proposals_by_id={"proposal-note": {"id": "proposal-note"}},
    )

    assert result.changes[0].status == ApplyChangeStatus.UNSUPPORTED
    assert graph.upserted_nodes == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_uses_registry_for_unsupported_action() -> None:
    graph = FakeGraph()
    proposal = {
        "id": "proposal-alias-merge",
        "type": "entity_alias_merge",
        "target": "node:flu-short",
        "proposed_change": "Merge duplicate aliases.",
        "reason": "Alias merge is not statically applyable yet.",
        "evidence": ["node:flu-short"],
        "action_payload": {"action": "entity_alias_merge"},
    }

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-alias-merge"}],
        proposals_by_id={"proposal-alias-merge": proposal},
    )

    assert result.changes[0].status == ApplyChangeStatus.UNSUPPORTED
    assert result.changes[0].action == "entity_alias_merge"
    assert result.changes[0].reason == proposal_apply_capability(proposal).reason
    assert graph.upserted_nodes == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_missing_proposal_definition_even_if_record_contains_mutation_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = FakeGraph()
    install_recording_keyed_lock(monkeypatch)

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[
            {
                "proposal_id": "proposal-missing",
                "proposal_type": "add_hierarchy_branch",
                "proposal_target": "hierarchy",
                "proposed_change": "Create branch diagnosis_testing.",
            }
        ],
        proposals_by_id={},
    )

    assert result.proposal_ids == ["proposal-missing"]
    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert result.changes[0].proposal_id == "proposal-missing"
    assert result.changes[0].proposal_type == "add_hierarchy_branch"
    assert result.changes[0].target == "hierarchy"
    assert result.changes[0].action == "add_hierarchy_branch"
    assert result.changes[0].reason == "Accepted proposal definition was not found."
    assert graph.upserted_nodes == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_blocks_historical_missing_proposal_definition() -> None:
    graph = FakeGraph()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[
            {
                "proposal_id": "proposal-historical",
                "proposal_type": "medical_relation_schema_migration",
                "proposal_target": "edge:historical",
                "recorded_at": "2026-06-20T00:00:00+00:00",
            }
        ],
        proposals_by_id={},
    )

    assert result.proposal_ids == ["proposal-historical"]
    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert result.changes[0].proposal_type == "medical_relation_schema_migration"
    assert result.changes[0].target == "edge:historical"
    assert result.changes[0].action == "medical_relation_schema_migration"
    assert result.changes[0].reason == "Accepted proposal definition was not found."
    assert graph.upserted_nodes == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_unknown_branch_key_is_blocked() -> None:
    graph = FakeGraph()

    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(graph),
        workspace="influenza_medical_v1",
        records=[{"proposal_id": "proposal-unknown"}],
        proposals_by_id={
            "proposal-unknown": {
                "id": "proposal-unknown",
                "type": "add_hierarchy_branch",
                "target": "hierarchy",
                "proposed_change": "Create branch administrative_notes.",
                "reason": "Missing branch administrative_notes.",
                "evidence": [],
            }
        },
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert result.changes[0].reason == "Missing or unsupported branch key."
    assert graph.upserted_nodes == []
    assert graph.upserted_edges == []
    assert graph.index_done_calls == 0


@pytest.mark.asyncio
async def test_apply_accepted_changes_without_graph_storage_is_blocked() -> None:
    result = await apply_accepted_changes_to_graph(
        rag=FakeRAG(),
        workspace="influenza_medical_v1",
        records=[
            {
                "proposal_id": "proposal-diagnosis-branch",
                "proposal_type": "add_hierarchy_branch",
                "proposal_target": "hierarchy",
            }
        ],
        proposals_by_id={
            "proposal-diagnosis-branch": {
                "id": "proposal-diagnosis-branch",
                "proposed_change": "Create branch diagnosis_testing.",
                "reason": "Missing required branch diagnosis_testing.",
                "evidence": [],
            }
        },
    )

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert result.changes[0].reason == "Graph storage is not available."


def test_quality_missing_branch_count_decreases_for_medical_group_property() -> None:
    sparse = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-19T00:00:00+00:00",
        source_files=[],
        nodes=[],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )
    with_branch = KGSnapshot(
        workspace="influenza_medical_v1",
        generated_at="2026-06-19T00:00:00+00:00",
        source_files=[],
        nodes=[
            SnapshotNode(
                "branch-diagnosis",
                "Diagnosis branch",
                "MedicalGroup",
                properties={"medical_group": "diagnosis_testing"},
            )
        ],
        edges=[],
        metadata={"profile": "clinical_guideline_zh"},
    )

    sparse_score = evaluate_snapshot_quality(sparse)
    with_branch_score = evaluate_snapshot_quality(with_branch)

    assert with_branch_score.metrics["hierarchy_missing_branch_count"] == (
        sparse_score.metrics["hierarchy_missing_branch_count"] - 1
    )
