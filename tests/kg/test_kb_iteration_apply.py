from __future__ import annotations

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
    apply_accepted_changes_to_graph,
    branch_key_from_proposal,
    category_for_branch_key,
    load_proposals_by_id,
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
        self.index_done_calls = 0

    async def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return (source_node_id, target_node_id) in self.edges

    async def upsert_nodes_batch(self, nodes: list[tuple[str, dict]]) -> None:
        self.upserted_nodes.extend(nodes)
        for node_id, node_data in nodes:
            self.nodes[node_id] = node_data

    async def upsert_edges_batch(self, edges: list[tuple[str, str, dict]]) -> None:
        self.upserted_edges.extend(edges)
        for source_node_id, target_node_id, edge_data in edges:
            self.edges[(source_node_id, target_node_id)] = edge_data

    async def index_done_callback(self) -> None:
        self.index_done_calls += 1


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

    assert result.changes[0].status == ApplyChangeStatus.BLOCKED
    assert result.changes[0].proposal_type == ""
    assert result.changes[0].target == ""
    assert result.changes[0].action == "resolve_proposal_definition"
    assert (
        result.changes[0].reason
        == "Accepted proposal definition was not found."
    )
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
