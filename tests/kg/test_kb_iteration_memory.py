import json
from pathlib import Path

from lightrag.kb_iteration.memory import (
    AGENT_MEMORY_SUMMARY_FILE,
    MAX_AGENT_MEMORY_SUMMARY_CHARS,
    build_agent_memory_summary,
    refresh_agent_memory_summary,
)


def test_refresh_agent_memory_summary_keeps_agent_context_small(tmp_path: Path):
    _write_decision_records(
        tmp_path / "accepted_changes.md",
        "Accepted Changes",
        "accept",
        count=18,
    )
    _write_decision_records(
        tmp_path / "rejected_changes.md",
        "Rejected Changes",
        "reject",
        count=18,
    )
    _write_revision_records(tmp_path / "proposal_revision_requests.md", count=18)

    path = refresh_agent_memory_summary(tmp_path)

    summary = path.read_text(encoding="utf-8")
    assert path == tmp_path / AGENT_MEMORY_SUMMARY_FILE
    assert len(summary) <= MAX_AGENT_MEMORY_SUMMARY_CHARS
    assert "## 已接受规则" in summary
    assert "## 已拒绝模式" in summary
    assert "## 返修要求" in summary
    assert "accept-17" in summary
    assert "reject-17" in summary
    assert "revision-17" in summary
    assert "accept-0" not in summary
    assert "reject-0" not in summary
    assert "revision-0" not in summary


def test_refresh_agent_memory_summary_ignores_placeholder_templates(tmp_path: Path):
    (tmp_path / "accepted_changes.md").write_text(
        "# Accepted Changes\n\n- Record approved KB changes here.\n",
        encoding="utf-8",
    )
    (tmp_path / "rejected_changes.md").write_text(
        "# Rejected Changes\n\n- Record rejected KB changes here.\n",
        encoding="utf-8",
    )
    (tmp_path / "proposal_revision_requests.md").write_text(
        "# Proposal Revision Requests\n\n"
        "- Queue maintainer feedback for proposals that need revision here.\n",
        encoding="utf-8",
    )

    path = refresh_agent_memory_summary(tmp_path)

    summary = path.read_text(encoding="utf-8")
    assert "Record approved KB changes here" not in summary
    assert "Record rejected KB changes here" not in summary
    assert "Queue maintainer feedback" not in summary
    assert "- 暂无。" in summary


def test_build_agent_memory_summary_distills_medical_schema_rules(tmp_path: Path):
    _write_schema_record(
        tmp_path / "accepted_changes.md",
        {
            "proposal_id": "schema-accepted-1",
            "proposal_type": "medical_relation_schema_migration",
            "proposal_target": "relation_schema",
            "reason": "Use has_manifestation for disease to symptom relations.",
        },
    )
    _write_schema_record(
        tmp_path / "rejected_changes.md",
        {
            "proposal_id": "schema-rejected-1",
            "proposal_type": "value_node_to_qualifier",
            "proposal_target": "dose-node-42",
            "reason": "剂量节点连接多条边时不能自动转换为 qualifier。",
        },
    )

    summary = build_agent_memory_summary(tmp_path)

    assert len(summary) <= MAX_AGENT_MEMORY_SUMMARY_CHARS
    assert "## 医学 schema 规则记忆" in summary
    assert "has_manifestation" in summary
    assert "不能自动转换" in summary

    path = refresh_agent_memory_summary(tmp_path)
    assert path == tmp_path / AGENT_MEMORY_SUMMARY_FILE
    assert path.read_text(encoding="utf-8") == summary


def test_build_agent_memory_summary_includes_generic_records_with_schema_keywords(
    tmp_path: Path,
):
    _write_schema_record(
        tmp_path / "accepted_changes.md",
        {
            "proposal_id": "generic-accepted-1",
            "proposal_type": "relation_rule_change",
            "proposal_target": "edge-manifestation",
            "reason": "Legacy migration should use has_manifestation for symptoms.",
        },
    )
    _write_schema_record(
        tmp_path / "proposal_revision_requests.md",
        {
            "proposal_id": "generic-revision-1",
            "proposal_type": "kg_fact_correction",
            "proposal_target": "dose-node-9",
            "instruction": "Revise dose modeling; qualifier is needed, not a value node.",
        },
    )

    summary = build_agent_memory_summary(tmp_path)
    schema_section = _section(
        summary,
        "## 医学 schema 规则记忆",
        "## 已接受规则",
    )

    assert "generic-accepted-1" in schema_section
    assert "has_manifestation" in schema_section
    assert "generic-revision-1" in schema_section
    assert "qualifier" in schema_section
    assert len(summary) <= MAX_AGENT_MEMORY_SUMMARY_CHARS


def test_schema_keyword_fallback_only_matches_lesson_text(tmp_path: Path):
    _write_schema_record(
        tmp_path / "accepted_changes.md",
        {
            "proposal_id": "diagnosis_analysis_false_positive",
            "proposal_type": "relation_rule_change",
            "proposal_target": "analysis_artifact",
            "reason": "Only audit metadata changed; no schema lesson here.",
        },
    )
    _write_schema_record(
        tmp_path / "proposal_revision_requests.md",
        {
            "proposal_id": "generic-revision-is-a",
            "proposal_type": "kg_fact_correction",
            "proposal_target": "hierarchy-edge",
            "instruction": "Use is_a only for class hierarchy links.",
        },
    )

    summary = build_agent_memory_summary(tmp_path)
    schema_section = _section(
        summary,
        "## 医学 schema 规则记忆",
        "## 已接受规则",
    )

    assert "diagnosis_analysis_false_positive" not in schema_section
    assert "analysis_artifact" not in schema_section
    assert "generic-revision-is-a" in schema_section
    assert "Use is_a only" in schema_section


def test_schema_memory_dedupes_by_latest_timestamp_across_sources(tmp_path: Path):
    duplicate_lesson = "Use has_manifestation for disease symptom edges."
    _write_schema_record(
        tmp_path / "accepted_changes.md",
        {
            "proposal_id": "newer-accepted-schema",
            "proposal_type": "medical_relation_schema_migration",
            "proposal_target": "relation_schema",
            "reason": duplicate_lesson,
            "recorded_at": "2026-06-20T10:00:00+00:00",
        },
    )
    _write_schema_record(
        tmp_path / "proposal_revision_requests.md",
        {
            "proposal_id": "older-revision-schema",
            "proposal_type": "medical_relation_schema_migration",
            "proposal_target": "relation_schema",
            "instruction": duplicate_lesson,
            "requested_at": "2026-06-19T10:00:00+00:00",
        },
    )

    summary = build_agent_memory_summary(tmp_path)
    schema_section = _section(
        summary,
        "## 医学 schema 规则记忆",
        "## 已接受规则",
    )

    assert "newer-accepted-schema" in schema_section
    assert "older-revision-schema" not in schema_section


def _write_decision_records(
    path: Path, title: str, decision: str, *, count: int
) -> None:
    records = [f"# {title}\n"]
    for index in range(count):
        records.append(
            f"\n## {decision}-{index}\n\n"
            "```json\n"
            "{\n"
            f'  "proposal_id": "{decision}-{index}",\n'
            f'  "proposal_type": "relation_rule_change",\n'
            f'  "proposal_target": "edge-{index}",\n'
            f'  "decision": "{decision}",\n'
            f'  "reason": "Use compact memory item number {index} with evidence and guardrails."\n'
            "}\n"
            "```\n"
        )
    path.write_text("".join(records), encoding="utf-8")


def _section(summary: str, heading: str, next_heading: str) -> str:
    start = summary.index(heading)
    end = summary.index(next_heading, start)
    return summary[start:end]


def _write_schema_record(path: Path, payload: dict[str, str]) -> None:
    path.write_text(
        "# Schema Records\n\n"
        f"## {payload['proposal_id']}\n\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "```\n",
        encoding="utf-8",
    )


def _write_revision_records(path: Path, *, count: int) -> None:
    records = ["# Proposal Revision Requests\n"]
    for index in range(count):
        records.append(
            f"\n## revision-{index}\n\n"
            "```json\n"
            "{\n"
            f'  "proposal_id": "revision-{index}",\n'
            f'  "proposal_type": "kg_fact_correction",\n'
            f'  "proposal_target": "edge-{index}",\n'
            f'  "reason": "Rejected direction needs tighter evidence {index}",\n'
            f'  "instruction": "Revise with source_id, file_path, and relation_id {index}."\n'
            "}\n"
            "```\n"
        )
    path.write_text("".join(records), encoding="utf-8")
