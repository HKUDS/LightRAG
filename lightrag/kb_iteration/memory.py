from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


AGENT_MEMORY_SUMMARY_FILE = "agent_memory_summary.md"
MAX_AGENT_MEMORY_SUMMARY_CHARS = 5000
MAX_MEMORY_ITEMS_PER_SECTION = 6
MAX_MEMORY_ITEM_CHARS = 420
SCHEMA_MEMORY_TYPES = {
    "medical_relation_schema_migration",
    "value_node_to_qualifier",
    "entity_alias_merge",
    "medical_fact_role_split",
}
SCHEMA_MEMORY_KEYWORDS = ("has_manifestation", "is_a", "qualifier")
MAX_SCHEMA_MEMORY_ITEMS = 10

_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_BULLET_PATTERN = re.compile(r"^\s*[-*]\s+(.+?)\s*$", re.MULTILINE)
_SCHEMA_MEMORY_KEYWORD_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_])(?:{'|'.join(map(re.escape, SCHEMA_MEMORY_KEYWORDS))})"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_PLACEHOLDER_SNIPPETS = (
    "record approved kb changes here",
    "record rejected kb changes here",
    "queue maintainer feedback for proposals that need revision here",
)


@dataclass(frozen=True)
class _MemoryRecord:
    proposal_id: str
    proposal_type: str
    target: str
    reason: str
    instruction: str
    source_file: str
    recorded_at: float | None
    read_order: int


def refresh_agent_memory_summary(package_dir: str | Path) -> Path:
    """Refresh the compact memory file consumed by the iteration agent."""

    package_path = Path(package_dir)
    package_path.mkdir(parents=True, exist_ok=True)
    summary_path = package_path / AGENT_MEMORY_SUMMARY_FILE
    summary_path.write_text(
        build_agent_memory_summary(package_path),
        encoding="utf-8",
    )
    return summary_path


def build_agent_memory_summary(package_dir: str | Path) -> str:
    package_path = Path(package_dir)
    accepted = _records_from_file(package_path / "accepted_changes.md", read_offset=0)
    rejected = _records_from_file(
        package_path / "rejected_changes.md",
        read_offset=len(accepted),
    )
    revisions = _records_from_file(
        package_path / "proposal_revision_requests.md",
        read_offset=len(accepted) + len(rejected),
    )
    schema_records = _schema_records(accepted, rejected, revisions)

    lines = [
        "# Agent Memory Summary",
        "",
        (
            "- 压缩长期记忆；原始 md 保留审计；"
            f"每区最多 {MAX_MEMORY_ITEMS_PER_SECTION} 条，最近优先。"
        ),
        "",
        "## 医学 schema 规则记忆",
        *_render_schema_section(schema_records),
        "",
        "## 已接受规则",
        *_render_section(accepted, preferred_field="reason"),
        "",
        "## 已拒绝模式",
        *_render_section(rejected, preferred_field="reason"),
        "",
        "## 返修要求",
        *_render_section(revisions, preferred_field="instruction"),
        "",
    ]
    return _fit_summary("\n".join(lines))


def _records_from_file(path: Path, *, read_offset: int) -> list[_MemoryRecord]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    records = [
        _record_from_payload(
            payload,
            source_file=path.name,
            read_order=read_offset + index,
        )
        for index, payload in enumerate(_json_payloads(text))
    ]
    records.extend(
        _manual_records(
            text,
            source_file=path.name,
            read_offset=read_offset + len(records),
        )
    )
    return [record for record in records if _record_text(record)]


def _schema_records(*record_groups: list[_MemoryRecord]) -> list[_MemoryRecord]:
    return [
        record
        for records in record_groups
        for record in records
        if _is_schema_memory_record(record)
    ]


def _is_schema_memory_record(record: _MemoryRecord) -> bool:
    if record.proposal_type in SCHEMA_MEMORY_TYPES:
        return True
    return _SCHEMA_MEMORY_KEYWORD_PATTERN.search(_record_lesson_text(record)) is not None


def _json_payloads(text: str) -> list[dict[str, Any]]:
    payloads = []
    for block in _JSON_BLOCK_PATTERN.findall(text):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _record_from_payload(
    payload: dict[str, Any], *, source_file: str, read_order: int
) -> _MemoryRecord:
    return _MemoryRecord(
        proposal_id=_string_field(payload, "proposal_id", "id"),
        proposal_type=_string_field(payload, "proposal_type", "type"),
        target=_string_field(payload, "proposal_target", "target"),
        reason=_string_field(payload, "reason", "proposal_reason", "proposed_change"),
        instruction=_string_field(payload, "instruction", "reason", "proposal_reason"),
        source_file=source_file,
        recorded_at=_timestamp_field(
            payload,
            "recorded_at",
            "requested_at",
            "applied_at",
            "decided_at",
            "created_at",
            "updated_at",
        ),
        read_order=read_order,
    )


def _manual_records(
    text: str, *, source_file: str, read_offset: int
) -> list[_MemoryRecord]:
    records = []
    for index, bullet in enumerate(_BULLET_PATTERN.findall(text)):
        cleaned = _clean_text(bullet)
        if not cleaned or _is_placeholder(cleaned):
            continue
        records.append(
            _MemoryRecord(
                proposal_id="manual",
                proposal_type="",
                target="",
                reason=cleaned,
                instruction=cleaned,
                source_file=source_file,
                recorded_at=None,
                read_order=read_offset + index,
            )
        )
    return records


def _render_section(
    records: list[_MemoryRecord], *, preferred_field: str
) -> list[str]:
    selected = _latest_unique(records)
    if not selected:
        return ["- 暂无。"]
    return [_render_record(record, preferred_field=preferred_field) for record in selected]


def _render_schema_section(records: list[_MemoryRecord]) -> list[str]:
    selected = _latest_unique(
        records,
        limit=MAX_SCHEMA_MEMORY_ITEMS,
        key_func=_schema_record_key,
    )
    if not selected:
        return ["- 暂无。"]
    return [_render_record(record, preferred_field="instruction") for record in selected]


def _latest_unique(
    records: list[_MemoryRecord],
    *,
    limit: int = MAX_MEMORY_ITEMS_PER_SECTION,
    key_func: Callable[[_MemoryRecord], object] | None = None,
) -> list[_MemoryRecord]:
    selected = []
    seen = set()
    key_func = key_func or _record_key
    for record in sorted(records, key=_record_recency_key, reverse=True):
        key = key_func(record)
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
        if len(selected) == limit:
            break
    return selected


def _record_recency_key(record: _MemoryRecord) -> tuple[int, float, int]:
    if record.recorded_at is None:
        return (0, float(record.read_order), record.read_order)
    return (1, record.recorded_at, record.read_order)


def _schema_record_key(record: _MemoryRecord) -> tuple[str, str, str]:
    body = record.instruction or record.reason
    return (
        record.proposal_type,
        record.target,
        body.casefold(),
    )


def _record_key(record: _MemoryRecord) -> tuple[str, str, str, str]:
    return (
        record.proposal_id,
        record.proposal_type,
        record.target,
        _record_text(record).casefold(),
    )


def _render_record(record: _MemoryRecord, *, preferred_field: str) -> str:
    body = record.instruction if preferred_field == "instruction" else record.reason
    body = body or record.reason or record.instruction
    details = []
    if record.proposal_type:
        details.append(record.proposal_type)
    if record.target:
        details.append(f"target={record.target}")
    if body:
        details.append(body)

    label = record.proposal_id or record.source_file
    if label == "manual":
        label = record.source_file
    rendered = f"- {label}: " + " | ".join(details)
    return _clip(rendered, MAX_MEMORY_ITEM_CHARS)


def _fit_summary(summary: str) -> str:
    summary = summary.rstrip() + "\n"
    if len(summary) <= MAX_AGENT_MEMORY_SUMMARY_CHARS:
        return summary

    suffix = "\n## 截断说明\n\n- 摘要超过上限，已保留最近的高优先级记忆。\n"
    limit = MAX_AGENT_MEMORY_SUMMARY_CHARS - len(suffix)
    return summary[:limit].rstrip() + suffix


def _string_field(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return _clean_text(value)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(value)
    return ""


def _record_text(record: _MemoryRecord) -> str:
    return " ".join(
        part
        for part in (
            record.proposal_id,
            record.proposal_type,
            record.target,
            record.reason,
            record.instruction,
        )
        if part
    )


def _record_lesson_text(record: _MemoryRecord) -> str:
    return _clean_text(f"{record.reason} {record.instruction}")


def _clean_text(value: str) -> str:
    return " ".join(str(value).split())


def _clip(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def _timestamp_field(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        parsed = _parse_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def _parse_timestamp(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def _is_placeholder(value: str) -> bool:
    normalized = value.casefold()
    return any(snippet in normalized for snippet in _PLACEHOLDER_SNIPPETS)
