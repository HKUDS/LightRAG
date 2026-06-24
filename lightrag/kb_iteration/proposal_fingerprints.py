from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .medical_schema import MEDICAL_RELATION_SCHEMA_VERSION

_EVIDENCE_STRING_FIELDS = (
    "source_id",
    "file_path",
    "evidence_quote",
    "relation_id",
)


@dataclass(frozen=True)
class ProposalFingerprints:
    semantic: str
    execution: str
    evidence: str


def candidate_fingerprints(
    candidate: Mapping[str, Any],
    *,
    schema_version: str | None = None,
) -> ProposalFingerprints:
    return _fingerprints(candidate, schema_version=schema_version)


def proposal_fingerprints(
    proposal: Mapping[str, Any],
    *,
    schema_version: str | None = None,
) -> ProposalFingerprints:
    normalized = {
        "proposal_type": _string_value(
            proposal.get("proposal_type") or proposal.get("type")
        ),
        "target": proposal.get("proposal_target") or proposal.get("target"),
        "schema_version": proposal.get("schema_version"),
        "evidence": proposal.get("evidence"),
        "evidence_spans": proposal.get("evidence_spans"),
        "action_payload": proposal.get("action_payload"),
        "proposed_change": proposal.get("proposed_change"),
    }
    return _fingerprints(normalized, schema_version=schema_version)


def decision_fingerprints(
    record: Mapping[str, Any],
) -> ProposalFingerprints | None:
    semantic = _string_value(record.get("semantic_fingerprint"))
    execution = _string_value(record.get("execution_fingerprint"))
    evidence = _string_value(record.get("evidence_fingerprint"))
    if not semantic or not execution or not evidence:
        return None
    return ProposalFingerprints(
        semantic=semantic,
        execution=execution,
        evidence=evidence,
    )


def decision_memory_suppresses_candidate(
    candidate: Mapping[str, Any],
    decisions: list[Mapping[str, Any]],
    *,
    schema_version: str | None = None,
) -> bool:
    candidate_schema_version = _schema_version(candidate, schema_version)
    candidate_fp = candidate_fingerprints(
        candidate,
        schema_version=candidate_schema_version,
    )
    for record in decisions:
        fingerprints = decision_fingerprints(record)
        if fingerprints is None:
            continue

        decision = _string_value(record.get("decision")).casefold()
        scope = _string_value(record.get("rejection_scope")).casefold()
        apply_status = _string_value(record.get("apply_status")).casefold()
        if decision in {"defer", "deferred"} or scope in {"defer_only", "defer"}:
            continue

        if decision in {"accept", "accepted", "apply", "applied"} or apply_status in {
            "applied",
            "succeeded",
            "success",
        }:
            if candidate_fp.execution == fingerprints.execution:
                return True
            continue

        if decision not in {"reject", "rejected"}:
            continue

        if scope in {"", "exact_action"}:
            if candidate_fp.execution == fingerprints.execution:
                return True
            continue

        if scope == "semantic_until_schema_change":
            record_schema = _string_value(record.get("schema_version"))
            if (
                record_schema == candidate_schema_version
                and candidate_fp.semantic == fingerprints.semantic
            ):
                return True
            continue

        if scope == "semantic_until_evidence_change":
            if (
                candidate_fp.semantic == fingerprints.semantic
                and candidate_fp.evidence == fingerprints.evidence
            ):
                return True

    return False


def _fingerprints(
    item: Mapping[str, Any],
    *,
    schema_version: str | None,
) -> ProposalFingerprints:
    semantic = _semantic_body(item, schema_version=schema_version)
    execution = _execution_body(item, semantic=semantic)
    evidence = _evidence_body(item)
    return ProposalFingerprints(
        semantic=_hash_payload("semantic", semantic),
        execution=_hash_payload("execution", execution),
        evidence=_hash_payload("evidence", evidence),
    )


def _semantic_body(
    item: Mapping[str, Any],
    *,
    schema_version: str | None,
) -> dict[str, Any]:
    action_payload = _mapping_value(item.get("action_payload"))
    proposal_type = _string_value(item.get("proposal_type") or item.get("type"))
    action = _string_value(action_payload.get("action"))
    version = _schema_version(item, schema_version)

    if action == "replace_relation":
        return {
            "proposal_type": proposal_type,
            "action": "replace_relation",
            "new_source": action_payload.get("new_source"),
            "new_target": action_payload.get("new_target"),
            "new_keywords": action_payload.get("new_keywords"),
            "qualifiers": _mapping_value(action_payload.get("qualifiers")),
            "schema_version": version,
        }

    return {
        "proposal_type": proposal_type,
        "target": item.get("target") or item.get("proposal_target"),
        "action_payload": action_payload,
        "proposed_change": item.get("proposed_change"),
        "schema_version": version,
    }


def _execution_body(
    item: Mapping[str, Any],
    *,
    semantic: dict[str, Any],
) -> dict[str, Any]:
    action_payload = _mapping_value(item.get("action_payload"))
    if action_payload.get("action") == "replace_relation":
        return {
            "semantic": semantic,
            "preconditions": {
                key: action_payload.get(key)
                for key in _replace_relation_execution_fields(action_payload)
            },
        }
    return {
        "semantic": semantic,
        "target": item.get("target") or item.get("proposal_target"),
        "action_payload": action_payload,
    }


def _replace_relation_execution_fields(
    action_payload: Mapping[str, Any],
) -> list[str]:
    preferred = [
        "edge_id",
        "expected_source",
        "expected_target",
        "current_keywords",
    ]
    extras = sorted(
        key
        for key in action_payload
        if (
            key.startswith("expected_")
            or key.startswith("current_")
            or key in {"edge_id"}
        )
        and key not in preferred
    )
    return [key for key in preferred if key in action_payload] + extras


def _evidence_body(item: Mapping[str, Any]) -> dict[str, Any]:
    evidence_spans = item.get("evidence_spans")
    if isinstance(evidence_spans, list):
        return {
            "evidence_spans": [
                _stable_value(span) for span in evidence_spans if isinstance(span, Mapping)
            ]
        }

    evidence = item.get("evidence")
    if isinstance(evidence, list):
        return {"evidence": [_evidence_string_value(item) for item in evidence]}

    action_payload = _mapping_value(item.get("action_payload"))
    evidence_tuple = {
        key: action_payload.get(key)
        for key in ("source_id", "file_path", "evidence_quote")
        if action_payload.get(key) not in {None, ""}
    }
    return {"evidence_spans": [evidence_tuple] if evidence_tuple else []}


def _schema_version(
    item: Mapping[str, Any],
    schema_version: str | None,
) -> str:
    return (
        _string_value(schema_version)
        or _string_value(item.get("schema_version"))
        or MEDICAL_RELATION_SCHEMA_VERSION
    )


def _mapping_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _evidence_string_value(value: Any) -> Any:
    if not isinstance(value, str):
        return _stable_value(value)
    parsed = _parse_evidence_string(value)
    if parsed:
        return parsed
    return _normalized_raw_string(value)


def _parse_evidence_string(value: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        key, separator, raw_field_value = part.partition(":")
        if not separator:
            return {}
        normalized_key = key.strip()
        if normalized_key not in _EVIDENCE_STRING_FIELDS:
            return {}
        field_value = raw_field_value.strip()
        if field_value:
            fields[normalized_key] = field_value
    return {key: fields[key] for key in _EVIDENCE_STRING_FIELDS if key in fields}


def _normalized_raw_string(value: str) -> str:
    return " ".join(value.split())


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    return value


def _hash_payload(kind: str, payload: Any) -> str:
    stable = json.dumps(
        _stable_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(f"{kind}:{stable}".encode("utf-8")).hexdigest()


def _string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
