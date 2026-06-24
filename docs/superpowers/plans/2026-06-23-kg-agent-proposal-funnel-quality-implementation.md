# KG Agent Proposal Funnel Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent rejected medical KG proposal patterns from reappearing under new proposal IDs, improve raw-issue-to-subagent throughput, and block unsafe medical relation proposals before they reach human approval.

**Architecture:** Add a canonical proposal/action fingerprint layer shared by decision records, deterministic scanning, and proposal merge filtering. Tighten medical validators and deterministic generators so unsafe `risk_factor_for`, vaccine `has_indication`, restrictive-treatment, and malformed evidence proposals are rejected before queueing. Extend LLM residual batching by clinical family so more raw issues reach subagents without mixing unrelated medical semantics.

**Tech Stack:** Python dataclasses/functions in `lightrag/kb_iteration`, FastAPI route helpers in `lightrag/api/routers/kb_iteration_routes.py`, pytest regression tests under `tests/kg` and `tests/api/routes`, React/WebUI TypeScript updates only for optional visibility controls.

---

## Current Failure Evidence

Latest two-round run on `work/kb-iteration/influenza_medical_v1` showed:

- Initial queue: `81` proposals -> `28` accepted / `53` rejected -> `28` applied / `0` blocked.
- Round 1: `55` proposals -> `8` accepted / `47` rejected -> `8` applied / `0` blocked.
- Round 2: `54` proposals -> `0` accepted / `54` rejected; queue cleared.
- Round 2 still generated suffix variants of rejected proposals, for example `prop-action-candidate-...-suffix`.
- Round 2 still had invalid subagent outputs:
  - `schema_repair-001`: evidence object used unknown structured fields.
  - `treatment-001`: `candidate_kg_expansion` edges omitted endpoint types.
- Latest funnel:
  - `727` raw issues.
  - `673` LLM residual issues.
  - only `28` residual issues actually packed into `8` task packs.
  - `54` proposals generated, all rejected.

## File Structure

Create:

- `lightrag/kb_iteration/proposal_fingerprints.py`
  - Owns canonical proposal/action fingerprints and semantic rejection classes.
  - Must not import API route modules.

Modify:

- `lightrag/api/routers/kb_iteration_routes.py`
  - Decision records must store fingerprint fields and enough proposal payload to support future rejection-memory checks.
- `lightrag/kb_iteration/issue_ledger.py`
  - Use the shared fingerprint module to route candidates/proposals matching rejected memory to `stale` or `blocked_safety`.
- `lightrag/kb_iteration/proposal_orchestrator.py`
  - Use shared fingerprints for duplicate proposal drops and same-target conflict handling.
- `lightrag/kb_iteration/proposals.py`
  - Add pre-queue medical safety validators for patterns proven unsafe by the two-round review.
- `lightrag/kb_iteration/deterministic_proposals/risk_safety.py`
  - Stop generating `risk_factor_for` for disease-to-complication/outcome/severity edges.
- `lightrag/kb_iteration/deterministic_proposals/treatment.py`
  - Stop generating broad `has_indication` when evidence says avoid, not recommend, or when source is antibacterial/antibiotic.
- `lightrag/kb_iteration/deterministic_proposals/prevention.py`
  - Convert vaccine-to-chronic-disease cases toward `recommended_for` only when population/condition/purpose qualifiers are explicit.
- `lightrag/kb_iteration/agent_outputs.py`
  - Reject malformed evidence objects with precise retry errors and normalize supported evidence forms consistently.
- `lightrag/kb_iteration/subagent_contracts.py`
  - Require endpoint types for `candidate_kg_expansion` candidate edges.
- `lightrag/kb_iteration/prompts/subagents/base_zh.md`
- `lightrag/kb_iteration/prompts/subagents/schema_repair_zh.md`
- `lightrag/kb_iteration/prompts/subagents/treatment_zh.md`
  - Make evidence format and endpoint-type requirements explicit in Chinese prompts.
- `lightrag/kb_iteration/agent_pipeline.py`
  - Add family-aware residual batching controls.
- `lightrag/api/routers/kb_iteration_routes.py`
  - Add API request fields for residual family caps if the pipeline config needs external control.

Tests:

- `tests/kg/test_kb_iteration_proposal_fingerprints.py`
- `tests/kg/test_kb_iteration_issue_ledger.py`
- `tests/kg/test_kb_iteration_proposal_orchestrator.py`
- `tests/kg/test_kb_iteration_proposals.py`
- `tests/kg/test_kb_iteration_agent_outputs.py`
- `tests/kg/test_kb_iteration_agent_pipeline.py`
- `tests/api/routes/test_kb_iteration_routes.py`

---

## Task 1: Add Canonical Proposal Fingerprints

**Files:**

- Create: `lightrag/kb_iteration/proposal_fingerprints.py`
- Test: `tests/kg/test_kb_iteration_proposal_fingerprints.py`

- [ ] **Step 1: Write failing tests for suffix-insensitive fingerprints**

Add `tests/kg/test_kb_iteration_proposal_fingerprints.py`:

```python
from lightrag.kb_iteration.proposal_fingerprints import (
    proposal_action_fingerprint,
    rejection_class_from_review_text,
)


def test_replace_relation_fingerprint_ignores_proposal_id_suffix():
    first = {
        "id": "prop-action-candidate-02152fcc7a94",
        "type": "medical_relation_schema_migration",
        "target": "edge:流行性感冒->流感重症",
        "action_payload": {
            "action": "replace_relation",
            "edge_id": "流行性感冒->流感重症",
            "expected_source": "流行性感冒",
            "expected_target": "流感重症",
            "current_keywords": "并发风险",
            "new_source": "流行性感冒",
            "new_target": "流感重症",
            "new_keywords": "risk_factor_for",
            "qualifiers": {},
        },
    }
    second = {
        **first,
        "id": "prop-action-candidate-02152fcc7a94-74a89dbd9fd0690f",
    }

    assert proposal_action_fingerprint(first) == proposal_action_fingerprint(second)


def test_replace_relation_fingerprint_changes_when_relation_changes():
    base = {
        "type": "medical_relation_schema_migration",
        "target": "edge:流行性感冒->死亡",
        "action_payload": {
            "action": "replace_relation",
            "edge_id": "流行性感冒->死亡",
            "new_source": "流行性感冒",
            "new_target": "死亡",
            "new_keywords": "risk_factor_for",
            "qualifiers": {},
        },
    }
    safer = {
        **base,
        "action_payload": {
            **base["action_payload"],
            "new_keywords": "increases_risk_of",
        },
    }

    assert proposal_action_fingerprint(base) != proposal_action_fingerprint(safer)


def test_rejection_class_detects_vaccine_indication_misuse():
    assert (
        rejection_class_from_review_text(
            "疫苗/接种对象关系仍被写成 has_indication，混淆治疗适应证和推荐接种人群"
        )
        == "vaccine_population_as_indication"
    )
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_proposal_fingerprints.py -q
```

Expected: import failure because `proposal_fingerprints.py` does not exist.

- [ ] **Step 3: Implement the fingerprint module**

Create `lightrag/kb_iteration/proposal_fingerprints.py`:

```python
from __future__ import annotations

import hashlib
import json
from typing import Any


def proposal_action_fingerprint(proposal: dict[str, Any]) -> str:
    payload = proposal.get("action_payload")
    if not isinstance(payload, dict) or not payload:
        identity = {
            "type": _string(proposal.get("type")),
            "target": _string(proposal.get("target")),
            "proposed_change": _string(proposal.get("proposed_change")),
        }
    else:
        identity = {
            "type": _string(proposal.get("type")),
            "action_payload": canonical_action_payload(payload),
        }
    return stable_hash(identity)


def decision_action_fingerprint(record: dict[str, Any]) -> str:
    explicit = _string(record.get("proposal_action_fingerprint"))
    if explicit:
        return explicit
    return proposal_action_fingerprint(record)


def canonical_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    action = _string(payload.get("action"))
    if action == "replace_relation":
        return {
            "action": "replace_relation",
            "edge_id": _string(payload.get("edge_id")),
            "new_source": _string(payload.get("new_source")),
            "new_target": _string(payload.get("new_target")),
            "new_keywords": _string(payload.get("new_keywords")),
            "qualifiers": _dict(payload.get("qualifiers")),
        }
    if action == "retire_relation":
        return {
            "action": "retire_relation",
            "edge_id": _string(payload.get("edge_id")),
            "expected_source": _string(payload.get("expected_source")),
            "expected_target": _string(payload.get("expected_target")),
            "current_keywords": _string(payload.get("current_keywords")),
        }
    if action == "value_node_to_qualifier":
        return {
            "action": "value_node_to_qualifier",
            "value_edge_id": _string(payload.get("value_edge_id")),
            "carrier_edge_id": _string(payload.get("carrier_edge_id")),
            "qualifier_key": _string(payload.get("qualifier_key")),
            "qualifier_value": _string(payload.get("qualifier_value")),
        }
    return _json_safe(payload)


def rejection_class_from_review_text(text: str) -> str:
    normalized = str(text or "").casefold()
    if "risk_factor_for" in normalized and any(
        marker in normalized for marker in ("并发", "重症", "死亡", "住院", "结局")
    ):
        return "risk_factor_outcome_or_complication_misuse"
    if "疫苗" in normalized and "has_indication" in normalized:
        return "vaccine_population_as_indication"
    if "抗菌" in normalized or "抗生素" in normalized:
        return "antibacterial_influenza_indication_misuse"
    if "糖皮质激素" in normalized and ("不建议" in normalized or "缺少" in normalized):
        return "corticosteroid_indication_missing_safety_scope"
    if "polarity" in normalized and "positive" in normalized:
        return "invalid_supports_or_refutes_polarity"
    if "performed_by_method" in normalized:
        return "performed_by_method_subject_misuse"
    return "maintainer_rejected_semantics"


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _dict(value: Any) -> dict[str, Any]:
    return _json_safe(value) if isinstance(value, dict) else {}


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_proposal_fingerprints.py -q
```

Expected: `3 passed`.

---

## Task 2: Store Fingerprints In Decision Records

**Files:**

- Modify: `lightrag/api/routers/kb_iteration_routes.py`
- Test: `tests/api/routes/test_kb_iteration_routes.py`

- [ ] **Step 1: Add failing route-helper test**

Add a test near existing decision-record tests in `tests/api/routes/test_kb_iteration_routes.py`:

```python
from lightrag.api.routers import kb_iteration_routes as routes


def test_decision_record_includes_action_fingerprint():
    proposal = {
        "id": "prop-action-candidate-risk",
        "type": "medical_relation_schema_migration",
        "target": "edge:流行性感冒->死亡",
        "risk": "medium",
        "reason": "unsafe risk relation",
        "evidence": ["source_id: x"],
        "action_payload": {
            "action": "replace_relation",
            "edge_id": "流行性感冒->死亡",
            "new_source": "流行性感冒",
            "new_target": "死亡",
            "new_keywords": "risk_factor_for",
            "qualifiers": {},
        },
    }
    request = routes.ProposalDecisionRequest(
        reviewer="reviewer",
        reason="risk_factor_for 不应指向死亡结局",
    )

    record = routes._build_proposal_decision_record(
        "prop-action-candidate-risk",
        "reject",
        proposal,
        request,
    )

    assert record["proposal_action_fingerprint"]
    assert record["semantic_rejection_class"] == "risk_factor_outcome_or_complication_misuse"
    assert record["action_payload"]["new_keywords"] == "risk_factor_for"
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/api/routes/test_kb_iteration_routes.py::test_decision_record_includes_action_fingerprint -q
```

Expected: FAIL with missing `proposal_action_fingerprint`.

- [ ] **Step 3: Update decision record construction**

In `lightrag/api/routers/kb_iteration_routes.py`, import:

```python
from lightrag.kb_iteration.proposal_fingerprints import (
    proposal_action_fingerprint,
    rejection_class_from_review_text,
)
```

Update `_build_proposal_decision_record()` so the returned dict includes:

```python
        "proposal_action_fingerprint": proposal_action_fingerprint(proposal),
        "semantic_rejection_class": (
            rejection_class_from_review_text(audit_review["reason"])
            if decision == "reject"
            else ""
        ),
        "action_payload": proposal.get("action_payload", {}),
```

Keep existing fields unchanged so old markdown records remain readable.

- [ ] **Step 4: Run the focused test**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/api/routes/test_kb_iteration_routes.py::test_decision_record_includes_action_fingerprint -q
```

Expected: `1 passed`.

---

## Task 3: Use Rejected Fingerprints During Deterministic Scan

**Files:**

- Modify: `lightrag/kb_iteration/issue_ledger.py`
- Test: `tests/kg/test_kb_iteration_issue_ledger.py`

- [ ] **Step 1: Add failing regression test**

Add a test that creates `rejected_changes.md` with a fingerprinted rejected decision, then scans a candidate with a different proposal ID but identical action:

```python
import json
from pathlib import Path

from lightrag.kb_iteration.issue_ledger import _rejected_action_fingerprints
from lightrag.kb_iteration.proposal_fingerprints import proposal_action_fingerprint


def test_rejected_action_fingerprints_read_new_decision_fields(tmp_path: Path):
    proposal = {
        "type": "medical_relation_schema_migration",
        "target": "edge:流行性感冒->流感重症",
        "action_payload": {
            "action": "replace_relation",
            "edge_id": "流行性感冒->流感重症",
            "new_source": "流行性感冒",
            "new_target": "流感重症",
            "new_keywords": "risk_factor_for",
            "qualifiers": {},
        },
    }
    record = {
        "proposal_id": "old-id",
        "decision": "reject",
        "proposal_type": proposal["type"],
        "proposal_target": proposal["target"],
        "proposal_action_fingerprint": proposal_action_fingerprint(proposal),
        "semantic_rejection_class": "risk_factor_outcome_or_complication_misuse",
        "action_payload": proposal["action_payload"],
    }
    (tmp_path / "rejected_changes.md").write_text(
        "# Rejected Changes\n\n## old-id\n\n```json\n"
        + json.dumps(record, ensure_ascii=False)
        + "\n```\n",
        encoding="utf-8",
    )

    assert proposal_action_fingerprint(proposal) in _rejected_action_fingerprints(tmp_path)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_issue_ledger.py::test_rejected_action_fingerprints_read_new_decision_fields -q
```

Expected: FAIL because `_rejected_action_fingerprints()` does not use the new fingerprint field.

- [ ] **Step 3: Update `issue_ledger.py` imports and helpers**

In `lightrag/kb_iteration/issue_ledger.py`, import:

```python
from .proposal_fingerprints import decision_action_fingerprint, proposal_action_fingerprint
```

Replace `_decision_payload_fingerprint()` implementation with:

```python
def _decision_payload_fingerprint(payload: dict[str, Any]) -> str:
    return decision_action_fingerprint(payload)
```

Replace `_candidate_action_fingerprint(candidate)` internals to call:

```python
return proposal_action_fingerprint(
    {
        "type": candidate.get("proposal_type") or "medical_relation_schema_migration",
        "target": candidate.get("target") or candidate.get("edge_id") or "",
        "action_payload": candidate.get("action_payload") or candidate,
    }
)
```

If `_candidate_action_fingerprint()` currently receives raw action payloads, preserve the current call sites and wrap the candidate without changing external behavior.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_issue_ledger.py tests/kg/test_kb_iteration_proposal_fingerprints.py -q
```

Expected: both files pass.

---

## Task 4: Drop Rejected Suffix Variants Before Approval Queue

**Files:**

- Modify: `lightrag/kb_iteration/proposal_orchestrator.py`
- Test: `tests/kg/test_kb_iteration_proposal_orchestrator.py`

- [ ] **Step 1: Add failing orchestrator test**

Add a test near duplicate/stale proposal tests:

```python
from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.proposal_orchestrator import merge_and_rank_proposals
from lightrag.kb_iteration.proposal_fingerprints import proposal_action_fingerprint


def test_merge_drops_rejected_semantic_suffix_variant(tmp_path):
    proposal = ImprovementProposal(
        id="prop-action-candidate-risk-new-suffix",
        type="medical_relation_schema_migration",
        target="edge:流行性感冒->流感重症",
        proposed_change="bad risk relation",
        reason="bad risk relation",
        evidence=["source_id: x"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={"medical_schema_issue_count": -1},
        action_payload={
            "action": "replace_relation",
            "edge_id": "流行性感冒->流感重症",
            "new_source": "流行性感冒",
            "new_target": "流感重症",
            "new_keywords": "risk_factor_for",
            "qualifiers": {},
        },
    )
    rejected_record = {
        "proposal_id": "old",
        "decision": "reject",
        "proposal_type": proposal.type,
        "proposal_target": proposal.target,
        "proposal_action_fingerprint": proposal_action_fingerprint(proposal.to_dict()),
        "semantic_rejection_class": "risk_factor_outcome_or_complication_misuse",
        "action_payload": proposal.action_payload,
    }

    selected, dropped = merge_and_rank_proposals(
        [proposal],
        max_proposals=20,
        rejected_decision_records=[rejected_record],
    )

    assert selected == []
    assert dropped[0]["reason"] == "rejected_decision_fingerprint"
```

If `merge_and_rank_proposals()` does not currently accept `rejected_decision_records`, add this parameter in the test and implementation.

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_proposal_orchestrator.py::test_merge_drops_rejected_semantic_suffix_variant -q
```

Expected: FAIL because the merge layer does not accept/use rejected fingerprints.

- [ ] **Step 3: Implement rejected fingerprint filtering**

In `proposal_orchestrator.py`:

```python
from .proposal_fingerprints import decision_action_fingerprint, proposal_action_fingerprint
```

Add helper:

```python
def _rejected_decision_fingerprint_set(records: list[dict[str, Any]]) -> set[str]:
    return {
        decision_action_fingerprint(record)
        for record in records
        if str(record.get("decision", "")).casefold() in {"reject", "rejected"}
        and decision_action_fingerprint(record)
    }
```

In `merge_and_rank_proposals(...)`, before duplicate ID filtering:

```python
    rejected_fingerprints = _rejected_decision_fingerprint_set(
        rejected_decision_records or []
    )
    surviving: list[ImprovementProposal] = []
    for proposal in proposals:
        if proposal_action_fingerprint(proposal.to_dict()) in rejected_fingerprints:
            dropped.append(_dropped_proposal(proposal, "rejected_decision_fingerprint"))
            continue
        surviving.append(proposal)
    proposals = surviving
```

Thread `rejected_decision_records` from `agent_pipeline.py` where accepted/rejected memory is loaded for the package.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_proposal_orchestrator.py tests/kg/test_kb_iteration_agent_pipeline.py -q
```

Expected: pass.

---

## Task 5: Add Medical Safety Validators Before Queueing

**Files:**

- Modify: `lightrag/kb_iteration/proposals.py`
- Test: `tests/kg/test_kb_iteration_proposals.py`

- [ ] **Step 1: Add failing validator tests**

Add tests:

```python
import pytest

from lightrag.kb_iteration.models import ImprovementProposal
from lightrag.kb_iteration.proposals import validate_proposal


def _migration(source, target, new_keywords):
    return ImprovementProposal(
        id="prop-test",
        type="medical_relation_schema_migration",
        target=f"edge:{source}->{target}",
        proposed_change="test",
        reason="test",
        evidence=["source_id: x"],
        confidence=0.9,
        risk="medium",
        requires_approval=True,
        expected_metric_change={"medical_schema_issue_count": -1},
        action_payload={
            "action": "replace_relation",
            "edge_id": f"{source}->{target}",
            "expected_source": source,
            "expected_target": target,
            "current_keywords": "legacy",
            "new_source": source,
            "new_target": target,
            "new_keywords": new_keywords,
            "qualifiers": {},
        },
    )


@pytest.mark.parametrize("target", ["哮喘", "糖尿病", "慢性阻塞性肺疾病", "急性心力衰竭"])
def test_rejects_vaccine_to_chronic_disease_has_indication(target):
    with pytest.raises(ValueError, match="vaccine recommended_for"):
        validate_proposal(_migration("流感疫苗", target, "has_indication"))


@pytest.mark.parametrize("target", ["死亡", "流感相关住院", "重症", "危重症"])
def test_rejects_disease_to_outcome_risk_factor_for(target):
    with pytest.raises(ValueError, match="risk_factor_for"):
        validate_proposal(_migration("流行性感冒", target, "risk_factor_for"))


def test_rejects_positive_polarity_for_supports_or_refutes():
    proposal = _migration("心电图", "心脏损伤", "supports_or_refutes")
    proposal.action_payload["qualifiers"] = {"polarity": "positive"}
    with pytest.raises(ValueError, match="supports_or_refutes qualifier polarity"):
        validate_proposal(proposal)
```

- [ ] **Step 2: Run tests and confirm failures**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_proposals.py -q
```

Expected: new tests fail for missing validator messages.

- [ ] **Step 3: Implement validator checks**

In `proposals.py`, inside `_validate_replace_relation_payload()` after canonical relation checks, add:

```python
    if new_keywords == "has_indication" and _looks_like_vaccine_to_chronic_condition(
        str(action_payload["new_source"]),
        str(action_payload["new_target"]),
    ):
        raise ValueError(
            "proposal action_payload vaccine recommended_for population/condition "
            "must not be modeled as has_indication"
        )
    if new_keywords == "risk_factor_for" and _looks_like_disease_to_outcome_or_severity(
        str(action_payload["new_source"]),
        str(action_payload["new_target"]),
    ):
        raise ValueError(
            "proposal action_payload risk_factor_for must not connect disease "
            "directly to outcome, hospitalization, death, or severity endpoints"
        )
```

Add helper functions near existing `_looks_like_*` helpers:

```python
def _looks_like_vaccine_to_chronic_condition(source: str, target: str) -> bool:
    return "疫苗" in source and any(
        term in target
        for term in (
            "哮喘",
            "糖尿病",
            "慢性阻塞性肺疾病",
            "慢性支气管炎",
            "炎症性肠病",
            "心力衰竭",
            "心血管病",
        )
    )


def _looks_like_disease_to_outcome_or_severity(source: str, target: str) -> bool:
    if not any(term in source for term in ("流感", "感染", "肺炎")):
        return False
    return any(
        term in target
        for term in (
            "死亡",
            "住院",
            "重症",
            "危重",
            "不良妊娠结局",
            "早产",
            "低出生体重",
            "流产",
            "心脏停搏",
        )
    )
```

The polarity enum is already defined in `medical_schema.py`; this task ensures any bypass path still hits `validate_relation_instance()`.

- [ ] **Step 4: Run proposal tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_proposals.py -q
```

Expected: pass.

---

## Task 6: Tighten Deterministic Medical Generators

**Files:**

- Modify: `lightrag/kb_iteration/deterministic_proposals/risk_safety.py`
- Modify: `lightrag/kb_iteration/deterministic_proposals/treatment.py`
- Modify: `lightrag/kb_iteration/deterministic_proposals/prevention.py`
- Test: `tests/kg/test_kb_iteration_proposal_orchestrator.py`

- [ ] **Step 1: Add generator regression tests**

Add tests that call `build_proposal_task_packs()` or the deterministic scan helper with synthetic issues:

```python
def test_risk_generator_rejects_flu_to_death_risk_factor(package_with_issue):
    issue = {
        "issue_ref": "medical_schema_issues:0",
        "issue_family": "risk_safety",
        "issue_kind": "legacy_overloaded_relation",
        "source": "流行性感冒",
        "source_type": "disease",
        "target": "死亡",
        "target_type": "complication",
        "keywords": "并发风险",
        "candidate_predicates": ["risk_factor_for"],
        "edge_id": "流行性感冒->死亡",
    }
    result = scan_deterministic_candidates(package_with_issue([issue]))
    assert result.candidates == []
    assert result.issue_routes[0].route_state in {"llm_residual", "blocked_safety"}
    assert result.rejections[0]["error_code"] in {
        "OUTCOME_OR_SEVERITY_NOT_RISK_FACTOR",
        "ACTION_CANDIDATE_INVALID",
    }
```

Use the existing fixture style in `tests/kg/test_kb_iteration_proposal_orchestrator.py`; do not introduce a new heavy fixture if the file already has a package builder.

- [ ] **Step 2: Run the test and confirm failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_proposal_orchestrator.py -q
```

Expected: at least the new test fails if the generator still emits unsafe candidates.

- [ ] **Step 3: Update generator guards**

In `risk_safety.py`:

- Treat target terms `死亡`, `住院`, `重症`, `危重症`, `早产`, `低出生体重`, `不良妊娠结局`, `流产` as outcome/severity endpoints.
- Generate `increases_risk_of` only when `candidate_predicates` contains `increases_risk_of` and the issue text/evidence explicitly says risk increase.
- Do not choose `risk_factor_for` for source type `Disease` and target type `Complication`/severity/outcome.

In `treatment.py`:

- Reject `has_indication` if source contains `抗菌药物` or `抗生素` and target contains `流感`.
- Reject `has_indication` if evidence text contains `不推荐`, `避免`, `不建议`, `慎用`, or `无获益`.
- Return structured rejection `RESTRICTIVE_EVIDENCE_NOT_INDICATION`.

In `prevention.py`:

- For vaccine source and chronic disease target, generate `recommended_for` only when qualifiers include `purpose=prevention` and one of `condition`, `population`, or `age`.
- Reject `targets_disease` where the target is a chronic condition or adverse event such as `格林-巴利综合征`.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_proposal_orchestrator.py tests/kg/test_kb_iteration_proposals.py -q
```

Expected: pass.

---

## Task 7: Fix Invalid Subagent Output Feedback

**Files:**

- Modify: `lightrag/kb_iteration/agent_outputs.py`
- Modify: `lightrag/kb_iteration/subagent_contracts.py`
- Modify: `lightrag/kb_iteration/prompts/subagents/base_zh.md`
- Modify: `lightrag/kb_iteration/prompts/subagents/schema_repair_zh.md`
- Modify: `lightrag/kb_iteration/prompts/subagents/treatment_zh.md`
- Test: `tests/kg/test_kb_iteration_agent_outputs.py`

- [ ] **Step 1: Add tests for exact invalid-output feedback**

Add tests:

```python
import pytest

from lightrag.kb_iteration.agent_outputs import parse_agent_json_output


def test_schema_repair_evidence_object_unknown_field_error_is_precise():
    raw = {
        "proposals": [
            {
                "id": "schema-repair-bad-evidence",
                "type": "medical_relation_schema_migration",
                "target": "edge:x->y",
                "proposed_change": "change",
                "reason": "reason",
                "evidence": [
                    {
                        "source_id": "doc-1",
                        "file_path": "a.pdf",
                        "evidence_quote": "quote",
                    }
                ],
                "confidence": 0.9,
                "risk": "medium",
                "requires_approval": True,
                "expected_metric_change": {},
                "action_payload": {
                    "action": "replace_relation",
                    "edge_id": "x->y",
                    "expected_source": "x",
                    "expected_target": "y",
                    "current_keywords": "检测方法",
                    "new_source": "x",
                    "new_target": "y",
                    "new_keywords": "performed_by_method",
                    "qualifiers": {},
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="EVIDENCE_MUST_BE_STRING"):
        parse_agent_json_output(raw)


def test_candidate_expansion_requires_endpoint_types():
    raw = {
        "proposals": [
            {
                "id": "candidate-expansion-missing-types",
                "type": "candidate_kg_expansion",
                "target": "edge:x->y",
                "proposed_change": "change",
                "reason": "reason",
                "evidence": ["source_id: doc-1; file_path: a.pdf; evidence_quote: 证据片段"],
                "confidence": 0.9,
                "risk": "medium",
                "requires_approval": True,
                "expected_metric_change": {},
                "action_payload": {
                    "candidate_nodes": [],
                    "candidate_edges": [
                        {
                            "source": "x",
                            "target": "y",
                            "keywords": "has_indication",
                            "source_id": "doc-1",
                            "file_path": "a.pdf",
                        }
                    ],
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="CANDIDATE_EDGE_TYPES_REQUIRED"):
        parse_agent_json_output(raw)
```

Adapt `parse_agent_json_output` name if the module uses a different public parser; use the existing parser entry point in the file.

- [ ] **Step 2: Run tests and confirm current behavior**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_agent_outputs.py -q
```

Expected: tests fail if error codes are too vague or parser entry point needs exact adaptation.

- [ ] **Step 3: Implement precise error codes and prompt constraints**

In `agent_outputs.py`:

- If `proposal["evidence"]` contains dicts, raise `ValueError("EVIDENCE_MUST_BE_STRING: proposal evidence items must be grounded strings copied from allowed evidence spans")`.
- If `candidate_kg_expansion.action_payload.candidate_edges[*]` lacks `source_type` or `target_type`, raise `ValueError("CANDIDATE_EDGE_TYPES_REQUIRED: candidate_edges[i] must provide endpoint types")`.

In `subagent_contracts.py`, ensure candidate expansion contracts require:

```python
"candidate_edges_required_fields": (
    "source",
    "target",
    "source_type",
    "target_type",
    "keywords",
    "source_id",
    "file_path",
)
```

In `prompts/subagents/base_zh.md`, add:

```markdown
- `evidence` 必须是字符串数组，不允许对象。格式示例：
  `"source_id: doc-xxx; file_path: 指南.pdf; evidence_quote: 原文短句"`
- `candidate_kg_expansion.action_payload.candidate_edges[]` 必须包含
  `source_type` 和 `target_type`。
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_agent_outputs.py tests/kg/test_kb_iteration_agent_pipeline.py -q
```

Expected: pass.

---

## Task 8: Add Family-Aware Residual Batching

**Files:**

- Modify: `lightrag/kb_iteration/agent_pipeline.py`
- Modify: `lightrag/kb_iteration/proposal_orchestrator.py`
- Modify: `lightrag/api/routers/kb_iteration_routes.py`
- Test: `tests/kg/test_kb_iteration_agent_pipeline.py`
- Test: `tests/api/routes/test_kb_iteration_routes.py`

- [ ] **Step 1: Add config fields**

Extend `LLMAgentPipelineConfig` with:

```python
llm_residual_family_caps: dict[str, int] | None = None
max_llm_residual_issues_per_run: int = 120
```

Default resolved caps:

```python
DEFAULT_LLM_RESIDUAL_FAMILY_CAPS = {
    "diagnosis": 20,
    "treatment": 30,
    "risk_safety": 30,
    "prevention": 20,
    "clinical_modeling": 15,
    "entity_cleanup": 15,
    "legacy_schema": 10,
}
```

- [ ] **Step 2: Add failing batching test**

Add to `tests/kg/test_kb_iteration_agent_pipeline.py`:

```python
def test_llm_residual_packs_use_family_caps(package_with_residual_issues, fake_client):
    package = package_with_residual_issues(
        treatment_count=60,
        diagnosis_count=50,
        risk_safety_count=50,
    )
    result = run_llm_agent_pipeline(
        workspace="influenza_medical_v1",
        package_dir=package,
        client=fake_client,
        config=LLMAgentPipelineConfig(
            max_subagent_tasks=12,
            max_subagent_issues_per_task=5,
            llm_residual_family_caps={
                "treatment": 10,
                "diagnosis": 10,
                "risk_safety": 10,
            },
            max_llm_residual_issues_per_run=30,
        ),
        profile="influenza",
    )
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_stage = next(stage for stage in trace["stages"] if stage["stage"] == "propose")
    assert propose_stage["llm_residual_issue_count"] == 30
```

Use existing fixtures in the file; if fixture names differ, implement a small local package builder using the file's established helper style.

- [ ] **Step 3: Implement batching**

In `proposal_orchestrator.py`, add a function:

```python
def select_llm_residual_issues_by_family(
    issues: list[dict[str, Any]],
    *,
    family_caps: dict[str, int],
    max_total: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for issue in issues:
        family = _issue_family(issue)
        cap = family_caps.get(family, 0)
        if cap <= 0:
            continue
        if counts.get(family, 0) >= cap:
            continue
        if len(selected) >= max_total:
            break
        selected.append(issue)
        counts[family] = counts.get(family, 0) + 1
    return selected
```

In `agent_pipeline.py`, call this before `build_llm_residual_task_packs()`. Record counts in trace:

```python
stage_trace["llm_residual_issue_count"] = len(selected_residual_issues)
stage_trace["llm_residual_family_caps"] = resolved_family_caps
```

In `kb_iteration_routes.py`, extend `RunLLMReviewRequest` with:

```python
llm_residual_family_caps: dict[str, int] | None = None
max_llm_residual_issues_per_run: int = Field(default=120, ge=1, le=500)
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/kg/test_kb_iteration_agent_pipeline.py tests/api/routes/test_kb_iteration_routes.py -q
```

Expected: pass.

---

## Task 9: Web UI Visibility For Funnel Blocks

**Files:**

- Modify: `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx`
- Modify: `lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx`
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/components/kg-maintenance/kgMaintenanceLLMReviewRequest.ts`

- [ ] **Step 1: Add API types**

In `lightrag_webui/src/api/lightrag.ts`, add request fields:

```ts
llm_residual_family_caps?: Record<string, number>
max_llm_residual_issues_per_run?: number
```

- [ ] **Step 2: Update default request**

In `kgMaintenanceLLMReviewRequest.ts`, default:

```ts
llm_residual_family_caps: {
  diagnosis: 20,
  treatment: 30,
  risk_safety: 30,
  prevention: 20,
  clinical_modeling: 15,
  entity_cleanup: 15,
  legacy_schema: 10
},
max_llm_residual_issues_per_run: 120
```

- [ ] **Step 3: Add display test**

In `LLMReviewPanels.test.tsx`, assert the panel renders:

```tsx
expect(screen.getByText('LLM 残余问题上限')).toBeInTheDocument()
expect(screen.getByText('语义指纹拦截')).toBeInTheDocument()
```

- [ ] **Step 4: Implement compact display**

In `LLMReviewPanels.tsx`, show:

- raw issue count;
- deterministic covered count;
- LLM residual selected count;
- rejected fingerprint drop count;
- invalid subagent output count.

Use existing panel styling; do not introduce a new visual system.

- [ ] **Step 5: Run frontend tests**

Run:

```powershell
cd lightrag_webui
bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx src/api/lightrag.test.ts
```

Expected: tests pass. If `bun` is not on PATH in the shell, record that explicitly and run available TypeScript/ESLint checks instead.

---

## Task 10: Real Run Verification

**Files:**

- No source files; this task verifies behavior on `influenza_medical_v1`.
- Artifacts under `work/kb-iteration/influenza_medical_v1/`.

- [ ] **Step 1: Run focused backend tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/kg/test_kb_iteration_proposal_fingerprints.py `
  tests/kg/test_kb_iteration_issue_ledger.py `
  tests/kg/test_kb_iteration_proposal_orchestrator.py `
  tests/kg/test_kb_iteration_proposals.py `
  tests/kg/test_kb_iteration_agent_outputs.py `
  tests/kg/test_kb_iteration_agent_pipeline.py `
  tests/api/routes/test_kb_iteration_routes.py `
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run ruff**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check `
  lightrag/kb_iteration `
  lightrag/api/routers/kb_iteration_routes.py `
  tests/kg/test_kb_iteration_proposal_fingerprints.py `
  tests/kg/test_kb_iteration_issue_ledger.py `
  tests/kg/test_kb_iteration_proposal_orchestrator.py `
  tests/kg/test_kb_iteration_proposals.py `
  tests/kg/test_kb_iteration_agent_outputs.py `
  tests/kg/test_kb_iteration_agent_pipeline.py `
  tests/api/routes/test_kb_iteration_routes.py
```

Expected: `All checks passed`.

- [ ] **Step 3: Run one real Agent validation round**

Use the same Flash + 200 configuration as the previous real run. Expected improvement:

- queued proposals should no longer include suffix variants of previously rejected fingerprints;
- invalid subagent outputs should be `0`, or their errors should be more precise and action-guiding;
- unsafe vaccine `has_indication`, antibacterial influenza indication, and disease-to-outcome `risk_factor_for` proposals should be absent from `approval_queue.md`.

- [ ] **Step 4: Verify artifacts**

Run:

```powershell
@'
import json, yaml
from pathlib import Path
pkg = Path("work/kb-iteration/influenza_medical_v1")
queue = (pkg / "approval_queue.md").read_text(encoding="utf-8")
start = queue.find("proposals:")
proposals = yaml.safe_load(queue[start:]).get("proposals", []) if start >= 0 else []
bad = []
for proposal in proposals:
    payload = proposal.get("action_payload") or {}
    source = str(payload.get("new_source") or payload.get("expected_source") or "")
    target = str(payload.get("new_target") or payload.get("expected_target") or "")
    keyword = str(payload.get("new_keywords") or "")
    if "疫苗" in source and keyword == "has_indication":
        bad.append((proposal["id"], "vaccine_has_indication"))
    if source in {"抗菌药物", "抗生素"} and "流感" in target and keyword == "has_indication":
        bad.append((proposal["id"], "antibacterial_has_indication"))
    if keyword == "risk_factor_for" and any(term in target for term in ("死亡", "住院", "重症", "危重")):
        bad.append((proposal["id"], "outcome_risk_factor"))
print({"proposal_count": len(proposals), "bad": bad})
raise SystemExit(1 if bad else 0)
'@ | .venv\Scripts\python.exe -
```

Expected: `bad: []`.

---

## Acceptance Criteria

- Rejected proposal suffix variants do not reappear in later queues when their action fingerprint is identical.
- Decision records include:
  - `proposal_action_fingerprint`;
  - `semantic_rejection_class`;
  - `action_payload`.
- Deterministic scan and proposal merge both use the same fingerprint implementation.
- Unsafe proposal patterns are blocked before approval queue:
  - vaccine/chronic disease as `has_indication`;
  - disease/outcome/severity as `risk_factor_for`;
  - antibacterial/antibiotic as influenza treatment indication;
  - `supports_or_refutes` with `polarity=positive`;
  - candidate expansion without endpoint types.
- LLM residual batching sends substantially more than 28 issues per run when raw residual volume is high, while preserving family caps.
- Focused pytest suite and ruff pass.
- One real Agent validation run produces no known-bad proposal patterns listed in Task 10.

## Risks And Constraints

- Do not delete existing `accepted_changes.md` or `rejected_changes.md`; new logic must remain backward-compatible with old records that lack fingerprints.
- Do not rely on proposal IDs for memory suppression.
- Do not let LLM proposals mutate KG directly; accepted proposals still go through deterministic Apply Engine.
- Do not broaden medical claims while fixing schema. If relation semantics need qualifiers and qualifiers are unavailable, reject or defer instead of inventing.
- Keep UI changes observational; the safety behavior belongs in backend validation and generation.

## Self-Review

- Spec coverage:
  - Repeated rejected proposals: Tasks 1-4.
  - Unsafe relation types: Tasks 5-6.
  - Invalid subagent outputs: Task 7.
  - Low residual throughput: Task 8.
  - Web visibility: Task 9.
  - Real verification: Task 10.
- Placeholder scan:
  - This plan contains no `TBD`, `TODO`, or unspecified implementation owner.
- Type consistency:
  - Fingerprint functions accept plain `dict[str, Any]`.
  - Decision records store explicit fingerprint fields while preserving existing markdown JSON structure.
  - Pipeline config fields are named consistently: `llm_residual_family_caps` and `max_llm_residual_issues_per_run`.
