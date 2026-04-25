from __future__ import annotations

from dataclasses import dataclass

from lightrag_enterprise.security.policies import detect_prompt_injection


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str
    requires_human_approval: bool = False


DESTRUCTIVE_ACTIONS = {
    "delete_document_by_id",
    "delete_entity",
    "delete_relation",
    "merge_entities",
    "reindex_workspace",
}


def evaluate_execution_guardrails(action: str, user_text: str = "") -> GuardrailDecision:
    if detect_prompt_injection(user_text):
        return GuardrailDecision(False, "Prompt-injection pattern detected.")
    if action in DESTRUCTIVE_ACTIONS:
        return GuardrailDecision(
            False,
            "Destructive action requires explicit human approval.",
            requires_human_approval=True,
        )
    return GuardrailDecision(True, "Action allowed by default guardrails.")
