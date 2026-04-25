from lightrag_enterprise.agents import AGENT_SPECS
from lightrag_enterprise.domain.crm import CRMContact, CRMTicket, InMemoryCRMRepository
from lightrag_enterprise.domain.internal_chat import (
    ChatMessage,
    ChatThread,
    InMemoryChatRepository,
)
from lightrag_enterprise.skills import DEFAULT_SKILL_REGISTRY
from lightrag_enterprise.subagents import SUBAGENT_SPECS


EXPECTED_SKILLS = {
    "query_lightrag",
    "query_lightrag_context_only",
    "ingest_document",
    "ingest_batch",
    "reindex_workspace",
    "delete_document_by_id",
    "delete_entity",
    "delete_relation",
    "merge_entities",
    "sync_model_catalog",
    "get_model_catalog",
    "route_model_by_policy",
    "check_cost_policy",
    "create_crm_contact",
    "update_crm_contact",
    "create_ticket",
    "update_ticket",
    "search_conversations",
    "summarize_thread",
    "generate_report",
    "audit_action",
    "validate_json_output",
}


def test_agent_subagent_and_skill_catalogs_are_declared():
    assert "orchestrator_agent" in AGENT_SPECS
    assert "prompt_injection_guard_subagent" in SUBAGENT_SPECS
    assert EXPECTED_SKILLS.issubset(set(DEFAULT_SKILL_REGISTRY.list_names()))


def test_crm_repository_contract():
    repo = InMemoryCRMRepository()
    contact = repo.create_contact(
        CRMContact(
            contact_id="c1",
            tenant_id="t1",
            workspace="sales",
            name="Ana Cliente",
        )
    )
    ticket = repo.create_ticket(
        CRMTicket(
            ticket_id="tck1",
            tenant_id="t1",
            workspace="sales",
            title="Need support",
            contact_id=contact.contact_id,
        )
    )

    assert repo.update_contact("c1", name="Ana C.").name == "Ana C."
    assert repo.update_ticket(ticket.ticket_id, status="closed").status == "closed"


def test_internal_chat_repository_contract():
    repo = InMemoryChatRepository()
    repo.threads["th1"] = ChatThread(
        thread_id="th1",
        tenant_id="t1",
        workspace_id="corp",
        channel_id="support",
    )
    repo.add_message(
        ChatMessage(
            message_id="m1",
            tenant_id="t1",
            workspace_id="corp",
            channel_id="support",
            sender_id="u1",
            body="LightRAG has cited support answers.",
        )
    )

    assert repo.search_messages("t1", "corp", "cited")
    assert repo.upsert_thread_memory("th1", "Summary").memory_summary == "Summary"
