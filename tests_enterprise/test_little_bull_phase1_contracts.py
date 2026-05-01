from __future__ import annotations

import inspect
import re
import sys
from types import SimpleNamespace

import pytest

from lightrag_enterprise.little_bull import admin_store
from lightrag_enterprise.little_bull import models
from lightrag_enterprise.system.little_bull_admin_schema import (
    LITTLE_BULL_ADMIN_SCHEMA_SQL,
)


PHASE1_TABLES = {
    "little_bull_provider_credentials": "ProviderCredential",
    "little_bull_model_catalog_snapshots": "ModelCatalogSnapshot",
    "little_bull_knowledge_groups": "KnowledgeGroup",
    "little_bull_knowledge_subgroups": "KnowledgeSubgroup",
    "little_bull_document_registry": "DocumentRegistry",
    "little_bull_note_registry": "NoteRegistry",
    "little_bull_embedding_index_versions": "EmbeddingIndexVersion",
    "little_bull_indexing_jobs": "IndexingJob",
    "little_bull_llm_usage_ledger": "LlmUsageLedger",
    "little_bull_graph_edge_origins": "GraphEdgeOrigin",
    "little_bull_graph_clusters": "GraphCluster",
    "little_bull_knowledge_trails": "KnowledgeTrail",
    "little_bull_knowledge_trail_steps": "KnowledgeTrailStep",
    "little_bull_backlinks": "Backlink",
    "little_bull_graph_chat_sessions": "GraphChatSession",
    "little_bull_agent_builder_sessions": "AgentBuilderSession",
    "little_bull_agent_context_budgets": "AgentContextBudget",
    "little_bull_markdown_notes": "MarkdownNote",
    "little_bull_wiki_links": "WikiLink",
    "little_bull_tag_registry": "TagRegistry",
    "little_bull_content_maps": "ContentMap",
    "little_bull_canvas_boards": "CanvasBoard",
    "little_bull_canvas_nodes": "CanvasNode",
    "little_bull_canvas_edges": "CanvasEdge",
    "little_bull_knowledge_inbox_items": "KnowledgeInboxItem",
    "little_bull_daily_notes": "DailyNote",
    "little_bull_note_templates": "NoteTemplate",
    "little_bull_command_palette_actions": "CommandPaletteAction",
    "little_bull_source_provenance": "SourceProvenance",
    "little_bull_knowledge_dossiers": "KnowledgeDossier",
    "little_bull_legal_matter_extraction_runs": "LegalMatterExtractionRun",
}

WORKSPACE_NULLABLE_TABLES = {
    "little_bull_provider_credentials",
    "little_bull_model_catalog_snapshots",
    "little_bull_command_palette_actions",
}

ALLOWED_SECRET_COLUMNS = {"secret_ref", "secret_fingerprint", "credential_kind"}
FORBIDDEN_RAW_SECRET_COLUMNS = {
    "api_key",
    "access_token",
    "refresh_token",
    "password",
    "secret_value",
}


def _table_blocks() -> dict[str, str]:
    blocks = {}
    pattern = re.compile(
        r"CREATE TABLE IF NOT EXISTS (little_bull_[a-z_]+) \((.*?)\n\);",
        re.DOTALL,
    )
    for match in pattern.finditer(LITTLE_BULL_ADMIN_SCHEMA_SQL):
        blocks[match.group(1)] = match.group(2)
    return blocks


def test_phase1_schema_tables_and_contract_classes_exist():
    blocks = _table_blocks()

    for table_name, class_name in PHASE1_TABLES.items():
        assert table_name in blocks
        assert hasattr(models, class_name)


def test_phase1_tables_have_scope_and_audit_columns():
    blocks = _table_blocks()

    for table_name in PHASE1_TABLES:
        block = blocks[table_name]
        assert "tenant_id TEXT NOT NULL" in block
        if table_name in WORKSPACE_NULLABLE_TABLES:
            assert "workspace_id TEXT REFERENCES system_workspaces" in block
        else:
            assert "workspace_id TEXT NOT NULL" in block
        for column in ("created_by", "updated_by", "created_at", "updated_at"):
            assert column in block


def test_phase1_schema_has_no_raw_secret_columns():
    blocks = _table_blocks()

    for table_name in PHASE1_TABLES:
        block = blocks[table_name]
        column_names = {
            match.group(1)
            for match in re.finditer(r"^\s+([a-z_]+)\s+[A-Z]", block, re.MULTILINE)
        }
        for column_name in column_names - ALLOWED_SECRET_COLUMNS:
            assert column_name not in FORBIDDEN_RAW_SECRET_COLUMNS, (
                table_name,
                column_name,
            )


def test_phase1_schema_guards_append_only_and_nullable_uniques():
    sql = LITTLE_BULL_ADMIN_SCHEMA_SQL

    assert "little_bull_prevent_usage_ledger_update" in sql
    assert "trg_little_bull_usage_ledger_append_only" in sql
    assert "idx_little_bull_llm_usage_ledger_group_scope" in sql
    assert "uq_little_bull_provider_credentials_tenant" in sql
    assert "uq_little_bull_command_palette_actions_tenant" in sql
    assert "uq_little_bull_agent_context_budgets_default" in sql
    assert "chk_little_bull_note_registry_markdown_classified" in sql
    assert "idx_little_bull_backlinks_source" in sql
    assert "idx_little_bull_source_provenance_document" in sql
    assert "idx_little_bull_source_provenance_note" in sql
    assert "idx_little_bull_canvas_nodes_ref" in sql
    assert "idx_little_bull_content_maps_root_note" in sql
    assert "idx_little_bull_knowledge_trail_steps_note" in sql
    assert "idx_little_bull_knowledge_trail_steps_document" in sql
    assert "idx_little_bull_knowledge_trail_steps_canvas" in sql
    assert "idx_little_bull_knowledge_inbox_items_source" in sql
    insert_source = inspect.getsource(
        admin_store.LittleBullAdminStore.insert_llm_usage_ledger
    )
    reserve_source = inspect.getsource(
        admin_store.LittleBullAdminStore.reserve_llm_usage_budget
    )
    assert "llm_usage_ledger" in insert_source
    assert insert_source.index("pg_advisory_xact_lock") < insert_source.index(
        "created_at = utc_now()"
    )
    assert reserve_source.index("llm_usage_ledger") < reserve_source.index(
        "created_at = utc_now()"
    )


def test_phase8_admin_store_upserts_have_id_update_paths_and_scope_guards():
    methods = {
        "canvas": (
            admin_store.LittleBullAdminStore.upsert_canvas_board,
            "canvas_board_scope_mismatch",
        ),
        "content_map": (
            admin_store.LittleBullAdminStore.upsert_content_map,
            "content_map_scope_mismatch",
        ),
        "knowledge_trail": (
            admin_store.LittleBullAdminStore.upsert_knowledge_trail,
            "knowledge_trail_scope_mismatch",
        ),
    }

    for name, (method, guard_token) in methods.items():
        source = inspect.getsource(method)
        assert "UPDATE little_bull_" in source, name
        assert "ON CONFLICT (workspace_id, slug)" in source, name
        assert "group_id IS NOT DISTINCT FROM" in source, name
        assert "subgroup_id IS NOT DISTINCT FROM" in source, name
        assert guard_token in source, name


def test_little_bull_conversation_upsert_is_scoped_before_rewriting_messages():
    source = inspect.getsource(admin_store.LittleBullAdminStore.save_conversation)

    assert "scope_snapshot JSONB" in LITTLE_BULL_ADMIN_SCHEMA_SQL
    assert "conversation_scope_mismatch" in source
    assert "scope_snapshot" in source
    assert (
        "little_bull_conversations.tenant_id IS NOT DISTINCT FROM EXCLUDED.tenant_id"
        in source
    )
    assert "little_bull_conversations.workspace_id = EXCLUDED.workspace_id" in source
    assert "little_bull_conversations.user_id = EXCLUDED.user_id" in source
    assert (
        "little_bull_conversations.scope_snapshot = EXCLUDED.scope_snapshot" in source
    )
    assert source.index("if row is None") < source.index(
        "DELETE FROM little_bull_conversation_messages"
    )


def test_phase1_contracts_serialize_minimal_payloads_without_secret_values():
    base = {
        "tenant_id": "default",
        "workspace_id": "default",
        "created_by": "usr_master",
        "updated_by": "usr_master",
    }
    instances = [
        models.ProviderCredential(
            tenant_id="default",
            provider="openrouter",
            label="Primary",
            secret_ref="vault://openrouter/default",
            created_by="usr_master",
            updated_by="usr_master",
        ),
        models.ModelCatalogSnapshot(
            tenant_id="default",
            provider="openrouter",
            source="openrouter:/models/user",
            catalog_hash="sha256:catalog",
            created_by="usr_master",
            updated_by="usr_master",
        ),
        models.KnowledgeGroup(**base, slug="juridico", name="Juridico"),
        models.KnowledgeSubgroup(
            **base, group_id="grp_1", slug="inicial", name="Inicial"
        ),
        models.DocumentRegistry(**base, title="Peticao inicial"),
        models.NoteRegistry(**base, title="Mapa do caso", slug="mapa-do-caso"),
        models.EmbeddingIndexVersion(
            **base,
            provider="openrouter",
            model_id="text-embedding/model",
            embedding_config_hash="sha256:embedding",
        ),
        models.IndexingJob(**base),
        models.LlmUsageLedger(
            **base,
            provider="openrouter",
            model_id="openai/gpt-4o-mini",
            operation="chat",
            request_hash="sha256:req",
            ledger_hash="sha256:ledger",
        ),
        models.GraphEdgeOrigin(
            **base,
            source_node_id="note_1",
            target_node_id="doc_1",
            edge_type="cites",
            origin_type="manual",
        ),
        models.GraphCluster(**base, label="Contratos"),
        models.KnowledgeTrail(**base, title="Leia isto antes", slug="leia-isto-antes"),
        models.KnowledgeTrailStep(
            **base,
            knowledge_trail_id="trail_1",
            step_order=1,
            title="Comece aqui",
        ),
        models.Backlink(
            **base,
            source_kind="note",
            source_id="note_1",
            target_kind="doc",
            target_id="doc_1",
        ),
        models.GraphChatSession(**base),
        models.AgentBuilderSession(**base, user_id="usr_master"),
        models.AgentContextBudget(**base, agent_id="agent_1"),
        models.MarkdownNote(
            **base, note_id="note_1", markdown="# Nota", content_hash="sha256:note"
        ),
        models.WikiLink(**base, source_note_id="note_1", target_label="Mapa"),
        models.TagRegistry(**base, tag="#juridico", label="Juridico"),
        models.ContentMap(**base, title="MOC Juridico", slug="moc-juridico"),
        models.CanvasBoard(**base, title="Canvas do caso", slug="canvas-caso"),
        models.CanvasNode(**base, canvas_board_id="canvas_1", node_kind="note"),
        models.CanvasEdge(
            **base,
            canvas_board_id="canvas_1",
            source_node_id="node_1",
            target_node_id="node_2",
        ),
        models.KnowledgeInboxItem(
            **base, item_kind="quick_note", title="Triar documento"
        ),
        models.DailyNote(**base, note_id="note_daily", note_date="2026-04-30"),
        models.NoteTemplate(**base, title="Ata", slug="ata", markdown_template="# Ata"),
        models.CommandPaletteAction(
            tenant_id="default",
            command_id="notes.new",
            title="Nova nota",
            handler_key="notes.new",
            created_by="usr_master",
            updated_by="usr_master",
        ),
        models.SourceProvenance(**base, source_kind="answer", source_id="msg_1"),
        models.KnowledgeDossier(**base, title="Dossie", slug="dossie"),
        models.LegalMatterExtractionRun(
            **base,
            schema_version="legal-matter/v1",
            source_refs=[{"document_id": "doc_1", "locator": "p.1"}],
        ),
    ]

    dumped = [item.model_dump() for item in instances]
    assert dumped[0]["secret_ref"].startswith("vault://")
    assert "api_key" not in dumped[0]
    assert dumped[-1]["requires_human_review"] is True


def test_phase0_feature_flags_have_safe_defaults(monkeypatch):
    from lightrag_enterprise.system import runtime

    for name in (
        "LITTLE_BULL_GRAPH_V2_ENABLED",
        "LITTLE_BULL_QDRANT_DATA_PLANE_ENABLED",
        "LITTLE_BULL_POSTGRES_CONTROL_PLANE_REQUIRED",
        "LITTLE_BULL_OBSIDIAN_WORKSPACE_ENABLED",
        "LITTLE_BULL_CLEAN_KNOWLEDGE_BASE_ALLOWED",
    ):
        monkeypatch.delenv(name, raising=False)

    assert runtime.little_bull_graph_v2_enabled() is False
    assert runtime.little_bull_qdrant_data_plane_enabled() is False
    assert runtime.little_bull_postgres_control_plane_required() is True
    assert runtime.little_bull_obsidian_workspace_enabled() is False
    assert runtime.little_bull_clean_knowledge_base_allowed() is False


@pytest.mark.asyncio
async def test_little_bull_admin_store_bootstraps_full_system_schema(monkeypatch):
    executed_sql: list[str] = []

    class FakeConnection:
        async def execute(self, sql):
            executed_sql.append(sql)

    class FakeAcquire:
        async def __aenter__(self):
            return FakeConnection()

        async def __aexit__(self, *_exc):
            return None

    class FakePool:
        def acquire(self):
            return FakeAcquire()

    async def create_pool(*_args, **_kwargs):
        return FakePool()

    monkeypatch.setitem(
        sys.modules, "asyncpg", SimpleNamespace(create_pool=create_pool)
    )

    store = admin_store.LittleBullAdminStore("postgresql://app:secret@localhost/db")
    await store._get_pool()

    assert executed_sql
    assert "CREATE TABLE IF NOT EXISTS system_users" in executed_sql[0]
    assert (
        "CREATE TABLE IF NOT EXISTS little_bull_provider_credentials" in executed_sql[0]
    )
