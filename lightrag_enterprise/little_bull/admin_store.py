from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from lightrag_enterprise.system.db import SCHEMA_SQL, get_database_url
from lightrag_enterprise.system.models import new_id, utc_now


class LittleBullAdminStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or get_database_url()
        self._pool: Any = None
        self._schema_ready = False

    async def _get_pool(self) -> Any:
        if not self.database_url:
            raise RuntimeError("Little Bull Admin requires LIGHTRAG_SYSTEM_DATABASE_URL or DATABASE_URL.")
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        if not self._schema_ready:
            async with self._pool.acquire() as conn:
                await conn.execute(SCHEMA_SQL)
            self._schema_ready = True
        return self._pool

    @staticmethod
    def _json(value: Any, fallback: Any) -> Any:
        if value is None:
            return fallback
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return fallback
        return value

    @staticmethod
    def _dt(value: Any) -> str | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _model_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "model_setting_id": row["model_setting_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "usage": row["usage"],
            "provider": row["provider"],
            "binding": row["binding"],
            "binding_host": row["binding_host"],
            "model_id": row["model_id"],
            "display_name": row["display_name"],
            "enabled": row["enabled"],
            "is_default": row["is_default"],
            "config": self._json(row["config"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _knowledge_group_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "group_id": row["group_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "slug": row["slug"],
            "name": row["name"],
            "description": row["description"],
            "privacy": row["privacy"],
            "color": row["color"],
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _knowledge_subgroup_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "subgroup_id": row["subgroup_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "group_id": row["group_id"],
            "slug": row["slug"],
            "name": row["name"],
            "description": row["description"],
            "privacy": row["privacy"],
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _document_registry_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "document_id": row["document_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "group_id": row["group_id"],
            "subgroup_id": row["subgroup_id"],
            "embedding_version_id": row["embedding_version_id"],
            "title": row["title"],
            "source_uri": row["source_uri"],
            "source_kind": row["source_kind"],
            "mime_type": row["mime_type"],
            "content_hash": row["content_hash"],
            "confidentiality": row["confidentiality"],
            "status": row["status"],
            "chunk_count": row["chunk_count"],
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _note_registry_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "note_id": row["note_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "group_id": row["group_id"],
            "subgroup_id": row["subgroup_id"],
            "title": row["title"],
            "slug": row["slug"],
            "note_type": row["note_type"],
            "privacy": row["privacy"],
            "status": row["status"],
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _markdown_note_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "markdown_note_id": row["markdown_note_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "note_id": row["note_id"],
            "version_number": row["version_number"],
            "markdown": row["markdown"],
            "rendered_summary": row["rendered_summary"],
            "content_hash": row["content_hash"],
            "status": row["status"],
            "source_document_id": row["source_document_id"],
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _wiki_link_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "wiki_link_id": row["wiki_link_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "source_note_id": row["source_note_id"],
            "target_note_id": row["target_note_id"],
            "target_label": row["target_label"],
            "link_text": row["link_text"],
            "link_status": row["link_status"],
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _tag_registry_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "tag_id": row["tag_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "tag": row["tag"],
            "label": row["label"],
            "description": row["description"],
            "color": row["color"],
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _backlink_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "backlink_id": row["backlink_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "source_kind": row["source_kind"],
            "source_id": row["source_id"],
            "target_kind": row["target_kind"],
            "target_id": row["target_id"],
            "link_text": row["link_text"],
            "origin_type": row["origin_type"],
            "graph_edge_origin_id": row["graph_edge_origin_id"],
            "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _source_provenance_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "source_provenance_id": row["source_provenance_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "source_kind": row["source_kind"],
            "source_id": row["source_id"],
            "document_id": row["document_id"],
            "note_id": row["note_id"],
            "chunk_id": row["chunk_id"],
            "model_id": row["model_id"],
            "agent_id": row["agent_id"],
            "usage_ledger_id": row["usage_ledger_id"],
            "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
            "locator": self._json(row["locator"], {}),
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _canvas_board_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "canvas_board_id": row["canvas_board_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "group_id": row["group_id"],
            "subgroup_id": row["subgroup_id"],
            "title": row["title"],
            "slug": row["slug"],
            "layout": self._json(row["layout"], {}),
            "status": row["status"],
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _canvas_node_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "canvas_node_id": row["canvas_node_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "canvas_board_id": row["canvas_board_id"],
            "node_kind": row["node_kind"],
            "ref_kind": row["ref_kind"],
            "ref_id": row["ref_id"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "width": float(row["width"]),
            "height": float(row["height"]),
            "content": self._json(row["content"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _canvas_edge_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "canvas_edge_id": row["canvas_edge_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "canvas_board_id": row["canvas_board_id"],
            "source_node_id": row["source_node_id"],
            "target_node_id": row["target_node_id"],
            "edge_kind": row["edge_kind"],
            "label": row["label"],
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _knowledge_dossier_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "knowledge_dossier_id": row["knowledge_dossier_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "group_id": row["group_id"],
            "subgroup_id": row["subgroup_id"],
            "title": row["title"],
            "slug": row["slug"],
            "dossier_kind": row["dossier_kind"],
            "status": row["status"],
            "content_refs": self._json(row["content_refs"], []),
            "export_policy": self._json(row["export_policy"], {}),
            "approval_id": row["approval_id"],
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _content_map_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "content_map_id": row["content_map_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "group_id": row["group_id"],
            "subgroup_id": row["subgroup_id"],
            "title": row["title"],
            "slug": row["slug"],
            "root_note_id": row["root_note_id"],
            "description": row["description"],
            "map_body": self._json(row["map_body"], {}),
            "status": row["status"],
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _knowledge_trail_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "knowledge_trail_id": row["knowledge_trail_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "group_id": row["group_id"],
            "subgroup_id": row["subgroup_id"],
            "title": row["title"],
            "slug": row["slug"],
            "trail_type": row["trail_type"],
            "description": row["description"],
            "status": row["status"],
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _knowledge_trail_step_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "knowledge_trail_step_id": row["knowledge_trail_step_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "knowledge_trail_id": row["knowledge_trail_id"],
            "step_order": row["step_order"],
            "title": row["title"],
            "step_kind": row["step_kind"],
            "note_id": row["note_id"],
            "document_id": row["document_id"],
            "canvas_board_id": row["canvas_board_id"],
            "instructions": row["instructions"],
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _knowledge_inbox_item_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "inbox_item_id": row["inbox_item_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "group_id": row["group_id"],
            "subgroup_id": row["subgroup_id"],
            "item_kind": row["item_kind"],
            "title": row["title"],
            "body": row["body"],
            "source_kind": row["source_kind"],
            "source_id": row["source_id"],
            "status": row["status"],
            "priority": row["priority"],
            "metadata": self._json(row["metadata"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _daily_note_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "daily_note_id": row["daily_note_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "note_id": row["note_id"],
            "note_date": str(row["note_date"]),
            "summary": row["summary"],
            "decisions": self._json(row["decisions"], []),
            "pending_items": self._json(row["pending_items"], []),
            "cost_snapshot": self._json(row["cost_snapshot"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _agent_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "agent_id": row["agent_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "name": row["name"],
            "description": row["description"],
            "enabled": row["enabled"],
            "model_setting_id": row["model_setting_id"],
            "system_prompt": row["system_prompt"],
            "response_rules": list(row["response_rules"] or []),
            "tools": list(row["tools"] or []),
            "config": self._json(row["config"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _agent_builder_session_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "agent_builder_session_id": row["agent_builder_session_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "user_id": row["user_id"],
            "agent_id": row["agent_id"],
            "model_setting_id": row["model_setting_id"],
            "status": row["status"],
            "current_step": row["current_step"],
            "builder_transcript": self._json(row["builder_transcript"], []),
            "generated_config": self._json(row["generated_config"], {}),
            "readiness_score": row["readiness_score"],
            "requires_review": row["requires_review"],
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _agent_context_budget_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "agent_context_budget_id": row["agent_context_budget_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "agent_id": row["agent_id"],
            "model_setting_id": row["model_setting_id"],
            "max_context_tokens": row["max_context_tokens"],
            "reserved_response_tokens": row["reserved_response_tokens"],
            "max_prompt_tokens": row["max_prompt_tokens"],
            "daily_cost_limit_usd": (
                float(row["daily_cost_limit_usd"]) if row["daily_cost_limit_usd"] is not None else None
            ),
            "monthly_cost_limit_usd": (
                float(row["monthly_cost_limit_usd"]) if row["monthly_cost_limit_usd"] is not None else None
            ),
            "policy": self._json(row["policy"], {}),
            "created_by": row["created_by"],
            "updated_by": row["updated_by"],
            "created_at": self._dt(row["created_at"]),
            "updated_at": self._dt(row["updated_at"]),
        }

    def _message_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "message_id": row["message_id"],
            "role": row["role"],
            "content": row["content"],
            "references": self._json(row["message_references"], []),
            "metadata": self._json(row["metadata"], {}),
            "created_at": self._dt(row["created_at"]),
        }

    def _conversation_from_row(self, row: Any, messages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        row_data = dict(row)
        data = {
            "conversation_id": row_data["conversation_id"],
            "tenant_id": row_data["tenant_id"],
            "workspace_id": row_data["workspace_id"],
            "user_id": row_data["user_id"],
            "title": row_data["title"],
            "agent_id": row_data["agent_id"],
            "model_profile": row_data["model_profile"],
            "confidentiality": row_data["confidentiality"],
            "message_count": int(row_data.get("message_count") or 0),
            "created_at": self._dt(row_data["created_at"]),
            "updated_at": self._dt(row_data["updated_at"]),
        }
        if messages is not None:
            data["messages"] = messages
        return data

    def _suggestion_from_row(self, row: Any) -> dict[str, Any]:
        return {
            "suggestion_id": row["suggestion_id"],
            "tenant_id": row["tenant_id"],
            "workspace_id": row["workspace_id"],
            "user_id": row["user_id"],
            "source_label": row["source_label"],
            "target_label": row["target_label"],
            "reason": row["reason"],
            "status": row["status"],
            "metadata": self._json(row["metadata"], {}),
            "created_at": self._dt(row["created_at"]),
            "decided_at": self._dt(row["decided_at"]),
            "decided_by": row["decided_by"],
        }

    async def list_model_settings(self, *, tenant_id: str | None, workspace_id: str | None) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_model_settings
            WHERE (tenant_id IS NULL OR tenant_id IS NOT DISTINCT FROM $1)
              AND (workspace_id IS NULL OR workspace_id IS NOT DISTINCT FROM $2)
            ORDER BY usage, is_default DESC, display_name
            """,
            tenant_id,
            workspace_id,
        )
        return [self._model_from_row(row) for row in rows]

    async def upsert_model_setting(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str | None,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        model_setting_id = str(payload.get("model_setting_id") or new_id("lbm"))
        usage = str(payload.get("usage") or "chat").strip().lower()
        is_default = bool(payload.get("is_default", False))
        async with pool.acquire() as conn:
            async with conn.transaction():
                if is_default:
                    await conn.execute(
                        """
                        UPDATE little_bull_model_settings
                        SET is_default=FALSE, updated_at=$4, updated_by=$5
                        WHERE usage=$1
                          AND tenant_id IS NOT DISTINCT FROM $2
                          AND workspace_id IS NOT DISTINCT FROM $3
                        """,
                        usage,
                        tenant_id,
                        workspace_id,
                        utc_now(),
                        user_id,
                    )
                row = await conn.fetchrow(
                    """
                    INSERT INTO little_bull_model_settings (
                        model_setting_id, tenant_id, workspace_id, usage, provider, binding,
                        binding_host, model_id, display_name, enabled, is_default, config,
                        created_by, updated_by, created_at, updated_at
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13,$13,$14,$14)
                    ON CONFLICT (model_setting_id) DO UPDATE SET
                        usage=EXCLUDED.usage,
                        provider=EXCLUDED.provider,
                        binding=EXCLUDED.binding,
                        binding_host=EXCLUDED.binding_host,
                        model_id=EXCLUDED.model_id,
                        display_name=EXCLUDED.display_name,
                        enabled=EXCLUDED.enabled,
                        is_default=EXCLUDED.is_default,
                        config=EXCLUDED.config,
                        updated_by=EXCLUDED.updated_by,
                        updated_at=EXCLUDED.updated_at
                    WHERE little_bull_model_settings.tenant_id IS NOT DISTINCT FROM EXCLUDED.tenant_id
                      AND little_bull_model_settings.workspace_id IS NOT DISTINCT FROM EXCLUDED.workspace_id
                    RETURNING *
                    """,
                    model_setting_id,
                    tenant_id,
                    workspace_id,
                    usage,
                    str(payload.get("provider") or "openrouter").strip().lower(),
                    str(payload.get("binding") or "openai").strip().lower(),
                    str(payload.get("binding_host") or "").strip(),
                    str(payload.get("model_id") or "").strip(),
                    str(payload.get("display_name") or payload.get("model_id") or "Modelo").strip(),
                    bool(payload.get("enabled", True)),
                    is_default,
                    json.dumps(payload.get("config") or {}),
                    user_id,
                    utc_now(),
                )
                if row is None:
                    raise ValueError("model_setting_scope_mismatch")
        return self._model_from_row(row)

    async def list_knowledge_groups(self, *, tenant_id: str | None, workspace_id: str) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_knowledge_groups
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
            ORDER BY name
            """,
            tenant_id,
            workspace_id,
        )
        return [self._knowledge_group_from_row(row) for row in rows]

    async def get_knowledge_group(
        self, group_id: str, *, tenant_id: str | None, workspace_id: str
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_knowledge_groups
            WHERE group_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            """,
            group_id,
            tenant_id,
            workspace_id,
        )
        return self._knowledge_group_from_row(row) if row else None

    async def upsert_knowledge_group(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        group_id = str(payload.get("group_id") or new_id("lbg"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_knowledge_groups (
                group_id, tenant_id, workspace_id, slug, name, description, privacy,
                color, metadata, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$10,$11,$11)
            ON CONFLICT (workspace_id, slug) DO UPDATE SET
                slug=EXCLUDED.slug,
                name=EXCLUDED.name,
                description=EXCLUDED.description,
                privacy=EXCLUDED.privacy,
                color=EXCLUDED.color,
                metadata=EXCLUDED.metadata,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            group_id,
            tenant_id,
            workspace_id,
            str(payload.get("slug") or "").strip(),
            str(payload.get("name") or "").strip(),
            str(payload.get("description") or "").strip(),
            str(payload.get("privacy") or "team").strip(),
            str(payload.get("color") or "#2563EB").strip(),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            utc_now(),
        )
        return self._knowledge_group_from_row(row)

    async def list_knowledge_subgroups(
        self, *, tenant_id: str | None, workspace_id: str, group_id: str | None = None
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_knowledge_subgroups
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR group_id=$3)
            ORDER BY name
            """,
            tenant_id,
            workspace_id,
            group_id,
        )
        return [self._knowledge_subgroup_from_row(row) for row in rows]

    async def get_knowledge_subgroup(
        self,
        subgroup_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
        group_id: str | None = None,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_knowledge_subgroups
            WHERE subgroup_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
              AND ($4::text IS NULL OR group_id=$4)
            """,
            subgroup_id,
            tenant_id,
            workspace_id,
            group_id,
        )
        return self._knowledge_subgroup_from_row(row) if row else None

    async def upsert_knowledge_subgroup(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        subgroup_id = str(payload.get("subgroup_id") or new_id("lbsg"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_knowledge_subgroups (
                subgroup_id, tenant_id, workspace_id, group_id, slug, name,
                description, privacy, metadata, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$10,$11,$11)
            ON CONFLICT (group_id, slug) DO UPDATE SET
                group_id=EXCLUDED.group_id,
                slug=EXCLUDED.slug,
                name=EXCLUDED.name,
                description=EXCLUDED.description,
                privacy=EXCLUDED.privacy,
                metadata=EXCLUDED.metadata,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            subgroup_id,
            tenant_id,
            workspace_id,
            str(payload.get("group_id") or "").strip(),
            str(payload.get("slug") or "").strip(),
            str(payload.get("name") or "").strip(),
            str(payload.get("description") or "").strip(),
            str(payload.get("privacy") or "team").strip(),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            utc_now(),
        )
        return self._knowledge_subgroup_from_row(row)

    async def register_document(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        source_kind = str(payload.get("source_kind") or "upload").strip()
        if source_kind == "upload" and (not payload.get("group_id") or not payload.get("subgroup_id")):
            raise ValueError("Upload documents require group_id and subgroup_id.")
        pool = await self._get_pool()
        document_id = str(payload.get("document_id") or new_id("lbdoc"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_document_registry (
                document_id, tenant_id, workspace_id, group_id, subgroup_id,
                embedding_version_id, title, source_uri, source_kind, mime_type,
                content_hash, confidentiality, status, chunk_count, metadata,
                created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15::jsonb,$16,$16,$17,$17)
            ON CONFLICT (document_id) DO UPDATE SET
                group_id=EXCLUDED.group_id,
                subgroup_id=EXCLUDED.subgroup_id,
                title=EXCLUDED.title,
                source_uri=EXCLUDED.source_uri,
                source_kind=EXCLUDED.source_kind,
                mime_type=EXCLUDED.mime_type,
                content_hash=EXCLUDED.content_hash,
                confidentiality=EXCLUDED.confidentiality,
                status=EXCLUDED.status,
                chunk_count=EXCLUDED.chunk_count,
                metadata=EXCLUDED.metadata,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            document_id,
            tenant_id,
            workspace_id,
            payload.get("group_id") or None,
            payload.get("subgroup_id") or None,
            payload.get("embedding_version_id") or None,
            str(payload.get("title") or "").strip(),
            str(payload.get("source_uri") or "").strip(),
            source_kind,
            str(payload.get("mime_type") or "").strip(),
            str(payload.get("content_hash") or "").strip(),
            str(payload.get("confidentiality") or "normal").strip(),
            str(payload.get("status") or "registered").strip(),
            int(payload.get("chunk_count") or 0),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            utc_now(),
        )
        return self._document_registry_from_row(row)

    async def list_document_registry(
        self, *, tenant_id: str | None, workspace_id: str
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_document_registry
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
            ORDER BY updated_at DESC
            """,
            tenant_id,
            workspace_id,
        )
        return [self._document_registry_from_row(row) for row in rows]

    async def get_document_registry(
        self,
        document_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_document_registry
            WHERE document_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            """,
            document_id,
            tenant_id,
            workspace_id,
        )
        return self._document_registry_from_row(row) if row else None

    async def list_note_registry(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_note_registry
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR group_id=$3)
              AND ($4::text IS NULL OR subgroup_id=$4)
            ORDER BY updated_at DESC
            """,
            tenant_id,
            workspace_id,
            group_id,
            subgroup_id,
        )
        return [self._note_registry_from_row(row) for row in rows]

    async def get_note_registry(
        self,
        note_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_note_registry
            WHERE note_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            """,
            note_id,
            tenant_id,
            workspace_id,
        )
        return self._note_registry_from_row(row) if row else None

    async def find_note_by_slug_or_title(
        self,
        *,
        slug: str,
        title: str,
        tenant_id: str | None,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_note_registry
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND (slug=$3 OR lower(title)=lower($4))
            ORDER BY CASE WHEN slug=$3 THEN 0 ELSE 1 END, updated_at DESC
            LIMIT 1
            """,
            tenant_id,
            workspace_id,
            slug,
            title,
        )
        return self._note_registry_from_row(row) if row else None

    async def upsert_note_registry(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        now = utc_now()
        note_id = str(payload.get("note_id") or new_id("lbnote"))
        if payload.get("note_id"):
            row = await pool.fetchrow(
                """
                UPDATE little_bull_note_registry
                SET group_id=$4,
                    subgroup_id=$5,
                    title=$6,
                    slug=$7,
                    note_type=$8,
                    privacy=$9,
                    status=$10,
                    metadata=$11::jsonb,
                    updated_by=$12,
                    updated_at=$13
                WHERE note_id=$1
                  AND tenant_id IS NOT DISTINCT FROM $2
                  AND workspace_id=$3
                RETURNING *
                """,
                note_id,
                tenant_id,
                workspace_id,
                payload.get("group_id") or None,
                payload.get("subgroup_id") or None,
                str(payload.get("title") or "").strip(),
                str(payload.get("slug") or "").strip(),
                str(payload.get("note_type") or "markdown").strip(),
                str(payload.get("privacy") or "team").strip(),
                str(payload.get("status") or "active").strip(),
                json.dumps(payload.get("metadata") or {}),
                user_id,
                now,
            )
            if row:
                return self._note_registry_from_row(row)

        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_note_registry (
                note_id, tenant_id, workspace_id, group_id, subgroup_id, title,
                slug, note_type, privacy, status, metadata, created_by, updated_by,
                created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12,$12,$13,$13)
            ON CONFLICT (workspace_id, slug) DO UPDATE SET
                group_id=EXCLUDED.group_id,
                subgroup_id=EXCLUDED.subgroup_id,
                title=EXCLUDED.title,
                note_type=EXCLUDED.note_type,
                privacy=EXCLUDED.privacy,
                status=EXCLUDED.status,
                metadata=EXCLUDED.metadata,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            note_id,
            tenant_id,
            workspace_id,
            payload.get("group_id") or None,
            payload.get("subgroup_id") or None,
            str(payload.get("title") or "").strip(),
            str(payload.get("slug") or "").strip(),
            str(payload.get("note_type") or "markdown").strip(),
            str(payload.get("privacy") or "team").strip(),
            str(payload.get("status") or "active").strip(),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            now,
        )
        return self._note_registry_from_row(row)

    async def insert_markdown_note_version(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        markdown_note_id = str(payload.get("markdown_note_id") or new_id("lbmd"))
        note_id = str(payload.get("note_id") or "").strip()
        now = utc_now()
        async with pool.acquire() as conn, conn.transaction():
            version_number = await conn.fetchval(
                """
                SELECT COALESCE(MAX(version_number), 0) + 1
                FROM little_bull_markdown_notes
                WHERE note_id=$1
                  AND tenant_id IS NOT DISTINCT FROM $2
                  AND workspace_id=$3
                """,
                note_id,
                tenant_id,
                workspace_id,
            )
            await conn.execute(
                """
                UPDATE little_bull_markdown_notes
                SET status='superseded',
                    updated_by=$4,
                    updated_at=$5
                WHERE note_id=$1
                  AND tenant_id IS NOT DISTINCT FROM $2
                  AND workspace_id=$3
                  AND status='current'
                """,
                note_id,
                tenant_id,
                workspace_id,
                user_id,
                now,
            )
            row = await conn.fetchrow(
                """
                INSERT INTO little_bull_markdown_notes (
                    markdown_note_id, tenant_id, workspace_id, note_id, version_number,
                    markdown, rendered_summary, content_hash, status, source_document_id,
                    created_by, updated_by, created_at, updated_at
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$11,$12,$12)
                RETURNING *
                """,
                markdown_note_id,
                tenant_id,
                workspace_id,
                note_id,
                int(version_number or 1),
                str(payload.get("markdown") or ""),
                str(payload.get("rendered_summary") or ""),
                str(payload.get("content_hash") or ""),
                str(payload.get("status") or "current"),
                payload.get("source_document_id") or None,
                user_id,
                now,
            )
        return self._markdown_note_from_row(row)

    async def get_latest_markdown_note(
        self,
        note_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_markdown_notes
            WHERE note_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            ORDER BY version_number DESC
            LIMIT 1
            """,
            note_id,
            tenant_id,
            workspace_id,
        )
        return self._markdown_note_from_row(row) if row else None

    async def replace_wiki_links(
        self,
        *,
        source_note_id: str,
        links: list[dict[str, Any]],
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        now = utc_now()
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                """
                DELETE FROM little_bull_wiki_links
                WHERE source_note_id=$1
                  AND tenant_id IS NOT DISTINCT FROM $2
                  AND workspace_id=$3
                """,
                source_note_id,
                tenant_id,
                workspace_id,
            )
            rows = []
            for link in links:
                row = await conn.fetchrow(
                    """
                    INSERT INTO little_bull_wiki_links (
                        wiki_link_id, tenant_id, workspace_id, source_note_id,
                        target_note_id, target_label, link_text, link_status, metadata,
                        created_by, updated_by, created_at, updated_at
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$10,$11,$11)
                    RETURNING *
                    """,
                    str(link.get("wiki_link_id") or new_id("lbwl")),
                    tenant_id,
                    workspace_id,
                    source_note_id,
                    link.get("target_note_id") or None,
                    str(link.get("target_label") or "").strip(),
                    str(link.get("link_text") or "").strip(),
                    str(link.get("link_status") or "unresolved").strip(),
                    json.dumps(link.get("metadata") or {}),
                    user_id,
                    now,
                )
                rows.append(self._wiki_link_from_row(row))
        return rows

    async def list_wiki_links(
        self,
        *,
        source_note_id: str,
        tenant_id: str | None,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_wiki_links
            WHERE source_note_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            ORDER BY created_at, target_label
            """,
            source_note_id,
            tenant_id,
            workspace_id,
        )
        return [self._wiki_link_from_row(row) for row in rows]

    async def upsert_tag_registry(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        tag_id = str(payload.get("tag_id") or new_id("lbtag"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_tag_registry (
                tag_id, tenant_id, workspace_id, tag, label, description, color,
                metadata, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,$9,$10,$10)
            ON CONFLICT (workspace_id, tag) DO UPDATE SET
                label=EXCLUDED.label,
                description=EXCLUDED.description,
                color=EXCLUDED.color,
                metadata=EXCLUDED.metadata,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            tag_id,
            tenant_id,
            workspace_id,
            str(payload.get("tag") or "").strip(),
            str(payload.get("label") or "").strip(),
            str(payload.get("description") or "").strip(),
            str(payload.get("color") or "#64748B").strip(),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            utc_now(),
        )
        return self._tag_registry_from_row(row)

    async def list_tag_registry(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_tag_registry
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
            ORDER BY tag
            """,
            tenant_id,
            workspace_id,
        )
        return [self._tag_registry_from_row(row) for row in rows]

    async def upsert_backlink(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_backlinks (
                backlink_id, tenant_id, workspace_id, source_kind, source_id,
                target_kind, target_id, link_text, origin_type, graph_edge_origin_id,
                confidence, metadata, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13,$13,$14,$14)
            ON CONFLICT (workspace_id, source_kind, source_id, target_kind, target_id, origin_type)
            DO UPDATE SET
                link_text=EXCLUDED.link_text,
                graph_edge_origin_id=EXCLUDED.graph_edge_origin_id,
                confidence=EXCLUDED.confidence,
                metadata=EXCLUDED.metadata,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            str(payload.get("backlink_id") or new_id("lbbl")),
            tenant_id,
            workspace_id,
            str(payload.get("source_kind") or "").strip(),
            str(payload.get("source_id") or "").strip(),
            str(payload.get("target_kind") or "").strip(),
            str(payload.get("target_id") or "").strip(),
            str(payload.get("link_text") or "").strip(),
            str(payload.get("origin_type") or "manual").strip(),
            payload.get("graph_edge_origin_id") or None,
            payload.get("confidence"),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            utc_now(),
        )
        return self._backlink_from_row(row)

    async def replace_backlinks_for_source(
        self,
        *,
        source_kind: str,
        source_id: str,
        origin_type: str,
        backlinks: list[dict[str, Any]],
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    DELETE FROM little_bull_backlinks
                    WHERE tenant_id IS NOT DISTINCT FROM $1
                      AND workspace_id=$2
                      AND source_kind=$3
                      AND source_id=$4
                      AND origin_type=$5
                    """,
                    tenant_id,
                    workspace_id,
                    source_kind,
                    source_id,
                    origin_type,
                )
                rows = []
                now = utc_now()
                for backlink in backlinks:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO little_bull_backlinks (
                            backlink_id, tenant_id, workspace_id, source_kind, source_id,
                            target_kind, target_id, link_text, origin_type, graph_edge_origin_id,
                            confidence, metadata, created_by, updated_by, created_at, updated_at
                        )
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13,$13,$14,$14)
                        RETURNING *
                        """,
                        str(backlink.get("backlink_id") or new_id("lbbl")),
                        tenant_id,
                        workspace_id,
                        str(backlink.get("source_kind") or source_kind).strip(),
                        str(backlink.get("source_id") or source_id).strip(),
                        str(backlink.get("target_kind") or "").strip(),
                        str(backlink.get("target_id") or "").strip(),
                        str(backlink.get("link_text") or "").strip(),
                        str(backlink.get("origin_type") or origin_type).strip(),
                        backlink.get("graph_edge_origin_id") or None,
                        backlink.get("confidence"),
                        json.dumps(backlink.get("metadata") or {}),
                        user_id,
                        now,
                    )
                    rows.append(self._backlink_from_row(row))
        return rows

    async def list_backlinks(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        source_kind: str | None = None,
        source_id: str | None = None,
        target_kind: str | None = None,
        target_id: str | None = None,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_backlinks
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR source_kind=$3)
              AND ($4::text IS NULL OR source_id=$4)
              AND ($5::text IS NULL OR target_kind=$5)
              AND ($6::text IS NULL OR target_id=$6)
            ORDER BY updated_at DESC, target_kind, target_id
            """,
            tenant_id,
            workspace_id,
            source_kind,
            source_id,
            target_kind,
            target_id,
        )
        return [self._backlink_from_row(row) for row in rows]

    async def insert_source_provenance(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_source_provenance (
                source_provenance_id, tenant_id, workspace_id, source_kind, source_id,
                document_id, note_id, chunk_id, model_id, agent_id, usage_ledger_id,
                confidence, locator, metadata, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb,$14::jsonb,$15,$15,$16,$16)
            RETURNING *
            """,
            str(payload.get("source_provenance_id") or new_id("lbsp")),
            tenant_id,
            workspace_id,
            str(payload.get("source_kind") or "").strip(),
            str(payload.get("source_id") or "").strip(),
            payload.get("document_id") or None,
            payload.get("note_id") or None,
            str(payload.get("chunk_id") or "").strip(),
            str(payload.get("model_id") or "").strip(),
            payload.get("agent_id") or None,
            payload.get("usage_ledger_id") or None,
            payload.get("confidence"),
            json.dumps(payload.get("locator") or {}),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            utc_now(),
        )
        return self._source_provenance_from_row(row)

    async def sum_llm_usage_cost(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        agent_id: str | None = None,
        since: datetime | None = None,
    ) -> float:
        pool = await self._get_pool()
        cost = await pool.fetchval(
            """
            SELECT COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0)
            FROM little_bull_llm_usage_ledger
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR agent_id=$3)
              AND ($4::timestamptz IS NULL OR created_at >= $4)
            """,
            tenant_id,
            workspace_id,
            agent_id,
            since,
        )
        return float(cost or 0)

    async def insert_llm_usage_ledger(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        created_at = utc_now()
        async with pool.acquire() as conn:
            previous_hash = await conn.fetchval(
                """
                SELECT ledger_hash
                FROM little_bull_llm_usage_ledger
                WHERE tenant_id IS NOT DISTINCT FROM $1
                  AND workspace_id=$2
                ORDER BY created_at DESC
                LIMIT 1
                """,
                tenant_id,
                workspace_id,
            )
            usage_ledger_id = str(payload.get("usage_ledger_id") or new_id("lblu"))
            ledger_material = {
                "usage_ledger_id": usage_ledger_id,
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "user_id": payload.get("user_id") or user_id,
                "agent_id": payload.get("agent_id"),
                "model_setting_id": payload.get("model_setting_id"),
                "provider": str(payload.get("provider") or "").strip(),
                "model_id": str(payload.get("model_id") or "").strip(),
                "operation": str(payload.get("operation") or "").strip(),
                "prompt_tokens": int(payload.get("prompt_tokens") or 0),
                "completion_tokens": int(payload.get("completion_tokens") or 0),
                "total_tokens": int(payload.get("total_tokens") or 0),
                "estimated_cost_usd": float(payload.get("estimated_cost_usd") or 0),
                "actual_cost_usd": payload.get("actual_cost_usd"),
                "currency": str(payload.get("currency") or "USD").strip(),
                "request_hash": str(payload.get("request_hash") or "").strip(),
                "response_hash": str(payload.get("response_hash") or "").strip(),
                "metadata": payload.get("metadata") or {},
                "previous_ledger_hash": previous_hash or "",
                "created_at": created_at.isoformat(),
            }
            ledger_hash = "sha256:" + hashlib.sha256(
                json.dumps(ledger_material, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            row = await conn.fetchrow(
                """
                INSERT INTO little_bull_llm_usage_ledger (
                    usage_ledger_id, tenant_id, workspace_id, user_id, agent_id,
                    conversation_id, model_setting_id, provider, model_id, operation,
                    prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd,
                    actual_cost_usd, currency, request_hash, response_hash, metadata,
                    previous_ledger_hash, ledger_hash, created_by, updated_by, created_at, updated_at
                )
                VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19::jsonb,
                    $20,$21,$22,$22,$23,$23
                )
                RETURNING *
                """,
                usage_ledger_id,
                tenant_id,
                workspace_id,
                payload.get("user_id") or user_id,
                payload.get("agent_id") or None,
                payload.get("conversation_id") or None,
                payload.get("model_setting_id") or None,
                ledger_material["provider"],
                ledger_material["model_id"],
                ledger_material["operation"],
                ledger_material["prompt_tokens"],
                ledger_material["completion_tokens"],
                ledger_material["total_tokens"],
                ledger_material["estimated_cost_usd"],
                payload.get("actual_cost_usd"),
                ledger_material["currency"],
                ledger_material["request_hash"],
                ledger_material["response_hash"],
                json.dumps(ledger_material["metadata"]),
                previous_hash or "",
                ledger_hash,
                user_id,
                created_at,
            )
        return {
            "usage_ledger_id": row["usage_ledger_id"],
            "ledger_hash": row["ledger_hash"],
            "previous_ledger_hash": row["previous_ledger_hash"],
            "estimated_cost_usd": float(row["estimated_cost_usd"] or 0),
        }

    async def reserve_llm_usage_budget(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
        agent_id: str | None,
        daily_limit_usd: float | None,
        monthly_limit_usd: float | None,
        daily_since: datetime,
        monthly_since: datetime,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        estimate = float(payload.get("estimated_cost_usd") or 0)
        created_at = utc_now()
        lock_key = f"{tenant_id or ''}:{workspace_id}:{agent_id or ''}:llm_budget"
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", lock_key)
                daily_used = await conn.fetchval(
                    """
                    SELECT COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0)
                    FROM little_bull_llm_usage_ledger
                    WHERE tenant_id IS NOT DISTINCT FROM $1
                      AND workspace_id=$2
                      AND ($3::text IS NULL OR agent_id=$3)
                      AND created_at >= $4
                    """,
                    tenant_id,
                    workspace_id,
                    agent_id,
                    daily_since,
                )
                if daily_limit_usd is not None and float(daily_used or 0) + estimate > daily_limit_usd:
                    raise ValueError("daily_cost_budget_exceeded")
                monthly_used = await conn.fetchval(
                    """
                    SELECT COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd)), 0)
                    FROM little_bull_llm_usage_ledger
                    WHERE tenant_id IS NOT DISTINCT FROM $1
                      AND workspace_id=$2
                      AND ($3::text IS NULL OR agent_id=$3)
                      AND created_at >= $4
                    """,
                    tenant_id,
                    workspace_id,
                    agent_id,
                    monthly_since,
                )
                if monthly_limit_usd is not None and float(monthly_used or 0) + estimate > monthly_limit_usd:
                    raise ValueError("monthly_cost_budget_exceeded")
                previous_hash = await conn.fetchval(
                    """
                    SELECT ledger_hash
                    FROM little_bull_llm_usage_ledger
                    WHERE tenant_id IS NOT DISTINCT FROM $1
                      AND workspace_id=$2
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    tenant_id,
                    workspace_id,
                )
                usage_ledger_id = str(payload.get("usage_ledger_id") or new_id("lblu"))
                metadata = payload.get("metadata") or {}
                ledger_material = {
                    "usage_ledger_id": usage_ledger_id,
                    "tenant_id": tenant_id,
                    "workspace_id": workspace_id,
                    "user_id": payload.get("user_id") or user_id,
                    "agent_id": payload.get("agent_id"),
                    "model_setting_id": payload.get("model_setting_id"),
                    "provider": str(payload.get("provider") or "").strip(),
                    "model_id": str(payload.get("model_id") or "").strip(),
                    "operation": str(payload.get("operation") or "").strip(),
                    "prompt_tokens": int(payload.get("prompt_tokens") or 0),
                    "completion_tokens": int(payload.get("completion_tokens") or 0),
                    "total_tokens": int(payload.get("total_tokens") or 0),
                    "estimated_cost_usd": estimate,
                    "actual_cost_usd": payload.get("actual_cost_usd"),
                    "currency": str(payload.get("currency") or "USD").strip(),
                    "request_hash": str(payload.get("request_hash") or "").strip(),
                    "response_hash": str(payload.get("response_hash") or "").strip(),
                    "metadata": metadata,
                    "previous_ledger_hash": previous_hash or "",
                    "created_at": created_at.isoformat(),
                }
                ledger_hash = "sha256:" + hashlib.sha256(
                    json.dumps(ledger_material, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
                row = await conn.fetchrow(
                    """
                    INSERT INTO little_bull_llm_usage_ledger (
                        usage_ledger_id, tenant_id, workspace_id, user_id, agent_id,
                        conversation_id, model_setting_id, provider, model_id, operation,
                        prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd,
                        actual_cost_usd, currency, request_hash, response_hash, metadata,
                        previous_ledger_hash, ledger_hash, created_by, updated_by, created_at, updated_at
                    )
                    VALUES (
                        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19::jsonb,
                        $20,$21,$22,$22,$23,$23
                    )
                    RETURNING *
                    """,
                    usage_ledger_id,
                    tenant_id,
                    workspace_id,
                    payload.get("user_id") or user_id,
                    payload.get("agent_id") or None,
                    payload.get("conversation_id") or None,
                    payload.get("model_setting_id") or None,
                    ledger_material["provider"],
                    ledger_material["model_id"],
                    ledger_material["operation"],
                    ledger_material["prompt_tokens"],
                    ledger_material["completion_tokens"],
                    ledger_material["total_tokens"],
                    ledger_material["estimated_cost_usd"],
                    payload.get("actual_cost_usd"),
                    ledger_material["currency"],
                    ledger_material["request_hash"],
                    ledger_material["response_hash"],
                    json.dumps(metadata),
                    previous_hash or "",
                    ledger_hash,
                    user_id,
                    created_at,
                )
        return {
            "usage_ledger_id": row["usage_ledger_id"],
            "ledger_hash": row["ledger_hash"],
            "previous_ledger_hash": row["previous_ledger_hash"],
            "estimated_cost_usd": float(row["estimated_cost_usd"] or 0),
        }

    async def list_source_provenance(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        source_kind: str | None = None,
        source_id: str | None = None,
        document_id: str | None = None,
        note_id: str | None = None,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_source_provenance
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR source_kind=$3)
              AND ($4::text IS NULL OR source_id=$4)
              AND ($5::text IS NULL OR document_id=$5)
              AND ($6::text IS NULL OR note_id=$6)
            ORDER BY created_at DESC
            """,
            tenant_id,
            workspace_id,
            source_kind,
            source_id,
            document_id,
            note_id,
        )
        return [self._source_provenance_from_row(row) for row in rows]

    async def list_canvas_boards(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_canvas_boards
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR group_id=$3)
              AND ($4::text IS NULL OR subgroup_id=$4)
            ORDER BY updated_at DESC
            """,
            tenant_id,
            workspace_id,
            group_id,
            subgroup_id,
        )
        return [self._canvas_board_from_row(row) for row in rows]

    async def get_canvas_board(
        self,
        canvas_board_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_canvas_boards
            WHERE canvas_board_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            """,
            canvas_board_id,
            tenant_id,
            workspace_id,
        )
        return self._canvas_board_from_row(row) if row else None

    async def upsert_canvas_board(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        canvas_board_id = str(payload.get("canvas_board_id") or new_id("lbcb"))
        updated_at = utc_now()
        if payload.get("canvas_board_id"):
            row = await pool.fetchrow(
                """
                UPDATE little_bull_canvas_boards
                SET title=$6,
                    slug=$7,
                    layout=$8::jsonb,
                    status=$9,
                    updated_by=$10,
                    updated_at=$11
                WHERE canvas_board_id=$1
                  AND tenant_id IS NOT DISTINCT FROM $2
                  AND workspace_id=$3
                  AND group_id IS NOT DISTINCT FROM $4
                  AND subgroup_id IS NOT DISTINCT FROM $5
                RETURNING *
                """,
                canvas_board_id,
                tenant_id,
                workspace_id,
                payload.get("group_id") or None,
                payload.get("subgroup_id") or None,
                str(payload.get("title") or "").strip(),
                str(payload.get("slug") or "").strip(),
                json.dumps(payload.get("layout") or {}),
                str(payload.get("status") or "active").strip(),
                user_id,
                updated_at,
            )
            if not row:
                raise ValueError("canvas_board_scope_mismatch")
            return self._canvas_board_from_row(row)
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_canvas_boards (
                canvas_board_id, tenant_id, workspace_id, group_id, subgroup_id,
                title, slug, layout, status, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,$10,$10,$11,$11)
            ON CONFLICT (workspace_id, slug) DO UPDATE SET
                group_id=EXCLUDED.group_id,
                subgroup_id=EXCLUDED.subgroup_id,
                title=EXCLUDED.title,
                layout=EXCLUDED.layout,
                status=EXCLUDED.status,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            WHERE little_bull_canvas_boards.group_id IS NOT DISTINCT FROM EXCLUDED.group_id
              AND little_bull_canvas_boards.subgroup_id IS NOT DISTINCT FROM EXCLUDED.subgroup_id
            RETURNING *
            """,
            canvas_board_id,
            tenant_id,
            workspace_id,
            payload.get("group_id") or None,
            payload.get("subgroup_id") or None,
            str(payload.get("title") or "").strip(),
            str(payload.get("slug") or "").strip(),
            json.dumps(payload.get("layout") or {}),
            str(payload.get("status") or "active").strip(),
            user_id,
            updated_at,
        )
        if not row:
            raise ValueError("canvas_board_scope_mismatch")
        return self._canvas_board_from_row(row)

    async def upsert_canvas_node(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        canvas_node_id = str(payload.get("canvas_node_id") or new_id("lbcn"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_canvas_nodes (
                canvas_node_id, tenant_id, workspace_id, canvas_board_id, node_kind,
                ref_kind, ref_id, x, y, width, height, content,
                created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13,$13,$14,$14)
            ON CONFLICT (canvas_node_id) DO UPDATE SET
                node_kind=EXCLUDED.node_kind,
                ref_kind=EXCLUDED.ref_kind,
                ref_id=EXCLUDED.ref_id,
                x=EXCLUDED.x,
                y=EXCLUDED.y,
                width=EXCLUDED.width,
                height=EXCLUDED.height,
                content=EXCLUDED.content,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            canvas_node_id,
            tenant_id,
            workspace_id,
            payload.get("canvas_board_id"),
            str(payload.get("node_kind") or "").strip(),
            str(payload.get("ref_kind") or "").strip(),
            str(payload.get("ref_id") or "").strip(),
            float(payload.get("x") or 0),
            float(payload.get("y") or 0),
            float(payload.get("width") or 280),
            float(payload.get("height") or 160),
            json.dumps(payload.get("content") or {}),
            user_id,
            utc_now(),
        )
        return self._canvas_node_from_row(row)

    async def get_canvas_node(
        self,
        canvas_node_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_canvas_nodes
            WHERE canvas_node_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            """,
            canvas_node_id,
            tenant_id,
            workspace_id,
        )
        return self._canvas_node_from_row(row) if row else None

    async def list_canvas_nodes(
        self,
        *,
        canvas_board_id: str,
        tenant_id: str | None,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_canvas_nodes
            WHERE canvas_board_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            ORDER BY created_at, canvas_node_id
            """,
            canvas_board_id,
            tenant_id,
            workspace_id,
        )
        return [self._canvas_node_from_row(row) for row in rows]

    async def upsert_canvas_edge(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        canvas_edge_id = str(payload.get("canvas_edge_id") or new_id("lbce"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_canvas_edges (
                canvas_edge_id, tenant_id, workspace_id, canvas_board_id,
                source_node_id, target_node_id, edge_kind, label, metadata,
                created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10,$10,$11,$11)
            ON CONFLICT (canvas_edge_id) DO UPDATE SET
                source_node_id=EXCLUDED.source_node_id,
                target_node_id=EXCLUDED.target_node_id,
                edge_kind=EXCLUDED.edge_kind,
                label=EXCLUDED.label,
                metadata=EXCLUDED.metadata,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            canvas_edge_id,
            tenant_id,
            workspace_id,
            payload.get("canvas_board_id"),
            payload.get("source_node_id"),
            payload.get("target_node_id"),
            str(payload.get("edge_kind") or "manual").strip(),
            str(payload.get("label") or "").strip(),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            utc_now(),
        )
        return self._canvas_edge_from_row(row)

    async def list_canvas_edges(
        self,
        *,
        canvas_board_id: str,
        tenant_id: str | None,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_canvas_edges
            WHERE canvas_board_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            ORDER BY created_at, canvas_edge_id
            """,
            canvas_board_id,
            tenant_id,
            workspace_id,
        )
        return [self._canvas_edge_from_row(row) for row in rows]

    async def upsert_knowledge_dossier(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        knowledge_dossier_id = str(payload.get("knowledge_dossier_id") or new_id("lbdos"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_knowledge_dossiers (
                knowledge_dossier_id, tenant_id, workspace_id, group_id, subgroup_id,
                title, slug, dossier_kind, status, content_refs, export_policy,
                approval_id, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11::jsonb,$12,$13,$13,$14,$14)
            ON CONFLICT (workspace_id, slug) DO UPDATE SET
                group_id=EXCLUDED.group_id,
                subgroup_id=EXCLUDED.subgroup_id,
                title=EXCLUDED.title,
                dossier_kind=EXCLUDED.dossier_kind,
                status=EXCLUDED.status,
                content_refs=EXCLUDED.content_refs,
                export_policy=EXCLUDED.export_policy,
                approval_id=EXCLUDED.approval_id,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            knowledge_dossier_id,
            tenant_id,
            workspace_id,
            payload.get("group_id") or None,
            payload.get("subgroup_id") or None,
            str(payload.get("title") or "").strip(),
            str(payload.get("slug") or "").strip(),
            str(payload.get("dossier_kind") or "knowledge").strip(),
            str(payload.get("status") or "draft").strip(),
            json.dumps(payload.get("content_refs") or []),
            json.dumps(payload.get("export_policy") or {}),
            payload.get("approval_id") or None,
            user_id,
            utc_now(),
        )
        return self._knowledge_dossier_from_row(row)

    async def list_content_maps(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_content_maps
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR group_id=$3)
              AND ($4::text IS NULL OR subgroup_id=$4)
            ORDER BY updated_at DESC
            """,
            tenant_id,
            workspace_id,
            group_id,
            subgroup_id,
        )
        return [self._content_map_from_row(row) for row in rows]

    async def upsert_content_map(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        content_map_id = str(payload.get("content_map_id") or new_id("lbmoc"))
        updated_at = utc_now()
        if payload.get("content_map_id"):
            row = await pool.fetchrow(
                """
                UPDATE little_bull_content_maps
                SET title=$6,
                    slug=$7,
                    root_note_id=$8,
                    description=$9,
                    map_body=$10::jsonb,
                    status=$11,
                    updated_by=$12,
                    updated_at=$13
                WHERE content_map_id=$1
                  AND tenant_id IS NOT DISTINCT FROM $2
                  AND workspace_id=$3
                  AND group_id IS NOT DISTINCT FROM $4
                  AND subgroup_id IS NOT DISTINCT FROM $5
                RETURNING *
                """,
                content_map_id,
                tenant_id,
                workspace_id,
                payload.get("group_id") or None,
                payload.get("subgroup_id") or None,
                str(payload.get("title") or "").strip(),
                str(payload.get("slug") or "").strip(),
                payload.get("root_note_id") or None,
                str(payload.get("description") or "").strip(),
                json.dumps(payload.get("map_body") or {}),
                str(payload.get("status") or "draft").strip(),
                user_id,
                updated_at,
            )
            if not row:
                raise ValueError("content_map_scope_mismatch")
            return self._content_map_from_row(row)
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_content_maps (
                content_map_id, tenant_id, workspace_id, group_id, subgroup_id,
                title, slug, root_note_id, description, map_body, status,
                created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11,$12,$12,$13,$13)
            ON CONFLICT (workspace_id, slug) DO UPDATE SET
                group_id=EXCLUDED.group_id,
                subgroup_id=EXCLUDED.subgroup_id,
                title=EXCLUDED.title,
                root_note_id=EXCLUDED.root_note_id,
                description=EXCLUDED.description,
                map_body=EXCLUDED.map_body,
                status=EXCLUDED.status,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            WHERE little_bull_content_maps.group_id IS NOT DISTINCT FROM EXCLUDED.group_id
              AND little_bull_content_maps.subgroup_id IS NOT DISTINCT FROM EXCLUDED.subgroup_id
            RETURNING *
            """,
            content_map_id,
            tenant_id,
            workspace_id,
            payload.get("group_id") or None,
            payload.get("subgroup_id") or None,
            str(payload.get("title") or "").strip(),
            str(payload.get("slug") or "").strip(),
            payload.get("root_note_id") or None,
            str(payload.get("description") or "").strip(),
            json.dumps(payload.get("map_body") or {}),
            str(payload.get("status") or "draft").strip(),
            user_id,
            updated_at,
        )
        if not row:
            raise ValueError("content_map_scope_mismatch")
        return self._content_map_from_row(row)

    async def list_knowledge_trails(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        group_id: str | None = None,
        subgroup_id: str | None = None,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_knowledge_trails
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR group_id=$3)
              AND ($4::text IS NULL OR subgroup_id=$4)
            ORDER BY updated_at DESC
            """,
            tenant_id,
            workspace_id,
            group_id,
            subgroup_id,
        )
        return [self._knowledge_trail_from_row(row) for row in rows]

    async def get_knowledge_trail(
        self,
        knowledge_trail_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_knowledge_trails
            WHERE knowledge_trail_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            """,
            knowledge_trail_id,
            tenant_id,
            workspace_id,
        )
        return self._knowledge_trail_from_row(row) if row else None

    async def upsert_knowledge_trail(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        knowledge_trail_id = str(payload.get("knowledge_trail_id") or new_id("lbtrail"))
        updated_at = utc_now()
        if payload.get("knowledge_trail_id"):
            row = await pool.fetchrow(
                """
                UPDATE little_bull_knowledge_trails
                SET title=$6,
                    slug=$7,
                    trail_type=$8,
                    description=$9,
                    status=$10,
                    metadata=$11::jsonb,
                    updated_by=$12,
                    updated_at=$13
                WHERE knowledge_trail_id=$1
                  AND tenant_id IS NOT DISTINCT FROM $2
                  AND workspace_id=$3
                  AND group_id IS NOT DISTINCT FROM $4
                  AND subgroup_id IS NOT DISTINCT FROM $5
                RETURNING *
                """,
                knowledge_trail_id,
                tenant_id,
                workspace_id,
                payload.get("group_id") or None,
                payload.get("subgroup_id") or None,
                str(payload.get("title") or "").strip(),
                str(payload.get("slug") or "").strip(),
                str(payload.get("trail_type") or "study").strip(),
                str(payload.get("description") or "").strip(),
                str(payload.get("status") or "draft").strip(),
                json.dumps(payload.get("metadata") or {}),
                user_id,
                updated_at,
            )
            if not row:
                raise ValueError("knowledge_trail_scope_mismatch")
            return self._knowledge_trail_from_row(row)
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_knowledge_trails (
                knowledge_trail_id, tenant_id, workspace_id, group_id, subgroup_id,
                title, slug, trail_type, description, status, metadata,
                created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12,$12,$13,$13)
            ON CONFLICT (workspace_id, slug) DO UPDATE SET
                group_id=EXCLUDED.group_id,
                subgroup_id=EXCLUDED.subgroup_id,
                title=EXCLUDED.title,
                trail_type=EXCLUDED.trail_type,
                description=EXCLUDED.description,
                status=EXCLUDED.status,
                metadata=EXCLUDED.metadata,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            WHERE little_bull_knowledge_trails.group_id IS NOT DISTINCT FROM EXCLUDED.group_id
              AND little_bull_knowledge_trails.subgroup_id IS NOT DISTINCT FROM EXCLUDED.subgroup_id
            RETURNING *
            """,
            knowledge_trail_id,
            tenant_id,
            workspace_id,
            payload.get("group_id") or None,
            payload.get("subgroup_id") or None,
            str(payload.get("title") or "").strip(),
            str(payload.get("slug") or "").strip(),
            str(payload.get("trail_type") or "study").strip(),
            str(payload.get("description") or "").strip(),
            str(payload.get("status") or "draft").strip(),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            updated_at,
        )
        if not row:
            raise ValueError("knowledge_trail_scope_mismatch")
        return self._knowledge_trail_from_row(row)

    async def list_knowledge_trail_steps(
        self,
        *,
        knowledge_trail_id: str,
        tenant_id: str | None,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_knowledge_trail_steps
            WHERE knowledge_trail_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            ORDER BY step_order, created_at
            """,
            knowledge_trail_id,
            tenant_id,
            workspace_id,
        )
        return [self._knowledge_trail_step_from_row(row) for row in rows]

    async def upsert_knowledge_trail_step(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        knowledge_trail_step_id = str(payload.get("knowledge_trail_step_id") or new_id("lbtrails"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_knowledge_trail_steps (
                knowledge_trail_step_id, tenant_id, workspace_id, knowledge_trail_id,
                step_order, title, step_kind, note_id, document_id, canvas_board_id,
                instructions, metadata, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,$13,$13,$14,$14)
            ON CONFLICT (knowledge_trail_id, step_order) DO UPDATE SET
                title=EXCLUDED.title,
                step_kind=EXCLUDED.step_kind,
                note_id=EXCLUDED.note_id,
                document_id=EXCLUDED.document_id,
                canvas_board_id=EXCLUDED.canvas_board_id,
                instructions=EXCLUDED.instructions,
                metadata=EXCLUDED.metadata,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            knowledge_trail_step_id,
            tenant_id,
            workspace_id,
            payload.get("knowledge_trail_id"),
            int(payload.get("step_order") or 0),
            str(payload.get("title") or "").strip(),
            str(payload.get("step_kind") or "note").strip(),
            payload.get("note_id") or None,
            payload.get("document_id") or None,
            payload.get("canvas_board_id") or None,
            str(payload.get("instructions") or "").strip(),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            utc_now(),
        )
        return self._knowledge_trail_step_from_row(row)

    async def list_knowledge_inbox_items(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        status: str | None = None,
        group_id: str | None = None,
        subgroup_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_knowledge_inbox_items
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR status=$3)
              AND ($4::text IS NULL OR group_id=$4)
              AND ($5::text IS NULL OR subgroup_id=$5)
            ORDER BY created_at DESC
            LIMIT $6
            """,
            tenant_id,
            workspace_id,
            status,
            group_id,
            subgroup_id,
            limit,
        )
        return [self._knowledge_inbox_item_from_row(row) for row in rows]

    async def get_knowledge_inbox_item(
        self,
        inbox_item_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_knowledge_inbox_items
            WHERE inbox_item_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            """,
            inbox_item_id,
            tenant_id,
            workspace_id,
        )
        return self._knowledge_inbox_item_from_row(row) if row else None

    async def upsert_knowledge_inbox_item(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        inbox_item_id = str(payload.get("inbox_item_id") or new_id("lbinbox"))
        now = utc_now()
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_knowledge_inbox_items (
                inbox_item_id, tenant_id, workspace_id, group_id, subgroup_id,
                item_kind, title, body, source_kind, source_id, status, priority,
                metadata, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb,$14,$14,$15,$15)
            ON CONFLICT (inbox_item_id) DO UPDATE SET
                group_id=EXCLUDED.group_id,
                subgroup_id=EXCLUDED.subgroup_id,
                item_kind=EXCLUDED.item_kind,
                title=EXCLUDED.title,
                body=EXCLUDED.body,
                source_kind=EXCLUDED.source_kind,
                source_id=EXCLUDED.source_id,
                status=EXCLUDED.status,
                priority=EXCLUDED.priority,
                metadata=EXCLUDED.metadata,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            inbox_item_id,
            tenant_id,
            workspace_id,
            payload.get("group_id") or None,
            payload.get("subgroup_id") or None,
            str(payload.get("item_kind") or "").strip(),
            str(payload.get("title") or "").strip(),
            str(payload.get("body") or "").strip(),
            str(payload.get("source_kind") or "").strip(),
            str(payload.get("source_id") or "").strip(),
            str(payload.get("status") or "open").strip(),
            str(payload.get("priority") or "normal").strip(),
            json.dumps(payload.get("metadata") or {}),
            user_id,
            now,
        )
        return self._knowledge_inbox_item_from_row(row)

    async def update_knowledge_inbox_item_status(
        self,
        inbox_item_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
        status: str,
        metadata: dict[str, Any] | None,
        user_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            UPDATE little_bull_knowledge_inbox_items
            SET status=$4,
                metadata=metadata || $5::jsonb,
                updated_by=$6,
                updated_at=$7
            WHERE inbox_item_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            RETURNING *
            """,
            inbox_item_id,
            tenant_id,
            workspace_id,
            status,
            json.dumps(metadata or {}),
            user_id,
            utc_now(),
        )
        return self._knowledge_inbox_item_from_row(row) if row else None

    async def list_daily_notes(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_daily_notes
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
            ORDER BY note_date DESC
            LIMIT $3
            """,
            tenant_id,
            workspace_id,
            limit,
        )
        return [self._daily_note_from_row(row) for row in rows]

    async def get_daily_note(
        self,
        note_date: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_daily_notes
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND note_date=$3::date
            """,
            tenant_id,
            workspace_id,
            note_date,
        )
        return self._daily_note_from_row(row) if row else None

    async def upsert_daily_note(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        daily_note_id = str(payload.get("daily_note_id") or new_id("lbdaily"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_daily_notes (
                daily_note_id, tenant_id, workspace_id, note_id, note_date,
                summary, decisions, pending_items, cost_snapshot,
                created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5::date,$6,$7::jsonb,$8::jsonb,$9::jsonb,$10,$10,$11,$11)
            ON CONFLICT (workspace_id, note_date) DO UPDATE SET
                note_id=EXCLUDED.note_id,
                summary=EXCLUDED.summary,
                decisions=EXCLUDED.decisions,
                pending_items=EXCLUDED.pending_items,
                cost_snapshot=EXCLUDED.cost_snapshot,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            RETURNING *
            """,
            daily_note_id,
            tenant_id,
            workspace_id,
            payload.get("note_id"),
            payload.get("note_date"),
            str(payload.get("summary") or "").strip(),
            json.dumps(payload.get("decisions") or []),
            json.dumps(payload.get("pending_items") or []),
            json.dumps(payload.get("cost_snapshot") or {}),
            user_id,
            utc_now(),
        )
        return self._daily_note_from_row(row)

    async def list_agent_configs(self, *, tenant_id: str | None, workspace_id: str | None) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_agent_configs
            WHERE (tenant_id IS NULL OR tenant_id IS NOT DISTINCT FROM $1)
              AND (workspace_id IS NULL OR workspace_id IS NOT DISTINCT FROM $2)
            ORDER BY enabled DESC, name
            """,
            tenant_id,
            workspace_id,
        )
        return [self._agent_from_row(row) for row in rows]

    async def upsert_agent_config(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str | None,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        agent_id = str(payload.get("agent_id") or new_id("lba"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_agent_configs (
                agent_id, tenant_id, workspace_id, name, description, enabled, model_setting_id,
                system_prompt, response_rules, tools, config, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12,$12,$13,$13)
            ON CONFLICT (agent_id) DO UPDATE SET
                name=EXCLUDED.name,
                description=EXCLUDED.description,
                enabled=EXCLUDED.enabled,
                model_setting_id=EXCLUDED.model_setting_id,
                system_prompt=EXCLUDED.system_prompt,
                response_rules=EXCLUDED.response_rules,
                tools=EXCLUDED.tools,
                config=EXCLUDED.config,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            WHERE little_bull_agent_configs.tenant_id IS NOT DISTINCT FROM EXCLUDED.tenant_id
              AND little_bull_agent_configs.workspace_id IS NOT DISTINCT FROM EXCLUDED.workspace_id
            RETURNING *
            """,
            agent_id,
            tenant_id,
            workspace_id,
            str(payload.get("name") or "Agente").strip(),
            str(payload.get("description") or "").strip(),
            bool(payload.get("enabled", True)),
            payload.get("model_setting_id") or None,
            str(payload.get("system_prompt") or "").strip(),
            [str(item) for item in payload.get("response_rules") or []],
            [str(item) for item in payload.get("tools") or []],
            json.dumps(payload.get("config") or {}),
            user_id,
            utc_now(),
        )
        if not row:
            raise ValueError("agent_config_scope_mismatch")
        return self._agent_from_row(row)

    async def list_agent_builder_sessions(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_agent_builder_sessions
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR user_id=$3)
              AND ($4::text IS NULL OR status=$4)
            ORDER BY updated_at DESC
            """,
            tenant_id,
            workspace_id,
            user_id,
            status,
        )
        return [self._agent_builder_session_from_row(row) for row in rows]

    async def get_agent_builder_session(
        self,
        agent_builder_session_id: str,
        *,
        tenant_id: str | None,
        workspace_id: str,
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT * FROM little_bull_agent_builder_sessions
            WHERE agent_builder_session_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND workspace_id=$3
            """,
            agent_builder_session_id,
            tenant_id,
            workspace_id,
        )
        return self._agent_builder_session_from_row(row) if row else None

    async def upsert_agent_builder_session(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        agent_builder_session_id = str(payload.get("agent_builder_session_id") or new_id("lbbuild"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_agent_builder_sessions (
                agent_builder_session_id, tenant_id, workspace_id, user_id, agent_id,
                model_setting_id, status, current_step, builder_transcript,
                generated_config, readiness_score, requires_review,
                created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::jsonb,$10::jsonb,$11,$12,$13,$13,$14,$14)
            ON CONFLICT (agent_builder_session_id) DO UPDATE SET
                agent_id=EXCLUDED.agent_id,
                model_setting_id=EXCLUDED.model_setting_id,
                status=EXCLUDED.status,
                current_step=EXCLUDED.current_step,
                builder_transcript=EXCLUDED.builder_transcript,
                generated_config=EXCLUDED.generated_config,
                readiness_score=EXCLUDED.readiness_score,
                requires_review=EXCLUDED.requires_review,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            WHERE little_bull_agent_builder_sessions.tenant_id IS NOT DISTINCT FROM EXCLUDED.tenant_id
              AND little_bull_agent_builder_sessions.workspace_id=EXCLUDED.workspace_id
            RETURNING *
            """,
            agent_builder_session_id,
            tenant_id,
            workspace_id,
            payload.get("user_id") or user_id,
            payload.get("agent_id") or None,
            payload.get("model_setting_id") or None,
            str(payload.get("status") or "draft").strip(),
            str(payload.get("current_step") or "intake").strip(),
            json.dumps(payload.get("builder_transcript") or []),
            json.dumps(payload.get("generated_config") or {}),
            int(payload.get("readiness_score") or 0),
            bool(payload.get("requires_review", True)),
            user_id,
            utc_now(),
        )
        if not row:
            raise ValueError("agent_builder_session_scope_mismatch")
        return self._agent_builder_session_from_row(row)

    async def list_agent_context_budgets(
        self,
        *,
        tenant_id: str | None,
        workspace_id: str,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_agent_context_budgets
            WHERE tenant_id IS NOT DISTINCT FROM $1
              AND workspace_id=$2
              AND ($3::text IS NULL OR agent_id=$3)
            ORDER BY updated_at DESC
            """,
            tenant_id,
            workspace_id,
            agent_id,
        )
        return [self._agent_context_budget_from_row(row) for row in rows]

    async def upsert_agent_context_budget(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        agent_context_budget_id = payload.get("agent_context_budget_id")
        if not agent_context_budget_id:
            agent_context_budget_id = await pool.fetchval(
                """
                SELECT agent_context_budget_id
                FROM little_bull_agent_context_budgets
                WHERE tenant_id IS NOT DISTINCT FROM $1
                  AND workspace_id=$2
                  AND agent_id=$3
                  AND model_setting_id IS NOT DISTINCT FROM $4
                """,
                tenant_id,
                workspace_id,
                payload.get("agent_id"),
                payload.get("model_setting_id") or None,
            )
        agent_context_budget_id = str(agent_context_budget_id or new_id("lbbudget"))
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_agent_context_budgets (
                agent_context_budget_id, tenant_id, workspace_id, agent_id,
                model_setting_id, max_context_tokens, reserved_response_tokens,
                max_prompt_tokens, daily_cost_limit_usd, monthly_cost_limit_usd,
                policy, created_by, updated_by, created_at, updated_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12,$12,$13,$13)
            ON CONFLICT (agent_context_budget_id) DO UPDATE SET
                model_setting_id=EXCLUDED.model_setting_id,
                max_context_tokens=EXCLUDED.max_context_tokens,
                reserved_response_tokens=EXCLUDED.reserved_response_tokens,
                max_prompt_tokens=EXCLUDED.max_prompt_tokens,
                daily_cost_limit_usd=EXCLUDED.daily_cost_limit_usd,
                monthly_cost_limit_usd=EXCLUDED.monthly_cost_limit_usd,
                policy=EXCLUDED.policy,
                updated_by=EXCLUDED.updated_by,
                updated_at=EXCLUDED.updated_at
            WHERE little_bull_agent_context_budgets.tenant_id IS NOT DISTINCT FROM EXCLUDED.tenant_id
              AND little_bull_agent_context_budgets.workspace_id=EXCLUDED.workspace_id
              AND little_bull_agent_context_budgets.agent_id=EXCLUDED.agent_id
            RETURNING *
            """,
            agent_context_budget_id,
            tenant_id,
            workspace_id,
            payload.get("agent_id"),
            payload.get("model_setting_id") or None,
            int(payload.get("max_context_tokens") or 0),
            int(payload.get("reserved_response_tokens") or 0),
            int(payload.get("max_prompt_tokens") or 0),
            payload.get("daily_cost_limit_usd"),
            payload.get("monthly_cost_limit_usd"),
            json.dumps(payload.get("policy") or {}),
            user_id,
            utc_now(),
        )
        if not row:
            raise ValueError("agent_context_budget_scope_mismatch")
        return self._agent_context_budget_from_row(row)

    async def save_conversation(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        conversation_id = str(payload.get("conversation_id") or new_id("lbc"))
        messages = list(payload.get("messages") or [])
        title = str(payload.get("title") or "").strip()
        if not title:
            first_user = next((message.get("content") for message in messages if message.get("role") == "user"), "")
            title = str(first_user or "Conversa Little Bull").strip()[:120]
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    INSERT INTO little_bull_conversations (
                        conversation_id, tenant_id, workspace_id, user_id, title, agent_id,
                        model_profile, confidentiality, created_at, updated_at
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$9)
                    ON CONFLICT (conversation_id) DO UPDATE SET
                        title=EXCLUDED.title,
                        agent_id=EXCLUDED.agent_id,
                        model_profile=EXCLUDED.model_profile,
                        confidentiality=EXCLUDED.confidentiality,
                        updated_at=EXCLUDED.updated_at
                    RETURNING *, 0::int AS message_count
                    """,
                    conversation_id,
                    tenant_id,
                    workspace_id,
                    user_id,
                    title,
                    payload.get("agent_id") or None,
                    str(payload.get("model_profile") or "equilibrado"),
                    str(payload.get("confidentiality") or "normal"),
                    utc_now(),
                )
                await conn.execute(
                    "DELETE FROM little_bull_conversation_messages WHERE conversation_id=$1",
                    conversation_id,
                )
                for index, message in enumerate(messages):
                    await conn.execute(
                        """
                        INSERT INTO little_bull_conversation_messages (
                            message_id, conversation_id, role, content, message_references, metadata, created_at
                        )
                        VALUES ($1,$2,$3,$4,$5::jsonb,$6::jsonb,$7)
                        """,
                        str(message.get("message_id") or message.get("id") or new_id("lbmsg")),
                        conversation_id,
                        str(message.get("role") or "user"),
                        str(message.get("content") or ""),
                        json.dumps(message.get("references") or []),
                        json.dumps({"order": index, **(message.get("metadata") or {})}),
                        utc_now(),
                    )
        saved = self._conversation_from_row(row, messages=await self.list_messages(conversation_id))
        saved["message_count"] = len(saved["messages"])
        return saved

    async def list_conversations(
        self, *, tenant_id: str | None, workspace_id: str, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT c.*, count(m.message_id)::int AS message_count
            FROM little_bull_conversations c
            LEFT JOIN little_bull_conversation_messages m ON m.conversation_id = c.conversation_id
            WHERE c.workspace_id=$1
              AND c.tenant_id IS NOT DISTINCT FROM $2
              AND ($3::text IS NULL OR c.user_id=$3)
            GROUP BY c.conversation_id
            ORDER BY c.updated_at DESC
            """,
            workspace_id,
            tenant_id,
            user_id,
        )
        return [self._conversation_from_row(row) for row in rows]

    async def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            SELECT c.*, count(m.message_id)::int AS message_count
            FROM little_bull_conversations c
            LEFT JOIN little_bull_conversation_messages m ON m.conversation_id = c.conversation_id
            WHERE c.conversation_id=$1
            GROUP BY c.conversation_id
            """,
            conversation_id,
        )
        if row is None:
            return None
        return self._conversation_from_row(row, messages=await self.list_messages(conversation_id))

    async def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_conversation_messages
            WHERE conversation_id=$1
            ORDER BY (metadata->>'order')::int NULLS LAST, created_at
            """,
            conversation_id,
        )
        return [self._message_from_row(row) for row in rows]

    async def create_correlation_suggestion(
        self,
        payload: dict[str, Any],
        *,
        tenant_id: str | None,
        workspace_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO little_bull_correlation_suggestions (
                suggestion_id, tenant_id, workspace_id, user_id, source_label, target_label,
                reason, status, metadata, created_at
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,'pending',$8::jsonb,$9)
            RETURNING *
            """,
            new_id("lbcorr"),
            tenant_id,
            workspace_id,
            user_id,
            str(payload.get("source_label") or "").strip(),
            str(payload.get("target_label") or "").strip(),
            str(payload.get("reason") or "").strip(),
            json.dumps(payload.get("metadata") or {}),
            utc_now(),
        )
        return self._suggestion_from_row(row)

    async def list_correlation_suggestions(
        self, *, tenant_id: str | None, workspace_id: str, status: str | None = None
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM little_bull_correlation_suggestions
            WHERE workspace_id=$1
              AND tenant_id IS NOT DISTINCT FROM $2
              AND ($3::text IS NULL OR status=$3)
            ORDER BY created_at DESC
            """,
            workspace_id,
            tenant_id,
            status,
        )
        return [self._suggestion_from_row(row) for row in rows]

    async def get_correlation_suggestion(self, suggestion_id: str) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM little_bull_correlation_suggestions WHERE suggestion_id=$1",
            suggestion_id,
        )
        return self._suggestion_from_row(row) if row else None

    async def decide_correlation_suggestion(
        self, suggestion_id: str, *, status: str, decided_by: str
    ) -> dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            """
            UPDATE little_bull_correlation_suggestions
            SET status=$2, decided_by=$3, decided_at=$4
            WHERE suggestion_id=$1
            RETURNING *
            """,
            suggestion_id,
            status,
            decided_by,
            utc_now(),
        )
        return self._suggestion_from_row(row) if row else None
