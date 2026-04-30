from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from lightrag_enterprise.system.db import get_database_url
from lightrag_enterprise.system.little_bull_admin_schema import LITTLE_BULL_ADMIN_SCHEMA_SQL
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
                await conn.execute(LITTLE_BULL_ADMIN_SCHEMA_SQL)
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
        return self._model_from_row(row)

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
        return self._agent_from_row(row)

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
