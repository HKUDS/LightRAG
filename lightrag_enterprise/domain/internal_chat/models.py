from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ChatWorkspace:
    workspace_id: str
    tenant_id: str
    name: str


@dataclass(frozen=True)
class ChatChannel:
    channel_id: str
    tenant_id: str
    workspace_id: str
    name: str
    is_private: bool = False


@dataclass(frozen=True)
class ChatParticipant:
    user_id: str
    tenant_id: str
    display_name: str
    role: str = "member"


@dataclass(frozen=True)
class ChatAttachment:
    attachment_id: str
    file_name: str
    content_type: str
    uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    message_id: str
    tenant_id: str
    workspace_id: str
    channel_id: str
    sender_id: str
    body: str
    thread_id: str | None = None
    citations: tuple[str, ...] = ()
    attachments: tuple[ChatAttachment, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ChatThread:
    thread_id: str
    tenant_id: str
    workspace_id: str
    channel_id: str
    participant_ids: tuple[str, ...] = ()
    memory_summary: str | None = None
    human_handoff: bool = False
    moderated: bool = False
