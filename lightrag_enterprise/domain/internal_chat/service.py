from __future__ import annotations

from dataclasses import dataclass, replace

from .models import ChatMessage, ChatThread


@dataclass
class InMemoryChatRepository:
    messages: dict[str, ChatMessage]
    threads: dict[str, ChatThread]

    def __init__(self) -> None:
        self.messages = {}
        self.threads = {}

    def add_message(self, message: ChatMessage) -> ChatMessage:
        if message.message_id in self.messages:
            raise ValueError("Message already exists")
        self.messages[message.message_id] = message
        return message

    def search_messages(self, tenant_id: str, workspace_id: str, text: str) -> list[ChatMessage]:
        needle = text.lower()
        return [
            message
            for message in self.messages.values()
            if message.tenant_id == tenant_id
            and message.workspace_id == workspace_id
            and needle in message.body.lower()
        ]

    def upsert_thread_memory(self, thread_id: str, summary: str) -> ChatThread:
        current = self.threads[thread_id]
        updated = replace(current, memory_summary=summary)
        self.threads[thread_id] = updated
        return updated
