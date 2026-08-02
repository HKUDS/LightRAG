"""Shared measurement of normalized model-facing conversation input."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def count_conversation_input_chars(
    query: str,
    instruction: str | None,
    history: Sequence[dict[str, Any]] | None,
) -> int:
    """Count the canonical conversation input sent into an LLM flow.

    ``history`` must be the representation the caller forwards downstream. Its
    serialized form deliberately counts roles, field names, and permitted extra
    fields, so either API surface cannot hide input outside the shared budget.
    """
    total = len(query) + len(instruction or "")
    if history:
        total += len(json.dumps(history, ensure_ascii=False))
    return total
