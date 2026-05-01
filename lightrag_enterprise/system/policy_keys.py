from __future__ import annotations

import hashlib
import json
from typing import Any


WORKSPACE_PRIVATE_POLICY = "little_bull.workspace_contains_private_data"
PRIVATE_DATA_HOSTED_LLM_EXCEPTION_POLICY = (
    "little_bull.private_data.hosted_llm_exception"
)
WORKSPACE_DATA_PLANE_POLICY = "little_bull.workspace_data_plane"


def stable_policy_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
