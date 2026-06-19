from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any


ZH_LABELS: dict[str, str] = {
    "critical_blockers": "关键阻断项",
    "details": "详情",
    "findings": "发现项",
    "hierarchy_branches": "层级分支",
    "hierarchy_missing_branch_count": "缺失层级分支数",
    "hierarchy_present_branch_count": "已有层级分支数",
    "hierarchy_required_branch_count": "必需层级分支数",
    "metrics": "指标",
    "missing": "缺失项",
    "overall": "总分",
    "present": "已有项",
    "required": "必需项",
    "subscores": "子分数",
}

_DOC_CHUNK_ID_PATTERN = re.compile(
    r"^doc-[0-9a-f]{16,}(?:-chunk-\d+)?$", re.IGNORECASE
)
_MODEL_TOKEN_PATTERN = re.compile(r"^(?=.*\d)[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")
_PROPOSAL_ID_PATTERN = re.compile(r"^prop-[a-z0-9]+(?:-[a-z0-9]+)+$")
_SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")


def artifact_zh_relative_path(path: Path) -> Path:
    """Return a sibling relative artifact path with `.zh` before the final suffix."""

    if path.suffix:
        return path.with_name(f"{path.stem}.zh{path.suffix}")
    return path.with_name(f"{path.name}.zh")


def build_zh_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Copy a JSON-like mapping and add `_zh_labels` maps for known keys."""

    return _label_json_value(payload)


def machine_token_should_be_preserved(value: str) -> bool:
    """Return whether a string looks like an identifier/path/model token."""

    text = value.strip()
    if not text or any(character.isspace() for character in text):
        return False
    if "/" in text or "\\" in text:
        return True
    if Path(text).suffix:
        return True
    if _SNAKE_CASE_PATTERN.fullmatch(text):
        return True
    if _PROPOSAL_ID_PATTERN.fullmatch(text):
        return True
    if _DOC_CHUNK_ID_PATTERN.fullmatch(text):
        return True
    return bool(_MODEL_TOKEN_PATTERN.fullmatch(text))


def _label_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        copied = {key: _label_json_value(item) for key, item in value.items()}
        labels = {
            key: ZH_LABELS[key]
            for key in value
            if isinstance(key, str) and key in ZH_LABELS
        }
        if labels:
            copied["_zh_labels"] = labels
        return copied
    if isinstance(value, list):
        return [_label_json_value(item) for item in value]
    return deepcopy(value)
