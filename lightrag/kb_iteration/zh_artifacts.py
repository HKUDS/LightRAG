from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Protocol


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
_MODEL_TOKEN_PATTERN = re.compile(
    r"^(?=.*\d)[a-z][a-z0-9]*(?:-[a-z0-9]+)+$", re.IGNORECASE
)
_PROPOSAL_ID_PATTERN = re.compile(
    r"^prop-[a-z0-9]+(?:-[a-z0-9]+)+$", re.IGNORECASE
)
_SNAKE_CASE_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$", re.IGNORECASE)


class ZhArtifactClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Return a synchronous completion for later OpenAI-compatible clients."""


@dataclass(frozen=True)
class ZhArtifactResult:
    artifact_key: str
    source_relative_path: Path
    zh_relative_path: Path
    content_type: str
    generated: bool
    fallback_to_source: bool
    generated_at: str
    model: str | None
    error: str | None
    payload: Any | None = None
    content: str | None = None


def artifact_zh_relative_path(path: Path) -> Path:
    """Return a sibling relative artifact path with `.zh` before the final suffix."""

    if path.suffix:
        return path.with_name(f"{path.stem}.zh{path.suffix}")
    return path.with_name(f"{path.name}.zh")


def build_zh_json_payload(payload: Any) -> Any:
    """Copy a JSON-like value and add `_zh_labels` maps for known object keys.

    `_zh_labels` is reserved for generated display labels; when known labels exist
    at a mapping level, generated labels replace any existing `_zh_labels` value.
    """

    return _label_json_value(payload)


def ensure_zh_artifact(
    source_path: Path | str,
    *,
    artifact_key: str,
    content_type: str,
    client: ZhArtifactClient,
    model: str | None = None,
    force: bool = False,
    artifact_root: Path | str | None = None,
) -> ZhArtifactResult:
    """Ensure a sibling Chinese display artifact exists for a source artifact."""

    source = Path(source_path)
    root = Path(artifact_root) if artifact_root is not None else source.parent
    zh_path = artifact_zh_relative_path(source)
    source_relative_path = _safe_relative_to(source, root)
    zh_relative_path = _safe_relative_to(zh_path, root)

    try:
        if zh_path.exists() and not force:
            return _existing_result(
                zh_path=zh_path,
                artifact_key=artifact_key,
                source_relative_path=source_relative_path,
                zh_relative_path=zh_relative_path,
                content_type=content_type,
                model=model,
            )

        if content_type == "application/json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            zh_payload = build_zh_json_payload(payload)
            zh_path.parent.mkdir(parents=True, exist_ok=True)
            zh_path.write_text(
                json.dumps(zh_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return _result(
                artifact_key=artifact_key,
                source_relative_path=source_relative_path,
                zh_relative_path=zh_relative_path,
                content_type=content_type,
                generated=True,
                fallback_to_source=False,
                model=model,
                payload=zh_payload,
            )

        source_content = source.read_text(encoding="utf-8")
        translated = client.complete(
            system_prompt=_build_markdown_system_prompt(),
            user_prompt=_build_markdown_user_prompt(
                source_relative_path, source_content
            ),
        )
        content = _normalize_text_artifact(translated)
        zh_path.parent.mkdir(parents=True, exist_ok=True)
        zh_path.write_text(content, encoding="utf-8")
        return _result(
            artifact_key=artifact_key,
            source_relative_path=source_relative_path,
            zh_relative_path=zh_relative_path,
            content_type=content_type,
            generated=True,
            fallback_to_source=False,
            model=model,
            content=content,
        )
    except Exception as exc:
        return _fallback_result(
            source=source,
            artifact_key=artifact_key,
            source_relative_path=source_relative_path,
            zh_relative_path=zh_relative_path,
            content_type=content_type,
            model=model,
            error=str(exc),
        )


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


def _build_markdown_system_prompt() -> str:
    return (
        "请将面向人的自然语言内容翻译成简洁中文，用于知识图谱迭代工作台展示。"
        "必须保留机器标识符和结构化字段原文，包括 proposal_id、source_id、"
        "doc_id、file_path、workspace、run_id、JSON/YAML keys、node/relation/"
        "evidence IDs、model names、paths、code blocks、API field names。"
        "不要翻译代码块、路径、模型名、键名或 ID。保持 Markdown 结构。"
    )


def _build_markdown_user_prompt(source_relative_path: Path, source_content: str) -> str:
    return (
        f"源文件: {source_relative_path.as_posix()}\n\n"
        "请翻译以下 Markdown/text 内容：\n\n"
        f"{source_content}"
    )


def _existing_result(
    *,
    zh_path: Path,
    artifact_key: str,
    source_relative_path: Path,
    zh_relative_path: Path,
    content_type: str,
    model: str | None,
) -> ZhArtifactResult:
    if content_type == "application/json":
        payload = json.loads(zh_path.read_text(encoding="utf-8"))
        return _result(
            artifact_key=artifact_key,
            source_relative_path=source_relative_path,
            zh_relative_path=zh_relative_path,
            content_type=content_type,
            generated=False,
            fallback_to_source=False,
            model=model,
            payload=payload,
        )

    return _result(
        artifact_key=artifact_key,
        source_relative_path=source_relative_path,
        zh_relative_path=zh_relative_path,
        content_type=content_type,
        generated=False,
        fallback_to_source=False,
        model=model,
        content=zh_path.read_text(encoding="utf-8"),
    )


def _fallback_result(
    *,
    source: Path,
    artifact_key: str,
    source_relative_path: Path,
    zh_relative_path: Path,
    content_type: str,
    model: str | None,
    error: str,
) -> ZhArtifactResult:
    content: str | None = None
    payload: Any | None = None
    if content_type == "application/json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            payload = None
    else:
        try:
            content = source.read_text(encoding="utf-8")
        except Exception:
            content = None

    return _result(
        artifact_key=artifact_key,
        source_relative_path=source_relative_path,
        zh_relative_path=zh_relative_path,
        content_type=content_type,
        generated=False,
        fallback_to_source=True,
        model=model,
        error=error,
        payload=payload,
        content=content,
    )


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


def _normalize_text_artifact(content: str) -> str:
    return content.strip() + "\n"


def _result(
    *,
    artifact_key: str,
    source_relative_path: Path,
    zh_relative_path: Path,
    content_type: str,
    generated: bool,
    fallback_to_source: bool,
    model: str | None,
    error: str | None = None,
    payload: Any | None = None,
    content: str | None = None,
) -> ZhArtifactResult:
    return ZhArtifactResult(
        artifact_key=artifact_key,
        source_relative_path=source_relative_path,
        zh_relative_path=zh_relative_path,
        content_type=content_type,
        generated=generated,
        fallback_to_source=fallback_to_source,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=model,
        error=error,
        payload=payload,
        content=content,
    )


def _safe_relative_to(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)
