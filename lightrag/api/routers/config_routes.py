"""
Configuration workbench routes for the LightRAG WebUI.

These endpoints edit operator-owned files such as `.env` and entity-type
prompt profiles. Changes are file-level only and require a server restart to
take effect in the running LightRAG instance.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import dotenv_values, set_key
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from lightrag.prompt import (
    PROMPTS,
    get_default_entity_extraction_prompt_profile,
    get_entity_type_prompt_dir,
    load_entity_extraction_prompt_profile,
    resolve_entity_type_prompt_path,
)
from lightrag.prompt_multimodal import MULTIMODAL_PROMPTS
from lightrag.utils import validate_workspace

from ..utils_api import get_combined_auth_dependency


SENSITIVE_KEY_PARTS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH_ACCOUNTS")
WORKSPACE_LIST_ENV_KEY = "LIGHTRAG_WEBUI_WORKSPACES"
CHUNK_STRATEGIES: dict[str, str] = {
    "F": "Fixed token",
    "R": "Recursive character",
    "V": "Semantic vector",
    "P": "Paragraph semantic",
}
CHUNK_ENV_FIELDS: list[tuple[str, str | None, str]] = [
    ("CHUNK_F_SIZE", None, "Fixed-token chunk size."),
    ("CHUNK_F_OVERLAP_SIZE", None, "Fixed-token overlap size."),
    ("CHUNK_F_SPLIT_BY_CHARACTER", None, "Fixed-token pre-split separator."),
    (
        "CHUNK_F_SPLIT_BY_CHARACTER_ONLY",
        None,
        "Use only the separator for fixed-token splitting.",
    ),
    ("CHUNK_R_SIZE", None, "Recursive-character chunk size."),
    ("CHUNK_R_OVERLAP_SIZE", None, "Recursive-character overlap size."),
    ("CHUNK_R_SEPARATORS", None, "Recursive-character separators JSON array."),
    ("CHUNK_V_SIZE", None, "Semantic-vector chunk size cap."),
    (
        "CHUNK_V_BREAKPOINT_THRESHOLD_TYPE",
        None,
        "Semantic-vector breakpoint threshold type.",
    ),
    (
        "CHUNK_V_BREAKPOINT_THRESHOLD_AMOUNT",
        None,
        "Semantic-vector breakpoint threshold amount.",
    ),
    ("CHUNK_V_BUFFER_SIZE", None, "Semantic-vector sentence buffer size."),
    (
        "CHUNK_V_SENTENCE_SPLIT_REGEX",
        None,
        "Semantic-vector sentence split regex.",
    ),
    ("CHUNK_P_SIZE", None, "Paragraph-semantic chunk size."),
    ("CHUNK_P_OVERLAP_SIZE", None, "Paragraph-semantic overlap size."),
]

PARSER_ENV_FIELDS: list[tuple[str, str | None, str]] = [
    (
        "LIGHTRAG_PARSER",
        None,
        "Default parser routing and process options.",
    ),
    (
        "RAGANYTHING_PATH",
        None,
        "Optional local path to the RAG-Anything repository or package root.",
    ),
    ("RAGANYTHING_PARSER", None, "RAG-Anything parser backend name."),
    ("RAGANYTHING_PARSE_METHOD", None, "RAG-Anything parse method."),
    ("RAGANYTHING_LANG", None, "Optional parser language hint."),
    (
        "RAGANYTHING_PARSE_KWARGS",
        None,
        "JSON object passed to RAG-Anything parser.parse_document.",
    ),
    (
        "LIGHTRAG_FORCE_REPARSE_RAGANYTHING",
        None,
        "Force the RAG-Anything raw bundle cache to be rebuilt.",
    ),
]


ENV_SECTIONS: list[dict[str, Any]] = [
    {
        "id": "workspace",
        "label": "Workspace",
        "description": "Data-isolation and filesystem root settings.",
        "fields": [
            ("WORKSPACE", "workspace", "Current LightRAG workspace."),
            ("WORKING_DIR", "working_dir", "Storage directory for LightRAG data."),
            ("INPUT_DIR", "input_dir", "Input document directory."),
        ],
    },
    {
        "id": "model_roles",
        "label": "Model and Roles",
        "description": "Base LLM and role-specific model limits.",
        "fields": [
            ("LLM_BINDING", "llm_binding", "Base LLM provider binding."),
            ("LLM_MODEL", "llm_model", "Base LLM model name."),
            ("LLM_BINDING_HOST", "llm_binding_host", "Base LLM API host."),
            ("LLM_BINDING_API_KEY", "llm_binding_api_key", "Base LLM API key."),
            ("MAX_ASYNC_LLM", "max_async", "Base LLM concurrency limit."),
            ("LLM_TIMEOUT", "llm_timeout", "Base LLM request timeout."),
            ("KEYWORD_LLM_MODEL", "keyword_llm_model", "Keyword role model."),
            (
                "KEYWORD_MAX_ASYNC_LLM",
                "keyword_llm_max_async",
                "Keyword role concurrency limit.",
            ),
            ("QUERY_LLM_MODEL", "query_llm_model", "Query role model."),
            (
                "QUERY_MAX_ASYNC_LLM",
                "query_llm_max_async",
                "Query role concurrency limit.",
            ),
            ("VLM_PROCESS_ENABLE", "vlm_process_enable", "Enable multimodal analysis."),
            ("VLM_LLM_MODEL", "vlm_llm_model", "VLM role model."),
        ],
    },
    {
        "id": "embedding",
        "label": "Embedding",
        "description": "Embedding provider, model, dimensions, and batching.",
        "fields": [
            ("EMBEDDING_BINDING", "embedding_binding", "Embedding provider binding."),
            ("EMBEDDING_MODEL", "embedding_model", "Embedding model name."),
            (
                "EMBEDDING_BINDING_HOST",
                "embedding_binding_host",
                "Embedding API host.",
            ),
            (
                "EMBEDDING_BINDING_API_KEY",
                "embedding_binding_api_key",
                "Embedding API key.",
            ),
            ("EMBEDDING_DIM", "embedding_dim", "Embedding vector dimension."),
            (
                "EMBEDDING_FUNC_MAX_ASYNC",
                "embedding_func_max_async",
                "Embedding concurrency limit.",
            ),
            ("EMBEDDING_BATCH_NUM", "embedding_batch_num", "Embedding batch size."),
            ("EMBEDDING_TIMEOUT", "embedding_timeout", "Embedding timeout."),
            (
                "EMBEDDING_TOKEN_LIMIT",
                "embedding_token_limit",
                "Maximum tokens per embedding input.",
            ),
        ],
    },
    {
        "id": "retrieval",
        "label": "Retrieval and Rerank",
        "description": "Query budgets, retrieval counts, reranker settings.",
        "fields": [
            ("TOP_K", "top_k", "Entity/relation retrieval count."),
            ("CHUNK_TOP_K", "chunk_top_k", "Chunk retrieval count."),
            (
                "MAX_ENTITY_TOKENS",
                "max_entity_tokens",
                "Entity context token budget.",
            ),
            (
                "MAX_RELATION_TOKENS",
                "max_relation_tokens",
                "Relation context token budget.",
            ),
            ("MAX_TOTAL_TOKENS", "max_total_tokens", "Total context token budget."),
            ("COSINE_THRESHOLD", "cosine_threshold", "Vector similarity threshold."),
            (
                "RELATED_CHUNK_NUMBER",
                "related_chunk_number",
                "Related chunk count per entity/relation.",
            ),
            ("RERANK_BINDING", "rerank_binding", "Rerank provider binding."),
            ("RERANK_MODEL", "rerank_model", "Rerank model."),
            ("RERANK_BY_DEFAULT", "enable_rerank", "Enable rerank by default."),
            ("MIN_RERANK_SCORE", "min_rerank_score", "Minimum rerank score."),
            (
                "MAX_ASYNC_RERANK",
                "rerank_max_async",
                "Rerank concurrency limit.",
            ),
            ("RERANK_TIMEOUT", "rerank_timeout", "Rerank timeout."),
        ],
    },
    {
        "id": "parser",
        "label": "Parser",
        "description": "Document parser engine routing and parser plugin settings.",
        "fields": PARSER_ENV_FIELDS,
    },
    {
        "id": "documents",
        "label": "Documents and Extraction",
        "description": "Upload, chunking, extraction, and summary limits.",
        "fields": [
            ("MAX_UPLOAD_SIZE", "max_upload_size", "Maximum upload size in bytes."),
            (
                "MAX_PARALLEL_INSERT",
                "max_parallel_insert",
                "Parallel document insertion limit.",
            ),
            ("CHUNK_SIZE", "chunk_size", "Default chunk token size."),
            (
                "CHUNK_OVERLAP_SIZE",
                "chunk_overlap_size",
                "Default chunk overlap size.",
            ),
            *CHUNK_ENV_FIELDS,
            (
                "ENTITY_TYPE_PROMPT_FILE",
                None,
                "Entity-type prompt profile file name.",
            ),
            ("PROMPT_DIR", None, "Prompt profile root directory."),
            ("SUMMARY_LANGUAGE", "summary_language", "Summary language."),
            (
                "SUMMARY_MAX_TOKENS",
                "summary_max_tokens",
                "Entity/relation summary token limit.",
            ),
            (
                "SUMMARY_CONTEXT_SIZE",
                "summary_context_size",
                "Summary context size.",
            ),
            (
                "MAX_EXTRACT_INPUT_TOKENS",
                None,
                "Maximum extraction prompt input tokens.",
            ),
            (
                "MAX_EXTRACTION_RECORDS",
                None,
                "Maximum extraction output records.",
            ),
            (
                "MAX_EXTRACTION_ENTITIES",
                None,
                "Maximum extraction entities.",
            ),
        ],
    },
    {
        "id": "storage",
        "label": "Storage",
        "description": "Storage backends and selected backend limits.",
        "fields": [
            ("LIGHTRAG_KV_STORAGE", "kv_storage", "KV storage backend."),
            (
                "LIGHTRAG_VECTOR_STORAGE",
                "vector_storage",
                "Vector storage backend.",
            ),
            ("LIGHTRAG_GRAPH_STORAGE", "graph_storage", "Graph storage backend."),
            (
                "LIGHTRAG_DOC_STATUS_STORAGE",
                "doc_status_storage",
                "Document status storage backend.",
            ),
            (
                "POSTGRES_MAX_CONNECTIONS",
                None,
                "PostgreSQL maximum connection count.",
            ),
            (
                "NEO4J_MAX_CONNECTION_POOL_SIZE",
                None,
                "Neo4j maximum connection pool size.",
            ),
            ("REDIS_MAX_CONNECTIONS", None, "Redis maximum connection count."),
        ],
    },
    {
        "id": "security",
        "label": "Security",
        "description": "Authentication and session settings.",
        "fields": [
            ("AUTH_ACCOUNTS", "auth_accounts", "Configured username/password entries."),
            ("LIGHTRAG_API_KEY", "key", "API key for server authentication."),
            ("TOKEN_EXPIRE_HOURS", "token_expire_hours", "User token expiry."),
            (
                "GUEST_TOKEN_EXPIRE_HOURS",
                "guest_token_expire_hours",
                "Guest token expiry.",
            ),
            ("TOKEN_AUTO_RENEW", "token_auto_renew", "Enable token renewal."),
            (
                "TOKEN_RENEW_THRESHOLD",
                "token_renew_threshold",
                "Token renewal threshold ratio.",
            ),
        ],
    },
]

ALLOWED_ENV_KEYS = {
    key for section in ENV_SECTIONS for key, _attr, _description in section["fields"]
} | {WORKSPACE_LIST_ENV_KEY}


class EnvUpdateRequest(BaseModel):
    values: dict[str, str | None] = Field(default_factory=dict)


class FolderPickRequest(BaseModel):
    initial_dir: str | None = None


class EntityPromptUpdateRequest(BaseModel):
    profile: str
    entity_types_guidance: str

    @field_validator("profile", "entity_types_guidance", mode="after")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value


def _is_sensitive(key: str) -> bool:
    return any(part in key for part in SENSITIVE_KEY_PARTS)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _runtime_value(args: Any, attr: str | None) -> str:
    if not attr:
        return ""
    return _stringify(getattr(args, attr, ""))


def _read_env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        key: value or ""
        for key, value in dotenv_values(path).items()
        if key is not None
    }


def _pick_directory_with_shell_dialog(initial_dir: str | None = None) -> str | None:
    if os.name != "nt":
        raise RuntimeError("Folder picker is only available on Windows desktop.")

    script = r"""
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = 'Select LightRAG workspace folder'
$dialog.ShowNewFolderButton = $true
if ($env:LIGHTRAG_PICKER_INITIAL_DIR -and (Test-Path -LiteralPath $env:LIGHTRAG_PICKER_INITIAL_DIR)) {
    $dialog.SelectedPath = $env:LIGHTRAG_PICKER_INITIAL_DIR
}
$result = $dialog.ShowDialog()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Out.WriteLine($dialog.SelectedPath)
}
"""
    env = os.environ.copy()
    if initial_dir:
        env["LIGHTRAG_PICKER_INITIAL_DIR"] = initial_dir

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())

    selected = completed.stdout.strip()
    return selected or None


def _env_or_runtime_value(
    args: Any, env_values: dict[str, str], key: str, attr: str | None = None
) -> str:
    if key in env_values:
        return env_values[key]
    return os.getenv(key, _runtime_value(args, attr))


def _resolve_config_path(cwd: Path, value: str) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return str(path)


def _workspace_child_path(root: str, workspace: str) -> str:
    if not root:
        return ""
    path = Path(root)
    return str(path / workspace) if workspace else str(path)


def _split_workspace_names(raw: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        name = item.strip()
        if not name or name in seen:
            continue
        validate_workspace(name)
        names.append(name)
        seen.add(name)
    return names


def _discover_workspace_dir_names(*roots: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root:
            continue
        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            continue
        for child in sorted(root_path.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir() or child.name in seen:
                continue
            try:
                validate_workspace(child.name)
            except ValueError:
                continue
            names.append(child.name)
            seen.add(child.name)
    return names


def _workspace_payload(rag: Any, args: Any, env_path: Path) -> dict[str, Any]:
    cwd = Path.cwd()
    env_values = _read_env_values(env_path)
    current = (
        env_values.get("WORKSPACE")
        or getattr(rag, "workspace", None)
        or getattr(args, "workspace", "")
        or ""
    )
    if current:
        validate_workspace(current)

    working_dir = _resolve_config_path(
        cwd, _env_or_runtime_value(args, env_values, "WORKING_DIR", "working_dir")
    )
    input_dir = _resolve_config_path(
        cwd, _env_or_runtime_value(args, env_values, "INPUT_DIR", "input_dir")
    )

    managed_raw = env_values.get(WORKSPACE_LIST_ENV_KEY, "")
    managed_names = (
        _split_workspace_names(managed_raw)
        if WORKSPACE_LIST_ENV_KEY in env_values
        else _discover_workspace_dir_names(working_dir, input_dir)
    )
    if current and current not in managed_names:
        managed_names.insert(0, current)

    return {
        "current": current,
        "dynamic_switching": False,
        "working_dir": working_dir,
        "input_dir": input_dir,
        "managed_env_key": WORKSPACE_LIST_ENV_KEY,
        "available": [
            {
                "name": name,
                "active": name == current,
                "working_path": _workspace_child_path(working_dir, name),
                "input_path": _workspace_child_path(input_dir, name),
                "working_exists": Path(_workspace_child_path(working_dir, name)).exists()
                if working_dir
                else False,
                "input_exists": Path(_workspace_child_path(input_dir, name)).exists()
                if input_dir
                else False,
            }
            for name in managed_names
        ],
    }


def _active_chunk_strategy(args: Any, env_path: Path) -> str:
    env_values = _read_env_values(env_path)
    parser_rules = _env_or_runtime_value(args, env_values, "LIGHTRAG_PARSER")
    for rule in parser_rules.replace(";", ",").split(","):
        if ":" not in rule:
            continue
        target = rule.split(":", 1)[1]
        if "-" not in target:
            continue
        options = target.split("-", 1)[1]
        for char in options:
            if char in CHUNK_STRATEGIES:
                return char
    return "F"


def _chunking_payload(args: Any, env_path: Path) -> dict[str, Any]:
    return {
        "active_strategy": _active_chunk_strategy(args, env_path),
        "strategies": [
            {"key": key, "label": label}
            for key, label in CHUNK_STRATEGIES.items()
        ],
    }


def _discover_env_profiles(cwd: Path, active_profile: str = ".env") -> list[dict[str, Any]]:
    candidates: list[Path] = [cwd / ".env"]
    for path in cwd.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name == ".env":
            continue
        if name == ".env" or name == "env.example":
            candidates.append(path)
        elif name.startswith(".env.") or name.startswith("env."):
            candidates.append(path)

    def sort_key(path: Path) -> tuple[int, str]:
        priority = {".env": 0, "env.example": 1}.get(path.name, 2)
        return priority, path.name

    return [
        {
            "name": path.name,
            "read_only": path.name != ".env",
            "active": path.name == active_profile,
            "exists": path.exists(),
        }
        for path in sorted(candidates, key=sort_key)
    ]


def _env_sections(args: Any, env_path: Path, *, editable: bool) -> list[dict[str, Any]]:
    file_values = _read_env_values(env_path)
    sections: list[dict[str, Any]] = []

    for section in ENV_SECTIONS:
        fields = []
        for key, attr, description in section["fields"]:
            runtime = os.getenv(key, _runtime_value(args, attr))
            raw_value = file_values.get(key, runtime)
            sensitive = _is_sensitive(key)
            configured = bool(file_values.get(key) or runtime)
            fields.append(
                {
                    "key": key,
                    "label": key,
                    "description": description,
                    "value": "" if sensitive else _stringify(raw_value),
                    "runtime_value": "" if sensitive else _stringify(runtime),
                    "configured": configured,
                    "sensitive": sensitive,
                    "editable": editable,
                    "requires_restart": True,
                    "source": "file" if key in file_values else "runtime",
                }
            )
        sections.append(
            {
                "id": section["id"],
                "label": section["label"],
                "description": section["description"],
                "fields": fields,
            }
        )

    return sections


def _entity_prompt_profiles() -> list[dict[str, Any]]:
    prompt_dir = get_entity_type_prompt_dir()
    if not prompt_dir.exists():
        return []
    profiles = []
    for path in sorted(prompt_dir.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.suffix.lower() in {".yml", ".yaml"}:
            profiles.append({"name": path.name, "read_only": False})
    return profiles


def _active_entity_prompt_file(rag: Any) -> str:
    addon_params = getattr(rag, "addon_params", {}) or {}
    return str(
        addon_params.get("entity_type_prompt_file")
        or os.getenv("ENTITY_TYPE_PROMPT_FILE", "")
    ).strip()


def _entity_prompt_content(active_profile: str) -> str:
    if active_profile:
        try:
            profile = load_entity_extraction_prompt_profile(
                resolve_entity_type_prompt_path(active_profile)
            )
            guidance = profile.get("entity_types_guidance")
            if guidance:
                return str(guidance)
        except (FileNotFoundError, OSError, ValueError):
            pass
    return get_default_entity_extraction_prompt_profile()["entity_types_guidance"]


def _resolve_existing_entity_prompt_profile(prompt_profile: str) -> str:
    try:
        profile_path = resolve_entity_type_prompt_path(prompt_profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not profile_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown prompt profile '{prompt_profile}'.",
        )
    return profile_path.name


def _prompt_stages(active_profile: str) -> list[dict[str, Any]]:
    return [
        {
            "key": "entity_type",
            "label": "Entity relationship extraction",
            "field": "entity_types_guidance",
            "profile": active_profile,
            "editable": True,
            "content": _entity_prompt_content(active_profile),
        },
        {
            "key": "keyword",
            "label": "Keyword extraction",
            "field": "keywords_extraction",
            "editable": False,
            "content": PROMPTS["keywords_extraction"],
        },
        {
            "key": "query",
            "label": "Query answer",
            "field": "rag_response",
            "editable": False,
            "content": PROMPTS["rag_response"],
        },
        {
            "key": "vlm",
            "label": "Multimodal analysis",
            "field": "image_analysis",
            "editable": False,
            "content": MULTIMODAL_PROMPTS["image_analysis"],
        },
    ]


def _resolve_env_profile_path(cwd: Path, env_profile: str) -> Path:
    profiles = {profile["name"] for profile in _discover_env_profiles(cwd)}
    if env_profile not in profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown env profile '{env_profile}'.",
        )
    return cwd / env_profile


def _workbench_payload(
    rag: Any,
    args: Any,
    *,
    env_profile: str = ".env",
    prompt_profile: str | None = None,
) -> dict[str, Any]:
    cwd = Path.cwd()
    env_path = _resolve_env_profile_path(cwd, env_profile)
    env_editable = env_profile == ".env"
    entity_prompt_profile = (
        _resolve_existing_entity_prompt_profile(prompt_profile)
        if prompt_profile
        else _active_entity_prompt_file(rag)
    )
    return {
        "workspace": _workspace_payload(rag, args, env_path),
        "chunking": _chunking_payload(args, env_path),
        "env": {
            "active_profile": env_profile,
            "profiles": _discover_env_profiles(cwd, env_profile),
            "sections": _env_sections(args, env_path, editable=env_editable),
        },
        "prompts": {
            "entity_type_active_profile": entity_prompt_profile,
            "entity_type_profiles": _entity_prompt_profiles(),
            "stages": _prompt_stages(entity_prompt_profile),
        },
        "requires_restart": True,
    }


def create_config_routes(
    rag: Any, args: Any, api_key: Optional[str] = None
) -> APIRouter:
    router = APIRouter(tags=["config"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/config/workbench", dependencies=[Depends(combined_auth)])
    async def get_config_workbench(
        env_profile: str = Query(".env", description="Env file profile to inspect"),
        prompt_profile: str | None = Query(
            None,
            description="Entity-type prompt profile file to inspect",
        ),
    ):
        return _workbench_payload(
            rag,
            args,
            env_profile=env_profile,
            prompt_profile=prompt_profile,
        )

    @router.post("/config/workbench/folders/pick", dependencies=[Depends(combined_auth)])
    async def pick_workspace_folder(request: FolderPickRequest):
        try:
            selected = _pick_directory_with_shell_dialog(request.initial_dir)
        except RuntimeError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

        if not selected:
            return {"selected_path": None}

        selected_path = Path(selected).resolve()
        workspace = selected_path.name
        if not workspace:
            raise HTTPException(
                status_code=400,
                detail="Selected folder must have a workspace directory name.",
            )
        try:
            validate_workspace(workspace)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "selected_path": str(selected_path),
            "workspace": workspace,
            "input_dir": str(selected_path.parent),
        }

    @router.put("/config/workbench/env", dependencies=[Depends(combined_auth)])
    async def update_env_config(request: EnvUpdateRequest):
        unknown = sorted(set(request.values) - ALLOWED_ENV_KEYS)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported configuration key(s): {', '.join(unknown)}",
            )
        try:
            if "WORKSPACE" in request.values and request.values["WORKSPACE"]:
                validate_workspace(str(request.values["WORKSPACE"]))
            if WORKSPACE_LIST_ENV_KEY in request.values:
                _split_workspace_names(str(request.values[WORKSPACE_LIST_ENV_KEY] or ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        env_path = Path.cwd() / ".env"
        env_path.touch(exist_ok=True)
        current_values = _read_env_values(env_path)
        updated: list[str] = []

        for key, value in request.values.items():
            if _is_sensitive(key) and (value is None or value == ""):
                continue
            value_to_write = "" if value is None else str(value)
            if current_values.get(key) == value_to_write:
                continue
            set_key(str(env_path), key, value_to_write, quote_mode="never")
            updated.append(key)

        return {
            "updated": updated,
            "requires_restart": True,
            "workbench": _workbench_payload(rag, args),
        }

    @router.put(
        "/config/workbench/prompts/entity-type",
        dependencies=[Depends(combined_auth)],
    )
    async def update_entity_type_prompt(request: EntityPromptUpdateRequest):
        try:
            profile_path = resolve_entity_type_prompt_path(request.profile)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        profile_path.parent.mkdir(parents=True, exist_ok=True)
        default_profile = get_default_entity_extraction_prompt_profile()
        raw_profile: dict[str, Any] = {}

        if profile_path.exists():
            try:
                loaded = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Existing prompt profile contains invalid YAML: {exc}",
                ) from exc
            if loaded is not None and not isinstance(loaded, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Existing prompt profile must contain a YAML mapping.",
                )
            raw_profile = loaded or {}

        raw_profile["entity_types_guidance"] = request.entity_types_guidance
        raw_profile.setdefault(
            "entity_extraction_examples",
            default_profile["entity_extraction_examples"],
        )
        raw_profile.setdefault(
            "entity_extraction_json_examples",
            default_profile["entity_extraction_json_examples"],
        )

        profile_path.write_text(
            yaml.safe_dump(raw_profile, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        return {
            "profile": profile_path.name,
            "requires_restart": True,
            "workbench": _workbench_payload(
                rag,
                args,
                prompt_profile=profile_path.name,
            ),
        }

    return router
