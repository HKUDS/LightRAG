#!/usr/bin/env python3
"""Non-destructive Little Bull Phase 3 data-plane pilot.

This harness is intentionally opt-in. It initializes a timestamped workspace
against PGKVStorage, PGDocStatusStorage, Neo4JStorage, and QdrantVectorDBStorage
without enabling Little Bull feature flags or printing credential values.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import numpy as np


TRUE_VALUES = {"1", "true", "yes", "on"}
EMBEDDING_DIM = 16
STORAGE_CONTRACT = {
    "kv_storage": "PGKVStorage",
    "doc_status_storage": "PGDocStatusStorage",
    "graph_storage": "Neo4JStorage",
    "vector_storage": "QdrantVectorDBStorage",
}
REQUIRED_ENV_VARS = (
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DATABASE",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "QDRANT_URL",
)
QDRANT_MAX_MINOR_VERSION_DELTA = 1
FORBIDDEN_STORAGE_WORKSPACE_ENVS = (
    "POSTGRES_WORKSPACE",
    "NEO4J_WORKSPACE",
    "QDRANT_WORKSPACE",
)
WORKSPACE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{2,79}$")


class PilotConfigError(RuntimeError):
    """Raised when the pilot is not explicitly and safely configured."""


@dataclass(frozen=True)
class Phase3PilotConfig:
    workspace: str
    working_dir: Path
    mode: str
    query_mode: str
    allow_storage_workspace_env: bool = False
    allow_neo4j_no_auth: bool = False
    allow_qdrant_version_mismatch: bool = False


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def _load_dotenv_without_echo() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(".env", override=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the opt-in Little Bull Phase 3 data-plane pilot."
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Pilot workspace id. Defaults to a new phase3-<uuid> workspace.",
    )
    parser.add_argument(
        "--working-dir",
        default=None,
        help="Local LightRAG working directory. Defaults to /tmp/trag-lightrag-phase3/<workspace>.",
    )
    parser.add_argument(
        "--mode",
        choices=("init", "insert-query"),
        default="init",
        help="Pilot depth: initialize/finalize only, or insert plus naive source retrieval.",
    )
    parser.add_argument(
        "--query-mode",
        choices=("naive",),
        default="naive",
        help="Retrieval mode used by insert-query. Only naive avoids external LLM generation.",
    )
    parser.add_argument(
        "--allow-storage-workspace-env",
        action="store_true",
        help="Allow storage workspace override env vars for isolated subprocess experiments.",
    )
    return parser


def build_config(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str | None] = os.environ,
) -> Phase3PilotConfig:
    args = _parser().parse_args(argv)
    if not _truthy(env.get("LITTLE_BULL_PHASE3_PILOT")):
        raise PilotConfigError("Set LITTLE_BULL_PHASE3_PILOT=1 to run this pilot.")

    forbidden = [
        name
        for name in FORBIDDEN_STORAGE_WORKSPACE_ENVS
        if (env.get(name) or "").strip()
    ]
    if forbidden and not args.allow_storage_workspace_env:
        names = ", ".join(sorted(forbidden))
        raise PilotConfigError(
            "Unset storage workspace override env vars before running the pilot: "
            f"{names}."
        )

    missing = [name for name in REQUIRED_ENV_VARS if not (env.get(name) or "").strip()]
    if missing:
        raise PilotConfigError(
            "Missing required data-plane env vars: " + ", ".join(sorted(missing)) + "."
        )

    workspace = args.workspace or f"phase3-{uuid4().hex[:12]}"
    if not WORKSPACE_PATTERN.fullmatch(workspace):
        raise PilotConfigError(
            "Workspace must be 3-80 characters and contain only letters, numbers, "
            "underscore, or hyphen."
        )

    working_dir = (
        Path(args.working_dir)
        if args.working_dir
        else Path(tempfile.gettempdir()) / "trag-lightrag-phase3" / workspace
    )
    return Phase3PilotConfig(
        workspace=workspace,
        working_dir=working_dir,
        mode=args.mode,
        query_mode=args.query_mode,
        allow_storage_workspace_env=args.allow_storage_workspace_env,
        allow_neo4j_no_auth=_truthy(env.get("LITTLE_BULL_PHASE3_ALLOW_NEO4J_NO_AUTH")),
        allow_qdrant_version_mismatch=_truthy(
            env.get("LITTLE_BULL_PHASE3_ALLOW_QDRANT_VERSION_MISMATCH")
        ),
    )


async def _fake_llm(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    **_: object,
) -> str:
    del prompt, system_prompt, history_messages
    return "<|COMPLETE|>"


async def _fake_embedding(texts: list[str], **_: object) -> np.ndarray:
    return np.ones((len(texts), EMBEDDING_DIM), dtype=np.float32)


def _redacted_env_status() -> dict[str, str]:
    return {name: "set" if os.getenv(name) else "missing" for name in REQUIRED_ENV_VARS}


def _major_minor(version: str) -> tuple[int, int]:
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError(f"Invalid version: {version}")
    return int(parts[0]), int(parts[1])


def qdrant_versions_are_compatible(client_version: str, server_version: str) -> bool:
    client_major, client_minor = _major_minor(client_version)
    server_major, server_minor = _major_minor(server_version)
    return (
        client_major == server_major
        and abs(client_minor - server_minor) <= QDRANT_MAX_MINOR_VERSION_DELTA
    )


async def _preflight_qdrant_version(config: Phase3PilotConfig) -> str:
    if config.allow_qdrant_version_mismatch:
        return "skipped"
    import httpx

    qdrant_url = os.environ["QDRANT_URL"].rstrip("/")
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{qdrant_url}/")
        response.raise_for_status()
    server_version = str(response.json().get("version", ""))
    client_version = importlib.metadata.version("qdrant-client")
    if not qdrant_versions_are_compatible(client_version, server_version):
        raise PilotConfigError(
            "Qdrant client/server versions are incompatible for the Phase 3 pilot. "
            "Align the local Qdrant image with qdrant-client, or set "
            "LITTLE_BULL_PHASE3_ALLOW_QDRANT_VERSION_MISMATCH=1 for an explicit "
            "diagnostic-only run."
        )
    return f"client={client_version};server={server_version}"


async def _preflight_neo4j_auth(config: Phase3PilotConfig) -> str:
    if config.allow_neo4j_no_auth:
        return "skipped"
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import AuthError

    uri = os.environ["NEO4J_URI"]
    bogus_user = f"phase3_auth_probe_{uuid4().hex}"
    bogus_password = f"phase3_auth_probe_{uuid4().hex}"
    driver = AsyncGraphDatabase.driver(uri, auth=(bogus_user, bogus_password))
    try:
        try:
            await driver.verify_connectivity()
        except AuthError:
            return "required"
        raise PilotConfigError(
            "Neo4j accepted bogus credentials. Enable Neo4j auth before running "
            "the Phase 3 pilot, or set LITTLE_BULL_PHASE3_ALLOW_NEO4J_NO_AUTH=1 "
            "for an explicit diagnostic-only run."
        )
    finally:
        await driver.close()


async def run_pilot(config: Phase3PilotConfig) -> dict[str, object]:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc

    preflight = {
        "qdrant_version": await _preflight_qdrant_version(config),
        "neo4j_auth": await _preflight_neo4j_auth(config),
    }
    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=2048,
        model_name=f"phase3-fake-local-{config.workspace}",
        func=_fake_embedding,
    )
    rag = LightRAG(
        working_dir=str(config.working_dir),
        workspace=config.workspace,
        llm_model_func=_fake_llm,
        embedding_func=embedding_func,
        **STORAGE_CONTRACT,
    )
    summary: dict[str, object] = {
        "workspace": config.workspace,
        "working_dir": str(config.working_dir),
        "mode": config.mode,
        "storages": STORAGE_CONTRACT,
        "env": _redacted_env_status(),
        "preflight": preflight,
        "initialized": False,
        "finalized": False,
    }
    try:
        await rag.initialize_storages()
        summary["initialized"] = True
        if config.mode == "insert-query":
            phrase = f"little bull phase3 source phrase {uuid4().hex}"
            file_path = f"{config.workspace}-pilot.txt"
            track_id = await rag.ainsert(
                phrase,
                ids=f"doc-{uuid4().hex}",
                file_paths=file_path,
                track_id=f"phase3-{uuid4().hex}",
            )
            data = await rag.aquery_data(
                f"What exact phase3 source phrase appears in {file_path}?",
                QueryParam(mode=config.query_mode, top_k=3),
            )
            data_section = data.get("data", {}) if isinstance(data, dict) else {}
            chunks = data_section.get("chunks", []) if isinstance(data_section, dict) else []
            references = (
                data_section.get("references", [])
                if isinstance(data_section, dict)
                else []
            )
            source_verified = any(phrase in str(chunk) for chunk in chunks)
            summary.update(
                {
                    "track_id": track_id,
                    "chunk_count": len(chunks),
                    "reference_count": len(references),
                    "source_verified": source_verified,
                }
            )
            if not source_verified:
                raise RuntimeError("Inserted pilot source was not retrieved.")
        return summary
    finally:
        await rag.finalize_storages()
        summary["finalized"] = True


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_without_echo()
    try:
        config = build_config(argv)
        summary = asyncio.run(run_pilot(config))
    except PilotConfigError as exc:
        print(f"Little Bull Phase 3 pilot configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            "Little Bull Phase 3 pilot failed: "
            f"{exc.__class__.__name__}. Secret values are intentionally not echoed.",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
