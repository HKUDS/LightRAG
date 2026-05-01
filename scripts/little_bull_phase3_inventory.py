#!/usr/bin/env python3
"""Inventory non-production Little Bull Phase 3 pilot artifacts.

This script is read-only. It prints artifact names for an isolated pilot
workspace so a human can approve any later cleanup precisely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any


EMBEDDING_DIM = 16
WORKSPACE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{2,79}$")
POSTGRES_TABLES = (
    "lightrag_doc_full",
    "lightrag_doc_chunks",
    "lightrag_llm_cache",
    "lightrag_doc_status",
    "lightrag_full_entities",
    "lightrag_full_relations",
    "lightrag_entity_chunks",
    "lightrag_relation_chunks",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only inventory for Little Bull Phase 3 pilot artifacts."
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--working-dir",
        default=None,
        help="Optional working dir to include in the inventory.",
    )
    return parser


def _safe_workspace(value: str) -> str:
    if not WORKSPACE_PATTERN.fullmatch(value):
        raise ValueError(
            "Workspace must be 3-80 characters and contain only letters, numbers, "
            "underscore, or hyphen."
        )
    return value


def _model_suffix(workspace: str) -> str:
    model_name = f"phase3-fake-local-{workspace}".lower()
    safe_model = re.sub(r"[^a-zA-Z0-9_]", "_", model_name)
    return f"{safe_model}_{EMBEDDING_DIM}d"


def expected_artifacts(
    workspace: str, working_dir: str | None = None
) -> dict[str, Any]:
    safe_workspace = _safe_workspace(workspace)
    suffix = _model_suffix(safe_workspace)
    return {
        "workspace": safe_workspace,
        "working_dir": working_dir,
        "postgres": {
            "workspace": safe_workspace,
            "tables": list(POSTGRES_TABLES),
        },
        "qdrant": {
            "collections": [
                f"lightrag_vdb_{namespace}_{suffix}"
                for namespace in ("entities", "relationships", "chunks")
            ]
        },
        "neo4j": {
            "workspace_label": safe_workspace,
            "fulltext_index": f"entity_id_fulltext_idx_{safe_workspace.replace('-', '_')}",
        },
    }


async def _qdrant_existing(expected: dict[str, Any]) -> list[str]:
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        return []
    import httpx

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{qdrant_url.rstrip('/')}/collections")
        response.raise_for_status()
    names = {
        item["name"]
        for item in response.json().get("result", {}).get("collections", [])
        if isinstance(item, dict) and item.get("name")
    }
    return [name for name in expected["qdrant"]["collections"] if name in names]


async def _postgres_counts(workspace: str) -> dict[str, int]:
    required = (
        "POSTGRES_HOST",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DATABASE",
    )
    if any(not os.getenv(name) for name in required):
        return {}
    try:
        import asyncpg
    except Exception:
        return {}
    connection = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DATABASE"),
    )
    try:
        counts: dict[str, int] = {}
        for table in POSTGRES_TABLES:
            exists = await connection.fetchval("SELECT to_regclass($1)", table)
            if exists:
                counts[table] = int(
                    await connection.fetchval(
                        f"SELECT count(*) FROM {table} WHERE workspace=$1", workspace
                    )
                    or 0
                )
        return counts
    finally:
        await connection.close()


async def _neo4j_indexes(expected: dict[str, Any]) -> list[str]:
    required = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")
    if any(not os.getenv(name) for name in required):
        return []
    try:
        from neo4j import AsyncGraphDatabase
    except Exception:
        return []
    driver = AsyncGraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        async with driver.session(
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        ) as session:
            result = await session.run(
                "SHOW INDEXES YIELD name WHERE name = $name RETURN name",
                name=expected["neo4j"]["fulltext_index"],
            )
            return [record["name"] async for record in result]
    finally:
        await driver.close()


async def inventory(workspace: str, working_dir: str | None = None) -> dict[str, Any]:
    expected = expected_artifacts(workspace, working_dir)
    observed = {
        "qdrant_collections": await _qdrant_existing(expected),
        "postgres_row_counts": await _postgres_counts(expected["workspace"]),
        "neo4j_indexes": await _neo4j_indexes(expected),
    }
    return {"expected": expected, "observed": observed, "destructive_actions": []}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = asyncio.run(inventory(args.workspace, args.working_dir))
    except Exception as exc:
        print(f"Phase 3 inventory failed: {exc.__class__.__name__}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
