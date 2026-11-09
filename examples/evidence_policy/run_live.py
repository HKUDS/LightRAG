#!/usr/bin/env python
"""F01-F04 through the real path: LightRAG retrieval -> version resolution ->
context -> ordinary model answer.

This is the application entry for the limited feature demonstration. It adds no new
API: ingestion uses ``LightRAG.ainsert``, the answer uses
``LightRAG.aquery_llm``, and version selection goes through the existing
``lightrag.evidence_policy`` resolver via ``QueryParam.evidence_policy``.

What is real here:

* storages: ``JsonKVStorage`` + ``NanoVectorDBStorage`` + ``NetworkXStorage``
  on disk -- the shipped defaults, no fake readers;
* retrieval: the chunk vector index is queried by the shipped ``naive`` path;
* catalog: chunk IDs, document IDs and span hashes are read back from
  ``text_chunks`` *after* ingestion (``build_catalog``), never invented;
* generation: whatever OpenAI-compatible endpoint ``LIGHTRAG_*`` names, with
  the answer cache disabled so a cached answer cannot hide the policy.

What is deliberately controlled, and reported as such:

* the corpus is two fictional documents carrying one 180 -> 270 revision;
* F02 removes the revision's *vector* entry so retrieval cannot recall it --
  the "missed amendment" case. Its ``text_chunks`` row and its catalog entry
  stay in place, so the resolver has something real to fetch by ID. On a
  two-document corpus this removal is the only way to make the case
  reproducible instead of hoping the retriever misses it.

Usage::

    export LIGHTRAG_LLM_API_KEY=sk-...          # DeepSeek key; the only paid leg

    .venv/bin/python \
        examples/evidence_policy/run_live.py \
        --workdir temp/evidence-policy-demo --out temp/evidence-policy-demo/record.json

Defaults: ``LIGHTRAG_LLM_BASE_URL=https://api.deepseek.com``,
``LIGHTRAG_LLM_MODEL=deepseek-flash``, and the embedding leg runs locally with
``LIGHTRAG_EMBED_BACKEND=fastembed`` on ``BAAI/bge-small-zh-v1.5`` (512 dims,
ONNX, no key). Set ``LIGHTRAG_EMBED_BACKEND=openai`` plus
``LIGHTRAG_EMBED_BASE_URL`` / ``LIGHTRAG_EMBED_MODEL`` / ``LIGHTRAG_EMBED_API_KEY``
to use a hosted embedding API instead.

``--preflight-only`` checks the endpoints without ingesting anything.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_3037 as corpus  # noqa: E402

from lightrag import LightRAG, QueryParam  # noqa: E402
from lightrag.evidence_policy import catalog_from_mapping  # noqa: E402
from lightrag.kg.shared_storage import initialize_pipeline_status  # noqa: E402
from lightrag.llm.openai import openai_complete_if_cache, openai_embed  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

POLICY = "authority-v1"
WORKSPACE_MAIN = "main"
WORKSPACE_UNRECALLED = "unrecalled"

#: ``expect`` is the number the answer must contain. Whether the *other* number
#: is also acceptable is a reading judgement: "180 was the rule before
#: 2026-07-01" is correct in a 270 answer, so this script does not assert on it.
CASES: list[dict[str, Any]] = [
    {
        "id": "F01",
        "scenario": "原文与已生效修订同时存在",
        "workspace": WORKSPACE_MAIN,
        "as_of": "2026-08-01",
        "expect": "270",
        "require_seed": ["V-old", "V-new"],
        "expect_selected": ["V-new"],
    },
    {
        "id": "F02",
        "scenario": "初始seed只含旧文，修订已入库且目录已登记",
        "workspace": WORKSPACE_UNRECALLED,
        "as_of": "2026-08-01",
        "expect": "270",
        "require_seed": ["V-old"],
        "forbid_seed": ["V-new"],
        "expect_expanded": ["V-new"],
        "expect_selected": ["V-new"],
    },
    {
        "id": "F03",
        "scenario": "查询修订前的历史日期",
        "workspace": WORKSPACE_MAIN,
        "as_of": "2026-03-01",
        "expect": "180",
        "expect_selected": ["V-old"],
    },
    {
        "id": "F04-pre",
        "scenario": "修订已公布（2026-05-20）但尚未生效",
        "workspace": WORKSPACE_MAIN,
        "as_of": "2026-06-15",
        "expect": "180",
        "expect_selected": ["V-old"],
    },
    {
        "id": "F04-post",
        "scenario": "同一修订生效之后",
        "workspace": WORKSPACE_MAIN,
        "as_of": "2026-08-01",
        "expect": "270",
        "expect_selected": ["V-new"],
    },
]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provider:
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    embed_backend: str
    embed_base_url: str
    embed_model: str
    embed_api_key: str
    embed_dim: int | None

    @classmethod
    def from_env(cls) -> "Provider":
        def need(name: str) -> str:
            value = os.environ.get(name, "").strip()
            if not value:
                raise SystemExit(
                    f"missing {name}: set LIGHTRAG_LLM_BASE_URL / "
                    "LIGHTRAG_LLM_MODEL / LIGHTRAG_LLM_API_KEY (see module "
                    "docstring); the default embedding backend runs locally"
                )
            return value

        backend = os.environ.get("LIGHTRAG_EMBED_BACKEND", "fastembed").strip().lower()
        if backend not in ("fastembed", "openai"):
            raise SystemExit(
                f"LIGHTRAG_EMBED_BACKEND must be 'fastembed' or 'openai', got {backend!r}"
            )
        llm_base = os.environ.get(
            "LIGHTRAG_LLM_BASE_URL", "https://api.deepseek.com"
        ).rstrip("/")
        llm_key = need("LIGHTRAG_LLM_API_KEY")
        dim_raw = os.environ.get("LIGHTRAG_EMBED_DIM", "").strip()

        if backend == "fastembed":
            embed_base = ""
            embed_model = os.environ.get(
                "LIGHTRAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5"
            ).strip()
            embed_key = ""
        else:
            embed_base = os.environ.get("LIGHTRAG_EMBED_BASE_URL", llm_base).rstrip("/")
            embed_model = need("LIGHTRAG_EMBED_MODEL")
            embed_key = os.environ.get("LIGHTRAG_EMBED_API_KEY", llm_key)

        return cls(
            llm_base_url=llm_base,
            llm_model=os.environ.get("LIGHTRAG_LLM_MODEL", "deepseek-flash").strip(),
            llm_api_key=llm_key,
            embed_backend=backend,
            embed_base_url=embed_base,
            embed_model=embed_model,
            embed_api_key=embed_key,
            embed_dim=int(dim_raw) if dim_raw else None,
        )

    def public(self) -> dict[str, Any]:
        return {
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "embed_backend": self.embed_backend,
            "embed_base_url": self.embed_base_url or None,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
        }


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


_LOCAL_MODELS: dict[str, Any] = {}


def _local_embedder(model_name: str):
    """BAAI/bge-small-zh-v1.5 through fastembed: ONNX, no torch, no API key.

    Asymmetric: BGE expects a retrieval instruction on the query side and none
    on the document side, which is what ``query_embed`` / ``embed`` differ on.
    The loaded model is cached by name because a workspace finalizes its own
    ``EmbeddingFunc`` wrapper, not the model.
    """
    import numpy as np
    from fastembed import TextEmbedding

    model = _LOCAL_MODELS.get(model_name)
    if model is None:
        model = TextEmbedding(model_name)
        _LOCAL_MODELS[model_name] = model

    def run(texts, context: str = "document"):
        batch = list(texts)
        if not batch:
            return np.zeros((0, 512), dtype=float)
        if context == "query":
            return np.array(list(model.query_embed(batch)), dtype=float)
        return np.array(list(model.embed(batch)), dtype=float)

    return run


async def build_embedding_func(provider: Provider, dim: int) -> EmbeddingFunc:
    if provider.embed_backend == "fastembed":
        run = _local_embedder(provider.embed_model)

        async def embed(texts, context: str = "document"):
            return await asyncio.to_thread(run, texts, context)

        return EmbeddingFunc(
            embedding_dim=dim,
            max_token_size=512,
            func=embed,
            supports_asymmetric=True,
            model_name=provider.embed_model,
        )

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=provider.embed_model,
            base_url=provider.embed_base_url,
            api_key=provider.embed_api_key,
        ),
        supports_asymmetric=True,
        model_name=provider.embed_model,
    )


async def preflight(provider: Provider, *, check_chat: bool = True) -> dict[str, Any]:
    """Probe embedding dimensions and optionally check chat connectivity."""
    from openai import AsyncOpenAI

    chat_text = None
    if check_chat:
        client = AsyncOpenAI(
            base_url=provider.llm_base_url,
            api_key=provider.llm_api_key,
            max_retries=0,
            timeout=30,
        )
        try:
            completion = await client.chat.completions.create(
                model=provider.llm_model,
                messages=[{"role": "user", "content": "reply with OK"}],
                max_tokens=16,
                temperature=0,
                extra_body={"thinking": {"type": "disabled"}},
            )
            chat_text = (completion.choices[0].message.content or "").strip()
        finally:
            await client.close()

    if provider.embed_backend == "fastembed":
        run = _local_embedder(provider.embed_model)
        vectors = await asyncio.to_thread(
            run, ["预检：标准成员办理申请的期限是多少天？"], "query"
        )
    else:
        vectors = await openai_embed.func(
            ["预检：标准成员办理申请的期限是多少天？"],
            model=provider.embed_model,
            base_url=provider.embed_base_url,
            api_key=provider.embed_api_key,
        )
    dim = int(vectors.shape[1]) if hasattr(vectors, "shape") else len(vectors[0])
    if provider.embed_dim is not None and provider.embed_dim != dim:
        raise SystemExit(
            f"LIGHTRAG_EMBED_DIM={provider.embed_dim} but the backend returned {dim}"
        )
    return {"chat_reply": chat_text, "chat_calls": int(check_chat), "embed_dim": dim}


# ---------------------------------------------------------------------------
# Model recorder
# ---------------------------------------------------------------------------


def make_recorder(inner):
    """Return ``(recorder, state)``.

    A plain async function, not a callable object: ``_build_global_config``
    deep-copies the config, and a function is atomic under ``deepcopy`` while an
    instance is not. A class-based recorder would be copied per request and the
    calls would land in the copy.
    """
    state: dict[str, Any] = {"calls": [], "all_calls": []}

    async def recorder(prompt, system_prompt=None, **kwargs):
        result = await inner(prompt, system_prompt=system_prompt, **kwargs)
        call = {
            "prompt": prompt,
            "system_prompt": system_prompt or "",
            "response": result if isinstance(result, str) else None,
        }
        state["calls"].append(call)
        state["all_calls"].append(call)
        return result

    return recorder, state


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def make_rag(
    workdir: Path, provider: Provider, llm_func, embedding: EmbeddingFunc
) -> LightRAG:
    return LightRAG(
        working_dir=str(workdir),
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
        llm_model_func=llm_func,
        llm_model_name=provider.llm_model,
        embedding_func=embedding,
        # Keep the answer cache off: a cached answer would
        # hide whether the policy actually ran for this request.
        enable_llm_cache=False,
        evidence_policy_timezone="Asia/Shanghai",
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=0,
    )


async def ingest(rag: LightRAG) -> None:
    await rag.initialize_storages()
    await initialize_pipeline_status()
    await rag.ainsert(
        [doc["content"] for doc in corpus.DOCUMENTS],
        ids=[doc["doc_id"] for doc in corpus.DOCUMENTS],
        file_paths=[doc["file_path"] for doc in corpus.DOCUMENTS],
    )


async def build_catalog(rag: LightRAG, revision: str):
    """Read the real chunk rows back and bind the catalog to them."""
    document_ids: dict[str, str] = {}
    chunk_refs: dict[str, list[dict[str, Any]]] = {}
    detail: dict[str, Any] = {}

    for spec in corpus.DOCUMENTS:
        rows = await rag.doc_status.get_by_ids([spec["doc_id"]])
        row = rows[0] if rows else None
        if not isinstance(row, dict):
            raise SystemExit(f"doc_status has no row for {spec['doc_id']}")
        chunk_ids = list(row.get("chunks_list") or [])
        if not chunk_ids:
            raise SystemExit(f"{spec['doc_id']} produced no chunks")

        fetched = await rag.text_chunks.get_by_ids(chunk_ids)
        by_id: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(fetched or []):
            if not isinstance(item, dict):
                continue
            key = (
                item.get("_id")
                or item.get("id")
                or (chunk_ids[index] if index < len(chunk_ids) else None)
            )
            if key:
                by_id[str(key)] = item

        refs: list[dict[str, Any]] = []
        doc_ids: set[str] = set()
        for chunk_id in chunk_ids:
            chunk_row = by_id.get(chunk_id)
            if not chunk_row or not chunk_row.get("content"):
                raise SystemExit(f"text_chunks row for {chunk_id} is unreadable")
            content = chunk_row["content"]
            full_doc_id = chunk_row.get("full_doc_id")
            if isinstance(full_doc_id, str) and full_doc_id:
                doc_ids.add(full_doc_id)
            refs.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": "",  # filled below, once the doc id is known
                    "span": [0, len(content)],
                    "content_sha256": hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest(),
                    "file_path": chunk_row.get("file_path") or spec["file_path"],
                }
            )
        doc_id = next(iter(doc_ids)) if len(doc_ids) == 1 else spec["doc_id"]
        for ref in refs:
            ref["doc_id"] = doc_id
        document_ids[spec["doc_id"]] = doc_id
        chunk_refs[spec["version_id"]] = refs
        detail[spec["version_id"]] = {
            "doc_id": doc_id,
            "chunk_ids": chunk_ids,
            "file_path": spec["file_path"],
            "span_sha256": [ref["content_sha256"] for ref in refs],
        }

    catalog = catalog_from_mapping(
        corpus.catalog_mapping(
            document_ids=document_ids,
            chunk_refs=chunk_refs,
            revision=revision,
        ),
        source_path=f"generated:{revision}",
    )
    return catalog, detail


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _versions_of(chunk_ids, detail: dict[str, Any]) -> list[str]:
    wanted = set(chunk_ids)
    return sorted(
        version_id
        for version_id, info in detail.items()
        if wanted & set(info["chunk_ids"])
    )


async def run_case(
    case: dict[str, Any],
    rag: LightRAG,
    state: dict[str, Any],
    detail: dict[str, Any],
) -> dict[str, Any]:
    state["calls"] = []
    param = QueryParam(
        mode="naive",
        evidence_policy=POLICY,
        as_of=case["as_of"],
        enable_rerank=False,
        stream=False,
    )
    started = dt.datetime.now(dt.timezone.utc)
    raw = await rag.aquery_llm(corpus.QUERY, param)
    elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()

    llm = raw.get("llm_response") or {}
    block = (raw.get("metadata") or {}).get("evidence_policy") or {}
    answer = llm.get("content") or ""
    calls = list(state["calls"])
    model_context = calls[-1]["system_prompt"] if calls else ""

    seed_ids = block.get("seed_chunk_ids") or []
    expanded_ids = block.get("expanded_chunk_ids") or []
    seed_versions = _versions_of(seed_ids, detail)
    expanded_versions = _versions_of(expanded_ids, detail)
    selected = block.get("selected_versions") or []

    checks: dict[str, Any] = {
        "status_resolved": block.get("status") == "resolved",
        "selected_versions_ok": sorted(selected) == sorted(case["expect_selected"]),
        "answer_contains_expected": case["expect"] in answer,
        "model_was_called": bool(calls),
        "context_was_rendered": "270" in model_context or "180" in model_context,
    }
    if "require_seed" in case:
        checks["seed_has_required"] = all(
            v in seed_versions for v in case["require_seed"]
        )
    if "forbid_seed" in case:
        checks["seed_excludes_forbidden"] = all(
            v not in seed_versions for v in case["forbid_seed"]
        )
    if "expect_expanded" in case:
        checks["expanded_has_required"] = all(
            v in expanded_versions for v in case["expect_expanded"]
        )

    return {
        "id": case["id"],
        "scenario": case["scenario"],
        "query": corpus.QUERY,
        "mode": "naive",
        "as_of": case["as_of"],
        "workspace": case["workspace"],
        "elapsed_seconds": round(elapsed, 2),
        "policy_status": block.get("status"),
        "selected_versions": selected,
        "seed_chunk_ids": seed_ids,
        "seed_versions": seed_versions,
        "expanded_chunk_ids": expanded_ids,
        "expanded_versions": expanded_versions,
        "required_evidence": block.get("required_evidence"),
        "citation_keys": block.get("citation_keys"),
        "model_calls": len(calls),
        "actual_model_context": model_context,
        "context_sha256": hashlib.sha256(model_context.encode("utf-8")).hexdigest(),
        "final_model_answer": answer,
        "llm_generated": llm.get("llm_generated"),
        "expected_contains": case["expect"],
        "checks": checks,
        "passed": all(checks.values()),
        "diagnostics": block,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _reset_workspace(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


async def main_async(args: argparse.Namespace) -> int:
    provider = Provider.from_env()
    preflight_result = await preflight(provider, check_chat=not args.reuse_ingestion)
    print(json.dumps({"preflight": preflight_result}, ensure_ascii=False))
    if args.preflight_only:
        return 0

    workdir = Path(args.workdir).resolve()
    record: dict[str, Any] = {
        "schema_version": 1,
        "issue": "https://github.com/HKUDS/LightRAG/issues/3037",
        "recorded_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "plan": "examples/evidence_policy/README.md",
        "provider": provider.public(),
        "preflight": preflight_result,
        "corpus": {
            "documents": [
                {
                    "doc_id": doc["doc_id"],
                    "file_path": doc["file_path"],
                    "published_at": doc["published_at"],
                    "content": doc["content"],
                }
                for doc in corpus.DOCUMENTS
            ],
            "version_specs": corpus.VERSION_SPECS,
            "scope": corpus.SCOPE,
        },
        "controlled_demo": {
            "corpus": "two fictional documents; not a real legal family",
            "F02": (
                "the revision's chunk vector entry is removed after ingestion so "
                "retrieval cannot recall it; its text_chunks row and its catalog "
                "entry stay in place, which is what the resolver fetches by ID"
            ),
        },
        "workspaces": {},
        "cases": [],
    }

    dim = preflight_result["embed_dim"]

    for name in (WORKSPACE_MAIN, WORKSPACE_UNRECALLED):
        ws_dir = workdir / name
        if args.reuse_ingestion:
            if not ws_dir.is_dir():
                raise SystemExit(f"cannot reuse missing workspace: {ws_dir}")
        elif name == WORKSPACE_MAIN:
            _reset_workspace(ws_dir)
        else:
            # The main instance is finalized before this copy. Reuse its real
            # ingestion output so the same two documents are not extracted twice.
            shutil.copytree(workdir / WORKSPACE_MAIN, ws_dir)
        # A fresh wrapper per workspace: finalize_storages() shuts down the
        # embedding worker pool of the instance it belongs to.
        embedding = await build_embedding_func(provider, dim)
        base_llm = partial(
            openai_complete_if_cache,
            provider.llm_model,
            base_url=provider.llm_base_url,
            api_key=provider.llm_api_key,
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=2048,
            timeout=60,
        )
        llm_func, state = make_recorder(base_llm)
        rag = make_rag(ws_dir, provider, llm_func, embedding)

        if name == WORKSPACE_MAIN and not args.reuse_ingestion:
            await ingest(rag)
        else:
            await rag.initialize_storages()
        catalog, detail = await build_catalog(rag, revision=f"demo-{name}-v1")
        rag.evidence_catalog = catalog
        record["workspaces"][name] = {
            "dir": str(ws_dir),
            "catalog_revision": catalog.revision,
            "catalog_fingerprint": catalog.fingerprint,
            "chunks": detail,
            "ingestion_reused_from": WORKSPACE_MAIN if name != WORKSPACE_MAIN else None,
            "ingestion_reused_from_previous_run": args.reuse_ingestion,
        }

        if name == WORKSPACE_UNRECALLED:
            removed = list(detail["V-new"]["chunk_ids"])
            await rag.chunks_vdb.delete(removed)
            await rag.chunks_vdb.index_done_callback()
            record["workspaces"][name]["vectors_removed_for_demo"] = removed

        for case in [c for c in CASES if c["workspace"] == name]:
            result = await run_case(case, rag, state, detail)
            record["cases"].append(result)
            print(
                json.dumps(
                    {
                        "case": result["id"],
                        "status": result["policy_status"],
                        "selected": result["selected_versions"],
                        "seeds": result["seed_versions"],
                        "expanded": result["expanded_versions"],
                        "passed": result["passed"],
                        "answer": result["final_model_answer"][:200],
                    },
                    ensure_ascii=False,
                )
            )
        record["workspaces"][name]["successful_chat_calls"] = len(state["all_calls"])
        await rag.finalize_storages()

    record["summary"] = {
        "cases": len(record["cases"]),
        "passed": sum(1 for case in record["cases"] if case["passed"]),
        "failed": [case["id"] for case in record["cases"] if not case["passed"]],
        "successful_chat_calls_including_preflight": preflight_result["chat_calls"]
        + sum(ws["successful_chat_calls"] for ws in record["workspaces"].values()),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record["summary"], ensure_ascii=False))
    print(f"record: {out}")
    return 0 if not record["summary"]["failed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Issue 3037 F01-F04 on a real LightRAG workspace"
    )
    parser.add_argument("--workdir", default="temp/evidence-policy-demo")
    parser.add_argument("--out", default="temp/evidence-policy-demo/record.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--reuse-ingestion",
        action="store_true",
        help="Reuse the existing demo workspaces and run only the five questions",
    )
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
