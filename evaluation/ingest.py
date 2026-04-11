"""
Document ingestion for the evaluation pipeline.

Usage:
    python -m evaluation.ingest                  # ingest all docs in data/documents/
    python -m evaluation.ingest --clear          # wipe storage first, then ingest
    python -m evaluation.ingest --dry-run        # list files that would be ingested
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from typing import List

from lightrag import LightRAG
from lightrag.utils import logger

from .config import (
    DOCUMENTS_DIR,
    EVAL_DIR,
    SUPPORTED_EXTENSIONS,
    EvalConfig,
    DEFAULT_CONFIG,
)


# ---------------------------------------------------------------------------
# LightRAG factory
# ---------------------------------------------------------------------------

def _build_lightrag(cfg: EvalConfig) -> LightRAG:
    """Instantiate LightRAG from the evaluation config."""
    # Import the right LLM / embedding functions based on the binding name.
    # We import lazily so that missing optional packages don't break the import.
    llm_func = _resolve_llm(cfg)
    embed_func = _resolve_embedding(cfg)

    return LightRAG(
        working_dir=cfg.working_dir,
        llm_model_func=llm_func,
        embedding_func=embed_func,
        max_parallel_insert=cfg.max_parallel_insert,
    )


def _resolve_llm(cfg: EvalConfig):
    """Return the async LLM callable matching cfg.llm_binding."""
    binding = cfg.llm_binding.lower()

    if binding == "openai":
        from lightrag.llm.openai import openai_complete_if_cache
        import functools

        return functools.partial(
            openai_complete_if_cache,
            model=cfg.llm_model,
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_binding_host,
        )

    if binding in ("azure_openai", "azure"):
        from lightrag.llm.openai import azure_openai_complete_if_cache
        import functools

        return functools.partial(
            azure_openai_complete_if_cache,
            model=cfg.llm_model,
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_binding_host,
        )

    if binding == "ollama":
        from lightrag.llm.ollama import ollama_model_complete
        import functools

        return functools.partial(
            ollama_model_complete,
            model=cfg.llm_model,
            host=cfg.llm_binding_host,
        )

    # Generic OpenAI-compatible fallback
    from lightrag.llm.openai import openai_complete_if_cache
    import functools

    logger.warning(
        "Unknown llm_binding %r, falling back to openai-compatible.", cfg.llm_binding
    )
    return functools.partial(
        openai_complete_if_cache,
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_binding_host,
    )


def _resolve_embedding(cfg: EvalConfig):
    """Return the EmbeddingFunc matching cfg.embedding_binding."""
    from lightrag.utils import EmbeddingFunc

    binding = cfg.embedding_binding.lower()

    if binding == "openai":
        from lightrag.llm.openai import openai_embed
        import functools, numpy as np

        async def _embed(texts: list[str]) -> np.ndarray:
            return await openai_embed(
                texts,
                model=cfg.embedding_model,
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_binding_host,
            )

        return EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=cfg.embedding_max_token_size,
            func=_embed,
        )

    if binding == "ollama":
        from lightrag.llm.ollama import ollama_embedding
        import functools, numpy as np

        async def _embed(texts: list[str]) -> np.ndarray:
            return await ollama_embedding(
                texts,
                embed_model=cfg.embedding_model,
                host=cfg.llm_binding_host,
            )

        return EmbeddingFunc(
            embedding_dim=cfg.embedding_dim,
            max_token_size=cfg.embedding_max_token_size,
            func=_embed,
        )

    raise ValueError(
        f"Unsupported embedding_binding: {cfg.embedding_binding!r}. "
        "Supported: openai, ollama. "
        "Open evaluation/ingest.py to add more."
    )


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------

def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        raise ImportError(
            "pypdf is required to read PDF files. "
            "Install it with: pip install pypdf"
        )


def _read_docx(path: Path) -> str:
    try:
        from docx import Document

        doc = Document(str(path))
        return "\n".join(para.text for para in doc.paragraphs)
    except ImportError:
        raise ImportError(
            "python-docx is required to read DOCX files. "
            "Install it with: pip install python-docx"
        )


def _read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return _read_txt(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    raise ValueError(f"Unsupported extension: {ext}")


# ---------------------------------------------------------------------------
# Ingestion state tracking
# ---------------------------------------------------------------------------

_INGESTED_MANIFEST = EVAL_DIR / "results" / "ingested_files.json"


def _load_manifest() -> dict:
    if _INGESTED_MANIFEST.exists():
        return json.loads(_INGESTED_MANIFEST.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    _INGESTED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _INGESTED_MANIFEST.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_documents(documents_dir: Path = DOCUMENTS_DIR) -> List[Path]:
    """Return all supported document files under documents_dir."""
    docs = []
    for ext in SUPPORTED_EXTENSIONS:
        docs.extend(sorted(documents_dir.rglob(f"*{ext}")))
    return docs


async def ingest_documents(
    cfg: EvalConfig = DEFAULT_CONFIG,
    *,
    clear: bool = False,
    skip_already_ingested: bool = True,
) -> List[str]:
    """
    Insert all documents from data/documents/ into LightRAG.

    Args:
        cfg: Evaluation config.
        clear: If True, delete the working_dir before ingesting (full re-index).
        skip_already_ingested: Skip files already recorded in the manifest.

    Returns:
        List of file names that were successfully ingested this run.
    """
    docs = collect_documents()
    if not docs:
        logger.warning(
            "No documents found in %s. "
            "Place your corpus files there and run ingest again.",
            DOCUMENTS_DIR,
        )
        return []

    if clear:
        logger.info("--clear requested: removing storage at %s", cfg.working_dir)
        shutil.rmtree(cfg.working_dir, ignore_errors=True)
        Path(cfg.working_dir).mkdir(parents=True, exist_ok=True)
        manifest = {}
    else:
        manifest = _load_manifest()

    rag = _build_lightrag(cfg)
    await rag.initialize_storages()

    ingested: List[str] = []

    for doc_path in docs:
        rel = str(doc_path.relative_to(DOCUMENTS_DIR))
        if skip_already_ingested and rel in manifest:
            logger.info("Skipping (already ingested): %s", rel)
            continue

        logger.info("Ingesting: %s", rel)
        try:
            text = _read_file(doc_path)
            if not text.strip():
                logger.warning("Empty document, skipping: %s", rel)
                continue
            await rag.ainsert(text, file_paths=[str(doc_path)])
            manifest[rel] = {"status": "ok", "size_chars": len(text)}
            ingested.append(rel)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", rel, exc)
            manifest[rel] = {"status": "error", "error": str(exc)}

    await rag.finalize_storages()
    _save_manifest(manifest)

    logger.info(
        "Ingestion complete. %d new document(s) processed. "
        "Total in manifest: %d.",
        len(ingested),
        len(manifest),
    )
    return ingested


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest documents into LightRAG.")
    p.add_argument(
        "--clear",
        action="store_true",
        help="Wipe the RAG storage before ingesting (full re-index).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List documents that would be ingested without actually running.",
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-ingest files that are already in the manifest.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    docs = collect_documents()

    if args.dry_run:
        print(f"Found {len(docs)} document(s) in {DOCUMENTS_DIR}:")
        for d in docs:
            print(f"  {d.relative_to(DOCUMENTS_DIR)}")
        return

    asyncio.run(
        ingest_documents(
            clear=args.clear,
            skip_already_ingested=not args.no_skip,
        )
    )


if __name__ == "__main__":
    main()
