"""
ArXiv Paper Knowledge Base with LightRAG
=========================================
This example demonstrates how to build a searchable knowledge base from arXiv
papers using LightRAG's graph-enhanced retrieval.

Workflow:
  1. Fetch paper abstracts from arXiv API (no API key needed)
  2. Insert them into LightRAG for indexing
  3. Query across papers with Local / Global / Hybrid modes

Usage:
  # Install dependencies
  pip install lightrag-hku arxiv

  # Set your LLM credentials (or use Ollama locally)
  export OPENAI_API_KEY="sk-..."

  # Run with a list of arXiv IDs
  python lightrag_arxiv_papers_demo.py --ids 2410.05779 1706.03762 2005.11401

  # Or use Ollama (free, no API key)
  python lightrag_arxiv_papers_demo.py \\
      --ids 2410.05779 1706.03762 \\
      --llm-model qwen2.5:7b \\
      --embed-model nomic-embed-text \\
      --ollama
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# ── LightRAG imports ──────────────────────────────────────────────────────────
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed

WORKING_DIR = Path("./arxiv_knowledge_base")


# ── ArXiv helpers (standalone, no extra deps) ─────────────────────────────────

@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str


def _parse_arxiv_id(raw: str) -> str:
    """Extract bare arXiv ID from URL or raw string."""
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]+)", raw)
    if m:
        return m.group(1)
    m = re.match(r"^([0-9]{4}\.[0-9]+)(?:v\d+)?$", raw.strip())
    if m:
        return m.group(1)
    raise ValueError(f"Cannot parse arXiv ID from: {raw!r}")


def fetch_papers(arxiv_ids: list[str], timeout: int = 20) -> list[ArxivPaper]:
    """Fetch paper metadata from arXiv API (free, no API key required)."""
    id_list = ",".join(arxiv_ids)
    url = f"https://export.arxiv.org/api/query?id_list={id_list}&max_results={len(arxiv_ids)}"
    req = urllib.request.Request(url, headers={"User-Agent": "lightrag-arxiv-demo/1.0"})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                xml_data = resp.read().decode("utf-8")
            break
        except OSError as e:
            if attempt == 2:
                raise ConnectionError(
                    f"arXiv API unreachable after 3 attempts: {e}\n"
                    "Tip: a proxy may be needed in some regions."
                ) from e
            time.sleep(2 ** attempt)

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id_raw = entry.findtext("atom:id", namespaces=ns, default="")
        # Extract bare ID from URL like http://arxiv.org/abs/2301.00001v3
        m = re.search(r"abs/([0-9]{4}\.[0-9]+)", arxiv_id_raw)
        arxiv_id = m.group(1) if m else arxiv_id_raw
        papers.append(ArxivPaper(
            arxiv_id=arxiv_id,
            title=entry.findtext("atom:title", namespaces=ns, default="").strip(),
            authors=[
                a.findtext("atom:name", namespaces=ns, default="")
                for a in entry.findall("atom:author", ns)
            ],
            abstract=entry.findtext("atom:summary", namespaces=ns, default="").strip(),
            published=entry.findtext("atom:published", namespaces=ns, default="")[:10],
        ))
    return papers


def paper_to_document(paper: ArxivPaper) -> str:
    """Format a paper as a plain-text document for LightRAG ingestion."""
    return (
        f"Title: {paper.title}\n"
        f"ArXiv ID: {paper.arxiv_id}\n"
        f"Authors: {', '.join(paper.authors[:5])}{'et al.' if len(paper.authors) > 5 else ''}\n"
        f"Published: {paper.published}\n\n"
        f"Abstract:\n{paper.abstract}"
    )


# ── LightRAG setup ────────────────────────────────────────────────────────────

def build_rag(llm_model: str, embed_model: str, use_ollama: bool) -> LightRAG:
    WORKING_DIR.mkdir(exist_ok=True)

    if use_ollama:
        from lightrag.llm.ollama import ollama_model_complete, ollama_embed
        from lightrag.utils import EmbeddingFunc

        rag = LightRAG(
            working_dir=str(WORKING_DIR),
            llm_model_func=ollama_model_complete,
            llm_model_name=llm_model,
            llm_model_max_async=4,
            llm_model_max_token_size=32768,
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda texts: ollama_embed(texts, embed_model=embed_model),
            ),
        )
    else:
        import os
        from lightrag.utils import EmbeddingFunc

        rag = LightRAG(
            working_dir=str(WORKING_DIR),
            llm_model_func=openai_complete_if_cache,
            llm_model_name=llm_model,
            llm_model_max_async=4,
            llm_model_max_token_size=32768,
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model=embed_model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                ),
            ),
        )
    return rag


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    arxiv_ids = [_parse_arxiv_id(x) for x in args.ids]

    print(f"📥 Fetching {len(arxiv_ids)} paper(s) from arXiv API...")
    papers = fetch_papers(arxiv_ids)
    print(f"✅ Fetched: {', '.join(p.title[:40] + '...' for p in papers)}\n")

    print("🔧 Initializing LightRAG...")
    rag = build_rag(args.llm_model, args.embed_model, args.ollama)
    await rag.initialize_storages()

    print("📚 Inserting papers into knowledge graph...")
    documents = [paper_to_document(p) for p in papers]
    await rag.ainsert(documents)
    print(f"✅ Inserted {len(documents)} document(s)\n")

    # ── Demo queries ──────────────────────────────────────────────────────────
    demo_queries = [
        "What are the main contributions of RAG-related papers?",
        "Which papers propose graph-based retrieval methods?",
        "Compare the approaches used in these papers.",
    ]

    for query in demo_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)

        for mode in ["local", "global", "hybrid"]:
            print(f"\n[{mode.upper()} mode]")
            result = await rag.aquery(query, param=QueryParam(mode=mode))
            print(result[:400] + ("..." if len(result) > 400 else ""))

    await rag.finalize_storages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build an arXiv paper knowledge base with LightRAG"
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=["2410.05779", "1706.03762", "2005.11401"],
        help="arXiv IDs or URLs (default: LightRAG, Attention, RAG papers)",
    )
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument(
        "--embed-model", default="text-embedding-3-small", help="Embedding model name"
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use local Ollama instead of OpenAI (free)",
    )
    args = parser.parse_args()

    # Override defaults for Ollama
    if args.ollama:
        if args.llm_model == "gpt-4o-mini":
            args.llm_model = "qwen2.5:7b"
        if args.embed_model == "text-embedding-3-small":
            args.embed_model = "nomic-embed-text"

    asyncio.run(main(args))
