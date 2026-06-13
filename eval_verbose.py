#!/usr/bin/env python3
"""
Evaluation rõ ràng từng bước: indexing → retrieval → RAGAS.

Chạy:
    python eval_verbose.py               # dùng graph hiện tại
    python eval_verbose.py --reindex     # xóa storage và index lại từ đầu

Requires:  server đang chạy ở localhost:9621
           pip install ragas datasets langchain-openai httpx networkx
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from textwrap import wrap

import httpx
import networkx as nx
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

# ─── Config ──────────────────────────────────────────────────────────────────
API_URL   = os.getenv("LIGHTRAG_API_URL", "http://localhost:9621")
API_KEY   = os.getenv("LIGHTRAG_API_KEY", "")
DOCS_DIR  = Path("lightrag/evaluation/sample_documents")
GRAPHML   = Path("rag_storage/graph_chunk_entity_relation.graphml")
DATASET   = Path("lightrag/evaluation/sample_dataset.json")
HEADERS   = {"X-API-Key": API_KEY} if API_KEY else {}

EVAL_LLM_KEY  = os.getenv("EVAL_LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY", "")
EVAL_LLM      = os.getenv("EVAL_LLM_MODEL", "gpt-4o-mini")
QUERY_MODE    = os.getenv("EVAL_QUERY_MODE", "hybrid")

SEP   = "=" * 70
SEP2  = "-" * 70


def hdr(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def sub(title: str):
    print(f"\n  {SEP2}")
    print(f"  {title}")
    print(f"  {SEP2}")


# ─── STEP 1: Graph state ─────────────────────────────────────────────────────

def show_graph_state():
    hdr("STEP 1: TRANG THAI KNOWLEDGE GRAPH")
    if not GRAPHML.exists():
        print(f"  [WARN] Chua co graph: {GRAPHML}")
        return 0, 0

    G = nx.read_graphml(str(GRAPHML))
    n, e = G.number_of_nodes(), G.number_of_edges()
    frame_mode = os.getenv("LIGHTRAG_FRAME_EXTRACTION_MODE", "full")
    print(f"  Mode extraction:   {frame_mode.upper()}")
    print(f"  Graph file:        {GRAPHML}")
    print(f"  Entities (nodes):  {n}")
    print(f"  Relations (edges): {e}")

    if n > 0:
        print(f"\n  Top entities (by degree):")
        top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:min(15, n)]
        for node, deg in top:
            etype = G.nodes[node].get("entity_type", "?")
            desc  = G.nodes[node].get("description", "")[:80]
            print(f"    [{deg:2d}] {str(node)[:40]:<40} type={etype}")
            if desc:
                for line in wrap(desc, 60):
                    print(f"           {line}")

    if e > 0:
        print(f"\n  Sample relations (up to 10):")
        for u, v, d in list(G.edges(data=True))[:10]:
            kw   = d.get("keywords", "")[:30]
            desc = d.get("description", "")[:60]
            print(f"    {str(u)[:22]:<22} --[{kw}]--> {str(v)[:22]}")
            if desc:
                print(f"      {desc}")

    return n, e


# ─── STEP 2: Re-index ────────────────────────────────────────────────────────

async def reindex():
    hdr("STEP 2: INDEX TAI LIEU")

    # Clear storage — recreate dir immediately so running server can still write
    storage = Path("rag_storage")
    if storage.exists():
        shutil.rmtree(storage)
        print(f"  Cleared: {storage}")
    storage.mkdir(parents=True, exist_ok=True)
    await asyncio.sleep(2)  # let server detect dir exists again

    files = sorted(DOCS_DIR.glob("*.md"))
    files = [f for f in files if f.name.lower() != "readme.md"]
    print(f"  Files to index: {len(files)}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30), headers=HEADERS) as c:
        for f in files:
            text = f.read_text(encoding="utf-8")
            r = await c.post(f"{API_URL}/documents/text",
                             json={"text": text, "file_path": f.name})
            status = "OK " if r.status_code == 200 else "ERR"
            print(f"  [{status}] {f.name:40} ({len(text):5d} chars)")

    print("\n  Dang doi indexing...")
    t0 = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(30), headers=HEADERS) as c:
        while time.time() - t0 < 600:
            await asyncio.sleep(5)
            data = (await c.get(f"{API_URL}/documents")).json()
            raw  = data.get("statuses", data)
            if isinstance(raw, dict) and any(isinstance(v, list) for v in raw.values()):
                stat = {k: len(v) for k, v in raw.items() if isinstance(v, list)}
            else:
                stat = raw
            pending = stat.get("pending", 0) + stat.get("processing", 0)
            failed  = stat.get("failed", 0)
            done    = stat.get("processed", stat.get("success", 0))
            print(f"  [{time.time()-t0:4.0f}s] {stat}")
            if pending == 0:
                if failed > 0 and done == 0:
                    print(f"  [ERROR] Tat ca {failed} docs BI FAILED. Kiem tra EMBEDDING_DIM va server log!")
                    sys.exit(1)
                break
    print(f"  Indexing xong: {time.time()-t0:.0f}s")


# ─── STEP 3: Query từng câu hỏi ─────────────────────────────────────────────

async def query_one(q: str, client: httpx.AsyncClient) -> dict:
    """
    Dùng /query/data để lấy đầy đủ structured data bao gồm chunks với content.
    /query endpoint có bug: file_path="unknown_source" → references=[] → contexts trống.
    /query/data luôn trả đủ data.chunks với content.
    """
    payload = {
        "query": q,
        "mode": QUERY_MODE,
        "include_references": True,
        "include_chunk_content": True,
        "response_type": "Multiple Paragraphs",
        "top_k": 10,
    }
    # Dùng /query/data để lấy full structured response
    r = await client.post(f"{API_URL}/query/data", json=payload, headers=HEADERS)
    r.raise_for_status()
    result = r.json()

    # Extract answer
    llm_resp = result.get("llm_response", {})
    if isinstance(llm_resp, dict):
        answer = llm_resp.get("content", "")
    else:
        answer = ""

    # Nếu /query/data không có llm_response, dùng /query để lấy text answer
    if not answer:
        r2 = await client.post(f"{API_URL}/query", json={**payload, "include_chunk_content": False}, headers=HEADERS)
        if r2.status_code == 200:
            answer = r2.json().get("response", "")

    # Extract contexts trực tiếp từ data.chunks (bypass broken references system)
    data = result.get("data", {})
    chunks = data.get("chunks", [])
    contexts: list[str] = []
    for chunk in chunks:
        content = chunk.get("content", "")
        if content:
            contexts.append(content)

    # Also collect references for display
    references_raw = data.get("references", [])
    entities = data.get("entities", [])
    relations = data.get("relationships", [])

    return {
        "answer": answer,
        "contexts": contexts,
        "references_raw": references_raw,
        "chunks_raw": chunks,
        "entities_found": entities,
        "relations_found": relations,
        "metadata": result.get("metadata", {}),
    }


async def run_retrieval(test_cases: list[dict]) -> list[dict]:
    hdr(f"STEP 3: RETRIEVAL (mode={QUERY_MODE})")
    results = []
    timeout = httpx.Timeout(180, connect=10)

    async with httpx.AsyncClient(timeout=timeout) as client:
        for i, case in enumerate(test_cases, 1):
            q  = case["question"]
            gt = case["ground_truth"]
            sub(f"Query {i}/{len(test_cases)}")
            print(f"  Q: {q}")
            print()

            t0 = time.time()
            try:
                r = await query_one(q, client)
                elapsed = time.time() - t0

                print(f"  ANSWER ({elapsed:.1f}s):")
                for line in wrap(r["answer"][:600], 65):
                    print(f"    {line}")
                if len(r["answer"]) > 600:
                    print(f"    ... [{len(r['answer'])} chars total]")

                # Graph retrieval info
                meta = r.get("metadata", {})
                proc = meta.get("processing_info", {})
                print(f"\n  GRAPH RETRIEVAL:")
                print(f"    Entities found:   {proc.get('total_entities_found', len(r.get('entities_found', [])))}")
                print(f"    Relations found:  {proc.get('total_relations_found', len(r.get('relations_found', [])))}")
                print(f"    Keywords hl:      {meta.get('keywords', {}).get('high_level', [])}")
                print(f"    Keywords ll:      {meta.get('keywords', {}).get('low_level', [])}")

                # Entity list (brief)
                ents = r.get("entities_found", [])
                if ents:
                    print(f"    Top entities: {[e.get('entity_name','?') for e in ents[:5]]}")

                print(f"\n  CONTEXTS RETRIEVED: {len(r['contexts'])} chunk(s)")
                for j, ctx in enumerate(r["contexts"][:3], 1):
                    print(f"    [{j}] {ctx[:200]}{'...' if len(ctx) > 200 else ''}")
                if len(r["contexts"]) > 3:
                    print(f"    ... ({len(r['contexts']) - 3} more chunks)")

                results.append({
                    "test_number": i,
                    "question": q,
                    "ground_truth": gt,
                    "answer": r["answer"],
                    "contexts": r["contexts"],
                    "contexts_count": len(r["contexts"]),
                    "elapsed_s": round(elapsed, 2),
                })

            except Exception as exc:
                print(f"  [ERROR] {exc}")
                results.append({
                    "test_number": i,
                    "question": q,
                    "ground_truth": gt,
                    "answer": "",
                    "contexts": [],
                    "contexts_count": 0,
                    "error": str(exc),
                    "elapsed_s": 0,
                })

    return results


# ─── STEP 4: RAGAS ──────────────────────────────────────────────────────────

def run_ragas(retrieval_results: list[dict]) -> list[dict]:
    hdr("STEP 4: RAGAS METRICS")
    if not EVAL_LLM_KEY:
        print("  [SKIP] Khong co EVAL_LLM_BINDING_API_KEY / OPENAI_API_KEY")
        return retrieval_results

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (AnswerRelevancy, ContextPrecision,
                                   ContextRecall, Faithfulness)
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as e:
        print(f"  [SKIP] Thieu thu vien RAGAS: {e}")
        return retrieval_results

    llm = LangchainLLMWrapper(
        langchain_llm=ChatOpenAI(model=EVAL_LLM, api_key=EVAL_LLM_KEY, max_retries=3),
        bypass_n=True,
    )
    emb = OpenAIEmbeddings(model="text-embedding-3-large", api_key=EVAL_LLM_KEY)

    for r in retrieval_results:
        if r.get("error") or not r.get("answer"):
            r["metrics"] = {}
            continue

        contexts = r["contexts"] if r["contexts"] else ["(no context retrieved)"]
        ds = Dataset.from_dict({
            "question":    [r["question"]],
            "answer":      [r["answer"]],
            "contexts":    [contexts],
            "ground_truth":[r["ground_truth"]],
        })

        try:
            eval_result = evaluate(
                dataset=ds,
                metrics=[Faithfulness(), AnswerRelevancy(),
                         ContextRecall(), ContextPrecision()],
                llm=llm,
                embeddings=emb,
            )
            row = eval_result.to_pandas().iloc[0]

            def fval(col):
                v = float(row.get(col, 0))
                return 0.0 if math.isnan(v) else v

            r["metrics"] = {
                "faithfulness":      fval("faithfulness"),
                "answer_relevance":  fval("answer_relevancy"),
                "context_recall":    fval("context_recall"),
                "context_precision": fval("context_precision"),
            }
            vals = list(r["metrics"].values())
            r["ragas_score"] = round(sum(vals) / len(vals), 4) if vals else 0.0

        except Exception as exc:
            print(f"  [ERROR] RAGAS query {r['test_number']}: {exc}")
            r["metrics"] = {}

        # Print per-query
        sub(f"Query {r['test_number']} — RAGAS scores")
        print(f"  Q: {r['question'][:70]}")
        print(f"  contexts_count = {r['contexts_count']}")
        m = r.get("metrics", {})
        print(f"  Faithfulness:      {m.get('faithfulness', 0):.4f}")
        print(f"  Answer Relevance:  {m.get('answer_relevance', 0):.4f}")
        print(f"  Context Recall:    {m.get('context_recall', 0):.4f}")
        print(f"  Context Precision: {m.get('context_precision', 0):.4f}")
        print(f"  RAGAS Score:       {r.get('ragas_score', 0):.4f}")

    return retrieval_results


# ─── STEP 5: Summary ─────────────────────────────────────────────────────────

def show_summary(results: list[dict], n_entities: int, n_relations: int):
    hdr("STEP 5: KET QUA TONG HOP")
    frame_mode = os.getenv("LIGHTRAG_FRAME_EXTRACTION_MODE", "full")
    print(f"  Extraction mode:   {frame_mode.upper()}")
    print(f"  Query mode:        {QUERY_MODE}")
    print(f"  Entities in graph: {n_entities}")
    print(f"  Relations in graph:{n_relations}")
    print()

    metric_names = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
    valid = [r for r in results if r.get("metrics")]
    avg = {}
    for m in metric_names:
        vals = [r["metrics"][m] for r in valid if m in r.get("metrics", {})]
        avg[m] = round(sum(vals) / len(vals), 4) if vals else 0.0
    scores = [r.get("ragas_score", 0) for r in valid]
    avg["ragas_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0

    print(f"  {'Query':<5} {'Ctx':>4} {'Faith':>7} {'AnswRel':>8} {'CtxRec':>7} {'CtxPrec':>8} {'RAGAS':>7}")
    print(f"  {'-'*50}")
    for r in results:
        m = r.get("metrics", {})
        print(
            f"  {r['test_number']:<5}"
            f" {r.get('contexts_count',0):>4}"
            f" {m.get('faithfulness',0):>7.4f}"
            f" {m.get('answer_relevance',0):>8.4f}"
            f" {m.get('context_recall',0):>7.4f}"
            f" {m.get('context_precision',0):>8.4f}"
            f" {r.get('ragas_score',0):>7.4f}"
        )
    print(f"  {'-'*50}")
    print(
        f"  {'AVG':<5}"
        f" {'':>4}"
        f" {avg['faithfulness']:>7.4f}"
        f" {avg['answer_relevance']:>8.4f}"
        f" {avg['context_recall']:>7.4f}"
        f" {avg['context_precision']:>8.4f}"
        f" {avg['ragas_score']:>7.4f}"
    )

    print()
    print(f"  Contexts count per query: {[r.get('contexts_count',0) for r in results]}")
    if all(r.get('contexts_count', 0) == 0 for r in results):
        print()
        print("  [WARN] contexts_count=0 cho moi query!")
        print("  Giai thich: references tra ve nhung 'content' chua duoc populate.")
        print("  Dieu nay xay ra khi: include_chunk_content=True nhung 'chunks' trong")
        print("  data tra ve bi rong. Context Recall / Precision se bi anh huong.")
        print("  Faithfulness va Answer Relevance van chay duoc (khong can context).")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = frame_mode
    out = Path(f"eval_results_{mode}_{ts}.json")
    out.write_text(json.dumps({
        "extraction_mode": mode,
        "query_mode": QUERY_MODE,
        "n_entities": n_entities,
        "n_relations": n_relations,
        "average_metrics": avg,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out.absolute()}")


# ─── Main ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reindex", action="store_true",
                        help="Xoa storage va index lai truoc khi evaluate")
    parser.add_argument("--skip-ragas", action="store_true",
                        help="Bo qua RAGAS (chi xem retrieval)")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  LightRAG Verbose Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Extraction mode: {os.getenv('LIGHTRAG_FRAME_EXTRACTION_MODE','full').upper()}")
    print(f"  Query mode:      {QUERY_MODE}")
    print(f"{'#'*70}")

    if args.reindex:
        await reindex()
        # Re-check server after re-index
        await asyncio.sleep(2)

    n_e, n_r = show_graph_state()

    with open(DATASET, encoding="utf-8") as f:
        test_cases = json.load(f)["test_cases"]
    print(f"\n  Dataset: {DATASET.name}  ({len(test_cases)} test cases)")

    retrieval = await run_retrieval(test_cases)

    if not args.skip_ragas:
        retrieval = run_ragas(retrieval)

    show_summary(retrieval, n_e, n_r)


if __name__ == "__main__":
    asyncio.run(main())
