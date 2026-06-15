"""Run FrameRAG ChronoQA evaluation for a SINGLE story in its own process.

Env knobs:
  MAX_EXCERPTS : cap excerpts indexed per story (0 = all). Default 0.
  MAX_LLM      : max_concurrent_llm for FrameRAG. Default 3.
"""
import asyncio
import json
import logging
import os
import sys

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logging.getLogger("lightrag").setLevel(logging.ERROR)

if not os.getenv("OPENAI_API_KEY"):
    try:
        with open(".env", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    os.environ.setdefault("OPENAI_API_KEY", line.split("=", 1)[1])
    except FileNotFoundError:
        pass

MAX_EXCERPTS = int(os.getenv("MAX_EXCERPTS", "0"))
MAX_LLM = int(os.getenv("MAX_LLM", "3"))


async def main(story_id, output_json, working_dir):
    from framerag import FrameRAG
    from framerag.evaluation.datasets import load_chronoqa
    from framerag.evaluation.paper_eval import (
        _make_llm, _make_embed, _collect_story_corpus,
        _judge_response, DEFAULT_JUDGE_MODELS, _summarise,
    )
    try:
        from lightrag.kg.shared_storage import initialize_share_data
        initialize_share_data(workers=1)
    except Exception:
        pass

    samples = load_chronoqa()
    grp = [s for s in samples if str(s.get("story_id")) == str(story_id)]
    if not grp:
        print(f"[story {story_id}] no samples", flush=True)
        return

    corpus = _collect_story_corpus(grp)
    if MAX_EXCERPTS > 0:
        corpus = corpus[:MAX_EXCERPTS]

    llm = _make_llm("gpt-4.1-mini")
    emb = _make_embed("text-embedding-3-small")
    story_dir = os.path.join(working_dir, "framerag", f"story_{story_id}")

    rag = FrameRAG(working_dir=story_dir, llm_func=llm, embed_func=emb,
                   embedding_dim=1536, enable_llm_coref_verify=False,
                   max_concurrent_llm=MAX_LLM)
    await rag.initialize()
    # corpus is now a single concatenated document (sorted by byte offset, overlaps merged)
    print(f"[story {story_id}] indexing {len(corpus)} doc(s) (~{sum(len(d) for d in corpus)} chars), {len(grp)} questions", flush=True)
    docs = [(i, d) for i, d in enumerate(corpus) if d.strip()]
    await rag.ainsert_batch(
        texts=[d for _, d in docs],
        source_docs=[f"story_{story_id}_doc{i}" for i, _ in docs],
        concurrency=1,
    )
    print(f"[story {story_id}] indexed; querying", flush=True)

    query_sem = asyncio.Semaphore(4)
    async def _q(s):
        async with query_sem:
            try:
                a = await rag.aquery(s["question"])
            except Exception as e:
                print(f"[story {story_id}] query err: {e}", flush=True)
                a = ""
        return {**s, "response": a, "system": "FrameRAG"}
    raw = list(await asyncio.gather(*[_q(s) for s in grp]))
    await rag.finalize()

    judge_llms = [_make_llm(m) for m in DEFAULT_JUDGE_MODELS]
    judge_sem = asyncio.Semaphore(8)
    async def _judge(sr):
        gold = sr.get("gold_answers") or [sr.get("ground_truth", "")]
        gt = "; ".join(str(a) for a in gold if a)
        v = await _judge_response(sr["question"], gt, sr.get("response", ""), judge_llms, judge_sem)
        return {**sr, "avg_score": v["avg_score"], "judges": v["judges"]}
    judged = await asyncio.gather(*[_judge(r) for r in raw])
    summary = _summarise(judged)
    out = {"story_id": story_id, "summary": summary, "samples": judged,
           "n_excerpts_indexed": len(docs)}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[story {story_id}] DONE avg={summary['avg_score']:.4f} n={summary['n']}", flush=True)


if __name__ == "__main__":
    sid = sys.argv[1]
    out = sys.argv[2]
    wd = sys.argv[3] if len(sys.argv) > 3 else "./eval_storage"
    asyncio.run(main(sid, out, wd))
