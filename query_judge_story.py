"""Query + judge a story against its ALREADY-INDEXED FrameRAG hypergraph."""
import asyncio, json, logging, os, sys
logging.basicConfig(level=logging.ERROR)
if not os.getenv("OPENAI_API_KEY"):
    with open(".env", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("OPENAI_API_KEY="):
                os.environ.setdefault("OPENAI_API_KEY", line.strip().split("=",1)[1])

async def main(sid, out):
    from framerag import FrameRAG
    from framerag.evaluation.datasets import load_chronoqa
    from framerag.evaluation.paper_eval import _make_llm, _make_embed, _judge_response, DEFAULT_JUDGE_MODELS, _summarise
    try:
        from lightrag.kg.shared_storage import initialize_share_data
        initialize_share_data(workers=1)
    except Exception: pass
    grp = [s for s in load_chronoqa() if str(s.get("story_id"))==str(sid)]
    llm=_make_llm("gpt-4o-mini"); emb=_make_embed("text-embedding-3-small")
    rag=FrameRAG(working_dir=f"eval_storage/framerag/story_{sid}", llm_func=llm, embed_func=emb, embedding_dim=1536, max_concurrent_llm=6)
    await rag.initialize()
    stats=await rag.get_stats()
    print("INDEX:", {k:stats[k] for k in ["chunks","events","frame_instances","causal_edges","frames_in_db"]}, flush=True)
    qsem=asyncio.Semaphore(6)
    async def _q(s):
        async with qsem:
            try: a=await rag.aquery(s["question"])
            except Exception as e: a=""; print("qerr",e,flush=True)
        return {**s,"response":a,"system":"FrameRAG"}
    raw=list(await asyncio.gather(*[_q(s) for s in grp]))
    await rag.finalize()
    jl=[_make_llm(m) for m in DEFAULT_JUDGE_MODELS]; jsem=asyncio.Semaphore(8)
    async def _j(sr):
        gold=sr.get("gold_answers") or [sr.get("ground_truth","")]
        gt="; ".join(str(a) for a in gold if a)
        v=await _judge_response(sr["question"],gt,sr.get("response",""),jl,jsem)
        return {**sr,"avg_score":v["avg_score"],"judges":v["judges"]}
    judged=await asyncio.gather(*[_j(r) for r in raw])
    summ=_summarise(judged)
    json.dump({"story_id":sid,"summary":summ,"samples":judged}, open(out,"w",encoding="utf-8"), ensure_ascii=False)
    print(f"DONE story {sid} avg={summ['avg_score']:.4f} n={summ['n']}", flush=True)

if __name__=="__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
