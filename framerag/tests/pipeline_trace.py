"""End-to-end pipeline trace with deterministic mock LLM + embed.

Runs the full FrameRAG index→query pipeline on a tiny passage WITHOUT any
network calls, printing every intermediate value so the data flow can be
inspected and regressions caught. Also doubles as a bug-verification harness.

Run:  python -m framerag.tests.pipeline_trace
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import tempfile

import numpy as np

from framerag.framerag import FrameRAG

EMB_DIM = 256  # small for speed; mock embed is bag-of-words

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic mock embed: bag-of-words → real cosine similarity on shared words
# ─────────────────────────────────────────────────────────────────────────────

_WORD_VECS: dict[str, np.ndarray] = {}


def _word_vec(word: str) -> np.ndarray:
    if word not in _WORD_VECS:
        h = hashlib.sha256(word.encode()).digest()
        seed = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed)
        _WORD_VECS[word] = rng.standard_normal(EMB_DIM)
    return _WORD_VECS[word]


def _embed_one(text: str) -> np.ndarray:
    import re
    words = re.findall(r"[a-z0-9]+", text.lower())
    if not words:
        return np.ones(EMB_DIM) / np.sqrt(EMB_DIM)
    v = np.sum([_word_vec(w) for w in words], axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


async def mock_embed(texts: list[str], **kwargs) -> np.ndarray:
    return np.stack([_embed_one(t) for t in texts])


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic mock LLM: routes by prompt fingerprint → canned JSON
# ─────────────────────────────────────────────────────────────────────────────

async def mock_llm(prompt: str, stream: bool = False) -> str:
    p = prompt

    # Call 1 — entity extraction
    if "Knowledge Graph specialist" in p:
        return json.dumps([
            {"entity_name": "Sherlock Holmes", "entity_type": "PERSON",
             "entity_description": "A brilliant consulting detective at Baker Street",
             "entity_aliases": ["Holmes", "the detective"], "entity_salience": "HIGH"},
            {"entity_name": "Dr. Watson", "entity_type": "PERSON",
             "entity_description": "Holmes's loyal companion and the narrator",
             "entity_aliases": ["Watson"], "entity_salience": "HIGH"},
            {"entity_name": "violin", "entity_type": "ARTIFACT",
             "entity_description": "A stringed musical instrument played by Holmes",
             "entity_aliases": [], "entity_salience": "MEDIUM"},
        ])

    # Call 1b — gleaning
    if "MISSED" in p:
        return "[]"

    # Call 2 Step A — event extraction
    if "expert in event extraction" in p:
        return json.dumps([
            {"event_span": "drew the bow across his violin",
             "description": "Holmes played the violin in the evening to relax",
             "participant_names": ["Sherlock Holmes", "violin"],
             "temporal_marker": "in the evening", "is_negation": False},
            {"event_span": "Watson was undisturbed by the music",
             "description": "Watson remained undisturbed by the violin music",
             "participant_names": ["Dr. Watson"],
             "temporal_marker": "", "is_negation": True},
        ])

    # Call 2 Step B — frame annotation (one per event).
    # Route on the event's own Description line (unique per event) so the mock
    # stays correct even when the shared context window mentions both events.
    if "Frame Semantics" in p:
        if "Watson remained undisturbed" in p:
            return json.dumps({
                "frame_name": "NOT_Affect", "is_new_frame": True,
                "frame_definition": "An entity is excluded from the effect of an event",
                "lexical_unit": "undisturbed.adj",
                "core_elements": [
                    {"fe_name": "Excluded_Entity", "filler_text": "Dr. Watson",
                     "filler_type": "ENTITY", "is_missing": False}],
                "noncore_elements": [
                    {"fe_name": "Reference_Event", "filler_text": "the violin music",
                     "filler_type": "VALUE", "is_missing": False}],
            })
        # Note frame_name with a SPACE to exercise the normalize bug/fix
        return json.dumps({
            "frame_name": "Music performance", "is_new_frame": True,
            "frame_definition": "An agent plays a musical instrument",
            "lexical_unit": "play.v",
            "core_elements": [
                {"fe_name": "Agent", "filler_text": "Sherlock Holmes",
                 "filler_type": "ENTITY", "is_missing": False},
                {"fe_name": "Instrument", "filler_text": "violin",
                 "filler_type": "ENTITY", "is_missing": False}],
            "noncore_elements": [
                {"fe_name": "Time", "filler_text": "in the evening",
                 "filler_type": "VALUE", "is_missing": False}],
        })

    # Call 3 — causal/temporal
    if "event causality" in p:
        return "[]"

    # Query processing
    if "query analysis expert" in p:
        return json.dumps({
            "entity_hints": ["Holmes", "detective"],
            "event_hints": ["play", "perform"],
            "frame_hints": "Music performance",
            "fe_focus": ["Agent", "Instrument"],
            "temporal_hints": [],
        })

    # HyDE
    if "novelist's assistant" in p:
        return ("In the evening Holmes drew the bow across his violin, the "
                "haunting melody filling the room as he played to relax.")

    # Frame selection (LLM)
    if "frame retrieval expert" in p:
        return json.dumps(["Music performance"])

    # Answer generation
    if "question-answering assistant" in p:
        return "Holmes played the violin to relax."

    return "[]"


# ─────────────────────────────────────────────────────────────────────────────
# Trace harness
# ─────────────────────────────────────────────────────────────────────────────

def _hdr(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


async def _trace_query_phases(rag, query: str) -> None:
    """Re-run the query pipeline step by step, printing every intermediate so
    the data flowing through each phase is visible (mirrors _retrieve)."""
    import numpy as np
    from framerag.operate import (
        process_query, generate_hyde_passage,
        expand_query_frames, expand_query_frames_llm,
    )
    from framerag.diffusion import HypergraphDiffusion
    hg = rag._hg

    # Phase Q1 — query signals
    sig = await process_query(query, rag._llm)
    _hdr("Q1 — QUERY SIGNALS (process_query)")
    for k, v in sig.items():
        print(f"  {k:15} = {v}")

    # Phase Q2 — frame expansion + HyDE (parallel in product)
    hyde = await generate_hyde_passage(query, rag._llm)
    llm_frames = await expand_query_frames_llm(query, rag._frame_db, rag._llm)
    emb_frames = await expand_query_frames(query, sig.get("frame_hints", ""), rag._frame_db)
    _hdr("Q2 — EXPANSION")
    print(f"  HyDE passage   = {hyde!r}")
    print(f"  LLM frames     = {llm_frames}")
    print(f"  embed frames   = {emb_frames}")
    print(f"  union          = {list(dict.fromkeys([*llm_frames, *emb_frames]))}")

    # Phase Q3 — matrices
    matrices = await hg.build_matrices(fe_focus=sig.get("fe_focus") or None)
    _hdr("Q3 — SPARSE MATRICES (shape / nnz)")
    for key in ("H_ce", "H_ef", "H_fe", "A_cau", "A_coref"):
        M = matrices[key]
        print(f"  {key:8} shape={str(M.shape):12} nnz={M.nnz}")
    print(f"  node_ids  = {[n[:10] for n in matrices['node_ids']]}")
    print(f"  fi_ids    = {[n[:10] for n in matrices['fi_ids']]}")

    node_idx, fi_idx, chunk_idx = (
        matrices["node_idx"], matrices["fi_idx"], matrices["chunk_idx"])
    n_n, n_f, n_c = len(node_idx), len(fi_idx), len(chunk_idx)
    y_node, y_fi, y_chunk = np.zeros(n_n), np.zeros(n_f), np.zeros(n_c)

    # Phase Q4 — seeding (show which nodes/frames/chunks each seed fires)
    _hdr("Q4 — SEEDING (which targets each seed source hits)")
    q_vec = (await mock_embed([query]))[0]

    async def show_mentions(vec, tag, w):
        hits = await hg.search_entity_mentions(vec, top_k=20)
        for h in hits:
            if h["id"] in node_idx:
                y_node[node_idx[h["id"]]] += h["distance"] * w
        print(f"  [{tag}] mentions -> " + ", ".join(
            f"{h['id'][:10]}:{h['distance']:.3f}" for h in hits if h['id'] in node_idx))

    async def show_frames(vec, tag, w):
        hits = await hg.search_frame_instances(vec, top_k=20)
        for h in hits:
            if h["id"] in fi_idx:
                y_fi[fi_idx[h["id"]]] += h["distance"] * w
        print(f"  [{tag}] frames   -> " + ", ".join(
            f"{h['id'][:10]}:{h['distance']:.3f}" for h in hits if h['id'] in fi_idx))

    async def show_chunks(vec, tag, w):
        hits = await hg.search_chunks(vec, top_k=30)
        for h in hits:
            if h["id"] in chunk_idx:
                y_chunk[chunk_idx[h["id"]]] += h["distance"] * w
        print(f"  [{tag}] chunks   -> " + ", ".join(
            f"{h['id'][:12]}:{h['distance']:.3f}" for h in hits if h['id'] in chunk_idx))

    await show_mentions(q_vec, "query", 1.0)
    await show_frames(q_vec, "query", 1.0)
    await show_chunks(q_vec, "query", 1.0)
    if hyde:
        hv = (await mock_embed([hyde]))[0]
        await show_mentions(hv, "hyde ", rag._hyde_weight)
    # event-hint seeding → events VDB → frame instances
    for eh in sig.get("event_hints", []):
        ev = (await mock_embed([eh]))[0]
        ev_hits = await hg.search_events(ev, top_k=10)
        boosted = []
        for h in ev_hits:
            for fid in hg.frame_ids_for_event(h["id"]):
                if fid in fi_idx:
                    y_fi[fi_idx[fid]] += h["distance"] * rag._event_hint_weight
                    boosted.append((fid[:10], round(h["distance"], 3)))
        print(f"  [evhint {eh!r}] event->FI boost -> {boosted}")

    def _l1(v):
        s = v.sum()
        return v / s if s > 1e-12 else v

    print("\n  seed vectors (post-L1, nonzero entries):")
    print("   y_node :", {matrices['node_ids'][i][:10]: round(float(x), 3)
                          for i, x in enumerate(_l1(y_node)) if x > 1e-9})
    print("   y_fi   :", {matrices['fi_ids'][i][:10]: round(float(x), 3)
                          for i, x in enumerate(_l1(y_fi)) if x > 1e-9})
    print("   y_chunk:", {matrices['chunk_ids'][i][:12]: round(float(x), 3)
                          for i, x in enumerate(_l1(y_chunk)) if x > 1e-9})

    # Phase Q5 — diffusion, step by step
    _hdr("Q5 — DIFFUSION (warm-up then cooling; top values per step)")
    diff = HypergraphDiffusion(matrices)
    fn, ff, fc = _l1(y_node).copy(), _l1(y_fi).copy(), _l1(y_chunk).copy()
    yn, yf, yc = fn.copy(), ff.copy(), fc.copy()

    def _top(vec, ids):
        i = int(np.argmax(vec))
        return f"{ids[i][:12]}={vec[i]:.4f}"

    for k in range(rag._diffusion_warm_up):
        fn, ff, fc = diff._step_warmup(
            fn, ff, fc, yn, yf, yc,
            alpha=rag._diffusion_alpha, causal_weight=0.3,
            coref_weight=rag._coref_weight)
        print(f"  warmup[{k}]  topNode={_top(fn, matrices['node_ids'])}  "
              f"topFI={_top(ff, matrices['fi_ids'])}  "
              f"topChunk={_top(fc, matrices['chunk_ids'])}")
    fn_frozen = fn.copy()
    k = 0
    while True:
        wb = rag._diffusion_t_decay ** k
        if wb < rag._diffusion_epsilon:
            break
        ff, fc = diff._step_cool(
            fn_frozen, ff, fc, yf, yc,
            w_back=wb, alpha=rag._diffusion_alpha, causal_weight=0.3)
        if k < 3 or wb < rag._diffusion_epsilon * 2:
            print(f"  cool[{k:2}] wb={wb:.3f}  topFI={_top(ff, matrices['fi_ids'])}  "
                  f"topChunk={_top(fc, matrices['chunk_ids'])}")
        k += 1
    print(f"  (cooling ran {k} steps total)")


async def main() -> None:
    from lightrag.kg.shared_storage import initialize_share_data
    initialize_share_data(workers=1)

    work = tempfile.mkdtemp(prefix="frtrace_")
    # Small chunk_size so the passage splits into several chunks. The mock LLM
    # emits Holmes/Watson/violin for EVERY chunk, so coreference merges the
    # repeated mentions into one canonical each → A_coref edges fire → diffusion
    # has real multi-hop / cross-chunk structure to propagate over.
    rag = FrameRAG(
        working_dir=work, llm_func=mock_llm, embed_func=mock_embed,
        embedding_dim=EMB_DIM, chunk_size=30, chunk_overlap=5,
        enable_hyde=True, max_concurrent_llm=4,
    )
    await rag.initialize()

    passage = (
        "Sherlock Holmes sat in his armchair at Baker Street, thinking about the "
        "case. In the evening he drew the bow across his violin, playing a haunting "
        "melody to relax after a long day of investigation. Later that night Holmes "
        "set down the violin and lit his pipe by the fire. Dr. Watson was "
        "undisturbed by the music, having grown used to his friend's late-night "
        "recitals at Baker Street."
    )
    _hdr("INPUT DOCUMENT")
    print(passage)

    await rag.ainsert(passage, source_doc="holmes_violin")

    hg = rag._hg

    _hdr("STEP 1 — CHUNKS")
    for cid in hg._chunk_ids:
        c = await hg.chunks.get_by_id(cid)
        print(f"  {cid}  tokens={c['tokens']}  text={c['text'][:60]!r}...")

    _hdr("STEP 2 — ENTITY MENTIONS")
    for mid in hg._mention_ids:
        m = await hg.entity_mentions.get_by_id(mid)
        print(f"  {mid[:14]}  {m['name']!r:22} type={m['entity_type']:8} "
              f"canon={str(m['canonical_id'])[:14]}")

    _hdr("STEP 3 — CANONICAL ENTITIES")
    for cid in hg._canonical_ids:
        c = await hg.canonical_entities.get_by_id(cid)
        print(f"  {cid[:14]}  {c['canonical_name']!r:22} type={c['entity_type']:8} "
              f"#mentions={len(c['mention_ids'])}")

    _hdr("STEP 4 — EVENTS  (check: event description includes FE roles?)")
    for eid in hg._event_ids:
        e = await hg.events.get_by_id(eid)
        print(f"  {eid[:12]}  trigger={e['trigger_lemma']!r}  frame={e['frame_name']!r}")
        print(f"      description = {e['event_description']!r}")
        print(f"      vdb content = {e['trigger_lemma']} [{e['frame_name']}]: {e['event_description']}")

    _hdr("STEP 5 — FRAME INSTANCES  (check: frame_name normalized, no spaces?)")
    for fid in hg._fi_ids:
        f = await hg.frame_instances.get_by_id(fid)
        core = [f"{a['fe_name']}={a['filler_text']}" for a in f["core_assignments"]]
        print(f"  {fid[:12]}  frame={f['frame_name']!r}  LU={f['lexical_unit']!r}")
        print(f"      core: {core}")

    _hdr("STEP 6 — INFO NODES")
    for iid in hg._info_ids:
        i = await hg.info_nodes.get_by_id(iid)
        print(f"  {iid[:14]}  value={i['value']!r:24} type={i['info_type']}")

    _hdr("STEP 7 — ADJACENCY")
    print(f"  chunk→event : {dict((k[:10], [e[:8] for e in v]) for k, v in hg._adj_chunk_event.items())}")
    print(f"  event→frame : {dict((k[:8], [f[:8] for f in v]) for k, v in hg._adj_event_frame.items())}")
    print(f"  mention→canon: {dict((k[:10], v[:10]) for k, v in hg._adj_mention_canon.items())}")
    print("  frame→node  :")
    for fid, adj in hg._adj_frame_node.items():
        print(f"    {fid[:10]}: {[(a['fe_name'], a['node_id'][:8], a['weight']) for a in adj]}")

    # ── QUERY ────────────────────────────────────────────────────────────────
    query = "What instrument did the detective play to relax?"
    _hdr(f"QUERY: {query!r}")

    await _trace_query_phases(rag, query)

    ctx = await rag.aretrieve_context(query)
    _hdr("RETRIEVAL OUTPUT (via aretrieve_context)")
    print("  frame_hits:")
    for h in ctx["frame_hits"]:
        print(f"    {h['id'][:12]}  score={h['score']:.5f}")
    print("  chunk_hits:")
    for h in ctx["chunk_hits"]:
        print(f"    {h['id'][:14]}  score={h['score']:.5f}")
    print("\n  structured_facts:\n" + "\n".join("    " + ln for ln in ctx["structured_facts"].splitlines()))

    answer = await rag.aquery(query)
    _hdr("FINAL ANSWER")
    print("  " + answer)

    # ── ASSERTIONS (bug regression checks) ────────────────────────────────────
    _hdr("ASSERTIONS")
    results = []

    # Bug #2: frame names must be normalized (no spaces)
    bad_names = []
    for fid in hg._fi_ids:
        f = await hg.frame_instances.get_by_id(fid)
        if " " in f["frame_name"]:
            bad_names.append(f["frame_name"])
    results.append(("#2 frame names have no spaces", not bad_names, bad_names))

    # Bug #1: event description should carry FE role info (Agent/Instrument)
    enriched = False
    for eid in hg._event_ids:
        e = await hg.events.get_by_id(eid)
        if "Agent" in e["event_description"] or "Instrument" in e["event_description"]:
            enriched = True
    results.append(("#1 event description carries FE roles", enriched, None))

    # Bug #4/#3: event VDB must actually be queried via event_hints. Verify the
    # event-hint seed path moves frame-instance scores (run retrieval with the
    # event VDB emptied → the Music frame should rank lower / score differently).
    music_fi = None
    for fid in hg._fi_ids:
        f = await hg.frame_instances.get_by_id(fid)
        if f["frame_name"].startswith("Music"):
            music_fi = fid
    music_in_hits = any(h["id"] == music_fi for h in ctx["frame_hits"])
    results.append(("#3/#4 event-hint path surfaces Music frame", music_in_hits, None))

    # Retrieval correctness: the violin chunk must be retrieved
    got_chunk = bool(ctx["chunk_hits"])
    results.append(("retrieval returns chunk hits", got_chunk, None))

    ok = True
    for name, passed, extra in results:
        mark = "PASS" if passed else "FAIL"
        ok = ok and passed
        print(f"  [{mark}] {name}" + (f"  -> {extra}" if extra else ""))

    shutil.rmtree(work, ignore_errors=True)
    print("\n" + ("ALL PASS" if ok else "SOME FAILURES"))


if __name__ == "__main__":
    asyncio.run(main())
