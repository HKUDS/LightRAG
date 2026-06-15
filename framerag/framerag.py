"""FrameRAG: Frame-Semantic Event Hypergraph RAG system.

Inherits LightRAG's infrastructure:
  - Storage   : JsonKVStorage + NanoVectorDBStorage
  - Chunking  : chunking_by_token_size (TiktokenTokenizer)
  - LLM Cache : JsonKVStorage (keyed by MD5 of prompt)
  - Entity merge: cross-doc description merging + LLM summarisation

Indexing pipeline (per document):
  chunk → (Call 1) entity extraction [+ gleaning]
        → (Call 2) event + frame + role extraction (with Frame DB hints)
        → (Call 3) causal edge extraction
        → entity coreference resolution
        → entity description merging (cross-doc)
        → hypergraph storage

Post-indexing (call post_process_frames() once after all docs):
  cluster near-duplicate frames → re-evoke canonical frame from event text
  → merge LUs + usage → delete duplicates from DB

Query pipeline:
  query → LLM seed signals → vector search → sparse matrix build
        → hypergraph diffusion → collect top context → LLM answer
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, Callable, Awaitable, Optional

import numpy as np

from lightrag.utils import logger, EmbeddingFunc, compute_mdhash_id, TiktokenTokenizer
from lightrag.operate import chunking_by_token_size

from .storage import make_kv, wrap_embed
from .rerank import RerankFunc, rerank_chunk_hits
from .doc_store import DocStore, DocStatus, DocRecord
from .constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_GLEANING,
    DEFAULT_DIFFUSION_WARM_UP,
    DEFAULT_DIFFUSION_ALPHA,
    DEFAULT_DIFFUSION_T_DECAY,
    DEFAULT_DIFFUSION_EPSILON,
    DEFAULT_TOP_CHUNKS,
    DEFAULT_TOP_FRAMES,
    DEFAULT_TOP_NODES,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_MAX_ASYNC,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_DESC_MERGE_THRESHOLD,
)
from .types import (
    ChunkSchema,
    EntityMentionSchema,
    CanonicalEntitySchema,
    EventSchema,
    FrameInstanceSchema,
    InfoNodeSchema,
    CausalEdgeSchema,
    QuerySignals,
)
from .frame_db import FrameDatabase
from .hypergraph import HypergraphStore
from .diffusion import HypergraphDiffusion
from .coreference.entity_coref import EntityCoreferenceResolver
from .operate import (
    extract_entities,
    glean_entities,
    extract_events_frames_two_step,
    extract_causal_edges,
    expand_query_frames,
    expand_query_frames_llm,
    process_query,
    generate_answer,
    _llm_with_retry,
)
from .prompt import PROMPTS


class FrameRAG:
    """Frame-Semantic Event Hypergraph RAG.

    Usage:
        rag = FrameRAG(
            working_dir="./framerag_storage",
            llm_func=my_llm,
            embed_func=my_embed,
        )
        await rag.initialize()
        await rag.ainsert("Document text here...")
        answer = await rag.aquery("What caused X?")

        # Streaming
        async for token in rag.aquery_stream("What caused X?"):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        working_dir: str,
        llm_func: Callable[..., Awaitable[str]],
        embed_func: Callable[[list[str]], Awaitable[np.ndarray]],
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        enable_causal: bool = True,
        enable_gleaning: bool = True,
        max_gleaning_rounds: int = DEFAULT_MAX_GLEANING,
        enable_llm_coref_verify: bool = False,
        diffusion_warm_up: int = DEFAULT_DIFFUSION_WARM_UP,
        diffusion_alpha: float = DEFAULT_DIFFUSION_ALPHA,
        diffusion_t_decay: float = DEFAULT_DIFFUSION_T_DECAY,
        diffusion_epsilon: float = DEFAULT_DIFFUSION_EPSILON,
        top_chunks: int = DEFAULT_TOP_CHUNKS,
        top_frames: int = DEFAULT_TOP_FRAMES,
        top_nodes: int = DEFAULT_TOP_NODES,
        rerank_func: Optional[Callable] = None,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
        llm_timeout: float = DEFAULT_LLM_TIMEOUT,
        embed_timeout: float = DEFAULT_EMBEDDING_TIMEOUT,
        max_concurrent_llm: int = DEFAULT_MAX_ASYNC,
    ):
        self._raw_llm = llm_func
        self._raw_embed = embed_func
        self._embedding_dim = embedding_dim
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._enable_causal = enable_causal
        self._enable_gleaning = enable_gleaning
        self._max_gleaning_rounds = max_gleaning_rounds
        self._enable_llm_coref_verify = enable_llm_coref_verify
        self._diffusion_warm_up = diffusion_warm_up
        self._diffusion_alpha = diffusion_alpha
        self._diffusion_t_decay = diffusion_t_decay
        self._diffusion_epsilon = diffusion_epsilon
        self._top_chunks = top_chunks
        self._top_frames = top_frames
        self._top_nodes = top_nodes
        self._rerank_func = rerank_func
        self._rerank_top_k = rerank_top_k
        self._llm_timeout = llm_timeout
        self._embed_timeout = embed_timeout
        self._llm_sem = asyncio.Semaphore(max_concurrent_llm)
        self._save_lock = asyncio.Lock()
        self._desc_merge_threshold = DEFAULT_DESC_MERGE_THRESHOLD

        self._working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)

        ef: EmbeddingFunc = wrap_embed(embed_func, embedding_dim)

        self._hg         = HypergraphStore(working_dir, ef)
        self._frame_db   = FrameDatabase(working_dir, ef)
        self._entity_coref = EntityCoreferenceResolver(
            embed_func, llm_func,
            enable_llm_verify=enable_llm_coref_verify,
        )
        self._doc_store  = DocStore(working_dir)

        # LLM response cache (JsonKVStorage)
        self._llm_cache = make_kv("llm_cache", working_dir)

        # Tokenizer for chunking (TiktokenTokenizer)
        try:
            self._tokenizer = TiktokenTokenizer("gpt-4o-mini")
        except Exception:
            self._tokenizer = None  # fallback: word-split in _chunk_text

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        await self._hg.initialize()
        await self._frame_db.initialize()
        await self._llm_cache.initialize()
        await self._doc_store.initialize()
        logger.info("[FrameRAG] Initialized")

    async def finalize(self) -> None:
        async with self._save_lock:
            await self._hg.index_done_callback()
            await self._frame_db.index_done_callback()
            await self._llm_cache.index_done_callback()
            await self._doc_store.index_done_callback()
        logger.info("[FrameRAG] Storage flushed")

    # ──────────────────────────────────────────────────────────────────────────
    # LLM with caching
    # ──────────────────────────────────────────────────────────────────────────

    async def _llm(self, prompt: str) -> str:
        """Call LLM with KV cache, semaphore throttle, and timeout."""
        cache_key = compute_mdhash_id(prompt, prefix="llm-")
        cached = await self._llm_cache.get_by_id(cache_key)
        if cached and cached.get("response"):
            return cached["response"]
        async with self._llm_sem:
            response = await asyncio.wait_for(
                _llm_with_retry(self._raw_llm, prompt),
                timeout=self._llm_timeout,
            )
        await self._llm_cache.upsert({cache_key: {"response": response}})
        return response

    async def _llm_stream(self, prompt: str) -> AsyncIterator[str]:
        """Streaming LLM call — tries async-generator path, falls back to cached non-streaming."""
        import inspect
        try:
            result = self._raw_llm(prompt, stream=True)
            if inspect.isasyncgen(result):
                # LLM returned an async generator directly
                async for chunk in result:
                    yield chunk
                return
            if asyncio.isfuture(result) or inspect.isawaitable(result):
                # LLM returned a coroutine (non-streaming with stream kwarg ignored)
                yield await result
                return
            # Sync iterable (shouldn't happen but guard)
            for chunk in result:
                yield chunk
        except TypeError:
            # LLM func doesn't accept stream=True — fall back to cached non-streaming
            yield await self._llm(prompt)

    # ──────────────────────────────────────────────────────────────────────────
    # Chunking
    # ──────────────────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, source_doc: str) -> list[ChunkSchema]:
        if self._tokenizer is not None:
            raw_chunks = chunking_by_token_size(
                self._tokenizer,
                text,
                chunk_overlap_token_size=self._chunk_overlap,
                chunk_token_size=self._chunk_size,
            )
            return [
                ChunkSchema(
                    chunk_id=compute_mdhash_id(
                        f"{source_doc}_{rc['chunk_order_index']}", prefix="chunk-"
                    ),
                    text=rc["content"],
                    source_doc=source_doc,
                    chunk_index=rc["chunk_order_index"],
                    tokens=rc["tokens"],
                )
                for rc in raw_chunks
                if rc["content"].strip()
            ]
        # Fallback: whitespace token split
        tokens = text.split()
        chunks: list[ChunkSchema] = []
        step = max(1, self._chunk_size - self._chunk_overlap)
        i = 0
        for start in range(0, max(1, len(tokens) - self._chunk_overlap), step):
            toks = tokens[start: start + self._chunk_size]
            if not toks:
                break
            content = " ".join(toks)
            chunks.append(ChunkSchema(
                chunk_id=compute_mdhash_id(f"{source_doc}_{i}", prefix="chunk-"),
                text=content,
                source_doc=source_doc,
                chunk_index=i,
                tokens=len(toks),
            ))
            i += 1
            if start + self._chunk_size >= len(tokens):
                break
        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────────────────

    async def ainsert(self, text: str, source_doc: str = "unknown") -> None:
        """Index a document into the frame-semantic hypergraph."""
        doc_id = DocStore.make_doc_id(source_doc)
        existing = await self._doc_store.get(doc_id)
        if existing and existing.status == DocStatus.PROCESSED:
            logger.info(f"[FrameRAG] Skipping '{source_doc}': already indexed")
            return
        record = DocRecord(doc_id=doc_id, source_doc=source_doc, status=DocStatus.PENDING)
        await self._doc_store.upsert(record)

        try:
            record.status = DocStatus.PROCESSING
            await self._doc_store.upsert(record)

            chunks = self._chunk_text(text, source_doc)
            logger.info(f"[FrameRAG] Inserting '{source_doc}': {len(chunks)} chunks")

            all_mentions: list[EntityMentionSchema] = []
            all_events:   list[EventSchema]          = []
            all_fi:       list[FrameInstanceSchema]  = []
            all_info:     list[InfoNodeSchema]       = []
            all_causal:   list[CausalEdgeSchema]     = []
            chunk_ids: list[str] = [c.chunk_id for c in chunks]

            # ── Per-chunk extraction (parallelised within this passage) ────────
            # Phase 1a: batch-embed all chunk texts at once, register in KV + VDB.
            # Runs concurrently with Phase 1b via asyncio.gather below.
            async def _register_chunks() -> None:
                """Batch-embed all chunks, write KV + VDB in one pass."""
                await asyncio.gather(*[self._hg.add_chunk(c) for c in chunks])

            # Phase 1b: LLM extraction — all chunks concurrently (throttled by _llm_sem)
            async def _extract_chunk(
                chunk: ChunkSchema,
            ) -> tuple[
                list[EntityMentionSchema],
                list[EventSchema],
                list[FrameInstanceSchema],
                list[InfoNodeSchema],
                list[CausalEdgeSchema],
                list,  # new_frames (FrameDefinitionSchema)
            ]:
                # Call 1: entity extraction
                mentions = await extract_entities(chunk, self._llm)
                if self._enable_gleaning and mentions:
                    gleaned = await glean_entities(
                        chunk, mentions, self._llm,
                        max_rounds=self._max_gleaning_rounds,
                    )
                    mentions = mentions + gleaned

                # Call 2: event + frame + role (2-step: detect events, then
                # annotate frames per event in parallel)
                events, fis, infos, new_frames = await extract_events_frames_two_step(
                    chunk, mentions, self._llm, self._frame_db
                )

                # Call 3: causal edges
                causal_edges: list[CausalEdgeSchema] = []
                if self._enable_causal and len(events) >= 2:
                    causal_edges = await extract_causal_edges(chunk, events, self._llm)

                return mentions, events, fis, infos, causal_edges, new_frames

            # Run chunk registration (embed) + LLM extraction concurrently
            results = await asyncio.gather(
                _register_chunks(),
                asyncio.gather(*[_extract_chunk(c) for c in chunks]),
            )
            chunk_results = results[1]

            # Phase 1c: aggregate + update frame DB (sequential — DB writes)
            seen_new_frames: set[str] = set()
            for mentions, events, fis, infos, causal_edges, new_frames in chunk_results:
                all_mentions.extend(mentions)
                all_events.extend(events)
                all_fi.extend(fis)
                all_info.extend(infos)
                all_causal.extend(causal_edges)
                for fd in new_frames:
                    if fd.frame_name not in seen_new_frames:
                        await self._frame_db.upsert_frame(fd)
                        seen_new_frames.add(fd.frame_name)
                # Count usage for frames that appeared in events but were NOT
                # created as new in ANY chunk of this document.
                for ev in events:
                    if ev.frame_name not in seen_new_frames:
                        await self._frame_db.increment_usage(ev.frame_name)

            # ── Entity coreference ───────────────────────────────────────────
            logger.info(f"[FrameRAG] Entity coref: {len(all_mentions)} mentions")
            canonicals = await self._entity_coref.resolve_with_llm_verify(all_mentions)

            # ── Cross-document entity description merging ────────────────────
            canonicals = await self._merge_entity_descriptions(canonicals, all_mentions)

            # ── Persist to hypergraph ────────────────────────────────────────
            for m in all_mentions:
                await self._hg.add_entity_mention(m)
            for canon in canonicals:
                await self._hg.add_canonical_entity(canon)
            for ev in all_events:
                await self._hg.add_event(ev)

            frame_core_fes: dict[str, set[str]] = {}
            frame_defs: dict[str, str] = {}
            for ev in all_events:
                if ev.frame_name not in frame_core_fes:
                    fd = await self._frame_db.get(ev.frame_name)
                    frame_core_fes[ev.frame_name] = fd.core_fe_names() if fd else set()
                    frame_defs[ev.frame_name] = fd.frame_definition if fd else ""

            mention_by_id = {m.mention_id: m for m in all_mentions}
            info_by_id    = {info.info_id: info for info in all_info}
            event_by_id   = {ev.event_id: ev for ev in all_events}

            for fi in all_fi:
                core_names = frame_core_fes.get(fi.frame_name, set())
                rich_content = self._build_fi_content(
                    fi, frame_defs.get(fi.frame_name, ""),
                    mention_by_id, info_by_id, event_by_id,
                )
                await self._hg.add_frame_instance(fi, core_names, rich_content)
            for info in all_info:
                await self._hg.add_info_node(info)
            for edge in all_causal:
                await self._hg.add_causal_edge(edge)

            record.status      = DocStatus.PROCESSED
            record.chunk_ids   = chunk_ids
            record.chunks_count = len(chunk_ids)
            await self._doc_store.upsert(record)
            await self.finalize()
            logger.info(
                f"[FrameRAG] Indexed '{source_doc}': {len(all_mentions)} mentions, "
                f"{len(canonicals)} canonicals, {len(all_events)} events, "
                f"{len(all_fi)} frame instances"
            )
        except Exception as e:
            record.status    = DocStatus.FAILED
            record.error_msg = str(e)
            await self._doc_store.upsert(record)
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Frame instance rich content builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_fi_content(
        self,
        fi: "FrameInstanceSchema",
        frame_definition: str,
        mention_by_id: dict,
        info_by_id: dict,
        event_by_id: dict,
    ) -> str:
        """Build semantically rich text for frame instance VDB embedding.

        Format:
          Frame: {name} — {frame_definition}
          Trigger: {event_span}
          {fe_name}: {filler_text} ({entity_description})
          {fe_name}: {value}  [info_type]
        """
        parts: list[str] = [f"Frame: {fi.frame_name}"]
        if frame_definition:
            parts[0] += f" — {frame_definition}"

        ev = event_by_id.get(fi.event_id)
        if ev and ev.event_span:
            parts.append(f"Trigger: {ev.event_span}")

        for a in fi.core_assignments + fi.noncore_assignments:
            if not a.filler_text:
                continue
            line = f"{a.fe_name}: {a.filler_text}"
            if a.filler_type == "ENTITY" and a.filler_id:
                m = mention_by_id.get(a.filler_id)
                if m and m.description:
                    line += f" ({m.description})"
            elif a.filler_type == "VALUE" and a.filler_id:
                info = info_by_id.get(a.filler_id)
                if info and info.info_type:
                    line += f" [{info.info_type}]"
            parts.append(line)

        return " | ".join(parts)

    # ──────────────────────────────────────────────────────────────────────────
    # Entity description merging (cross-document)
    # ──────────────────────────────────────────────────────────────────────────

    async def _merge_entity_descriptions(
        self,
        canonicals: list[CanonicalEntitySchema],
        all_mentions: list,
    ) -> list[CanonicalEntitySchema]:
        """Cross-passage canonical ID unification.

        Identifies mentions of the same real-world entity across passages and
        maps them to a single canonical_id so hyperedges connect correctly.
        No description management — just ID unification.

        Two-pass lookup:
        1. Normalized name match — strips honorifics then does O(1) hash lookup.
        2. Embedding similarity >= 0.70 against the canonical entity VDB.
        """
        import re as _re
        CROSS_EMBED_THRESHOLD = 0.70
        _TITLE_RE = _re.compile(
            r"^(mr\.?|mrs\.?|ms\.?|miss\.?|dr\.?|prof\.?|sir\.?|lord\.?|lady\.?)\s+",
            _re.IGNORECASE,
        )

        def _normalize(name: str) -> str:
            return _TITLE_RE.sub("", name.strip()).lower()

        mention_by_canon: dict[str, list] = {}
        for m in all_mentions:
            mention_by_canon.setdefault(m.canonical_id, []).append(m)

        def _unify(canon: CanonicalEntitySchema, existing_id: str) -> None:
            old_id = canon.canonical_id
            canon.canonical_id = existing_id
            for m in mention_by_canon.get(old_id, []):
                m.canonical_id = existing_id
            # Register this surface form so future Pass-1 lookups find it.
            norm = _normalize(canon.canonical_name)
            self._hg._canonical_name_idx.setdefault(canon.canonical_name.lower().strip(), existing_id)
            self._hg._canonical_name_idx.setdefault(norm, existing_id)

        for canon in canonicals:
            # Pass 1: normalized name match (handles "Mr. X" vs "X" variants)
            norm = _normalize(canon.canonical_name)
            existing_id = (
                self._hg.get_canonical_id_by_name(canon.canonical_name)
                or self._hg.get_canonical_id_by_name(norm)
            )
            if existing_id and existing_id != canon.canonical_id:
                _unify(canon, existing_id)
                continue

            # Pass 2: embedding similarity match
            search_text = f"{canon.canonical_name}: {' '.join(canon.descriptions[:1])}"
            try:
                vec = await self._raw_embed([search_text])
                hits = await self._hg.search_canonical(vec[0], top_k=5)
                for hit in hits:
                    if hit.get("distance", 0.0) >= CROSS_EMBED_THRESHOLD:
                        hit_id = hit["id"]
                        if hit_id != canon.canonical_id:
                            _unify(canon, hit_id)
                            break
            except Exception as e:
                logger.debug(f"[FrameRAG] Cross-passage unify failed for '{canon.canonical_name}': {e}")

        return canonicals

    # ──────────────────────────────────────────────────────────────────────────
    # Query
    # ──────────────────────────────────────────────────────────────────────────

    async def _retrieve(
        self,
        query: str,
        top_chunks: int,
        top_frames: int,
        top_nodes: int,
        diffusion_warm_up: int,
    ) -> tuple[list[dict], list[dict]]:
        """Shared retrieval logic: returns (frame_hits, chunk_hits)."""
        # Step 1: Extract query signals
        signals_raw = await process_query(query, self._llm)
        signals = QuerySignals(
            entity_hints=signals_raw.get("entity_hints", []),
            event_hints=signals_raw.get("event_hints", []),
            frame_hints=signals_raw.get("frame_hints", ""),
            fe_focus=signals_raw.get("fe_focus", []),
            temporal_hints=signals_raw.get("temporal_hints", []),
        )

        # Frame expansion — LLM selects from actual frame DB (primary path)
        expanded_frames: list[str] = await expand_query_frames_llm(
            query, self._frame_db, self._llm
        )
        # Fallback: embedding-based expansion from query_processing hint
        if not expanded_frames and signals.frame_hints:
            expanded_frames = await expand_query_frames(
                query, signals.frame_hints, self._frame_db
            )
        logger.info(f"[FrameRAG] Expanded frames: {expanded_frames}")

        # Step 2: Build sparse matrices
        matrices = await self._hg.build_matrices(fe_focus=signals.fe_focus or None)
        if not matrices:
            return [], []

        n_nodes  = len(matrices["node_ids"])
        n_frames = len(matrices["fi_ids"])
        n_chunks = len(matrices["chunk_ids"])

        # Step 3: Embed query once
        q_emb = await self._raw_embed([query])
        q_vec = q_emb[0]

        y_node  = np.zeros(n_nodes,  dtype=np.float64)
        y_fi    = np.zeros(n_frames, dtype=np.float64)
        y_chunk = np.zeros(n_chunks, dtype=np.float64)

        node_idx  = matrices["node_idx"]
        fi_idx    = matrices["fi_idx"]
        chunk_idx = matrices["chunk_idx"]

        # Seed from canonical entity similarity
        # node_idx now uses canonical_ids (not mention_ids), so index directly
        for hit in await self._hg.search_canonical(q_vec, top_k=20):
            cid = hit["id"]
            if cid in node_idx:
                y_node[node_idx[cid]] += hit.get("distance", 0.0)

        # Seed from frame instance similarity
        for hit in await self._hg.search_frame_instances(q_vec, top_k=20):
            fid = hit["id"]
            if fid in fi_idx:
                y_fi[fi_idx[fid]] += hit.get("distance", 0.0)

        # Boost frame instances for expanded related frames (re-embed frame names)
        if expanded_frames:
            frame_embs = await self._raw_embed(expanded_frames)
            for frame_emb in frame_embs:
                for hit in await self._hg.search_frame_instances_by_vec(
                    frame_emb, top_k=10
                ):
                    fid = hit["id"]
                    if fid in fi_idx:
                        y_fi[fi_idx[fid]] += hit.get("distance", 0.0) * 0.5

        # Seed from chunk similarity
        for hit in await self._hg.search_chunks(q_vec, top_k=30):
            cid = hit["id"]
            if cid in chunk_idx:
                y_chunk[chunk_idx[cid]] += hit.get("distance", 0.0)

        # Seed from info node similarity (temporal hints)
        if signals.temporal_hints:
            t_emb = await self._raw_embed([" ".join(signals.temporal_hints)])
            for hit in await self._hg.search_info_nodes(t_emb[0], top_k=10):
                iid = hit["id"]
                if iid in node_idx:
                    y_node[node_idx[iid]] += hit.get("distance", 0.0) * 0.5

        def _l1(v: np.ndarray) -> np.ndarray:
            s = v.sum()
            return v / s if s > 1e-12 else v

        # Step 4: Hypergraph diffusion
        diffusion = HypergraphDiffusion(matrices)
        f_node, f_fi, f_chunk = diffusion.run(
            _l1(y_node), _l1(y_fi), _l1(y_chunk),
            alpha=self._diffusion_alpha,
            warm_up_steps=self._diffusion_warm_up,
            t_decay=self._diffusion_t_decay,
            epsilon=self._diffusion_epsilon,
        )

        # Step 5: Top results — retrieve extra chunks when reranker is active
        fetch_chunks = self._rerank_top_k if self._rerank_func else top_chunks
        results = diffusion.top_results(
            f_node, f_fi, f_chunk,
            top_chunks=fetch_chunks,
            top_frames=top_frames,
            top_nodes=top_nodes,
        )
        frame_hits = results["frame_hits"]
        chunk_hits = results["chunk_hits"]

        # Step 6: Rerank chunks (optional)
        if self._rerank_func and chunk_hits:
            chunk_texts: dict[str, str] = {}
            for hit in chunk_hits:
                cdata = await self._hg.chunks.get_by_id(hit["id"])
                if cdata:
                    chunk_texts[hit["id"]] = cdata.get("content", "")
            chunk_hits = await rerank_chunk_hits(
                query, chunk_hits, chunk_texts, self._rerank_func, top_n=top_chunks
            )
            logger.info(f"[FrameRAG] Reranked to {len(chunk_hits)} chunks")

        return frame_hits, chunk_hits

    async def aquery(
        self,
        query: str,
        top_chunks: Optional[int] = None,
        top_frames: Optional[int] = None,
        top_nodes: Optional[int] = None,
        diffusion_warm_up: Optional[int] = None,
    ) -> str:
        """Answer a query using hypergraph diffusion retrieval."""
        answer, _ = await self.aquery_with_context(
            query,
            top_chunks=top_chunks,
            top_frames=top_frames,
            top_nodes=top_nodes,
            diffusion_warm_up=diffusion_warm_up,
        )
        return answer

    async def aquery_with_context(
        self,
        query: str,
        top_chunks: Optional[int] = None,
        top_frames: Optional[int] = None,
        top_nodes: Optional[int] = None,
        diffusion_warm_up: Optional[int] = None,
    ) -> tuple[str, list[str]]:
        """Answer a query and return (answer, retrieved_passages).

        The retrieved_passages list is suitable for RAGAS context metrics.
        Each element is one text chunk retrieved from the hypergraph.
        """
        frame_hits, chunk_hits = await self._retrieve(
            query,
            top_chunks=top_chunks or self._top_chunks,
            top_frames=top_frames or self._top_frames,
            top_nodes=top_nodes or self._top_nodes,
            diffusion_warm_up=diffusion_warm_up or self._diffusion_warm_up,
        )
        if not frame_hits and not chunk_hits:
            return "No indexed documents found. Please insert documents first.", []

        structured_facts = await self._build_structured_facts(frame_hits)
        text_passages    = await self._build_text_passages(chunk_hits)
        answer = await generate_answer(query, structured_facts, text_passages, self._llm)

        # Split joined passages back into individual strings for RAGAS
        passages = [
            p.strip()
            for p in text_passages.split("\n\n---\n\n")
            if p.strip() and p.strip() != "(no passages retrieved)"
        ]
        return answer, passages

    async def aquery_stream(
        self,
        query: str,
        top_chunks: Optional[int] = None,
        top_frames: Optional[int] = None,
        top_nodes: Optional[int] = None,
        diffusion_warm_up: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Streaming version of aquery — yields answer tokens as they arrive."""
        frame_hits, chunk_hits = await self._retrieve(
            query,
            top_chunks=top_chunks or self._top_chunks,
            top_frames=top_frames or self._top_frames,
            top_nodes=top_nodes or self._top_nodes,
            diffusion_warm_up=diffusion_warm_up or self._diffusion_warm_up,
        )
        if not frame_hits and not chunk_hits:
            yield "No indexed documents found. Please insert documents first."
            return

        structured_facts = await self._build_structured_facts(frame_hits)
        text_passages    = await self._build_text_passages(chunk_hits)
        prompt = PROMPTS["answer_generation"].format(
            structured_facts=structured_facts,
            text_passages=text_passages,
            query=query,
        )
        async for token in self._llm_stream(prompt):
            yield token

    # ──────────────────────────────────────────────────────────────────────────
    # Context builders
    # ──────────────────────────────────────────────────────────────────────────

    async def _build_structured_facts(self, frame_hits: list[dict]) -> str:
        lines = []
        seen_frame_defs: set[str] = set()
        for i, hit in enumerate(frame_hits, 1):
            fid = hit.get("id", "")
            fi_data = await self._hg.frame_instances.get_by_id(fid)
            if not fi_data:
                continue
            frame_name = fi_data.get("frame_name", "")
            lu = fi_data.get("lexical_unit", "")

            # Include frame definition on first occurrence of each frame type
            frame_def_line = ""
            if frame_name not in seen_frame_defs:
                fd = await self._frame_db.get(frame_name)
                if fd and fd.frame_definition:
                    frame_def_line = f"  [def: {fd.frame_definition}]"
                seen_frame_defs.add(frame_name)

            parts = [f"[{i}] Frame: {frame_name} (LU: {lu}){frame_def_line}"]
            for slot in ("core_assignments", "noncore_assignments"):
                for a in fi_data.get(slot, []):
                    if a.get("filler_text"):
                        parts.append(f"  {a['fe_name']}: {a['filler_text']}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines) if lines else "(no structured facts retrieved)"

    async def _build_text_passages(self, chunk_hits: list[dict]) -> str:
        passages = []
        for hit in chunk_hits:
            cid = hit.get("id", "")
            chunk_data = await self._hg.chunks.get_by_id(cid)
            if chunk_data:
                passages.append(chunk_data.get("text", ""))
        return "\n\n---\n\n".join(passages) if passages else "(no passages retrieved)"

    # ──────────────────────────────────────────────────────────────────────────
    # Batch insert
    # ──────────────────────────────────────────────────────────────────────────

    async def ainsert_batch(
        self,
        texts: list[str],
        source_docs: Optional[list[str]] = None,
        concurrency: int = 2,
    ) -> None:
        """Insert multiple documents with controlled concurrency.

        Args:
            texts:       List of document strings to index.
            source_docs: Optional list of source doc labels (same length as texts).
            concurrency: Max number of documents indexed in parallel.
        """
        if source_docs is None:
            source_docs = [f"doc_{i}" for i in range(len(texts))]
        if len(texts) != len(source_docs):
            raise ValueError("texts and source_docs must have the same length")

        semaphore = asyncio.Semaphore(concurrency)

        async def _safe_insert(text: str, doc: str) -> None:
            async with semaphore:
                await self.ainsert(text, source_doc=doc)

        await asyncio.gather(*[_safe_insert(t, d) for t, d in zip(texts, source_docs)])

    async def adelete(self, doc_id: str) -> bool:
        """Delete a document and all its indexed data.

        Args:
            doc_id: The document ID as returned by ``DocStore.make_doc_id(source_doc)``.

        Returns:
            True if the document was found and deleted; False if not tracked.
        """
        record = await self._doc_store.get(doc_id)
        if record is None:
            return False
        if record.chunk_ids:
            await self._hg.delete_chunks(record.chunk_ids)
        await self._doc_store.delete(doc_id)
        await self.finalize()
        logger.info(
            f"[FrameRAG] Deleted doc '{record.source_doc}' ({len(record.chunk_ids)} chunks)"
        )
        return True

    async def list_documents(self, status: Optional[str] = None) -> list[dict]:
        """List all tracked documents, optionally filtered by status string.

        Valid status values: ``pending``, ``processing``, ``processed``, ``failed``.
        """
        if status:
            try:
                st = DocStatus(status)
            except ValueError:
                valid = [s.value for s in DocStatus]
                raise ValueError(f"Unknown status {status!r}. Valid: {valid}")
            records = await self._doc_store.list_by_status(st)
        else:
            records = await self._doc_store.list_all()
        return [r.to_dict() for r in records]

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval context (without answer generation — useful for evaluation)
    # ──────────────────────────────────────────────────────────────────────────

    async def aretrieve_context(
        self,
        query: str,
        top_chunks: Optional[int] = None,
        top_frames: Optional[int] = None,
        top_nodes: Optional[int] = None,
        diffusion_warm_up: Optional[int] = None,
    ) -> dict:
        """Return raw retrieval context without calling the answer LLM.

        Returns dict with keys:
            structured_facts : str   — formatted frame instance context
            text_passages    : str   — formatted chunk context
            frame_hits       : list  — raw frame instance hits with scores
            chunk_hits       : list  — raw chunk hits with scores
        """
        frame_hits, chunk_hits = await self._retrieve(
            query,
            top_chunks=top_chunks or self._top_chunks,
            top_frames=top_frames or self._top_frames,
            top_nodes=top_nodes or self._top_nodes,
            diffusion_warm_up=diffusion_warm_up or self._diffusion_warm_up,
        )
        structured_facts = await self._build_structured_facts(frame_hits)
        text_passages    = await self._build_text_passages(chunk_hits)
        return {
            "structured_facts": structured_facts,
            "text_passages":    text_passages,
            "frame_hits":       frame_hits,
            "chunk_hits":       chunk_hits,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Introspection / statistics
    # ──────────────────────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        """Return a summary of all indexed data."""
        all_frames = await self._frame_db.all_frames()
        return {
            "chunks":           len(self._hg._chunk_ids),
            "entity_mentions":  len(self._hg._mention_ids),
            "canonical_entities": len(self._hg._canonical_ids),
            "events":           len(self._hg._event_ids),
            "frame_instances":  len(self._hg._fi_ids),
            "info_nodes":       len(self._hg._info_ids),
            "causal_edges":     len(self._hg._adj_causal),
            "frames_in_db":     len(self._frame_db._frame_names),
            "top_frames_by_usage": sorted(
                [
                    {"frame_name": f.frame_name, "usage_count": f.usage_count}
                    for f in all_frames
                ],
                key=lambda x: x["usage_count"],
                reverse=True,
            )[:10],
        }

    async def get_canonical_entity(self, name: str) -> Optional[dict]:
        """Look up a canonical entity by name."""
        cid = self._hg.get_canonical_id_by_name(name)
        if not cid:
            return None
        return await self._hg.canonical_entities.get_by_id(cid)

    async def get_frame_instances_for_entity(
        self, entity_name: str, top_k: int = 10
    ) -> list[dict]:
        """Return frame instances where the given entity is a participant."""
        cid = self._hg.get_canonical_id_by_name(entity_name)
        if not cid:
            return []
        mention_ids = await self._hg.get_all_mention_ids_for_canonical(cid)
        seen_fis: set[str] = set()
        result: list[dict] = []
        for mid in mention_ids:
            for fi_id, adj_list in self._hg._adj_frame_node.items():
                if fi_id in seen_fis:
                    continue
                for entry in adj_list:
                    if entry["node_id"] == mid:
                        fi_data = await self._hg.frame_instances.get_by_id(fi_id)
                        if fi_data:
                            seen_fis.add(fi_id)
                            result.append(fi_data)
                            break
                if len(result) >= top_k:
                    break
        return result[:top_k]

    async def get_causal_chain(
        self, start_event_trigger: str, depth: int = 3
    ) -> list[dict]:
        """Trace a causal chain starting from events matching a trigger word.

        Returns list of (source_event, relation, target_event) dicts.
        """
        causal_map: dict[str, list[dict]] = {}
        for edge in self._hg._adj_causal:
            src = edge["source"]
            if src not in causal_map:
                causal_map[src] = []
            causal_map[src].append(edge)

        # Find seed events by trigger word
        trigger_low = start_event_trigger.lower()
        seeds: list[str] = []
        for eid in self._hg._event_ids:
            ev = await self._hg.events.get_by_id(eid)
            if ev and trigger_low in ev.get("trigger_lemma", "").lower():
                seeds.append(eid)

        chain: list[dict] = []
        visited: set[str] = set()
        queue = [(s, 0) for s in seeds]

        while queue:
            eid, d = queue.pop(0)
            if eid in visited or d > depth:
                continue
            visited.add(eid)
            for edge in causal_map.get(eid, []):
                src_ev  = await self._hg.events.get_by_id(edge["source"])
                tgt_ev  = await self._hg.events.get_by_id(edge["target"])
                if src_ev and tgt_ev:
                    chain.append({
                        "source":      src_ev.get("trigger_lemma", edge["source"]),
                        "relation":    "CAUSES",
                        "target":      tgt_ev.get("trigger_lemma", edge["target"]),
                        "confidence":  edge["confidence"],
                        "source_frame": src_ev.get("frame_name", ""),
                        "target_frame": tgt_ev.get("frame_name", ""),
                    })
                    queue.append((edge["target"], d + 1))
        return chain
