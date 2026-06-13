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
        → coreference resolution (entity then event)
        → entity description merging (cross-doc)
        → hypergraph storage

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
from .coreference.event_coref import EventCoreferenceResolver
from .operate import (
    extract_entities,
    glean_entities,
    extract_events_frames_roles,
    extract_causal_edges,
    expand_query_frames,
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

    _DESC_MERGE_THRESHOLD = 3_500   # chars; summarise when descriptions exceed this

    def __init__(
        self,
        working_dir: str,
        llm_func: Callable[..., Awaitable[str]],
        embed_func: Callable[[list[str]], Awaitable[np.ndarray]],
        embedding_dim: int = 1536,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        enable_causal: bool = True,
        enable_gleaning: bool = True,
        enable_event_coref: bool = True,
        diffusion_steps: int = 3,
        diffusion_alpha: float = 0.15,
        top_chunks: int = 20,
        top_frames: int = 10,
        top_nodes: int = 15,
    ):
        self._raw_llm = llm_func
        self._raw_embed = embed_func
        self._embedding_dim = embedding_dim
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._enable_causal = enable_causal
        self._enable_gleaning = enable_gleaning
        self._enable_event_coref = enable_event_coref
        self._diffusion_steps = diffusion_steps
        self._diffusion_alpha = diffusion_alpha
        self._top_chunks = top_chunks
        self._top_frames = top_frames
        self._top_nodes = top_nodes

        self._working_dir = working_dir
        os.makedirs(working_dir, exist_ok=True)

        ef: EmbeddingFunc = wrap_embed(embed_func, embedding_dim)

        self._hg         = HypergraphStore(working_dir, ef)
        self._frame_db   = FrameDatabase(working_dir, ef)
        self._entity_coref = EntityCoreferenceResolver(embed_func, llm_func)
        self._event_coref  = EventCoreferenceResolver(embed_func, llm_func)

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
        logger.info("[FrameRAG] Initialized")

    async def finalize(self) -> None:
        await self._hg.index_done_callback()
        await self._frame_db.index_done_callback()
        await self._llm_cache.index_done_callback()
        logger.info("[FrameRAG] Storage flushed")

    # ──────────────────────────────────────────────────────────────────────────
    # LLM with caching
    # ──────────────────────────────────────────────────────────────────────────

    async def _llm(self, prompt: str) -> str:
        """Call LLM with JSON KV cache (MD5 key)."""
        cache_key = compute_mdhash_id(prompt, prefix="llm-")
        cached = await self._llm_cache.get_by_id(cache_key)
        if cached and cached.get("response"):
            return cached["response"]
        response = await _llm_with_retry(self._raw_llm, prompt)
        await self._llm_cache.upsert({cache_key: {"response": response}})
        return response

    async def _llm_stream(self, prompt: str) -> AsyncIterator[str]:
        """Streaming LLM call (falls back to non-streaming if not supported)."""
        try:
            result = self._raw_llm(prompt, stream=True)
            if hasattr(result, "__aiter__"):
                async for chunk in result:
                    yield chunk
                return
            text = await result
            yield text
        except TypeError:
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
        chunks = self._chunk_text(text, source_doc)
        logger.info(f"[FrameRAG] Inserting '{source_doc}': {len(chunks)} chunks")

        all_mentions: list[EntityMentionSchema] = []
        all_events:   list[EventSchema]          = []
        all_fi:       list[FrameInstanceSchema]  = []
        all_info:     list[InfoNodeSchema]       = []
        all_causal:   list[CausalEdgeSchema]     = []

        # ── Per-chunk extraction ─────────────────────────────────────────────
        for chunk in chunks:
            # Store chunk (NanoVDB auto-embeds the text content)
            await self._hg.add_chunk(chunk)

            # Call 1: entity extraction (cached)
            mentions = await extract_entities(chunk, self._llm)
            if self._enable_gleaning and mentions:
                gleaned = await glean_entities(chunk, mentions, self._llm)
                mentions = mentions + gleaned

            # Call 2: event + frame + role (cached)
            events, fis, infos, new_frames = await extract_events_frames_roles(
                chunk, mentions, self._llm, self._frame_db
            )

            # Update Frame DB
            for fd in new_frames:
                await self._frame_db.upsert_frame(fd)
            reused = {ev.frame_name for ev in events} - {f.frame_name for f in new_frames}
            for fname in reused:
                await self._frame_db.increment_usage(fname)

            # Call 3: causal edges (cached)
            causal_edges: list[CausalEdgeSchema] = []
            if self._enable_causal and len(events) >= 2:
                causal_edges = await extract_causal_edges(chunk, events, self._llm)

            all_mentions.extend(mentions)
            all_events.extend(events)
            all_fi.extend(fis)
            all_info.extend(infos)
            all_causal.extend(causal_edges)

        # ── Entity coreference ───────────────────────────────────────────────
        logger.info(f"[FrameRAG] Entity coref: {len(all_mentions)} mentions")
        canonicals = await self._entity_coref.resolve_with_llm_verify(all_mentions)

        # ── Cross-document entity description merging ────────────────────────
        canonicals = await self._merge_entity_descriptions(canonicals)

        # ── Event coreference ────────────────────────────────────────────────
        if self._enable_event_coref and all_events:
            logger.info(f"[FrameRAG] Event coref: {len(all_events)} events")
            ev_to_canon = await self._event_coref.resolve(all_events, canonicals)
            for ev in all_events:
                ev.canonical_event_id = ev_to_canon.get(ev.event_id, ev.event_id)

        # ── Persist to hypergraph ────────────────────────────────────────────
        for m in all_mentions:
            await self._hg.add_entity_mention(m)
        for canon in canonicals:
            await self._hg.add_canonical_entity(canon)
        for ev in all_events:
            await self._hg.add_event(ev)

        frame_core_fes: dict[str, set[str]] = {}
        for ev in all_events:
            if ev.frame_name not in frame_core_fes:
                fd = await self._frame_db.get(ev.frame_name)
                frame_core_fes[ev.frame_name] = fd.core_fe_names() if fd else set()

        for fi in all_fi:
            core_names = frame_core_fes.get(fi.frame_name, set())
            await self._hg.add_frame_instance(fi, core_names)
        for info in all_info:
            await self._hg.add_info_node(info)
        for edge in all_causal:
            await self._hg.add_causal_edge(edge)

        await self.finalize()
        logger.info(
            f"[FrameRAG] Indexed '{source_doc}': {len(all_mentions)} mentions, "
            f"{len(canonicals)} canonicals, {len(all_events)} events, "
            f"{len(all_fi)} frame instances"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Entity description merging (cross-document)
    # ──────────────────────────────────────────────────────────────────────────

    async def _merge_entity_descriptions(
        self, canonicals: list[CanonicalEntitySchema]
    ) -> list[CanonicalEntitySchema]:
        """Merge each canonical's descriptions; summarise if over threshold."""
        for canon in canonicals:
            # Check whether this entity is already in storage (previous ainsert)
            existing_id = self._hg.get_canonical_id_by_name(canon.canonical_name)
            if existing_id and existing_id != canon.canonical_id:
                existing = await self._hg.canonical_entities.get_by_id(existing_id)
                if existing:
                    all_descs = list(
                        dict.fromkeys(
                            existing.get("descriptions", []) + canon.descriptions
                        )
                    )
                    if sum(len(d) for d in all_descs) > self._DESC_MERGE_THRESHOLD:
                        merged = await self._summarise_descriptions(
                            canon.canonical_name, all_descs
                        )
                        all_descs = [merged]
                    await self._hg.update_canonical_descriptions(existing_id, all_descs)
                    canon.canonical_id = existing_id
                    canon.descriptions = all_descs
                    continue

            # New canonical: check if descriptions alone are already too long
            if sum(len(d) for d in canon.descriptions) > self._DESC_MERGE_THRESHOLD:
                merged = await self._summarise_descriptions(
                    canon.canonical_name, canon.descriptions
                )
                canon.descriptions = [merged]
        return canonicals

    async def _summarise_descriptions(
        self, entity_name: str, descriptions: list[str]
    ) -> str:
        desc_text = "\n".join(f"- {d}" for d in descriptions)
        prompt = PROMPTS["entity_description_merge"].format(
            entity_name=entity_name,
            descriptions=desc_text,
        )
        try:
            return await self._llm(prompt)
        except Exception as e:
            logger.warning(f"[FrameRAG] Description merge failed for {entity_name}: {e}")
            return descriptions[0] if descriptions else ""

    # ──────────────────────────────────────────────────────────────────────────
    # Query
    # ──────────────────────────────────────────────────────────────────────────

    async def _retrieve(
        self,
        query: str,
        top_chunks: int,
        top_frames: int,
        top_nodes: int,
        diffusion_steps: int,
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

        # Frame expansion for broader coverage
        expanded_frames: list[str] = []
        if signals.frame_hints:
            expanded_frames = await expand_query_frames(
                query, signals.frame_hints, self._llm
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
        for hit in await self._hg.search_canonical(q_vec, top_k=20):
            for mid in await self._hg.get_all_mention_ids_for_canonical(hit["id"]):
                if mid in node_idx:
                    y_node[node_idx[mid]] += hit.get("distance", 0.0)

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
        for hit in await self._hg.search_chunks(q_vec, top_k=10):
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
            T=diffusion_steps,
        )

        # Step 5: Top results
        results = diffusion.top_results(
            f_node, f_fi, f_chunk,
            top_chunks=top_chunks,
            top_frames=top_frames,
            top_nodes=top_nodes,
        )
        return results["frame_hits"], results["chunk_hits"]

    async def aquery(
        self,
        query: str,
        top_chunks: Optional[int] = None,
        top_frames: Optional[int] = None,
        top_nodes: Optional[int] = None,
        diffusion_steps: Optional[int] = None,
    ) -> str:
        """Answer a query using hypergraph diffusion retrieval."""
        frame_hits, chunk_hits = await self._retrieve(
            query,
            top_chunks=top_chunks or self._top_chunks,
            top_frames=top_frames or self._top_frames,
            top_nodes=top_nodes or self._top_nodes,
            diffusion_steps=diffusion_steps or self._diffusion_steps,
        )
        if not frame_hits and not chunk_hits:
            return "No indexed documents found. Please insert documents first."

        structured_facts = await self._build_structured_facts(frame_hits)
        text_passages    = await self._build_text_passages(chunk_hits)
        return await generate_answer(query, structured_facts, text_passages, self._llm)

    async def aquery_stream(
        self,
        query: str,
        top_chunks: Optional[int] = None,
        top_frames: Optional[int] = None,
        top_nodes: Optional[int] = None,
        diffusion_steps: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Streaming version of aquery — yields answer tokens as they arrive."""
        frame_hits, chunk_hits = await self._retrieve(
            query,
            top_chunks=top_chunks or self._top_chunks,
            top_frames=top_frames or self._top_frames,
            top_nodes=top_nodes or self._top_nodes,
            diffusion_steps=diffusion_steps or self._diffusion_steps,
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
        for i, hit in enumerate(frame_hits, 1):
            fid = hit.get("id", "")
            fi_data = await self._hg.frame_instances.get_by_id(fid)
            if not fi_data:
                continue
            parts = [f"[{i}] Frame: {fi_data.get('frame_name','')} (LU: {fi_data.get('lexical_unit','')})"]
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
                try:
                    await self.ainsert(text, source_doc=doc)
                except Exception as e:
                    logger.error(f"[FrameRAG] Batch insert failed for '{doc}': {e}")

        await asyncio.gather(*[_safe_insert(t, d) for t, d in zip(texts, source_docs)])

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval context (without answer generation — useful for evaluation)
    # ──────────────────────────────────────────────────────────────────────────

    async def aretrieve_context(
        self,
        query: str,
        top_chunks: Optional[int] = None,
        top_frames: Optional[int] = None,
        top_nodes: Optional[int] = None,
        diffusion_steps: Optional[int] = None,
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
            diffusion_steps=diffusion_steps or self._diffusion_steps,
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
