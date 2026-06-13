"""FrameRAG: Frame-Semantic Event Hypergraph RAG system.

Indexing pipeline (per document):
  chunk → (Call 1) entity extraction
        → (Call 2) event + frame + role extraction (with Frame DB hints)
        → (Call 3) causal edge extraction
        → entity embed + event embed + frame instance embed
        → coreference resolution (entity then event)
        → hypergraph storage

Query pipeline:
  query → (LLM) seed signal extraction
        → vector search for seed nodes (entities, frame instances, events, chunks)
        → build sparse matrices + hypergraph diffusion
        → collect top chunks / frame instances / entities
        → (LLM) answer generation
"""
from __future__ import annotations

import uuid
from typing import Callable, Awaitable, Optional

import numpy as np

from lightrag.utils import logger

from .types import (
    ChunkSchema,
    EntityMentionSchema,
    CanonicalEntitySchema,
    EventSchema,
    FrameInstanceSchema,
    InfoNodeSchema,
    CausalEdgeSchema,
    QuerySignals,
    RetrievalResult,
)
from .frame_db import FrameDatabase
from .hypergraph import HypergraphStore
from .diffusion import HypergraphDiffusion
from .coreference.entity_coref import EntityCoreferenceResolver
from .coreference.event_coref import EventCoreferenceResolver
from .operate import (
    extract_entities,
    extract_events_frames_roles,
    extract_causal_edges,
    process_query,
    generate_answer,
)


def _chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 100,
    source_doc: str = "unknown",
) -> list[ChunkSchema]:
    """Simple token-based chunking with overlap."""
    tokens = text.split()
    chunks: list[ChunkSchema] = []
    step = max(1, chunk_size - overlap)
    for i, start in enumerate(range(0, max(1, len(tokens) - overlap), step)):
        chunk_tokens = tokens[start : start + chunk_size]
        if not chunk_tokens:
            break
        content = " ".join(chunk_tokens)
        chunks.append(ChunkSchema(
            chunk_id=f"chunk_{uuid.uuid4().hex[:12]}",
            text=content,
            source_doc=source_doc,
            chunk_index=i,
            tokens=len(chunk_tokens),
        ))
        if start + chunk_size >= len(tokens):
            break
    return chunks


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
    """

    def __init__(
        self,
        working_dir: str,
        llm_func: Callable[..., Awaitable[str]],
        embed_func: Callable[[list[str]], Awaitable[np.ndarray]],
        embedding_dim: int = 1536,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        enable_causal: bool = True,
        enable_event_coref: bool = True,
        diffusion_steps: int = 3,
        diffusion_alpha: float = 0.15,
        top_chunks: int = 20,
        top_frames: int = 10,
        top_nodes: int = 15,
    ):
        self._llm = llm_func
        self._embed = embed_func
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._enable_causal = enable_causal
        self._enable_event_coref = enable_event_coref
        self._diffusion_steps = diffusion_steps
        self._diffusion_alpha = diffusion_alpha
        self._top_chunks = top_chunks
        self._top_frames = top_frames
        self._top_nodes = top_nodes

        self._hg = HypergraphStore(working_dir, embedding_dim)
        self._frame_db = FrameDatabase(working_dir, embed_func, embedding_dim)
        self._entity_coref = EntityCoreferenceResolver(embed_func, llm_func)
        self._event_coref = EventCoreferenceResolver(embed_func, llm_func)

    async def initialize(self) -> None:
        """Initialize all storage backends."""
        await self._hg.initialize()
        await self._frame_db.initialize()
        logger.info("[FrameRAG] Initialized")

    async def finalize(self) -> None:
        """Persist all storage backends to disk."""
        await self._hg.save()
        await self._frame_db.save()
        logger.info("[FrameRAG] Saved all stores")

    # ──────────────────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────────────────

    async def ainsert(
        self,
        text: str,
        source_doc: str = "unknown",
    ) -> None:
        """Index a single document into the hypergraph."""
        chunks = _chunk_text(text, self._chunk_size, self._chunk_overlap, source_doc)
        logger.info(f"[FrameRAG] Inserting '{source_doc}': {len(chunks)} chunks")

        all_mentions: list[EntityMentionSchema] = []
        all_events: list[EventSchema] = []
        all_fi: list[FrameInstanceSchema] = []
        all_info: list[InfoNodeSchema] = []
        all_causal: list[CausalEdgeSchema] = []

        # Per-chunk extraction
        for chunk in chunks:
            # Embed and store chunk
            chunk_emb = await self._embed([chunk.text])
            await self._hg.add_chunk(chunk, chunk_emb[0])

            # Call 1: Entity extraction
            mentions = await extract_entities(chunk, self._llm)

            # Embed entity mentions
            if mentions:
                emb_texts = [f"{m.name} [SEP] {m.description}" for m in mentions]
                embs = await self._embed(emb_texts)
                for m, emb in zip(mentions, embs):
                    m.embedding = emb.tolist()

            # Call 2: Event + Frame + Role extraction
            events, fis, infos, new_frames = await extract_events_frames_roles(
                chunk, mentions, self._llm, self._frame_db
            )

            # Upsert new frames into Frame DB
            for frame_def in new_frames:
                await self._frame_db.upsert_frame(frame_def)

            # Increment usage for reused frames
            reused_frames = {ev.frame_name for ev in events} - {f.frame_name for f in new_frames}
            for fname in reused_frames:
                await self._frame_db.increment_usage(fname)

            # Embed events and frame instances
            if events:
                ev_texts = [
                    f"{ev.trigger_lemma} [{ev.frame_name}]: {ev.event_description}"
                    for ev in events
                ]
                ev_embs = await self._embed(ev_texts)
                for ev, emb in zip(events, ev_embs):
                    ev.embedding = emb.tolist()

            if fis:
                fi_texts = []
                for fi in fis:
                    parts = [f"Frame: {fi.frame_name}"]
                    for a in fi.core_assignments + fi.noncore_assignments:
                        if a.filler_id and a.filler_text:
                            parts.append(f"{a.fe_name}: {a.filler_text}")
                    fi_texts.append(" | ".join(parts))
                fi_embs = await self._embed(fi_texts)
                for fi, emb in zip(fis, fi_embs):
                    fi.embedding = emb.tolist()

            # Embed info nodes
            if infos:
                info_texts = [info.value for info in infos]
                info_embs = await self._embed(info_texts)
                for info, emb in zip(infos, info_embs):
                    info.embedding = emb.tolist()

            # Call 3: Causal edges (optional)
            causal_edges: list[CausalEdgeSchema] = []
            if self._enable_causal and len(events) >= 2:
                causal_edges = await extract_causal_edges(chunk, events, self._llm)

            all_mentions.extend(mentions)
            all_events.extend(events)
            all_fi.extend(fis)
            all_info.extend(infos)
            all_causal.extend(causal_edges)

        # Entity coreference resolution (across all chunks)
        logger.info(f"[FrameRAG] Running entity coref on {len(all_mentions)} mentions")
        canonicals = await self._entity_coref.resolve_with_llm_verify(all_mentions)

        # Event coreference resolution (across all chunks)
        if self._enable_event_coref and all_events:
            logger.info(f"[FrameRAG] Running event coref on {len(all_events)} events")
            event_to_canon = await self._event_coref.resolve(all_events, canonicals)
            for ev in all_events:
                ev.canonical_event_id = event_to_canon.get(ev.event_id, ev.event_id)

        # Build Frame DB core FE name lookup for hypergraph FE weight computation
        frame_core_fes: dict[str, set[str]] = {}
        for ev in all_events:
            if ev.frame_name not in frame_core_fes:
                frame_def = await self._frame_db.get(ev.frame_name)
                if frame_def:
                    frame_core_fes[ev.frame_name] = frame_def.core_fe_names()
                else:
                    frame_core_fes[ev.frame_name] = set()

        # Persist to hypergraph
        for m in all_mentions:
            await self._hg.add_entity_mention(m)

        for canon in canonicals:
            await self._hg.add_canonical_entity(canon)

        for ev in all_events:
            await self._hg.add_event(ev)

        for fi in all_fi:
            core_fe_names = frame_core_fes.get(fi.frame_name, set())
            await self._hg.add_frame_instance(fi, core_fe_names)

        for info in all_info:
            await self._hg.add_info_node(info)

        for edge in all_causal:
            await self._hg.add_causal_edge(edge)

        await self.finalize()
        logger.info(f"[FrameRAG] Indexed '{source_doc}': "
                    f"{len(all_mentions)} mentions, {len(canonicals)} canonicals, "
                    f"{len(all_events)} events, {len(all_fi)} frame instances")

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────────

    async def aquery(
        self,
        query: str,
        top_chunks: Optional[int] = None,
        top_frames: Optional[int] = None,
        top_nodes: Optional[int] = None,
        diffusion_steps: Optional[int] = None,
    ) -> str:
        """Answer a natural language query using hypergraph diffusion retrieval."""
        top_chunks = top_chunks or self._top_chunks
        top_frames = top_frames or self._top_frames
        top_nodes = top_nodes or self._top_nodes
        T = diffusion_steps or self._diffusion_steps

        # Step 1: Extract query signals
        signals_raw = await process_query(query, self._llm)
        signals = QuerySignals(
            entity_hints=signals_raw.get("entity_hints", []),
            event_hints=signals_raw.get("event_hints", []),
            frame_hints=signals_raw.get("frame_hints", ""),
            fe_focus=signals_raw.get("fe_focus", []),
            temporal_hints=signals_raw.get("temporal_hints", []),
        )

        # Step 2: Build sparse matrices
        matrices = await self._hg.build_matrices(fe_focus=signals.fe_focus or None)
        if not matrices:
            return "No indexed documents found. Please insert documents first."

        n_nodes = len(matrices["node_ids"])
        n_frames = len(matrices["fi_ids"])
        n_chunks = len(matrices["chunk_ids"])

        # Step 3: Build seed vectors via vector search
        query_emb = await self._embed([query])
        q_vec = query_emb[0]

        y_node = np.zeros(n_nodes, dtype=np.float64)
        y_fi = np.zeros(n_frames, dtype=np.float64)
        y_chunk = np.zeros(n_chunks, dtype=np.float64)

        node_idx = matrices["node_idx"]
        fi_idx = matrices["fi_idx"]
        chunk_idx = matrices["chunk_idx"]

        # Seed from entity mention similarity
        entity_hits = await self._hg.vdb_canonical_entities.search(q_vec, top_k=20)
        for hit in entity_hits:
            # canonical entity → all mention ids
            mention_ids_for_canon = await self._hg.get_all_mention_ids_for_canonical(
                hit["id"] if "id" in hit else hit.get("canonical_id", "")
            )
            for mid in mention_ids_for_canon:
                if mid in node_idx:
                    y_node[node_idx[mid]] += hit.get("score", 0.0)

        # Seed from event/frame instance similarity
        fi_hits = await self._hg.vdb_frame_instances.search(q_vec, top_k=20)
        for hit in fi_hits:
            fid = hit.get("id", hit.get("fi_id", ""))
            if fid in fi_idx:
                y_fi[fi_idx[fid]] += hit.get("score", 0.0)

        # Seed from chunk similarity
        chunk_hits = await self._hg.vdb_chunks.search(q_vec, top_k=10)
        for hit in chunk_hits:
            cid = hit.get("id", hit.get("chunk_id", ""))
            if cid in chunk_idx:
                y_chunk[chunk_idx[cid]] += hit.get("score", 0.0)

        # Seed from info node similarity (temporal hints etc.)
        if signals.temporal_hints:
            temporal_query = " ".join(signals.temporal_hints)
            t_emb = await self._embed([temporal_query])
            info_hits = await self._hg.vdb_info_nodes.search(t_emb[0], top_k=10)
            for hit in info_hits:
                iid = hit.get("id", hit.get("info_id", ""))
                if iid in node_idx:
                    y_node[node_idx[iid]] += hit.get("score", 0.0) * 0.5

        # Normalize seeds
        def _safe_l1(v: np.ndarray) -> np.ndarray:
            s = v.sum()
            return v / s if s > 1e-12 else v

        y_node = _safe_l1(y_node)
        y_fi = _safe_l1(y_fi)
        y_chunk = _safe_l1(y_chunk)

        # Step 4: Hypergraph diffusion
        diffusion = HypergraphDiffusion(matrices)
        f_node, f_fi, f_chunk = diffusion.run(
            y_node, y_fi, y_chunk,
            alpha=self._diffusion_alpha,
            T=T,
        )

        # Step 5: Extract top results
        results = diffusion.top_results(
            f_node, f_fi, f_chunk,
            top_chunks=top_chunks,
            top_frames=top_frames,
            top_nodes=top_nodes,
        )

        # Step 6: Build context for answer generation
        structured_facts = await self._build_structured_facts(results["frame_hits"])
        text_passages = await self._build_text_passages(results["chunk_hits"])

        # Step 7: Generate answer
        answer = await generate_answer(query, structured_facts, text_passages, self._llm)
        return answer

    # ──────────────────────────────────────────────────────────────────────────
    # Context building helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _build_structured_facts(self, frame_hits: list[dict]) -> str:
        lines = []
        for i, hit in enumerate(frame_hits, 1):
            fid = hit.get("id", "")
            fi_data = await self._hg.frame_instances.get(fid)
            if not fi_data:
                continue
            frame_name = fi_data.get("frame_name", "")
            lu = fi_data.get("lexical_unit", "")
            parts = [f"[{i}] Frame: {frame_name} (LU: {lu})"]
            for slot_key in ("core_assignments", "noncore_assignments"):
                for a in fi_data.get(slot_key, []):
                    if a.get("filler_text"):
                        parts.append(f"  {a['fe_name']}: {a['filler_text']}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines) if lines else "(no structured facts retrieved)"

    async def _build_text_passages(self, chunk_hits: list[dict]) -> str:
        passages = []
        for hit in chunk_hits:
            cid = hit.get("id", "")
            chunk_data = await self._hg.chunks.get(cid)
            if chunk_data:
                passages.append(chunk_data.get("text", ""))
        return "\n\n---\n\n".join(passages) if passages else "(no passages retrieved)"
