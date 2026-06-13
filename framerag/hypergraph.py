"""Hypergraph storage: adjacency lists + sparse matrix construction for diffusion."""
from __future__ import annotations

import os
import json
from typing import Optional

import numpy as np
from scipy import sparse

from .storage import SimpleKVStore, SimpleVectorStore
from .types import (
    ChunkSchema,
    EntityMentionSchema,
    CanonicalEntitySchema,
    EventSchema,
    FrameInstanceSchema,
    InfoNodeSchema,
    CausalEdgeSchema,
    FEAssignment,
)


# Weights for FE roles in the incidence matrix
CORE_FE_WEIGHT = 1.0
NONCORE_FE_WEIGHT = 0.4
FE_FOCUS_BOOST = 1.5     # multiplier when FE matches query focus


class HypergraphStore:
    """Stores all nodes and edges; builds scipy sparse matrices on demand.

    Directory layout under working_dir/hypergraph/:
      chunks.json           SimpleKVStore  chunk_id  → ChunkSchema dict
      entity_mentions.json  SimpleKVStore  mention_id → EntityMentionSchema dict
      canonical_entities.json SimpleKVStore canonical_id → CanonicalEntitySchema dict
      events.json           SimpleKVStore  event_id  → EventSchema dict
      frame_instances.json  SimpleKVStore  fi_id     → FrameInstanceSchema dict
      info_nodes.json       SimpleKVStore  info_id   → InfoNodeSchema dict
      causal_edges.json     SimpleKVStore  edge_id   → CausalEdgeSchema dict

      adj_chunk_event.json  {chunk_id: [event_id, ...]}
      adj_event_frame.json  {event_id: [fi_id, ...]}
      adj_frame_node.json   {fi_id: [{node_id, fe_name, weight, is_core},...]}
      adj_mention_canon.json {mention_id: canonical_id}
      adj_causal.json       [{source, target, confidence}, ...]

    Vector stores (for embedding search):
      vdb_entity_mentions.json
      vdb_canonical_entities.json
      vdb_events.json
      vdb_frame_instances.json
      vdb_info_nodes.json
      vdb_chunks.json
    """

    def __init__(self, working_dir: str, embedding_dim: int = 1536):
        hg_dir = os.path.join(working_dir, "hypergraph")
        os.makedirs(hg_dir, exist_ok=True)
        self._dir = hg_dir
        self._dim = embedding_dim

        # KV stores
        self.chunks = SimpleKVStore(os.path.join(hg_dir, "chunks.json"))
        self.entity_mentions = SimpleKVStore(os.path.join(hg_dir, "entity_mentions.json"))
        self.canonical_entities = SimpleKVStore(os.path.join(hg_dir, "canonical_entities.json"))
        self.events = SimpleKVStore(os.path.join(hg_dir, "events.json"))
        self.frame_instances = SimpleKVStore(os.path.join(hg_dir, "frame_instances.json"))
        self.info_nodes = SimpleKVStore(os.path.join(hg_dir, "info_nodes.json"))
        self.causal_edges = SimpleKVStore(os.path.join(hg_dir, "causal_edges.json"))

        # Vector stores
        self.vdb_entity_mentions = SimpleVectorStore(
            os.path.join(hg_dir, "vdb_entity_mentions.json"), embedding_dim
        )
        self.vdb_canonical_entities = SimpleVectorStore(
            os.path.join(hg_dir, "vdb_canonical_entities.json"), embedding_dim
        )
        self.vdb_events = SimpleVectorStore(
            os.path.join(hg_dir, "vdb_events.json"), embedding_dim
        )
        self.vdb_frame_instances = SimpleVectorStore(
            os.path.join(hg_dir, "vdb_frame_instances.json"), embedding_dim
        )
        self.vdb_info_nodes = SimpleVectorStore(
            os.path.join(hg_dir, "vdb_info_nodes.json"), embedding_dim
        )
        self.vdb_chunks = SimpleVectorStore(
            os.path.join(hg_dir, "vdb_chunks.json"), embedding_dim
        )

        # Adjacency lists (loaded into memory, saved to JSON)
        self._adj_chunk_event: dict[str, list[str]] = {}      # chunk_id → [event_id]
        self._adj_event_frame: dict[str, list[str]] = {}      # event_id → [fi_id]
        self._adj_frame_node: dict[str, list[dict]] = {}      # fi_id → [{node_id, fe_name, weight, is_core}]
        self._adj_mention_canon: dict[str, str] = {}          # mention_id → canonical_id
        self._adj_causal: list[dict] = []                     # [{source, target, confidence}]

    # ──────────────────────────────────────────────────────────────────────────
    # Init / Persistence
    # ──────────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        for store in [
            self.chunks, self.entity_mentions, self.canonical_entities,
            self.events, self.frame_instances, self.info_nodes, self.causal_edges,
        ]:
            await store.initialize()
        for vdb in [
            self.vdb_entity_mentions, self.vdb_canonical_entities, self.vdb_events,
            self.vdb_frame_instances, self.vdb_info_nodes, self.vdb_chunks,
        ]:
            await vdb.initialize()
        self._load_adj()

    def _adj_path(self, name: str) -> str:
        return os.path.join(self._dir, f"adj_{name}.json")

    def _load_adj(self) -> None:
        for attr, fname in [
            ("_adj_chunk_event", "chunk_event"),
            ("_adj_event_frame", "event_frame"),
            ("_adj_frame_node",  "frame_node"),
            ("_adj_mention_canon", "mention_canon"),
            ("_adj_causal",      "causal"),
        ]:
            path = self._adj_path(fname)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    setattr(self, attr, json.load(f))

    async def save(self) -> None:
        for store in [
            self.chunks, self.entity_mentions, self.canonical_entities,
            self.events, self.frame_instances, self.info_nodes, self.causal_edges,
        ]:
            await store.save()
        for vdb in [
            self.vdb_entity_mentions, self.vdb_canonical_entities, self.vdb_events,
            self.vdb_frame_instances, self.vdb_info_nodes, self.vdb_chunks,
        ]:
            await vdb.save()
        for attr, fname in [
            ("_adj_chunk_event", "chunk_event"),
            ("_adj_event_frame", "event_frame"),
            ("_adj_frame_node",  "frame_node"),
            ("_adj_mention_canon", "mention_canon"),
            ("_adj_causal",      "causal"),
        ]:
            path = self._adj_path(fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(getattr(self, attr), f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    # Add nodes
    # ──────────────────────────────────────────────────────────────────────────

    async def add_chunk(self, chunk: ChunkSchema, embedding: np.ndarray) -> None:
        data = {"chunk_id": chunk.chunk_id, "text": chunk.text,
                "source_doc": chunk.source_doc, "chunk_index": chunk.chunk_index,
                "tokens": chunk.tokens}
        await self.chunks.set(chunk.chunk_id, data)
        await self.vdb_chunks.upsert(chunk.chunk_id, embedding.tolist(),
                                      metadata={"text": chunk.text,
                                                "source_doc": chunk.source_doc})

    async def add_entity_mention(self, mention: EntityMentionSchema) -> None:
        data = {
            "mention_id": mention.mention_id,
            "chunk_id": mention.chunk_id,
            "name": mention.name,
            "entity_type": mention.entity_type,
            "description": mention.description,
            "aliases": mention.aliases,
            "salience": mention.salience,
            "canonical_id": mention.canonical_id,
        }
        await self.entity_mentions.set(mention.mention_id, data)
        if mention.embedding:
            await self.vdb_entity_mentions.upsert(
                mention.mention_id, mention.embedding,
                metadata={"name": mention.name, "description": mention.description,
                          "entity_type": mention.entity_type}
            )

    async def add_canonical_entity(self, canonical: CanonicalEntitySchema) -> None:
        data = {
            "canonical_id": canonical.canonical_id,
            "canonical_name": canonical.canonical_name,
            "entity_type": canonical.entity_type,
            "descriptions": canonical.descriptions,
            "mention_ids": canonical.mention_ids,
        }
        await self.canonical_entities.set(canonical.canonical_id, data)
        if canonical.embedding:
            await self.vdb_canonical_entities.upsert(
                canonical.canonical_id, canonical.embedding,
                metadata={"canonical_name": canonical.canonical_name,
                          "entity_type": canonical.entity_type,
                          "mention_ids": canonical.mention_ids}
            )
        # Update mention → canonical mapping
        for mid in canonical.mention_ids:
            self._adj_mention_canon[mid] = canonical.canonical_id
            mention_data = await self.entity_mentions.get(mid)
            if mention_data:
                mention_data["canonical_id"] = canonical.canonical_id
                await self.entity_mentions.set(mid, mention_data)

    async def add_event(self, event: EventSchema) -> None:
        data = {
            "event_id": event.event_id,
            "chunk_id": event.chunk_id,
            "trigger": event.trigger,
            "trigger_lemma": event.trigger_lemma,
            "trigger_pos": event.trigger_pos,
            "event_span": event.event_span,
            "event_description": event.event_description,
            "frame_name": event.frame_name,
            "participant_mention_ids": event.participant_mention_ids,
            "frame_instance_ids": event.frame_instance_ids,
            "canonical_event_id": event.canonical_event_id,
        }
        await self.events.set(event.event_id, data)
        if event.embedding:
            await self.vdb_events.upsert(
                event.event_id, event.embedding,
                metadata={"trigger_lemma": event.trigger_lemma,
                          "event_description": event.event_description}
            )
        # Chunk → Event adjacency
        if event.chunk_id not in self._adj_chunk_event:
            self._adj_chunk_event[event.chunk_id] = []
        if event.event_id not in self._adj_chunk_event[event.chunk_id]:
            self._adj_chunk_event[event.chunk_id].append(event.event_id)

    async def add_frame_instance(
        self,
        fi: FrameInstanceSchema,
        frame_core_fe_names: set[str],
    ) -> None:
        data = {
            "fi_id": fi.fi_id,
            "event_id": fi.event_id,
            "frame_name": fi.frame_name,
            "lexical_unit": fi.lexical_unit,
            "core_assignments": [
                {"fe_name": a.fe_name, "filler_id": a.filler_id,
                 "filler_type": a.filler_type, "filler_text": a.filler_text,
                 "is_core": a.is_core}
                for a in fi.core_assignments
            ],
            "noncore_assignments": [
                {"fe_name": a.fe_name, "filler_id": a.filler_id,
                 "filler_type": a.filler_type, "filler_text": a.filler_text,
                 "is_core": a.is_core}
                for a in fi.noncore_assignments
            ],
        }
        await self.frame_instances.set(fi.fi_id, data)
        if fi.embedding:
            await self.vdb_frame_instances.upsert(
                fi.fi_id, fi.embedding,
                metadata={"frame_name": fi.frame_name, "event_id": fi.event_id,
                          "lexical_unit": fi.lexical_unit}
            )
        # Event → FrameInstance adjacency
        if fi.event_id not in self._adj_event_frame:
            self._adj_event_frame[fi.event_id] = []
        if fi.fi_id not in self._adj_event_frame[fi.event_id]:
            self._adj_event_frame[fi.event_id].append(fi.fi_id)

        # FrameInstance → Node (entity/info) adjacency with FE weights
        if fi.fi_id not in self._adj_frame_node:
            self._adj_frame_node[fi.fi_id] = []
        for assignment in fi.core_assignments + fi.noncore_assignments:
            if assignment.filler_id is None:
                continue
            is_core = assignment.fe_name in frame_core_fe_names
            weight = CORE_FE_WEIGHT if is_core else NONCORE_FE_WEIGHT
            self._adj_frame_node[fi.fi_id].append({
                "node_id": assignment.filler_id,
                "fe_name": assignment.fe_name,
                "weight": weight,
                "is_core": is_core,
            })

    async def add_info_node(self, info: InfoNodeSchema) -> None:
        data = {"info_id": info.info_id, "value": info.value, "info_type": info.info_type}
        await self.info_nodes.set(info.info_id, data)
        if info.embedding:
            await self.vdb_info_nodes.upsert(
                info.info_id, info.embedding,
                metadata={"value": info.value, "info_type": info.info_type}
            )

    async def add_causal_edge(self, edge: CausalEdgeSchema) -> None:
        data = {
            "edge_id": edge.edge_id,
            "source_event_id": edge.source_event_id,
            "target_event_id": edge.target_event_id,
            "relation_type": edge.relation_type,
            "confidence": edge.confidence,
            "evidence_span": edge.evidence_span,
        }
        await self.causal_edges.set(edge.edge_id, data)
        self._adj_causal.append({
            "source": edge.source_event_id,
            "target": edge.target_event_id,
            "confidence": edge.confidence,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # Sparse matrix construction (called at retrieval time)
    # ──────────────────────────────────────────────────────────────────────────

    async def build_matrices(
        self, fe_focus: Optional[list[str]] = None
    ) -> dict[str, sparse.csr_matrix]:
        """Build normalized sparse incidence matrices for diffusion.

        Returns dict with keys:
          H_ce  : [n_chunks × n_events]   chunk contains event
          H_ef  : [n_events × n_frames]   event instantiates frame
          H_fe  : [n_frames × n_nodes]    frame connects to entity/info (FE-weighted)
          A_cau : [n_events × n_events]   causal/temporal edges
        """
        chunk_ids = await self.chunks.all_keys()
        event_ids = await self.events.all_keys()
        fi_ids = await self.frame_instances.all_keys()
        mention_ids = await self.entity_mentions.all_keys()
        info_ids = await self.info_nodes.all_keys()

        # Combined node index: entity mentions + info nodes
        all_node_ids = mention_ids + info_ids

        if not event_ids or not fi_ids:
            return {}

        chunk_idx = {c: i for i, c in enumerate(chunk_ids)}
        event_idx = {e: i for i, e in enumerate(event_ids)}
        fi_idx    = {f: i for i, f in enumerate(fi_ids)}
        node_idx  = {n: i for i, n in enumerate(all_node_ids)}

        n_c, n_e, n_f, n_n = len(chunk_ids), len(event_ids), len(fi_ids), len(all_node_ids)

        # ── H_ce: chunk → event ──
        rows, cols = [], []
        for cid, eids in self._adj_chunk_event.items():
            if cid not in chunk_idx:
                continue
            for eid in eids:
                if eid in event_idx:
                    rows.append(chunk_idx[cid])
                    cols.append(event_idx[eid])
        H_ce = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n_c, n_e)
        ) if rows else sparse.csr_matrix((n_c, n_e))

        # ── H_ef: event → frame_instance ──
        rows, cols = [], []
        for eid, fids in self._adj_event_frame.items():
            if eid not in event_idx:
                continue
            for fid in fids:
                if fid in fi_idx:
                    rows.append(event_idx[eid])
                    cols.append(fi_idx[fid])
        H_ef = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n_e, n_f)
        ) if rows else sparse.csr_matrix((n_e, n_f))

        # ── H_fe: frame_instance → node (FE-weighted) ──
        rows, cols, data = [], [], []
        for fid, adj_list in self._adj_frame_node.items():
            if fid not in fi_idx:
                continue
            for entry in adj_list:
                nid = entry["node_id"]
                if nid not in node_idx:
                    continue
                weight = entry["weight"]
                # Boost if this FE is in query focus
                if fe_focus and entry["fe_name"] in fe_focus:
                    weight *= FE_FOCUS_BOOST
                rows.append(fi_idx[fid])
                cols.append(node_idx[nid])
                data.append(weight)
        H_fe = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_f, n_n)
        ) if data else sparse.csr_matrix((n_f, n_n))

        # ── A_cau: event → event (causal) ──
        rows, cols, data = [], [], []
        for entry in self._adj_causal:
            s, t = entry["source"], entry["target"]
            if s in event_idx and t in event_idx:
                rows.append(event_idx[s])
                cols.append(event_idx[t])
                data.append(entry["confidence"])
        A_cau = sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_e, n_e)
        ) if data else sparse.csr_matrix((n_e, n_e))

        return {
            "H_ce": H_ce, "H_ef": H_ef, "H_fe": H_fe, "A_cau": A_cau,
            "chunk_ids": chunk_ids, "event_ids": event_ids,
            "fi_ids": fi_ids, "node_ids": all_node_ids,
            "chunk_idx": chunk_idx, "event_idx": event_idx,
            "fi_idx": fi_idx, "node_idx": node_idx,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def get_events_in_chunk(self, chunk_id: str) -> list[dict]:
        eids = self._adj_chunk_event.get(chunk_id, [])
        results = []
        for eid in eids:
            ev = await self.events.get(eid)
            if ev:
                results.append(ev)
        return results

    async def get_frame_instances_for_event(self, event_id: str) -> list[dict]:
        fids = self._adj_event_frame.get(event_id, [])
        results = []
        for fid in fids:
            fi = await self.frame_instances.get(fid)
            if fi:
                results.append(fi)
        return results

    async def get_canonical_id(self, mention_id: str) -> Optional[str]:
        return self._adj_mention_canon.get(mention_id)

    async def get_all_mention_ids_for_canonical(self, canonical_id: str) -> list[str]:
        canon_data = await self.canonical_entities.get(canonical_id)
        if canon_data:
            return canon_data.get("mention_ids", [])
        return []
