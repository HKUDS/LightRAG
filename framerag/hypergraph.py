"""Hypergraph storage: adjacency lists + sparse matrix construction for diffusion.

Uses LightRAG's JsonKVStorage for KV data and NanoVectorDBStorage for vectors.
Local ID sets (persisted in ids.json) provide all_keys() semantics without
full-table scans, since JsonKVStorage doesn't expose a keys() method.

Directory layout under working_dir/hypergraph/:
  kv_store_chunks.json           chunks KV
  kv_store_entity_mentions.json  entity mention KV
  kv_store_canonical_entities.json canonical entity KV
  kv_store_events.json           event KV
  kv_store_frame_instances.json  frame instance KV
  kv_store_info_nodes.json       info node KV
  kv_store_causal_edges.json     causal edge KV

  vdb_chunk_vec.json      chunk vectors
  vdb_mention_vec.json    entity mention vectors
  vdb_canonical_vec.json  canonical entity vectors
  vdb_event_vec.json      event vectors
  vdb_fi_vec.json         frame instance vectors
  vdb_info_vec.json       info node vectors

  ids.json               local ID tracking + canonical-name index
  adj_chunk_event.json   chunk → [event_id]
  adj_event_frame.json   event → [fi_id]
  adj_frame_node.json    fi → [{node_id, fe_name, weight, is_core}]
  adj_mention_canon.json mention → canonical_id
  adj_causal.json        [{source, target, confidence}]
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
from scipy import sparse

from lightrag.utils import EmbeddingFunc

from .storage import make_kv, make_vdb
from .types import (
    ChunkSchema,
    EntityMentionSchema,
    CanonicalEntitySchema,
    EventSchema,
    FrameInstanceSchema,
    InfoNodeSchema,
    CausalEdgeSchema,
)

CORE_FE_WEIGHT = 1.0
NONCORE_FE_WEIGHT = 0.4
FE_FOCUS_BOOST = 1.5


class HypergraphStore:
    """All hypergraph nodes/edges using LightRAG storage backends."""

    def __init__(self, working_dir: str, embedding_func: EmbeddingFunc):
        hg_dir = os.path.join(working_dir, "hypergraph")
        os.makedirs(hg_dir, exist_ok=True)
        self._dir = hg_dir
        ef = embedding_func

        # ── KV stores (JsonKVStorage) ─────────────────────────────────────────
        self.chunks               = make_kv("chunks",             hg_dir)
        self.entity_mentions      = make_kv("entity_mentions",    hg_dir)
        self.canonical_entities   = make_kv("canonical_entities", hg_dir)
        self.events               = make_kv("events",             hg_dir)
        self.frame_instances      = make_kv("frame_instances",    hg_dir)
        self.info_nodes           = make_kv("info_nodes",         hg_dir)
        self.causal_edges         = make_kv("causal_edges",       hg_dir)

        # ── Vector stores (NanoVectorDBStorage) ──────────────────────────────
        self.vdb_chunks = make_vdb(
            "chunk_vec", hg_dir, ef, {"text", "source_doc"}
        )
        self.vdb_entity_mentions = make_vdb(
            "mention_vec", hg_dir, ef,
            {"name", "entity_type", "description"},
        )
        self.vdb_canonical_entities = make_vdb(
            "canonical_vec", hg_dir, ef,
            {"canonical_name", "entity_type"},
        )
        self.vdb_events = make_vdb(
            "event_vec", hg_dir, ef,
            {"trigger_lemma", "frame_name", "event_description"},
        )
        self.vdb_frame_instances = make_vdb(
            "fi_vec", hg_dir, ef,
            {"frame_name", "event_id", "lexical_unit"},
        )
        self.vdb_info_nodes = make_vdb(
            "info_vec", hg_dir, ef, {"value", "info_type"}
        )

        # ── Local ID sets (provide all_keys() semantics) ──────────────────────
        self._chunk_ids: set[str] = set()
        self._mention_ids: set[str] = set()
        self._canonical_ids: set[str] = set()
        self._event_ids: set[str] = set()
        self._fi_ids: set[str] = set()
        self._info_ids: set[str] = set()
        # canonical_name (lowercase) → canonical_id
        self._canonical_name_idx: dict[str, str] = {}

        # ── Adjacency lists ───────────────────────────────────────────────────
        self._adj_chunk_event: dict[str, list[str]] = {}
        self._adj_event_frame: dict[str, list[str]] = {}
        self._adj_frame_node: dict[str, list[dict]] = {}
        self._adj_mention_canon: dict[str, str] = {}
        self._adj_causal: list[dict] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        for store in (
            self.chunks, self.entity_mentions, self.canonical_entities,
            self.events, self.frame_instances, self.info_nodes, self.causal_edges,
        ):
            await store.initialize()
        for vdb in (
            self.vdb_chunks, self.vdb_entity_mentions, self.vdb_canonical_entities,
            self.vdb_events, self.vdb_frame_instances, self.vdb_info_nodes,
        ):
            await vdb.initialize()
        self._load_local()

    async def index_done_callback(self) -> None:
        """Flush KV stores to disk (LightRAG pattern)."""
        for store in (
            self.chunks, self.entity_mentions, self.canonical_entities,
            self.events, self.frame_instances, self.info_nodes, self.causal_edges,
        ):
            await store.index_done_callback()
        for vdb in (
            self.vdb_chunks, self.vdb_entity_mentions, self.vdb_canonical_entities,
            self.vdb_events, self.vdb_frame_instances, self.vdb_info_nodes,
        ):
            await vdb.index_done_callback()
        self._save_local()

    # ──────────────────────────────────────────────────────────────────────────
    # Local data: ID sets + adjacency
    # ──────────────────────────────────────────────────────────────────────────

    def _ids_path(self) -> str:
        return os.path.join(self._dir, "ids.json")

    def _adj_path(self, name: str) -> str:
        return os.path.join(self._dir, f"adj_{name}.json")

    def _load_local(self) -> None:
        ids_path = self._ids_path()
        if os.path.exists(ids_path):
            with open(ids_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self._chunk_ids     = set(d.get("chunk_ids", []))
            self._mention_ids   = set(d.get("mention_ids", []))
            self._canonical_ids = set(d.get("canonical_ids", []))
            self._event_ids     = set(d.get("event_ids", []))
            self._fi_ids        = set(d.get("fi_ids", []))
            self._info_ids      = set(d.get("info_ids", []))
            self._canonical_name_idx = d.get("canonical_name_idx", {})

        for attr, fname in (
            ("_adj_chunk_event",   "chunk_event"),
            ("_adj_event_frame",   "event_frame"),
            ("_adj_frame_node",    "frame_node"),
            ("_adj_mention_canon", "mention_canon"),
            ("_adj_causal",        "causal"),
        ):
            path = self._adj_path(fname)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    setattr(self, attr, json.load(f))

    def _save_local(self) -> None:
        ids_data = {
            "chunk_ids":     list(self._chunk_ids),
            "mention_ids":   list(self._mention_ids),
            "canonical_ids": list(self._canonical_ids),
            "event_ids":     list(self._event_ids),
            "fi_ids":        list(self._fi_ids),
            "info_ids":      list(self._info_ids),
            "canonical_name_idx": self._canonical_name_idx,
        }
        with open(self._ids_path(), "w", encoding="utf-8") as f:
            json.dump(ids_data, f, ensure_ascii=False, indent=2)

        for attr, fname in (
            ("_adj_chunk_event",   "chunk_event"),
            ("_adj_event_frame",   "event_frame"),
            ("_adj_frame_node",    "frame_node"),
            ("_adj_mention_canon", "mention_canon"),
            ("_adj_causal",        "causal"),
        ):
            with open(self._adj_path(fname), "w", encoding="utf-8") as f:
                json.dump(getattr(self, attr), f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    # Add nodes
    # ──────────────────────────────────────────────────────────────────────────

    async def add_chunk(self, chunk: ChunkSchema) -> None:
        await self.chunks.upsert({
            chunk.chunk_id: {
                "chunk_id": chunk.chunk_id, "text": chunk.text,
                "source_doc": chunk.source_doc, "chunk_index": chunk.chunk_index,
                "tokens": chunk.tokens,
            }
        })
        await self.vdb_chunks.upsert({
            chunk.chunk_id: {
                "content": chunk.text,
                "text": chunk.text,
                "source_doc": chunk.source_doc,
            }
        })
        self._chunk_ids.add(chunk.chunk_id)

    async def add_entity_mention(self, mention: EntityMentionSchema) -> None:
        await self.entity_mentions.upsert({
            mention.mention_id: {
                "mention_id": mention.mention_id,
                "chunk_id":   mention.chunk_id,
                "name":       mention.name,
                "entity_type": mention.entity_type,
                "description": mention.description,
                "aliases":     mention.aliases,
                "salience":    mention.salience,
                "canonical_id": mention.canonical_id,
            }
        })
        await self.vdb_entity_mentions.upsert({
            mention.mention_id: {
                "content": f"{mention.name} [SEP] {mention.description}",
                "name":        mention.name,
                "entity_type": mention.entity_type,
                "description": mention.description,
            }
        })
        self._mention_ids.add(mention.mention_id)

    async def add_canonical_entity(self, canonical: CanonicalEntitySchema) -> None:
        """Add or merge a canonical entity. Returns the effective canonical_id."""
        norm_name = canonical.canonical_name.lower().strip()

        # Merge into existing canonical if same name seen before
        existing_id = self._canonical_name_idx.get(norm_name)
        if existing_id:
            existing = await self.canonical_entities.get_by_id(existing_id)
            if existing:
                merged_descs = list(
                    dict.fromkeys(existing.get("descriptions", []) + canonical.descriptions)
                )
                merged_mentions = list(
                    dict.fromkeys(existing.get("mention_ids", []) + canonical.mention_ids)
                )
                await self.canonical_entities.upsert({
                    existing_id: {
                        "canonical_id":   existing_id,
                        "canonical_name": existing.get("canonical_name", canonical.canonical_name),
                        "entity_type":    existing.get("entity_type", canonical.entity_type),
                        "descriptions":   merged_descs,
                        "mention_ids":    merged_mentions,
                    }
                })
                # Point new mention IDs to existing canonical
                for mid in canonical.mention_ids:
                    self._adj_mention_canon[mid] = existing_id
                    m_data = await self.entity_mentions.get_by_id(mid)
                    if m_data:
                        m_data["canonical_id"] = existing_id
                        await self.entity_mentions.upsert({mid: m_data})
                return

        # New canonical
        await self.canonical_entities.upsert({
            canonical.canonical_id: {
                "canonical_id":   canonical.canonical_id,
                "canonical_name": canonical.canonical_name,
                "entity_type":    canonical.entity_type,
                "descriptions":   canonical.descriptions,
                "mention_ids":    canonical.mention_ids,
            }
        })
        combined_desc = " ".join(canonical.descriptions[:3])
        await self.vdb_canonical_entities.upsert({
            canonical.canonical_id: {
                "content":        f"{canonical.canonical_name} [SEP] {combined_desc}",
                "canonical_name": canonical.canonical_name,
                "entity_type":    canonical.entity_type,
            }
        })
        self._canonical_ids.add(canonical.canonical_id)
        self._canonical_name_idx[norm_name] = canonical.canonical_id

        # Update mention → canonical mapping
        for mid in canonical.mention_ids:
            self._adj_mention_canon[mid] = canonical.canonical_id
            m_data = await self.entity_mentions.get_by_id(mid)
            if m_data:
                m_data["canonical_id"] = canonical.canonical_id
                await self.entity_mentions.upsert({mid: m_data})

    async def update_canonical_descriptions(
        self, canonical_id: str, descriptions: list[str]
    ) -> None:
        """Replace descriptions for a canonical entity (after LLM merge)."""
        data = await self.canonical_entities.get_by_id(canonical_id)
        if not data:
            return
        data["descriptions"] = descriptions
        await self.canonical_entities.upsert({canonical_id: data})
        combined_desc = " ".join(descriptions[:3])
        await self.vdb_canonical_entities.upsert({
            canonical_id: {
                "content":        f"{data['canonical_name']} [SEP] {combined_desc}",
                "canonical_name": data["canonical_name"],
                "entity_type":    data["entity_type"],
            }
        })

    async def add_event(self, event: EventSchema) -> None:
        await self.events.upsert({
            event.event_id: {
                "event_id":                event.event_id,
                "chunk_id":                event.chunk_id,
                "trigger":                 event.trigger,
                "trigger_lemma":           event.trigger_lemma,
                "trigger_pos":             event.trigger_pos,
                "event_span":              event.event_span,
                "event_description":       event.event_description,
                "frame_name":              event.frame_name,
                "participant_mention_ids": event.participant_mention_ids,
                "frame_instance_ids":      event.frame_instance_ids,
                "canonical_event_id":      event.canonical_event_id,
            }
        })
        content = (
            f"{event.trigger_lemma} [{event.frame_name}]: {event.event_description}"
        )
        await self.vdb_events.upsert({
            event.event_id: {
                "content":           content,
                "trigger_lemma":     event.trigger_lemma,
                "frame_name":        event.frame_name,
                "event_description": event.event_description,
            }
        })
        self._event_ids.add(event.event_id)

        if event.chunk_id not in self._adj_chunk_event:
            self._adj_chunk_event[event.chunk_id] = []
        if event.event_id not in self._adj_chunk_event[event.chunk_id]:
            self._adj_chunk_event[event.chunk_id].append(event.event_id)

    async def add_frame_instance(
        self,
        fi: FrameInstanceSchema,
        frame_core_fe_names: set[str],
    ) -> None:
        await self.frame_instances.upsert({
            fi.fi_id: {
                "fi_id":      fi.fi_id,
                "event_id":   fi.event_id,
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
        })
        parts = [f"Frame: {fi.frame_name}"]
        for a in fi.core_assignments + fi.noncore_assignments:
            if a.filler_id and a.filler_text:
                parts.append(f"{a.fe_name}: {a.filler_text}")
        content = " | ".join(parts)
        await self.vdb_frame_instances.upsert({
            fi.fi_id: {
                "content":      content,
                "frame_name":   fi.frame_name,
                "event_id":     fi.event_id,
                "lexical_unit": fi.lexical_unit,
            }
        })
        self._fi_ids.add(fi.fi_id)

        if fi.event_id not in self._adj_event_frame:
            self._adj_event_frame[fi.event_id] = []
        if fi.fi_id not in self._adj_event_frame[fi.event_id]:
            self._adj_event_frame[fi.event_id].append(fi.fi_id)

        if fi.fi_id not in self._adj_frame_node:
            self._adj_frame_node[fi.fi_id] = []
        for a in fi.core_assignments + fi.noncore_assignments:
            if a.filler_id is None:
                continue
            is_core = a.fe_name in frame_core_fe_names
            weight = CORE_FE_WEIGHT if is_core else NONCORE_FE_WEIGHT
            self._adj_frame_node[fi.fi_id].append({
                "node_id": a.filler_id,
                "fe_name": a.fe_name,
                "weight":  weight,
                "is_core": is_core,
            })

    async def add_info_node(self, info: InfoNodeSchema) -> None:
        await self.info_nodes.upsert({
            info.info_id: {
                "info_id":   info.info_id,
                "value":     info.value,
                "info_type": info.info_type,
            }
        })
        await self.vdb_info_nodes.upsert({
            info.info_id: {
                "content":   info.value,
                "value":     info.value,
                "info_type": info.info_type,
            }
        })
        self._info_ids.add(info.info_id)

    async def add_causal_edge(self, edge: CausalEdgeSchema) -> None:
        await self.causal_edges.upsert({
            edge.edge_id: {
                "edge_id":         edge.edge_id,
                "source_event_id": edge.source_event_id,
                "target_event_id": edge.target_event_id,
                "relation_type":   edge.relation_type,
                "confidence":      edge.confidence,
                "evidence_span":   edge.evidence_span,
            }
        })
        self._adj_causal.append({
            "source":     edge.source_event_id,
            "target":     edge.target_event_id,
            "confidence": edge.confidence,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # Sparse matrix construction (called at retrieval time)
    # ──────────────────────────────────────────────────────────────────────────

    async def build_matrices(
        self, fe_focus: Optional[list[str]] = None
    ) -> dict:
        """Build normalized sparse incidence matrices for hypergraph diffusion.

        Returns dict with keys:
          H_ce  : [n_chunks × n_events]
          H_ef  : [n_events × n_frames]
          H_fe  : [n_frames × n_nodes] (FE-weighted)
          A_cau : [n_events × n_events] (causal)
          plus ID lists and index dicts.
        """
        chunk_ids  = list(self._chunk_ids)
        event_ids  = list(self._event_ids)
        fi_ids     = list(self._fi_ids)
        mention_ids = list(self._mention_ids)
        info_ids   = list(self._info_ids)
        all_node_ids = mention_ids + info_ids

        if not event_ids or not fi_ids:
            return {}

        chunk_idx = {c: i for i, c in enumerate(chunk_ids)}
        event_idx = {e: i for i, e in enumerate(event_ids)}
        fi_idx    = {f: i for i, f in enumerate(fi_ids)}
        node_idx  = {n: i for i, n in enumerate(all_node_ids)}

        n_c = len(chunk_ids)
        n_e = len(event_ids)
        n_f = len(fi_ids)
        n_n = len(all_node_ids)

        # H_ce
        rows, cols = [], []
        for cid, eids in self._adj_chunk_event.items():
            if cid not in chunk_idx:
                continue
            for eid in eids:
                if eid in event_idx:
                    rows.append(chunk_idx[cid])
                    cols.append(event_idx[eid])
        H_ce = (
            sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_c, n_e))
            if rows else sparse.csr_matrix((n_c, n_e))
        )

        # H_ef
        rows, cols = [], []
        for eid, fids in self._adj_event_frame.items():
            if eid not in event_idx:
                continue
            for fid in fids:
                if fid in fi_idx:
                    rows.append(event_idx[eid])
                    cols.append(fi_idx[fid])
        H_ef = (
            sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_e, n_f))
            if rows else sparse.csr_matrix((n_e, n_f))
        )

        # H_fe (FE-weighted)
        rows, cols, data = [], [], []
        for fid, adj_list in self._adj_frame_node.items():
            if fid not in fi_idx:
                continue
            for entry in adj_list:
                nid = entry["node_id"]
                if nid not in node_idx:
                    continue
                w = entry["weight"]
                if fe_focus and entry["fe_name"] in fe_focus:
                    w *= FE_FOCUS_BOOST
                rows.append(fi_idx[fid])
                cols.append(node_idx[nid])
                data.append(w)
        H_fe = (
            sparse.csr_matrix((data, (rows, cols)), shape=(n_f, n_n))
            if data else sparse.csr_matrix((n_f, n_n))
        )

        # A_cau
        rows, cols, data = [], [], []
        for entry in self._adj_causal:
            s, t = entry["source"], entry["target"]
            if s in event_idx and t in event_idx:
                rows.append(event_idx[s])
                cols.append(event_idx[t])
                data.append(entry["confidence"])
        A_cau = (
            sparse.csr_matrix((data, (rows, cols)), shape=(n_e, n_e))
            if data else sparse.csr_matrix((n_e, n_e))
        )

        return {
            "H_ce": H_ce, "H_ef": H_ef, "H_fe": H_fe, "A_cau": A_cau,
            "chunk_ids": chunk_ids, "event_ids": event_ids,
            "fi_ids": fi_ids,       "node_ids": all_node_ids,
            "chunk_idx": chunk_idx, "event_idx": event_idx,
            "fi_idx": fi_idx,       "node_idx": node_idx,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Vector search helpers (pass pre-computed embedding as query_embedding)
    # ──────────────────────────────────────────────────────────────────────────

    async def search_canonical(self, q_vec: np.ndarray, top_k: int) -> list[dict]:
        return await self.vdb_canonical_entities.query(
            "", top_k=top_k, query_embedding=q_vec
        )

    async def search_frame_instances(self, q_vec: np.ndarray, top_k: int) -> list[dict]:
        return await self.vdb_frame_instances.query(
            "", top_k=top_k, query_embedding=q_vec
        )

    async def search_chunks(self, q_vec: np.ndarray, top_k: int) -> list[dict]:
        return await self.vdb_chunks.query(
            "", top_k=top_k, query_embedding=q_vec
        )

    async def search_info_nodes(self, q_vec: np.ndarray, top_k: int) -> list[dict]:
        return await self.vdb_info_nodes.query(
            "", top_k=top_k, query_embedding=q_vec
        )

    async def search_frame_instances_by_vec(
        self, vec: np.ndarray, top_k: int
    ) -> list[dict]:
        return await self.vdb_frame_instances.query(
            "", top_k=top_k, query_embedding=vec
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_canonical_id_by_name(self, name: str) -> Optional[str]:
        return self._canonical_name_idx.get(name.lower().strip())

    async def get_all_mention_ids_for_canonical(self, canonical_id: str) -> list[str]:
        data = await self.canonical_entities.get_by_id(canonical_id)
        return data.get("mention_ids", []) if data else []

    async def get_canonical_id(self, mention_id: str) -> Optional[str]:
        return self._adj_mention_canon.get(mention_id)

    async def get_events_in_chunk(self, chunk_id: str) -> list[dict]:
        results = []
        for eid in self._adj_chunk_event.get(chunk_id, []):
            ev = await self.events.get_by_id(eid)
            if ev:
                results.append(ev)
        return results

    async def get_frame_instances_for_event(self, event_id: str) -> list[dict]:
        results = []
        for fid in self._adj_event_frame.get(event_id, []):
            fi = await self.frame_instances.get_by_id(fid)
            if fi:
                results.append(fi)
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Deletion: cascade-remove all data for a set of chunk_ids
    # ──────────────────────────────────────────────────────────────────────────

    async def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Remove chunks and all downstream data (events, frames, mentions)."""
        if not chunk_ids:
            return
        chunk_set = set(chunk_ids)

        # Collect IDs to delete
        event_ids: set[str] = set()
        for cid in chunk_set:
            for eid in self._adj_chunk_event.get(cid, []):
                event_ids.add(eid)

        fi_ids: set[str] = set()
        for eid in event_ids:
            for fid in self._adj_event_frame.get(eid, []):
                fi_ids.add(fid)

        # Collect mention_ids from chunk data
        mention_ids: set[str] = set()
        info_ids: set[str] = set()
        for cid in chunk_set:
            cdata = await self.chunks.get_by_id(cid)
            if cdata:
                for mid in cdata.get("mention_ids", []):
                    mention_ids.add(mid)

        # Collect info_ids from frame instances
        for fid in fi_ids:
            fi_data = await self.frame_instances.get_by_id(fid)
            if fi_data:
                for slot in ("core_assignments", "noncore_assignments"):
                    for assign in fi_data.get(slot, []):
                        if assign.get("filler_type") == "VALUE":
                            iid = assign.get("filler_id")
                            if iid:
                                info_ids.add(iid)

        # Delete from KV stores
        if chunk_ids:
            await self.chunks.delete(list(chunk_set))
            await self.vdb_chunks.delete(list(chunk_set))
        if event_ids:
            await self.events.delete(list(event_ids))
            await self.vdb_events.delete(list(event_ids))
        if fi_ids:
            await self.frame_instances.delete(list(fi_ids))
            await self.vdb_frame_instances.delete(list(fi_ids))
        if mention_ids:
            await self.entity_mentions.delete(list(mention_ids))
            await self.vdb_entity_mentions.delete(list(mention_ids))
        if info_ids:
            await self.info_nodes.delete(list(info_ids))
            await self.vdb_info_nodes.delete(list(info_ids))

        # Prune causal edges involving deleted events
        self._adj_causal = [
            e for e in self._adj_causal
            if e.get("source") not in event_ids and e.get("target") not in event_ids
        ]

        # Prune adjacency lists
        for cid in chunk_set:
            self._adj_chunk_event.pop(cid, None)
            self._chunk_ids.discard(cid)
        for eid in event_ids:
            self._adj_event_frame.pop(eid, None)
            self._event_ids.discard(eid)
        for fid in fi_ids:
            self._adj_frame_node.pop(fid, None)
            self._fi_ids.discard(fid)
        for mid in mention_ids:
            self._adj_mention_canon.pop(mid, None)
            self._mention_ids.discard(mid)
        for iid in info_ids:
            self._info_ids.discard(iid)
