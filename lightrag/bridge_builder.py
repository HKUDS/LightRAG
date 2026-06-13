"""
bridge_builder.py — Cross-event hyperedge and bridge edge builder.

After all chunks have been indexed (llm_frames mode), this module builds
three tiers of cross-event connections to break data isolation:

  Tier 1 — Similarity resolution (embedding-based, no LLM):
    • Entities cosine_sim >= ENTITY_MERGE_THRESHOLD  → merged as one node
    • Entities ENTITY_SIM_THRESHOLD <= sim < MERGE   → "entity_similarity" edge
    • Frames  FRAME_SIM_THRESHOLD  <= sim < 1.0      → "frame_similarity" edge

  Tier 2 — Structural bridges (zero LLM cost):
    Binary edge  (1-1):  2 events share the same frame/entity   → binary edge
    Hub hyperedge (1-N): >=HUB_MIN_EVENTS share same frame/entity → HUB node + star
    N-M hyperedge:       >=NM_MIN_EVENTS share >=NM_MIN_SHARED  → CLUSTER node + star

  Tier 3 — Causal / temporal edges (LLM):
    For high-weight candidate pairs from Tier 2, LLM classifies:
    precedes | causes | enables | contradicts
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from lightrag.utils import logger

# ── Thresholds ────────────────────────────────────────────────────────────────
ENTITY_MERGE_THRESHOLD: float = 0.80  # cosine >= this → merge into one node
ENTITY_SIM_THRESHOLD:   float = 0.50  # cosine in [this, MERGE) → sim edge
FRAME_SIM_THRESHOLD:    float = 0.50  # frames: sim-only edges (no merge)

HUB_MIN_EVENTS: int = 3   # events sharing same frame/entity → hub hyperedge
NM_MIN_EVENTS:  int = 3   # events sharing multiple items   → N-M hyperedge
NM_MIN_SHARED:  int = 2   # minimum combined shared frame+entity count for N-M

MAX_CAUSAL_PAIRS:          int   = 50
CAUSAL_CANDIDATE_MIN_WEIGHT: float = 0.30  # min edge weight to be causal candidate

# Node-type prefix for bridge nodes
HUB_PREFIX     = "HUB:"
CLUSTER_PREFIX = "CLUSTER:"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe  = np.where(norms == 0, 1e-9, norms)
    normed = embeddings / safe
    return (normed @ normed.T).astype(np.float32)


def _union_find_canonical(names: list[str], sim_matrix: np.ndarray,
                           threshold: float, freq: dict[str, int]) -> dict[str, str]:
    """Union-Find: merge pairs >= threshold; higher-freq name wins."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        # Two-pass path compression: first find root, then compress path
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def union(a: str, b: str):
        ca, cb = find(a), find(b)
        if ca == cb:
            return
        if freq.get(ca, 0) >= freq.get(cb, 0):
            parent[cb] = ca
        else:
            parent[ca] = cb

    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            if float(sim_matrix[i, j]) >= threshold:
                union(names[i], names[j])

    return {name: find(name) for name in names}


def _apply_merge_nodes(nodes: dict, merge_map: dict[str, str]) -> dict:
    merged: dict[str, list] = defaultdict(list)
    for name, entries in nodes.items():
        canon = merge_map.get(name, name)
        for e in entries:
            merged[canon].append({**e, "entity_name": canon})
    return dict(merged)


def _apply_merge_edges(edges: dict, merge_map: dict[str, str]) -> dict:
    merged: dict[tuple, list] = defaultdict(list)
    for (src, tgt), edge_list in edges.items():
        cs = merge_map.get(src, src)
        ct = merge_map.get(tgt, tgt)
        if cs == ct:
            continue
        key = (cs, ct) if cs <= ct else (ct, cs)
        for e in edge_list:
            # Preserve original directed src/tgt from the record, remapped to canonical
            orig_src = merge_map.get(e.get("src_id", src), e.get("src_id", src))
            orig_tgt = merge_map.get(e.get("tgt_id", tgt), e.get("tgt_id", tgt))
            merged[key].append({**e, "src_id": orig_src, "tgt_id": orig_tgt})
    return dict(merged)


def _apply_merge_metadata(metadata_list: list[dict],
                           merge_map: dict[str, str]) -> list[dict]:
    updated = []
    for m in metadata_list:
        canon_entities = list({merge_map.get(e, e) for e in m.get("entity_names", [])})
        updated.append({**m, "entity_names": canon_entities})
    return updated


# ══════════════════════════════════════════════════════════════════════════════
# Tier 1 — Similarity resolution
# ══════════════════════════════════════════════════════════════════════════════

async def _resolve_similarity(
    nodes: dict,
    edges: dict,
    metadata_list: list[dict],
    embed_func,
    entity_merge_threshold: float,
    entity_sim_threshold: float,
    frame_sim_threshold: float,
) -> tuple[dict, dict, list[dict], dict[str, str], dict]:
    """
    Compute embedding-based similarity for entity and frame nodes.

    Returns:
        merged_nodes, merged_edges, updated_metadata,
        entity_merge_map,
        sim_edges
    """
    timestamp = int(time.time())
    entity_merge_map: dict[str, str] = {}
    sim_edges: dict[tuple, list] = defaultdict(list)

    entity_names = [
        n for n, entries in nodes.items()
        if entries and entries[0].get("entity_type") not in ("event", "frame")
    ]
    frame_names = [
        n for n, entries in nodes.items()
        if entries and entries[0].get("entity_type") == "frame"
    ]

    # ── Entity similarity ────────────────────────────────────────────────────
    if len(entity_names) >= 2:
        try:
            emb = np.array(await embed_func(entity_names))
            sim = _cosine_sim_matrix(emb)
            freq = {n: len(nodes[n]) for n in entity_names}
            entity_merge_map = _union_find_canonical(
                entity_names, sim, entity_merge_threshold, freq
            )
            merged_count = sum(1 for k, v in entity_merge_map.items() if k != v)
            if merged_count:
                logger.info("[bridge] Entity merge: %d → canonical names", merged_count)

            for i, a in enumerate(entity_names):
                for j, b in enumerate(entity_names):
                    if j <= i:
                        continue
                    s = float(sim[i, j])
                    if entity_sim_threshold <= s < entity_merge_threshold:
                        ca = entity_merge_map.get(a, a)
                        cb = entity_merge_map.get(b, b)
                        if ca == cb:
                            continue
                        key = (ca, cb) if ca < cb else (cb, ca)
                        sim_edges[key].append({
                            "src_id": key[0], "tgt_id": key[1],
                            "weight": s,
                            "keywords": "entity_similarity",
                            "description": (
                                f'"{ca}" and "{cb}" are semantically similar '
                                f"(cosine={s:.2f})."
                            ),
                            "source_id": "bridge_builder",
                            "file_path": "bridge",
                            "timestamp": timestamp,
                        })
        except Exception as exc:
            logger.warning("[bridge] Entity embedding failed: %s — skipped", exc)

    # ── Frame similarity (sim edges only, no merge) ──────────────────────────
    if len(frame_names) >= 2:
        try:
            frame_texts = [
                f"{n}: {(nodes[n][0].get('description', '') if nodes[n] else '')[:120]}"
                for n in frame_names
            ]
            emb_f = np.array(await embed_func(frame_texts))
            sim_f = _cosine_sim_matrix(emb_f)
            for i, a in enumerate(frame_names):
                for j, b in enumerate(frame_names):
                    if j <= i:
                        continue
                    s = float(sim_f[i, j])
                    if frame_sim_threshold <= s < 1.0:
                        key = (a, b) if a < b else (b, a)
                        sim_edges[key].append({
                            "src_id": key[0], "tgt_id": key[1],
                            "weight": s,
                            "keywords": "frame_similarity",
                            "description": (
                                f'Frames "{a}" and "{b}" are semantically similar '
                                f"(cosine={s:.2f})."
                            ),
                            "source_id": "bridge_builder",
                            "file_path": "bridge",
                            "timestamp": timestamp,
                        })
        except Exception as exc:
            logger.warning("[bridge] Frame embedding failed: %s — skipped", exc)

    merged_nodes = _apply_merge_nodes(nodes, entity_merge_map) if entity_merge_map else nodes
    merged_edges = _apply_merge_edges(edges, entity_merge_map) if entity_merge_map else edges
    updated_meta = _apply_merge_metadata(metadata_list, entity_merge_map) if entity_merge_map else metadata_list

    return merged_nodes, merged_edges, updated_meta, entity_merge_map, dict(sim_edges)


# ══════════════════════════════════════════════════════════════════════════════
# Tier 2 — Structural bridges
# ══════════════════════════════════════════════════════════════════════════════

def _build_co_frame_edges(
    metadata_list: list[dict],
    hub_min: int = HUB_MIN_EVENTS,
) -> tuple[dict, dict]:
    """
    Co-frame connections between events.

    - 2 events share a frame  → binary edge (1-1)
    - >=hub_min events share  → HUB node + star edges (1-N hyperedge)

    Returns: (binary_edges, new_hub_nodes_and_edges)
              where new_hub_nodes_and_edges = (hub_nodes, hub_edges)
    """
    timestamp = int(time.time())
    frame_to_events: dict[str, list[str]] = defaultdict(list)
    event_frames: dict[str, set[str]] = {}

    for m in metadata_list:
        eid = m["event_id"]
        fnames = set(m.get("frame_names", []))
        event_frames[eid] = fnames
        for fn in fnames:
            frame_to_events[fn].append(eid)

    binary_edges: dict[tuple, list] = defaultdict(list)
    hub_nodes: dict[str, list] = defaultdict(list)
    hub_edges: dict[tuple, list] = defaultdict(list)

    # Collect all frames shared by each pair (avoid one record per frame per pair)
    pair_shared_frames: dict[tuple, set[str]] = defaultdict(set)

    for frame_name, event_ids in frame_to_events.items():
        unique_events = sorted(set(event_ids))
        n = len(unique_events)
        if n < 2:
            continue

        if n < hub_min:
            for a, b in combinations(unique_events, 2):
                key = (a, b) if a < b else (b, a)
                pair_shared_frames[key].add(frame_name)
        else:
            # 1-N hyperedge: create HUB node
            hub_id = f"{HUB_PREFIX}frame:{frame_name}"
            hub_nodes[hub_id].append({
                "entity_name": hub_id,
                "entity_type": "hub",
                "description": (
                    f"Hub hyperedge: {n} events all evoke frame '{frame_name}'."
                ),
                "source_id": "bridge_builder",
                "file_path": "bridge",
                "timestamp": timestamp,
            })
            for eid in unique_events:
                key = (eid, hub_id) if eid < hub_id else (hub_id, eid)
                hub_edges[key].append({
                    "src_id": eid, "tgt_id": hub_id,
                    "weight": 1.0,
                    "keywords": "hub_member",
                    "description": f"Event {eid} is a member of hub {hub_id}.",
                    "source_id": "bridge_builder",
                    "file_path": "bridge",
                    "timestamp": timestamp,
                })

    # Emit one binary edge record per pair with the full set of shared frames
    for key, shared in pair_shared_frames.items():
        a, b = key
        fa, fb = event_frames.get(a, set()), event_frames.get(b, set())
        w = len(shared) / len(fa | fb) if (fa | fb) else 0.0
        binary_edges[key].append({
            "src_id": key[0], "tgt_id": key[1],
            "weight": w,
            "keywords": "co_frame",
            "description": f"Events share frame(s): {', '.join(sorted(shared))}.",
            "source_id": "bridge_builder",
            "file_path": "bridge",
            "timestamp": timestamp,
        })

    logger.info(
        "[bridge] Co-frame — binary_edges=%d  hub_nodes=%d",
        len(binary_edges), len(hub_nodes),
    )
    return dict(binary_edges), (dict(hub_nodes), dict(hub_edges))


def _build_co_entity_edges(
    metadata_list: list[dict],
    hub_min: int = HUB_MIN_EVENTS,
) -> tuple[dict, dict]:
    """
    Co-entity connections between events.

    Same 1-1 / hub logic as co-frame.
    Returns: (binary_edges, (hub_nodes, hub_edges))
    """
    timestamp = int(time.time())
    entity_to_events: dict[str, list[str]] = defaultdict(list)
    event_entities: dict[str, set[str]] = {}

    for m in metadata_list:
        eid = m["event_id"]
        enames = set(m.get("entity_names", []))
        event_entities[eid] = enames
        for en in enames:
            entity_to_events[en].append(eid)

    binary_edges: dict[tuple, list] = defaultdict(list)
    hub_nodes: dict[str, list] = defaultdict(list)
    hub_edges: dict[tuple, list] = defaultdict(list)

    # Collect all entities shared by each pair (avoid one record per entity per pair)
    pair_shared_entities: dict[tuple, set[str]] = defaultdict(set)

    for entity_name, event_ids in entity_to_events.items():
        unique_events = sorted(set(event_ids))
        n = len(unique_events)
        if n < 2:
            continue

        if n < hub_min:
            for a, b in combinations(unique_events, 2):
                key = (a, b) if a < b else (b, a)
                pair_shared_entities[key].add(entity_name)
        else:
            # Add hash suffix to prevent collision for long/similar entity names
            entity_hash = hashlib.md5(entity_name.encode()).hexdigest()[:6]
            hub_id = f"{HUB_PREFIX}entity:{entity_name[:34]}_{entity_hash}"
            hub_nodes[hub_id].append({
                "entity_name": hub_id,
                "entity_type": "hub",
                "description": (
                    f"Hub hyperedge: {n} events all involve entity '{entity_name}'."
                ),
                "source_id": "bridge_builder",
                "file_path": "bridge",
                "timestamp": timestamp,
            })
            for eid in unique_events:
                key = (eid, hub_id) if eid < hub_id else (hub_id, eid)
                hub_edges[key].append({
                    "src_id": eid, "tgt_id": hub_id,
                    "weight": 1.0,
                    "keywords": "hub_member",
                    "description": f"Event {eid} is a member of hub {hub_id}.",
                    "source_id": "bridge_builder",
                    "file_path": "bridge",
                    "timestamp": timestamp,
                })

    # Emit one binary edge record per pair with the full set of shared entities
    for key, shared in pair_shared_entities.items():
        a, b = key
        ea, eb = event_entities.get(a, set()), event_entities.get(b, set())
        w = len(shared) / len(ea | eb) if (ea | eb) else 0.0
        binary_edges[key].append({
            "src_id": key[0], "tgt_id": key[1],
            "weight": w,
            "keywords": "co_entity",
            "description": (
                f"Events share entities: "
                f"{', '.join(sorted(shared)[:5])}."
            ),
            "source_id": "bridge_builder",
            "file_path": "bridge",
            "timestamp": timestamp,
        })

    logger.info(
        "[bridge] Co-entity — binary_edges=%d  hub_nodes=%d",
        len(binary_edges), len(hub_nodes),
    )
    return dict(binary_edges), (dict(hub_nodes), dict(hub_edges))


def _build_nm_hyperedges(
    metadata_list: list[dict],
    nm_min_events: int = NM_MIN_EVENTS,
    nm_min_shared: int = NM_MIN_SHARED,
) -> tuple[dict, dict]:
    """
    N-M hyperedges: groups of >=nm_min_events events that share
    >=nm_min_shared (frames + entities) in common.

    Creates a CLUSTER node that all member events connect to.

    Returns: (cluster_nodes, cluster_edges)
    """
    timestamp = int(time.time())

    # Build combined shared-item sets per event
    event_items: dict[str, set[str]] = {}
    for m in metadata_list:
        eid = m["event_id"]
        items: set[str] = set()
        items.update(f"F:{fn}" for fn in m.get("frame_names", []))
        items.update(f"E:{en}" for en in m.get("entity_names", []))
        event_items[eid] = items

    event_ids = sorted(event_items.keys())
    cluster_nodes: dict[str, list] = defaultdict(list)
    cluster_edges: dict[tuple, list] = defaultdict(list)
    seen_clusters: set[frozenset] = set()

    # Find groups sharing >= nm_min_shared items
    # To avoid O(2^N), only consider groups formed by a common shared item
    item_to_events: dict[str, list[str]] = defaultdict(list)
    for eid, items in event_items.items():
        for item in items:
            item_to_events[item].append(eid)

    # Candidate groups: events sharing >=nm_min_shared items
    # Build via pair intersection
    event_shared: dict[tuple[str, str], set[str]] = {}
    for item, eids in item_to_events.items():
        for a, b in combinations(sorted(set(eids)), 2):
            key = (a, b) if a < b else (b, a)
            event_shared.setdefault(key, set()).add(item)

    # Find cliques of >=nm_min_events all sharing >=nm_min_shared items
    # Simple approach: group events that all mutually share >=nm_min_shared items
    # (approximate via seed expansion)
    processed: set[str] = set()
    for eid in event_ids:
        if eid in processed:
            continue
        # Find events that share >= nm_min_shared items with eid
        candidates = [eid]
        for other in event_ids:
            if other == eid:
                continue
            pair = (eid, other) if eid < other else (other, eid)
            shared = event_shared.get(pair, set())
            if len(shared) >= nm_min_shared:
                candidates.append(other)

        if len(candidates) < nm_min_events:
            continue

        # Find common shared items across ALL candidates
        common_items = event_items[candidates[0]].copy()
        for c in candidates[1:]:
            common_items &= event_items[c]

        if len(common_items) < nm_min_shared:
            continue

        group_key = frozenset(candidates)
        if group_key in seen_clusters:
            continue
        seen_clusters.add(group_key)
        processed.update(candidates)

        # Create cluster node
        items_label = "+".join(sorted(common_items)[:3])
        cluster_hash = hashlib.md5(
            "|".join(sorted(candidates)).encode()
        ).hexdigest()[:8]
        cluster_id = f"{CLUSTER_PREFIX}{cluster_hash}"

        cluster_nodes[cluster_id].append({
            "entity_name": cluster_id,
            "entity_type": "cluster",
            "description": (
                f"N-M hyperedge: {len(candidates)} events share "
                f"{len(common_items)} items ({items_label}...)."
            ),
            "source_id": "bridge_builder",
            "file_path": "bridge",
            "timestamp": timestamp,
        })

        for c in candidates:
            key = (c, cluster_id) if c < cluster_id else (cluster_id, c)
            cluster_edges[key].append({
                "src_id": c, "tgt_id": cluster_id,
                "weight": len(common_items) / max(len(event_items[c]), 1),
                "keywords": "cluster_member",
                "description": (
                    f"Event {c} is a member of cluster {cluster_id}."
                ),
                "source_id": "bridge_builder",
                "file_path": "bridge",
                "timestamp": timestamp,
            })

    logger.info(
        "[bridge] N-M hyperedges — clusters=%d", len(cluster_nodes)
    )
    return dict(cluster_nodes), dict(cluster_edges)


# ══════════════════════════════════════════════════════════════════════════════
# Tier 3 — Causal / temporal edges (LLM)
# ══════════════════════════════════════════════════════════════════════════════

async def _classify_pair(meta_a: dict, meta_b: dict, llm_func) -> dict | None:
    """LLM-classify the directed relation from event_a → event_b."""
    from lightrag.prompt import PROMPTS
    from lightrag.llm_frame_extractor import _parse_json_response

    prompt = PROMPTS["llm_event_relation"].format(
        event_a_id=meta_a["event_id"],
        event_a_desc=meta_a.get("chunk_content", "")[:600],
        frames_a=", ".join(meta_a.get("frame_names", [])),
        entities_a=", ".join(meta_a.get("entity_names", [])[:10]),
        event_b_id=meta_b["event_id"],
        event_b_desc=meta_b.get("chunk_content", "")[:600],
        frames_b=", ".join(meta_b.get("frame_names", [])),
        entities_b=", ".join(meta_b.get("entity_names", [])[:10]),
    )
    try:
        raw  = await llm_func(prompt)
        data = _parse_json_response(raw)
        relation   = data.get("relation", "none")
        confidence = float(data.get("confidence", 0.0))
        if relation == "none" or confidence < 0.5:
            return None
        ts = int(time.time())
        return {
            "src_id": meta_a["event_id"],
            "tgt_id": meta_b["event_id"],
            "weight": confidence,
            "keywords": relation,
            "description": (
                f'[EVENT relation] {meta_a["event_id"]} {relation} '
                f'{meta_b["event_id"]}. {data.get("reasoning", "")}'
            ),
            "source_id": "bridge_builder",
            "file_path": "bridge",
            "timestamp": ts,
        }
    except Exception as exc:
        logger.warning(
            "[bridge] causal classify (%s, %s): %s",
            meta_a["event_id"], meta_b["event_id"], exc,
        )
        return None


async def _build_causal_edges(
    metadata_list: list[dict],
    candidate_edges: dict[tuple, list],
    llm_func,
    max_pairs: int,
    weight_threshold: float,
) -> dict[tuple, list]:
    weighted: list[tuple[float, tuple]] = []
    for (a, b), edge_list in candidate_edges.items():
        w = sum(e.get("weight", 0.0) for e in edge_list)
        if w >= weight_threshold:
            weighted.append((w, (a, b)))
    weighted.sort(reverse=True)
    top_pairs = [p for _, p in weighted[:max_pairs]]
    if not top_pairs:
        return {}

    meta_by_id = {m["event_id"]: m for m in metadata_list}
    tasks, valid = [], []
    for a, b in top_pairs:
        ma, mb = meta_by_id.get(a), meta_by_id.get(b)
        if ma and mb:
            tasks.append(_classify_pair(ma, mb, llm_func))
            valid.append((a, b))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    edges: dict[tuple, list] = defaultdict(list)
    classified = 0
    for (a, b), res in zip(valid, results):
        if isinstance(res, dict) and res:
            edges[(a, b)].append(res)
            classified += 1

    logger.info(
        "[bridge] Causal: %d/%d pairs classified", classified, len(valid)
    )
    return dict(edges)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

async def build_all_bridges(
    metadata_list: list[dict],
    all_nodes: dict,
    all_edges: dict,
    embed_func,
    llm_func,
    working_dir: str = ".",
    entity_merge_threshold: float = ENTITY_MERGE_THRESHOLD,
    entity_sim_threshold:   float = ENTITY_SIM_THRESHOLD,
    frame_sim_threshold:    float = FRAME_SIM_THRESHOLD,
    hub_min_events:         int   = HUB_MIN_EVENTS,
    nm_min_events:          int   = NM_MIN_EVENTS,
    nm_min_shared:          int   = NM_MIN_SHARED,
    max_causal_pairs:       int   = MAX_CAUSAL_PAIRS,
    causal_weight_threshold: float = CAUSAL_CANDIDATE_MIN_WEIGHT,
) -> tuple[dict, dict]:
    """
    Build all bridge edges and apply entity merging.

    Args:
        metadata_list : [{event_id, frame_names, entity_names, chunk_content}]
        all_nodes     : accumulated maybe_nodes from all chunks
        all_edges     : accumulated maybe_edges from all chunks
        embed_func    : async embedding function (list[str] → np.ndarray)
        llm_func      : async LLM call function

    Returns:
        (final_nodes, final_edges) — ready for storage upsert
    """
    if not metadata_list:
        logger.debug("[bridge] No metadata — skipping bridge building")
        return all_nodes, all_edges

    logger.info("[bridge] Building bridges for %d events ...", len(metadata_list))

    # ── Tier 1: Similarity resolution ────────────────────────────────────────
    (
        merged_nodes, merged_edges, updated_meta,
        _entity_merge_map, sim_edges,
    ) = await _resolve_similarity(
        all_nodes, all_edges, metadata_list, embed_func,
        entity_merge_threshold, entity_sim_threshold, frame_sim_threshold,
    )

    # ── Tier 2: Structural bridges ────────────────────────────────────────────
    co_frame_bin, (frame_hub_nodes, frame_hub_edges) = _build_co_frame_edges(
        updated_meta, hub_min=hub_min_events
    )
    co_entity_bin, (entity_hub_nodes, entity_hub_edges) = _build_co_entity_edges(
        updated_meta, hub_min=hub_min_events
    )
    cluster_nodes, cluster_edges = _build_nm_hyperedges(
        updated_meta, nm_min_events=nm_min_events, nm_min_shared=nm_min_shared
    )

    # Tier-3 candidates = all binary bridge pairs with meaningful weight
    causal_candidates: dict[tuple, list] = defaultdict(list)
    for d in (co_frame_bin, co_entity_bin):
        for k, v in d.items():
            causal_candidates[k].extend(v)

    # ── Tier 3: Causal / temporal (LLM) ─────────────────────────────────────
    causal_edges = await _build_causal_edges(
        updated_meta, dict(causal_candidates),
        llm_func, max_causal_pairs, causal_weight_threshold,
    )

    # ── Merge everything ──────────────────────────────────────────────────────
    final_nodes: dict[str, list] = defaultdict(list)
    for d in (merged_nodes, frame_hub_nodes, entity_hub_nodes, cluster_nodes):
        for k, v in d.items():
            final_nodes[k].extend(v)

    final_edges: dict[tuple, list] = defaultdict(list)
    for d in (
        merged_edges, sim_edges,
        co_frame_bin, frame_hub_edges,
        co_entity_bin, entity_hub_edges,
        cluster_edges, causal_edges,
    ):
        for k, v in d.items():
            final_edges[k].extend(v)

    new_node_count = (
        len(frame_hub_nodes) + len(entity_hub_nodes) + len(cluster_nodes)
    )
    new_edge_count = (
        len(sim_edges) + len(co_frame_bin) + len(frame_hub_edges) +
        len(co_entity_bin) + len(entity_hub_edges) +
        len(cluster_edges) + len(causal_edges)
    )
    logger.info(
        "[bridge] Done — new_nodes=%d  new_edges=%d  "
        "(sim=%d co_frame=%d co_entity=%d cluster=%d causal=%d)",
        new_node_count, new_edge_count,
        len(sim_edges), len(co_frame_bin) + len(frame_hub_edges),
        len(co_entity_bin) + len(entity_hub_edges),
        len(cluster_edges), len(causal_edges),
    )
    return dict(final_nodes), dict(final_edges)
