"""
LLM-based semantic frame extractor — FSRAG approach with hyperedge graph.

Replaces the frame-semantic-transformer (FrameNet-constrained) with a fully
LLM-driven pipeline that generates domain-specific frames dynamically.

Offline indexing — three-layer hyperedge graph:

  Layer 0 — EVENT nodes  (one per chunk, entity_type="event")
    Created for every chunk that evokes at least one frame.

  Layer 1 — FRAME nodes  (one per frame type, entity_type="frame")
    Frame names from DynamicFrameDatabase; shared across events.
    EVENT → FRAME edges carry  keywords="evokes", weight=1.0.

  Layer 2 — ENTITY nodes  (leaf entities filling FE roles)
    FRAME → ENTITY edges carry  keywords=<FE_role_name>, weight=1.0.
    The FE role name ("LanguageModel", "KnowledgeSource", …) is chosen over
    generic labels ("agent", "patient") because it encodes domain-specific
    semantics while remaining compact. A secondary proto-role could be added
    as a separate edge attribute later.

  An entity that participates in multiple frames within one chunk will have
  multiple FRAME→ENTITY edges (with different role keywords) all originating
  from the same EVENT via those frames — preserving the multi-role information
  without duplicating the entity node.

Online retrieval (query keyword extraction):
  - High-level keywords: frame names → hit FRAME nodes → expand to entities
  - Low-level keywords: specific entity texts → hit ENTITY nodes directly
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from typing import Any

import json_repair

from lightrag.utils import logger, use_llm_func_with_cache
from lightrag.constants import DEFAULT_ENTITY_NAME_MAX_LENGTH
from lightrag.llm_frame_db import DynamicFrameDatabase, get_frame_db

# Prefix constants — kept here so retrieval code can import them for filtering
EVENT_PREFIX = "EVENT:"
FRAME_PREFIX = "FRAME:"  # frame nodes store name without prefix in entity_name

# ── Internal helpers ──────────────────────────────────────────────────────────

_IDENTIFY_LOCK = asyncio.Lock()  # Serialize frame DB mutations


def _parse_json_response(raw: str) -> Any:
    """Best-effort JSON parse with repair fallback."""
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json_repair.repair_json(raw, return_objects=True)
        except Exception:
            return {}


async def _identify_frames(text: str, llm_func, llm_response_cache=None) -> list[str]:
    """Ask the LLM to identify 1-3 semantic frames evoked by *text*."""
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS["llm_frame_identify"].format(text=text[:3000])
    try:
        raw, _ = await use_llm_func_with_cache(
            prompt, llm_func, llm_response_cache=llm_response_cache, cache_type="extract"
        )
        data = _parse_json_response(raw)
        frames = data.get("frames", [])
        if isinstance(frames, list):
            return [str(f).strip() for f in frames if f]
    except Exception as exc:
        logger.warning("[llm_frame] identify_frames failed: %s", exc)
    return []


async def _define_frame(
    frame_name: str,
    context: str,
    existing_summary: str,
    llm_func,
    llm_response_cache=None,
) -> dict:
    """Ask the LLM to define a new semantic frame."""
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS["llm_frame_define"].format(
        frame_name=frame_name,
        context=context[:1500],
        existing_frames=existing_summary,
    )
    try:
        raw, _ = await use_llm_func_with_cache(
            prompt, llm_func, llm_response_cache=llm_response_cache, cache_type="extract"
        )
        data = _parse_json_response(raw)
        if isinstance(data, dict) and data.get("name"):
            data["name"] = frame_name  # Enforce the requested name
            return data
    except Exception as exc:
        logger.warning("[llm_frame] define_frame '%s' failed: %s", frame_name, exc)
    return {
        "name": frame_name,
        "definition": f"Semantic frame representing '{frame_name}' events.",
        "frame_elements": {},
        "lexical_units": [],
        "relations": [],
    }


async def _check_duplicate(
    new_name: str,
    new_def: str,
    existing_summary: str,
    llm_func,
    llm_response_cache=None,
) -> str | None:
    """Return the name of an existing frame to merge into, or None if distinct."""
    if not existing_summary or existing_summary.startswith("(empty"):
        return None
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS["llm_frame_check_duplicate"].format(
        new_name=new_name,
        new_definition=new_def[:200],
        existing_frames=existing_summary,
    )
    try:
        raw, _ = await use_llm_func_with_cache(
            prompt, llm_func, llm_response_cache=llm_response_cache, cache_type="extract"
        )
        data = _parse_json_response(raw)
        if isinstance(data, dict) and data.get("is_duplicate"):
            merge = data.get("merge_with")
            if merge and isinstance(merge, str):
                return merge.strip()
    except Exception as exc:
        logger.warning("[llm_frame] check_duplicate '%s' failed: %s", new_name, exc)
    return None


async def _score_frame_representativeness(
    frame_name: str,
    frame_definition: str,
    text: str,
    llm_func,
    llm_response_cache=None,
) -> float:
    """
    Ask LLM to score how representative *frame_name* is for the event in *text*.

    Returns a float in [0.0, 1.0].  Falls back to 1.0 on any failure so that
    missing scores never silently zero-out event→frame edges.
    """
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS["llm_frame_representativeness"].format(
        frame_name=frame_name,
        frame_definition=frame_definition[:200],
        text=text[:1500],
    )
    try:
        raw, _ = await use_llm_func_with_cache(
            prompt, llm_func, llm_response_cache=llm_response_cache, cache_type="extract"
        )
        data = _parse_json_response(raw)
        if isinstance(data, dict):
            score = data.get("score", 1.0)
            score = float(score)
            score = max(0.0, min(1.0, score))   # clamp to [0, 1]
            reasoning = data.get("reasoning", "")
            logger.debug(
                "[llm_frame] representativeness '%s' = %.2f  (%s)",
                frame_name, score, reasoning[:80],
            )
            return score
    except Exception as exc:
        logger.warning(
            "[llm_frame] score_representativeness '%s' failed: %s — using 1.0",
            frame_name, exc,
        )
    return 1.0


async def _extract_instances(
    text: str,
    frames_info: str,
    llm_func,
    llm_response_cache=None,
) -> list[dict]:
    """Extract entity texts filling FE roles within each detected frame."""
    from lightrag.prompt import PROMPTS

    prompt = PROMPTS["llm_frame_extract_instances"].format(
        frames_info=frames_info,
        text=text[:3000],
    )
    try:
        raw, _ = await use_llm_func_with_cache(
            prompt, llm_func, llm_response_cache=llm_response_cache, cache_type="extract"
        )
        data = _parse_json_response(raw)
        instances = data.get("instances", [])
        if isinstance(instances, list):
            return instances
    except Exception as exc:
        logger.warning("[llm_frame] extract_instances failed: %s", exc)
    return []


def _build_frames_info(frames: list[dict]) -> str:
    """Format frame definitions as a compact text block for the extraction prompt."""
    lines: list[str] = []
    for f in frames:
        name = f.get("name", "?")
        defn = f.get("definition", "")[:120]
        fes = f.get("frame_elements", {})
        fe_lines = [
            f"    - {role} ({v.get('type','?')}): {v.get('definition','')[:80]}"
            for role, v in fes.items()
        ]
        lines.append(f"Frame: {name}\nDefinition: {defn}")
        if fe_lines:
            lines.append("Frame Elements:\n" + "\n".join(fe_lines))
        lines.append("")
    return "\n".join(lines)


# ── Ensure frame exists in DB, creating it if necessary ───────────────────────

async def _ensure_frame(
    frame_name: str,
    context: str,
    frame_db: DynamicFrameDatabase,
    llm_func,
    llm_response_cache=None,
) -> str:
    """
    Ensure *frame_name* is in *frame_db*.

    If a case-insensitive match already exists, return that canonical name.
    Otherwise, define the frame (LLM), check for duplicates (LLM), and add it.
    Returns the canonical frame name to use.

    Lock discipline: hold lock only for cheap DB reads/writes; LLM calls run
    outside the lock so concurrent chunks don't serialize on LLM latency.
    """
    # 1. Fast path: check DB under lock (no LLM needed)
    async with _IDENTIFY_LOCK:
        candidate = frame_db.find_candidate(frame_name)
        if candidate:
            return candidate
        existing_summary = frame_db.summary_for_llm()
        frame_count = len(frame_db.frames)

    # 2. LLM calls outside the lock — these are slow and can run concurrently
    frame_dict = await _define_frame(
        frame_name, context, existing_summary, llm_func, llm_response_cache
    )

    merge_target: str | None = None
    if frame_count >= 3:
        merge_target = await _check_duplicate(
            frame_name,
            frame_dict.get("definition", ""),
            existing_summary,
            llm_func,
            llm_response_cache,
        )

    # 3. Re-check and write under lock (another coroutine may have added it)
    async with _IDENTIFY_LOCK:
        candidate = frame_db.find_candidate(frame_name)
        if candidate:
            return candidate

        if merge_target and frame_db.has_frame(merge_target):
            logger.info(
                "[frame_db] '%s' merged into existing frame '%s'",
                frame_name,
                merge_target,
            )
            return merge_target

        frame_db.add_frame(frame_dict)
        await frame_db.save()
        logger.info("[frame_db] New frame created: '%s'", frame_name)
        return frame_name


# ── Public API — Offline indexing ──────────────────────────────────────────────

async def extract_entities_from_frames_llm(
    text: str,
    chunk_key: str,
    file_path: str,
    llm_func,
    working_dir: str,
    llm_response_cache=None,
) -> tuple[dict, dict]:
    """
    Build a three-layer hyperedge graph from *text* using LLM-generated frames.

    Graph layers
    ───────────
    EVENT  (entity_type="event")  — one node per chunk, id=EVENT:<hash8>
      │  edge keywords="evokes", weight=1.0
    FRAME  (entity_type="frame") — one node per frame type (shared across chunks)
      │  edge keywords=<FE_role_name>, weight=1.0
    ENTITY (entity_type=<role>)  — leaf entity texts filling FE slots

    Returns:
        (maybe_nodes, maybe_edges) — compatible with _process_extraction_result()
    """
    timestamp = int(time.time())
    frame_db = get_frame_db(working_dir)

    # Step 1: Identify frames
    chunk_hash = hashlib.md5(chunk_key.encode()).hexdigest()[:8]
    event_id_early = f"{EVENT_PREFIX}{chunk_hash}"

    frame_names = await _identify_frames(text, llm_func, llm_response_cache)
    if not frame_names:
        logger.debug("[llm_frame] No frames detected in chunk %s", chunk_key)
        return {}, {}, {"event_id": event_id_early, "frame_names": []}

    # Step 2: Ensure each frame is in the DB (create if missing)
    canonical_names: list[str] = []
    for name in frame_names:
        canon = await _ensure_frame(name, text, frame_db, llm_func, llm_response_cache)
        if canon not in canonical_names:
            canonical_names.append(canon)

    # Step 3: Extract entity texts filling FE roles
    frames_info = _build_frames_info(
        [frame_db.get_frame(n) for n in canonical_names if frame_db.get_frame(n)]
    )
    instances = await _extract_instances(text, frames_info, llm_func, llm_response_cache)

    # Step 4: Build three-layer hyperedge graph
    maybe_nodes: dict[str, list[dict]] = defaultdict(list)
    maybe_edges: dict[tuple[str, str], list[dict]] = defaultdict(list)

    # ── Layer 0: EVENT node ────────────────────────────────────────────────────
    # One event per chunk; short hash suffix keeps the ID human-readable.
    chunk_hash = hashlib.md5(chunk_key.encode()).hexdigest()[:8]
    event_id = f"{EVENT_PREFIX}{chunk_hash}"
    event_desc = (
        f"Semantic event extracted from chunk {chunk_key}. "
        f"Evokes frames: {', '.join(canonical_names)}."
    )
    maybe_nodes[event_id].append(
        {
            "entity_name": event_id,
            "entity_type": "event",
            "description": event_desc,
            "source_id": chunk_key,
            "file_path": file_path,
            "timestamp": timestamp,
        }
    )

    # ── Layer 1: FRAME nodes + EVENT→FRAME edges (LLM-scored weights) ───────────
    # Score all frames concurrently to avoid sequential LLM latency.
    score_tasks = {}
    for frame_name in canonical_names:
        frame_data = frame_db.get_frame(frame_name)
        if not frame_data:
            continue
        defn = frame_data.get("definition", "")
        score_tasks[frame_name] = asyncio.create_task(
            _score_frame_representativeness(frame_name, defn, text, llm_func, llm_response_cache)
        )

    # Await all scores (dict: frame_name → float)
    frame_scores: dict[str, float] = {}
    for fname, task in score_tasks.items():
        frame_scores[fname] = await task

    for frame_name in canonical_names:
        frame_data = frame_db.get_frame(frame_name)
        if not frame_data:
            continue

        # Frame node — entity_type="frame" distinguishes it from leaf entities.
        # The same FRAME node is created/updated across chunks; LightRAG's
        # entity-merging accumulates descriptions from every chunk that evokes it.
        frame_defn = frame_data.get("definition", f"Semantic frame: {frame_name}")
        fe_names = ", ".join(frame_data.get("frame_elements", {}).keys())
        frame_desc = f"{frame_defn}  [FEs: {fe_names}]"
        maybe_nodes[frame_name].append(
            {
                "entity_name": frame_name,
                "entity_type": "frame",
                "description": frame_desc,
                "source_id": chunk_key,
                "file_path": file_path,
                "timestamp": timestamp,
            }
        )

        # EVENT → FRAME edge.
        # weight = LLM representativeness score (0.0–1.0):
        #   1.0 → frame perfectly captures the central event
        #   0.4 → frame is tangential / a minor aspect
        # This weight is used by retrieval to rank frame paths.
        rep_score = frame_scores.get(frame_name, 1.0)
        maybe_edges[(event_id, frame_name)].append(
            {
                "src_id": event_id,
                "tgt_id": frame_name,
                "weight": rep_score,
                "keywords": "evokes",
                "description": (
                    f"[EVENT→FRAME] Chunk {chunk_key} evokes frame {frame_name} "
                    f"(representativeness={rep_score:.2f})."
                ),
                "source_id": chunk_key,
                "file_path": file_path,
                "timestamp": timestamp,
            }
        )

    # ── Layer 2: ENTITY nodes + FRAME→ENTITY edges ────────────────────────────
    # Group instances by frame so we can aggregate roles for the same entity.
    for inst in instances:
        frame_name = inst.get("frame", "")
        elements: dict[str, str] = inst.get("elements", {})
        if not isinstance(elements, dict) or not elements:
            continue
        if frame_name not in canonical_names:
            continue

        frame_data = frame_db.get_frame(frame_name) or {}
        fe_defs: dict = frame_data.get("frame_elements", {})

        for role, entity_text in elements.items():
            if not entity_text or not isinstance(entity_text, str):
                continue
            entity_text = entity_text.strip()
            if not entity_text:
                continue

            entity_name = entity_text[:DEFAULT_ENTITY_NAME_MAX_LENGTH]

            # Entity node.
            # entity_type = FE role lowercased — retains domain semantics.
            # An entity appearing in multiple frames accumulates descriptions;
            # the merged node will reflect all its roles across frames.
            fe_info = fe_defs.get(role, {})
            fe_type = fe_info.get("type", "core")  # "core" | "peripheral"
            entity_desc = (
                f'"{entity_text}" fills the {role} ({fe_type}) role '
                f'in the "{frame_name}" frame.'
            )
            maybe_nodes[entity_name].append(
                {
                    "entity_name": entity_name,
                    "entity_type": role.lower().replace(" ", "_"),
                    "description": entity_desc,
                    "source_id": chunk_key,
                    "file_path": file_path,
                    "timestamp": timestamp,
                }
            )

            # FRAME → ENTITY edge.
            # keywords = FE role name — the most semantically precise label
            # available: it names exactly *how* this entity participates in
            # the frame ("LanguageModel", "KnowledgeSource", …).  A future
            # enhancement can add a proto-role ("agent", "patient", …) as a
            # secondary attribute without changing this field.
            fe_defn_snippet = fe_info.get("definition", "")[:100]
            edge_desc = (
                f"[{frame_name}/{role}] "
                f'"{entity_name}" is the {role} ({fe_type}). '
                + (f"FE def: {fe_defn_snippet}" if fe_defn_snippet else "")
            )
            maybe_edges[(frame_name, entity_name)].append(
                {
                    "src_id": frame_name,
                    "tgt_id": entity_name,
                    "weight": 1.0,
                    "keywords": role,
                    "description": edge_desc,
                    "source_id": chunk_key,
                    "file_path": file_path,
                    "timestamp": timestamp,
                }
            )

    logger.debug(
        "[llm_frame] chunk %s → event=%s frames=%s | nodes=%d edges=%d",
        chunk_key,
        event_id,
        canonical_names,
        len(maybe_nodes),
        len(maybe_edges),
    )
    # frame_meta is consumed by bridge_builder to build cross-event connections.
    frame_meta = {
        "event_id": event_id,
        "frame_names": canonical_names,
    }
    return dict(maybe_nodes), dict(maybe_edges), frame_meta


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _cosine_sim(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    import numpy as np
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


async def _canonicalize_frame_names(
    raw_names: list[str],
    frame_db: "DynamicFrameDatabase",
    embed_func,
    sim_threshold: float = 0.75,
) -> list[str]:
    """
    Snap LLM-generated frame names to canonical names in frame_db.

    For each raw_name:
    1. Exact / case-insensitive match → use canonical name.
    2. Embedding cosine similarity >= sim_threshold → snap to best match.
    3. No match → keep raw name (may still be valid if graph has it literally).

    Returns deduplicated list preserving input order.
    """
    if not frame_db.frames or embed_func is None:
        return raw_names

    known_names = frame_db.get_all_names()
    if not known_names:
        return raw_names

    canonicalized: list[str] = []
    seen: set[str] = set()

    # Batch-embed all names in one call
    all_texts = raw_names + known_names
    try:
        all_vecs = await embed_func(all_texts)
    except Exception as exc:
        logger.warning("[llm_frame] canonicalize embed failed: %s — skipping snap", exc)
        return raw_names

    import numpy as np
    raw_vecs   = all_vecs[: len(raw_names)]
    known_vecs = all_vecs[len(raw_names) :]

    for i, raw in enumerate(raw_names):
        # 1. Exact / case-insensitive match (free, no embedding needed)
        candidate = frame_db.find_candidate(raw)
        if candidate:
            if candidate not in seen:
                canonicalized.append(candidate)
                seen.add(candidate)
            continue

        # 2. Embedding similarity snap
        sims = np.array([_cosine_sim(raw_vecs[i], known_vecs[j])
                         for j in range(len(known_names))])
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= sim_threshold:
            canon = known_names[best_idx]
            logger.debug(
                "[llm_frame] canonicalize '%s' → '%s' (sim=%.3f)",
                raw, canon, best_sim,
            )
            if canon not in seen:
                canonicalized.append(canon)
                seen.add(canon)
        else:
            # Keep as-is; FAGE will just find nothing and move on
            if raw not in seen:
                canonicalized.append(raw)
                seen.add(raw)

    return canonicalized


async def _expand_related_frames(
    query_frames: list[str],
    frame_db: "DynamicFrameDatabase",
    llm_func,
    llm_response_cache=None,
    max_related: int = 3,
) -> list[str]:
    """
    Latent Frame Relation expansion (FS-RAG paper, §3.2).

    Ask the LLM: given the frames the query directly evokes, which OTHER
    frames in the database likely contain facts needed to answer the query?

    Returns a list of canonical frame names (subset of frame_db), deduplicated
    against query_frames.
    """
    if not query_frames or not frame_db.frames:
        return []

    # Only worth calling when DB has enough frames to expand into
    known_names = frame_db.get_all_names()
    # Remove frames already in query_frames to keep prompt focused
    candidates = [n for n in known_names if n not in query_frames]
    if not candidates:
        return []

    from lightrag.prompt import PROMPTS

    query_frames_str = ", ".join(query_frames)
    frame_db_summary = frame_db.summary_for_llm(max_frames=30)

    prompt = PROMPTS["llm_frame_related_for_query"].format(
        query_frames=query_frames_str,
        frame_db_summary=frame_db_summary,
    )
    try:
        raw, _ = await use_llm_func_with_cache(
            prompt, llm_func, llm_response_cache=llm_response_cache, cache_type="extract"
        )
        data = _parse_json_response(raw)
        related = data.get("related_frames", [])
        if not isinstance(related, list):
            return []

        # Only accept names that actually exist in DB
        valid = [
            str(r).strip() for r in related
            if isinstance(r, str) and frame_db.has_frame(str(r).strip())
        ]
        result = valid[:max_related]
        if result:
            logger.info(
                "[llm_frame] latent expansion: %s → +%s",
                query_frames, result,
            )
        return result
    except Exception as exc:
        logger.warning("[llm_frame] expand_related_frames failed: %s", exc)
    return []


# ── Public API — Online retrieval ──────────────────────────────────────────────

async def extract_keywords_from_frames_llm(
    text: str,
    llm_func,
    working_dir: str,
    llm_response_cache=None,
    embed_func=None,
) -> tuple[list[str], list[str]]:
    """
    Extract (hl_keywords, ll_keywords) from *text* using LLM frame analysis.

    Pipeline:
    1. LLM extracts raw frame names (hl) + entity names (ll) from query.
    2. Canonicalize hl frame names against frame_db via embedding similarity
       (Improvement 2 — fixes silent mismatch between LLM-generated names and
       the canonical names actually stored in the graph).
    3. Latent Frame Relation expansion (Improvement 1 — FS-RAG §3.2): ask LLM
       which other frames in the DB likely contain facts needed to answer the
       query, and append those to hl_keywords.

    With the hyperedge graph structure:
    - hl_keywords (canonical frame names) → FAGE hits FRAME nodes directly
    - ll_keywords (entity texts) → hit ENTITY leaf nodes via VDB
    """
    from lightrag.prompt import PROMPTS

    frame_db = get_frame_db(working_dir)
    prompt = PROMPTS["llm_frame_keywords"].format(text=text[:2000])
    try:
        raw, _ = await use_llm_func_with_cache(
            prompt, llm_func, llm_response_cache=llm_response_cache, cache_type="extract"
        )
        data = _parse_json_response(raw)
        hl = data.get("high_level_keywords", [])
        ll = data.get("low_level_keywords", [])
        if isinstance(hl, list) and isinstance(ll, list):
            hl_keywords = [str(k).strip() for k in hl if k]
            ll_keywords = [str(k).strip() for k in ll if k]

            # ── Improvement 2: canonicalize frame names ────────────────────────
            # Snap LLM-generated names ("VectorDatabase") to canonical DB names
            # ("VectorStorage") using embedding similarity. Falls back to exact
            # match when embed_func is None.
            hl_keywords = await _canonicalize_frame_names(
                hl_keywords, frame_db, embed_func
            )

            # ── Improvement 1: latent frame relation expansion ─────────────────
            # Ask LLM which other frames in the DB are needed to answer the query.
            # Run concurrently with ll_keywords ll enrichment below.
            related_task = asyncio.create_task(
                _expand_related_frames(
                    hl_keywords, frame_db, llm_func, llm_response_cache
                )
            )

            # Enrich hl_keywords: if an ll entity exactly matches a known frame name,
            # promote it to hl (existing behaviour, kept for backward compat).
            known = set(frame_db.get_all_names())
            for word in ll_keywords[:]:
                if word in known and word not in hl_keywords:
                    hl_keywords.append(word)

            # Await latent expansion and merge (deduplicated)
            related = await related_task
            for fname in related:
                if fname not in hl_keywords:
                    hl_keywords.append(fname)

            logger.debug(
                "[llm_frame] keywords — hl=%s ll=%s", hl_keywords, ll_keywords
            )
            return hl_keywords, ll_keywords
    except Exception as exc:
        logger.warning("[llm_frame] extract_keywords_llm failed: %s", exc)
    return [], []
