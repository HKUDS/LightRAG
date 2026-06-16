"""Core extraction pipeline for FrameRAG.

Three extraction calls per chunk:
  Call 1: Entity extraction (extended from LightRAG)
  Call 2: Event detection + Frame induction + Role assignment (combined, with Frame DB hints)
  Call 3: Causal/temporal edge extraction (optional)
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Callable, Awaitable, Optional

from lightrag.utils import logger

from .types import (
    ChunkSchema,
    EntityMentionSchema,
    EventSchema,
    FrameInstanceSchema,
    FrameDefinitionSchema,
    InfoNodeSchema,
    CausalEdgeSchema,
    FEAssignment,
    CoreFESchema,
    NonCoreFESchema,
)

from .frame_db import FrameDatabase
from .prompt import PROMPTS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_json(text: str) -> list | dict | None:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Use raw_decode to correctly handle prefix/suffix text and embedded
        # brackets inside string values (rfind-based approach breaks on those).
        decoder = json.JSONDecoder()
        for start_char in ("[", "{"):
            idx = text.find(start_char)
            if idx == -1:
                continue
            try:
                result, _ = decoder.raw_decode(text, idx)
                return result
            except json.JSONDecodeError:
                pass
        return None


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _normalize_frame_name(name: str) -> str:
    """Enforce consistent PascalCase_Underscore format for frame names.

    e.g. 'communication_tell' → 'Communication_Tell'
         'NOT_affect'         → 'NOT_Affect'
    """
    if not name:
        return name
    parts = name.replace("-", "_").split("_")
    normalized = []
    for p in parts:
        if not p:
            continue
        if p.upper() == "NOT":
            normalized.append("NOT")
        else:
            normalized.append(p.capitalize())
    return "_".join(normalized)


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper
# ─────────────────────────────────────────────────────────────────────────────

async def _llm_with_retry(
    llm_func: Callable[..., Awaitable[str]],
    prompt: str,
    max_retries: int = 7,
) -> str:
    """LLM call with exponential backoff. Handles 429 rate limits with longer waits."""
    # Backoff schedule (seconds): 2, 5, 15, 30, 60, 120, 180
    _BACKOFF = [2, 5, 15, 30, 60, 120, 180]
    for attempt in range(max_retries):
        try:
            return await llm_func(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"[operate] LLM failed after {max_retries} attempts: {e}")
                raise
            err_str = str(e)
            # 429 rate-limit or 500 server error: use longer wait
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
            elif "500" in err_str or "server" in err_str.lower():
                wait = _BACKOFF[min(attempt, len(_BACKOFF) - 1)]
            else:
                wait = 2 ** attempt
            logger.warning(f"[operate] LLM attempt {attempt + 1} failed, retry in {wait}s: {e}")
            await asyncio.sleep(wait)


# ─────────────────────────────────────────────────────────────────────────────
# Call 1b: Entity Extraction Gleaning
# ─────────────────────────────────────────────────────────────────────────────

async def glean_entities(
    chunk: ChunkSchema,
    existing_mentions: list[EntityMentionSchema],
    llm_func: Callable[..., Awaitable[str]],
    max_rounds: int = 1,
) -> list[EntityMentionSchema]:
    """Multi-round gleaning — repeatedly ask LLM for missed entities.

    Each round passes the *accumulated* entity list so the LLM knows what
    has already been found. Stops early when a round adds nothing new.

    Args:
        chunk:             The source chunk being processed.
        existing_mentions: Entities found in the initial extraction pass.
        llm_func:          Cached LLM callable.
        max_rounds:        Maximum gleaning iterations (0 = disabled, default 1).
    """
    if max_rounds <= 0 or not existing_mentions:
        return []

    all_new: list[EntityMentionSchema] = []
    accumulated = list(existing_mentions)

    for round_idx in range(max_rounds):
        existing_list = "\n".join(
            f"  - {m.name} [{m.entity_type}]" for m in accumulated
        )
        prompt = PROMPTS["entity_extraction_glean"].format(
            existing_entities=existing_list,
            chunk_text=chunk.text,
        )
        try:
            response = await _llm_with_retry(llm_func, prompt)
        except Exception as e:
            logger.warning(
                f"[operate] Gleaning round {round_idx + 1} failed for "
                f"chunk {chunk.chunk_id}: {e}"
            )
            break

        raw = _safe_json(response)
        if not isinstance(raw, list):
            break

        known_names = {m.name.lower() for m in accumulated}
        round_new: list[EntityMentionSchema] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("entity_name", "").strip()
            if not name or name.lower() in known_names:
                continue
            mention = EntityMentionSchema(
                mention_id=_gen_id("em"),
                chunk_id=chunk.chunk_id,
                name=name,
                entity_type=item.get("entity_type", "OTHER"),
                description=item.get("entity_description", ""),
                aliases=item.get("entity_aliases", []),
                salience=item.get("entity_salience", "MEDIUM"),
            )
            round_new.append(mention)
            accumulated.append(mention)

        logger.debug(
            f"[operate] Chunk {chunk.chunk_id}: gleaning round {round_idx + 1} "
            f"+{len(round_new)} entities"
        )
        all_new.extend(round_new)

        # Early stop: nothing new this round, further rounds won't help
        if not round_new:
            logger.debug(
                f"[operate] Chunk {chunk.chunk_id}: gleaning stopped early "
                f"after {round_idx + 1} round(s)"
            )
            break

    return all_new


# ─────────────────────────────────────────────────────────────────────────────
# Query-time Frame Relation Expansion
# ─────────────────────────────────────────────────────────────────────────────

async def expand_query_frames(
    query: str,
    primary_frame: str,
    frame_db: FrameDatabase,
    top_k: int = 8,
    threshold: float = 0.50,
) -> list[str]:
    """Expand primary frame to related frames via embedding-based DB search."""
    if not primary_frame:
        return []
    search_text = f"{primary_frame}: {query}"
    try:
        return await frame_db.search_related_frames(
            search_text,
            top_k=top_k,
            threshold=threshold,
            exclude={primary_frame},
        )
    except Exception as e:
        logger.warning(f"[operate] Embedding-based frame expansion failed: {e}")
        return []


async def expand_query_frames_llm(
    query: str,
    frame_db: FrameDatabase,
    llm_func: Callable[..., Awaitable[str]],
    max_frames_in_prompt: int = 300,
) -> list[str]:
    """Use LLM to select relevant frames directly from the frame DB.

    More accurate than embedding-based expansion for narrative QA because the
    LLM can reason about which event types are relevant without needing a
    primary frame hint. Falls back to [] on any error.
    """
    try:
        all_frames = await frame_db.all_frames()
        if not all_frames:
            return []

        # Sort by usage count desc, cap to avoid overflowing context
        sorted_frames = sorted(all_frames, key=lambda f: f.usage_count, reverse=True)
        names = [f.frame_name for f in sorted_frames[:max_frames_in_prompt]]
        frame_list = ", ".join(names)

        prompt = PROMPTS["frame_selection"].format(
            frame_list=frame_list,
            query=query,
        )
        response = await _llm_with_retry(llm_func, prompt)
        parsed = _safe_json(response)
        if isinstance(parsed, list):
            valid = [f for f in parsed if isinstance(f, str) and f in set(names)]
            logger.debug(f"[operate] LLM frame selection: {valid}")
            return valid
    except Exception as e:
        logger.warning(f"[operate] LLM frame expansion failed: {e}")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Call 1: Entity Extraction
# ─────────────────────────────────────────────────────────────────────────────

async def extract_entities(
    chunk: ChunkSchema,
    llm_func: Callable[..., Awaitable[str]],
) -> list[EntityMentionSchema]:
    """Extract entity mentions from a single chunk via LLM (Call 1)."""
    prompt = PROMPTS["entity_extraction"].format(chunk_text=chunk.text)
    try:
        response = await _llm_with_retry(llm_func, prompt)
    except Exception as e:
        logger.error(f"[operate] Entity extraction failed for chunk {chunk.chunk_id}: {e}")
        return []

    raw = _safe_json(response)
    if not isinstance(raw, list):
        logger.warning(f"[operate] Entity extraction returned non-list for chunk {chunk.chunk_id}")
        return []

    mentions: list[EntityMentionSchema] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("entity_name", "").strip()
        if not name:
            continue
        mentions.append(EntityMentionSchema(
            mention_id=_gen_id("em"),
            chunk_id=chunk.chunk_id,
            name=name,
            entity_type=item.get("entity_type", "OTHER"),
            description=item.get("entity_description", ""),
            aliases=item.get("entity_aliases", []),
            salience=item.get("entity_salience", "MEDIUM"),
        ))

    logger.debug(f"[operate] Chunk {chunk.chunk_id}: {len(mentions)} entity mentions extracted")
    return mentions


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for FE-filler resolution
# ─────────────────────────────────────────────────────────────────────────────

def _build_mention_lookup(
    entity_mentions: list[EntityMentionSchema],
) -> dict[str, EntityMentionSchema]:
    mention_by_name: dict[str, EntityMentionSchema] = {}
    for m in entity_mentions:
        mention_by_name[m.name.lower()] = m
        for alias in (m.aliases or []):
            mention_by_name[alias.lower()] = m
    return mention_by_name


def _make_filler_resolver(mention_by_name: dict[str, EntityMentionSchema]):
    """Return a closure that resolves a filler string to an EntityMentionSchema.

    Exact match first; fall back to longest-overlap substring match.
    Returns None for ambiguous matches (multiple mentions tie on overlap).
    """
    def _resolve_filler(filler_text: str) -> Optional[EntityMentionSchema]:
        key = filler_text.lower().strip()
        if key in mention_by_name:
            return mention_by_name[key]
        if len(key) < 3:
            return None
        candidates: list[tuple[int, EntityMentionSchema]] = []
        for name, m in mention_by_name.items():
            if key in name or name in key:
                overlap = len(set(key.split()) & set(name.split()))
                candidates.append((overlap, m))
        if not candidates:
            return None
        max_overlap = max(c[0] for c in candidates)
        best = [m for score, m in candidates if score == max_overlap]
        return best[0] if len(best) == 1 else None

    return _resolve_filler


def _format_entity_list(entity_mentions: list[EntityMentionSchema]) -> str:
    entity_lines = []
    for m in entity_mentions:
        aliases_str = f" (also: {', '.join(m.aliases)})" if m.aliases else ""
        entity_lines.append(f"  - {m.name} [{m.entity_type}]{aliases_str}: {m.description}")
    return "\n".join(entity_lines) if entity_lines else "  (none)"


# ─────────────────────────────────────────────────────────────────────────────
# Call 2 — Step A: Event Detection ONLY (one LLM call per chunk)
# ─────────────────────────────────────────────────────────────────────────────

async def extract_events(
    chunk: ChunkSchema,
    entity_mentions: list[EntityMentionSchema],
    llm_func: Callable[..., Awaitable[str]],
) -> list[dict]:
    """Step A — detect raw events in a chunk (no frames).

    Returns a list of raw event dicts with keys:
        event_span, description, participant_names, temporal_marker, is_negation
    Frame assignment happens later in Step B (annotate_frames_for_event).
    """
    entity_list_text = _format_entity_list(entity_mentions)
    prompt = PROMPTS["event_extraction"].format(
        entity_list=entity_list_text,
        chunk_text=chunk.text,
    )
    try:
        response = await _llm_with_retry(llm_func, prompt)
    except Exception as e:
        logger.error(f"[operate] Event extraction failed for chunk {chunk.chunk_id}: {e}")
        return []

    raw = _safe_json(response)
    # Unwrap {"events": [...]} or {"results": [...]} if LLM wrapped the array
    if isinstance(raw, dict):
        for key in ("events", "results", "items", "data"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        preview = (response or "")[:300].replace("\n", " ")
        logger.warning(
            f"[operate] Event extraction returned non-list for chunk {chunk.chunk_id} "
            f"(type={type(raw).__name__}); LLM response preview: {preview!r}"
        )
        return []

    events: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        span = (item.get("event_span") or item.get("trigger") or "").strip()
        desc = (item.get("description") or item.get("event_description") or "").strip()
        if not span and not desc:
            continue
        events.append({
            "event_span":       span,
            "description":      desc or span,
            "participant_names": [
                str(p).strip() for p in (item.get("participant_names") or []) if str(p).strip()
            ],
            "temporal_marker":  (item.get("temporal_marker") or "").strip(),
            "is_negation":      bool(item.get("is_negation", False)),
        })

    logger.debug(f"[operate] Chunk {chunk.chunk_id}: {len(events)} raw events (Step A)")
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Call 2 — Step B: Frame Annotation for ONE event (parallelizable)
# ─────────────────────────────────────────────────────────────────────────────

def _context_window(chunk_text: str, event_span: str, window: int = 300) -> str:
    """Return up to `window` chars around event_span in chunk_text."""
    if not event_span:
        return chunk_text[:window * 2]
    idx = chunk_text.lower().find(event_span.lower()[:40])
    if idx == -1:
        return chunk_text[:window * 2]
    start = max(0, idx - window)
    end = min(len(chunk_text), idx + len(event_span) + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(chunk_text) else ""
    return prefix + chunk_text[start:end] + suffix


async def annotate_frames_for_event(
    event: dict,
    chunk_text: str,
    entity_mentions: list[EntityMentionSchema],
    llm_func: Callable[..., Awaitable[str]],
    frame_db: FrameDatabase,
) -> Optional[dict]:
    """Step B — assign a FrameNet-style frame + FEs to a single raw event.

    One LLM call per event. Returns the parsed frame annotation dict, or None on failure.
    Keys: frame_name, frame_definition, is_new_frame, lexical_unit,
          core_elements, noncore_elements.
    """
    event_span = event.get("event_span", "")
    hint_text = f"{event_span} {event.get('description', '')}".strip()
    hints = await frame_db.get_hints_for_chunk(hint_text or chunk_text)
    hints_text = frame_db.format_hints_for_prompt(hints)
    entity_list_text = _format_entity_list(entity_mentions)
    # Pass only a small context window around the event span, not the full chunk
    context_snippet = _context_window(chunk_text, event_span, window=300)

    prompt = PROMPTS["frame_annotation"].format(
        frame_db_hints=hints_text,
        entity_list=entity_list_text,
        event_span=event_span,
        event_description=event.get("description", ""),
        participant_names=", ".join(event.get("participant_names", [])) or "(none)",
        is_negation=event.get("is_negation", False),
        chunk_text=context_snippet,
    )
    try:
        response = await _llm_with_retry(llm_func, prompt)
    except Exception as e:
        logger.warning(f"[operate] Frame annotation failed: {e}")
        return None

    parsed = _safe_json(response)
    if not isinstance(parsed, dict):
        preview = (response or "")[:300].replace("\n", " ")
        logger.warning(
            f"[operate] Frame annotation returned non-dict for event '{event_span[:60]}' "
            f"(type={type(parsed).__name__}); LLM response preview: {preview!r}"
        )
        return None
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# Call 2 — Combine: run Step A then parallel Step B, build final schemas
# ─────────────────────────────────────────────────────────────────────────────

async def extract_events_frames_two_step(
    chunk: ChunkSchema,
    entity_mentions: list[EntityMentionSchema],
    llm_func: Callable[..., Awaitable[str]],
    frame_db: FrameDatabase,
) -> tuple[
    list[EventSchema],
    list[FrameInstanceSchema],
    list[InfoNodeSchema],
    list[FrameDefinitionSchema],
]:
    """2-step Call 2: (A) detect events, then (B) annotate frames per event in parallel.

    Returns the same 4-tuple as the legacy ``extract_events_frames_roles``:
        events, frame_instances, info_nodes, new_frames
    """
    raw_events = await extract_events(chunk, entity_mentions, llm_func)
    if not raw_events:
        return [], [], [], []

    # Step B — annotate frames per event, fully parallel.
    annotations = await asyncio.gather(*[
        annotate_frames_for_event(ev, chunk.text, entity_mentions, llm_func, frame_db)
        for ev in raw_events
    ])

    mention_by_name = _build_mention_lookup(entity_mentions)
    resolve_filler = _make_filler_resolver(mention_by_name)

    events: list[EventSchema] = []
    frame_instances: list[FrameInstanceSchema] = []
    info_nodes: list[InfoNodeSchema] = []
    new_frames: list[FrameDefinitionSchema] = []

    def _build_assignments(
        elements: list, is_core: bool
    ) -> list[FEAssignment]:
        out: list[FEAssignment] = []
        for fe_item in elements or []:
            if not isinstance(fe_item, dict):
                continue
            fe_name = (fe_item.get("fe_name") or "").strip()
            if not fe_name:
                continue
            filler_text = fe_item.get("filler_text")
            filler_type_raw = fe_item.get("filler_type", "ENTITY" if is_core else "VALUE")
            is_missing = fe_item.get("is_missing", filler_text is None)

            if is_missing or not filler_text:
                out.append(FEAssignment(
                    fe_name=fe_name, filler_id=None, filler_type="MISSING",
                    filler_text="", is_core=is_core,
                ))
                continue

            if filler_type_raw == "ENTITY":
                matched = resolve_filler(str(filler_text))
                out.append(FEAssignment(
                    fe_name=fe_name,
                    filler_id=matched.mention_id if matched else None,
                    filler_type="ENTITY" if matched else "MISSING",
                    filler_text=str(filler_text),
                    is_core=is_core,
                ))
            else:
                info_id = _gen_id("info")
                info_nodes.append(InfoNodeSchema(
                    info_id=info_id,
                    value=str(filler_text),
                    info_type=filler_type_raw,
                ))
                out.append(FEAssignment(
                    fe_name=fe_name, filler_id=info_id, filler_type="VALUE",
                    filler_text=str(filler_text), is_core=is_core,
                ))
        return out

    for raw_ev, ann in zip(raw_events, annotations):
        if not ann:
            continue
        frame_name = _normalize_frame_name((ann.get("frame_name") or "").strip())
        if not frame_name:
            continue

        # Derive trigger lemma from event span / lexical unit.
        lexical_unit = ann.get("lexical_unit") or ""
        trigger_lemma = lexical_unit.split(".")[0].strip() if lexical_unit else ""
        if not trigger_lemma:
            trigger_lemma = (raw_ev.get("event_span", "").split() or ["event"])[0]
        if not lexical_unit:
            lexical_unit = f"{trigger_lemma}.v"

        participant_mention_ids: list[str] = []
        for pname in raw_ev.get("participant_names", []):
            matched = resolve_filler(pname)
            if matched and matched.mention_id not in participant_mention_ids:
                participant_mention_ids.append(matched.mention_id)

        event_id = _gen_id("ev")
        event = EventSchema(
            event_id=event_id,
            chunk_id=chunk.chunk_id,
            trigger=raw_ev.get("event_span", trigger_lemma),
            trigger_lemma=trigger_lemma,
            trigger_pos="VERB",
            event_span=raw_ev.get("event_span", ""),
            event_description=raw_ev.get("description", ""),
            frame_name=frame_name,
            participant_mention_ids=participant_mention_ids,
        )
        events.append(event)

        core_assignments = _build_assignments(ann.get("core_elements"), is_core=True)
        noncore_assignments = _build_assignments(ann.get("noncore_elements"), is_core=False)

        # Promote temporal_marker to a non-core Time FE if the LLM did not capture it.
        tmarker = raw_ev.get("temporal_marker", "")
        if tmarker and not any(
            a.fe_name.lower() in ("time", "duration") for a in noncore_assignments
        ):
            info_id = _gen_id("info")
            info_nodes.append(InfoNodeSchema(info_id=info_id, value=tmarker, info_type="TIME"))
            noncore_assignments.append(FEAssignment(
                fe_name="Time", filler_id=info_id, filler_type="VALUE",
                filler_text=tmarker, is_core=False,
            ))

        fi_id = _gen_id("fi")
        frame_instance = FrameInstanceSchema(
            fi_id=fi_id,
            event_id=event_id,
            frame_name=frame_name,
            lexical_unit=lexical_unit,
            core_assignments=core_assignments,
            noncore_assignments=noncore_assignments,
        )
        frame_instances.append(frame_instance)
        event.frame_instance_ids.append(fi_id)

        if ann.get("is_new_frame", True) and ann.get("frame_definition"):
            new_frames.append(FrameDefinitionSchema(
                frame_name=frame_name,
                lexical_units=[lexical_unit],
                frame_definition=ann.get("frame_definition", ""),
                core_fes=[
                    CoreFESchema(fe_name=a.fe_name, fe_definition="", semantic_type="")
                    for a in core_assignments
                ],
                noncore_fes=[
                    NonCoreFESchema(fe_name=a.fe_name, fe_definition="", semantic_type="")
                    for a in noncore_assignments
                ],
                is_from_framenet=False,
                usage_count=1,
            ))

    logger.debug(
        f"[operate] Chunk {chunk.chunk_id}: {len(events)} events, "
        f"{len(frame_instances)} FIs, {len(info_nodes)} info nodes, "
        f"{len(new_frames)} new frames (2-step)"
    )
    return events, frame_instances, info_nodes, new_frames


# ─────────────────────────────────────────────────────────────────────────────
# Call 3: Causal / Temporal Edge Extraction
# ─────────────────────────────────────────────────────────────────────────────

async def extract_causal_edges(
    chunk: ChunkSchema,
    events: list[EventSchema],
    llm_func: Callable[..., Awaitable[str]],
) -> list[CausalEdgeSchema]:
    """Extract causal/temporal edges between events in the same chunk (Call 3).

    Skipped when fewer than 2 events in chunk.
    """
    if len(events) < 2:
        return []

    event_lines = [
        f'  - event_id: "{ev.event_id}" | trigger: "{ev.trigger_lemma}" '
        f'| frame: "{ev.frame_name}" | span: "{ev.event_span}" '
        f'| desc: "{ev.event_description}"'
        for ev in events
    ]
    prompt = PROMPTS["causal_temporal"].format(
        chunk_text=chunk.text,
        event_list="\n".join(event_lines),
    )

    try:
        response = await _llm_with_retry(llm_func, prompt)
    except Exception as e:
        logger.warning(f"[operate] Causal extraction failed for chunk {chunk.chunk_id}: {e}")
        return []

    raw = _safe_json(response)
    if not isinstance(raw, list):
        logger.warning(
            f"[operate] Causal extraction returned non-list for chunk {chunk.chunk_id}; "
            f"raw head: {response[:120]!r}"
        )
        return []

    event_ids = {ev.event_id for ev in events}
    edges: list[CausalEdgeSchema] = []
    skipped_unknown = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source_event_id", "")).strip()
        tgt = str(item.get("target_event_id", "")).strip()
        rel = str(item.get("relation_type", "")).strip().upper()
        if src not in event_ids or tgt not in event_ids:
            skipped_unknown += 1
            continue
        if src == tgt:
            continue
        if rel not in ("CAUSES", "PRECEDES", "ENABLES"):
            continue
        edges.append(CausalEdgeSchema(
            edge_id=_gen_id("cau"),
            source_event_id=src,
            target_event_id=tgt,
            relation_type=rel,
            confidence=float(item.get("confidence", 0.8)),
            evidence_span=item.get("evidence_span", ""),
        ))

    if skipped_unknown:
        logger.debug(
            f"[operate] Chunk {chunk.chunk_id}: skipped {skipped_unknown} causal "
            f"edges with unknown event_ids"
        )
    logger.debug(
        f"[operate] Chunk {chunk.chunk_id}: {len(edges)} causal edges from "
        f"{len(raw)} candidates ({len(events)} events)"
    )
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Query Seed Extraction
# ─────────────────────────────────────────────────────────────────────────────

async def generate_hyde_passage(
    query: str,
    llm_func: Callable[..., Awaitable[str]],
) -> str:
    """Generate a hypothetical answer passage (HyDE) to bridge the query→content gap.

    For paraphrased / disguised questions the raw query embedding lands far from
    the actual narrative wording. A hypothetical excerpt — written in the plain
    voice of the novel — embeds much closer to the real chunks that hold the
    answer, dramatically improving seed recall. Returns "" on any failure so the
    caller can silently fall back to query-only seeding.
    """
    prompt = PROMPTS["hyde_passage"].format(query=query)
    try:
        passage = await _llm_with_retry(llm_func, prompt)
        return (passage or "").strip()
    except Exception as e:
        logger.warning(f"[operate] HyDE generation failed: {e}")
        return ""


async def process_query(
    query: str,
    llm_func: Callable[..., Awaitable[str]],
) -> dict:
    """Extract retrieval signals from a natural language query."""
    prompt = PROMPTS["query_processing"].format(query=query)
    try:
        response = await _llm_with_retry(llm_func, prompt)
        parsed = _safe_json(response)
        if isinstance(parsed, dict):
            return parsed
    except Exception as e:
        logger.warning(f"[operate] Query processing failed: {e}")
    return {
        "entity_hints": [],
        "event_hints": [],
        "frame_hints": "",
        "fe_focus": [],
        "temporal_hints": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Answer Generation
# ─────────────────────────────────────────────────────────────────────────────

async def generate_answer(
    query: str,
    structured_facts: str,
    text_passages: str,
    llm_func: Callable[..., Awaitable[str]],
) -> str:
    """Generate final answer using retrieved frame instances and text passages."""
    prompt = PROMPTS["answer_generation"].format(
        structured_facts=structured_facts,
        text_passages=text_passages,
        query=query,
    )
    try:
        return await _llm_with_retry(llm_func, prompt)
    except Exception as e:
        logger.error(f"[operate] Answer generation failed: {e}")
        return "Unable to generate answer due to an internal error."
