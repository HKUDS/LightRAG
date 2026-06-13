"""Core extraction pipeline for FrameRAG.

Three extraction calls per chunk:
  Call 1: Entity extraction (extended from LightRAG)
  Call 2: Event detection + Frame induction + Role assignment (combined, with Frame DB hints)
  Call 3: Causal/temporal edge extraction (optional)
"""
from __future__ import annotations

import json
import uuid
from typing import Callable, Awaitable

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
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            idx = text.find(start_char)
            if idx != -1:
                ridx = text.rfind(end_char)
                if ridx > idx:
                    try:
                        return json.loads(text[idx : ridx + 1])
                    except json.JSONDecodeError:
                        pass
        return None


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


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
        response = await llm_func(prompt)
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
# Call 2: Event Detection + Frame Induction + Role Assignment
# ─────────────────────────────────────────────────────────────────────────────

async def extract_events_frames_roles(
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
    """Combined event detection + frame induction + role assignment (Call 2).

    Returns:
        events         : one EventSchema per event trigger
        frame_instances: one FrameInstanceSchema per event's frame instantiation
        info_nodes     : non-entity FE fillers (dates, prices, etc.)
        new_frames     : FrameDefinitionSchema for frames to save to Frame DB
    """
    hints = await frame_db.get_hints_for_chunk(chunk.text)
    hints_text = frame_db.format_hints_for_prompt(hints)

    entity_lines = []
    for m in entity_mentions:
        aliases_str = f" (also: {', '.join(m.aliases)})" if m.aliases else ""
        entity_lines.append(f"  - {m.name} [{m.entity_type}]{aliases_str}: {m.description}")
    entity_list_text = "\n".join(entity_lines) if entity_lines else "  (none)"

    prompt = PROMPTS["event_frame_role"].format(
        frame_db_hints=hints_text,
        entity_list=entity_list_text,
        chunk_text=chunk.text,
    )

    try:
        response = await llm_func(prompt)
    except Exception as e:
        logger.error(f"[operate] Event/frame/role failed for chunk {chunk.chunk_id}: {e}")
        return [], [], [], []

    raw = _safe_json(response)
    if not isinstance(raw, list):
        logger.warning(f"[operate] Event/frame/role returned non-list for chunk {chunk.chunk_id}")
        return [], [], [], []

    # Build entity mention lookup by name (and aliases) for linking
    mention_by_name: dict[str, EntityMentionSchema] = {}
    for m in entity_mentions:
        mention_by_name[m.name.lower()] = m
        for alias in (m.aliases or []):
            mention_by_name[alias.lower()] = m

    events: list[EventSchema] = []
    frame_instances: list[FrameInstanceSchema] = []
    info_nodes: list[InfoNodeSchema] = []
    new_frames: list[FrameDefinitionSchema] = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        trigger_lemma = item.get("trigger_lemma", item.get("trigger", "")).strip()
        if not trigger_lemma:
            continue

        frame_data = item.get("frame", {})
        frame_name = frame_data.get("frame_name", "").strip()
        if not frame_name:
            continue

        lexical_unit = frame_data.get("lexical_unit", f"{trigger_lemma}.v")

        # Resolve participant entity mentions
        participant_mention_ids: list[str] = []
        for pname in item.get("participant_entity_names", []):
            matched = mention_by_name.get(pname.lower())
            if matched:
                participant_mention_ids.append(matched.mention_id)

        event_id = _gen_id("ev")
        event = EventSchema(
            event_id=event_id,
            chunk_id=chunk.chunk_id,
            trigger=item.get("trigger", trigger_lemma),
            trigger_lemma=trigger_lemma,
            trigger_pos=item.get("trigger_pos", "VERB"),
            event_span=item.get("event_span", ""),
            event_description=item.get("event_description", ""),
            frame_name=frame_name,
            participant_mention_ids=participant_mention_ids,
        )
        events.append(event)

        # Build FE assignments
        role_data = item.get("role_assignments", {})
        core_assignments: list[FEAssignment] = []
        noncore_assignments: list[FEAssignment] = []

        for fe_item in role_data.get("core", []):
            fe_name = fe_item.get("fe_name", "").strip()
            filler_text = fe_item.get("filler_text")
            filler_type_raw = fe_item.get("filler_type", "ENTITY")
            is_missing = fe_item.get("is_missing", filler_text is None)

            if is_missing or not filler_text:
                core_assignments.append(FEAssignment(
                    fe_name=fe_name, filler_id=None, filler_type="MISSING",
                    filler_text="", is_core=True,
                ))
                continue

            if filler_type_raw == "ENTITY":
                matched = mention_by_name.get(str(filler_text).lower())
                core_assignments.append(FEAssignment(
                    fe_name=fe_name,
                    filler_id=matched.mention_id if matched else None,
                    filler_type="ENTITY" if matched else "MISSING",
                    filler_text=str(filler_text),
                    is_core=True,
                ))
            else:
                info_id = _gen_id("info")
                info_nodes.append(InfoNodeSchema(
                    info_id=info_id,
                    value=str(filler_text),
                    info_type=filler_type_raw,
                ))
                core_assignments.append(FEAssignment(
                    fe_name=fe_name, filler_id=info_id, filler_type="VALUE",
                    filler_text=str(filler_text), is_core=True,
                ))

        for fe_item in role_data.get("noncore", []):
            fe_name = fe_item.get("fe_name", "").strip()
            filler_text = fe_item.get("filler_text")
            filler_type_raw = fe_item.get("filler_type", "VALUE")
            is_missing = fe_item.get("is_missing", filler_text is None)

            if is_missing or not filler_text:
                noncore_assignments.append(FEAssignment(
                    fe_name=fe_name, filler_id=None, filler_type="MISSING",
                    filler_text="", is_core=False,
                ))
                continue

            if filler_type_raw == "ENTITY":
                matched = mention_by_name.get(str(filler_text).lower())
                noncore_assignments.append(FEAssignment(
                    fe_name=fe_name,
                    filler_id=matched.mention_id if matched else None,
                    filler_type="ENTITY" if matched else "MISSING",
                    filler_text=str(filler_text),
                    is_core=False,
                ))
            else:
                info_id = _gen_id("info")
                info_nodes.append(InfoNodeSchema(
                    info_id=info_id,
                    value=str(filler_text),
                    info_type=filler_type_raw,
                ))
                noncore_assignments.append(FEAssignment(
                    fe_name=fe_name, filler_id=info_id, filler_type="VALUE",
                    filler_text=str(filler_text), is_core=False,
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

        # Update event to track its frame instance
        event.frame_instance_ids.append(fi_id)

        # Collect new frame definitions for Frame DB
        if frame_data.get("is_new_frame", False):
            core_fes = [
                CoreFESchema(
                    fe_name=fe.get("fe_name", ""),
                    fe_definition=fe.get("fe_definition", ""),
                    semantic_type=fe.get("semantic_type", ""),
                )
                for fe in frame_data.get("core_fes", [])
            ]
            noncore_fes = [
                NonCoreFESchema(
                    fe_name=fe.get("fe_name", ""),
                    fe_definition=fe.get("fe_definition", ""),
                    semantic_type=fe.get("semantic_type", ""),
                )
                for fe in frame_data.get("noncore_fes", [])
            ]
            new_frames.append(FrameDefinitionSchema(
                frame_name=frame_name,
                lexical_units=[lexical_unit],
                frame_definition=frame_data.get("frame_definition", ""),
                core_fes=core_fes,
                noncore_fes=noncore_fes,
                is_from_framenet=False,
                usage_count=1,
            ))

    logger.debug(
        f"[operate] Chunk {chunk.chunk_id}: {len(events)} events, "
        f"{len(frame_instances)} FIs, {len(info_nodes)} info nodes, "
        f"{len(new_frames)} new frames"
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
        f'| frame: "{ev.frame_name}" | desc: "{ev.event_description}"'
        for ev in events
    ]
    prompt = PROMPTS["causal_temporal"].format(event_list="\n".join(event_lines))

    try:
        response = await llm_func(prompt)
    except Exception as e:
        logger.warning(f"[operate] Causal extraction failed for chunk {chunk.chunk_id}: {e}")
        return []

    raw = _safe_json(response)
    if not isinstance(raw, list):
        return []

    event_ids = {ev.event_id for ev in events}
    edges: list[CausalEdgeSchema] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        src = item.get("source_event_id", "").strip()
        tgt = item.get("target_event_id", "").strip()
        rel = item.get("relation_type", "").strip()
        if src not in event_ids or tgt not in event_ids:
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

    logger.debug(f"[operate] Chunk {chunk.chunk_id}: {len(edges)} causal edges")
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# Query Seed Extraction
# ─────────────────────────────────────────────────────────────────────────────

async def process_query(
    query: str,
    llm_func: Callable[..., Awaitable[str]],
) -> dict:
    """Extract retrieval signals from a natural language query."""
    prompt = PROMPTS["query_processing"].format(query=query)
    try:
        response = await llm_func(prompt)
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
        return await llm_func(prompt)
    except Exception as e:
        logger.error(f"[operate] Answer generation failed: {e}")
        return "Unable to generate answer due to an internal error."
