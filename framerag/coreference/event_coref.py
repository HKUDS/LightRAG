"""Event Coreference Resolution.

Pipeline:
  1. Blocking   — group events by frame_name (same frame = candidate co-referents)
  2. Embedding  — encode each event as: trigger_lemma + frame_name + canonical entity names
  3. Clustering — agglomerative (cosine) within each frame block
  4. LLM verify — borderline pairs in [0.75, 0.88]: check argument compatibility
  5. Merge      — assign canonical_event_id to duplicate events
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Callable, Awaitable

import numpy as np

from lightrag.utils import logger

from ..types import EventSchema, CanonicalEntitySchema
from ..prompt import PROMPTS

AUTO_SAME_THRESHOLD = 0.88
AUTO_DIFF_THRESHOLD = 0.75


class EventCoreferenceResolver:
    """Resolve event coreference using encoder + optional LLM argument-compatibility verify."""

    def __init__(
        self,
        embed_func: Callable[[list[str]], Awaitable[np.ndarray]],
        llm_func: Callable[..., Awaitable[str]],
        cluster_threshold: float = AUTO_SAME_THRESHOLD,
    ):
        self._embed = embed_func
        self._llm = llm_func
        self._threshold = cluster_threshold

    async def resolve(
        self,
        events: list[EventSchema],
        canonical_entities: list[CanonicalEntitySchema],
    ) -> dict[str, str]:
        """Resolve event coreference.

        Returns a dict mapping event_id → canonical_event_id.
        Events that are unique get their own event_id as canonical.
        Events that corefer share the same canonical_event_id.
        """
        if not events:
            return {}

        logger.info(f"[EventCoref] Resolving {len(events)} events")

        # Build lookup: mention_id → canonical entity name (for event embedding)
        canon_name_by_mention: dict[str, str] = {}
        for ce in canonical_entities:
            for mid in ce.mention_ids:
                canon_name_by_mention[mid] = ce.canonical_name

        # Step 1: Build event text representations for embedding
        event_texts = self._build_event_texts(events, canon_name_by_mention)

        # Step 2: Embed all events
        all_embeddings = await self._embed(event_texts)
        emb_by_id = {ev.event_id: all_embeddings[i] for i, ev in enumerate(events)}

        # Step 3: Block by frame_name
        blocks = self._build_blocks(events)
        logger.info(f"[EventCoref] {len(blocks)} frame blocks for event coref")

        # Step 4: Cluster within each block
        uf = _UnionFind([ev.event_id for ev in events])
        event_by_id = {ev.event_id: ev for ev in events}

        for frame_name, block_ids in blocks.items():
            if len(block_ids) < 2:
                continue
            embs = np.stack([emb_by_id[eid] for eid in block_ids])
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
            normed = embs / norms
            sim_matrix = normed @ normed.T

            for i in range(len(block_ids)):
                for j in range(i + 1, len(block_ids)):
                    sim = float(sim_matrix[i, j])
                    eid_i, eid_j = block_ids[i], block_ids[j]

                    if sim >= AUTO_SAME_THRESHOLD:
                        uf.union(eid_i, eid_j)
                    elif sim >= AUTO_DIFF_THRESHOLD:
                        ev_i = event_by_id[eid_i]
                        ev_j = event_by_id[eid_j]
                        same = await self._llm_verify_pair(ev_i, ev_j, canon_name_by_mention)
                        if same:
                            uf.union(eid_i, eid_j)

        # Step 5: Assign canonical_event_id
        clusters = uf.get_clusters()
        event_to_canonical: dict[str, str] = {}
        for root, member_ids in clusters.items():
            if len(member_ids) == 1:
                event_to_canonical[member_ids[0]] = member_ids[0]
            else:
                canonical_event_id = f"canon_ev_{uuid.uuid4().hex[:12]}"
                for eid in member_ids:
                    event_to_canonical[eid] = canonical_event_id

        logger.info(
            f"[EventCoref] {len(clusters)} canonical events from {len(events)} events"
        )
        return event_to_canonical

    # ──────────────────────────────────────────────────────────────────────────

    def _build_event_texts(
        self,
        events: list[EventSchema],
        canon_name_by_mention: dict[str, str],
    ) -> list[str]:
        """Build embedding text for each event: trigger + frame + participant canonical names."""
        texts = []
        for ev in events:
            participant_names = []
            for mid in ev.participant_mention_ids:
                name = canon_name_by_mention.get(mid, mid)
                participant_names.append(name)
            participant_str = ", ".join(participant_names) if participant_names else "none"
            text = (
                f"{ev.trigger_lemma} [{ev.frame_name}]: {ev.event_description}"
                f" | participants: {participant_str}"
            )
            texts.append(text)
        return texts

    def _build_blocks(self, events: list[EventSchema]) -> dict[str, list[str]]:
        """Group events by frame_name — same frame = co-reference candidates."""
        blocks: dict[str, list[str]] = defaultdict(list)
        for ev in events:
            blocks[ev.frame_name].append(ev.event_id)
        return {k: list(set(v)) for k, v in blocks.items() if len(v) >= 2}

    async def _llm_verify_pair(
        self,
        ev_a: EventSchema,
        ev_b: EventSchema,
        canon_name_by_mention: dict[str, str],
    ) -> bool:
        """Ask LLM whether two events with same frame describe the same real-world occurrence."""
        def _participant_str(ev: EventSchema) -> str:
            names = [canon_name_by_mention.get(mid, mid) for mid in ev.participant_mention_ids]
            return ", ".join(names) if names else "none"

        prompt = PROMPTS["event_coref_verify"].format(
            trigger_a=ev_a.trigger_lemma,
            frame_a=ev_a.frame_name,
            desc_a=ev_a.event_description,
            participants_a=_participant_str(ev_a),
            trigger_b=ev_b.trigger_lemma,
            frame_b=ev_b.frame_name,
            desc_b=ev_b.event_description,
            participants_b=_participant_str(ev_b),
        )
        try:
            response = await self._llm(prompt)
            return response.strip().upper().startswith("SAME")
        except Exception as e:
            logger.warning(f"[EventCoref] LLM verify failed: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, ids: list[str]):
        self._parent = {i: i for i in ids}
        self._rank = {i: 0 for i in ids}

    def find(self, x: str) -> str:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def get_clusters(self) -> dict[str, list[str]]:
        clusters: dict[str, list[str]] = defaultdict(list)
        for node in self._parent:
            root = self.find(node)
            clusters[root].append(node)
        return dict(clusters)
