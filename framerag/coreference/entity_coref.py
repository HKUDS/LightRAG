"""Entity Coreference Resolution.

Pipeline:
  1. Blocking   — group mentions by normalized name / first token
  2. Clustering — agglomerative clustering on (name + description) embeddings
  3. LLM verify — borderline pairs in [0.75, 0.88]
  4. Merge      — create CanonicalEntityNodes and update MENTION_OF edges

Only entity.name + entity.description are used for embedding — role is NOT used.
"""
from __future__ import annotations

import re
import uuid
from typing import Callable, Awaitable, Optional
from collections import defaultdict

import numpy as np

from lightrag.utils import logger

from ..types import EntityMentionSchema, CanonicalEntitySchema
from ..prompt import PROMPTS

# Similarity thresholds
AUTO_SAME_THRESHOLD = 0.88
AUTO_DIFF_THRESHOLD = 0.75
LLM_VERIFY_RANGE = (AUTO_DIFF_THRESHOLD, AUTO_SAME_THRESHOLD)


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().strip())


def _first_token(name: str) -> str:
    return _normalize_name(name).split()[0] if name.strip() else ""


class EntityCoreferenceResolver:
    """Resolve entity mention coreference using encoder + optional LLM verify."""

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
        self, mentions: list[EntityMentionSchema]
    ) -> list[CanonicalEntitySchema]:
        """Run full coreference resolution on a list of entity mentions.

        Returns a list of CanonicalEntitySchema.
        Each mention's canonical_id is updated in-place.
        """
        if not mentions:
            return []

        logger.info(f"[EntityCoref] Resolving {len(mentions)} entity mentions")

        # Step 1: Blocking
        blocks = self._build_blocks(mentions)
        logger.info(f"[EntityCoref] {len(blocks)} blocks created")

        # Step 2: Embed all mentions once (name + description)
        all_texts = [
            f"{m.name} [SEP] {m.description}" for m in mentions
        ]
        all_embeddings = await self._embed(all_texts)  # [n_mentions, dim]

        mention_by_id = {m.mention_id: m for m in mentions}
        emb_by_id = {m.mention_id: all_embeddings[i] for i, m in enumerate(mentions)}

        # Step 3: Cluster within each block
        union_find = _UnionFind([m.mention_id for m in mentions])

        for block_ids in blocks.values():
            if len(block_ids) < 2:
                continue
            embs = np.stack([emb_by_id[mid] for mid in block_ids])
            await self._cluster_block(block_ids, embs, union_find)

        # Step 4: Build canonical entities from clusters
        clusters = union_find.get_clusters()
        canonicals = []
        for root, member_ids in clusters.items():
            canonical = await self._build_canonical(
                member_ids, mention_by_id, emb_by_id
            )
            # Update canonical_id on each mention object
            for mid in member_ids:
                mention_by_id[mid].canonical_id = canonical.canonical_id
            canonicals.append(canonical)

        logger.info(f"[EntityCoref] {len(canonicals)} canonical entities created")
        return canonicals

    # ──────────────────────────────────────────────────────────────────────────

    def _build_blocks(
        self, mentions: list[EntityMentionSchema]
    ) -> dict[str, list[str]]:
        """Group mentions into blocks by normalized name and first token."""
        blocks: dict[str, list[str]] = defaultdict(list)
        for m in mentions:
            norm = _normalize_name(m.name)
            first = _first_token(m.name)
            blocks[norm].append(m.mention_id)
            if first and first != norm:
                blocks[first].append(m.mention_id)
            for alias in (m.aliases or []):
                a_norm = _normalize_name(alias)
                blocks[a_norm].append(m.mention_id)
        # Deduplicate within each block
        return {k: list(set(v)) for k, v in blocks.items() if v}

    async def _cluster_block(
        self,
        block_ids: list[str],
        embeddings: np.ndarray,
        union_find: "_UnionFind",
    ) -> None:
        """Agglomerative single-pass clustering within a block."""
        n = len(block_ids)
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        normed = embeddings / norms
        sim_matrix = normed @ normed.T  # [n, n]

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                mid_i, mid_j = block_ids[i], block_ids[j]

                if sim >= AUTO_SAME_THRESHOLD:
                    union_find.union(mid_i, mid_j)
                elif sim >= AUTO_DIFF_THRESHOLD:
                    # LLM verify borderline case
                    same = await self._llm_verify(mid_i, mid_j, union_find)
                    if same:
                        union_find.union(mid_i, mid_j)

    async def _llm_verify(
        self,
        mid_a: str,
        mid_b: str,
        uf: "_UnionFind",
    ) -> bool:
        """Placeholder — full implementation needs mention data passed in."""
        # In practice, we'd look up mention data here.
        # This is wired up in resolve() via closure or by restructuring.
        return False

    async def resolve_with_llm_verify(
        self,
        mentions: list[EntityMentionSchema],
    ) -> list[CanonicalEntitySchema]:
        """Full resolve with LLM verification for borderline pairs."""
        if not mentions:
            return []

        mention_by_id = {m.mention_id: m for m in mentions}
        all_texts = [f"{m.name} [SEP] {m.description}" for m in mentions]
        all_embeddings = await self._embed(all_texts)
        emb_by_id = {m.mention_id: all_embeddings[i] for i, m in enumerate(mentions)}

        blocks = self._build_blocks(mentions)
        union_find = _UnionFind([m.mention_id for m in mentions])

        for block_ids in blocks.values():
            if len(block_ids) < 2:
                continue
            embs = np.stack([emb_by_id[mid] for mid in block_ids])
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
            normed = embs / norms
            sim_matrix = normed @ normed.T

            for i in range(len(block_ids)):
                for j in range(i + 1, len(block_ids)):
                    sim = float(sim_matrix[i, j])
                    mid_i, mid_j = block_ids[i], block_ids[j]

                    if sim >= AUTO_SAME_THRESHOLD:
                        union_find.union(mid_i, mid_j)
                    elif sim >= AUTO_DIFF_THRESHOLD:
                        m_i = mention_by_id[mid_i]
                        m_j = mention_by_id[mid_j]
                        same = await self._llm_verify_pair(m_i, m_j)
                        if same:
                            union_find.union(mid_i, mid_j)

        clusters = union_find.get_clusters()
        canonicals = []
        for _, member_ids in clusters.items():
            canonical = await self._build_canonical(member_ids, mention_by_id, emb_by_id)
            for mid in member_ids:
                mention_by_id[mid].canonical_id = canonical.canonical_id
            canonicals.append(canonical)

        logger.info(f"[EntityCoref] {len(canonicals)} canonical entities from {len(mentions)} mentions")
        return canonicals

    async def _llm_verify_pair(
        self,
        m_a: EntityMentionSchema,
        m_b: EntityMentionSchema,
    ) -> bool:
        prompt = PROMPTS["entity_coref_verify"].format(
            name_a=m_a.name, desc_a=m_a.description,
            name_b=m_b.name, desc_b=m_b.description,
        )
        try:
            response = await self._llm(prompt)
            return response.strip().upper().startswith("SAME")
        except Exception as e:
            logger.warning(f"[EntityCoref] LLM verify failed: {e}")
            return False

    async def _build_canonical(
        self,
        member_ids: list[str],
        mention_by_id: dict[str, EntityMentionSchema],
        emb_by_id: dict[str, np.ndarray],
    ) -> CanonicalEntitySchema:
        members = [mention_by_id[mid] for mid in member_ids]

        # Pick most frequent name as canonical name
        name_counts: dict[str, int] = defaultdict(int)
        for m in members:
            name_counts[m.name] += 1
        canonical_name = max(name_counts, key=lambda k: name_counts[k])

        # Pick entity_type by majority vote
        type_counts: dict[str, int] = defaultdict(int)
        for m in members:
            type_counts[m.entity_type] += 1
        canonical_type = max(type_counts, key=lambda k: type_counts[k])

        descriptions = list({m.description for m in members if m.description})

        # Canonical embedding: average of member embeddings
        embs = np.stack([emb_by_id[mid] for mid in member_ids])
        canonical_emb = embs.mean(axis=0)

        canonical_id = f"canon_{uuid.uuid4().hex[:12]}"
        return CanonicalEntitySchema(
            canonical_id=canonical_id,
            canonical_name=canonical_name,
            entity_type=canonical_type,
            descriptions=descriptions,
            mention_ids=member_ids,
            embedding=canonical_emb.tolist(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Union-Find for cluster management
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
