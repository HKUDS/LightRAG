"""Entity Resolution - 3-Layer Approach

Layer 1: Case normalization (exact match)
Layer 2: Fuzzy string matching (>85% = typos)
Layer 3: Vector similarity + LLM verification (semantic matches)

Uses the same LLM that LightRAG is configured with.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np

from lightrag.utils import logger

from .config import DEFAULT_CONFIG, EntityResolutionConfig


@dataclass
class ResolutionResult:
    """Result of entity resolution attempt."""

    action: str  # "match" | "new"
    matched_entity: str | None
    confidence: float
    method: str  # "exact" | "fuzzy" | "llm" | "none" | "disabled"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def fuzzy_similarity(a: str, b: str) -> float:
    """Calculate fuzzy string similarity (0-1)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def find_vector_candidates(
    query_embedding: list[float],
    existing_entities: list[tuple[str, list[float]]],
    threshold: float,
) -> list[tuple[str, float]]:
    """Find entities with vector similarity above threshold."""
    candidates = []
    for name, embedding in existing_entities:
        sim = cosine_similarity(query_embedding, embedding)
        if sim >= threshold:
            candidates.append((name, sim))
    # Sort by similarity descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


async def llm_verify(
    term_a: str,
    term_b: str,
    llm_fn: Callable[[str], Awaitable[str]],
    prompt_template: str,
) -> bool:
    """Ask LLM if two terms refer to the same entity.

    Uses strict parsing with exact token matching only. Accepted responses:
    - Positive: "YES", "TRUE", "SAME", "MATCH"
    - Negative: "NO", "FALSE", "DIFFERENT", "NOT SAME"
    Any other response defaults to False to avoid false positive merges.
    """
    prompt = prompt_template.format(term_a=term_a, term_b=term_b)
    response = await llm_fn(prompt)

    # Normalize response: strip whitespace, take first line only
    normalized = response.strip().split('\n')[0].strip().upper()

    # Remove common trailing punctuation
    normalized = normalized.rstrip('.!,')

    # Only accept exact tokens (no prefix/substring matching)
    if normalized in ('YES', 'TRUE', 'SAME', 'MATCH'):
        return True
    if normalized in ('NO', 'FALSE', 'DIFFERENT', 'NOT SAME'):
        return False

    # Default to False for ambiguous responses (safer than false positive)
    return False


async def resolve_entity(
    entity_name: str,
    existing_entities: list[tuple[str, list[float]]],
    embed_fn: Callable[[str], Awaitable[list[float]]],
    llm_fn: Callable[[str], Awaitable[str]],
    config: EntityResolutionConfig = DEFAULT_CONFIG,
) -> ResolutionResult:
    """Resolve an entity against existing entities using 3-layer approach.

    Args:
        entity_name: The new entity name to resolve
        existing_entities: List of (name, embedding) tuples for existing entities
        embed_fn: Async function to get embedding for a string (same as LightRAG uses)
        llm_fn: Async function to query LLM (same as LightRAG uses)
        config: Resolution configuration

    Returns:
        ResolutionResult with action ("match" or "new"), matched entity,
        confidence, and method used.
    """
    if not config.enabled:
        return ResolutionResult('new', None, 0.0, 'disabled')

    if not existing_entities:
        return ResolutionResult('new', None, 0.0, 'none')

    normalized = entity_name.lower().strip()

    # Layer 1: Case-insensitive exact match
    for name, _ in existing_entities:
        if name.lower().strip() == normalized:
            return ResolutionResult('match', name, 1.0, 'exact')

    # Layer 2: Fuzzy string matching (catches typos)
    best_fuzzy_match = None
    best_fuzzy_score = 0.0

    for name, _ in existing_entities:
        similarity = fuzzy_similarity(entity_name, name)
        if similarity > best_fuzzy_score:
            best_fuzzy_score = similarity
            best_fuzzy_match = name

    if best_fuzzy_score >= config.fuzzy_threshold:
        return ResolutionResult('match', best_fuzzy_match, best_fuzzy_score, 'fuzzy')

    # Layer 3: Vector similarity + LLM verification
    embedding = await embed_fn(entity_name)
    candidates = find_vector_candidates(
        embedding,
        existing_entities,
        config.vector_threshold,
    )

    # Verify top candidates with LLM
    for candidate_name, similarity in candidates[: config.max_candidates]:
        is_same = await llm_verify(
            entity_name,
            candidate_name,
            llm_fn,
            config.llm_prompt_template,
        )
        if is_same:
            return ResolutionResult('match', candidate_name, similarity, 'llm')

    # No match found - this is a new entity
    return ResolutionResult('new', None, 0.0, 'none')


async def resolve_entity_with_vdb(
    entity_name: str,
    entity_vdb,  # BaseVectorStorage - imported dynamically to avoid circular imports
    llm_fn: Callable[[str], Awaitable[str]],
    config: EntityResolutionConfig = DEFAULT_CONFIG,
) -> ResolutionResult:
    """Resolve an entity using VDB for similarity search.

    This is the production integration that uses LightRAG's vector database
    directly instead of requiring pre-computed embeddings.

    Args:
        entity_name: The new entity name to resolve
        entity_vdb: LightRAG's entity vector database (BaseVectorStorage)
        llm_fn: Async function to query LLM (same as LightRAG uses)
        config: Resolution configuration

    Returns:
        ResolutionResult with action ("match" or "new"), matched entity,
        confidence, and method used.
    """
    if not config.enabled:
        return ResolutionResult('new', None, 0.0, 'disabled')

    if entity_vdb is None:
        return ResolutionResult('new', None, 0.0, 'none')

    normalized = entity_name.lower().strip()

    # Query VDB for similar entities - cast wide net, LLM will verify
    # top_k is doubled to have enough candidates after filtering
    try:
        candidates = await entity_vdb.query(entity_name, top_k=config.max_candidates * 3)
    except Exception as e:
        # Log and skip resolution if VDB query fails
        logger.debug(f"VDB query failed for '{entity_name}': {e}")
        return ResolutionResult('new', None, 0.0, 'none')

    if not candidates:
        return ResolutionResult('new', None, 0.0, 'none')

    # Layer 1: Case-insensitive exact match among candidates
    for candidate in candidates:
        candidate_name = candidate.get('entity_name')
        if candidate_name and candidate_name.lower().strip() == normalized:
            return ResolutionResult('match', candidate_name, 1.0, 'exact')

    # Layer 2: Fuzzy string matching (catches typos)
    best_fuzzy_match = None
    best_fuzzy_score = 0.0

    for candidate in candidates:
        candidate_name = candidate.get('entity_name')
        if not candidate_name:
            continue
        similarity = fuzzy_similarity(entity_name, candidate_name)
        if similarity > best_fuzzy_score:
            best_fuzzy_score = similarity
            best_fuzzy_match = candidate_name

    if best_fuzzy_score >= config.fuzzy_threshold:
        return ResolutionResult('match', best_fuzzy_match, best_fuzzy_score, 'fuzzy')

    # Layer 3: LLM verification on top candidates
    verified_count = 0
    for candidate in candidates:
        if verified_count >= config.max_candidates:
            break
        candidate_name = candidate.get('entity_name')
        if not candidate_name:
            continue

        is_same = await llm_verify(
            entity_name,
            candidate_name,
            llm_fn,
            config.llm_prompt_template,
        )
        verified_count += 1

        if is_same:
            # Use distance from VDB if available (converted to similarity)
            similarity = 0.7  # Default confidence for LLM match
            return ResolutionResult('match', candidate_name, similarity, 'llm')

    # No match found - this is a new entity
    return ResolutionResult('new', None, 0.0, 'none')


# --- Alias Cache Functions (PostgreSQL) ---


async def get_cached_alias(
    alias: str,
    db,  # PostgresDB instance
    workspace: str,
) -> tuple[str, str, float] | None:
    """Check if alias is already resolved in cache.

    Args:
        alias: The entity name to look up
        db: PostgresDB instance with query method
        workspace: Workspace for isolation

    Returns:
        Tuple of (canonical_entity, method, confidence) if found, None otherwise
    """
    import logging

    from lightrag.kg.postgres_impl import SQL_TEMPLATES

    logger = logging.getLogger(__name__)
    normalized_alias = alias.lower().strip()

    sql = SQL_TEMPLATES['get_alias']
    try:
        result = await db.query(sql, params=[workspace, normalized_alias])
        if result:
            return (
                result['canonical_entity'],
                result['method'],
                result['confidence'],
            )
    except Exception as e:
        logger.debug(f'Alias cache lookup error: {e}')
    return None


async def store_alias(
    alias: str,
    canonical: str,
    method: str,
    confidence: float,
    db,  # PostgresDB instance
    workspace: str,
) -> None:
    """Store a resolution in the alias cache.

    Args:
        alias: The variant name (e.g., "FDA")
        canonical: The resolved canonical name (e.g., "US Food and Drug Administration")
        method: How it was resolved ('exact', 'fuzzy', 'llm', 'manual')
        confidence: Resolution confidence (0-1)
        db: PostgresDB instance with execute method
        workspace: Workspace for isolation
    """
    import logging
    from datetime import datetime, timezone

    from lightrag.kg.postgres_impl import SQL_TEMPLATES

    logger = logging.getLogger(__name__)
    normalized_alias = alias.lower().strip()

    # Don't store self-referential aliases (e.g., "FDA" → "FDA")
    if normalized_alias == canonical.lower().strip():
        return

    sql = SQL_TEMPLATES['upsert_alias']
    try:
        await db.execute(
            sql,
            data={
                'workspace': workspace,
                'alias': normalized_alias,
                'canonical_entity': canonical,
                'method': method,
                'confidence': confidence,
                'create_time': datetime.now(timezone.utc).replace(tzinfo=None),
            },
        )
    except Exception as e:
        logger.debug(f'Alias cache store error: {e}')
