"""
Unit tests for Entity Resolution

Tests the 3-layer approach with mock embed_fn and llm_fn.
No database or external services required.
"""

import pytest

from lightrag.entity_resolution import (
    EntityResolutionConfig,
    resolve_entity,
)

# Mock embeddings - pre-computed for test entities
# These simulate what an embedding model would return
MOCK_EMBEDDINGS = {
    # FDA and full name have ~0.67 similarity (based on real test)
    'fda': [0.1, 0.2, 0.3, 0.4, 0.5],
    'us food and drug administration': [0.15, 0.25, 0.28, 0.38, 0.52],
    # Dupixent and dupilumab have ~0.63 similarity
    'dupixent': [0.5, 0.6, 0.7, 0.8, 0.9],
    'dupilumab': [0.48, 0.58, 0.72, 0.78, 0.88],
    # Celebrex and Cerebyx are different (low similarity)
    'celebrex': [0.9, 0.1, 0.2, 0.3, 0.4],
    'cerebyx': [0.1, 0.9, 0.8, 0.7, 0.6],
    # Default for unknown entities
    'default': [0.0, 0.0, 0.0, 0.0, 0.0],
}

# Mock LLM responses
MOCK_LLM_RESPONSES = {
    ('fda', 'us food and drug administration'): 'YES',
    ('us food and drug administration', 'fda'): 'YES',
    ('dupixent', 'dupilumab'): 'YES',
    ('dupilumab', 'dupixent'): 'YES',
    ('heart attack', 'myocardial infarction'): 'YES',
    ('celebrex', 'cerebyx'): 'NO',
    ('metformin', 'metoprolol'): 'NO',
}


async def mock_embed_fn(text: str) -> list[float]:
    """Mock embedding function."""
    key = text.lower().strip()
    return MOCK_EMBEDDINGS.get(key, MOCK_EMBEDDINGS['default'])


async def mock_llm_fn(prompt: str) -> str:
    """Mock LLM function that parses the prompt and returns YES/NO."""
    # Extract term_a and term_b from the prompt
    lines = prompt.strip().split('\n')
    term_a = None
    term_b = None
    for line in lines:
        if line.startswith('Term A:'):
            term_a = line.replace('Term A:', '').strip().lower()
        elif line.startswith('Term B:'):
            term_b = line.replace('Term B:', '').strip().lower()

    if term_a and term_b:
        # Check both orderings
        response = MOCK_LLM_RESPONSES.get((term_a, term_b))
        if response is None:
            response = MOCK_LLM_RESPONSES.get((term_b, term_a), 'NO')
        return response
    return 'NO'


# Test fixtures
@pytest.fixture
def existing_entities():
    """Existing entities in the knowledge graph."""
    return [
        (
            'US Food and Drug Administration',
            MOCK_EMBEDDINGS['us food and drug administration'],
        ),
        ('Dupixent', MOCK_EMBEDDINGS['dupixent']),
        ('Celebrex', MOCK_EMBEDDINGS['celebrex']),
    ]


@pytest.fixture
def config():
    """Default resolution config."""
    return EntityResolutionConfig()


# Layer 1: Case normalization tests
class TestCaseNormalization:
    @pytest.mark.asyncio
    async def test_exact_match_same_case(self, existing_entities, config):
        """Exact match with same case."""
        result = await resolve_entity(
            'Dupixent',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'Dupixent'
        assert result.method == 'exact'
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_different_case(self, existing_entities, config):
        """DUPIXENT should match Dupixent via case normalization."""
        result = await resolve_entity(
            'DUPIXENT',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'Dupixent'
        assert result.method == 'exact'

    @pytest.mark.asyncio
    async def test_exact_match_lowercase(self, existing_entities, config):
        """dupixent should match Dupixent."""
        result = await resolve_entity(
            'dupixent',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.method == 'exact'


# Layer 2: Fuzzy matching tests
class TestFuzzyMatching:
    @pytest.mark.asyncio
    async def test_fuzzy_match_typo(self, existing_entities, config):
        """Dupixant (typo) should match Dupixent via fuzzy matching (88%)."""
        result = await resolve_entity(
            'Dupixant',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'Dupixent'
        assert result.method == 'fuzzy'
        assert result.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_fuzzy_rejects_below_threshold(self, existing_entities, config):
        """Celebrex vs Cerebyx is 67% - should NOT fuzzy match."""
        # Add Cerebyx as the query (Celebrex exists)
        result = await resolve_entity(
            'Cerebyx',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        # Should not be fuzzy match (67% < 85%)
        assert result.method != 'fuzzy' or result.action == 'new'


# Layer 3: LLM verification tests
class TestLLMVerification:
    @pytest.mark.asyncio
    async def test_llm_matches_acronym(self, existing_entities, config):
        """FDA should match US Food and Drug Administration via LLM."""
        result = await resolve_entity(
            'FDA',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'US Food and Drug Administration'
        assert result.method == 'llm'

    @pytest.mark.asyncio
    async def test_llm_matches_brand_generic(self, config):
        """Dupixent should match dupilumab via LLM."""
        existing = [
            ('dupilumab', MOCK_EMBEDDINGS['dupilumab']),
        ]
        result = await resolve_entity(
            'Dupixent',
            existing,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'match'
        assert result.matched_entity == 'dupilumab'
        assert result.method == 'llm'


# Edge cases
class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_existing_entities(self, config):
        """New entity when no existing entities."""
        result = await resolve_entity(
            'NewEntity',
            [],
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'new'

    @pytest.mark.asyncio
    async def test_disabled_resolution(self, existing_entities):
        """Resolution disabled returns new."""
        config = EntityResolutionConfig(enabled=False)
        result = await resolve_entity(
            'Dupixent',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'new'
        assert result.method == 'disabled'

    @pytest.mark.asyncio
    async def test_genuinely_new_entity(self, existing_entities, config):
        """Completely new entity should return 'new'."""
        result = await resolve_entity(
            'CompletelyNewDrug',
            existing_entities,
            mock_embed_fn,
            mock_llm_fn,
            config,
        )
        assert result.action == 'new'
        assert result.method == 'none'
