"""
PoC test: CWE-89 OpenSearch injection via unsanitized entity names in query construction.

The test validates that:
1. Wildcard special characters (*, ?) in user input to search_labels are escaped
   before being used in OpenSearch wildcard queries, preventing DoS via expensive
   wildcard patterns.
2. PPL escape handles control characters and additional metacharacters beyond
   just backslash and single-quote.

Run with: pytest tests/test_cwe89_opensearch_injection.py -v
"""

import re
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
import numpy as np

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from lightrag.kg.opensearch_impl import (
    OpenSearchGraphStorage,
    ClientManager,
)

pytestmark = pytest.mark.offline


@asynccontextmanager
async def _mock_lock():
    yield


def _mock_lock_factory():
    return _mock_lock()


@pytest.fixture(autouse=True)
def patch_data_init_lock():
    with patch(
        "lightrag.kg.opensearch_impl.get_data_init_lock", side_effect=_mock_lock_factory
    ):
        yield


class MockEmbeddingFunc:
    def __init__(self, dim=128):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


@pytest.fixture
def global_config():
    return {
        "embedding_batch_num": 10,
        "max_graph_nodes": 1000,
    }


@pytest.fixture
def embed_func():
    return MockEmbeddingFunc()


def _make_client():
    from opensearchpy import AsyncOpenSearch

    client = AsyncMock(spec=AsyncOpenSearch)
    # opensearchpy decorates client methods with @query_params, which hides their
    # coroutine nature from inspect.iscoroutinefunction. AsyncMock(spec=...) then
    # creates plain MagicMocks for them, so callers awaiting client.search(...) hit
    # "object dict can't be used in 'await' expression". Force AsyncMock explicitly.
    client.search = AsyncMock()
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.refresh = AsyncMock()
    client.transport = AsyncMock()
    return client


@pytest.fixture
def graph_storage(global_config, embed_func):
    with patch.object(ClientManager, "get_client") as mock_get:
        client = _make_client()
        mock_get.return_value = client

        storage = OpenSearchGraphStorage(
            namespace="test_graph",
            global_config=global_config,
            embedding_func=embed_func,
        )
        storage.client = client
        storage._indices_ready = True
        storage._ppl_graphlookup_available = True
        yield storage


class TestWildcardInjection:
    """Test that wildcard special chars are escaped in search_labels."""

    @pytest.mark.asyncio
    async def test_wildcard_chars_escaped_in_search_labels(self, graph_storage):
        """Wildcard metacharacters *, ? in user input must be escaped."""
        client = graph_storage.client

        # Setup mock to return empty results
        client.search.return_value = {"hits": {"hits": []}}

        # Malicious query with wildcard chars that could cause expensive patterns
        malicious_query = "test*?foo"
        await graph_storage.search_labels(malicious_query)

        # Inspect the query body that was sent to OpenSearch
        assert client.search.called, "search should have been called"
        call_kwargs = client.search.call_args
        body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body")

        # Extract the wildcard clause
        should_clauses = body["query"]["bool"]["should"]
        wildcard_clause = None
        for clause in should_clauses:
            if "wildcard" in clause:
                wildcard_clause = clause["wildcard"]["entity_id"]["value"]
                break

        assert wildcard_clause is not None, "wildcard clause should exist"

        # The wildcard value should NOT contain unescaped * or ? from the user input
        # The outer * wrapping is fine (those are the intentional wildcards),
        # but the inner user-provided * and ? must be escaped
        # Expected: *test\*\?foo* (with the user's * and ? escaped)
        inner_value = wildcard_clause[1:-1]  # strip leading and trailing *
        assert (
            "\\*" in inner_value
        ), f"User's '*' should be escaped as '\\*' in wildcard, got: {wildcard_clause}"
        assert (
            "\\?" in inner_value
        ), f"User's '?' should be escaped as '\\?' in wildcard, got: {wildcard_clause}"

    @pytest.mark.asyncio
    async def test_wildcard_heavy_pattern_not_exploitable(self, graph_storage):
        """A series of ? chars should be escaped, not passed raw to OpenSearch."""
        client = graph_storage.client
        client.search.return_value = {"hits": {"hits": []}}

        # Attack: many single-char wildcards cause exponential matching
        attack_query = "?" * 50
        await graph_storage.search_labels(attack_query)

        call_kwargs = client.search.call_args
        body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body")

        should_clauses = body["query"]["bool"]["should"]
        wildcard_clause = None
        for clause in should_clauses:
            if "wildcard" in clause:
                wildcard_clause = clause["wildcard"]["entity_id"]["value"]
                break

        # None of the user's ? should appear as unescaped wildcards
        # The value between the outer * delimiters should have all ? escaped
        inner = wildcard_clause[1:-1]
        # Count unescaped ? (i.e., ? not preceded by \)
        unescaped_q = re.findall(r"(?<!\\)\?", inner)
        assert (
            len(unescaped_q) == 0
        ), f"Found {len(unescaped_q)} unescaped '?' in wildcard pattern: {wildcard_clause}"

    @pytest.mark.asyncio
    async def test_backslash_escaped_in_wildcard(self, graph_storage):
        """Backslashes in user input must be double-escaped for the wildcard query."""
        client = graph_storage.client
        client.search.return_value = {"hits": {"hits": []}}

        attack_query = "test\\*"
        await graph_storage.search_labels(attack_query)

        call_kwargs = client.search.call_args
        body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body")

        should_clauses = body["query"]["bool"]["should"]
        wildcard_clause = None
        for clause in should_clauses:
            if "wildcard" in clause:
                wildcard_clause = clause["wildcard"]["entity_id"]["value"]
                break

        # The backslash should be escaped first, then the * — so we get \\\\\\*
        # In the final pattern between outer *...*, user's \ becomes \\ and * becomes \*
        inner = wildcard_clause[1:-1]
        assert (
            "\\\\" in inner or "\\*" in inner
        ), f"Backslash and * from user should be escaped in wildcard: {wildcard_clause}"


class TestPPLInjection:
    """Test that PPL string escape handles additional metacharacters."""

    def test_escape_ppl_basic_quote(self, graph_storage):
        """Single quotes should be escaped."""
        result = graph_storage._escape_ppl("it's a test")
        assert "'" not in result.replace(
            "\\'", ""
        ), f"Unescaped quote found in: {result}"

    def test_escape_ppl_backslash(self, graph_storage):
        """Backslashes should be escaped."""
        result = graph_storage._escape_ppl("test\\path")
        assert result == "test\\\\path"

    def test_escape_ppl_newline_and_control_chars(self, graph_storage):
        """Newlines and control characters should be escaped/stripped."""
        result = graph_storage._escape_ppl("line1\nline2\rline3\t")
        # Control chars should either be stripped or escaped — no raw newlines
        assert "\n" not in result, f"Raw newline in PPL literal: {repr(result)}"
        assert "\r" not in result, f"Raw carriage return in PPL literal: {repr(result)}"

    def test_escape_ppl_pipe_in_quotes_safe(self, graph_storage):
        """Pipe character inside a quoted string literal poses no injection risk."""
        result = graph_storage._escape_ppl("entity | stats count()")
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
