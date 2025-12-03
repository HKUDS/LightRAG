"""
Unit tests for rerank document chunking functionality.

Tests the chunk_documents_for_rerank and aggregate_chunk_scores functions
in lightrag/rerank.py to ensure proper document splitting and score aggregation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from lightrag.rerank import (
    chunk_documents_for_rerank,
    aggregate_chunk_scores,
    cohere_rerank,
)


class TestChunkDocumentsForRerank:
    """Test suite for chunk_documents_for_rerank function"""

    def test_no_chunking_needed_for_short_docs(self):
        """Documents shorter than max_tokens should not be chunked"""
        documents = [
            "Short doc 1",
            "Short doc 2",
            "Short doc 3",
        ]

        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=100, overlap_tokens=10
        )

        # No chunking should occur
        assert len(chunked_docs) == 3
        assert chunked_docs == documents
        assert doc_indices == [0, 1, 2]

    def test_chunking_with_character_fallback(self):
        """Test chunking falls back to character-based when tokenizer unavailable"""
        # Create a very long document that exceeds character limit
        long_doc = "a" * 2000  # 2000 characters
        documents = [long_doc, "short doc"]

        with patch("lightrag.rerank.TiktokenTokenizer", side_effect=ImportError):
            chunked_docs, doc_indices = chunk_documents_for_rerank(
                documents,
                max_tokens=100,  # 100 tokens = ~400 chars
                overlap_tokens=10,  # 10 tokens = ~40 chars
            )

        # First doc should be split into chunks, second doc stays whole
        assert len(chunked_docs) > 2  # At least one chunk from first doc + second doc
        assert chunked_docs[-1] == "short doc"  # Last chunk is the short doc
        # Verify doc_indices maps chunks to correct original document
        assert doc_indices[-1] == 1  # Last chunk maps to document 1

    def test_chunking_with_tiktoken_tokenizer(self):
        """Test chunking with actual tokenizer"""
        # Create document with known token count
        # Approximate: "word " = ~1 token, so 200 words ~ 200 tokens
        long_doc = " ".join([f"word{i}" for i in range(200)])
        documents = [long_doc, "short"]

        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=50, overlap_tokens=10
        )

        # Long doc should be split, short doc should remain
        assert len(chunked_docs) > 2
        assert doc_indices[-1] == 1  # Last chunk is from second document

        # Verify overlapping chunks contain overlapping content
        if len(chunked_docs) > 2:
            # Check that consecutive chunks from same doc have some overlap
            for i in range(len(doc_indices) - 1):
                if doc_indices[i] == doc_indices[i + 1] == 0:
                    # Both chunks from first doc, should have overlap
                    chunk1_words = chunked_docs[i].split()
                    chunk2_words = chunked_docs[i + 1].split()
                    # At least one word should be common due to overlap
                    assert any(word in chunk2_words for word in chunk1_words[-5:])

    def test_empty_documents(self):
        """Test handling of empty document list"""
        documents = []
        chunked_docs, doc_indices = chunk_documents_for_rerank(documents)

        assert chunked_docs == []
        assert doc_indices == []

    def test_single_document_chunking(self):
        """Test chunking of a single long document"""
        # Create document with ~100 tokens
        long_doc = " ".join([f"token{i}" for i in range(100)])
        documents = [long_doc]

        chunked_docs, doc_indices = chunk_documents_for_rerank(
            documents, max_tokens=30, overlap_tokens=5
        )

        # Should create multiple chunks
        assert len(chunked_docs) > 1
        # All chunks should map to document 0
        assert all(idx == 0 for idx in doc_indices)


class TestAggregateChunkScores:
    """Test suite for aggregate_chunk_scores function"""

    def test_no_chunking_simple_aggregation(self):
        """Test aggregation when no chunking occurred (1:1 mapping)"""
        chunk_results = [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.7},
            {"index": 2, "relevance_score": 0.5},
        ]
        doc_indices = [0, 1, 2]  # 1:1 mapping
        num_original_docs = 3

        aggregated = aggregate_chunk_scores(
            chunk_results, doc_indices, num_original_docs, aggregation="max"
        )

        # Results should be sorted by score
        assert len(aggregated) == 3
        assert aggregated[0]["index"] == 0
        assert aggregated[0]["relevance_score"] == 0.9
        assert aggregated[1]["index"] == 1
        assert aggregated[1]["relevance_score"] == 0.7
        assert aggregated[2]["index"] == 2
        assert aggregated[2]["relevance_score"] == 0.5

    def test_max_aggregation_with_chunks(self):
        """Test max aggregation strategy with multiple chunks per document"""
        # 5 chunks: first 3 from doc 0, last 2 from doc 1
        chunk_results = [
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.8},
            {"index": 2, "relevance_score": 0.6},
            {"index": 3, "relevance_score": 0.7},
            {"index": 4, "relevance_score": 0.4},
        ]
        doc_indices = [0, 0, 0, 1, 1]
        num_original_docs = 2

        aggregated = aggregate_chunk_scores(
            chunk_results, doc_indices, num_original_docs, aggregation="max"
        )

        # Should take max score for each document
        assert len(aggregated) == 2
        assert aggregated[0]["index"] == 0
        assert aggregated[0]["relevance_score"] == 0.8  # max of 0.5, 0.8, 0.6
        assert aggregated[1]["index"] == 1
        assert aggregated[1]["relevance_score"] == 0.7  # max of 0.7, 0.4

    def test_mean_aggregation_with_chunks(self):
        """Test mean aggregation strategy"""
        chunk_results = [
            {"index": 0, "relevance_score": 0.6},
            {"index": 1, "relevance_score": 0.8},
            {"index": 2, "relevance_score": 0.4},
        ]
        doc_indices = [0, 0, 1]  # First two chunks from doc 0, last from doc 1
        num_original_docs = 2

        aggregated = aggregate_chunk_scores(
            chunk_results, doc_indices, num_original_docs, aggregation="mean"
        )

        assert len(aggregated) == 2
        assert aggregated[0]["index"] == 0
        assert aggregated[0]["relevance_score"] == pytest.approx(0.7)  # (0.6 + 0.8) / 2
        assert aggregated[1]["index"] == 1
        assert aggregated[1]["relevance_score"] == 0.4

    def test_first_aggregation_with_chunks(self):
        """Test first aggregation strategy"""
        chunk_results = [
            {"index": 0, "relevance_score": 0.6},
            {"index": 1, "relevance_score": 0.8},
            {"index": 2, "relevance_score": 0.4},
        ]
        doc_indices = [0, 0, 1]
        num_original_docs = 2

        aggregated = aggregate_chunk_scores(
            chunk_results, doc_indices, num_original_docs, aggregation="first"
        )

        assert len(aggregated) == 2
        # First should use first score seen for each doc
        assert aggregated[0]["index"] == 0
        assert aggregated[0]["relevance_score"] == 0.6  # First score for doc 0
        assert aggregated[1]["index"] == 1
        assert aggregated[1]["relevance_score"] == 0.4

    def test_empty_chunk_results(self):
        """Test handling of empty results"""
        aggregated = aggregate_chunk_scores([], [], 3, aggregation="max")
        assert aggregated == []

    def test_documents_with_no_scores(self):
        """Test when some documents have no chunks/scores"""
        chunk_results = [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.7},
        ]
        doc_indices = [0, 0]  # Both chunks from document 0
        num_original_docs = 3  # But we have 3 documents total

        aggregated = aggregate_chunk_scores(
            chunk_results, doc_indices, num_original_docs, aggregation="max"
        )

        # Only doc 0 should appear in results
        assert len(aggregated) == 1
        assert aggregated[0]["index"] == 0

    def test_unknown_aggregation_strategy(self):
        """Test that unknown strategy falls back to max"""
        chunk_results = [
            {"index": 0, "relevance_score": 0.6},
            {"index": 1, "relevance_score": 0.8},
        ]
        doc_indices = [0, 0]
        num_original_docs = 1

        # Use invalid strategy
        aggregated = aggregate_chunk_scores(
            chunk_results, doc_indices, num_original_docs, aggregation="invalid"
        )

        # Should fall back to max
        assert aggregated[0]["relevance_score"] == 0.8


@pytest.mark.offline
class TestTopNWithChunking:
    """Tests for top_n behavior when chunking is enabled (Bug fix verification)"""

    @pytest.mark.asyncio
    async def test_top_n_limits_documents_not_chunks(self):
        """
        Test that top_n correctly limits documents (not chunks) when chunking is enabled.

        Bug scenario: 10 docs expand to 50 chunks. With old behavior, top_n=5 would
        return scores for only 5 chunks (possibly all from 1-2 docs). After aggregation,
        fewer than 5 documents would be returned.

        Fixed behavior: top_n=5 should return exactly 5 documents after aggregation.
        """
        # Setup: 5 documents, each producing multiple chunks when chunked
        # Using small max_tokens to force chunking
        long_docs = [" ".join([f"doc{i}_word{j}" for j in range(50)]) for i in range(5)]
        query = "test query"

        # First, determine how many chunks will be created by actual chunking
        _, doc_indices = chunk_documents_for_rerank(
            long_docs, max_tokens=50, overlap_tokens=10
        )
        num_chunks = len(doc_indices)

        # Mock API returns scores for ALL chunks (simulating disabled API-level top_n)
        # Give different scores to ensure doc 0 gets highest, doc 1 second, etc.
        # Assign scores based on original document index (lower doc index = higher score)
        mock_chunk_scores = []
        for i in range(num_chunks):
            original_doc = doc_indices[i]
            # Higher score for lower doc index, with small variation per chunk
            base_score = 0.9 - (original_doc * 0.1)
            mock_chunk_scores.append({"index": i, "relevance_score": base_score})

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"results": mock_chunk_scores})
        mock_response.request_info = None
        mock_response.history = None
        mock_response.headers = {}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=mock_session):
            result = await cohere_rerank(
                query=query,
                documents=long_docs,
                api_key="test-key",
                base_url="http://test.com/rerank",
                enable_chunking=True,
                max_tokens_per_doc=50,  # Match chunking above
                top_n=3,  # Request top 3 documents
            )

            # Verify: should get exactly 3 documents (not unlimited chunks)
            assert len(result) == 3
            # All results should have valid document indices (0-4)
            assert all(0 <= r["index"] < 5 for r in result)
            # Results should be sorted by score (descending)
            assert all(
                result[i]["relevance_score"] >= result[i + 1]["relevance_score"]
                for i in range(len(result) - 1)
            )
            # The top 3 docs should be 0, 1, 2 (highest scores)
            result_indices = [r["index"] for r in result]
            assert set(result_indices) == {0, 1, 2}

    @pytest.mark.asyncio
    async def test_api_receives_no_top_n_when_chunking_enabled(self):
        """
        Test that the API request does NOT include top_n when chunking is enabled.

        This ensures all chunk scores are retrieved for proper aggregation.
        """
        documents = [" ".join([f"word{i}" for i in range(100)]), "short doc"]
        query = "test query"

        captured_payload = {}

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 2, "relevance_score": 0.7},
                ]
            }
        )
        mock_response.request_info = None
        mock_response.history = None
        mock_response.headers = {}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return mock_response

        mock_session = Mock()
        mock_session.post = Mock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=mock_session):
            await cohere_rerank(
                query=query,
                documents=documents,
                api_key="test-key",
                base_url="http://test.com/rerank",
                enable_chunking=True,
                max_tokens_per_doc=30,
                top_n=1,  # User wants top 1 document
            )

            # Verify: API payload should NOT have top_n (disabled for chunking)
            assert "top_n" not in captured_payload

    @pytest.mark.asyncio
    async def test_top_n_not_modified_when_chunking_disabled(self):
        """
        Test that top_n is passed through to API when chunking is disabled.
        """
        documents = ["doc1", "doc2"]
        query = "test query"

        captured_payload = {}

        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": 0.9},
                ]
            }
        )
        mock_response.request_info = None
        mock_response.history = None
        mock_response.headers = {}
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        def capture_post(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return mock_response

        mock_session = Mock()
        mock_session.post = Mock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=mock_session):
            await cohere_rerank(
                query=query,
                documents=documents,
                api_key="test-key",
                base_url="http://test.com/rerank",
                enable_chunking=False,  # Chunking disabled
                top_n=1,
            )

            # Verify: API payload should have top_n when chunking is disabled
            assert captured_payload.get("top_n") == 1


@pytest.mark.offline
class TestCohereRerankChunking:
    """Integration tests for cohere_rerank with chunking enabled"""

    @pytest.mark.asyncio
    async def test_cohere_rerank_with_chunking_disabled(self):
        """Test that chunking can be disabled"""
        documents = ["doc1", "doc2"]
        query = "test query"

        # Mock the generic_rerank_api
        with patch(
            "lightrag.rerank.generic_rerank_api", new_callable=AsyncMock
        ) as mock_api:
            mock_api.return_value = [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
            ]

            result = await cohere_rerank(
                query=query,
                documents=documents,
                api_key="test-key",
                enable_chunking=False,
                max_tokens_per_doc=100,
            )

            # Verify generic_rerank_api was called with correct parameters
            mock_api.assert_called_once()
            call_kwargs = mock_api.call_args[1]
            assert call_kwargs["enable_chunking"] is False
            assert call_kwargs["max_tokens_per_doc"] == 100
            # Result should mirror mocked scores
            assert len(result) == 2
            assert result[0]["index"] == 0
            assert result[0]["relevance_score"] == 0.9
            assert result[1]["index"] == 1
            assert result[1]["relevance_score"] == 0.7

    @pytest.mark.asyncio
    async def test_cohere_rerank_with_chunking_enabled(self):
        """Test that chunking parameters are passed through"""
        documents = ["doc1", "doc2"]
        query = "test query"

        with patch(
            "lightrag.rerank.generic_rerank_api", new_callable=AsyncMock
        ) as mock_api:
            mock_api.return_value = [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
            ]

            result = await cohere_rerank(
                query=query,
                documents=documents,
                api_key="test-key",
                enable_chunking=True,
                max_tokens_per_doc=480,
            )

            # Verify parameters were passed
            call_kwargs = mock_api.call_args[1]
            assert call_kwargs["enable_chunking"] is True
            assert call_kwargs["max_tokens_per_doc"] == 480
            # Result should mirror mocked scores
            assert len(result) == 2
            assert result[0]["index"] == 0
            assert result[0]["relevance_score"] == 0.9
            assert result[1]["index"] == 1
            assert result[1]["relevance_score"] == 0.7

    @pytest.mark.asyncio
    async def test_cohere_rerank_default_parameters(self):
        """Test default parameter values for cohere_rerank"""
        documents = ["doc1"]
        query = "test"

        with patch(
            "lightrag.rerank.generic_rerank_api", new_callable=AsyncMock
        ) as mock_api:
            mock_api.return_value = [{"index": 0, "relevance_score": 0.9}]

            result = await cohere_rerank(
                query=query, documents=documents, api_key="test-key"
            )

            # Verify default values
            call_kwargs = mock_api.call_args[1]
            assert call_kwargs["enable_chunking"] is False
            assert call_kwargs["max_tokens_per_doc"] == 4096
            assert call_kwargs["model"] == "rerank-v3.5"
            # Result should mirror mocked scores
            assert len(result) == 1
            assert result[0]["index"] == 0
            assert result[0]["relevance_score"] == 0.9


@pytest.mark.offline
class TestEndToEndChunking:
    """End-to-end tests for chunking workflow"""

    @pytest.mark.asyncio
    async def test_end_to_end_chunking_workflow(self):
        """Test complete chunking workflow from documents to aggregated results"""
        # Create documents where first one needs chunking
        long_doc = " ".join([f"word{i}" for i in range(100)])
        documents = [long_doc, "short doc"]
        query = "test query"

        # Mock the HTTP call inside generic_rerank_api
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "results": [
                    {"index": 0, "relevance_score": 0.5},  # chunk 0 from doc 0
                    {"index": 1, "relevance_score": 0.8},  # chunk 1 from doc 0
                    {"index": 2, "relevance_score": 0.6},  # chunk 2 from doc 0
                    {"index": 3, "relevance_score": 0.7},  # doc 1 (short)
                ]
            }
        )
        mock_response.request_info = None
        mock_response.history = None
        mock_response.headers = {}
        # Make mock_response an async context manager (for `async with session.post() as response`)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = Mock()
        # session.post() returns an async context manager, so return mock_response which is now one
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("lightrag.rerank.aiohttp.ClientSession", return_value=mock_session):
            result = await cohere_rerank(
                query=query,
                documents=documents,
                api_key="test-key",
                base_url="http://test.com/rerank",
                enable_chunking=True,
                max_tokens_per_doc=30,  # Force chunking of long doc
            )

            # Should get 2 results (one per original document)
            # The long doc's chunks should be aggregated
            assert len(result) <= len(documents)
            # Results should be sorted by score
            assert all(
                result[i]["relevance_score"] >= result[i + 1]["relevance_score"]
                for i in range(len(result) - 1)
            )
