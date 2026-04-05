"""Tests for batch entity extraction pipeline."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.batch_provider import (
    BatchJobState,
    BatchJobStatus,
    BatchProvider,
    BatchRequest,
    BatchResponse,
)
from lightrag.operate import _merge_gleaning_results
from lightrag.utils import (
    Tokenizer,
    TokenizerInterface,
    compute_args_hash,
    generate_cache_key,
    sanitize_text_for_encoding,
)


# ── Helpers ───────────────────────────────────────────────────────────


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping for testing."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


# Minimal valid extraction result that _process_extraction_result can parse
_EXTRACTION_RESULT = (
    "(entity<|#|>TEST_ENTITY<|#|>CONCEPT<|#|>A test entity)<|COMPLETE|>"
)

_EXTRACTION_RESULT_2 = (
    "(entity<|#|>ANOTHER_ENTITY<|#|>CONCEPT<|#|>Another entity)<|COMPLETE|>"
)


def _make_chunks(n: int = 3) -> dict[str, dict]:
    """Create n test chunks."""
    return {
        f"chunk-{i:03d}": {
            "tokens": 20,
            "content": f"Test content for chunk {i}.",
            "full_doc_id": "doc-001",
            "chunk_order_index": i,
        }
        for i in range(n)
    }


def _make_global_config(
    batch_provider=None,
    entity_extract_max_gleaning: int = 0,
    max_extract_input_tokens: int = 999999,
) -> dict:
    """Build a minimal global_config dict for batch extraction."""
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    return {
        "llm_model_func": AsyncMock(return_value=_EXTRACTION_RESULT),
        "entity_extract_max_gleaning": entity_extract_max_gleaning,
        "addon_params": {},
        "tokenizer": tokenizer,
        "max_extract_input_tokens": max_extract_input_tokens,
        "llm_model_max_async": 4,
        "llm_model_name": "test-model",
        "batch_provider": batch_provider,
        "llm_batch_timeout": 60.0,
        "llm_batch_poll_interval": 0.01,
    }


class MockBatchProvider(BatchProvider):
    """Test batch provider that returns canned responses."""

    def __init__(
        self,
        responses: list[BatchResponse] | None = None,
        fail_state: BatchJobState | None = None,
        error_code: str | None = None,
    ):
        self._responses = responses or []
        self._fail_state = fail_state
        self._error_code = error_code
        self._submitted: list[list[BatchRequest]] = []
        self._cancelled: list[str] = []
        self._job_counter = 0

    async def submit_completion_batch(self, requests, model, **kwargs):
        self._submitted.append(list(requests))
        self._job_counter += 1
        return f"test-job-{self._job_counter}"

    async def get_job_status(self, job_id):
        state = self._fail_state or BatchJobState.SUCCEEDED
        return BatchJobStatus(
            job_id=job_id,
            state=state,
            total=len(self._responses),
            succeeded=sum(1 for r in self._responses if not r.error),
            failed=sum(1 for r in self._responses if r.error),
            error_code=self._error_code,
        )

    async def get_results(self, job_id):
        return self._responses

    async def cancel_job(self, job_id):
        self._cancelled.append(job_id)


def _make_mock_cache(cached_keys: dict | None = None):
    """Create a mock llm_response_cache.

    cached_keys: dict of chunk_key -> content string for simulating cache hits.
    """
    cache = AsyncMock()
    cache.global_config = {"enable_llm_cache_for_entity_extract": True}

    # get_by_id for persisted batch state lookup — default to None
    cache.get_by_id = AsyncMock(return_value=None)
    cache.upsert = AsyncMock()
    cache.delete = AsyncMock()

    return cache


# ── Tests: _merge_gleaning_results ────────────────────────────────────


@pytest.mark.offline
class TestMergeGleaningResults:
    def test_new_entity_added(self):
        nodes = {}
        edges = {}
        glean_nodes = {
            "ENTITY_A": [{"description": "desc A", "entity_name": "ENTITY_A"}]
        }
        glean_edges = {}
        _merge_gleaning_results(nodes, edges, glean_nodes, glean_edges)
        assert "ENTITY_A" in nodes

    def test_longer_description_wins(self):
        nodes = {"ENTITY_A": [{"description": "short", "entity_name": "ENTITY_A"}]}
        edges = {}
        glean_nodes = {
            "ENTITY_A": [
                {"description": "a much longer description", "entity_name": "ENTITY_A"}
            ]
        }
        _merge_gleaning_results(nodes, edges, glean_nodes, glean_edges={})
        assert nodes["ENTITY_A"][0]["description"] == "a much longer description"

    def test_shorter_description_rejected(self):
        nodes = {
            "ENTITY_A": [
                {"description": "original longer desc", "entity_name": "ENTITY_A"}
            ]
        }
        edges = {}
        glean_nodes = {
            "ENTITY_A": [{"description": "short", "entity_name": "ENTITY_A"}]
        }
        _merge_gleaning_results(nodes, edges, glean_nodes, glean_edges={})
        assert nodes["ENTITY_A"][0]["description"] == "original longer desc"

    def test_new_edge_added(self):
        nodes = {}
        edges = {}
        glean_edges = {
            ("A", "B"): [{"description": "relates to", "src_id": "A", "tgt_id": "B"}]
        }
        _merge_gleaning_results(nodes, edges, glean_nodes={}, glean_edges=glean_edges)
        assert ("A", "B") in edges

    def test_edge_longer_description_wins(self):
        edges = {("A", "B"): [{"description": "short", "src_id": "A", "tgt_id": "B"}]}
        glean_edges = {
            ("A", "B"): [
                {
                    "description": "a much longer edge description",
                    "src_id": "A",
                    "tgt_id": "B",
                }
            ]
        }
        _merge_gleaning_results(
            maybe_nodes={}, maybe_edges=edges, glean_nodes={}, glean_edges=glean_edges
        )
        assert edges[("A", "B")][0]["description"] == "a much longer edge description"

    def test_empty_inputs(self):
        nodes = {}
        edges = {}
        _merge_gleaning_results(nodes, edges, {}, {})
        assert nodes == {}
        assert edges == {}

    def test_none_descriptions_handled(self):
        nodes = {"ENTITY_A": [{"description": None, "entity_name": "ENTITY_A"}]}
        glean_nodes = {
            "ENTITY_A": [{"description": "has desc", "entity_name": "ENTITY_A"}]
        }
        _merge_gleaning_results(nodes, {}, glean_nodes, {})
        assert nodes["ENTITY_A"][0]["description"] == "has desc"


# ── Tests: Batch extraction pipeline ─────────────────────────────────


@pytest.mark.offline
@pytest.mark.asyncio
async def test_batch_all_cached():
    """When all chunks are cached, no batch is submitted."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks(2)
    provider = MockBatchProvider()
    config = _make_global_config(batch_provider=provider)

    # Patch handle_cache to return hits for all chunks
    with patch(
        "lightrag.operate.handle_cache",
        new_callable=AsyncMock,
        return_value=(_EXTRACTION_RESULT, 1000),
    ):
        results = await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=_make_mock_cache(),
        )

    assert len(results) == 2
    assert len(provider._submitted) == 0  # No batch submitted


@pytest.mark.offline
@pytest.mark.asyncio
async def test_batch_all_miss():
    """When no chunks are cached, all are submitted in a batch."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks(3)
    chunk_keys = list(chunks.keys())

    responses = [BatchResponse(key=k, content=_EXTRACTION_RESULT) for k in chunk_keys]
    provider = MockBatchProvider(responses=responses)
    config = _make_global_config(batch_provider=provider)

    with patch(
        "lightrag.operate.handle_cache",
        new_callable=AsyncMock,
        return_value=None,
    ):
        results = await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=_make_mock_cache(),
        )

    assert len(results) == 3
    assert len(provider._submitted) == 1
    assert len(provider._submitted[0]) == 3


@pytest.mark.offline
@pytest.mark.asyncio
async def test_batch_mixed_cache():
    """Mix of cached and uncached chunks — only misses submitted."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks(3)
    chunk_keys = list(chunks.keys())

    call_count = 0

    async def mock_handle_cache(cache, arg_hash, prompt, mode, cache_type=None):
        nonlocal call_count
        call_count += 1
        # First chunk is cached, rest are not
        if call_count == 1:
            return (_EXTRACTION_RESULT, 1000)
        return None

    # Only 2 chunks need batch submission
    responses = [
        BatchResponse(key=chunk_keys[1], content=_EXTRACTION_RESULT),
        BatchResponse(key=chunk_keys[2], content=_EXTRACTION_RESULT),
    ]
    provider = MockBatchProvider(responses=responses)
    config = _make_global_config(batch_provider=provider)

    with patch("lightrag.operate.handle_cache", side_effect=mock_handle_cache):
        results = await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=_make_mock_cache(),
        )

    assert len(results) == 3
    assert len(provider._submitted) == 1
    assert len(provider._submitted[0]) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_batch_per_row_failure_fallback():
    """Per-row failures fall back to live API."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks(2)
    chunk_keys = list(chunks.keys())

    # First succeeds, second fails
    responses = [
        BatchResponse(key=chunk_keys[0], content=_EXTRACTION_RESULT),
        BatchResponse(key=chunk_keys[1], error="Content filtered"),
    ]
    provider = MockBatchProvider(responses=responses)
    config = _make_global_config(batch_provider=provider)

    with (
        patch(
            "lightrag.operate.handle_cache",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("lightrag.operate.use_llm_func_with_cache") as mock_live,
    ):
        mock_live.return_value = (_EXTRACTION_RESULT, 2000)
        results = await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=_make_mock_cache(),
        )

    assert len(results) == 2
    # Live API called once for the failed row
    assert mock_live.await_count == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_batch_entire_failure_fallback():
    """Entire batch failure falls back to live API for all requests."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks(2)
    provider = MockBatchProvider(fail_state=BatchJobState.FAILED)
    config = _make_global_config(batch_provider=provider)

    with (
        patch(
            "lightrag.operate.handle_cache",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("lightrag.operate.use_llm_func_with_cache") as mock_live,
    ):
        mock_live.return_value = (_EXTRACTION_RESULT, 2000)
        results = await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=_make_mock_cache(),
        )

    assert len(results) == 2
    # Live API called for all chunks
    assert mock_live.await_count == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_batch_token_limit_auto_split():
    """Token limit failure triggers auto-split and resubmission."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks(4)

    class SplittingProvider(BatchProvider):
        def __init__(self):
            self.submitted = []
            self.job_counter = 0

        async def submit_completion_batch(self, requests, model, **kwargs):
            self.submitted.append(list(requests))
            self.job_counter += 1
            return f"job-{self.job_counter}"

        async def get_job_status(self, job_id):
            # First job fails with token limit, subsequent succeed
            if job_id == "job-1":
                # total=0 matches real OpenAI behavior (counts not populated on rejection)
                return BatchJobStatus(
                    job_id=job_id,
                    state=BatchJobState.FAILED,
                    total=0,
                    error_code="token_limit_exceeded",
                )
            return BatchJobStatus(job_id=job_id, state=BatchJobState.SUCCEEDED, total=2)

        async def get_results(self, job_id):
            # Return results for whichever requests were in this sub-batch
            idx = int(job_id.split("-")[1]) - 1
            if idx < len(self.submitted):
                return [
                    BatchResponse(key=r.key, content=_EXTRACTION_RESULT)
                    for r in self.submitted[idx]
                ]
            return []

        async def cancel_job(self, job_id):
            pass

    provider = SplittingProvider()
    config = _make_global_config(batch_provider=provider)

    with patch(
        "lightrag.operate.handle_cache",
        new_callable=AsyncMock,
        return_value=None,
    ):
        results = await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=_make_mock_cache(),
        )

    assert len(results) == 4
    # First submission (4 requests), then two halves (2+2)
    assert len(provider.submitted) == 3
    assert len(provider.submitted[0]) == 4
    assert len(provider.submitted[1]) == 2
    assert len(provider.submitted[2]) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_batch_with_gleaning():
    """Gleaning runs as a second batch when enabled."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks(1)
    chunk_keys = list(chunks.keys())

    responses = [
        BatchResponse(key=chunk_keys[0], content=_EXTRACTION_RESULT),
    ]
    glean_responses = [
        BatchResponse(key=chunk_keys[0], content=_EXTRACTION_RESULT_2),
    ]

    class GleanProvider(BatchProvider):
        def __init__(self):
            self.submitted = []
            self.job_counter = 0

        async def submit_completion_batch(self, requests, model, **kwargs):
            self.submitted.append(list(requests))
            self.job_counter += 1
            return f"job-{self.job_counter}"

        async def get_job_status(self, job_id):
            return BatchJobStatus(job_id=job_id, state=BatchJobState.SUCCEEDED, total=1)

        async def get_results(self, job_id):
            if job_id == "job-1":
                return responses
            return glean_responses

        async def cancel_job(self, job_id):
            pass

    provider = GleanProvider()
    config = _make_global_config(batch_provider=provider, entity_extract_max_gleaning=1)

    with patch(
        "lightrag.operate.handle_cache",
        new_callable=AsyncMock,
        return_value=None,
    ):
        results = await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=_make_mock_cache(),
        )

    assert len(results) == 1
    # Two batches: extraction + gleaning
    assert len(provider.submitted) == 2


# ── Tests: Cache key compatibility ────────────────────────────────────


@pytest.mark.offline
def test_cache_key_matches_live_path():
    """Batch path should produce the same cache key as the live path
    for identical prompts."""
    user_prompt = "Extract entities from: Test content."
    system_prompt = "You are an entity extractor."

    # Replicate batch path logic
    safe_user = sanitize_text_for_encoding(user_prompt)
    safe_system = sanitize_text_for_encoding(system_prompt)
    _prompt = "\n".join([safe_user, safe_system])
    arg_hash = compute_args_hash(_prompt)
    batch_key = generate_cache_key("default", "extract", arg_hash)

    # Replicate live path logic (from use_llm_func_with_cache)
    live_prompt_parts = [safe_user, safe_system]
    live_prompt = "\n".join(live_prompt_parts)
    live_hash = compute_args_hash(live_prompt)
    live_key = generate_cache_key("default", "extract", live_hash)

    assert batch_key == live_key


# ── Tests: Job persistence ────────────────────────────────────────────


@pytest.mark.offline
@pytest.mark.asyncio
async def test_batch_job_persisted_after_submission():
    """After submitting a batch, job state should be persisted."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks(2)
    chunk_keys = list(chunks.keys())
    responses = [BatchResponse(key=k, content=_EXTRACTION_RESULT) for k in chunk_keys]
    provider = MockBatchProvider(responses=responses)
    config = _make_global_config(batch_provider=provider)
    cache = _make_mock_cache()

    with patch(
        "lightrag.operate.handle_cache",
        new_callable=AsyncMock,
        return_value=None,
    ):
        await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=cache,
        )

    # upsert should have been called to persist job state (among other calls)
    batch_key = "__batch_extraction_job__"
    persisted = False
    for call in cache.upsert.call_args_list:
        data = call[0][0]
        if batch_key in data:
            persisted = True
            break
    assert persisted, "Batch job state was not persisted via upsert"

    # delete should have been called to clear it after completion
    cache.delete.assert_any_call([batch_key])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_batch_job_resumed_on_matching_keys():
    """If a persisted job has matching keys, it should resume polling."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks(2)
    chunk_keys = sorted(chunks.keys())
    responses = [BatchResponse(key=k, content=_EXTRACTION_RESULT) for k in chunk_keys]
    provider = MockBatchProvider(responses=responses)
    config = _make_global_config(batch_provider=provider)

    # Simulate a persisted job with matching keys
    persisted_state = {
        "original_prompt": "__batch_extraction_job__",
        "return": json.dumps(
            {
                "job_ids": ["existing-job-123"],
                "submitted_keys": chunk_keys,
                "submitted_at": 1000,
            }
        ),
        "cache_type": "batch_job",
    }
    cache = _make_mock_cache()
    cache.get_by_id = AsyncMock(return_value=persisted_state)

    with patch(
        "lightrag.operate.handle_cache",
        new_callable=AsyncMock,
        return_value=None,
    ):
        await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=cache,
        )

    # Should NOT submit a new batch — should resume the existing one
    assert len(provider._submitted) == 0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_batch_job_discarded_on_mismatched_keys():
    """If persisted job has different keys, discard and submit new."""
    from lightrag.operate import extract_entities

    chunks = _make_chunks(2)
    chunk_keys = sorted(chunks.keys())
    responses = [BatchResponse(key=k, content=_EXTRACTION_RESULT) for k in chunk_keys]
    provider = MockBatchProvider(responses=responses)
    config = _make_global_config(batch_provider=provider)

    # Persisted job has different keys
    persisted_state = {
        "original_prompt": "__batch_extraction_job__",
        "return": json.dumps(
            {
                "job_ids": ["old-job-456"],
                "submitted_keys": ["chunk-999", "chunk-998"],
                "submitted_at": 1000,
            }
        ),
        "cache_type": "batch_job",
    }
    cache = _make_mock_cache()
    cache.get_by_id = AsyncMock(return_value=persisted_state)

    with patch(
        "lightrag.operate.handle_cache",
        new_callable=AsyncMock,
        return_value=None,
    ):
        await extract_entities(
            chunks=chunks,
            global_config=config,
            llm_response_cache=cache,
        )

    # Old state deleted, new batch submitted
    cache.delete.assert_any_call(["__batch_extraction_job__"])
    assert len(provider._submitted) == 1


# ── Tests: Cancellation ──────────────────────────────────────────────


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cancellation_cancels_batch_job():
    """Pipeline cancellation should cancel the provider-side job."""
    from lightrag.exceptions import PipelineCancelledException
    from lightrag.operate import _await_batch_with_cancellation

    provider = MockBatchProvider()
    pipeline_status = {"cancellation_requested": True}
    pipeline_status_lock = asyncio.Lock()

    with pytest.raises(PipelineCancelledException):
        await _await_batch_with_cancellation(
            provider, "job-123", 0.01, 60.0, pipeline_status, pipeline_status_lock
        )

    assert "job-123" in provider._cancelled
