# Batch API Support for LightRAG Ingestion Pipeline

## Context

LightRAG's entity extraction pipeline (`extract_entities()` in operate.py) processes chunks one-at-a-time through a semaphore that drip-feeds 4 concurrent LLM calls. For large ingestion jobs, provider batch APIs (Gemini, OpenAI) offer 50% cost savings and higher throughput by submitting all requests at once.

The semaphore at operate.py makes transparent adapter-level batching impossible. Batching must happen **above** the semaphore, at the orchestration level. This plan adds an opt-in batch code path that bypasses the semaphore, pre-generates all prompts, checks the cache, and submits cache misses as a single batch.

Users opt in by selecting `gemini_batch` or `openai_batch` as their provider in `make env-base`.

---

## Phase 1: BatchProvider Abstraction

### `lightrag/llm/batch_provider.py`

Provider-agnostic abstraction:

```python
class BatchJobState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BatchRequest:
    key: str                    # Caller-assigned (chunk_key), for result reordering
    prompt: str                 # User prompt
    system_prompt: str | None = None
    history_messages: list[dict] | None = None

@dataclass
class BatchResponse:
    key: str                    # Matches BatchRequest.key
    content: str | None = None  # LLM response text (None if failed)
    error: str | None = None    # Error message if this row failed
    prompt_tokens: int = 0
    completion_tokens: int = 0

@dataclass
class BatchJobStatus:
    job_id: str
    state: BatchJobState
    total: int = 0
    succeeded: int = 0
    failed: int = 0

class BatchProvider(ABC):
    async def submit_completion_batch(self, requests: list[BatchRequest], model: str, **kwargs) -> str:
        """Submit batch. Returns job_id."""
    async def get_job_status(self, job_id: str) -> BatchJobStatus:
    async def get_results(self, job_id: str) -> list[BatchResponse]:
    async def cancel_job(self, job_id: str) -> None:
    async def await_completion(self, job_id, poll_interval=30.0, timeout=3600.0) -> BatchJobStatus:
        """Default polling implementation."""
```

---

## Phase 2: Provider Implementations

### 2A. `lightrag/llm/gemini_batch.py` — GeminiBatchProvider

- Reuses `_get_gemini_client()` from `gemini.py` (LRU-cached)
- Reuses `_build_generation_config()` for system prompt / config construction
- Uses `client.aio.batches.create(model=model, src=[InlinedRequest(...)])` for submission
- Uses `InlinedRequest.metadata` dict to store the `key` for result reordering
- Uses `client.aio.batches.get(name=job_id)` for polling
- Maps Gemini's `JobState` to `BatchJobState`
- Formats `history_messages` into the prompt the same way `gemini_complete_if_cache` does

### 2B. `lightrag/llm/openai_batch.py` — OpenAIBatchProvider

Uses OpenAI's file-based Batch API (different flow from Gemini):

1. **Build JSONL** — each line: `{"custom_id": key, "method": "POST", "url": "/v1/chat/completions", "body": {"model": ..., "messages": [...]}}`
2. **Upload file** via Files API: `client.files.create(file=..., purpose="batch")`
3. **Create batch**: `client.batches.create(input_file_id=..., endpoint="/v1/chat/completions", completion_window="24h")`
4. **Poll**: `client.batches.retrieve(batch_id)` — states: `validating`, `in_progress`, `finalizing`, `completed`, `failed`, `expired`, `cancelled`
5. **Download results**: `client.files.content(batch.output_file_id)` — JSONL with `{"custom_id": ..., "response": {"status_code": 200, "body": {"choices": [...], "usage": {...}}}}`

Key differences from Gemini:
- Reuses `create_openai_async_client()` from `openai.py`
- Messages formatted in OpenAI format: `[{"role": "system", ...}, {"role": "user", ...}]`
- Accepts `extra_params` dict (from `OpenAILLMOptions`) to pass temperature, max_tokens, etc. into each request body
- Results keyed by `custom_id` → maps back to `BatchRequest.key`
- Error handling: checks both `output_file_id` and `error_file_id`

---

## Phase 3: Provider Configuration — Batch Bindings

Both `gemini_batch` and `openai_batch` bindings use the same live API for queries but additionally create a `BatchProvider` for ingestion.

### 3A. Setup wizard: `scripts/setup/setup.sh`

- `collect_llm_config()`: `gemini_batch` and `openai_batch` added to options array
- `gemini_batch` shares `gemini)` case; `openai_batch` falls through to `*)` default (same as `openai`)
- `default_llm_model_for_binding()`: `openai|openai_batch|azure_openai)` → `gpt-5-mini`; `gemini|gemini_batch)` → `gemini-flash-latest`
- `collect_embedding_config()`: both batch bindings added to options
- `default_embedding_model_for_binding()` / `default_embedding_dim_for_binding()`: batch bindings alongside their live counterparts

### 3B. Config: `lightrag/api/config.py`

- `get_default_host()`: `openai_batch` → `https://api.openai.com/v1`; `gemini_batch` → Gemini endpoint
- LLM binding arg registration: `openai_batch` uses `OpenAILLMOptions`; `gemini_batch` uses `GeminiLLMOptions`
- Embedding binding arg registration: same pattern

### 3C. Server: `lightrag/api/lightrag_server.py`

- Binding validation allowlists: both `gemini_batch` and `openai_batch` added to LLM and embedding lists
- `LLMConfigCache`: `openai_batch` triggers `OpenAILLMOptions` initialization; `gemini_batch` triggers `GeminiLLMOptions`
- `create_llm_model_func()`: `gemini_batch` → live Gemini function; `openai_batch` falls through to `else` (same as `openai`)
- `create_optimized_embedding_function()`: `openai_batch` uses `openai_embed`; `gemini_batch` uses `gemini_embed`
- Batch provider creation at LightRAG instantiation:
  ```python
  if args.llm_binding == "gemini_batch":
      batch_provider = GeminiBatchProvider(api_key=..., base_url=...)
  elif args.llm_binding == "openai_batch":
      batch_provider = OpenAIBatchProvider(api_key=..., base_url=..., extra_params=openai_llm_options)
  ```
- Passes `batch_provider`, `llm_batch_timeout`, `llm_batch_poll_interval` to LightRAG constructor

### 3D. `env.example`

```bash
### OpenAI Batch API example
# # LLM_BINDING=openai_batch
# # LLM_BINDING_HOST=https://api.openai.com/v1
# # LLM_BINDING_API_KEY=your_api_key
# # LLM_MODEL=gpt-4o-mini
# # LLM_BATCH_TIMEOUT=86400
# # LLM_BATCH_POLL_INTERVAL=30

### Google Gemini Batch API example
# # LLM_BINDING=gemini_batch
# # LLM_BINDING_API_KEY=your_gemini_api_key
# # LLM_BINDING_HOST=https://generativelanguage.googleapis.com
# # LLM_MODEL=gemini-flash-latest
# # LLM_BATCH_TIMEOUT=3600
# # LLM_BATCH_POLL_INTERVAL=30
```

---

## Phase 4: LightRAG Config Fields

### `lightrag/lightrag.py`

Three new fields on the `LightRAG` dataclass:

```python
batch_provider: Any = field(default=None)
"""Optional BatchProvider instance for batch API ingestion. When set, extraction
uses batch submission instead of per-chunk concurrent calls."""

llm_batch_timeout: float = field(
    default=float(os.getenv("LLM_BATCH_TIMEOUT", "3600"))
)
"""Maximum time (seconds) to wait for a batch job to complete."""

llm_batch_poll_interval: float = field(
    default=float(os.getenv("LLM_BATCH_POLL_INTERVAL", "30"))
)
"""Polling interval (seconds) for batch job status checks."""
```

### `asdict()` handling

`BatchProvider` contains unpicklable objects (API clients with thread locks) that cause `dataclasses.asdict()` to crash. Handled by:

1. `__post_init__`: temporarily sets `self.batch_provider = None` before `asdict()`, restores after
2. `_build_global_config()`: extracted helper method that all call sites use instead of raw `asdict(self)`, ensuring `batch_provider` and `embedding_func` are correctly preserved

---

## Phase 5: Batch Extraction in operate.py

### 5A. Gate in `extract_entities()`

After `context_base` setup, before `_process_single_content`:

```python
batch_provider = global_config.get("batch_provider")
if batch_provider is not None:
    return await _extract_entities_batch(...)
```

### 5B. `_extract_entities_batch()`

**Step 1 — Pre-generate all prompts** for all chunks upfront (system, user, continue).

**Step 2 — Check cache** for each prompt, replicating exact cache key logic from `use_llm_func_with_cache` (sanitize → hash → `generate_cache_key("default", "extract", hash)`).

**Step 3 — Submit batch or resume persisted job:**
- Before submitting, check `llm_response_cache` for a persisted job state (key `__batch_extraction_job__`)
- If a persisted job exists and its `submitted_keys` match the current cache misses, resume polling it instead of resubmitting
- If no match or no persisted job, submit a new batch via `batch_provider.submit_completion_batch()`
- After submitting, persist `{job_id, submitted_keys, submitted_at}` to `llm_response_cache`
- Poll via `_await_batch_with_cancellation()` (checks pipeline cancellation each cycle, logs status to pipeline history and server logs)
- On terminal state, clear persisted job state
- On success: distribute results, save each to cache via `save_to_cache()`
- On per-row failure: add to `failed_keys` list

**Step 4 — Fallback for failed rows:** call `use_llm_func_with_cache()` individually via the live API.

**Step 5 — Parse all results:** `_process_extraction_result()` for each chunk → `(maybe_nodes, maybe_edges)`.

**Step 6 — Gleaning batch (if enabled):**
- Generate gleaning prompts (depend on first-round results → must be 2nd batch)
- Check gleaning cache for each
- Submit cache misses as second batch
- Gleaning failures are **non-fatal** — proceed with initial extraction only, log warning
- Apply results with `_merge_gleaning_results()` (description length comparison)

**Step 7 — Return:** update chunk cache lists, return `list[tuple[dict, dict]]` — same format as live path.

### 5C. Helper: `_merge_gleaning_results()`

Shared function (used by both batch and live paths) that merges gleaning results by description length comparison — keeps whichever version has the longer description, adds new entities/edges directly.

### 5D. Helper: `_await_batch_with_cancellation()`

Polls batch status while checking pipeline cancellation. Logs state, elapsed time, and progress to both `pipeline_status["history_messages"]` and `logger.info()` each poll cycle. When the user cancels the pipeline, calls `batch_provider.cancel_job()` to cancel the provider-side job (works for both OpenAI and Gemini).

### 5E. Batch Job Persistence

Batch job state is persisted in `llm_response_cache` (existing KV store, workspace-scoped, disk-backed) under key `__batch_extraction_job__`.

**Storage format compatibility:** The Postgres KV backend has a rigid schema for the LLM cache namespace, requiring `original_prompt`, `return`, and `cache_type` fields. The batch job state is packed/unpacked through helper functions:
- `_pack_batch_state()`: wraps the state dict as `{"original_prompt": key, "return": json.dumps(state), "cache_type": "batch_job"}`
- `_unpack_batch_state()`: handles both JSON-based backends (which store dicts directly) and Postgres (which wraps in the rigid schema)

On re-entry after restart:
- Same document re-ingested → same chunks → same cache misses → keys match → resumes polling
- Different document → keys don't match → discards old state, submits new batch
- Job completes or fails → state is cleared

### 5F. Error Handling and Retry Strategies

The batch path has three layers of error handling, each addressing different failure modes.

#### Layer 1: Submission-time errors (transient)

When `batch_provider.submit_completion_batch()` raises an exception:
- If the error message contains "limit" and the sub-batch has >1 request → halve `max_sub_batch` and retry
- Otherwise → mark remaining requests as failed, fall back to live API

This catches synchronous rejections (e.g., request too large, invalid model).

#### Layer 2: Post-submission batch failures (token limits)

When a batch job reaches `FAILED` state after polling:
- `BatchJobStatus.error_code` is checked for token/quota limit errors (e.g., OpenAI's `token_limit_exceeded`)
- If limit-related and batch had >1 request → split unresolved requests in half, resubmit as new sub-batches, and continue polling
- Otherwise → mark unresolved requests as failed

This handles provider-side validation that happens after submission (e.g., OpenAI's enqueued token limit of 2M tokens per model, which is only checked asynchronously).

The splitting is recursive — if the halved sub-batch still exceeds the limit, it will be split again on the next failure, converging via binary search.

#### Layer 3: Per-row failures within succeeded batches

When a batch completes with `SUCCEEDED` state but individual rows have errors:
- Each `BatchResponse` with `error` set is added to `failed_keys`
- All failed keys fall back to live API individually (Step 4)

This handles partial failures — e.g., a single prompt that triggers content filtering while the rest succeed.

#### Live API retry (fallback path)

Failed batch rows fall back to `use_llm_func_with_cache()` which goes through the standard live API path. This inherits the existing retry decorators:

**Gemini** (`@retry` in `gemini.py`):
- 3 attempts, exponential backoff (4-60s wait)
- Retries on: `InternalServerError`, `ServiceUnavailable`, `ResourceExhausted`, `GatewayTimeout`, `BadGateway`, `DeadlineExceeded`, `Aborted`, `Unknown`, `InvalidResponseError`
- Non-retryable (immediate failure): `InvalidArgument` (400), `PermissionDenied` (401/403), `NotFound` (404)

**OpenAI** (`@retry` in `openai.py`):
- 3 attempts, exponential backoff (4-10s wait)
- Retries on: `RateLimitError`, `APIConnectionError`, `APITimeoutError`, `InvalidResponseError`
- Non-retryable: `AuthenticationError`, `NotFoundError`, `BadRequestError`

#### Gleaning batch failures

Gleaning (optional refinement pass) has relaxed error handling:
- Entire gleaning batch failure → log warning, proceed with initial extraction results
- `PipelineCancelledException` is re-raised (user intent respected)
- Per-row gleaning failures → silently skipped (initial extraction is sufficient)

---

## Phase 6: Batch Embeddings (Deferred)

Add batch embedding functions using provider-specific batch APIs. When `EMBEDDING_BINDING=gemini_batch` or `openai_batch`, the server creates the embedding function using the batch variant.

Lower priority than extraction batching — implement after Phase 5 is validated.

---

## Files Modified/Created Summary

| File | Action |
|------|--------|
| `lightrag/llm/batch_provider.py` | **New** — BatchProvider ABC + data types |
| `lightrag/llm/gemini_batch.py` | **New** — GeminiBatchProvider (inlined requests) |
| `lightrag/llm/openai_batch.py` | **New** — OpenAIBatchProvider (JSONL file upload) |
| `lightrag/operate.py` | Batch extraction path, helpers, persistence with Postgres compat |
| `lightrag/lightrag.py` | 3 config fields, `_build_global_config()`, asdict fix |
| `lightrag/api/lightrag_server.py` | Batch binding wiring, validation, provider creation |
| `lightrag/api/config.py` | Batch binding resolution |
| `scripts/setup/setup.sh` | Batch bindings in wizard choices |
| `env.example` | Batch env vars documentation |

---

## Edge Cases

| Scenario | Handling |
|---|---|
| All chunks cached | `cache_misses` is empty, skip batch submit, all results from cache |
| Single chunk | Batch of 1 is valid |
| Cancellation during batch wait | `_await_batch_with_cancellation` checks each poll, cancels provider-side job |
| Per-row failure | Failed rows fall back to live API individually |
| Entire batch failure (token limit) | Auto-split in half, resubmit smaller sub-batches (recursive) |
| Entire batch failure (other) | All requests fall back to live API |
| Gleaning batch failure | Non-fatal — proceed with initial extraction, log warning |
| Gleaning disabled | Skip step 6 entirely |
| `batch_provider` is None | Gate falls through to existing live path (default) |
| Server restart mid-batch | Job state persisted in KV store, resumed on re-ingestion |
| Restart with different document | Persisted state discarded, new batch submitted |
| Postgres KV backend | Batch state packed into rigid `original_prompt`/`return` schema |

---

## Usage

### Via Server — OpenAI Batch
```bash
# In .env:
LLM_BINDING=openai_batch
LLM_BINDING_API_KEY=your_key
LLM_MODEL=gpt-4o-mini
LLM_BATCH_TIMEOUT=86400
LLM_BATCH_POLL_INTERVAL=30

# Start server:
lightrag-server
```

### Via Server — Gemini Batch
```bash
# In .env:
LLM_BINDING=gemini_batch
LLM_BINDING_API_KEY=your_key
LLM_MODEL=gemini-2.0-flash
LLM_BATCH_TIMEOUT=86400
LLM_BATCH_POLL_INTERVAL=30

# Start server:
lightrag-server
```

### Programmatic
```python
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.llm.openai_batch import OpenAIBatchProvider

rag = LightRAG(
    working_dir="./storage",
    llm_model_func=openai_complete_if_cache,
    embedding_func=openai_embed,
    batch_provider=OpenAIBatchProvider(api_key="..."),
    llm_batch_timeout=86400,
)
await rag.initialize_storages()
await rag.ainsert(large_document_list)  # Uses batch API for extraction
```

---

## Important Notes

- **Batch API latency**: Both Gemini and OpenAI batch APIs are designed for non-time-critical workloads. Jobs can stay in `pending` state for minutes to hours while queued. The cost savings comes at the expense of latency.
- **Best for large jobs**: Batch mode is most valuable for ingesting thousands of chunks where cost savings outweigh wait time. For small/interactive ingestion, use the live API binding (`gemini` or `openai`) with `MAX_ASYNC=16`.
- **Queries always use live API**: Batch bindings only affect ingestion. Queries (`aquery`) always use the live API for real-time responses.
- **Embedding binding is independent**: `LLM_BINDING=openai_batch` only changes entity extraction. Set `EMBEDDING_BINDING` independently to match your existing vector store dimensions.

---

## Verification Plan

1. **Unit tests** (offline, no API calls):
   - Mock `BatchProvider` returning canned responses
   - Test `_extract_entities_batch()`: all-cached, all-miss, mixed, per-row failures, gleaning enabled/disabled
   - Test cache key compatibility: batch path produces same keys as live path for identical prompts
   - Test `_merge_gleaning_results()` independently
   - Test job persistence: save state (JSON and Postgres backends), simulate restart, verify resume

2. **Integration tests** (requires API keys, `--run-integration`):
   - `GeminiBatchProvider`: submit 3-request batch, poll, verify results
   - `OpenAIBatchProvider`: submit 3-request batch, poll, download results file, verify
   - End-to-end: insert document with `batch_provider`, verify entities match live-path
   - Cancellation: start batch, cancel pipeline, verify provider-side job cancelled

3. **Manual smoke test**:
   - Insert multi-page PDF with ~50 chunks using both live and batch paths
   - Compare entity/relation counts
   - Verify cache: second run hits cache on both paths
