# LightRAG Multi-Document Processing: Concurrent Control Strategy Analysis

LightRAG employs a multi-layered concurrent control strategy when processing multiple documents. This article provides an in-depth analysis of the concurrent control mechanisms at document level, chunk level, and LLM request level, helping you understand why specific concurrent behaviors occur.

## Overview

LightRAG's concurrent control is divided into three layers:

1. **Document-level concurrency**: Controls the number of documents processed simultaneously
2. **Chunk-level concurrency**: Controls the number of chunks processed simultaneously within a single document
3. **LLM request-level concurrency**: Controls the global concurrent number of LLM requests

## 1. Document-Level Concurrent Control

**Control Parameter**: `max_parallel_insert`

Document-level concurrency is controlled by the `max_parallel_insert` parameter, with a default value of 2.

```python
# lightrag/lightrag.py
max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 2)))
```

### Implementation Mechanism

In the `apipeline_process_enqueue_documents` method, a semaphore is used to control document concurrency:

```python
# lightrag/lightrag.py - apipeline_process_enqueue_documents method
async def process_document(
    doc_id: str,
    status_doc: DocProcessingStatus,
    split_by_character: str | None,
    split_by_character_only: bool,
    pipeline_status: dict,
    pipeline_status_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,  # Document-level semaphore
) -> None:
    """Process single document"""
    async with semaphore:  # 🔥 Document-level concurrent control
        # ... Process all chunks of a single document

# Create document-level semaphore
semaphore = asyncio.Semaphore(self.max_parallel_insert)  # Default 2

# Create processing tasks for each document
doc_tasks = []
for doc_id, status_doc in to_process_docs.items():
    doc_tasks.append(
        process_document(
            doc_id, status_doc, split_by_character, split_by_character_only,
            pipeline_status, pipeline_status_lock, semaphore
        )
    )

# Wait for all documents to complete processing
await asyncio.gather(*doc_tasks)
```

## 2. Chunk-Level Concurrent Control

**Control Parameter**: `llm_model_max_async`

**Key Point**: Each document independently creates its own chunk semaphore!

```python
# lightrag/lightrag.py
llm_model_max_async: int = field(default=int(os.getenv("MAX_ASYNC", 4)))
```

### Implementation Mechanism

In the `extract_entities` function, **each document independently creates** its own chunk semaphore:

```python
# lightrag/operate.py - extract_entities function
async def extract_entities(chunks: dict[str, TextChunkSchema], global_config: dict[str, str], ...):
    # 🔥 Key: Each document independently creates this semaphore!
    llm_model_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(llm_model_max_async)  # Chunk semaphore for each document

    async def _process_with_semaphore(chunk):
        async with semaphore:  # 🔥 Chunk concurrent control within document
            return await _process_single_content(chunk)

    # Create tasks for each chunk
    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # Wait for all chunks to complete processing
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    chunk_results = [task.result() for task in tasks]
    return chunk_results
```

### Important Inference: System Overall Chunk Concurrency

Since each document independently creates chunk semaphores, the theoretical chunk concurrency of the system is:

**Theoretical Chunk Concurrency = max_parallel_insert × llm_model_max_async**

For example:
- `max_parallel_insert = 2` (process 2 documents simultaneously)
- `llm_model_max_async = 4` (maximum 4 chunk concurrency per document)
- **Theoretical result**: Maximum 2 × 4 = 8 chunks simultaneously in "processing" state

## 3. LLM Request-Level Concurrent Control (The Real Bottleneck)

**Control Parameter**: `llm_model_max_async` (globally shared)

**Key**: Although there might be 8 chunks "in processing", all LLM requests share the same global priority queue!

```python
# lightrag/lightrag.py - __post_init__ method
self.llm_model_func = priority_limit_async_func_call(self.llm_model_max_async)(
    partial(
        self.llm_model_func,
        hashing_kv=hashing_kv,
        **self.llm_model_kwargs,
    )
)
# 🔥 Global LLM queue size = llm_model_max_async = 4
```

### Priority Queue Implementation

```python
# lightrag/utils.py - priority_limit_async_func_call function
def priority_limit_async_func_call(max_size: int, max_queue_size: int = 1000):
    def final_decro(func):
        queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        tasks = set()

        async def worker():
            """Worker that processes tasks in the priority queue"""
            while not shutdown_event.is_set():
                try:
                    priority, count, future, args, kwargs = await asyncio.wait_for(queue.get(), timeout=1.0)
                    result = await func(*args, **kwargs)  # 🔥 Actual LLM call
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    # Error handling...
                finally:
                    queue.task_done()

        # 🔥 Create fixed number of workers (max_size), this is the real concurrency limit
        for _ in range(max_size):
            task = asyncio.create_task(worker())
            tasks.add(task)
```

## 4. Chunk Internal Processing Mechanism (Serial)

### Why Serial?

Internal processing of each chunk strictly follows this serial execution order:

```python
# lightrag/operate.py - _process_single_content function
async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
    # Step 1: Initial entity extraction
    hint_prompt = entity_extract_prompt.format(**{**context_base, "input_text": content})
    final_result = await use_llm_func_with_cache(hint_prompt, use_llm_func, ...)

    # Process initial extraction results
    maybe_nodes, maybe_edges = await _process_extraction_result(final_result, chunk_key, file_path)

    # Step 2: Gleaning phase
    for now_glean_index in range(entity_extract_max_gleaning):
        # 🔥 Serial wait for gleaning results
        glean_result = await use_llm_func_with_cache(
            continue_prompt, use_llm_func,
            llm_response_cache=llm_response_cache,
            history_messages=history, cache_type="extract"
        )

        # Process gleaning results
        glean_nodes, glean_edges = await _process_extraction_result(glean_result, chunk_key, file_path)

        # Merge results...

        # Step 3: Determine whether to continue loop
        if now_glean_index == entity_extract_max_gleaning - 1:
            break

        # 🔥 Serial wait for loop decision results
        if_loop_result = await use_llm_func_with_cache(
            if_loop_prompt, use_llm_func,
            llm_response_cache=llm_response_cache,
            history_messages=history, cache_type="extract"
        )

        if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
            break

    return maybe_nodes, maybe_edges
```

## 5. Complete Concurrent Hierarchy Diagram
![lightrag_indexing.png](assets%2Flightrag_indexing.png)

### Chunk Internal Processing (Serial)
```
Initial Extraction → Gleaning → Loop Decision → Complete
```

## 6. Real-World Scenario Analysis

### Scenario 1: Single Document with Multiple Chunks
Assume 1 document with 6 chunks:

- **Document level**: Only 1 document, not limited by `max_parallel_insert`
- **Chunk level**: Maximum 4 chunks processed simultaneously (limited by `llm_model_max_async=4`)
- **LLM level**: Global maximum 4 LLM requests concurrent

**Expected behavior**: 4 chunks process concurrently, remaining 2 chunks wait.

### Scenario 2: Multiple Documents with Multiple Chunks
Assume 3 documents, each with 10 chunks:

- **Document level**: Maximum 2 documents processed simultaneously
- **Chunk level**: Maximum 4 chunks per document processed simultaneously
- **Theoretical Chunk concurrency**: 2 × 4 = 8 chunks processed simultaneously
- **Actual LLM concurrency**: Only 4 LLM requests actually execute

**Actual state distribution**:
```
# Possible system state:
Document 1: 4 chunks "processing" (2 executing LLM, 2 waiting for LLM response)
Document 2: 4 chunks "processing" (2 executing LLM, 2 waiting for LLM response)
Document 3: Waiting for document-level semaphore

Total:
- 8 chunks in "processing" state
- 4 LLM requests actually executing
- 4 chunks waiting for LLM response
```

## 7. Performance Optimization Recommendations

### Understanding the Bottleneck

The real bottleneck is the global LLM queue, not the chunk semaphores!

### Adjustment Strategies

**Strategy 1: Increase LLM Concurrent Capacity**

```bash
# Environment variable configuration
export MAX_PARALLEL_INSERT=2    # Keep document concurrency
export MAX_ASYNC=8              # 🔥 Increase LLM request concurrency
```

**Strategy 2: Balance Document and LLM Concurrency**

```python
rag = LightRAG(
    max_parallel_insert=3,      # Moderately increase document concurrency
    llm_model_max_async=12,     # Significantly increase LLM concurrency
    entity_extract_max_gleaning=0,  # Reduce serial steps within chunks
)
```

## 8. Summary

Key characteristics of LightRAG's multi-document concurrent processing mechanism:

### Concurrent Layers
1. **Inter-document competition**: Controlled by `max_parallel_insert`, default 2 documents concurrent
2. **Theoretical Chunk concurrency**: Each document independently creates semaphores, total = max_parallel_insert × llm_model_max_async
3. **Actual LLM concurrency**: All chunks share global LLM queue, controlled by `llm_model_max_async`
4. **Intra-chunk serial**: Multiple LLM requests within each chunk execute strictly serially

### Key Insights
- **Theoretical vs Actual**: System may have many chunks "in processing", but only few are actually executing LLM requests
- **Real Bottleneck**: Global LLM request queue is the performance bottleneck, not chunk semaphores
- **Optimization Focus**: Increasing `llm_model_max_async` is more effective than increasing `max_parallel_insert`
