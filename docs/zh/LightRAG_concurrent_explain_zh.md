# LightRAG 多文档并发控制机制详解

LightRAG 在处理多个文档时采用了多层次的并发控制策略。本文将深入分析文档级别、chunk级别和LLM请求级别的并发控制机制，帮助您理解为什么会出现特定的并发行为。

## 概述

LightRAG 的并发控制分为三个层次：

1. 文档级别并发：控制同时处理的文档数量
2. Chunk级别并发：控制单个文档内同时处理的chunk数量
3. LLM请求级别并发：控制全局LLM请求的并发数量

## 1. 文档级别并发控制

**控制参数**：`max_parallel_insert`

文档级别的并发由 `max_parallel_insert` 参数控制，默认值为2。

```python
# lightrag/lightrag.py
max_parallel_insert: int = field(default=int(os.getenv("MAX_PARALLEL_INSERT", 2)))
```

### 实现机制

在 `apipeline_process_enqueue_documents` 方法中，使用信号量控制文档并发：

```python
# lightrag/lightrag.py - apipeline_process_enqueue_documents方法
async def process_document(
    doc_id: str,
    status_doc: DocProcessingStatus,
    split_by_character: str | None,
    split_by_character_only: bool,
    pipeline_status: dict,
    pipeline_status_lock: asyncio.Lock,
    semaphore: asyncio.Semaphore,  # 文档级别信号量
) -> None:
    """Process single document"""
    async with semaphore:  # 🔥 文档级别并发控制
        # ... 处理单个文档的所有chunks

# 创建文档级别信号量
semaphore = asyncio.Semaphore(self.max_parallel_insert)  # 默认2

# 为每个文档创建处理任务
doc_tasks = []
for doc_id, status_doc in to_process_docs.items():
    doc_tasks.append(
        process_document(
            doc_id, status_doc, split_by_character, split_by_character_only,
            pipeline_status, pipeline_status_lock, semaphore
        )
    )

# 等待所有文档处理完成
await asyncio.gather(*doc_tasks)
```

## 2. Chunk级别并发控制

**控制参数**：`llm_model_max_async`

**关键点**：每个文档都会独立创建自己的chunk信号量！

```python
# lightrag/lightrag.py
llm_model_max_async: int = field(default=int(os.getenv("MAX_ASYNC", 4)))
```

### 实现机制

在 `extract_entities` 函数中，**每个文档独立创建**自己的chunk信号量：

```python
# lightrag/operate.py - extract_entities函数
async def extract_entities(chunks: dict[str, TextChunkSchema], global_config: dict[str, str], ...):
    # 🔥 关键：每个文档都会独立创建这个信号量！
    llm_model_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(llm_model_max_async)  # 每个文档的chunk信号量

    async def _process_with_semaphore(chunk):
        async with semaphore:  # 🔥 文档内部的chunk并发控制
            return await _process_single_content(chunk)

    # 为每个chunk创建任务
    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # 等待所有chunk处理完成
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    chunk_results = [task.result() for task in tasks]
    return chunk_results
```

### 重要推论：系统整体Chunk并发数

由于每个文档独立创建chunk信号量，系统理论上的chunk并发数是：

**理论Chunk并发数 = max_parallel_insert × llm_model_max_async**

例如：
- `max_parallel_insert = 2`（同时处理2个文档）
- `llm_model_max_async = 4`（每个文档最多4个chunk并发）
- 理论结果：最多 2 × 4 = 8个chunk同时处于"处理中"状态

## 3. LLM请求级别并发控制（真正的瓶颈）

**控制参数**：`llm_model_max_async`（全局共享）

**关键**：尽管可能有8个chunk在"处理中"，但所有LLM请求共享同一个全局优先级队列！

```python
# lightrag/lightrag.py - __post_init__方法
self.llm_model_func = priority_limit_async_func_call(self.llm_model_max_async)(
    partial(
        self.llm_model_func,
        hashing_kv=hashing_kv,
        **self.llm_model_kwargs,
    )
)
# 🔥 全局LLM队列大小 = llm_model_max_async = 4
```

### 优先级队列实现

```python
# lightrag/utils.py - priority_limit_async_func_call函数
def priority_limit_async_func_call(max_size: int, max_queue_size: int = 1000):
    def final_decro(func):
        queue = asyncio.PriorityQueue(maxsize=max_queue_size)
        tasks = set()

        async def worker():
            """Worker that processes tasks in the priority queue"""
            while not shutdown_event.is_set():
                try:
                    priority, count, future, args, kwargs = await asyncio.wait_for(queue.get(), timeout=1.0)
                    result = await func(*args, **kwargs)  # 🔥 实际LLM调用
                    if not future.done():
                        future.set_result(result)
                except Exception as e:
                    # 错误处理...
                finally:
                    queue.task_done()

        # 🔥 创建固定数量的worker（max_size个），这是真正的并发限制
        for _ in range(max_size):
            task = asyncio.create_task(worker())
            tasks.add(task)
```

## 4. Chunk内部处理机制（串行）

### 为什么是串行？

每个chunk内部的处理严格按照以下顺序串行执行：

```python
# lightrag/operate.py - _process_single_content函数
async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
    # 步骤1：初始实体提取
    hint_prompt = entity_extract_prompt.format(**{**context_base, "input_text": content})
    final_result = await use_llm_func_with_cache(hint_prompt, use_llm_func, ...)

    # 处理初始提取结果
    maybe_nodes, maybe_edges = await _process_extraction_result(final_result, chunk_key, file_path)

    # 步骤2：Gleaning（深挖）阶段
    for now_glean_index in range(entity_extract_max_gleaning):
        # 🔥 串行等待gleaning结果
        glean_result = await use_llm_func_with_cache(
            continue_prompt, use_llm_func,
            llm_response_cache=llm_response_cache,
            history_messages=history, cache_type="extract"
        )

        # 处理gleaning结果
        glean_nodes, glean_edges = await _process_extraction_result(glean_result, chunk_key, file_path)

        # 合并结果...

        # 步骤3：判断是否继续循环
        if now_glean_index == entity_extract_max_gleaning - 1:
            break

        # 🔥 串行等待循环判断结果
        if_loop_result = await use_llm_func_with_cache(
            if_loop_prompt, use_llm_func,
            llm_response_cache=llm_response_cache,
            history_messages=history, cache_type="extract"
        )

        if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
            break

    return maybe_nodes, maybe_edges
```

## 5. 完整的并发层次图
![lightrag_indexing.png](..%2Fassets%2Flightrag_indexing.png)


## 6. 实际运行场景分析

### 场景1：单文档多Chunk
假设有1个文档，包含6个chunks：

- 文档级别：只有1个文档，不受 `max_parallel_insert` 限制
- Chunk级别：最多4个chunks同时处理（受 `llm_model_max_async=4` 限制）
- LLM级别：全局最多4个LLM请求并发

**预期行为**：4个chunks并发处理，剩余2个chunks等待。

### 场景2：多文档多Chunk
假设有3个文档，每个文档包含10个chunks：

- 文档级别：最多2个文档同时处理
- Chunk级别：每个文档最多4个chunks同时处理
- 理论Chunk并发：2 × 4 = 8个chunks同时处理
- 实际LLM并发：只有4个LLM请求真正执行

**实际状态分布**：
```
# 可能的系统状态：
文档1: 4个chunks"处理中"（其中2个在执行LLM，2个在等待LLM响应）
文档2: 4个chunks"处理中"（其中2个在执行LLM，2个在等待LLM响应）
文档3: 等待文档级别信号量

总计：
- 8个chunks处于"处理中"状态
- 4个LLM请求真正执行
- 4个chunks等待LLM响应
```

## 7. 性能优化建议

### 理解瓶颈

**真正的瓶颈是全局LLM队列，而不是chunk信号量！**

### 调整策略

**策略1：提高LLM并发能力**

```bash
# 环境变量配置
export MAX_PARALLEL_INSERT=2    # 保持文档并发
export MAX_ASYNC=8              # 🔥 增加LLM请求并发数
```

**策略2：平衡文档和LLM并发**

```python
rag = LightRAG(
    max_parallel_insert=3,      # 适度增加文档并发
    llm_model_max_async=12,     # 大幅增加LLM并发
    entity_extract_max_gleaning=0,  # 减少chunk内串行步骤
)
```

## 8. 总结

LightRAG的多文档并发处理机制的关键特点：

### 并发层次
1. **文档间争抢**：受 `max_parallel_insert` 控制，默认2个文档并发
2. **理论Chunk并发**：每个文档独立创建信号量，总数 = `max_parallel_insert × llm_model_max_async`
3. **实际LLM并发**：所有chunk共享全局LLM队列，受 `llm_model_max_async` 控制
4. **单Chunk内串行**：每个chunk内的多个LLM请求严格串行执行

### 关键洞察
- **理论vs实际**：系统可能有很多chunk在"处理中"，但只有少数在真正执行LLM请求
- **真正瓶颈**：全局LLM请求队列是性能瓶颈，而不是chunk信号量
- **优化重点**：提高 `llm_model_max_async` 比增加 `max_parallel_insert` 更有效
