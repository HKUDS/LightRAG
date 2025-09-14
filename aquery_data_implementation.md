# aquery_data 函数实现需求

## 项目背景

为 LightRAG 项目添加一个 aquery_data 方法用于返回与送给LLM完全一致结构化原始数据。

## 目标

1. **逻辑复用**：最大程度复用现有的查询处理逻辑， 复用 `aquery` 的逻辑来实现 aquery_data，仅在调用LLM之前返回查询结果的原始数据
2. **数据一致性**：确保 `aquery_data` 返回的数据与 `aquery` 发送给LLM的数据完全一致, 确保返回的数据与送给LLM的实际情况完全相符合，包括所有token截断和处理步骤
3. **向后兼容**：不影响现有的 `aquery` 功能

## 实现方案

统一通过 `_build_llm_context` 获取LLM上下文和原始数据。修改 `kg_query` 和 `naive_query` 让它们同时返回原始数据和LLM响应, 通过添加 `return_raw_data` 参数来控制底层函数是否调用LLM，这样可以：
- 最小化代码改动
- 保持逻辑一致性
- 确保数据同步更新

## 数据结构设计

### 返回的原始数据结构

```python
{
    "entities": [
        {
            "entity_name": str,
            "entity_type": str,
            "description": str,
            "source_id": str,
            "file_path": str,
            "created_at": int,
            # ... 其他完整字段
        }
    ],
    "relationships": [
        {
            "src_id": str,
            "tgt_id": str,
            "description": str,
            "keywords": str,
            "weight": float,
            "source_id": str,
            "file_path": str,
            # ... 其他完整字段
        }
    ],
    "chunks": [
        {
            "content": str,
            "file_path": str,
            "chunk_id": str,
            # ... 其他完整字段
        }
    ],
    "metadata": {
        "query_mode": str,
        "keywords": {
            "high_level": list[str],
            "low_level": list[str]
        },
        "processing_info": {
            "total_entities_found": int,
            "total_relations_found": int,
            "entities_after_truncation": int,
            "relations_after_truncation": int,
            "merged_chunks_found": int,
            "chunks_after_truncation": int
        }
    }
}
