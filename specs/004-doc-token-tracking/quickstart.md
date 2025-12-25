# Quickstart: Document Token Tracking for Cleo Billing

**Feature**: 004-doc-token-tracking
**For**: Cleo IA integration

## Overview

This feature adds token usage tracking to document processing, enabling Cleo to bill users based on actual resource consumption during RAG indexation.

## Integration Steps

### 1. Upload Document (existing flow)

```bash
curl -X POST "https://your-lightrag-server/documents/text" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "LIGHTRAG-WORKSPACE: user_abc123" \
  -d '{
    "text": "Your document content here...",
    "metadata": {"source": "manual_upload"}
  }'
```

**Response**:
```json
{
  "message": "Text successfully inserted",
  "track_id": "insert-7c3a3abb-5bc1-4d16-821a-36e7cc07ad5b"
}
```

### 2. Poll for Processing Status

```bash
curl -X GET "https://your-lightrag-server/documents/track_status/insert-7c3a3abb-5bc1-4d16-821a-36e7cc07ad5b" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "LIGHTRAG-WORKSPACE: user_abc123"
```

### 3. Check Response for Token Usage

**While processing**:
```json
{
  "track_id": "insert-7c3a3abb-5bc1-4d16-821a-36e7cc07ad5b",
  "documents": [{
    "id": "doc-xxx",
    "status": "processing",
    "metadata": {}
  }],
  "status_summary": {"pending": 0, "processing": 1, "processed": 0, "failed": 0}
}
```

**When processed**:
```json
{
  "track_id": "insert-7c3a3abb-5bc1-4d16-821a-36e7cc07ad5b",
  "documents": [{
    "id": "doc-xxx",
    "status": "processed",
    "chunks_count": 1,
    "metadata": {
      "processing_start_time": 1766671645,
      "processing_end_time": 1766671752,
      "token_usage": {
        "embedding_tokens": 93,
        "llm_input_tokens": 7850,
        "llm_output_tokens": 462,
        "total_chunks": 1,
        "embedding_model": "text-embedding-3-small",
        "llm_model": "openai/gpt-4o-mini"
      }
    }
  }],
  "status_summary": {"pending": 0, "processing": 0, "processed": 1, "failed": 0}
}
```

## Cleo Integration Code

### Python Example

```python
import httpx
import asyncio

async def get_document_billing_data(track_id: str, workspace: str) -> dict:
    """Poll track_status until processed, then extract billing data."""

    async with httpx.AsyncClient() as client:
        while True:
            response = await client.get(
                f"https://your-lightrag-server/documents/track_status/{track_id}",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "LIGHTRAG-WORKSPACE": workspace,
                }
            )
            data = response.json()

            # Check if all documents are processed
            summary = data["status_summary"]
            if summary["pending"] == 0 and summary["processing"] == 0:
                break

            await asyncio.sleep(1)  # Poll every second

        # Aggregate token usage across all documents
        total_embedding = 0
        total_llm_input = 0
        total_llm_output = 0
        total_chunks = 0

        for doc in data["documents"]:
            if doc["status"] == "processed" and doc.get("metadata", {}).get("token_usage"):
                usage = doc["metadata"]["token_usage"]
                total_embedding += usage.get("embedding_tokens", 0)
                total_llm_input += usage.get("llm_input_tokens", 0)
                total_llm_output += usage.get("llm_output_tokens", 0)
                total_chunks += usage.get("total_chunks", 0)

        return {
            "embedding_tokens": total_embedding,
            "llm_input_tokens": total_llm_input,
            "llm_output_tokens": total_llm_output,
            "total_chunks": total_chunks,
        }
```

### Pricing Integration

In Cleo `/admin/pricing`, add these new pricing entries:

| Code | Description | Unit |
|------|-------------|------|
| `rag_embedding` | Embedding tokens | per 1K tokens |
| `rag_llm_input` | LLM input tokens | per 1K tokens |
| `rag_llm_output` | LLM output tokens | per 1K tokens |
| `rag_indexation` | Chunk indexation | per chunk |

### Cost Calculation Example

```python
def calculate_indexation_cost(token_usage: dict, pricing: dict) -> float:
    """Calculate total cost for document indexation."""

    cost = 0.0

    # Embedding cost
    embedding_tokens = token_usage.get("embedding_tokens", 0)
    cost += (embedding_tokens / 1000) * pricing.get("rag_embedding", 0.0001)

    # LLM input cost
    llm_input = token_usage.get("llm_input_tokens", 0)
    cost += (llm_input / 1000) * pricing.get("rag_llm_input", 0.0015)

    # LLM output cost
    llm_output = token_usage.get("llm_output_tokens", 0)
    cost += (llm_output / 1000) * pricing.get("rag_llm_output", 0.002)

    # Chunk indexation cost (optional)
    chunks = token_usage.get("total_chunks", 0)
    cost += chunks * pricing.get("rag_indexation", 0.001)

    return cost
```

## Edge Cases

### Failed Processing

If document processing fails, `token_usage` may contain partial data:

```json
{
  "id": "doc-xxx",
  "status": "failed",
  "error_msg": "Entity extraction failed: API timeout",
  "metadata": {
    "token_usage": {
      "embedding_tokens": 93,
      "llm_input_tokens": 500,
      "llm_output_tokens": 0,
      "total_chunks": 1
    }
  }
}
```

**Recommendation**: Bill for tokens consumed even on failure.

### Local Models

When using local embedding or LLM models, model names will be null:

```json
{
  "token_usage": {
    "embedding_tokens": 0,
    "llm_input_tokens": 0,
    "llm_output_tokens": 0,
    "embedding_model": null,
    "llm_model": null
  }
}
```

**Note**: Local models don't report token usage.

### Empty Documents

Documents with no content produce zero tokens:

```json
{
  "token_usage": {
    "embedding_tokens": 0,
    "llm_input_tokens": 0,
    "llm_output_tokens": 0,
    "total_chunks": 0
  }
}
```

## Verification Checklist

- [ ] Upload a test document
- [ ] Poll track_status until status is "processed"
- [ ] Verify token_usage contains all expected fields
- [ ] Verify embedding_tokens > 0 (for non-empty documents)
- [ ] Verify llm_input_tokens > 0 (for entity extraction)
- [ ] Verify model names match configured models
- [ ] Test with multiple documents in one upload
- [ ] Test with failed processing (verify partial usage)
