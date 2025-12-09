"""
Query explain routes for debugging and performance analysis.

This module provides an endpoint for understanding query execution,
including timing breakdown, retrieval statistics, and token usage.

Endpoints:
- POST /query/explain: Execute query with detailed execution breakdown
"""

from __future__ import annotations

import time
from typing import Any, ClassVar, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.base import QueryParam
from lightrag.metrics import record_query_metric
from lightrag.utils import logger


class TimingBreakdown(BaseModel):
    """Timing breakdown in milliseconds."""

    total_ms: float = Field(description='Total query time in ms')
    context_building_ms: float = Field(default=0, description='Time for building context')
    llm_generation_ms: float = Field(default=0, description='Time for LLM response')


class RetrievalStats(BaseModel):
    """Statistics about retrieval results."""

    entities_found: int = Field(description='Number of entities retrieved')
    relations_found: int = Field(description='Number of relations retrieved')
    chunks_found: int = Field(description='Number of chunks retrieved')
    connectivity_passed: bool = Field(description='Whether connectivity check passed')


class TokenStats(BaseModel):
    """Token usage statistics."""

    context_tokens: int = Field(description='Estimated tokens in context')


class ExplainRequest(BaseModel):
    """Request model for query explain endpoint."""

    query: str = Field(min_length=1, description='The query to explain')
    mode: Literal['local', 'global', 'hybrid', 'naive', 'mix'] = Field(
        default='mix', description='Query mode'
    )
    top_k: int | None = Field(default=None, ge=1, description='Number of top items')
    chunk_top_k: int | None = Field(default=None, ge=1, description='Number of chunks')
    only_need_context: bool = Field(
        default=True,
        description='Only retrieve context without LLM generation (faster for explain)',
    )


class ExplainResponse(BaseModel):
    """Response model for query explain endpoint."""

    query: str = Field(description='Original query')
    mode: str = Field(description='Query mode used')
    timing: TimingBreakdown = Field(description='Timing breakdown')
    retrieval: RetrievalStats = Field(description='Retrieval statistics')
    tokens: TokenStats = Field(description='Token statistics')
    context_preview: str | None = Field(
        default=None, description='Preview of retrieved context (first 500 chars)'
    )
    success: bool = Field(description='Whether query returned results')

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            'example': {
                'query': 'What are the key features?',
                'mode': 'mix',
                'timing': {
                    'total_ms': 1234.5,
                    'context_building_ms': 800.0,
                    'llm_generation_ms': 0.0,
                },
                'retrieval': {
                    'entities_found': 15,
                    'relations_found': 23,
                    'chunks_found': 8,
                    'connectivity_passed': True,
                },
                'tokens': {'context_tokens': 4500},
                'context_preview': 'Entity: Example\nDescription: This is...',
                'success': True,
            }
        }


def create_explain_routes(rag, api_key: str | None = None) -> APIRouter:
    """Create explain routes for query debugging.

    Args:
        rag: LightRAG instance
        api_key: Optional API key for authentication

    Returns:
        FastAPI router with explain endpoints
    """
    router = APIRouter(tags=['explain'])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        '/query/explain',
        response_model=ExplainResponse,
        dependencies=[Depends(combined_auth)],
        summary='Explain query execution',
        description=(
            'Execute a query and return detailed execution breakdown including '
            'timing, retrieval statistics, and token usage. By default, only '
            'retrieves context without LLM generation for faster analysis.'
        ),
    )
    async def explain_query(request: ExplainRequest) -> ExplainResponse:
        """Execute query in explain mode and return execution breakdown."""
        start_time = time.perf_counter()

        try:
            # Build query parameters
            param = QueryParam(
                mode=request.mode,
                only_need_context=request.only_need_context,
            )
            if request.top_k:
                param.top_k = request.top_k
            if request.chunk_top_k:
                param.chunk_top_k = request.chunk_top_k

            # Execute query using aquery_data for raw statistics
            context_start = time.perf_counter()
            result = await rag.aquery_data(request.query, param=param)
            context_end = time.perf_counter()

            total_ms = (time.perf_counter() - start_time) * 1000
            context_ms = (context_end - context_start) * 1000

            # Extract statistics from result (aquery_data returns dict directly)
            entities_count = 0
            relations_count = 0
            chunks_count = 0
            context_tokens = 0
            context_preview = None
            success = False

            if result and isinstance(result, dict):
                data = result.get('data', {})
                entities = data.get('entities', [])
                relationships = data.get('relationships', [])
                chunks = data.get('chunks', [])

                entities_count = len(entities)
                relations_count = len(relationships)
                chunks_count = len(chunks)
                success = result.get('status') != 'failure'

                # Estimate context tokens from all retrieved content
                total_content = ''
                for e in entities[:10]:  # Sample first 10
                    total_content += str(e.get('description', ''))
                for r in relationships[:10]:
                    total_content += str(r.get('description', ''))
                for c in chunks[:5]:
                    total_content += str(c.get('content', ''))

                if total_content:
                    # Rough estimate: scale up based on actual counts
                    sample_tokens = len(total_content) // 4
                    scale = max(entities_count / 10, relations_count / 10, chunks_count / 5, 1)
                    context_tokens = int(sample_tokens * scale)
                    context_preview = total_content[:500] + '...' if len(total_content) > 500 else total_content

            # Record metric
            await record_query_metric(
                duration_ms=total_ms,
                mode=request.mode,
                cache_hit=False,
                entities_count=entities_count,
                relations_count=relations_count,
                chunks_count=chunks_count,
                tokens_used=context_tokens,
            )

            return ExplainResponse(
                query=request.query,
                mode=request.mode,
                timing=TimingBreakdown(
                    total_ms=round(total_ms, 2),
                    context_building_ms=round(context_ms, 2),
                    llm_generation_ms=0.0,  # We use only_need_context=True
                ),
                retrieval=RetrievalStats(
                    entities_found=entities_count,
                    relations_found=relations_count,
                    chunks_found=chunks_count,
                    connectivity_passed=success,
                ),
                tokens=TokenStats(context_tokens=context_tokens),
                context_preview=context_preview,
                success=success,
            )

        except Exception as e:
            logger.error(f'Error in query explain: {e}', exc_info=True)
            total_ms = (time.perf_counter() - start_time) * 1000

            return ExplainResponse(
                query=request.query,
                mode=request.mode,
                timing=TimingBreakdown(total_ms=round(total_ms, 2)),
                retrieval=RetrievalStats(
                    entities_found=0,
                    relations_found=0,
                    chunks_found=0,
                    connectivity_passed=False,
                ),
                tokens=TokenStats(context_tokens=0),
                context_preview=f'Error: {str(e)}',
                success=False,
            )

    return router
