"""
Search routes for BM25 full-text search.

This module provides a direct keyword search endpoint that bypasses the LLM
and returns matching chunks directly. This is complementary to the /query
endpoint which uses semantic RAG.

Use cases:
- /query (existing): Semantic RAG - "What causes X?" -> LLM-generated answer
- /search (this): Keyword search - "Find docs about X" -> Direct chunk results
"""

from typing import Annotated, Any, ClassVar

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.kg.postgres_impl import PostgreSQLDB
from lightrag.utils import logger


class SearchResult(BaseModel):
    """A single search result (chunk match)."""

    id: str = Field(description='Chunk ID')
    full_doc_id: str = Field(description='Parent document ID')
    chunk_order_index: int = Field(description='Position in document')
    tokens: int = Field(description='Token count')
    content: str = Field(description='Chunk content')
    file_path: str | None = Field(default=None, description='Source file path')
    s3_key: str | None = Field(default=None, description='S3 key for source document')
    char_start: int | None = Field(default=None, description='Character offset start in source document')
    char_end: int | None = Field(default=None, description='Character offset end in source document')
    score: float = Field(description='BM25 relevance score')


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    query: str = Field(description='Original search query')
    results: list[SearchResult] = Field(description='Matching chunks')
    count: int = Field(description='Number of results returned')
    workspace: str = Field(description='Workspace searched')

    class Config:
        json_schema_extra: ClassVar[dict[str, Any]] = {
            'example': {
                'query': 'machine learning algorithms',
                'results': [
                    {
                        'id': 'chunk-abc123',
                        'full_doc_id': 'doc-xyz789',
                        'chunk_order_index': 5,
                        'tokens': 250,
                        'content': 'Machine learning algorithms can be categorized into...',
                        'file_path': 's3://lightrag/archive/default/doc-xyz789/report.pdf',
                        's3_key': 'archive/default/doc-xyz789/report.pdf',
                        'score': 0.85,
                    }
                ],
                'count': 1,
                'workspace': 'default',
            }
        }


def create_search_routes(
    kv_storage: Any,
    api_key: str | None = None,
) -> APIRouter:
    """
    Create search routes for BM25 full-text search.

    Args:
        kv_storage: PGKVStorage instance (db accessed lazily at request time)
        api_key: Optional API key for authentication

    Returns:
        FastAPI router with search endpoints
    """
    router = APIRouter(
        prefix='/search',
        tags=['search'],
    )

    optional_api_key = get_combined_auth_dependency(api_key)

    def get_db() -> PostgreSQLDB:
        """Get db lazily - initialized after app startup."""
        db = getattr(kv_storage, 'db', None)
        if db is None:
            raise HTTPException(status_code=503, detail='Database not yet initialized')
        return db

    @router.get(
        '',
        response_model=SearchResponse,
        summary='BM25 keyword search',
        description="""
        Perform BM25-style full-text search on document chunks.

        This endpoint provides direct keyword search without LLM processing.
        It's faster than /query for simple keyword lookups and returns
        matching chunks directly.

        The search uses PostgreSQL's native full-text search with ts_rank
        for relevance scoring.

        **Use cases:**
        - Quick keyword lookups
        - Finding specific terms or phrases
        - Browsing chunks containing specific content
        - Export/citation workflows where you need exact matches

        **Differences from /query:**
        - /query: Semantic search + LLM generation -> Natural language answer
        - /search: Keyword search -> Direct chunk results (no LLM)
        """,
    )
    async def search(
        q: Annotated[str, Query(description='Search query', min_length=1)],
        limit: Annotated[int, Query(description='Max results to return', ge=1, le=100)] = 10,
        workspace: Annotated[str, Query(description='Workspace to search')] = 'default',
        _: Annotated[bool, Depends(optional_api_key)] = True,
    ) -> SearchResponse:
        """Perform BM25 full-text search on chunks."""
        try:
            db = get_db()
            results = await db.full_text_search(
                query=q,
                workspace=workspace,
                limit=limit,
            )

            search_results = [
                SearchResult(
                    id=r.get('id', ''),
                    full_doc_id=r.get('full_doc_id', ''),
                    chunk_order_index=r.get('chunk_order_index', 0),
                    tokens=r.get('tokens', 0),
                    content=r.get('content', ''),
                    file_path=r.get('file_path'),
                    s3_key=r.get('s3_key'),
                    char_start=r.get('char_start'),
                    char_end=r.get('char_end'),
                    score=float(r.get('score', 0)),
                )
                for r in results
            ]

            return SearchResponse(
                query=q,
                results=search_results,
                count=len(search_results),
                workspace=workspace,
            )

        except Exception as e:
            logger.error(f'Search failed: {e}')
            raise HTTPException(status_code=500, detail=f'Search failed: {e}') from e

    return router
