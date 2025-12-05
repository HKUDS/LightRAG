import re
import traceback
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from lightrag import LightRAG
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.kg.postgres_impl import TABLES
from lightrag.utils import logger


def get_order_clause(ddl: str) -> str:
    """Determine the best ORDER BY clause based on available columns in DDL."""
    ddl_lower = ddl.lower()
    if 'update_time' in ddl_lower:
        return 'ORDER BY update_time DESC'
    elif 'updated_at' in ddl_lower:
        return 'ORDER BY updated_at DESC'
    elif 'create_time' in ddl_lower:
        return 'ORDER BY create_time DESC'
    elif 'created_at' in ddl_lower:
        return 'ORDER BY created_at DESC'
    elif 'id' in ddl_lower:
        return 'ORDER BY id ASC'
    return ''


def create_table_routes(rag: LightRAG, api_key: str | None = None) -> APIRouter:
    router = APIRouter(tags=['Tables'])
    combined_auth = get_combined_auth_dependency(api_key)

    def get_workspace_from_request(request: Request) -> str | None:
        workspace = request.headers.get('LIGHTRAG-WORKSPACE', '').strip()
        return workspace if workspace else None

    @router.get('/list', dependencies=[Depends(combined_auth)])
    async def list_tables() -> list[str]:
        """List all available LightRAG tables."""
        try:
            return list(TABLES.keys())
        except Exception as e:
            logger.error(f'Error listing tables: {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error listing tables: {e!s}') from e

    @router.get('/{table_name}/schema', dependencies=[Depends(combined_auth)])
    async def get_table_schema(table_name: str) -> dict[str, Any]:
        """Get DDL/schema for a specific table."""
        if table_name not in TABLES:
            raise HTTPException(status_code=404, detail=f'Table {table_name} not found')
        return TABLES[table_name]

    @router.get('/{table_name}/data', dependencies=[Depends(combined_auth)])
    async def get_table_data(
        request: Request,
        table_name: str,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        workspace: str | None = None,
    ) -> dict[str, Any]:
        """Get paginated data from a table."""
        # Strict validation: table name must be alphanumeric + underscores only
        if not re.match(r'^[a-zA-Z0-9_]+$', table_name):
            raise HTTPException(status_code=400, detail='Invalid table name format')

        if table_name not in TABLES:
            raise HTTPException(status_code=404, detail=f'Table {table_name} not found')

        try:
            req_workspace = get_workspace_from_request(request)
            # Priority: query param > header > rag default > "default"
            target_workspace = workspace or req_workspace or rag.workspace or 'default'

            # Access the database connection from an initialized storage
            # full_docs is a KV storage that has the db connection when using PostgreSQL
            db = getattr(rag.full_docs, 'db', None)
            if db is None:
                raise HTTPException(status_code=500, detail='PostgreSQL storage not available')

            offset = (page - 1) * page_size

            # 1. Get total count
            count_sql = f'SELECT COUNT(*) as count FROM {table_name} WHERE workspace = $1'
            count_res = await db.query(count_sql, [target_workspace])

            total = 0
            if isinstance(count_res, dict):
                total = count_res.get('count', 0)
            elif isinstance(count_res, list) and len(count_res) > 0:
                first_row = count_res[0]
                if isinstance(first_row, dict):
                    total = first_row.get('count', 0)

            # 2. Get data
            # Try to determine order column
            if 'ddl' in TABLES[table_name]:
                ddl = TABLES[table_name]['ddl']
                order_clause = get_order_clause(ddl)
            else:
                order_clause = ''

            sql = f'SELECT * FROM {table_name} WHERE workspace = $1 {order_clause} LIMIT $2 OFFSET $3'
            rows = await db.query(sql, [target_workspace, page_size, offset], multirows=True)

            return {
                'data': rows or [],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size if page_size > 0 else 0,
            }
        except Exception as e:
            logger.error(f'Error fetching table data for {table_name}: {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Database error: {e!s}') from e

    return router
