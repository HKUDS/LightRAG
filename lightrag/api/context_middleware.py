import base64
from pathlib import Path
from urllib.parse import unquote
from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from lightrag.lightrag import LightRAG

async def get_rag(request: Request) -> LightRAG:
    return request.state.rag

class ContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app:ASGIApp, initialize_rag, args):
        super().__init__(app)
        self.initialize_rag = initialize_rag
        self.args = args

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        args = self.args
        initialize_rag = self.initialize_rag
        working_dir_header_value = request.headers.get("X-Workspace") or None
        print(f"working_dir_header_value: {working_dir_header_value}")
        working_dir = args.working_dir
        if not working_dir_header_value:
            rag = await initialize_rag(working_dir)
        else:
            working_dir_header_value = unquote(working_dir_header_value)
            workspace_path = Path(
                working_dir,
                base64.urlsafe_b64encode(
                    working_dir_header_value.encode("utf-8")
                ).decode("utf-8"),
            )
            if not Path(workspace_path).exists() or not Path(workspace_path).is_dir():
                raise HTTPException(status_code=404, detail="Workspace not found")
            rag = await initialize_rag(workspace_path)

        request.state.rag = rag

        response = await call_next(request)
        return response
