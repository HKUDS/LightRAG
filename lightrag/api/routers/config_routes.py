import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger


class LangfuseConfigResponse(BaseModel):
    public_key: Optional[str] = None
    secret_key_set: bool = False
    host: Optional[str] = None
    enabled: bool = False


class LangfuseConfigUpdateRequest(BaseModel):
    public_key: Optional[str] = Field(None, description="Langfuse public key")
    secret_key: Optional[str] = Field(None, description="Langfuse secret key")
    host: Optional[str] = Field(None, description="Langfuse host URL")


def update_env_file(updates: dict[str, Optional[str]], env_path: str = ".env") -> None:
    """Update or append key-value pairs in the specified .env file."""
    path = Path(env_path)
    lines: list[str] = []
    keys_written = set()

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                val = updates[key]
                if val is not None and val != "":
                    new_lines.append(f"{key}={val}\n")
                keys_written.add(key)
                continue
        new_lines.append(line)

    for key, val in updates.items():
        if key not in keys_written and val is not None and val != "":
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines.append("\n")
            new_lines.append(f"{key}={val}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def create_config_routes(api_key: Optional[str] = None) -> APIRouter:
    router = APIRouter(prefix="/api/config", tags=["config"])
    auth_dep = get_combined_auth_dependency(api_key)

    @router.get(
        "/langfuse",
        response_model=LangfuseConfigResponse,
        dependencies=[Depends(auth_dep)],
    )
    async def get_langfuse_config():
        pk = os.environ.get("LANGFUSE_PUBLIC_KEY") or ""
        sk = os.environ.get("LANGFUSE_SECRET_KEY") or ""
        host = (
            os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASEURL") or ""
        )

        return LangfuseConfigResponse(
            public_key=pk if pk else None,
            secret_key_set=bool(sk),
            host=host if host else None,
            enabled=bool(pk and sk),
        )

    @router.post(
        "/langfuse",
        response_model=LangfuseConfigResponse,
        dependencies=[Depends(auth_dep)],
    )
    async def update_langfuse_config(request: LangfuseConfigUpdateRequest):
        env_updates: dict[str, Optional[str]] = {}

        if request.public_key is not None:
            os.environ["LANGFUSE_PUBLIC_KEY"] = request.public_key
            env_updates["LANGFUSE_PUBLIC_KEY"] = request.public_key

        if request.secret_key is not None:
            os.environ["LANGFUSE_SECRET_KEY"] = request.secret_key
            env_updates["LANGFUSE_SECRET_KEY"] = request.secret_key

        if request.host is not None:
            os.environ["LANGFUSE_HOST"] = request.host
            env_updates["LANGFUSE_HOST"] = request.host

        try:
            update_env_file(env_updates)
        except Exception as e:
            logger.error(f"Failed to persist Langfuse configuration to .env: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to write configuration to .env: {str(e)}",
            )

        pk = os.environ.get("LANGFUSE_PUBLIC_KEY") or ""
        sk = os.environ.get("LANGFUSE_SECRET_KEY") or ""
        host = os.environ.get("LANGFUSE_HOST") or ""

        return LangfuseConfigResponse(
            public_key=pk if pk else None,
            secret_key_set=bool(sk),
            host=host if host else None,
            enabled=bool(pk and sk),
        )

    return router
