import ast
import asyncio
import base64
from enum import Enum
import json
import logging
import os
from pathlib import Path
from pdb import pm
import shutil
from typing import Callable, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from lightrag.api.utils_api import get_api_key_dependency
from lightrag.base import  QueryParam
from ascii_colors import trace_exception
from starlette.status import HTTP_403_FORBIDDEN

router = APIRouter(prefix="/workspaces", tags=["workspaces"])

class DataResponse(BaseModel):
    status: str
    message: str
    data: Any

class CreateWorkspaceRequest(BaseModel):
    workspace: str


class UpdateWorkspaceRequest(BaseModel):
    workspace: str

def create_new_workspace_routes(
    args,
    api_key: Optional[str] = None
):
    # Setup logging
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=getattr(logging, args.log_level)
    )

    optional_api_key = get_api_key_dependency(api_key)

    # 获取所有工作空间
    @router.get(
        "/all",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_workspaces():
        try:
            working_dir = args.working_dir
            workspaces = []
            for item_name in os.listdir(working_dir):
                item_path = os.path.join(working_dir, item_name)
                if os.path.isdir(item_path):
                    # 获取文件夹相关信息
                    dir_info = os.stat(item_path)
                    workspaces.append(
                        {
                            "name": base64.urlsafe_b64decode(
                                item_name.encode("utf-8")
                            ).decode("utf-8"),
                            "mtime": dir_info.st_mtime,
                            "birthtime": getattr(
                                dir_info, "st_birthtime", dir_info.st_mtime
                            ),
                        }
                    )
            return DataResponse(
                status="success",
                message="ok",
                data=workspaces,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 获取工作空间详情
    @router.get(
        "/{workspace}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_workspace_detail(workspace: str):
        try:
            workspace_path = Path(
                args.working_dir,
                base64.urlsafe_b64encode(workspace.encode("utf-8")).decode("utf-8"),
            )
            # 如果文件夹已存在，抛出异常
            if not os.path.exists(workspace_path):
                raise FileExistsError(f"Workspace not found: {workspace}")
            # 获取文件夹相关信息
            dir_info = os.stat(workspace_path)
            data = {
                "name": workspace,
                "mtime": dir_info.st_mtime,
                "birthtime": getattr(dir_info, "st_birthtime", dir_info.st_mtime),
            }
            return DataResponse(status="success", message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 创建新的工作空间
    @router.post(
        "",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def create_workspace(request: CreateWorkspaceRequest):
        try:
            workspace = request.workspace
            # 如果为空
            if not workspace:
                raise ValueError("Workspace name is required")

            workspace_path = Path(
                args.working_dir,
                base64.urlsafe_b64encode(workspace.encode("utf-8")).decode("utf-8"),
            )
            # 如果文件夹已存在，抛出异常
            if os.path.exists(workspace_path):
                raise FileExistsError(f"Workspace already exists: {workspace}")
            Path(workspace_path).mkdir(parents=False, exist_ok=True)
            data = {
                "name": workspace,
            }
            return DataResponse(status="success", message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 修改工作空间
    @router.put(
        "/{workspace}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def update_workspace(workspace: str, request: UpdateWorkspaceRequest):
        try:
            new_workspace = request.workspace
            # 如果为空
            if not workspace:
                raise ValueError("Workspace name is required")
            if not new_workspace:
                raise ValueError("New workspace name is required")
            # 如果工作区名称与新工作区名称相同，抛出异常
            if workspace == new_workspace:
                raise ValueError("New workspace name is the same as the original one")
            print("Renaming workspace...")
            print(f"Original workspace: {workspace}")
            print(f"New workspace: {new_workspace}")
            origin_workspace_path = Path(
                args.working_dir,
                base64.urlsafe_b64encode(workspace.encode("utf-8")).decode("utf-8"),
            )

            # 如果文件夹不存在，抛出异常
            if not Path(origin_workspace_path).exists():
                raise FileExistsError(f"Workspace not exists: {workspace}")
            new_workspace_path = Path(
                args.working_dir,
                base64.urlsafe_b64encode(new_workspace.encode("utf-8")).decode("utf-8"),
            )
            # 如果文件夹已存在，抛出异常
            if Path(new_workspace_path).exists():
                raise FileExistsError(f"Workspace '{new_workspace}' already exists")
            # 修改工作空间
            Path(origin_workspace_path).rename(new_workspace_path)
            data = {
                "name": new_workspace,
            }
            return DataResponse(status="success", message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 删除工作空间
    @router.delete("/{workspace}", response_model=DataResponse)
    async def delete_workspace(workspace: str):
        try:
            # 如果为空
            if not workspace:
                raise ValueError("Workspace name is required")
            workspace_path = Path(
                args.working_dir,
                base64.urlsafe_b64encode(workspace.encode("utf-8")).decode("utf-8"),
            )
            # 如果存在 workspace，则删除它
            if Path(workspace_path).exists():
                shutil.rmtree(workspace_path)
            return DataResponse(status="success", message="ok", data=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
