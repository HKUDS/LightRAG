from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import os

from starlette.responses import JSONResponse

from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from typing import Optional
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
import sqlite3
# Load environment variables
load_dotenv()

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Default directory for RAG index
DEFAULT_RAG_DIR = "index_default"
WORKING_DIR = os.getenv("RAG_DIR", DEFAULT_RAG_DIR)
UPLOAD_DIR=os.getenv("UPLOAD_DIR")
SQLITE_DIR=os.getenv("SQLITE_DIR")
# Ensure the working directory exists
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(SQLITE_DIR):
    os.makedirs(SQLITE_DIR)
# 数据库初始化
# 创建或连接到 SQLite 数据库文件
conn = sqlite3.connect(SQLITE_DIR+'/filesystem.db')  # 连接到数据库，如果没有会自动创建

# 创建一个 cursor 对象
cursor = conn.cursor()

# 创建一个表格（如果表格已存在，则不创建）
cursor.execute('''
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    embeded INTEGER NOT NULL,
    path TEXT NOT NULL
)
''')

# 提交事务
conn.commit()
cursor.close()

# Initialize FastAPI app
app = FastAPI(title="LightRAG API", description="API for RAG operations")

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains, can be restricted to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Configure working directory
WORKING_DIR = os.environ.get("RAG_DIR", f"{DEFAULT_RAG_DIR}")
print(f"WORKING_DIR: {WORKING_DIR}")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# LLM model function
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "glm-4-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
        **kwargs,
    )

# Embedding function
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICON_API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
    )

# Initialize RAG instance
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024, max_token_size=8192, func=embedding_func
    ),
)

# Data models for API requests
class QueryRequest(BaseModel):
    query: str
    mode: str = "local"

class InsertRequest(BaseModel):
    text: str

class InsertFileRequest(BaseModel):
    filename: str

class Response(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None

# API routes
@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rag.query(
                request.query,
                param=QueryParam(
                    mode=request.mode, only_need_context=request.only_need_context
                ),
            ),
        )
        return Response(status="success", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
@app.get("/getfilelist")
async def getfilelist():
    cursor=conn.cursor()
    cursor.execute("SELECT id,name,embeded FROM files")
    filelist=cursor.fetchall()
    # 格式化查询结果
    files = [{"id": row[0], "name": row[1], "embeded": row[2]} for row in filelist]

    # 返回 JSON 响应
    return JSONResponse(content={"status": "success", "data": files})


@app.post("/insert", response_model=Response)
async def insert_endpoint(request: InsertRequest):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(request.text))
        return Response(status="success", message="Text inserted successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/uploadfile")
async def upload_file(file: UploadFile = File(...)):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM files WHERE name = ?", (file.filename,))
    fileflag = 0 if cursor.fetchall() else 1
    if fileflag:
        try:
            # 生成保存文件的路径
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            file_path=file_path.replace('\\','/')
            print(file.filename)
            # 将文件保存到服务器
            with open(file_path, "wb") as f:
                f.write(await file.read())
            cursor = conn.cursor()
            cursor.execute("INSERT INTO files (name,embeded,path) VALUES (?,?,?)",(file.filename,0,file_path))
            conn.commit()
            cursor.close()
            return Response(status="success", message= file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    else:
        return Response(status="failed",message='files repeating')
@app.post("/insert_file", response_model=Response)
async def insert_file(file: UploadFile = File(...)):
    try:
        filename=request.filename
        cursor=conn.cursor()
        cursor.execute("SELECT path FROM files WHERE name =?",(filename,))
        file_path=cursor.fetchall()
        cursor.close()
        # print(file_path[0][0])
        if not os.path.exists(file_path[0][0]):
            raise HTTPException(
                status_code=404, detail=f"File not found: {file_path[0][0]}"
            )

        # Read file content
        try:
            with open(file_path[0][0], "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path[0][0], "r", encoding="gbk") as f:
                content = f.read()

        # Insert file content into RAG system
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(content))
        cursor=conn.cursor()
        cursor.execute("UPDATE files SET embeded=? WHERE name=?",(1,filename))
        conn.commit()
        cursor.close()

        return Response(
            status="success",
            message=f"File content from {file_path[0][0]} inserted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8020)

# Usage example:
# To run the server, use the following command in your terminal:
# python lightrag_api_openai_compatible_demo.py
