from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import inspect
import json
from pydantic import BaseModel
from typing import Optional

import os
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

import nest_asyncio

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:latest",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts=texts, embed_model="bge-m3:latest", host="http://127.0.0.1:11434"
        ),
    ),
)

with open("./book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

app = FastAPI(title="LightRAG", description="LightRAG API open-webui")


# Data models
MODEL_NAME = "LightRAG:latest"


class Message(BaseModel):
    role: Optional[str] = None
    content: str


class OpenWebUIRequest(BaseModel):
    stream: Optional[bool] = None
    model: Optional[str] = None
    messages: list[Message]


# API routes


@app.get("/")
async def index():
    return "Set Ollama link to http://ip:port/ollama in Open-WebUI Settings"


@app.get("/ollama/api/version")
async def ollama_version():
    return {"version": "0.4.7"}


@app.get("/ollama/api/tags")
async def ollama_tags():
    return {
        "models": [
            {
                "name": MODEL_NAME,
                "model": MODEL_NAME,
                "modified_at": "2024-11-12T20:22:37.561463923+08:00",
                "size": 4683087332,
                "digest": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "qwen2",
                    "families": ["qwen2"],
                    "parameter_size": "7.6B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    }


@app.post("/ollama/api/chat")
async def ollama_chat(request: OpenWebUIRequest):
    resp = rag.query(
        request.messages[-1].content, param=QueryParam(mode="hybrid", stream=True)
    )
    if inspect.isasyncgen(resp):

        async def ollama_resp(chunks):
            async for chunk in chunks:
                yield (
                    json.dumps(
                        {
                            "model": MODEL_NAME,
                            "created_at": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%S.%fZ"
                            ),
                            "message": {
                                "role": "assistant",
                                "content": chunk,
                            },
                            "done": False,
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    + b"\n"
                )  # the b"\n" is important

        return StreamingResponse(ollama_resp(resp), media_type="application/json")
    else:
        return resp


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)
