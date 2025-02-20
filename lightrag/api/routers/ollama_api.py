from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import time
import json
import re
from enum import Enum
from fastapi.responses import StreamingResponse
import asyncio
from ascii_colors import trace_exception
from lightrag import LightRAG, QueryParam
from lightrag.utils import encode_string_by_tiktoken
from ..utils_api import ollama_server_infos


# query mode according to query prefix (bypass is not LightRAG quer mode)
class SearchMode(str, Enum):
    naive = "naive"
    local = "local"
    global_ = "global"
    hybrid = "hybrid"
    mix = "mix"
    bypass = "bypass"


class OllamaMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaMessage]
    stream: bool = True
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None


class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaMessage
    done: bool


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    stream: bool = False
    options: Optional[Dict[str, Any]] = None


class OllamaGenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]]
    total_duration: Optional[int]
    load_duration: Optional[int]
    prompt_eval_count: Optional[int]
    prompt_eval_duration: Optional[int]
    eval_count: Optional[int]
    eval_duration: Optional[int]


class OllamaVersionResponse(BaseModel):
    version: str


class OllamaModelDetails(BaseModel):
    parent_model: str
    format: str
    family: str
    families: List[str]
    parameter_size: str
    quantization_level: str


class OllamaModel(BaseModel):
    name: str
    model: str
    size: int
    digest: str
    modified_at: str
    details: OllamaModelDetails


class OllamaTagResponse(BaseModel):
    models: List[OllamaModel]


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text using tiktoken"""
    tokens = encode_string_by_tiktoken(text)
    return len(tokens)


def parse_query_mode(query: str) -> tuple[str, SearchMode]:
    """Parse query prefix to determine search mode
    Returns tuple of (cleaned_query, search_mode)
    """
    mode_map = {
        "/local ": SearchMode.local,
        "/global ": SearchMode.global_,  # global_ is used because 'global' is a Python keyword
        "/naive ": SearchMode.naive,
        "/hybrid ": SearchMode.hybrid,
        "/mix ": SearchMode.mix,
        "/bypass ": SearchMode.bypass,
    }

    for prefix, mode in mode_map.items():
        if query.startswith(prefix):
            # After removing prefix an leading spaces
            cleaned_query = query[len(prefix) :].lstrip()
            return cleaned_query, mode

    return query, SearchMode.hybrid


class OllamaAPI:
    def __init__(self, rag: LightRAG, top_k: int = 60):
        self.rag = rag
        self.ollama_server_infos = ollama_server_infos
        self.top_k = top_k
        self.router = APIRouter(tags=["ollama"])
        self.setup_routes()

    def setup_routes(self):
        @self.router.get("/version")
        async def get_version():
            """Get Ollama version information"""
            return OllamaVersionResponse(version="0.5.4")

        @self.router.get("/tags")
        async def get_tags():
            """Return available models acting as an Ollama server"""
            return OllamaTagResponse(
                models=[
                    {
                        "name": self.ollama_server_infos.LIGHTRAG_MODEL,
                        "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                        "size": self.ollama_server_infos.LIGHTRAG_SIZE,
                        "digest": self.ollama_server_infos.LIGHTRAG_DIGEST,
                        "modified_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": self.ollama_server_infos.LIGHTRAG_NAME,
                            "families": [self.ollama_server_infos.LIGHTRAG_NAME],
                            "parameter_size": "13B",
                            "quantization_level": "Q4_0",
                        },
                    }
                ]
            )

        @self.router.post("/generate")
        async def generate(raw_request: Request, request: OllamaGenerateRequest):
            """Handle generate completion requests acting as an Ollama model
            For compatibility purpose, the request is not processed by LightRAG,
            and will be handled by underlying LLM model.
            """
            try:
                query = request.prompt
                start_time = time.time_ns()
                prompt_tokens = estimate_tokens(query)

                if request.system:
                    self.rag.llm_model_kwargs["system_prompt"] = request.system

                if request.stream:
                    response = await self.rag.llm_model_func(
                        query, stream=True, **self.rag.llm_model_kwargs
                    )

                    async def stream_generator():
                        try:
                            first_chunk_time = None
                            last_chunk_time = time.time_ns()
                            total_response = ""

                            # Ensure response is an async generator
                            if isinstance(response, str):
                                # If it's a string, send in two parts
                                first_chunk_time = start_time
                                last_chunk_time = time.time_ns()
                                total_response = response

                                data = {
                                    "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                    "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                    "response": response,
                                    "done": False,
                                }
                                yield f"{json.dumps(data, ensure_ascii=False)}\n"

                                completion_tokens = estimate_tokens(total_response)
                                total_time = last_chunk_time - start_time
                                prompt_eval_time = first_chunk_time - start_time
                                eval_time = last_chunk_time - first_chunk_time

                                data = {
                                    "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                    "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                    "done": True,
                                    "total_duration": total_time,
                                    "load_duration": 0,
                                    "prompt_eval_count": prompt_tokens,
                                    "prompt_eval_duration": prompt_eval_time,
                                    "eval_count": completion_tokens,
                                    "eval_duration": eval_time,
                                }
                                yield f"{json.dumps(data, ensure_ascii=False)}\n"
                            else:
                                try:
                                    async for chunk in response:
                                        if chunk:
                                            if first_chunk_time is None:
                                                first_chunk_time = time.time_ns()

                                            last_chunk_time = time.time_ns()

                                            total_response += chunk
                                            data = {
                                                "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                                "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                                "response": chunk,
                                                "done": False,
                                            }
                                            yield f"{json.dumps(data, ensure_ascii=False)}\n"
                                except (asyncio.CancelledError, Exception) as e:
                                    error_msg = str(e)
                                    if isinstance(e, asyncio.CancelledError):
                                        error_msg = "Stream was cancelled by server"
                                    else:
                                        error_msg = f"Provider error: {error_msg}"

                                    logging.error(f"Stream error: {error_msg}")

                                    # Send error message to client
                                    error_data = {
                                        "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                        "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                        "response": f"\n\nError: {error_msg}",
                                        "done": False,
                                    }
                                    yield f"{json.dumps(error_data, ensure_ascii=False)}\n"

                                    # Send final message to close the stream
                                    final_data = {
                                        "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                        "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                        "done": True,
                                    }
                                    yield f"{json.dumps(final_data, ensure_ascii=False)}\n"
                                    return
                                if first_chunk_time is None:
                                    first_chunk_time = start_time
                                completion_tokens = estimate_tokens(total_response)
                                total_time = last_chunk_time - start_time
                                prompt_eval_time = first_chunk_time - start_time
                                eval_time = last_chunk_time - first_chunk_time

                                data = {
                                    "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                    "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                    "done": True,
                                    "total_duration": total_time,
                                    "load_duration": 0,
                                    "prompt_eval_count": prompt_tokens,
                                    "prompt_eval_duration": prompt_eval_time,
                                    "eval_count": completion_tokens,
                                    "eval_duration": eval_time,
                                }
                                yield f"{json.dumps(data, ensure_ascii=False)}\n"
                                return

                        except Exception as e:
                            trace_exception(e)
                            raise

                    return StreamingResponse(
                        stream_generator(),
                        media_type="application/x-ndjson",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "Content-Type": "application/x-ndjson",
                            "X-Accel-Buffering": "no",  # 确保在Nginx代理时正确处理流式响应
                        },
                    )
                else:
                    first_chunk_time = time.time_ns()
                    response_text = await self.rag.llm_model_func(
                        query, stream=False, **self.rag.llm_model_kwargs
                    )
                    last_chunk_time = time.time_ns()

                    if not response_text:
                        response_text = "No response generated"

                    completion_tokens = estimate_tokens(str(response_text))
                    total_time = last_chunk_time - start_time
                    prompt_eval_time = first_chunk_time - start_time
                    eval_time = last_chunk_time - first_chunk_time

                    return {
                        "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                        "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                        "response": str(response_text),
                        "done": True,
                        "total_duration": total_time,
                        "load_duration": 0,
                        "prompt_eval_count": prompt_tokens,
                        "prompt_eval_duration": prompt_eval_time,
                        "eval_count": completion_tokens,
                        "eval_duration": eval_time,
                    }
            except Exception as e:
                trace_exception(e)
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/chat")
        async def chat(raw_request: Request, request: OllamaChatRequest):
            """Process chat completion requests acting as an Ollama model
            Routes user queries through LightRAG by selecting query mode based on prefix indicators.
            Detects and forwards OpenWebUI session-related requests (for meta data generation task) directly to LLM.
            """
            try:
                # Get all messages
                messages = request.messages
                if not messages:
                    raise HTTPException(status_code=400, detail="No messages provided")

                # Get the last message as query and previous messages as history
                query = messages[-1].content
                # Convert OllamaMessage objects to dictionaries
                conversation_history = [
                    {"role": msg.role, "content": msg.content} for msg in messages[:-1]
                ]

                # Check for query prefix
                cleaned_query, mode = parse_query_mode(query)

                start_time = time.time_ns()
                prompt_tokens = estimate_tokens(cleaned_query)

                param_dict = {
                    "mode": mode,
                    "stream": request.stream,
                    "only_need_context": False,
                    "conversation_history": conversation_history,
                    "top_k": self.top_k,
                }

                if (
                    hasattr(self.rag, "args")
                    and self.rag.args.history_turns is not None
                ):
                    param_dict["history_turns"] = self.rag.args.history_turns

                query_param = QueryParam(**param_dict)

                if request.stream:
                    # Determine if the request is prefix with "/bypass"
                    if mode == SearchMode.bypass:
                        if request.system:
                            self.rag.llm_model_kwargs["system_prompt"] = request.system
                        response = await self.rag.llm_model_func(
                            cleaned_query,
                            stream=True,
                            history_messages=conversation_history,
                            **self.rag.llm_model_kwargs,
                        )
                    else:
                        response = await self.rag.aquery(
                            cleaned_query, param=query_param
                        )

                    async def stream_generator():
                        try:
                            first_chunk_time = None
                            last_chunk_time = time.time_ns()
                            total_response = ""

                            # Ensure response is an async generator
                            if isinstance(response, str):
                                # If it's a string, send in two parts
                                first_chunk_time = start_time
                                last_chunk_time = time.time_ns()
                                total_response = response

                                data = {
                                    "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                    "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                    "message": {
                                        "role": "assistant",
                                        "content": response,
                                        "images": None,
                                    },
                                    "done": False,
                                }
                                yield f"{json.dumps(data, ensure_ascii=False)}\n"

                                completion_tokens = estimate_tokens(total_response)
                                total_time = last_chunk_time - start_time
                                prompt_eval_time = first_chunk_time - start_time
                                eval_time = last_chunk_time - first_chunk_time

                                data = {
                                    "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                    "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                    "done": True,
                                    "total_duration": total_time,
                                    "load_duration": 0,
                                    "prompt_eval_count": prompt_tokens,
                                    "prompt_eval_duration": prompt_eval_time,
                                    "eval_count": completion_tokens,
                                    "eval_duration": eval_time,
                                }
                                yield f"{json.dumps(data, ensure_ascii=False)}\n"
                            else:
                                try:
                                    async for chunk in response:
                                        if chunk:
                                            if first_chunk_time is None:
                                                first_chunk_time = time.time_ns()

                                            last_chunk_time = time.time_ns()

                                            total_response += chunk
                                            data = {
                                                "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                                "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                                "message": {
                                                    "role": "assistant",
                                                    "content": chunk,
                                                    "images": None,
                                                },
                                                "done": False,
                                            }
                                            yield f"{json.dumps(data, ensure_ascii=False)}\n"
                                except (asyncio.CancelledError, Exception) as e:
                                    error_msg = str(e)
                                    if isinstance(e, asyncio.CancelledError):
                                        error_msg = "Stream was cancelled by server"
                                    else:
                                        error_msg = f"Provider error: {error_msg}"

                                    logging.error(f"Stream error: {error_msg}")

                                    # Send error message to client
                                    error_data = {
                                        "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                        "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                        "message": {
                                            "role": "assistant",
                                            "content": f"\n\nError: {error_msg}",
                                            "images": None,
                                        },
                                        "done": False,
                                    }
                                    yield f"{json.dumps(error_data, ensure_ascii=False)}\n"

                                    # Send final message to close the stream
                                    final_data = {
                                        "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                        "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                        "done": True,
                                    }
                                    yield f"{json.dumps(final_data, ensure_ascii=False)}\n"
                                    return

                                if first_chunk_time is None:
                                    first_chunk_time = start_time
                                completion_tokens = estimate_tokens(total_response)
                                total_time = last_chunk_time - start_time
                                prompt_eval_time = first_chunk_time - start_time
                                eval_time = last_chunk_time - first_chunk_time

                                data = {
                                    "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                    "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                    "done": True,
                                    "total_duration": total_time,
                                    "load_duration": 0,
                                    "prompt_eval_count": prompt_tokens,
                                    "prompt_eval_duration": prompt_eval_time,
                                    "eval_count": completion_tokens,
                                    "eval_duration": eval_time,
                                }
                                yield f"{json.dumps(data, ensure_ascii=False)}\n"

                        except Exception as e:
                            trace_exception(e)
                            raise

                    return StreamingResponse(
                        stream_generator(),
                        media_type="application/x-ndjson",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "Content-Type": "application/x-ndjson",
                            "X-Accel-Buffering": "no",  # 确保在Nginx代理时正确处理流式响应
                        },
                    )
                else:
                    first_chunk_time = time.time_ns()

                    # Determine if the request is prefix with "/bypass" or from Open WebUI's session title and session keyword generation task
                    match_result = re.search(
                        r"\n<chat_history>\nUSER:", cleaned_query, re.MULTILINE
                    )
                    if match_result or mode == SearchMode.bypass:
                        if request.system:
                            self.rag.llm_model_kwargs["system_prompt"] = request.system

                        response_text = await self.rag.llm_model_func(
                            cleaned_query,
                            stream=False,
                            history_messages=conversation_history,
                            **self.rag.llm_model_kwargs,
                        )
                    else:
                        response_text = await self.rag.aquery(
                            cleaned_query, param=query_param
                        )

                    last_chunk_time = time.time_ns()

                    if not response_text:
                        response_text = "No response generated"

                    completion_tokens = estimate_tokens(str(response_text))
                    total_time = last_chunk_time - start_time
                    prompt_eval_time = first_chunk_time - start_time
                    eval_time = last_chunk_time - first_chunk_time

                    return {
                        "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                        "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                        "message": {
                            "role": "assistant",
                            "content": str(response_text),
                            "images": None,
                        },
                        "done": True,
                        "total_duration": total_time,
                        "load_duration": 0,
                        "prompt_eval_count": prompt_tokens,
                        "prompt_eval_duration": prompt_eval_time,
                        "eval_count": completion_tokens,
                        "eval_duration": eval_time,
                    }
            except Exception as e:
                trace_exception(e)
                raise HTTPException(status_code=500, detail=str(e))
