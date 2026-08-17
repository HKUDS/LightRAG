from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import List, Dict, Any, Optional, Type
from lightrag.utils import logger
import threading
import time
import json
import re
from enum import Enum
from fastapi.responses import StreamingResponse
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.api.input_limits import count_conversation_input_chars
from lightrag.constants import (
    DEFAULT_QUERY_PRIORITY,
    MAX_IMAGES_PER_MESSAGE,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES_PER_REQUEST,
    MAX_MODEL_NAME_CHARS,
    MAX_QUERY_CHARS,
    MAX_REQUEST_TEXT_CHARS,
    MAX_ROLE_CHARS,
)
from lightrag.utils import TiktokenTokenizer, acount_tokens
from lightrag.api.utils_api import get_combined_auth_dependency, internal_server_error
from fastapi import Depends


# query mode according to query prefix (bypass is not LightRAG quer mode)
class SearchMode(str, Enum):
    naive = "naive"
    local = "local"
    global_ = "global"
    hybrid = "hybrid"
    mix = "mix"
    bypass = "bypass"
    context = "context"


class PayloadTooLargeError(ValueError):
    """Marks a validation failure that should answer 413 rather than 400.

    Raised by the aggregate size checks below. A dedicated type keeps
    ``parse_request_body`` from having to match on error strings, which would
    silently stop working the next time a message is reworded.
    """


def _bound_input_chars(total: int) -> None:
    """Reject an already-measured normalized model-input budget."""
    if total > MAX_REQUEST_TEXT_CHARS:
        raise PayloadTooLargeError(
            f"total request text is {total} characters, over the "
            f"{MAX_REQUEST_TEXT_CHARS} character limit"
        )


class OllamaMessage(BaseModel):
    role: str = Field(max_length=MAX_ROLE_CHARS)
    content: str = Field(max_length=MAX_MESSAGE_CHARS)
    images: Optional[List[str]] = Field(default=None, max_length=MAX_IMAGES_PER_MESSAGE)


def _split_chat_messages(
    messages: List[OllamaMessage],
) -> tuple[str, list[dict[str, str]]]:
    """Return the exact query/history split the chat handler forwards downstream."""
    if not messages:
        return "", []
    return (
        messages[-1].content,
        [
            {"role": message.role, "content": message.content}
            for message in messages[:-1]
        ],
    )


class OllamaChatRequest(BaseModel):
    model: str = Field(max_length=MAX_MODEL_NAME_CHARS)
    messages: List[OllamaMessage] = Field(max_length=MAX_MESSAGES_PER_REQUEST)
    stream: bool = True
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = Field(default=None, max_length=MAX_MESSAGE_CHARS)

    @model_validator(mode="after")
    def _bound_aggregate_text(self) -> "OllamaChatRequest":
        # The final message becomes the query; preceding messages become the
        # history passed to the LLM. Count that exact normalized representation
        # to match QueryRequest's aggregate budget.
        query, history = _split_chat_messages(self.messages)
        _bound_input_chars(count_conversation_input_chars(query, self.system, history))
        return self


class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaMessage
    done: bool


class OllamaGenerateRequest(BaseModel):
    model: str = Field(max_length=MAX_MODEL_NAME_CHARS)
    prompt: str = Field(max_length=MAX_QUERY_CHARS)
    system: Optional[str] = Field(default=None, max_length=MAX_MESSAGE_CHARS)
    stream: bool = False
    options: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _bound_aggregate_text(self) -> "OllamaGenerateRequest":
        _bound_input_chars(
            count_conversation_input_chars(self.prompt, self.system, None)
        )
        return self


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


class OllamaRunningModelDetails(BaseModel):
    parent_model: str
    format: str
    family: str
    families: List[str]
    parameter_size: str
    quantization_level: str


class OllamaRunningModel(BaseModel):
    name: str
    model: str
    size: int
    digest: str
    details: OllamaRunningModelDetails
    expires_at: str
    size_vram: int


class OllamaPsResponse(BaseModel):
    models: List[OllamaRunningModel]


async def parse_request_body(
    request: Request, model_class: Type[BaseModel]
) -> BaseModel:
    """
    Parse request body based on Content-Type header.
    Supports both application/json and application/octet-stream.

    Args:
        request: The FastAPI Request object
        model_class: The Pydantic model class to parse the request into

    Returns:
        An instance of the provided model_class
    """
    content_type = request.headers.get("content-type", "").lower()

    try:
        if content_type.startswith("application/json"):
            # FastAPI already handles JSON parsing for us
            body = await request.json()
        elif content_type.startswith("application/octet-stream"):
            # Manually parse octet-stream as JSON
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode("utf-8"))
        else:
            # Try to parse as JSON for any other content type
            body_bytes = await request.body()
            body = json.loads(body_bytes.decode("utf-8"))

        # Create an instance of the model
        return model_class(**body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except ValidationError as e:
        # Size failures answer 413 so an operator can tell "you sent too much"
        # apart from "you sent the wrong shape". The detail names the offending
        # fields only; echoing the input back would defeat the point of refusing
        # to hold it.
        if _is_size_violation(e):
            fields = ", ".join(
                ".".join(str(part) for part in error["loc"]) or "body"
                for error in e.errors()
            )
            raise HTTPException(
                status_code=413,
                detail=f"Request payload exceeds the allowed size ({fields}).",
            )
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e!s}")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error parsing request body: {str(e)}"
        )


_SIZE_ERROR_TYPES = frozenset({"string_too_long", "too_long"})


def _is_size_violation(error: ValidationError) -> bool:
    for detail in error.errors():
        if detail.get("type") in _SIZE_ERROR_TYPES:
            return True
        if isinstance(detail.get("ctx", {}).get("error"), PayloadTooLargeError):
            return True
    return False


# Built once, on first use. ``TiktokenTokenizer()`` runs
# ``tiktoken.encoding_for_model`` on construction — which used to happen on every
# single token estimate — but building it at import time would move tiktoken's
# BPE fetch into module import, where an offline or air-gapped install would trip
# over it before the server ever handles a request.
_ESTIMATE_TOKENIZER: Optional[TiktokenTokenizer] = None
_ESTIMATE_TOKENIZER_LOCK = threading.Lock()


def _estimate_tokenizer() -> TiktokenTokenizer:
    global _ESTIMATE_TOKENIZER
    if _ESTIMATE_TOKENIZER is None:
        with _ESTIMATE_TOKENIZER_LOCK:
            if _ESTIMATE_TOKENIZER is None:
                _ESTIMATE_TOKENIZER = TiktokenTokenizer()
    return _ESTIMATE_TOKENIZER


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text using tiktoken.

    Synchronous; kept for callers outside the request path. Handlers must use
    :func:`aestimate_tokens` — this is CPU-bound and blocks the event loop for
    roughly half a second per MiB.
    """
    return len(_estimate_tokenizer().encode(text))


async def aestimate_tokens(text: str) -> int:
    """Estimate tokens without occupying the event loop."""
    return await acount_tokens(_estimate_tokenizer(), text)


def parse_query_mode(query: str) -> tuple[str, SearchMode, bool, Optional[str]]:
    """Parse query prefix to determine search mode
    Returns tuple of (cleaned_query, search_mode, only_need_context, user_prompt)

    Examples:
    - "/local[use mermaid format for diagrams] query string" -> (cleaned_query, SearchMode.local, False, "use mermaid format for diagrams")
    - "/[use mermaid format for diagrams] query string" -> (cleaned_query, SearchMode.mix, False, "use mermaid format for diagrams")
    - "/local  query string" -> (cleaned_query, SearchMode.local, False, None)
    - "/local[use mermaid format for diagrams]" -> ("", SearchMode.local, False, "use mermaid format for diagrams")
    """
    # Initialize user_prompt as None
    user_prompt = None

    # First check if there's a bracket format for user prompt. The trailing
    # group spans newlines so a multi-line question is not truncated at the
    # first one; the prompt group stays single-line, as before.
    bracket_pattern = r"^/([a-z]*)\[(.*?)\]([\s\S]*)"
    bracket_match = re.match(bracket_pattern, query)

    if bracket_match:
        mode_prefix = bracket_match.group(1)
        user_prompt = bracket_match.group(2)
        remaining_query = bracket_match.group(3).lstrip()

        # Reconstruct query, removing the bracket part. Keep the separator the
        # space-suffixed mode keys match on, and emit no bare "/" without a mode.
        query = f"/{mode_prefix} {remaining_query}" if mode_prefix else remaining_query

    # Unified handling of mode and only_need_context determination
    mode_map = {
        "/local ": (SearchMode.local, False),
        "/global ": (
            SearchMode.global_,
            False,
        ),  # global_ is used because 'global' is a Python keyword
        "/naive ": (SearchMode.naive, False),
        "/hybrid ": (SearchMode.hybrid, False),
        "/mix ": (SearchMode.mix, False),
        "/bypass ": (SearchMode.bypass, False),
        "/context": (
            SearchMode.mix,
            True,
        ),
        "/localcontext": (SearchMode.local, True),
        "/globalcontext": (SearchMode.global_, True),
        "/hybridcontext": (SearchMode.hybrid, True),
        "/naivecontext": (SearchMode.naive, True),
        "/mixcontext": (SearchMode.mix, True),
    }

    for prefix, (mode, only_need_context) in mode_map.items():
        if query.startswith(prefix):
            # After removing prefix and leading spaces
            cleaned_query = query[len(prefix) :].lstrip()
            return cleaned_query, mode, only_need_context, user_prompt

    return query, SearchMode.mix, False, user_prompt


class OllamaAPI:
    def __init__(self, rag: LightRAG, top_k: int = 60, api_key: Optional[str] = None):
        self.rag = rag
        self.ollama_server_infos = rag.ollama_server_infos
        self.top_k = top_k
        self.api_key = api_key
        self.router = APIRouter(tags=["ollama"])
        self.setup_routes()

    def setup_routes(self):
        # Create combined auth dependency for Ollama API routes
        combined_auth = get_combined_auth_dependency(self.api_key)

        @self.router.get("/version", dependencies=[Depends(combined_auth)])
        async def get_version():
            """Get Ollama version information"""
            return OllamaVersionResponse(version="0.9.3")

        @self.router.get("/tags", dependencies=[Depends(combined_auth)])
        async def get_tags():
            """Return available models acting as an Ollama server"""
            return OllamaTagResponse(
                models=[
                    {
                        "name": self.ollama_server_infos.LIGHTRAG_MODEL,
                        "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                        "modified_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                        "size": self.ollama_server_infos.LIGHTRAG_SIZE,
                        "digest": self.ollama_server_infos.LIGHTRAG_DIGEST,
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

        @self.router.get("/ps", dependencies=[Depends(combined_auth)])
        async def get_running_models():
            """List Running Models - returns currently running models"""
            return OllamaPsResponse(
                models=[
                    {
                        "name": self.ollama_server_infos.LIGHTRAG_MODEL,
                        "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                        "size": self.ollama_server_infos.LIGHTRAG_SIZE,
                        "digest": self.ollama_server_infos.LIGHTRAG_DIGEST,
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": "llama",
                            "families": ["llama"],
                            "parameter_size": "7.2B",
                            "quantization_level": "Q4_0",
                        },
                        "expires_at": "2050-12-31T14:38:31.83753-07:00",
                        "size_vram": self.ollama_server_infos.LIGHTRAG_SIZE,
                    }
                ]
            )

        @self.router.post(
            "/generate", dependencies=[Depends(combined_auth)], include_in_schema=True
        )
        async def generate(raw_request: Request):
            """Handle generate completion requests acting as an Ollama model
            For compatibility purpose, the request is not processed by LightRAG,
            and will be handled by underlying LLM model.
            Supports both application/json and application/octet-stream Content-Types.
            """
            try:
                # Parse the request body manually
                request = await parse_request_body(raw_request, OllamaGenerateRequest)

                query = request.prompt
                start_time = time.time_ns()
                prompt_tokens = await aestimate_tokens(query)

                role_kwargs = (
                    dict(self.rag.role_llm_kwargs["query"])
                    if self.rag.role_llm_kwargs["query"] is not None
                    else dict(self.rag.llm_model_kwargs)
                )
                if request.system:
                    role_kwargs["system_prompt"] = request.system

                if request.stream:
                    response = await (self.rag.role_llm_funcs["query"])(
                        query,
                        stream=True,
                        _priority=DEFAULT_QUERY_PRIORITY,
                        **role_kwargs,
                    )

                    async def stream_generator():
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

                            completion_tokens = await aestimate_tokens(total_response)
                            total_time = last_chunk_time - start_time
                            prompt_eval_time = first_chunk_time - start_time
                            eval_time = last_chunk_time - first_chunk_time

                            data = {
                                "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                "response": "",
                                "done": True,
                                "done_reason": "stop",
                                "context": [],
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

                                logger.error(f"Stream error: {error_msg}")

                                # Send error message to client
                                error_data = {
                                    "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                    "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                    "response": f"\n\nError: {error_msg}",
                                    "error": f"\n\nError: {error_msg}",
                                    "done": False,
                                }
                                yield f"{json.dumps(error_data, ensure_ascii=False)}\n"

                                # Send final message to close the stream
                                final_data = {
                                    "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                    "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                    "response": "",
                                    "done": True,
                                }
                                yield f"{json.dumps(final_data, ensure_ascii=False)}\n"
                                return
                            if first_chunk_time is None:
                                first_chunk_time = start_time
                            completion_tokens = await aestimate_tokens(total_response)
                            total_time = last_chunk_time - start_time
                            prompt_eval_time = first_chunk_time - start_time
                            eval_time = last_chunk_time - first_chunk_time

                            data = {
                                "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                "response": "",
                                "done": True,
                                "done_reason": "stop",
                                "context": [],
                                "total_duration": total_time,
                                "load_duration": 0,
                                "prompt_eval_count": prompt_tokens,
                                "prompt_eval_duration": prompt_eval_time,
                                "eval_count": completion_tokens,
                                "eval_duration": eval_time,
                            }
                            yield f"{json.dumps(data, ensure_ascii=False)}\n"
                            return

                    return StreamingResponse(
                        stream_generator(),
                        media_type="application/x-ndjson",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "Content-Type": "application/x-ndjson",
                            "X-Accel-Buffering": "no",  # Ensure proper handling of streaming responses in Nginx proxy
                        },
                    )
                else:
                    first_chunk_time = time.time_ns()
                    response_text = await (self.rag.role_llm_funcs["query"])(
                        query,
                        stream=False,
                        _priority=DEFAULT_QUERY_PRIORITY,
                        **role_kwargs,
                    )
                    last_chunk_time = time.time_ns()

                    if not response_text:
                        response_text = "No response generated"

                    completion_tokens = await aestimate_tokens(str(response_text))
                    total_time = last_chunk_time - start_time
                    prompt_eval_time = first_chunk_time - start_time
                    eval_time = last_chunk_time - first_chunk_time

                    return {
                        "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                        "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                        "response": str(response_text),
                        "done": True,
                        "done_reason": "stop",
                        "context": [],
                        "total_duration": total_time,
                        "load_duration": 0,
                        "prompt_eval_count": prompt_tokens,
                        "prompt_eval_duration": prompt_eval_time,
                        "eval_count": completion_tokens,
                        "eval_duration": eval_time,
                    }
            except HTTPException:
                # Deliberate client-facing statuses — the 413 of an oversized
                # payload, the 400 of a malformed one — must reach the caller.
                # The catch-all below is for genuinely unexpected failures, and
                # relabelling these as 500 both misleads the client and hides the
                # refusal from anything watching status codes.
                raise
            except Exception as e:
                logger.error(f"Ollama generate error: {str(e)}", exc_info=True)
                raise internal_server_error(e)

        @self.router.post(
            "/chat", dependencies=[Depends(combined_auth)], include_in_schema=True
        )
        async def chat(raw_request: Request):
            """Process chat completion requests by acting as an Ollama model.
            Routes user queries through LightRAG by selecting query mode based on query prefix.
            Detects and forwards OpenWebUI session-related requests (for meta data generation task) directly to LLM.
            Supports both application/json and application/octet-stream Content-Types.
            """
            try:
                # Parse the request body manually
                request = await parse_request_body(raw_request, OllamaChatRequest)

                # Get all messages
                messages = request.messages
                if not messages:
                    raise HTTPException(status_code=400, detail="No messages provided")

                # Validate that the last message is from a user
                if messages[-1].role != "user":
                    raise HTTPException(
                        status_code=400, detail="Last message must be from user role"
                    )

                query, conversation_history = _split_chat_messages(messages)

                # Check for query prefix
                cleaned_query, mode, only_need_context, user_prompt = parse_query_mode(
                    query
                )

                start_time = time.time_ns()
                prompt_tokens = await aestimate_tokens(cleaned_query)

                param_dict = {
                    "mode": mode.value,
                    "stream": request.stream,
                    "only_need_context": only_need_context,
                    "conversation_history": conversation_history,
                    "top_k": self.top_k,
                }

                # Add user_prompt to param_dict
                if user_prompt is not None:
                    param_dict["user_prompt"] = user_prompt

                query_param = QueryParam(**param_dict)

                if request.stream:
                    # Determine if the request is prefix with "/bypass"
                    if mode == SearchMode.bypass:
                        role_kwargs = (
                            dict(self.rag.role_llm_kwargs["query"])
                            if self.rag.role_llm_kwargs["query"] is not None
                            else dict(self.rag.llm_model_kwargs)
                        )
                        if request.system:
                            role_kwargs["system_prompt"] = request.system
                        response = await (self.rag.role_llm_funcs["query"])(
                            cleaned_query,
                            stream=True,
                            history_messages=conversation_history,
                            _priority=DEFAULT_QUERY_PRIORITY,
                            **role_kwargs,
                        )
                    else:
                        response = await self.rag.aquery(
                            cleaned_query, param=query_param
                        )

                    async def stream_generator():
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

                            completion_tokens = await aestimate_tokens(total_response)
                            total_time = last_chunk_time - start_time
                            prompt_eval_time = first_chunk_time - start_time
                            eval_time = last_chunk_time - first_chunk_time

                            data = {
                                "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "images": None,
                                },
                                "done_reason": "stop",
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

                                logger.error(f"Stream error: {error_msg}")

                                # Send error message to client
                                error_data = {
                                    "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                    "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                    "message": {
                                        "role": "assistant",
                                        "content": f"\n\nError: {error_msg}",
                                        "images": None,
                                    },
                                    "error": f"\n\nError: {error_msg}",
                                    "done": False,
                                }
                                yield f"{json.dumps(error_data, ensure_ascii=False)}\n"

                                # Send final message to close the stream
                                final_data = {
                                    "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                    "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                    "message": {
                                        "role": "assistant",
                                        "content": "",
                                        "images": None,
                                    },
                                    "done": True,
                                }
                                yield f"{json.dumps(final_data, ensure_ascii=False)}\n"
                                return

                            if first_chunk_time is None:
                                first_chunk_time = start_time
                            completion_tokens = await aestimate_tokens(total_response)
                            total_time = last_chunk_time - start_time
                            prompt_eval_time = first_chunk_time - start_time
                            eval_time = last_chunk_time - first_chunk_time

                            data = {
                                "model": self.ollama_server_infos.LIGHTRAG_MODEL,
                                "created_at": self.ollama_server_infos.LIGHTRAG_CREATED_AT,
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "images": None,
                                },
                                "done_reason": "stop",
                                "done": True,
                                "total_duration": total_time,
                                "load_duration": 0,
                                "prompt_eval_count": prompt_tokens,
                                "prompt_eval_duration": prompt_eval_time,
                                "eval_count": completion_tokens,
                                "eval_duration": eval_time,
                            }
                            yield f"{json.dumps(data, ensure_ascii=False)}\n"

                    return StreamingResponse(
                        stream_generator(),
                        media_type="application/x-ndjson",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "Content-Type": "application/x-ndjson",
                            "X-Accel-Buffering": "no",  # Ensure proper handling of streaming responses in Nginx proxy
                        },
                    )
                else:
                    first_chunk_time = time.time_ns()

                    # Determine if the request is prefix with "/bypass" or from Open WebUI's session title and session keyword generation task
                    match_result = re.search(
                        r"\n<chat_history>\nUSER:", cleaned_query, re.MULTILINE
                    )
                    if match_result or mode == SearchMode.bypass:
                        role_kwargs = (
                            dict(self.rag.role_llm_kwargs["query"])
                            if self.rag.role_llm_kwargs["query"] is not None
                            else dict(self.rag.llm_model_kwargs)
                        )
                        if request.system:
                            role_kwargs["system_prompt"] = request.system

                        response_text = await (self.rag.role_llm_funcs["query"])(
                            cleaned_query,
                            stream=False,
                            history_messages=conversation_history,
                            _priority=DEFAULT_QUERY_PRIORITY,
                            **role_kwargs,
                        )
                    else:
                        response_text = await self.rag.aquery(
                            cleaned_query, param=query_param
                        )

                    last_chunk_time = time.time_ns()

                    if not response_text:
                        response_text = "No response generated"

                    completion_tokens = await aestimate_tokens(str(response_text))
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
                        "done_reason": "stop",
                        "done": True,
                        "total_duration": total_time,
                        "load_duration": 0,
                        "prompt_eval_count": prompt_tokens,
                        "prompt_eval_duration": prompt_eval_time,
                        "eval_count": completion_tokens,
                        "eval_duration": eval_time,
                    }
            except HTTPException:
                # Deliberate client-facing statuses — the 413 of an oversized
                # payload, the 400 of a malformed one — must reach the caller.
                # The catch-all below is for genuinely unexpected failures, and
                # relabelling these as 500 both misleads the client and hides the
                # refusal from anything watching status codes.
                raise
            except Exception as e:
                logger.error(f"Ollama chat error: {str(e)}", exc_info=True)
                raise internal_server_error(e)
