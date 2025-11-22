#!/usr/bin/env python3
"""
Mock OpenAI-compatible API server for integration testing.

This server mocks OpenAI's API endpoints for:
- Chat completions (LLM)
- Embeddings

Used for integration tests to avoid requiring actual API keys.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict
import numpy as np

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock OpenAI API")


def generate_mock_embedding(text: str, dimensions: int = 3072) -> List[float]:
    """Generate deterministic mock embedding based on text content."""
    # Use hash of text to generate deterministic embeddings
    hash_value = hash(text)
    np.random.seed(abs(hash_value) % (2**32))
    embedding = np.random.randn(dimensions).astype(float)
    # Normalize to unit vector
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.tolist()


def generate_mock_chat_response(messages: List[Dict], model: str = "gpt-5") -> str:
    """Generate mock chat completion response based on the query."""
    # Extract the user's query
    user_query = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    # Generate contextual responses based on keywords
    if "entity" in user_query.lower() or "extract" in user_query.lower():
        # Entity extraction response
        response = json.dumps(
            {
                "entities": [
                    {"entity_name": "SampleClass", "entity_type": "Class"},
                    {"entity_name": "main", "entity_type": "Function"},
                    {"entity_name": "std::cout", "entity_type": "Component"},
                ],
                "relationships": [
                    {
                        "src_id": "main",
                        "tgt_id": "SampleClass",
                        "description": "main function creates and uses SampleClass",
                        "keywords": "instantiation,usage",
                    }
                ],
            }
        )
    elif "summary" in user_query.lower() or "summarize" in user_query.lower():
        response = "This is a sample C++ program that demonstrates basic class usage and console output."
    elif "theme" in user_query.lower():
        response = "The main themes in this code are object-oriented programming, console I/O, and basic C++ syntax."
    elif "describe" in user_query.lower():
        response = "The code defines a simple C++ class with basic functionality and a main function that instantiates and uses the class."
    else:
        # Generic response
        response = f"Mock response for query: {user_query[:100]}"

    return response


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    """Mock chat completions endpoint."""
    try:
        data = await request.json()
        logger.info(f"Received chat completion request: model={data.get('model')}")

        messages = data.get("messages", [])
        model = data.get("model", "gpt-5")
        stream = data.get("stream", False)

        response_text = generate_mock_chat_response(messages, model)

        if stream:
            # Streaming response
            async def generate_stream():
                # Split response into chunks
                words = response_text.split()
                for i, word in enumerate(words):
                    chunk = {
                        "id": f"chatcmpl-mock-{i}",
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": word + " "}
                                if i > 0
                                else {"role": "assistant", "content": word + " "},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)

                # Final chunk
                final_chunk = {
                    "id": "chatcmpl-mock-final",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            # Non-streaming response
            response = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                    "total_tokens": 150,
                },
            }
            return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: Request):
    """Mock embeddings endpoint."""
    try:
        data = await request.json()
        logger.info(f"Received embeddings request: model={data.get('model')}")

        input_texts = data.get("input", [])
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        model = data.get("model", "text-embedding-3-large")
        dimensions = data.get("dimensions", 3072)

        # Generate embeddings for each text
        embeddings_data = []
        for i, text in enumerate(input_texts):
            embedding = generate_mock_embedding(text, dimensions)
            embeddings_data.append(
                {"object": "embedding", "embedding": embedding, "index": i}
            )

        response = {
            "object": "list",
            "data": embeddings_data,
            "model": model,
            "usage": {
                "prompt_tokens": len(input_texts) * 10,
                "total_tokens": len(input_texts) * 10,
            },
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error in embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Run the mock OpenAI server."""
    import argparse

    parser = argparse.ArgumentParser(description="Mock OpenAI API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    logger.info(f"Starting Mock OpenAI API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
