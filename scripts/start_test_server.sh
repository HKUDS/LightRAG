#!/bin/bash
# Start LightRAG server for workspace isolation testing
# Uses default storage (JSON/NanoVectorDB/NetworkX) — no external DB needed

export WORKING_DIR="/tmp/lightrag_ws_test"

# Clean up any previous test data
rm -rf "$WORKING_DIR"
mkdir -p "$WORKING_DIR"

# LLM Configuration (OpenAI-compatible endpoint)
export LLM_BINDING=openai
export LLM_BINDING_HOST=https://llm.daoduc.org/v1
export LLM_MODEL=quick
export LLM_BINDING_API_KEY="${LLM_API_KEY:?LLM_API_KEY is required}"

# Embedding Configuration (OpenAI)
export EMBEDDING_BINDING=openai
export EMBEDDING_BINDING_HOST=https://api.openai.com/v1
export EMBEDDING_MODEL=text-embedding-3-small
export EMBEDDING_BINDING_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
export EMBEDDING_DIM=1536

# Start server
echo "Starting LightRAG server on port 9621..."
echo "Working dir: $WORKING_DIR"
python -m lightrag.api.lightrag_server --port 9621 --working-dir "$WORKING_DIR"
