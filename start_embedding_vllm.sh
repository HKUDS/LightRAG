#!/bin/bash
# Start vLLM Embedding Service
LIGHTRAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LIGHTRAG_ROOT/activate_venv.sh"
PROJECT_NAME="$(basename $(dirname "${BASH_SOURCE[0]}"))"
mkdir -p "$LIGHTRAG_ROOT/logs"
echo "🚀 Starting vLLM Embedding Service (BAAI/bge-m3)..."
vllm serve "BAAI/bge-m3" \
    --host "0.0.0.0" \
    --port 8000 \
    --dtype float16 \
    --max-model-len 6144 \
    --gpu-memory-utilization 0.3 \
    --trust-remote-code \
    --disable-log-requests \
    --disable-log-stats &> "$LIGHTRAG_ROOT/logs/embedding-vllm.log" &
echo "✅ Embedding service started on port 8000"
echo "📊 Monitor logs: tail -f $LIGHTRAG_ROOT/logs/embedding-vllm.log"
