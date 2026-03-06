#!/bin/bash
# Start vLLM Reranking Service
LIGHTRAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LIGHTRAG_ROOT/activate_venv.sh"
PROJECT_NAME="$(basename $(dirname "${BASH_SOURCE[0]}"))"
mkdir -p "$LIGHTRAG_ROOT/logs"
echo "🚀 Starting vLLM Reranking Service (BAAI/bge-reranker-v2-m3)..."
vllm serve "BAAI/bge-reranker-v2-m3" \
    --host "0.0.0.0" \
    --port 8001 \
    --dtype float16 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.3 \
    --trust-remote-code \
    --disable-log-requests \
    --disable-log-stats &> "$LIGHTRAG_ROOT/logs/rerank-vllm.log" &
echo "✅ Reranking service started on port 8001"
echo "📊 Monitor logs: tail -f $LIGHTRAG_ROOT/logs/rerank-vllm.log"
