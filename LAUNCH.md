# LightRAG Local Deployment Guide

## Quick Start

```bash
# Clone and install
cd /home/ailab/LightRAG
./install.sh

# Activate environment
source ./activate_venv.sh

# Start services
./start_embedding_vllm.sh &
./start_rerank_vllm.sh
```

## Services

| Service | Port | URL |
|---------|------|-----|
| Embedding API | 8000 | http://localhost:8000 |
| Reranking API | 8001 | http://localhost:8001 |
| LightRAG API | 9621 | http://localhost:9621 |
| Web UI | 5173 | http://localhost:5173/webui/ |

## Configuration

Edit `.env`:
```bash
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_BINDING_HOST=http://localhost:8000
EMBEDDING_BINDING_API_KEY=sk-local-vllm

ENABLE_RERANK=true
RERANK_BINDING=local
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=http://localhost:8001
RERANK_BINDING_API_KEY=sk-local-vllm
```

## Stop Services
```bash
killall vllm
killall lightrag-server
```

## Document Processing
- **Locally:** PDF, Word, Excel, PowerPoint
- **Conversion:** Runs through embedded Python packages

## Version Info
- LightRAG v1.4.11
- vLLM 0.16.0
- BAAI/bge-m3 (embedding)
- BAAI/bge-reranker-v2-m3 (reranking)