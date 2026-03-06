# LightRAG Installation Guide

## Quick Install

```bash
./install.sh
```

This script installs everything needed for LightRAG in 5 steps.

## What Gets Installed

### Virtual Environment
- Location: `$HOME/venv/LightRAG`
- Activated via: `source ./activate_venv.sh`

### Dependencies
- **Core**: LightRAG package (editable mode)
- **LLM**: vLLM 0.16.0 & transformers
- **Compatibility**: OpenAI 1.96.0 (patched for compatibility)
- **Packages**: ~900MB torch download

### Startup Scripts Created

1. `activate_venv.sh` - Activates the virtual environment
2. `start_embedding_vllm.sh` - Starts embedding service on port 8000
3. `start_rerank_vllm.sh` - Starts reranking service on port 8001

## After Installation

```bash
# 1. Activate virtual environment
source ./activate_venv.sh

# 2. Start embedding service
./start_embedding_vllm.sh

# 3. Start reranking service
./start_rerank_vllm.sh

# 4. Run LightRAG
./run.sh
```

## Services

| Service | Port | Model | URL |
|---------|------|-------|-----|
| Embedding | 8000 | BAAI/bge-m3 | http://localhost:8000 |
| Reranking | 8001 | BAAI/bge-reranker-v2-m3 | http://localhost:8001 |
| API | 9621 | - | http://localhost:9621 |
| Web UI | 9622 | - | http://localhost:9622 |

## Troubleshooting

### Stop services
```bash
killall vllm
```

### Monitor logs
```bash
tail -f logs/embedding-vllm.log
tail -f logs/rerank-vllm.log
```

### Clean installation
```bash
rm -rf $HOME/venv/LightRAG
./install.sh
```