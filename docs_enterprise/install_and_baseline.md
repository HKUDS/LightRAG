# Installation And Baseline

## Prerequisites Observed

- `uv` is required by `make dev`.
- `bun` is required by `make dev` for WebUI dependencies and build.
- Python runtime must satisfy `requires-python >=3.10`; `uv` selected Python 3.11.14.
- The interactive setup wizard requires Bash 4+. On this macOS machine, Homebrew
  `bash` was installed because `/bin/bash` was 3.2.

## Commands Used

```bash
git clone https://github.com/HKUDS/LightRAG.git .
make dev
make env-validate
make env-security-check
uv run lightrag-server
curl -sS http://127.0.0.1:9621/health
curl -sS -H 'X-API-Key: dev-local-api-key-change-me' \
  http://127.0.0.1:9621/documents/status_counts
```

## Minimal Local `.env`

```dotenv
LIGHTRAG_RUNTIME_TARGET=host
HOST=127.0.0.1
PORT=9621
WORKSPACE=enterprise_baseline
LOG_LEVEL=INFO
LIGHTRAG_API_KEY=dev-local-api-key-change-me
WHITELIST_PATHS=/health,/docs,/docs/oauth2-redirect,/static/*

LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage
LIGHTRAG_GRAPH_STORAGE=NetworkXStorage
LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage

LLM_BINDING=openai
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=not-a-real-key
LLM_MODEL=gpt-4o-mini

EMBEDDING_BINDING=openai
EMBEDDING_BINDING_HOST=https://api.openai.com/v1
EMBEDDING_BINDING_API_KEY=not-a-real-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_TOKEN_LIMIT=8192
EMBEDDING_SEND_DIM=false

RERANK_BINDING=null
ENABLE_LLM_CACHE=true
ENABLE_LLM_CACHE_FOR_EXTRACT=true
```

The dummy provider keys are sufficient for startup and health checks only. Real
ingestion/query through hosted models requires real provider credentials, or a
local binding such as Ollama/vLLM.
