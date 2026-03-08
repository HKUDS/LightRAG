# Interactive Setup (Make Targets)

Use the Make targets below to configure and deploy LightRAG with an interactive wizard.

## Targets

- `make setup`: Full wizard. Choose development/production/custom and all backends.
- `make setup-quick`: Development preset, minimal prompts (API keys only).
- `make setup-quick-vllm`: Development preset + local vLLM embedding (fixed) + optional reranker.
- `make setup-production`: Production preset with security and SSL prompts.
- `make setup-validate`: Validate current `.env`.
- `make setup-backup`: Backup current `.env`.
- `make setup-help`: Show CLI help.

## Install Types

- **development**: Local JSON/NetworkX defaults. Quickest start.
- **production**: Database-backed defaults with security prompts.
- **custom**: Manual selection of each storage backend.

## Compose Output

The wizard writes a dedicated compose file to avoid overwriting `docker-compose.yml`:

- `docker-compose.development.yml`
- `docker-compose.production.yml`
- `docker-compose.custom.yml`

You can let the wizard start the services immediately after generation.
The generated `.env` remains host-usable; any container-only hostnames or SSL paths are injected into `docker-compose.*.yml` under the `lightrag` service.

## Image Settings

The wizard lists Docker image settings for selected services and lets you override them.
You can also edit these in `.env`:

- `POSTGRES_IMAGE`
- `NEO4J_IMAGE_TAG`
- `MONGODB_IMAGE_TAG`
- `REDIS_IMAGE_TAG`
- `MILVUS_IMAGE_TAG`
- `QDRANT_IMAGE_TAG`
- `MEMGRAPH_IMAGE_TAG`
- `VLLM_RERANK_IMAGE_TAG`
- `VLLM_EMBED_IMAGE_TAG`

## Tips

- Add `SETUP_OPTS=--debug` to `make` for debug logging.
- Use `SETUP_WAIT_TIMEOUT=120` to increase the startup wait for dependent services.
- Set `NO_COLOR=1` to disable colored output.
- Choose `vllm` in the rerank prompt to add a local vLLM reranker service to `docker-compose.yml`.
- Use `make setup-quick-vllm` to use `BAAI/bge-m3` via a local vLLM embedding server (port 8001) with no API key required. The reranker defaults to enabled.
- When you expose bundled PostgreSQL on a custom host port, `.env` keeps `POSTGRES_HOST=localhost` and `POSTGRES_PORT=<host-port>` while the generated compose file overrides the container to `postgres:5432`.
- For GPU setups, set `VLLM_RERANK_DEVICE=cuda` and `VLLM_RERANK_DTYPE=float16` (requires NVIDIA Container Toolkit).
- CPU vLLM services use the official `vllm/vllm-openai-cpu:latest` image; GPU mode uses `vllm/vllm-openai:latest`.
- Host-run local model endpoints stay as `localhost` in `.env`; generated Docker stacks inject `host.docker.internal` for the `lightrag` container when needed.
- PostgreSQL defaults to `gzdaniel/postgres-for-rag:16.6`, which bundles both Apache AGE and pgvector for LightRAG's PostgreSQL graph/vector storage modes.
- If you enable SSL in the wizard, the selected certificate and key are copied into `./data/certs/` and mounted into the `lightrag` container from there, while `.env` keeps the original host paths.
