# Interactive Setup (Make Targets)

Use the Make targets below to configure and deploy LightRAG with an interactive wizard.

## Targets

- `make setup`: Full wizard. Choose development/production/custom and all backends.
- `make setup-quick`: Development preset, minimal prompts (API keys only).
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

## Image Tags

The wizard lists Docker image tags for selected services and lets you override them.
You can also edit these in `.env`:

- `POSTGRES_IMAGE_TAG`
- `NEO4J_IMAGE_TAG`
- `MONGODB_IMAGE_TAG`
- `REDIS_IMAGE_TAG`
- `MILVUS_IMAGE_TAG`
- `QDRANT_IMAGE_TAG`
- `MEMGRAPH_IMAGE_TAG`
- `VLLM_RERANK_IMAGE_TAG`

## Tips

- Add `SETUP_OPTS=--debug` to `make` for debug logging.
- Use `SETUP_WAIT_TIMEOUT=120` to increase the startup wait for dependent services.
- Set `NO_COLOR=1` to disable colored output.
- Choose `vllm` in the rerank prompt to add a local vLLM reranker service to `docker-compose.yml`.
- For GPU setups, set `VLLM_RERANK_DEVICE=cuda` and `VLLM_RERANK_DTYPE=float16` (requires NVIDIA Container Toolkit).
- CPU-only vLLM requires a CPU-compatible image tag (the default image is GPU-only).
