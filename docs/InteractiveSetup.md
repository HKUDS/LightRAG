# Interactive Setup With Make Tool

Use the Make targets below to configure and deploy LightRAG with an interactive wizard.
All targets require Bash 4+ and auto-detect the correct interpreter (`/opt/homebrew/bin/bash`,
`/usr/local/bin/bash`, or the system `bash`).

## Quick Reference

| Target | Script flag | Use case |
|---|---|---|
| `make setup` | _(none)_ | Full wizard — choose install type and all backends |
| `make setup-quick` | `--quick` | Development preset, API keys only |
| `make setup-quick-vllm` | `--quick-vllm` | Development preset + local vLLM embedding + optional reranker |
| `make setup-production` | `--production` | Production preset with security and SSL prompts |
| `make setup-validate` | `--validate` | Validate current `.env` |
| `make setup-backup` | `--backup` | Backup current `.env` |
| `make setup-help` | `--help` | Show script CLI help |

## Make Variable Overrides

| Variable | Default | Description |
|---|---|---|
| `SETUP_OPTS` | _(empty)_ | Extra flags passed directly to the setup script |
| `NO_COLOR` | _(unset)_ | Set to `1` to disable colored output |

Examples:

```bash
make setup-production SETUP_OPTS=--debug
make setup-quick NO_COLOR=1
SETUP_WAIT_TIMEOUT=120 make setup-quick-vllm   # Increase startup wait (seconds)
```

## Target Details

### `make setup` — Full Interactive Wizard

Launches the full wizard. The first prompt asks you to choose an install type:

- **development** — Local file-based storage (JSON/NetworkX). Zero external dependencies. Fastest
  path from clone to running.
- **production** — Database-backed storage (PostgreSQL + Neo4j by default). Includes security
  prompts for AUTH_ACCOUNTS, TOKEN_SECRET, LIGHTRAG_API_KEY, and optional SSL.
- **custom** — Manual selection of every storage backend (KV, vector, graph, doc-status).

After install type, the wizard walks through 9–10 steps:

1. LLM provider and API key
2. Embedding provider and model (dimension auto-set where possible)
3. Reranker (none / Jina / vLLM)
4. Storage backends (varies by install type)
5. Docker image versions for selected services
6. Docker Compose generation
7. Security configuration (production only)
8. SSL configuration (production only)
9. Service startup (optional)

**Output files:**

- `.env` — written on host; usable directly for `lightrag-server`
- `docker-compose.development.yml` / `docker-compose.production.yml` / `docker-compose.custom.yml`
  — compose file for the chosen install type; never overwrites `docker-compose.yml`

### `make setup-quick` — Development Preset, Minimal Prompts

Applies a fixed development preset (JSON/NetworkX storage) and then only asks:

- LLM provider / API key
- Embedding provider / API key
- Reranker choice (optional)
- Whether to generate a Docker Compose file

**What is preserved on re-run:** If `.env` already exists with LLM/embedding settings, those
values are kept (fill-only). Storage backends are always reset to the development preset.
Security-specific keys (SSL, AUTH_ACCOUNTS, TOKEN_SECRET, LIGHTRAG_API_KEY, LANGFUSE_*) are
cleared to prevent stale production credentials from leaking into a development environment.

### `make setup-quick-vllm` — Development Preset + Local vLLM Embedding

Same as `setup-quick` but hard-codes the embedding backend to a local vLLM server:

- Embedding: `BAAI/bge-m3` on `http://localhost:8001` (no API key required)
- Reranker: prompted once with `confirm_default_yes` (defaults to enabling it)
  - Reranker model: `BAAI/bge-reranker-v2-m3` on `http://localhost:8002`

A `vllm-embed` (and optionally `vllm-rerank`) service is added to the generated
`docker-compose.development.yml`.

**GPU support:** Set `VLLM_EMBED_DEVICE=cuda` and `VLLM_EMBED_DTYPE=float16` in `.env` before
running, or answer the device prompt in the wizard. CPU mode uses `vllm/vllm-openai-cpu:latest`;
GPU mode uses `vllm/vllm-openai:latest` (requires NVIDIA Container Toolkit).

### `make setup-production` — Production Preset + Security

Applies PostgreSQL + Neo4j storage defaults and then prompts for:

- LLM provider / API key
- Embedding provider / API key
- Reranker
- Docker image versions
- Security: AUTH_ACCOUNTS, TOKEN_SECRET, TOKEN_EXPIRE_HOURS, LIGHTRAG_API_KEY, WHITELIST_PATHS
- SSL: certificate and key paths (copied into `./data/certs/` and mounted in the container)
- Docker Compose generation and optional service startup

### `make setup-validate` — Validate Existing `.env`

Reads the current `.env` and reports:

- Missing required keys
- Keys present in `.env` that are absent from `env.example`
- Type or format issues (ports, boolean values, etc.)

Does not modify any files.

### `make setup-backup` — Backup `.env`

Copies `.env` to `.env.bak.<timestamp>`. Safe to run at any time before a re-run.

### `make setup-help` — Script Help

Prints the setup script's built-in `--help` output listing all supported flags and environment
variables.

## Generated Files

### `.env`

Written to the project root. Contains all runtime configuration for `lightrag-server` and the
Docker services. The wizard uses `env.example` as a template:

- Active (uncommented) keys in `env.example` provide **default values** used when a key is not
  already set.
- Commented keys in `env.example` are activated only when the wizard has a value to write,
  positioned at the template line whose commented value best matches what is being written.

The `.env` file can be used directly on the host (`lightrag-server`) without Docker. When a
Docker Compose file is generated, any container-internal hostnames or SSL paths are injected into
the compose file's `lightrag` service `environment:` block instead of overwriting `.env`.

### `docker-compose.<type>.yml`

Generated alongside `.env`. The suffix matches the install type (`development`, `production`,
`custom`). The base `docker-compose.yml` is never touched.

When a local vLLM service is selected, the compose file includes `host.docker.internal` overrides
so the `lightrag` container can reach the vLLM endpoint running on the Docker host — `.env` keeps
`localhost` for direct host access.

## Image Settings

The wizard lists Docker image tags for selected services and lets you override them before
generating the compose file. You can also edit these directly in `.env`:

| Variable | Default |
|---|---|
| `POSTGRES_IMAGE` | `gzdaniel/postgres-for-rag:16.6` (bundles Apache AGE + pgvector) |
| `NEO4J_IMAGE_TAG` | latest Neo4j tag at release time |
| `MONGODB_IMAGE_TAG` | — |
| `REDIS_IMAGE_TAG` | — |
| `MILVUS_IMAGE_TAG` | — |
| `QDRANT_IMAGE_TAG` | — |
| `MEMGRAPH_IMAGE_TAG` | — |
| `VLLM_EMBED_IMAGE_TAG` | `vllm/vllm-openai-cpu:latest` (CPU) / `vllm/vllm-openai:latest` (GPU) |
| `VLLM_RERANK_IMAGE_TAG` | same as embed |

## Common Workflows

### First-time local development

```bash
make setup-quick
# Answer: LLM provider, API key
# Skip or configure reranker
# docker-compose.development.yml is generated
docker compose -f docker-compose.development.yml up -d
lightrag-server
```

### Local development with local models (no API key)

```bash
make setup-quick-vllm
# Embedding and reranker use local vLLM — no external API key needed
docker compose -f docker-compose.development.yml up -d   # starts vllm-embed + vllm-rerank
lightrag-server
```

### Production deployment

```bash
make setup-production
# Answer: LLM API key, security tokens, optional SSL
docker compose -f docker-compose.production.yml up -d
```

### Re-running setup after initial configuration

```bash
make setup-backup          # save current .env
make setup-quick           # re-run; existing LLM/embedding settings are preserved
```

### Validating before deployment

```bash
make setup-validate
```

## Tips

- Pass `SETUP_OPTS=--debug` to any target for verbose logging: `make setup SETUP_OPTS=--debug`
- Set `SETUP_WAIT_TIMEOUT=120` to increase the service startup wait (default 60 s).
- Set `NO_COLOR=1` to disable colored output in CI or terminals without color support.
- When exposing PostgreSQL on a custom host port, `.env` keeps `POSTGRES_HOST=localhost` and
  `POSTGRES_PORT=<host-port>` while the compose file overrides the container to `postgres:5432`.
- For GPU vLLM services, set `VLLM_EMBED_DEVICE=cuda` / `VLLM_RERANK_DEVICE=cuda` and the
  corresponding `DTYPE=float16`. Requires the NVIDIA Container Toolkit installed on the host.
- SSL certificates are copied into `./data/certs/` and mounted into the container; `.env` keeps
  the original host paths for direct host usage.
- The `configure` Make target is an alias for `make setup`.
