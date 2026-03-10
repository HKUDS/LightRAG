# Interactive Setup With Make Tool

Use the Make targets below to configure and deploy LightRAG with an interactive wizard.
All targets require Bash 4+ and auto-detect the correct interpreter (`/opt/homebrew/bin/bash`,
`/usr/local/bin/bash`, or the system `bash`).

## Quick Reference

| Target | Script flag | Use case |
|---|---|---|
| `make env-base` | `--base` | Configure LLM, embedding, and reranker (run first) |
| `make env-storage` | `--storage` | Configure storage backends and databases |
| `make env-server` | `--server` | Configure server, security, and SSL |
| `make env-validate` | `--validate` | Validate current `.env` |
| `make env-backup` | `--backup` | Backup current `.env` |
| `make help` | `--help` | Show script CLI help |

## Modular Configuration

The three wizards are designed to be run independently and in any order, allowing you to update
one configuration domain without touching the others:

```bash
make env-base     # Step 1: always required — sets up LLM / embedding / reranker
make env-storage  # Step 2: optional — configures storage backends; requires .env from env-base
make env-server   # Step 3: optional — configures port, auth, SSL; requires .env from env-base
```

**Compose file merging:** When a `docker-compose.final.yml` (or a previously-generated compose
file) already exists, each wizard preserves the services belonging to the *other* wizards:

- `env-base` detects and keeps any storage services (postgres, neo4j, …) in the compose file
  while updating vLLM services.
- `env-storage` detects and keeps any vLLM services (vllm-embed, vllm-rerank) while updating
  storage services.
- `env-server` keeps all existing services unchanged and only rebuilds the `lightrag` service's
  environment overrides (port, SSL paths, etc.).

The generated compose file is always written as `docker-compose.final.yml`. Older files named
`docker-compose.development.yml`, `docker-compose.production.yml`, etc. are detected automatically
for backwards compatibility.

## Make Variable Overrides

| Variable | Default | Description |
|---|---|---|
| `SETUP_OPTS` | (empty) | Extra flags passed directly to the setup script |
| `NO_COLOR` | (unset) | Set to `1` to disable colored output |

Examples:

```bash
make env-base SETUP_OPTS=--debug
make env-storage NO_COLOR=1
SETUP_WAIT_TIMEOUT=120 make env-server SETUP_OPTS=--debug
```

## Target Details

### `make env-base` — LLM / Embedding / Reranker

Configures the three inference providers:

1. **LLM** — provider, model, endpoint, API key
2. **Embedding** — choice between a remote provider (OpenAI, Ollama, Jina, …) or a local Docker
   vLLM service (`BAAI/bge-m3` by default). When Docker is chosen, a `vllm-embed` service is added
   to the compose file and `EMBEDDING_BINDING_HOST` is wired to the container hostname.
3. **Reranker** — same choice: remote provider or local Docker vLLM reranker.

**First run:** creates `.env` (and optionally `docker-compose.final.yml`) from scratch.

**Re-run:** loads the existing `.env` as defaults. If a compose file already exists, its storage
services are detected and preserved when the compose file is regenerated.

### `make env-storage` — Storage Backends

Configures which storage backends LightRAG uses and how to reach them. Requires `.env` to exist
(run `make env-base` first).

Prompts for:

- KV, vector, graph, and doc-status storage backend selection
- Connection settings for each required database
- Whether to run each database as a Docker service

When Docker storage services are selected, `docker-compose.final.yml` is generated (or updated).
Any existing vLLM services are detected and preserved in the compose file.

### `make env-server` — Server / Security / SSL

Configures server-level settings. Requires `.env` to exist (run `make env-base` first).

Prompts for:

- Server host and port
- WebUI title and description
- Authentication: AUTH_ACCOUNTS, TOKEN_SECRET, TOKEN_EXPIRE_HOURS, LIGHTRAG_API_KEY, WHITELIST_PATHS
- SSL: certificate and key paths (copied into `./data/certs/` and mounted in the container)

If a compose file already exists, all services are preserved and the `lightrag` container's
environment overrides are updated to reflect the new port and SSL settings.

### `make env-validate` — Validate Existing `.env`

Reads the current `.env` and reports:

- Missing required keys
- Keys present in `.env` that are absent from `env.example`
- Type or format issues (ports, boolean values, etc.)

Does not modify any files.

### `make env-backup` — Backup `.env`

Copies `.env` to `.env.bak.<timestamp>`. Safe to run at any time before a re-run.

### `make help` — Script Help

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

### `docker-compose.final.yml`

Generated by `env-base` or `env-storage` when Docker services are selected. The base
`docker-compose.yml` is never touched.

Previously generated files named `docker-compose.development.yml`, `docker-compose.production.yml`,
or `docker-compose.custom.yml` are automatically detected for backwards compatibility; new runs
write to `docker-compose.final.yml`.

When a local vLLM service is selected, the compose file includes `host.docker.internal` overrides
so the `lightrag` container can reach the vLLM endpoint running on the Docker host — `.env` keeps
`localhost` for direct host access.

## Image Settings

Bundled service images are defined by the Docker Compose templates in
`scripts/setup/templates/*.yml`. The modular setup wizards do not prompt for image overrides and
do not manage image-selection environment variables in `.env`.

## Common Workflows

### First-time local development

```bash
make env-base
# Answer: LLM provider, API key, embedding provider
# docker-compose.final.yml is generated if Docker services are selected
docker compose -f docker-compose.final.yml up -d
lightrag-server
```

### Local development with local models (no API key)

```bash
make env-base
# When prompted "Run embedding model locally via Docker (vLLM)?", answer yes
# When prompted "Run reranker locally via Docker (vLLM)?", answer yes
docker compose -f docker-compose.final.yml up -d   # starts vllm-embed + vllm-rerank
lightrag-server
```

### Adding a database backend after initial setup

```bash
make env-storage
# Select PostgreSQL, answer the connection prompts
# docker-compose.final.yml is updated; existing vLLM services are preserved
docker compose -f docker-compose.final.yml up -d
```

### Production-style deployment with security and SSL

```bash
make env-base
make env-storage   # configure PostgreSQL + Neo4j
make env-server    # set auth tokens, SSL certificate paths
docker compose -f docker-compose.final.yml up -d
```

### Re-running setup after initial configuration

```bash
make env-backup    # save current .env
make env-base      # re-run; existing .env values are shown as defaults
```

### Validating before deployment

```bash
make env-validate
```

## Tips

- Pass `SETUP_OPTS=--debug` to any target for verbose logging: `make env-base SETUP_OPTS=--debug`
- Set `SETUP_WAIT_TIMEOUT=120` to increase the service startup wait (default 60 s).
- Set `NO_COLOR=1` to disable colored output in CI or terminals without color support.
- When exposing PostgreSQL on a custom host port, `.env` keeps `POSTGRES_HOST=localhost` and
  `POSTGRES_PORT=<host-port>` while the compose file overrides the container to `postgres:5432`.
- For GPU vLLM services, set `VLLM_EMBED_DEVICE=cuda` / `VLLM_RERANK_DEVICE=cuda` and the
  corresponding `DTYPE=float16`. Requires the NVIDIA Container Toolkit installed on the host.
- SSL certificates are copied into `./data/certs/` and mounted into the container; `.env` keeps
  the original host paths for direct host usage.
- Each wizard can be re-run independently; current `.env` values are loaded as defaults so you
  only need to change what has actually changed.
