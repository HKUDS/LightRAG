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
| `make env-security-check` | `--security-check` | Audit current `.env` for security risks |
| `make env-backup` | `--backup` | Backup current `.env` |
| `make help` | `--help` | Show script CLI help |

## Modular Configuration

The three wizards are designed to be run independently and in any order, allowing you to update
one configuration domain without touching the others:

```bash
make env-base     # Step 1: always required — sets up LLM / embedding / reranker
make env-storage  # Step 2: optional — configures storage backends; requires .env from env-base
make env-server   # Step 3: optional — configures port, auth, SSL; requires .env from env-base
make env-security-check  # Optional audit: report risky auth / whitelist / secret settings
```

The modular wizards focus on collecting and updating configuration. They do **not** block writes
based on deployment profile or security posture. Use `make env-security-check` to audit the
current `.env` after editing or before exposing the service.

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
for backwards compatibility. If a legacy `.env` still contains `LIGHTRAG_SETUP_PROFILE`, that
value is used only to pick the matching legacy compose file during migration; new `.env` files do
not persist this variable.

**`.env` ownership principle:** The setup wizard does not promise that one `.env` file will be
simultaneously correct for both host startup and Docker Compose startup. Instead, each wizard run
updates `.env` for the runtime you are configuring at that moment, and rerunning the wizard later
may rewrite `.env` again when you switch between host-oriented and compose-oriented settings.

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
make env-security-check NO_COLOR=1
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

On reruns, `env-storage` reloads wizard-only `LIGHTRAG_SETUP_*_DEPLOYMENT=docker`
metadata from `.env` so each database's Docker prompt defaults to the previously
selected deployment mode.

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
- Invalid runtime combinations such as malformed `AUTH_ACCOUNTS`, missing `TOKEN_SECRET` when
  auth is enabled, bad URIs, or invalid ports
- File/path problems such as missing SSL assets

Does not modify any files.

`env-validate` checks whether the configuration is internally consistent and loadable. It does not
enforce a deployment profile or judge whether the resulting setup is sufficiently hardened for
internet exposure.

### `make env-security-check` — Audit Existing `.env`

Reads the current `.env` and reports security risks such as:

- No API protection configured (`AUTH_ACCOUNTS` and `LIGHTRAG_API_KEY` both unset)
- `AUTH_ACCOUNTS` enabled with a missing or default `TOKEN_SECRET`
- `WHITELIST_PATHS` exposing `/api` routes while account-based auth is enabled
- Sensitive values that still contain `${...}` interpolation syntax

The command does not modify any files. It exits with a non-zero status when any security issues are
found, which makes it suitable for CI or pre-deploy checks.

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

The wizard does not guarantee that a single `.env` stays valid for both host startup and Docker
Compose startup forever. Treat `.env` as the runtime configuration produced by the most recent
wizard run. If you later switch from host execution to Docker Compose, or back again, rerun the
relevant setup target so `.env` is rewritten for that target runtime.

The generated `.env` includes `LIGHTRAG_RUNTIME_TARGET=host|compose` near the top as wizard
metadata describing which runtime the current file is meant for.

### `docker-compose.final.yml`

Generated by `env-base` or `env-storage` when Docker services are selected. The base
`docker-compose.yml` is never touched.

Previously generated files named `docker-compose.development.yml`, `docker-compose.production.yml`,
or `docker-compose.custom.yml` are automatically detected for backwards compatibility; new runs
write to `docker-compose.final.yml`. If a legacy `.env` still carries `LIGHTRAG_SETUP_PROFILE`,
the migration step prefers the matching legacy compose file before falling back to the default
search order.

When a local vLLM service is selected, the compose file includes the overrides needed for the
containerized `lightrag` service to reach that endpoint from inside Docker. If you later switch
back to host startup, rerun the relevant setup target so `.env` is rewritten for host execution.

## Image Settings

Bundled service images are defined by the Docker Compose templates in
`scripts/setup/templates/*.yml`. The modular setup wizards do not prompt for image overrides and
do not manage image-selection environment variables in `.env`.

When the wizard includes the bundled Redis service, it stages `./data/config/redis.conf` and mounts
that file into the container. The file is created only if missing, so local edits are preserved on
later setup reruns.

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
make env-security-check
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
make env-security-check
```

## Tips

- Pass `SETUP_OPTS=--debug` to any target for verbose logging: `make env-base SETUP_OPTS=--debug`
- Set `SETUP_WAIT_TIMEOUT=120` to increase the service startup wait (default 60 s).
- Set `NO_COLOR=1` to disable colored output in CI or terminals without color support.
- When exposing PostgreSQL on a custom host port, `.env` keeps `POSTGRES_HOST=localhost` and
  `POSTGRES_PORT=<host-port>` while the compose file overrides the container to `postgres:5432`.
- For GPU vLLM services, set `VLLM_EMBED_DEVICE=cuda` / `VLLM_RERANK_DEVICE=cuda`. On reruns, the
  saved `VLLM_*_DEVICE` value takes precedence over auto-detection so an existing CPU deployment is
  not silently switched to GPU mode.
- SSL certificates are copied into `./data/certs/` for Docker use, but `.env` should still be
  treated as mode-specific output from the latest wizard run rather than a permanent host/compose
  hybrid configuration.
- Each wizard can be re-run independently; current `.env` values are loaded as defaults so you
  only need to change what has actually changed.
