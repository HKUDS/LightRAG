# Nebula Setup Wizard Integration Design

## Problem

NebulaGraphStorage has a complete implementation (1654 lines) and test suite (1462 lines),
but the interactive setup wizard (`make env-storage`) does not include it as an option.
Users must manually edit `.env` to use Nebula. Additionally, `nebula3-python` is incorrectly
listed in core `dependencies` in `pyproject.toml` — all other storage backends are optional extras only.

## Scope

- **In scope**: wizard integration (external cluster only), dependency cleanup, env documentation
- **Out of scope**: Docker Compose template (Nebula clusters require 3+ services — too complex for a single template), connectivity testing during setup

## Changes

### 1. `pyproject.toml` — Dependency Cleanup

Remove `nebula3-python>=3.8.3` from core `dependencies` (line 30).
Keep `nebula3-python>=3.8.3,<4.0.0` in `offline-storage` extras (line 116).

This aligns Nebula with every other storage backend (Neo4j, Redis, Milvus, etc.) which are
all optional-only.

### 2. `scripts/setup/lib/storage_requirements.sh` — Register Backend

Add `NebulaGraphStorage` to three data structures:

```bash
# GRAPH_STORAGE_OPTIONS array — add after MemgraphStorage
"NebulaGraphStorage"

# STORAGE_ENV_REQUIREMENTS map
["NebulaGraphStorage"]="NEBULA_HOSTS NEBULA_USER NEBULA_PASSWORD"

# STORAGE_DB_TYPES map
["NebulaGraphStorage"]="nebula"
```

### 3. `scripts/setup/lib/validation.sh` — Host Validation + Compatibility

Add a `validate_nebula_hosts_format()` function:
- Accepts comma-separated `host:port` entries (e.g. `127.0.0.1:9669` or `host1:9669,host2:9669`)
- Rejects URI schemes (`nebula://...` is not valid — the SDK uses bare host:port)
- Validates port range (1-65535)

Add Nebula-specific storage compatibility warning in `check_storage_compatibility()`:
- When `NebulaGraphStorage` is selected, warn that full-text search quality depends on
  Elasticsearch + Listener; without them, `search_labels` falls back to substring matching.

### 4. `scripts/setup/setup.sh` — Wizard Flow

#### 4a. Constants

Add `"nebula"` to `STORAGE_SERVICES` array (line ~63).

#### 4b. `collect_nebula_config()` Function

Interactive flow (mirrors `collect_neo4j_config` pattern):

1. **NEBULA_HOSTS** — required, default `127.0.0.1:9669`, validated by `validate_nebula_hosts_format`
2. **NEBULA_USER** — required, default `root`
3. **NEBULA_PASSWORD** — allowed empty, default `nebula`
4. **Advanced settings prompt** — `[y/N]`
   - **NEBULA_SPACE_PREFIX** — default `lightrag`
   - **NEBULA_LISTENER_HOSTS** — optional, `host:port` format, for full-text search auto-registration
   - **NEBULA_SSL** — `true`/`false`, default `false`

No `add_docker_service` call — external cluster only.

#### 4c. Switch/Case Updates

All locations that enumerate db types need a `nebula)` branch:
- `determine_compose_overrides()` — no compose overrides for nebula (no Docker)
- `resolve_db_type_collection_order()` — add "nebula" to `db_types` array
- `collect_db_config_by_type()` — route to `collect_nebula_config`
- `restore_storage_docker_services_from_env()` — skip nebula (no Docker service)
- `env_validate_flow()` — add nebula hosts validation
- `env_security_check_flow()` — no special security checks needed for Nebula

#### 4d. Compose Overrides

No compose overrides needed. Nebula uses environment variables directly
(`NEBULA_HOSTS`, `NEBULA_USER`, `NEBULA_PASSWORD`), not a Docker-internal service URI.

### 5. `env.example` — Complete Documentation

Expand the NebulaGraph section (currently lines 650-665) with all supported environment variables:

```ini
### NebulaGraph Configuration
### Select NebulaGraphStorage via env if needed:
# LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage
NEBULA_HOSTS=127.0.0.1:9669
NEBULA_USER=root
### Password may be left empty if your NebulaGraph deployment allows it
NEBULA_PASSWORD=nebula
### Optional: set a dedicated workspace override or space prefix
# NEBULA_WORKSPACE=forced_workspace_name
# NEBULA_SPACE_PREFIX=lightrag
### Optional: space partitions and replica factor for newly created spaces
# NEBULA_SPACE_PARTITIONS=10
# NEBULA_SPACE_REPLICA_FACTOR=1
### Optional but recommended when using Nebula full-text search:
### provide listener host(s) so newly created workspaces can auto-register listeners
# NEBULA_LISTENER_HOSTS=172.28.0.10:9789
### Full-text index quality for search_labels depends on
### NebulaGraph Full-Text service backed by Elasticsearch + Listener.
### Without it, search_labels falls back to substring matching.
### Transport and connection tuning
# NEBULA_SSL=false
# NEBULA_USE_HTTP2=false
# NEBULA_TIMEOUT_MS=60000
# NEBULA_MAX_CONNECTION_POOL_SIZE=100
```

## Non-Goals

- No Docker Compose template — Nebula cluster deployment (graphd + metad + storaged + optional ES + Listener) is too complex for a single wizard-managed template.
- No connectivity test during wizard — would require `nebula3-python` installed at setup time, which may not be the case.
- No migration tooling from other graph backends.
