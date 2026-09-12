# Hologres Storage Guide

This guide explains how to configure LightRAG to use Hologres for all four storage roles, how the two graph implementations differ, and how to run the backend's offline and live tests.

It is intended for both operators deploying LightRAG against Hologres and maintainers changing the Hologres backend.

## Contents

- [1. Requirements and installation](#1-requirements-and-installation)
- [2. Quick start](#2-quick-start)
- [3. Environment variables](#3-environment-variables)
- [4. Choosing a graph backend](#4-choosing-a-graph-backend)
- [5. Schema lifecycle and workspace isolation](#5-schema-lifecycle-and-workspace-isolation)
- [6. Python SDK usage](#6-python-sdk-usage)
- [7. Operational notes](#7-operational-notes)
- [8. Testing and coverage](#8-testing-and-coverage)
- [9. Troubleshooting](#9-troubleshooting)

## 1. Requirements and installation

### Server requirements

- Hologres **5.0 or newer**.
- A reachable database that has already been created. LightRAG creates schemas and tables inside the database; it does not create the database instance.
- A user that can:
  - create the configured schema and its migration ledger;
  - create and alter the backend's tables;
  - execute catalog reads used by capability probes and schema verification;
  - create and drop AGE graph objects when `HologresAGEGraphStorage` is selected.

The default four-backend stack is:

| LightRAG role | Implementation |
|---|---|
| KV storage | `HologresKVStorage` |
| Vector storage | `HologresVectorStorage` |
| Graph storage | `HologresGraphStorage` |
| Document status storage | `HologresDocStatusStorage` |

`HologresAGEGraphStorage` is an optional replacement for the graph role.

### Install the Python dependency

Hologres uses `asyncpg`, which is included in the `offline-storage` extra.

From a source checkout:

```bash
uv sync --extra offline-storage
```

For a package deployment:

```bash
pip install lightrag-hku[offline-storage]
```

If you run the API server, use the normal server startup procedure after installing the dependency and writing `.env`.

## 2. Quick start

Add the Hologres connection settings and select all four Hologres implementations in `.env`:

```ini
# Replace these with your instance's values.
HOLOGRES_HOST=example.hologres.aliyuncs.com
HOLOGRES_PORT=80
HOLOGRES_USER=your_username
HOLOGRES_PASSWORD=your_password
HOLOGRES_DATABASE=your_database
HOLOGRES_SCHEMA=lightrag

# Use one logical LightRAG instance/workspace.
WORKSPACE=project_a

# Select the all-in-one Hologres stack.
LIGHTRAG_KV_STORAGE=HologresKVStorage
LIGHTRAG_VECTOR_STORAGE=HologresVectorStorage
LIGHTRAG_GRAPH_STORAGE=HologresGraphStorage
LIGHTRAG_DOC_STATUS_STORAGE=HologresDocStatusStorage
```

Keep your existing LLM and embedding configuration unchanged. In particular, `EMBEDDING_MODEL` and `EMBEDDING_DIM` must match the model used to create the vectors.

To try the AGE-backed graph instead, change only the graph selection:

```ini
LIGHTRAG_GRAPH_STORAGE=HologresAGEGraphStorage
```

See [Choosing a graph backend](#4-choosing-a-graph-backend) before selecting the AGE variant.

## 3. Environment variables

All Hologres settings use the `HOLOGRES_` prefix. The backend does not define `HOLOGRES_WORKSPACE`; it uses LightRAG's common `WORKSPACE`.

### Connection settings

| Variable | Default | Valid values | Purpose |
|---|---:|---|---|
| `HOLOGRES_HOST` | required | non-empty string | Hologres endpoint |
| `HOLOGRES_PORT` | `80` | `1`–`65535` | Hologres port |
| `HOLOGRES_USER` | required | non-empty string | database user |
| `HOLOGRES_PASSWORD` | required | non-empty string | database password |
| `HOLOGRES_DATABASE` | required | non-empty string | existing database |
| `HOLOGRES_SCHEMA` | `public` | SQL identifier, at most 63 bytes | schema holding LightRAG objects |
| `HOLOGRES_SSL_MODE` | `prefer` | `disable`, `allow`, `prefer`, `require`, `verify-ca`, `verify-full` | asyncpg SSL mode |

The schema identifier must start with an ASCII letter or `_`, and its remaining characters must be ASCII letters, digits, or `_`.

### Pool, timeout, and retry settings

| Variable | Default | Valid range | Purpose |
|---|---:|---|---|
| `HOLOGRES_POOL_MIN_SIZE` | `1` | `1`–`1000` | minimum pool size |
| `HOLOGRES_POOL_MAX_SIZE` | `10` | `1`–`1000` | maximum pool size; must be at least the minimum |
| `HOLOGRES_CONNECT_TIMEOUT` | `10` | `0.001`–`600` seconds | connection establishment timeout |
| `HOLOGRES_COMMAND_TIMEOUT` | `60` | `0.001`–`3600` seconds | default SQL command timeout |
| `HOLOGRES_POOL_ACQUIRE_TIMEOUT` | `30` | `0.001`–`600` seconds | wait time to acquire a pooled connection |
| `HOLOGRES_POOL_CLOSE_TIMEOUT` | `5` | `0.001`–`60` seconds | pool close timeout |
| `HOLOGRES_CONNECTION_RETRIES` | `2` | `0`–`20` | connection retries |
| `HOLOGRES_RETRY_BACKOFF` | `0.5` | `0`–`60` seconds | exponential retry backoff |
| `HOLOGRES_STATEMENT_CACHE_SIZE` | `100` | `0`–`10000` | asyncpg statement cache size |

### Capability switches

| Variable | Default | Purpose |
|---|---:|---|
| `HOLOGRES_STREAM_COPY_ENABLED` | `false` | Opt in to stream COPY for qualifying bulk writes. |
| `HOLOGRES_AGE_SEARCH_PATH` | `false` | Internal compatibility switch for a caller-supplied dedicated AGE client. |

Boolean values accept `1/true/yes/on` and `0/false/no/off`, case-insensitively.

Leave `HOLOGRES_AGE_SEARCH_PATH=false` in normal deployments. `HologresAGEGraphStorage` creates its own dedicated client and applies the required AGE search path itself.

### Stream COPY behavior

Stream COPY is deliberately opt-in:

1. `HOLOGRES_STREAM_COPY_ENABLED=true` enables the configuration gate.
2. Initialization runs a non-blocking capability proof.
3. Bulk writes use stream COPY only when both the setting and proof pass.
4. If the proof fails, startup continues and writes use the parameterized-INSERT path.

The switch is a write-path optimization, not a correctness requirement.

## 4. Choosing a graph backend

### `HologresGraphStorage` — default

The default graph backend stores nodes and edges in two shared Hologres tables:

```text
lightrag_hologres_graph_nodes
lightrag_hologres_graph_edges
```

It is the recommended default because it works on the supported Hologres baseline and does not depend on Apache AGE.

Use it when you want:

- the most compatible Hologres graph backend;
- shared tables partitioned by LightRAG workspace;
- predictable behavior without an AGE capability probe.

### `HologresAGEGraphStorage` — optional

The AGE backend uses Hologres' embedded Apache AGE engine:

```ini
LIGHTRAG_GRAPH_STORAGE=HologresAGEGraphStorage
```

During initialization it probes the AGE contract. If the probe passes, LightRAG uses one physical AGE graph per workspace:

```text
lightrag_age_<workspace>
```

If the probe fails, initialization falls back to the two-table `HologresGraphStorage` implementation and delegates graph operations there.

Operational cautions:

- AGE and the two-table backend use different physical storage. Switching between them does not migrate graph data.
- If AGE previously held a workspace's graph but a later startup falls back, queries through the fallback backend will not see the AGE graph's data.
- Use the AGE variant only when you specifically want AGE semantics and can monitor whether its startup probe passed.
- Do not set `HOLOGRES_AGE_SEARCH_PATH=true` manually for the ordinary shared client; the AGE storage manages its dedicated client.

## 5. Schema lifecycle and workspace isolation

### Objects created in the configured schema

On initialization, LightRAG creates the configured schema if it does not exist and applies the required objects.

Fixed object names include:

| Object | Purpose |
|---|---|
| `lightrag_hologres_schema_ledger` | idempotent, resumable migration ledger |
| `lightrag_hologres_kv` | shared KV table |
| `lightrag_hologres_doc_status` | document status table |
| `lightrag_hologres_vectors` | shared vector table |
| `lightrag_hologres_graph_nodes` | two-table graph nodes |
| `lightrag_hologres_graph_edges` | two-table graph edges |

The AGE backend also creates AGE graph namespaces named `lightrag_age_<workspace>`.

### Migration behavior

The schema manager:

- creates the schema and ledger with idempotent single statements;
- records every descriptor's digest and state in the ledger;
- applies additive schema changes in deterministic order;
- handles concurrent initializers with ownership and lease bookkeeping;
- verifies catalog postconditions after applying descriptors;
- fails closed when migration state or catalog shape cannot be trusted.

Do not manually edit these tables to work around a migration failure. Preserve the ledger and server error details, stop other writers, and retry initialization or investigate from the recorded state.

### Workspace isolation

The common `WORKSPACE` setting isolates LightRAG data:

| Backend | Isolation |
|---|---|
| `HologresKVStorage` | `workspace` column |
| `HologresVectorStorage` | `workspace` and namespace columns |
| `HologresDocStatusStorage` | `workspace` column |
| `HologresGraphStorage` | `workspace` and namespace columns |
| `HologresAGEGraphStorage` | one physical AGE graph per workspace |

Workspace names may use ASCII letters, digits, and `_`. Once a LightRAG instance has initialized a workspace, keep its workspace value unchanged.

## 6. Python SDK usage

The same storage names are available through the SDK:

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",
    workspace="project_a",
    kv_storage="HologresKVStorage",
    vector_storage="HologresVectorStorage",
    graph_storage="HologresGraphStorage",
    doc_status_storage="HologresDocStatusStorage",
    # Supply the same LLM, tokenizer, and embedding configuration used by
    # your server deployment.
)

# Mandatory: initialize Hologres clients and reconcile schemas.
await rag.initialize_storages()

try:
    await rag.ainsert("Hologres is a real-time data warehouse.")
finally:
    await rag.finalize_storages()
```

For AGE:

```python
graph_storage="HologresAGEGraphStorage"
```

Connection values still come from the `HOLOGRES_*` environment variables unless you construct and inject backend clients yourself.

## 7. Operational notes

### Embedding model stability

Vector tables are dimension-bound. Use the same embedding model and dimension for indexing and querying.

If the model or dimension changes:

1. stop writers;
2. preserve graph and KV sources;
3. recreate or rebuild vector storage with the new embedding model and dimension;
4. resume queries and ingestion only after the rebuild succeeds.

### Credential safety

Hologres configuration and backend diagnostics redact the host, user, password, and database. Nevertheless, treat `.env` and deployment secrets as sensitive and avoid binding the API server to a non-loopback address without authentication.

### Concurrent writers

Use one `WORKSPACE` per logical knowledge base. Multiple server workers or storage instances may initialize the same schema concurrently; the migration ledger arbitrates additive descriptor application and fails closed rather than guessing when catalog state is inconsistent.

### AGE graph cleanup

Dropping data through `HologresAGEGraphStorage` removes the workspace's graph contents. Live tests additionally drop their randomly created AGE graph namespaces during cleanup so repeated test runs do not accumulate test graphs.

## 8. Testing and coverage

### Offline unit tests

Offline tests use fake clients and do not require a Hologres instance:

```bash
./scripts/test.sh tests/kg/hologres_impl
```

Current expectation: all tests pass, while live-only tests are skipped by default.

### Live integration tests

Live tests require real Hologres credentials in the same `HOLOGRES_*` variables used by the backend:

```bash
export HOLOGRES_HOST=...
export HOLOGRES_PORT=80
export HOLOGRES_USER=...
export HOLOGRES_PASSWORD=...
export HOLOGRES_DATABASE=...

./scripts/test.sh tests/kg/hologres_impl/test_hologres_live.py \
  --run-hologres-live -q
```

The dedicated `--run-hologres-live` option is sufficient; `--run-integration` is not required. The fixture creates isolated random `lightrag_test_*` schemas and cleans them up. It does not use your application workspace.

Live tests can take several minutes because they exercise real capability probes, schema recovery, CRUD, similarity queries, graph behavior, and an AGE cross-chunk hydration regression.

### Coverage

The maintenance threshold for this backend is **at least 90% statement coverage for every module in `lightrag/kg/hologres/`**.

Measure it without adding `coverage` to project dependencies:

```bash
uv run --extra pytest --with coverage coverage run \
  --data-file=/tmp/hologres.coverage \
  --source=lightrag/kg/hologres \
  -m pytest tests/kg/hologres_impl -q

uv run --extra pytest --with coverage coverage report \
  --data-file=/tmp/hologres.coverage -m
```

At commit `2f250901`, the measured statement coverage was:

| Module | Coverage |
|---|---:|
| `__init__.py` | 100.00% |
| `capabilities.py` | 91.83% |
| `client.py` | 91.84% |
| `config.py` | 93.16% |
| `doc_status.py` | 90.81% |
| `graph.py` | 95.25% |
| `graph_age.py` | 92.75% |
| `kv.py` | 90.62% |
| `schema.py` | 94.90% |
| `vector.py` | 90.44% |

These numbers are a point-in-time snapshot. The durable requirement is the 90% per-module threshold.

## 9. Troubleshooting

### Startup reports an invalid Hologres configuration

Check for missing required values, invalid numeric ranges, an invalid SQL schema identifier, or `HOLOGRES_POOL_MAX_SIZE` smaller than `HOLOGRES_POOL_MIN_SIZE`. Error messages identify the offending variable without echoing credentials.

### Startup rejects the Hologres version

The instance must identify itself as Hologres 5.0 or newer. Connect to the intended endpoint and verify its version before retrying.

### A blocking capability probe fails

Blocking probes indicate that the server cannot safely support the SQL or driver behavior required by the backend. Do not bypass them. Use Hologres 5.0+ and capture the probe detail plus server-side diagnostics.

### AGE startup falls back to the two-table backend

This is expected when the AGE probe fails. Decide whether to continue with `HologresGraphStorage` or investigate AGE support on the instance. Do not switch repeatedly if existing graph data is important; the two backends use different storage.

### Schema initialization fails

Stop concurrent writers and retry with the same schema and workspace. If it continues to fail, inspect the `lightrag_hologres_schema_ledger` state and the server error. Do not delete only the tables while preserving application data; the backend fails closed when migration evidence and catalog state disagree.

### Pool acquisition or connection timeouts occur

Check network reachability, Hologres quotas, maximum connections, and credential validity. Increase `HOLOGRES_POOL_ACQUIRE_TIMEOUT` or connection retries only after confirming that the instance is healthy; long timeouts can hide an outage.

### Vector dimension errors occur

Confirm `EMBEDDING_DIM` and the selected embedding model. A mismatch after tables exist normally requires rebuilding vector data with one consistent model and dimension.
