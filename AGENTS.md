# Repository Guidelines

## Project Overview

LightRAG is a Retrieval-Augmented Generation (RAG) framework that uses graph-based knowledge representation for enhanced information retrieval. The system extracts entities and relationships from documents, builds a knowledge graph, and uses multiple retrieval modes (`local`, `global`, `hybrid`, `mix`, `naive`) for queries.

## Project Structure

Top-level directories:

- **lightrag/**: Core Python package — see *Module Layout* below.
- **lightrag_webui/**: React 19 + TypeScript client (Bun + Vite + Tailwind). UI components in `src/`.
- **scripts/**: `test.sh` (preferred test runner), `setup/` interactive environment wizard (use `make env-*` rather than calling `setup.sh` directly — see *Configuration > Setup Wizard Outputs*), and release tooling.
- **tests/**: Pytest coverage, organized into subdirectories that mirror `lightrag/` (see *Testing* below for layout). Working datasets stay in `inputs/`, `rag_storage/`, and `temp/`; deployment collateral lives in `docs/`, `k8s-deploy/`, and compose files.

### Module Layout (`lightrag/`)

- **lightrag.py**: Main orchestrator class (`LightRAG`) — assembled from mixins (see *LightRAG class composition*). Hosts `ainsert_custom_kg`, `_insert_done`, `_process_extract_entities`, `_refresh_addon_params_cache`, and `addon_params` accessors. Critical: always call `await rag.initialize_storages()` after instantiation.
- **pipeline.py**: `_PipelineMixin` — owns the document ingestion pipeline (`apipeline_enqueue_documents`, `apipeline_process_enqueue_documents`, `apipeline_process_error_documents`), the `parse_native` / `parse_mineru` / `parse_docling` parser dispatchers, multimodal analysis, validation, and the worker scaffolding.
- **utils_pipeline.py**: Pure helpers shared by the pipeline mixin and other entry points: doc-status field access, document identity (source key, content hash), parsed-artifact path resolution, parser payload normalization, multimodal entity augmentation, and `make_lightrag_doc_content`.
- **llm_roles.py**: `RoleSpec` / `RoleLLMConfig` / `_RoleLLMState` / `ROLES` registry plus `_RoleLLMMixin` — role normalization, builder registration, wrapper rebuild, runtime config update, queue cleanup, sanitized config export, queue status reporting. Route role-specific behavior here rather than into provider modules.
- **storage_migrations.py**: `_StorageMigrationMixin` — `check_and_migrate_data`, `_migrate_entity_relation_data`, `_migrate_chunk_tracking_storage`.
- **addon_params.py**: `ObservableAddonParams` plus `default_addon_params` / `normalize_addon_params` helpers.
- **operate.py**: Core extraction and query operations including entity/relation extraction, chunking, and multi-mode retrieval logic.
- **base.py**: Abstract base classes for storage backends (`BaseKVStorage`, `BaseVectorStorage`, `BaseGraphStorage`, `BaseDocStatusStorage`).
- **kg/**: Storage implementations (JSON, NetworkX, Neo4j, PostgreSQL, MongoDB, Redis, Milvus, Qdrant, Faiss, Memgraph, OpenSearch, NanoVectorDB). The backend registry (`STORAGE_IMPLEMENTATIONS` / `STORAGES`) lives in `kg/__init__.py`; `kg/factory.py::get_storage_class()` resolves backend classes from configuration.
- **llm/**: LLM and embedding provider bindings (OpenAI, Ollama, Azure, Gemini, Bedrock, Anthropic, etc.). All async with caching support.
- **parser/**: Unified parsing layer. `parser/routing.py` resolves engine and filename hints for `legacy`, `native`, `mineru`, and `docling` flows; `parser/debug.py` provides an offline LightRAG stub for the `parser/cli.py` debug entry point (`python -m lightrag.parser.cli`). Native format parsers live as sibling sub-packages under `parser/` (currently `parser/docx/`); external HTTP-based adapters live under `parser/external/` (`mineru`, `docling`) with shared helpers in `parser/external/_common.py`, `_manifest.py`, `_zip.py`.
- **chunker/**: Chunking strategies (token-size, recursive character, semantic vector, paragraph semantic).
- **api/**: FastAPI service (`lightrag_server.py`) with REST endpoints and Ollama-compatible API; routers under `routers/`, static Swagger assets, packaged WebUI output, and Gunicorn launcher.

## Core Architecture

### LightRAG class composition

`LightRAG` is assembled from focused mixins (split out of the previously monolithic `lightrag.py`):

```
LightRAG → _RoleLLMMixin → _StorageMigrationMixin → _PipelineMixin → object
```

The `@final` decorator on `LightRAG` is preserved — the mixin layering is an internal implementation detail, not an external subclassing surface. The public API (`ainsert`, `aquery`, `ainsert_custom_kg`, `initialize_storages`, etc.) is unchanged. `ainsert_custom_kg` and its internal construction logic, `_insert_done`, `_process_extract_entities`, `_refresh_addon_params_cache`, and the `addon_params` property accessors stay on `LightRAG` itself because they cut across multiple flows or depend on prompt-profile state.

### Storage Layer

LightRAG uses 4 storage types with pluggable backends:
- **KV_STORAGE**: LLM response cache, text chunks, document info
- **VECTOR_STORAGE**: Entity/relation/chunk embeddings
- **GRAPH_STORAGE**: Entity-relation graph structure
- **DOC_STATUS_STORAGE**: Document processing status tracking

Each `LightRAG` instance can pass a `workspace` parameter for data isolation. Implementation differs per storage type:
- **File-based**: subdirectories under `working_dir`.
- **Collection-based**: collection name prefixes.
- **Relational DB**: workspace column filtering.
- **Qdrant**: payload-based partitioning.

### Pipeline concurrency contract

The document ingestion pipeline coordinates concurrent writers through `pipeline_status` (a per-workspace shared dict in `lightrag.kg.shared_storage`). These fields are mutated under `get_namespace_lock("pipeline_status", workspace=...)`:

- **`busy`**: any pipeline-busy state. Set by both the processing loop AND destructive jobs (clear / per-doc delete). On its own, `busy=True` does NOT block enqueue — see `destructive_busy` for the exclusive subset.
- **`destructive_busy`**: the busy job is `/documents/clear` or `/documents/{doc_id}` (delete). These DROP storages and remove input files; a concurrent enqueue accepted in this window would write to storage being torn down and silently lose the document. Reservation and the enqueue last-line guard reject when this is True.
- **`scanning`**: a `/documents/scan` task is running (whole lifecycle: classification + processing). Used by the `/scan` endpoint to refuse overlapping scans. Does NOT on its own block uploads/inserts.
- **`scanning_exclusive`**: True only during the scan task's classification phase, when `run_scanning_process` is reading `doc_status` to classify files (PROCESSED → archive, FAILED-without-`full_docs` → retry-as-new, etc.) and possibly deleting stale stubs. Reservation and the enqueue last-line guard reject when this is set. Cleared before the scan transitions to its processing phase, allowing concurrent uploads to land while scan-driven processing finishes.
- **`pending_enqueues`**: count of `/upload`, `/text`, `/texts` endpoints that have reserved a slot (via `_reserve_enqueue_slot`) but whose bg task has not yet completed. Only the scan endpoint reads this — to refuse starting while uploads are mid-flight.
- **`request_pending`**: a nudge to the running processing loop. Set by either (a) `apipeline_process_enqueue_documents` when called while `busy=True` or (b) `apipeline_enqueue_documents` after writing to `doc_status` while `busy=True`. The loop checks it after each batch and re-queries `doc_status` if set.

Mutual-exclusion rules (all checked atomically inside the lock):

| Operation | Refuses if | Writes |
|---|---|---|
| `_reserve_enqueue_slot` | `scanning_exclusive` or `destructive_busy` | `pending_enqueues++` |
| `apipeline_enqueue_documents` (last-line guard) | (`scanning_exclusive` and not `from_scan`) or `destructive_busy` | — |
| Scan endpoint reservation | `busy or scanning or pending_enqueues > 0` | `scanning = True` |
| `apipeline_process_enqueue_documents` entry | (already busy → set `request_pending`, return) | `busy = True` (NOT `destructive_busy`) |
| `clear_documents` / `delete_document` (synchronous reservation) | `busy or scanning or pending_enqueues > 0` | `busy = True`, `destructive_busy = True` |

The contract permits **concurrent enqueue + processing**: a freshly-uploaded doc lands in `doc_status` while the loop is mid-batch, the loop sees `request_pending` after the current batch, re-queries `doc_status`, and picks up the new PENDING row.

For the rest — write ordering of `full_docs` vs `doc_status`, the workspace-scoped `enqueue_serialize` lock around dedup-and-upsert, and the `from_scan=True` bypass — see the docstrings on `apipeline_enqueue_documents` and `apipeline_process_enqueue_documents` in `lightrag/pipeline.py`.

### Query Modes

- **local**: Context-dependent retrieval focused on specific entities
- **global**: Community/summary-based broad knowledge retrieval
- **hybrid**: Combines local and global
- **naive**: Direct vector search without graph
- **mix**: Integrates KG and vector retrieval (recommended with reranker)

## Development Commands

### Setup
```bash
# Install with uv
uv sync
source .venv/bin/activate  # Or: .venv\Scripts\activate on Windows

# Install with API support
uv sync --extra api

# Install specific extras
uv sync --extra offline-storage  # Storage backends
uv sync --extra offline-llm      # LLM providers
uv sync --extra test             # Testing dependencies
```

### API Server
```bash
# Copy and configure environment
cp env.example .env  # Edit with your LLM/embedding configs

# Build WebUI
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..

# Run server
lightrag-server                                           # Production
uvicorn lightrag.api.lightrag_server:app --reload        # Development
lightrag-gunicorn                                         # Multi-worker (gunicorn)
```

### WebUI
```bash
cd lightrag_webui
bun install --frozen-lockfile      # Install dependencies
bun run dev                        # Dev server (Node + Vite)
bun run dev:bun                    # Dev server (Bun native)
bun run build                      # Production build
bun run preview                    # Preview production build
bun run lint                       # ESLint over *.ts/tsx/js/jsx

# Testing — Bun built-in runner (NOT Vitest/Jest)
bun test                           # All tests
bun test --watch                   # Watch mode
bun test --coverage                # With coverage report
bun test src/api/lightrag.test.ts  # Single test file
```

### Testing

- Use mock-based tests for external services (Redis, httpx, etc.) — do not depend on live services in unit tests.
- Add regression tests for every bug fix.
- Run the full test suite (or relevant subset) and report pass counts before declaring done.
- Backend tests use pytest; frontend unit tests use Bun's built-in runner — see *WebUI* above.

```bash
# Preferred for fresh shells and automation; resolves PYTHON, venv, uv, .venv, venv, python, python3
./scripts/test.sh tests

# Run specific test file
./scripts/test.sh tests/kg/test_graph_storage.py

# Run with custom workers
./scripts/test.sh tests --test-workers 4
```

- `tests/`: main test suite, mirrors feature folders. Place new tests under the subdirectory matching the module under test:
  - `tests/api/{auth,config,routes}/` for FastAPI server tests (auth/token, config loading, route handlers); top-level `tests/api/` for app-wide concerns (path prefixes, Ollama-compatible endpoint).
  - `tests/chunker/`, `tests/evaluation/`, `tests/extraction/` for the like-named modules.
  - `tests/kg/<backend>/` (e.g. `postgres/`, `opensearch/`, `milvus/`, `faiss/`, `json/`, `neo4j/`, `networkx/`, `nano/`, `memgraph/`, `mongo/`, `qdrant/`, `redis/`) for backend-specific storage tests; `tests/kg/` root for cross-backend tests (`test_graph_storage`, `test_batch_graph_operations`, `test_unified_lock_safety`, `test_file_atomic`).
  - `tests/llm/<provider>/` (e.g. `bedrock/`, `gemini/`, `ollama/`, `openai/`, `voyageai/`, `zhipu/`) for provider-specific behavior; `tests/llm/` root for cross-provider concerns (embedding, VLM, cache, role).
  - `tests/parser/`, `tests/parser/docx/`, `tests/parser/external/{mineru,docling}/` for parser implementations.
  - `tests/pipeline/` for ingestion pipeline and doc-status behavior (including `test_pipeline_*`, `test_doc_status_*`, `test_multimodal_*`, `test_graph_keyed_locks`).
  - `tests/sidecar/`, `tests/setup/`, `tests/workspace/` for the like-named cross-cutting concerns.
  - When adding a new backend or LLM provider, create a new subdirectory plus an empty `__init__.py` rather than dropping the file in the parent directory root.
- Markers (see `tests/pytest.ini`): `offline`, `integration`, `requires_db`, `requires_api`. Integration tests are skipped by default via `-m "not integration"`.
- Integration env vars: `LIGHTRAG_RUN_INTEGRATION=true`, `LIGHTRAG_KEEP_ARTIFACTS=true`, `LIGHTRAG_TEST_WORKERS=4`, plus storage-specific connection strings.

### Linting
```bash
ruff check .
```

## Key Implementation Patterns

### LightRAG Initialization (Critical)

The most common error is forgetting to initialize storages (manifests as `AttributeError: __aenter__` or `KeyError: 'history_messages'`):

```python
import asyncio
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed
    )

    # REQUIRED: Initialize storage backends
    await rag.initialize_storages()

    # Now safe to use
    await rag.ainsert("Your text here")
    result = await rag.aquery("Your question", param=QueryParam(mode="hybrid"))

    # Cleanup
    await rag.finalize_storages()

asyncio.run(main())
```

### Custom Embedding Functions

Use `@wrap_embedding_func_with_attrs` decorator and call `.func` when wrapping (already-decorated functions cannot be wrapped again — access the underlying via `.func`):

```python
from lightrag.utils import wrap_embedding_func_with_attrs

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def custom_embed(texts: list[str]) -> np.ndarray:
    # Call underlying function, not wrapped version
    return await openai_embed.func(texts, model="text-embedding-3-large")

# Wrong: EmbeddingFunc(func=openai_embed)
# Right: EmbeddingFunc(func=openai_embed.func)
```

> **Pitfall — switching embedding models**: when changing the embedding model you MUST clear the data directory (optionally keeping `kv_store_llm_response_cache.json` for LLM cache). Existing vectors will not match the new model's space.

### Storage Configuration

Configure via environment variables or constructor params:

```python
# Environment-based (recommended for production)
# See env.example for full list

# Constructor-based
rag = LightRAG(
    working_dir="./storage",
    workspace="project_name",  # For data isolation
    kv_storage="PGKVStorage",
    vector_storage="PGVectorStorage",
    graph_storage="Neo4JStorage",
    doc_status_storage="PGDocStatusStorage",
    vector_db_storage_cls_kwargs={
        "cosine_better_than_threshold": 0.2
    }
)
```

### Document Insertion

```python
# Single document
await rag.ainsert("Text content")

# Batch insertion
await rag.ainsert(["Text 1", "Text 2", ...])

# With custom IDs
await rag.ainsert("Text", ids=["doc-123"])

# With file paths (for citation)
await rag.ainsert(["Text 1", "Text 2"], file_paths=["doc1.pdf", "doc2.pdf"])

# Configure batch size
rag = LightRAG(..., max_parallel_insert=4)  # Default: 2, max recommended: 10
```

### Query Configuration

```python
from lightrag import QueryParam

result = await rag.aquery(
    "Your question",
    param=QueryParam(
        mode="mix",                    # Recommended with reranker
        top_k=60,                      # KG entities/relations to retrieve
        chunk_top_k=20,                # Text chunks to retrieve
        max_entity_tokens=6000,
        max_relation_tokens=8000,
        max_total_tokens=30000,
        enable_rerank=True,
        user_prompt="Additional instructions for LLM",
        stream=False
    )
)
```

## Frontend Debugging via Playwright

For WebUI bugs whose symptoms only surface in the rendered DOM — layout/overflow/scrollbar issues, transient flashes, third-party libraries attaching helpers to `<body>` outside React's tree, or end-to-end verification of a fix — drive the running dev server (`http://localhost:5173`) with the `document-skills:webapp-testing` skill instead of reasoning from source alone. Seed state directly via `localStorage` (persist key `settings-storage`, schema in `lightrag_webui/src/stores/settings.ts`) to skip live LLM calls. Use `wait_until="domcontentloaded"` plus a selector wait — Vite dev's long-lived polling makes `networkidle` time out.

## Configuration

### .env Configuration
Primary configuration file for API server. Generate it with `make env-base` or copy `env.example` manually. Key sections:
- Server settings (HOST, PORT, CORS)
- Storage backends (connection strings via environment variables)
- Query parameters (TOP_K, MAX_TOTAL_TOKENS, etc.)
- Reranking configuration (RERANK_BINDING, RERANK_MODEL)
- Authentication (AUTH_ACCOUNTS, LIGHTRAG_API_KEY)

See `env.example` for comprehensive template.

### Setup Wizard Outputs
- Keep `.env` host-usable. Container-only hostnames and staged SSL paths belong in the wizard-managed compose layer, not persisted back into `.env`.
- Treat `docker-compose.final.yml` as generated output assembled from `scripts/setup/templates/*.yml`.
- For setup workflow changes, prefer `make env-*` targets over direct `scripts/setup/setup.sh` calls.

## Code Style

### Language
Comments, backend code, and log messages in English. Frontend uses i18next for multi-language support.

### Python
- Follow PEP 8 with 4-space indentation
- Use type annotations
- Prefer dataclasses for state management
- Use `lightrag.utils.logger` instead of print
- Async/await patterns throughout

### TypeScript / React (incl. WebUI ESLint)
- Functional components with hooks; PascalCase for components
- 2-space indentation, single quotes (enforced by `@stylistic` rules)
- Tailwind utility-first styling
- ESLint stack: TypeScript-ESLint + React Hooks plugin + Prettier; `@typescript-eslint/no-explicit-any` is disabled (allowed)

## Commit and Pull Request Guidance

- If this repo is a fork of `HKUDS/LightRAG`. Target to `HKUDS/LightRAG` when creating PRs, not the fork's own repo.
- PR descriptions should include: summary, motivation, linked issues if applyed, what's changed, what's broken and how it works.
