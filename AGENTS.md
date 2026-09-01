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

**Workspace pipeline ingress** (`lightrag/kg/pipeline_ingress.py`, resolved via `get_pipeline_ingress(workspace)`): a three-channel mailbox living beside `pipeline_status` (never inside it — the status dict is serialized into API responses). It is the pipeline's only wake-up channel; `doc_status` stays the source of truth (a dropped notification is recovered by the next run's initial strict scan). Enqueue publishes document messages under `pipeline_status_lock` (one `put_documents` batch RPC); a busy-refused `apipeline_process_enqueue_documents` arms the **auto-rescan** flag inside `acquire_processing_reservation`'s own critical section. At every quiescence point the loop decides, atomically under `pipeline_status_lock`, cancellation first (consumes nothing), then: earliest sticky **manual retry** request (peeked, one per cycle) > **auto-rescan** dirty flag (consumed atomically; the loop is the sole consumer and re-arms it if the follow-up strict query fails) > **document** channel non-empty (peeked via `counts()`; resolved by a bounded drain-then-strict-scan refetch that compacts provably-stale messages) > release `busy` (same critical section).

**FAILED retry semantics**: automatic runs resume only `_AUTO_RESUME_DOC_STATUSES` (PENDING + PROCESSING/PARSING/ANALYZING dead-process orphans). A FAILED document re-enters the pipeline exclusively through a sticky manual retry request published by `/documents/scan` (after its reservation is granted) or `/documents/reprocess_failed` (publish-first; pure storage-driven, no filesystem scan, no custom-chunk rollback). Each request grants at most ONE retry attempt (`_MANUAL_RETRY_DOC_STATUSES`, initial scan only) and is ACKed only after the FAILED→PENDING resets persist — a crash re-executes the request or leaves the docs PENDING for automatic recovery; a doc failing again stays FAILED until the next explicit request. All scheduling-control-plane `doc_status` queries use `get_docs_by_statuses(..., strict=True)` (complete-or-raise), and scheduler `full_docs` reads distinguish confirmed-absent (`None`) from backend errors (raise). Manual-intent endpoints start their work through `start_committed_background_task` (fence recheck + publish in one critical section; a post-commit cancellation never cancels the child).

Mutual-exclusion rules (all checked atomically inside the lock):

| Operation | Refuses if | Writes |
|---|---|---|
| `_reserve_enqueue_slot` | `scanning_exclusive` or `destructive_busy` | `pending_enqueues++` |
| `apipeline_enqueue_documents` (last-line guard) | (`scanning_exclusive` and not `from_scan`) or `destructive_busy` | — |
| Scan endpoint reservation | `busy or scanning or pending_enqueues > 0` | `scanning = True` |
| `apipeline_process_enqueue_documents` entry | (already busy → arm ingress auto-rescan, return) | `busy = True` (NOT `destructive_busy`) |
| `clear_documents` / `delete_document` (synchronous reservation) | `busy or scanning or pending_enqueues > 0` | `busy = True`, `destructive_busy = True` |

The contract permits **concurrent enqueue + processing**: a freshly-uploaded doc lands in `doc_status` while the loop is mid-batch, its document message is routed into the running batch by the in-batch feeder (or resolved at the batch boundary by the quiescence decision), and the doc processes without waiting for a new run.

For the rest — write ordering of `full_docs` vs `doc_status`, the workspace-scoped `enqueue_serialize` lock around dedup-and-upsert, and the `from_scan=True` bypass — see the docstrings on `apipeline_enqueue_documents` and `apipeline_process_enqueue_documents` in `lightrag/pipeline.py`.

### Purge recovery contract

The KG is shared across documents, so "what did this document contribute?" can only be answered from the per-document **write-ahead recovery anchors** (`full_entities` / `full_relations`, written and flushed in `merge_nodes_and_edges` Phase 0 *before* the first graph mutation). The reverse lookup — graph `source_id` → `text_chunks` → `full_doc_id` — is not a fallback, because purge deletes those chunks.

The governing invariant is narrower than "every purge needs a proof":

> **A purge must never delete something that CARRIES attribution — a chunk row or an anchor row that names objects — and leave those objects behind.** An operation that removes no such carrier cannot strand anything and needs no proof.

`_purge_kg_contributions` therefore **fails closed** (`RecoveryAnchorMissingError`, surfaced as HTTP 409, nothing deleted) when it would remove a carrier without one of these proofs. Treating absent anchors as an empty candidate list was issue #3400's silent-skip defect: graph cleanup was skipped while the chunks went anyway, stranding unattributable entities that `audit_kg_integrity` can only report as unrecoverable orphans.

| Proof | Established by |
|---|---|
| `anchors` | Both anchor ROWS present and structurally usable. **Row presence is the test, never list truthiness** — an empty row is a document that extracted no entities, and conflating the two is the original bug. |
| `pre_graph` | `doc_status.metadata.kg_write_state`. Stamped `pre_graph` at enqueue so every pre-merge failure state inherits it by carry-over; advanced to `graph_mutation_started` only by `merge_nodes_and_edges`' `on_anchors_durable` hook. **Monotonic** — nothing writes it back, because re-stamping `pre_graph` on reprocess would let the resume purge skip and orphan the previous run's contributions. Absent means UNKNOWN (pre-#3416), which fails closed. |
| `journal` | `doc_status.metadata.kg_purge` at a phase past `prepared`, i.e. a previous attempt got far enough to have deleted the anchors itself. |
| `empty_scope` | No chunks AND no anchor row that names anything — so the delete removes no carrier at all and the invariant is satisfied outright. This is what lets a row enqueued before the marker existed, still holding no chunks, be deleted directly (no scan, no audit). |

**`kg_write_state` must never be inferred.** `pre_graph` asserts "this document never touched the graph", which licenses deleting its chunks while *skipping the graph* — sound only because the marker is written once, at enqueue, when it is necessarily true and the document has no history to misread. A backfill keying off a momentarily-empty `chunks_list` would stamp a document that does own graph objects, and because the stamp is durable the damage lands later, when the chunks reappear: chunks deleted, graph skipped, issue #3400 reproduced exactly. `empty_scope` is safe where such a backfill is not, because it is re-evaluated against live state on every call and grants nothing beyond that call. `tests/pipeline/test_purge_fail_closed.py::test_a_false_pre_graph_marker_would_reproduce_the_original_defect` pins the cost.

Anchor-driven whole-document purge is **journaled and resumable** through four ordered phases — `prepared` → `derived_committed` → `anchors_pending` → `completed` — keyed by an operation id over the document key plus its chunk SET. The journal is *required by* fail-closed rather than an optimisation: purge's last step deletes the anchors, so without it any later failure would make every retry refuse forever. A resumed purge skips exactly the phases already persisted (so it never re-runs the LLM-cache-backed rebuild); an in-flight journal for a different operation is refused (`KGPurgeOperationConflictError`), while a stale `completed` one is ignored as dead bookkeeping.

Both metadata keys are in the `_DOC_STATUS_METADATA_CARRY_OVER_KEYS` **and** `_DOC_STATUS_METADATA_DIRECTIVE_KEYS` whitelists in `lightrag/utils_pipeline.py`; dropping either at a transition or a FAILED→PENDING reset turns a resumable purge into a permanent refusal. Retiring one requires `doc_status_transition_metadata(..., drop=...)` — passing it via `extra` would persist the value, and omitting it lets carry-over restore it.

Callers: `adelete_by_doc_id` (delegates wholly to the primitive; the chunk-less branch runs it too), and the pipeline's resume path `_purge_stale_extraction_if_resuming` (which retires the journal and persists `chunks_list=[]` in one targeted write). Explicit-candidate mode — custom-chunk patch rollback — is neither journaled nor proof-checked, because its own operation journal already names the complete candidate superset; the primitive reads that journal to union in candidates no anchor row can name yet.

A document can legitimately own nothing: `skip_kg` (`process_options` `'!'`) skips extraction and the merge, so no anchor rows are ever written. Post-change those documents carry `pre_graph` and delete normally; older ones have neither proof, and anchor repair has nothing to rebuild from.

**Chunk tracking outranks graph `source_id`.** Within a surviving entity or relation, the `entity_chunks` / `relation_chunks` row is the authoritative chunk list; the graph node's `source_id` is only a truncated view of it (`apply_source_ids_limit`) and may legitimately still name chunks a previous purge already pruned — `_purge_kg_contributions` reads tracking first, falls back to `source_id` only when the row is absent, and its `graph_references_deleted_chunks` branch exists to repair exactly that lag. So code that folds a `source_id` delta back into tracking must append genuine additions only: restoring an ID that is in the graph but not in tracking writes stale attribution into the authoritative store, and a later purge would rebuild or retain KG objects from chunks that no longer exist. `compute_incremental_chunk_ids` carries this rule and `tests/utils/test_compute_incremental_chunk_ids.py` pins it. Genuinely missing attribution is repaired by `audit_kg_integrity`, never by the incremental path.

### Relation weight contract

Relation `weight` is bounded below by the number of distinct real IDs in the
graph edge's `source_id`; a larger value is an optional importance boost.
Empty IDs and the legacy no-source placeholders `manual_creation` and
`UNKNOWN` do not count as evidence. A source-less relation may therefore use
any non-negative fractional weight. Public ingress paths (`create_relation`,
`edit_relation`, and `insert_custom_kg`) must validate the complete relation
before the first storage mutation. To request a weight below the current
evidence count, creation callers omit `source_id`, while edit callers set it to
an empty string in the same operation.

Entity merges use `max(all input weights, distinct merged real source IDs)`.
Every extraction merge or entity-rename rewrite that rewrites the edge is a
repair point for legacy rows: it must lift an undersized stored weight to the
current evidence floor while preserving any larger explicit boost. Repair is
opportunistic, not a sweep: `_merge_edges_then_upsert`'s KEEP-cap skip branch
returns the stored edge without writing the graph or the vector record, so a
legacy row there stays undersized until a merge, an unrelated relation edit, or
a rebuild rewrites it. Rebuilds from surviving chunks
(`_rebuild_single_relationship`, reached only through `_purge_kg_contributions`
-> `rebuild_knowledge_from_chunks`, i.e. document purge, resume, and
custom-chunk rollback) are a repair point for the floor only: they re-derive
weight from the surviving cached fragments and then lift it to the surviving
evidence count, so weight tracks evidence down as purge removes chunks and an
explicit boost is not carried across — exactly as the rebuilt description and
keywords replace their edited values. The degraded path, having no fragments to
re-derive from, keeps the stored weight instead. Relation chunk tracking is the
authoritative chunk list, so the no-source placeholders must never be written
into it. Keep this contract synchronized across the core API docstrings, REST
graph documentation, `ProgramingWithCore.md`, and custom-KG examples whenever
relation write behavior changes.

The offline remedy for a document with no proof is `audit_kg_integrity(..., apply=True)` (`lightrag/tools/kg_integrity_repair.py`): it rebuilds anchors from surviving chunk provenance, and — because it enumerates the **whole** graph, which the hot paths never do — it can additionally certify that a document appearing nowhere in that scan owns nothing, writing it the empty anchor rows that are the normal proof for such a document (`anchorless_docs` in the report). Absence is only ever concluded from the completed scan; a document that does own graph objects is repaired with its real names, never blanked.

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

Every command below is run **from `lightrag_webui/`**, not the repository root.
`bun test` in particular resolves `bunfig.toml` — and the preload paths inside
it — relative to the working directory, so running it from the root silently
loads no DOM (see *React component tests*).

```bash
cd lightrag_webui
bun install --frozen-lockfile      # REQUIRED after any change to package.json /
                                   # bun.lock, including a branch switch across
                                   # one: `git checkout` does not update
                                   # node_modules, and a stale tree surfaces as
                                   # a wall of TS2307 "Cannot find module".
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
bunx tsc --noEmit                  # Typecheck (`bun run build` does NOT typecheck)
```

### Testing

- Use mock-based tests for external services (Redis, httpx, etc.) — do not depend on live services in unit tests.
- Add regression tests for every bug fix.
- **Run only the test directories that mirror the modules you changed**, and report which subset you ran plus its pass count. The suite is ~7000 tests and a full run takes over 6 minutes, which is too slow for the edit loop. Every PR's CI runs the full suite — proving nothing else broke is its job, not yours.
- Derive the subset from the mirror layout below: `lightrag/api/config.py` → `tests/api/config/`, `lightrag/kg/redis_impl.py` → `tests/kg/redis_impl/`, `lightrag/chunker/` → `tests/chunker/`. When a change spans several modules, run each of their directories rather than widening to `tests/`.
- Run the full suite locally only at a milestone, or when the change is genuinely cross-cutting (`lightrag/base.py`, `lightrag/utils.py`, `lightrag/kg/shared_storage.py`, or anything every backend inherits).
- Backend tests use pytest; frontend unit tests use Bun's built-in runner — see *WebUI* above and *React component tests* below.
- **A WebUI change runs the WHOLE frontend check set**, from `lightrag_webui/`: `bun install --frozen-lockfile` (see *WebUI* above — skip it after a branch switch and every later step fails on missing modules), then `bun test`, `bunx tsc --noEmit`, and `bun run lint`. The subsetting rule above is a backend rule and does not apply — all three together take well under a minute (test ~2 s, typecheck ~14 s, lint ~21 s), so there is nothing to save by running less. Report the pass count. `bun run build` transpiles WITHOUT checking types, so skipping `tsc --noEmit` means nothing checks them.

#### React component tests

WebUI tests are **colocated** next to the module they cover
(`src/features/SiteHeader.test.ts`), not mirrored into a separate tree — the
`tests/` mirror layout above is a backend rule and does not apply here. A test
file containing JSX must be named `.test.tsx`.

`bun test` has a DOM: `bunfig.toml` preloads `src/test/happydom.ts` (registers
happy-dom globally) and then `src/test/setup.ts` (jest-dom matchers plus
Testing Library's `cleanup` in `afterEach`). Order is load-bearing — Testing
Library binds to whatever `document` exists when it is first evaluated.

That preload is found relative to the WORKING DIRECTORY, so `bun test` must be
run from `lightrag_webui/`. From the repository root no preload loads at all
and the failure is silent in the worst way: pure logic tests still pass and
only the component tests break. `src/test/render.tsx` calls
`assertDomAvailable()` at import time to turn that into a message naming the
cause and the fix; `bun test --config <path>` does NOT work around it, because
the preload paths inside the file are still resolved against the CWD.

Rules for new tests:

- **Test rendered behavior by rendering it.** Assert what the user gets —
  roles, accessible names, visibility, what a click does. Do NOT write new
  tests that `readFileSync` a `.tsx` and match substrings: that style cannot
  see whether Radix's `asChild` actually wired the trigger up, and it breaks on
  equivalent rewrites. Several older tests still do this; converting one while
  working nearby is welcome. String and AST assertions stay correct for what
  genuinely IS a source-level property — an i18n key present in every locale, a
  forbidden import — just not for what the component renders.
- **Seed the stores a page reads, rather than stubbing its requests.**
  Pages behind `useCustomizedContent` render NOTHING until the first
  customization response settles, so an unseeded render finds an empty page:
  use `seedCustomization()` from `src/test/customization.ts`. Where a page
  really does call the API on mount, stub the module and mind the IMPORT
  ORDER — `mock.module` only reaches importers that have not been evaluated
  yet, so the component must be imported dynamically AFTER the mock (a static
  top-level import binds the real module first and the request goes out for
  real). Restore the module in `afterAll`.
- **Render through `renderWithProviders`** (`src/test/render.tsx`), not
  Testing Library's bare `render`. It supplies a fixed English i18n instance
  built from `locales/en.json` and deliberately does not import `@/i18n`, whose
  bootstrap resolves a language from `localStorage` and runs the settings
  migration — ambient state that asserted strings must not depend on.
- **Never assert `toBeNull()` / `not.toBeInTheDocument()` on a DOM element.**
  Use a count instead — `expect(screen.queryAllByRole(...)).toHaveLength(0)`.
  When such an assertion fails, Bun serialises the entire happy-dom element
  it received, which is large enough that the run appears to HANG rather than
  report a failure. The count form fails instantly and legibly
  (`Expected length: 0, Received length: 1`). The same applies to any
  assertion whose failure message would carry a DOM node: compare a boolean
  or a string you extracted (`expect(card.contains(footer)).toBe(false)`,
  `expect(el.getAttribute('href')).toBe('./')`).
- **Prove the test can fail.** Before calling it done, break the behavior it
  pins (flip the `aria-label`, drop the guard), confirm it goes red, then
  restore. A test written against already-passing code is worth nothing until
  it has been seen to fail: `harnessIsolation.test.ts` originally matched only
  `from '…'` and silently let a bare side-effect `import '…'` through, which
  only the mutation check surfaced.
- **The DOM is process-wide.** Bun evaluates every test file in one process, so
  `delete globalThis.window` in one file removes it for every file that runs
  later — and the failure surfaces somewhere else entirely. To exercise a
  DOM-less code path use `withoutDomGlobals(body, keys?)` from
  `src/test/domGlobals.ts`; to undo a stubbed global use `restoreDomGlobals()`
  in an `afterEach`. Never leave a bare `delete` of the `window` or `document`
  GLOBAL behind — `harnessIsolation.test.ts` fails on one anywhere but the
  helper. (Deleting a property OF window, such as `__LIGHTRAG_CONFIG__`, is
  fine and is not what the guard matches.)
- **Never import the test harness from production code.** Vite bundles from the
  import graph rooted at `index.html` / `workspace.html`, and the harness is
  reached only through the runner's preload — one import from `src/` would ship
  happy-dom to the browser. `src/test/harnessIsolation.test.ts` pins this for
  every import form (bare, dynamic, `require`), and `vite.config.ts`'s
  first-load byte budget backs it up. Dependency-section placement is not what
  decides this: `@faker-js/faker` is a runtime `dependencies` entry and ships
  because `hooks/useRandomGraph.tsx` imports it.

```bash
# Preferred for fresh shells and automation; resolves PYTHON, venv, uv, .venv, venv, python, python3
# Default during development: only the directories mirroring the changed modules
./scripts/test.sh tests/api/config
./scripts/test.sh tests/kg/redis_impl

# Run specific test file
./scripts/test.sh tests/kg/test_graph_storage.py

# Full suite — ~7000 tests, >6 min; milestones and cross-cutting changes only
./scripts/test.sh tests

# Run with custom workers
./scripts/test.sh tests --test-workers 4
```

- `tests/`: main test suite, mirrors feature folders. Place new tests under the subdirectory matching the module under test:
  - `tests/api/{auth,config,routes}/` for FastAPI server tests (auth/token, config loading, route handlers); top-level `tests/api/` for app-wide concerns (path prefixes, Ollama-compatible endpoint).
  - `tests/chunker/`, `tests/evaluation/`, `tests/extraction/` for the like-named modules.
  - `tests/kg/<backend>_impl/` for backend-specific storage tests, mirroring the `lightrag/kg/<backend>_impl.py` file naming. The `_impl` suffix on every subdirectory keeps the layout uniform and avoids `sys.path` shadowing on names that overlap with top-level PyPI/stdlib packages (`faiss`, `json`, `neo4j`, `networkx`, `redis`) when a test is launched directly via `python tests/kg/...`. Current backends: `faiss_impl/`, `json_impl/`, `memgraph_impl/`, `milvus_impl/`, `mongo_impl/`, `nano_impl/`, `neo4j_impl/`, `networkx_impl/`, `opensearch_impl/`, `postgres_impl/`, `qdrant_impl/`, `redis_impl/`. `tests/kg/` root holds cross-backend tests (`test_graph_storage`, `test_batch_graph_operations`, `test_unified_lock_safety`, `test_file_atomic`).
  - `tests/llm/<provider>_impl/` for provider-specific behavior, same `_impl` convention: `bedrock_impl/`, `gemini_impl/`, `ollama_impl/`, `openai_impl/`, `voyageai_impl/`, `zhipu_impl/`. `tests/llm/` root holds cross-provider concerns (embedding, VLM, cache, role).
  - `tests/parser/`, `tests/parser/docx/`, `tests/parser/external/{mineru,docling}/` for parser implementations.
  - `tests/pipeline/` for ingestion pipeline and doc-status behavior (including `test_pipeline_*`, `test_doc_status_*`, `test_multimodal_*`, `test_graph_keyed_locks`).
  - `tests/sidecar/`, `tests/setup/`, `tests/workspace/` for the like-named cross-cutting concerns.
  - When adding a new backend or LLM provider, create a new subdirectory plus an empty `__init__.py` rather than dropping the file in the parent directory root.
- Markers (registered in `[tool.pytest.ini_options]` in `pyproject.toml`): `offline`, `integration`, `requires_db`, `requires_api`, `pg_smoke`. Integration tests are skipped by default via `-m "not integration"`; opt in with `--run-integration`.
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
rag = LightRAG(..., max_parallel_insert=4)  # Default: 3, max recommended: 10
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
Comments, backend code, log messages, and Git commit messages in English. Frontend uses i18next for multi-language support.

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
- Write commit messages (subject and body) in English. Commit messages are repository artifacts — like code comments and log messages — not conversational replies, so they follow the English code-style rule above regardless of any per-conversation working language.
