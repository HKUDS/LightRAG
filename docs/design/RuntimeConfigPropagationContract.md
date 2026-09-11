# Runtime Configuration Propagation Contract

Read this before modifying `lightrag/addon_params.py`, `lightrag/llm_roles.py`, `lightrag/kg/shared_storage.py`, or any runtime configuration / role hot-swap paths. Summary referenced in `AGENTS.md`.

---

## 1. Motivation & Scope

In multi-worker deployments (`lightrag-gunicorn --workers N`, where `preload_app = True`), the `LightRAG` orchestrator is constructed once in the master process and inherited across worker forks via copy-on-write memory. Process-local mutations to runtime configuration surfaces in worker $A$ remain completely invisible to peer workers $B_1 \dots B_{N-1}$.

This contract governs cross-worker propagation of two runtime-mutable surfaces:

1. **`addon_params`**: workspace-level settings including `language`, `entity_type_prompt_file`, `entity_types_guidance`, and `chunker`.
2. **Role LLM configuration**: process-wide settings for LLM roles (`extract`, `keyword`, `query`, `vlm`), covering `binding`, `model`, `host`, `max_async`, `timeout`, and `model_kwargs`.

### Out of Scope (Non-Goals)

* Durable persistence across process restarts (environment variables remain boot authority; shared runtime state expires with the Manager process).
* Cross-host / multi-node clustering (the multiprocessing Manager is single-host).
* Runtime modification of `embedding_func` or `rerank_model_func`.
* REST endpoints and WebUI toggles (phased for subsequent PRs; this contract governs the core propagation protocol and SDK methods).

---

## 2. Namespace Layout & Scoping

To preserve workspace isolation without entangling process-wide LLM configuration with future multi-workspace models, runtime configuration is partitioned across two namespaces in `lightrag.kg.shared_storage`:

| Configuration Surface | Namespace Key | Workspace Scope | Keyed Lock Name |
| :--- | :--- | :--- | :--- |
| **`addon_params`** | `runtime_config:addon_params` | `workspace=<workspace_id>` | `addon_params` |
| **`role_llm` + `max_async`** | `runtime_config:role_llm` | `workspace=""` | `role_llm` |

`runtime_config:role_llm` uses `workspace=""` just like `_CONCURRENCY_LEASE_NAMESPACE` and `_QUEUE_STATS_NAMESPACE` in `lightrag/kg/shared_storage.py`, consistent with its process-wide (non-workspace-scoped) status.

### Shared State Schemas

#### Addon Params Namespace (`workspace=<workspace_id>`)

```python
{
    "version": int,               # Monotonically increasing version counter
    "updated_by_pid": int,        # Publisher PID for observability/diagnostics
    "updated_at": float,          # Epoch timestamp (time.time())
    "payload": {
        "language": str,
        "entity_type_prompt_file": str | None,
        "entity_type_prompt_hash": str | None,  # SHA-256 of file contents at publish
        "entity_types_guidance": str | None,
        "chunker": dict[str, Any],              # Normalized chunker configuration
    }
}
```

#### Role LLM Namespace (`workspace=""`)

```python
{
    "version": int,               # Monotonically increasing version counter
    "updated_by_pid": int,        # Publisher PID
    "updated_at": float,          # Epoch timestamp
    "roles": {
        "<role_name>": {
            "binding": str | None,
            "model": str | None,
            "host": str | None,
            "max_async": int | None,
            "timeout": int | None,
            "model_kwargs": dict[str, Any] | None,
            "metadata": dict[str, Any],         # Strictly sanitized (no credentials)
        }
    },
    "global_concurrency_limits": dict[str, int],  # Group -> dynamic limit (e.g. "llm:extract": 8)
}
```

---

## 3. Publication Protocol

Publishing is explicit, never an implicit side-effect of local in-place dictionary mutation. This prevents normal ingestion passes (such as `resolve_chunk_options()`) from firing continuous broadcast RPCs.

### Public SDK Ingress Points

* `await rag.apublish_runtime_config(workspace=None)`: Publishes local `addon_params` and role configurations.
* `await rag.aupdate_llm_role_config(..., publish=True)`: Atomically applies a local role update and publishes to peers.

### Publication Invariants

* **Locking & Serialization**: Writers acquire `get_storage_keyed_lock(name, namespace=...)` for the target namespace. Concurrent publications from different workers are strictly serialized.
* **Strict Monotonicity**: Inside the keyed lock, the writer reads the current shared version, increments it by 1, writes the new record, and invokes `set_all_update_flags(namespace, workspace=...)`.
* **Secret Scrubbing (Zero Escape Hatch)**:
  * Before writing to shared state, all role metadata is scrubbed against `_SECRET_MARKERS` (honoring `_SAFE_OPTION_KEYS`).
  * Field values matching auth/tokens/passwords are stripped entirely.
  * Workers resolve credentials locally via their own registered role builders and local environment variables.
* **Prompt Profile Identity**: When publishing `entity_type_prompt_file`, the publisher computes and includes the SHA-256 hash of the content. The hash is the canonical identity; the file path serves as an operational hint.

---

## 4. Convergence & Application Invariants

### Apply Boundaries (Bounded Staleness)

Workers do not run background polling loops. Flag evaluation and state refresh occur strictly at predefined execution boundaries:

* **HTTP Request Boundary**: Fast-path check executed via FastAPI request lifecycle dependency.
* **Pipeline Document Boundary**: Evaluated at each document boundary within `apipeline_process_enqueue_documents`.
* **Long-running SDK Boundary**: Evaluated inside `_ensure_addon_params_cache()` throttled by a 250 ms monotonic debounce window to minimize IPC overhead during compute-intensive loops.

### Worker Local Caches & Fast Gate Reads

* **Zero IPC on Concurrency Slots**: `get_global_concurrency_limit()` reads an in-process local dictionary cache. It must never perform a live Manager read in `_acquire_global_slot()`.
* The local cache of concurrency limits is updated exclusively when the worker processes its update flag.

### Atomic Per-Worker Application & Rollback

* **Version Gate**: A worker inspects `shared["version"]`. If `shared["version"] <= worker.applied_version`, the update is ignored.
* **Addon Params Application**: Applied via `_replace_addon_params(payload, mark_dirty=True)` followed by `_apply_chunk_size_overlay()`. If the file content on disk differs from the published hash, the worker re-reads the file, emits an actionable warning regarding the hash divergence, and converges rather than entering a permanent refuse-and-retry loop.
* **Role LLM Application**:
  * Applied as an all-or-nothing batch across all roles in the payload using `_apply_llm_role_config_update()`.
  * Each role wrapper is reconstructed with the local role builder.
  * If builder resolution fails for any role (e.g., missing local API credentials for a newly assigned binding), the worker rolls back all roles to their pre-update snapshots, does not advance `applied_version`, logs an explicit warning, and retries on the next boundary check.
* **Retired Wrapper Lifecycle**: Replaced wrappers are scheduled for background queue drainage via `_schedule_retired_llm_queue_cleanup()`. Global concurrency leases are returned to shared storage upon drain or task completion; leases are never leaked.

### Document Invariant

Documents enqueued prior to a configuration publish maintain their frozen `chunk_options` and `process_options` snapshots in `full_docs`. Only documents enqueued subsequent to a local apply observe new chunker settings.

---

## 5. Single-Worker Mode Invariant (`workers == 1`)

When `initialize_share_data(workers=1)` is used:

* The Manager process is omitted.
* Shared dictionary proxies, keyed holder tables, and IPC update flags are bypassed.
* Calls to `rag.addon_params[...] = ...` and `update_llm_role_config()` function purely in-process with zero IPC overhead and zero regression in latency.

---

## 6. Observability Invariants

* `/health` and `rag.get_llm_queue_status()` report `applied_addon_params_version` and `applied_role_llm_version` **per worker**, not a single worker's scalar view presented as global state. While a publish is in flight, the response identifies which version each worker is on, e.g. `{"applied_versions": {<pid>: <version>, ...}}` for each of `applied_addon_params_version` and `applied_role_llm_version`.
* The aggregated queue status exposes this per-worker version mapping so an operator can instantly detect a diverging or un-converged worker during in-flight propagation.
* Direct runtime assignment `rag.role_llm_configs = ...` is explicitly disallowed and must raise `AttributeError` or `TypeError` instead of failing as a silent no-op.

---

## 7. Consistency Without Transactions: Failure Model

Because LightRAG distributes configuration across decoupled worker processes without distributed two-phase commit, intermediate states can exist. In alignment with `AGENTS.md` (Consistency without transactions), the table below records every accepted residue, its rationale, and its self-healing path.

| Inconsistent State | Root Cause | Why It Is Accepted | Recovery Path |
| :--- | :--- | :--- | :--- |
| Worker config skew during propagation window | A publish completed, but peer workers have not reached an apply boundary. | Bounded staleness (≤ 250 ms or next request/doc). Prevents high-frequency polling IPC overhead. | Self-heals automatically at the next HTTP request, pipeline document boundary, or debounced SDK check. |
| Worker stuck on previous version following apply failure | One worker lacks local environment credentials for a newly published binding or prompt file is unreadable. | Prevents complete process failure or unauthenticated calls. System fails safe by retaining working state. | Worker logs actionable error with PID. Operator provides local credentials or updates file; worker retries and converges on next boundary. |
| Old LLM response cache entries orphaned | Model name or host changed, shifting cache key identity. | Partitioning is deliberate (binding/model/host are part of the cache key). No cross-talk or corruption can occur. | Orphaned keys remain inert. They are not forwarded to #3833 GC to ensure cache maintenance remains decoupled from propagation state. |
| Retired role wrappers draining concurrently | Rapid successive publishes cause overlapping background wrapper drains. | In-flight requests must finish without data loss or severed connections. | Each retired queue drains gracefully bounded by `max_task_duration` (2 × timeout + 15s). Expired leases are cleaned by heartbeat sweepers. |
| Unshared concurrency limit during transition | Concurrency group transitions between unlimited and limited states while old wrappers exist. | Wrappers rebuild upon role update, automatically re-resolving `use_global_limit`. | Rebuilding wrappers binds them to the new gate limit. In-flight tasks under retired wrappers finish under their original lease terms. |

---

## 8. Rejected Remedies

* **Live Manager Read on Every Slot Acquisition**: Reading concurrency limits directly from `shared_storage` inside `_acquire_global_slot()` would introduce a synchronous Manager IPC round-trip onto the inner LLM invocation path, causing severe throughput loss.
* **Auto-Publish on Addon Mutation**: Implicit broadcast on `ObservableAddonParams` mutation was rejected because ingestion passes (`resolve_chunk_options`) modify chunker options dynamically, which would flood the system with broadcast storms.
* **Reporting Retired LLM Cache Spaces to #3833 GC**: Coupling retired key spaces to the maintenance garbage collector was rejected because cache GC must operate independently of real-time worker synchronization state.
* **Permanent Worker Refusal on Prompt File Hash Mismatch**: Refusing to advance version permanently when a content hash mismatches would cause permanent divergence if files change out of sync on a single host. Re-reading and logging a clear diagnostic error prevents permanent silent split-brain.
