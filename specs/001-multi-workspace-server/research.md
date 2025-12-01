# Research: Multi-Workspace Server Support

**Date**: 2025-12-01
**Feature**: 001-multi-workspace-server

## Executive Summary

Research confirms that the existing LightRAG codebase provides solid foundation for multi-workspace support at the server level. The core library already has workspace isolation; the gap is purely at the API server layer.

## Research Findings

### 1. Existing Workspace Support in LightRAG Core

**Decision**: Leverage existing `workspace` parameter in `LightRAG` class

**Findings**:
- `LightRAG` class accepts `workspace: str` parameter (default: `os.getenv("WORKSPACE", "")`)
- Storage implementations use `get_final_namespace(namespace, workspace)` to create isolated keys
- Namespace format: `"{workspace}:{namespace}"` when workspace is set, else just `"{namespace}"`
- Pipeline status, locks, and in-memory state are all workspace-aware via `shared_storage.py`
- `DocumentManager` creates workspace-specific input directories

**Evidence**:
```python
# lightrag/lightrag.py
workspace: str = field(default_factory=lambda: os.getenv("WORKSPACE", ""))

# lightrag/kg/shared_storage.py
def get_final_namespace(namespace: str, workspace: str | None = None) -> str:
    if workspace is None:
        workspace = get_default_workspace()
    if not workspace:
        return namespace
    return f"{workspace}:{namespace}"
```

**Implications**: No changes needed to core isolation; just need to instantiate separate `LightRAG` objects with different `workspace` values.

### 2. Current Server Architecture

**Decision**: Refactor from closure pattern to FastAPI dependency injection

**Findings**:
- Server creates a single global `LightRAG` instance in `create_app(args)`
- Routes receive the RAG instance via closure (factory function pattern):
  ```python
  def create_document_routes(rag: LightRAG, doc_manager, api_key):
      @router.post("/scan")
      async def scan_for_new_documents(...):
          # rag captured from enclosing scope
  ```
- This pattern prevents per-request workspace switching

**Alternative Considered**: Keep closure pattern and add workspace switching to existing instance
- **Rejected Because**: LightRAG instance configuration is immutable after creation; switching workspace would require re-initializing storage connections

**Chosen Approach**: Replace closure with FastAPI `Depends()` that resolves workspace → instance

### 3. Instance Pool Design

**Decision**: Use `asyncio.Lock` protected dictionary with LRU eviction

**Findings**:
- Python's `asyncio.Lock` is appropriate for protecting async operations
- LRU eviction via `collections.OrderedDict` or manual tracking
- Instance initialization is async (`await rag.initialize_storages()`)
- Concurrent requests for same new workspace must share initialization

**Pattern**:
```python
_instances: dict[str, LightRAG] = {}
_lock = asyncio.Lock()
_lru_order: list[str] = []  # Most recent at end

async def get_instance(workspace: str) -> LightRAG:
    async with _lock:
        if workspace in _instances:
            # Move to end of LRU list
            _lru_order.remove(workspace)
            _lru_order.append(workspace)
            return _instances[workspace]

        # Evict if at capacity
        if len(_instances) >= max_pool_size:
            oldest = _lru_order.pop(0)
            await _instances[oldest].finalize_storages()
            del _instances[oldest]

        # Create and initialize
        instance = LightRAG(workspace=workspace, ...)
        await instance.initialize_storages()
        _instances[workspace] = instance
        _lru_order.append(workspace)
        return instance
```

**Alternative Considered**: Use `async_lru` library or `cachetools.TTLCache`
- **Rejected Because**: Adds external dependency; simple dict+lock is sufficient and well-understood

### 4. Header Routing Strategy

**Decision**: `LIGHTRAG-WORKSPACE` primary, `X-Workspace-ID` fallback

**Findings**:
- Custom headers conventionally use `X-` prefix, but this is deprecated per RFC 6648
- Product-specific headers (e.g., `LIGHTRAG-WORKSPACE`) are clearer and recommended
- Fallback to common convention (`X-Workspace-ID`) aids adoption

**Implementation**:
```python
def get_workspace_from_request(request: Request) -> str | None:
    workspace = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()
    if not workspace:
        workspace = request.headers.get("X-Workspace-ID", "").strip()
    return workspace or None
```

### 5. Configuration Schema

**Decision**: Three new environment variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LIGHTRAG_DEFAULT_WORKSPACE` | str | `""` (from `WORKSPACE`) | Default workspace when no header |
| `LIGHTRAG_ALLOW_DEFAULT_WORKSPACE` | bool | `true` | If false, reject requests without header |
| `LIGHTRAG_MAX_WORKSPACES_IN_POOL` | int | `50` | Maximum concurrent workspace instances |

**Rationale**:
- `LIGHTRAG_` prefix namespaces new vars to avoid conflicts
- `ALLOW_DEFAULT_WORKSPACE=false` enables strict multi-tenant mode
- Default pool size of 50 balances memory vs. reinitialization overhead

### 6. Workspace Identifier Validation

**Decision**: Alphanumeric, hyphens, underscores; 1-64 characters

**Findings**:
- Must be safe for filesystem paths (workspace creates subdirectories)
- Must be safe for database keys (used in storage namespacing)
- Must prevent injection attacks (path traversal, SQL injection)

**Validation Regex**: `^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$`
- Starts with alphanumeric (prevents hidden directories like `.hidden`)
- Allows hyphens and underscores for readability
- Max 64 chars (reasonable for identifiers, fits in most DB column sizes)

### 7. Error Handling

**Decision**: Return 400 for missing/invalid workspace; 503 for initialization failures

| Scenario | HTTP Status | Error Message |
|----------|-------------|---------------|
| Missing header, default disabled | 400 | `Missing LIGHTRAG-WORKSPACE header` |
| Invalid workspace identifier | 400 | `Invalid workspace identifier: must be alphanumeric...` |
| Workspace initialization fails | 503 | `Failed to initialize workspace: {details}` |

### 8. Logging Strategy

**Decision**: Log workspace identifier at INFO level; never log credentials

**Implementation**:
- Log workspace on request: `logger.info(f"Request to workspace: {workspace}")`
- Log pool events: `logger.info(f"Initialized workspace: {workspace}")`
- Log evictions: `logger.info(f"Evicted workspace from pool: {workspace}")`
- NEVER log: API keys, storage credentials, auth tokens

### 9. Test Strategy

**Decision**: Pytest with markers following existing patterns

**Test Categories**:
1. **Unit tests** (`@pytest.mark.offline`): Workspace resolution, validation, pool logic
2. **Integration tests** (`@pytest.mark.integration`): Full request flow with mock LLM/embedding
3. **Backward compatibility tests** (`@pytest.mark.offline`): Single-workspace mode unchanged

**Key Test Scenarios**:
- Two workspaces → ingest document in A → query from B returns nothing
- No header + `ALLOW_DEFAULT_WORKSPACE=true` → uses default
- No header + `ALLOW_DEFAULT_WORKSPACE=false` → returns 400
- Pool at capacity → evicts LRU → new workspace initializes

## Resolved Questions

| Question | Resolution |
|----------|------------|
| How to handle concurrent init of same workspace? | `asyncio.Lock` ensures single initialization; others wait |
| Should evicted workspace finalize storage? | Yes, call `finalize_storages()` to release resources |
| How to share config between instances? | Clone config; only `workspace` differs per instance |
| Where to put pool management code? | New module `workspace_manager.py` |

## Next Steps

1. Create `data-model.md` with entity definitions
2. Document contracts (no new API endpoints; header-based routing is transparent)
3. Create `quickstart.md` for multi-workspace deployment
