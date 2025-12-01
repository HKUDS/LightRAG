# Data Model: Multi-Workspace Server Support

**Date**: 2025-12-01
**Feature**: 001-multi-workspace-server

## Overview

This feature introduces server-level workspace management without adding new persistent data models. The data model focuses on runtime entities that manage workspace instances.

## Entities

### WorkspaceInstance

Represents a running LightRAG instance serving requests for a specific workspace.

| Attribute | Type | Description |
|-----------|------|-------------|
| `workspace_id` | `str` | Unique identifier for the workspace (validated, 1-64 chars) |
| `rag_instance` | `LightRAG` | The initialized LightRAG object |
| `created_at` | `datetime` | When the instance was first created |
| `last_accessed_at` | `datetime` | When the instance was last used (for LRU) |
| `status` | `enum` | `initializing`, `ready`, `finalizing`, `error` |

**Validation Rules**:
- `workspace_id` must match: `^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$`
- `workspace_id` must not be empty string (use explicit default workspace)

**State Transitions**:
```
┌─────────────┐     ┌───────┐     ┌────────────┐
│ initializing│ ──► │ ready │ ──► │ finalizing │
└─────────────┘     └───────┘     └────────────┘
      │                 │
      ▼                 ▼
  ┌───────┐         ┌───────┐
  │ error │         │ error │
  └───────┘         └───────┘
```

### WorkspacePool

Collection managing active WorkspaceInstance objects.

| Attribute | Type | Description |
|-----------|------|-------------|
| `max_size` | `int` | Maximum concurrent instances (from config) |
| `instances` | `dict[str, WorkspaceInstance]` | Active instances by workspace_id |
| `lru_order` | `list[str]` | Workspace IDs ordered by last access |
| `lock` | `asyncio.Lock` | Protects concurrent access |

**Invariants**:
- `len(instances) <= max_size`
- `set(lru_order) == set(instances.keys())`
- Only one instance per workspace_id

**Operations**:

| Operation | Description | Complexity |
|-----------|-------------|------------|
| `get(workspace_id)` | Get or create instance, updates LRU | O(1) amortized |
| `evict_lru()` | Remove least recently used instance | O(1) |
| `finalize_all()` | Clean shutdown of all instances | O(n) |

### WorkspaceConfig

Configuration for multi-workspace behavior (runtime, not persisted).

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_workspace` | `str` | `""` | Workspace when no header present |
| `allow_default_workspace` | `bool` | `true` | Allow requests without header |
| `max_workspaces_in_pool` | `int` | `50` | Pool size limit |

**Sources** (in priority order):
1. Environment variables (`LIGHTRAG_DEFAULT_WORKSPACE`, etc.)
2. Existing `WORKSPACE` env var (backward compatibility)
3. Hardcoded defaults

## Relationships

```
┌─────────────────┐
│ WorkspaceConfig │
└────────┬────────┘
         │ configures
         ▼
┌─────────────────┐       contains        ┌───────────────────┐
│  WorkspacePool  │◄─────────────────────►│ WorkspaceInstance │
└─────────────────┘                       └───────────────────┘
         │                                         │
         │ validates workspace_id                  │ wraps
         ▼                                         ▼
┌─────────────────┐                       ┌───────────────────┐
│ HTTP Request    │                       │ LightRAG (core)   │
│ (workspace hdr) │                       │                   │
└─────────────────┘                       └───────────────────┘
```

## Data Flow

### Request Processing

```
1. HTTP Request arrives
   │
2. Extract workspace from headers
   │  ├─ LIGHTRAG-WORKSPACE header (primary)
   │  └─ X-Workspace-ID header (fallback)
   │
3. If no header:
   │  ├─ allow_default_workspace=true → use default_workspace
   │  └─ allow_default_workspace=false → return 400
   │
4. Validate workspace_id format
   │  └─ Invalid → return 400
   │
5. WorkspacePool.get(workspace_id)
   │  ├─ Instance exists → update LRU, return instance
   │  └─ Instance missing:
   │       ├─ Pool full → evict LRU instance
   │       └─ Create new instance, initialize, add to pool
   │
6. Route handler receives LightRAG instance
   │
7. Process request using instance
   │
8. Return response
```

### Instance Lifecycle

```
1. First request for workspace arrives
   │
2. WorkspacePool creates WorkspaceInstance
   │  status: initializing
   │
3. LightRAG object created with workspace parameter
   │
4. await rag.initialize_storages()
   │
5. Instance status → ready
   │  Added to pool and LRU list
   │
6. Instance serves requests...
   │  last_accessed_at updated on each access
   │
7. Pool reaches max_size, this instance is LRU
   │
8. Instance status → finalizing
   │
9. await rag.finalize_storages()
   │
10. Instance removed from pool
```

## No Persistent Schema Changes

This feature does not modify:
- Storage schemas (KV, vector, graph)
- Database tables
- File formats

Workspace isolation at the data layer is already handled by the LightRAG core using namespace prefixing.
