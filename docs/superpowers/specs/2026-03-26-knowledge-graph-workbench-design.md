# Knowledge Graph Workbench Design

## Problem

The current LightRAG knowledge-graph experience already has useful primitives, but the workflow is fragmented:

- The WebUI graph page is still centered around `label + depth + max_nodes`, not multi-dimensional filtering.
- Entity and relation create/edit/merge capabilities already exist in the backend, but they are not presented as a coherent graph-governance workflow in the UI.
- Delete capabilities live under document routes, which makes graph operations feel split across domains.
- Existing merge support is useful for disambiguation, but there is no first-class workbench for manual merge and candidate-assisted merge review.
- Internationalization exists in the WebUI, but new graph-governance features would easily regress into hardcoded strings unless i18n is designed up front.

The goal of this design is to turn the graph page into a true workbench for graph exploration, maintenance, and disambiguation without forcing a rewrite of the current graph viewer stack.

## Goals

- Add richer filtering controls for graph exploration.
- Add a first-class CRUD workflow for entities and relations inside the graph experience.
- Add node-merge functionality for disambiguation, including both manual merge and suggestion-assisted merge.
- Preserve compatibility with existing graph routes and graph storage contracts where possible.
- Keep the implementation aligned with existing `GraphViewer`, `PropertiesView`, `graph_routes`, and `utils_graph` patterns.
- Explicitly support multi-language UI behavior from the first implementation pass.

## Scope

### In Scope

- Graph workbench UI redesign around the existing graph page
- Structured graph query/filter API
- Graph-domain delete route normalization
- Manual merge UX
- Merge suggestion API and UX
- Frontend state changes required for graph workbench behavior
- Backend tests and frontend tests for the new graph workbench flows
- Internationalization requirements for the new graph workbench

### Out of Scope

- Automatic merge execution without user confirmation
- True undo/rollback after merge or delete
- Cross-workspace graph operations
- Reworking all graph storage backends to natively implement advanced filtering
- Full review-history or audit-log subsystem for graph governance

## Existing Constraints and Reusable Foundations

### Backend

- `lightrag/api/routers/graph_routes.py` already provides:
  - graph label listing/search
  - base graph retrieval via `GET /graphs`
  - entity create/edit
  - relation create/edit
  - entity merge
- `lightrag/api/routers/document_routes.py` already provides:
  - entity delete
  - relation delete
- `lightrag/lightrag.py` and `lightrag/utils_graph.py` already provide:
  - `acreate_entity`
  - `acreate_relation`
  - `aedit_entity`
  - `aedit_relation`
  - `adelete_by_entity`
  - `adelete_by_relation`
  - `amerge_entities`

### Frontend

- `lightrag_webui/src/features/GraphViewer.tsx` already supplies the graph page shell.
- `lightrag_webui/src/components/graph/PropertiesView.tsx` already supports entity/edge inspection.
- `lightrag_webui/src/components/graph/EditablePropertyRow.tsx` already supports inline edit behavior and existing merge-on-rename cases.
- `lightrag_webui/src/hooks/useLightragGraph.tsx` already owns graph fetching, hydration into `RawGraph`, Sigma graph creation, and partial node expand/prune behavior.
- `lightrag_webui/src/stores/graph.ts` already manages graph rendering state and partial in-memory graph updates.
- `lightrag_webui/src/i18n.ts` and `lightrag_webui/src/locales/*.json` already establish a multilingual frontend foundation.

### Architecture Constraint

The design should avoid forcing every `BaseGraphStorage` implementation to grow a backend-specific advanced filter engine. The current graph storage abstraction is suitable for base graph retrieval and CRUD primitives, but not for a broad new filtering contract across all backends.

### Scale and Safety Constraint

The repository already has `max_graph_nodes` protections across runtime, API config, storage backends, and WebUI settings. However, once advanced filtering is performed after base graph retrieval, those existing limits need to be treated as an explicit design boundary rather than an implicit implementation detail. Query-size protection, mutation concurrency, and destructive-operation safeguards must therefore be part of the workbench contract.

## High-Level Design

The knowledge-graph page becomes a three-column workbench built on top of the current graph viewer:

1. **Left column: `FilterWorkbench`**
   - Owns structured filtering and query presets
   - Covers node filters, edge filters, scope filters, source filters, and view controls

2. **Center column: Graph canvas**
   - Keeps Sigma as the main visualization surface
   - Adds a top command bar for workbench actions
   - Adds a bottom work area for merge suggestions and current selection queue

3. **Right column: `ActionInspector`**
   - Upgrades the current property panel into a tabbed action area:
     - `Inspect`
     - `Create`
     - `Delete`
     - `Merge`

This preserves the existing graph stack while turning the page into a continuous graph-governance workflow.

## UI Design

### 1. FilterWorkbench

The left workbench supports five filter families:

- **Node filters**
  - entity type
  - name / description text search
  - degree range
  - isolated node toggle
- **Edge filters**
  - relation type
  - keyword text search
  - weight range
  - source-type / target-type constraints
- **Scope filters**
  - start node
  - max depth
  - max nodes
  - only matched neighborhood toggle
- **Source filters**
  - `source_id`
  - file path
  - time range
- **View controls**
  - show only nodes / only edges
  - hide low-weight edges
  - hide empty-description objects
  - highlight matched items

The filter UI should support:

- draft state before apply
- explicit apply/reset actions
- reusable filter presets in a later-compatible shape
- result summary metadata, such as matched count and truncation hints

### 2. ActionInspector

The right-side action area is split into four tabs.

#### `Inspect`

- Reuses the current `PropertiesView` behavior
- Continues to support property inspection and direct edits
- Continues to show adjacency/navigation affordances

#### `Create`

- Split into:
  - `Create Node`
  - `Create Relation`
- `Create Node` requires a minimal valid payload:
  - `entity_name`
  - `description`
- `Create Relation` should:
  - prefer prefilled source/target from current graph selection
  - support manual override
  - expose relation description/keywords/weight

#### `Delete`

- Distinguishes entity deletion from relation deletion
- Must always show a confirmation view before execution
- Entity deletion must clearly state that related edges will also be removed
- Relation deletion must clearly identify both endpoints and the relation summary

#### `Merge`

Supports two entry paths:

- **Manual merge**
  - user selects source entity/entities and target entity
  - user can edit target-preserved values before execution
- **Suggested merge**
  - user enters from a candidate list generated by the suggestion API
  - candidate evidence is displayed before execution

Both paths converge into one confirmation panel showing:

- source entities to be removed
- target entity to retain
- merge evidence
- merged preview / retained fields

After a successful merge, the UI should offer a post-action choice:

- focus the merged target entity
- refresh the current filtered result
- continue reviewing the next merge suggestion

## Backend Design

### 1. Structured Graph Query API

Add a graph-domain structured query endpoint in `graph_routes.py`.

Recommended route:

- `POST /graph/query`

Recommended request shape:

```json
{
  "scope": {
    "label": "Tesla",
    "max_depth": 2,
    "max_nodes": 800,
    "only_matched_neighborhood": true
  },
  "node_filters": {
    "entity_types": ["ORGANIZATION", "PERSON"],
    "name_query": "tesla",
    "description_query": "",
    "degree_min": 1,
    "degree_max": 50,
    "isolated_only": false
  },
  "edge_filters": {
    "relation_types": ["acquires", "owns"],
    "keyword_query": "",
    "weight_min": 0.3,
    "weight_max": 10,
    "source_entity_types": [],
    "target_entity_types": []
  },
  "source_filters": {
    "source_id_query": "",
    "file_paths": [],
    "time_from": null,
    "time_to": null
  },
  "view_options": {
    "show_nodes_only": false,
    "show_edges_only": false,
    "hide_low_weight_edges": true,
    "hide_empty_description": true,
    "highlight_matches": true
  }
}
```

Semantics for v1 must be explicit:

- filter groups are combined with **AND**
- different fields inside a group are combined with **AND**
- array values inside a single field are combined with **OR**

Examples:

- `entity_types: ["PERSON", "ORGANIZATION"]` means `PERSON OR ORGANIZATION`
- `entity_types + name_query` means `(PERSON OR ORGANIZATION) AND name_query`

The first version does not attempt to model arbitrary nested boolean expressions, but the request contract should remain future-extensible rather than implying the current flat shape is the permanent upper bound.

Implementation principle:

- Graph storage backends continue to provide the base graph via the existing retrieval contract.
- Advanced filtering happens in the API/workbench layer after base graph retrieval.
- This minimizes cross-backend disruption and keeps the first implementation practical.

### 1a. Performance Guardrails

The structured query path must include explicit protection against oversized in-memory filtering:

- the API must always apply a hard node cap before in-process filtering
- the effective cap must never exceed backend/runtime `max_graph_nodes`
- structured query responses should report the effective cap and truncation metadata
- if the safe base graph must be truncated before filtering, that fact must be visible to the client

Recommended metadata shape:

```json
{
  "meta": {
    "requested_max_nodes": 800,
    "effective_max_nodes": 800,
    "was_truncated_before_filtering": true,
    "was_truncated_after_filtering": false
  }
}
```

### 1b. Optional Filter Pushdown

The default implementation remains API-layer filtering, but the design should leave room for capable backends to optimize:

- graph storage backends are not required to implement advanced filters for correctness
- backends may optionally expose pushdown helpers for selected predicates
- the API layer may use those helpers when available to reduce memory pressure and latency

This preserves compatibility while allowing storage-specific optimization for large graphs.

### 2. Compatibility Route Preservation

The existing `GET /graphs` route remains in place.

Behavior:

- existing route is preserved for compatibility
- internally, it can adapt its inputs into the structured query shape
- legacy frontend flows remain functional during rollout

### 3. Graph-Domain Delete Route Normalization

Add new graph-domain delete endpoints in `graph_routes.py`:

- `DELETE /graph/entity`
- `DELETE /graph/relation`

These routes should delegate to the same underlying LightRAG deletion primitives already used by `document_routes.py`.

Compatibility strategy:

- keep `/delete_entity` and `/delete_relation` in `document_routes.py`
- treat them as compatibility aliases during migration
- avoid breaking existing external integrations

### 4. Merge Suggestion API

Add a dedicated merge suggestion endpoint rather than overloading graph query.

Recommended route:

- `POST /graph/merge/suggestions`

Recommended response shape:

```json
{
  "candidates": [
    {
      "target_entity": "Tesla",
      "source_entities": ["Tesla Inc.", "Tesla Motors"],
      "score": 0.87,
      "reasons": [
        {"code": "name_similarity", "score": 0.94},
        {"code": "shared_neighbors", "score": 0.73},
        {"code": "shared_sources", "score": 0.66}
      ]
    }
  ]
}
```

The first implementation should use explainable heuristics instead of opaque model-driven merging:

- normalized-name similarity
- edit distance / alias closeness
- same entity type
- overlapping neighbors
- overlapping `source_id`
- overlapping file-path evidence

The system should recommend candidates but never auto-execute a merge.

### 5. Concurrency Control for Mutations

Graph mutations should include optimistic concurrency protection.

Problem:

- multiple users may inspect the same node or relation in parallel
- a later mutation may silently overwrite an earlier one
- merge/delete actions are especially sensitive because they can remove graph structure

Design requirement:

- read responses for editable graph objects should expose a concurrency token
- mutation routes should accept that token back on edit/delete/merge requests
- the server should reject stale writes when the token no longer matches

Preferred token shape:

- `updated_at` if graph mutation timestamps are standardized across stores
- otherwise a dedicated `version` or opaque `revision_token`

This is a deliberate extension to the current graph contract rather than an already-existing capability.

## Frontend State Design

Do not overload `settings.ts` with workbench runtime state.

### Keep in `graph.ts`

- raw graph
- sigma graph
- selected/focused node and edge
- graph render update helpers
- expand/prune interaction state

### Keep in `settings.ts`

- persistent user preferences
- graph defaults such as max depth and max nodes
- panel/legend/visual toggles

### Add a new graph workbench store

Recommended ownership:

- filter drafts
- applied query model
- query result metadata
- create/delete/merge form state
- merge suggestion data
- selection queue for manual merge and relation creation
- error state specific to workbench operations

This keeps rendering state and workbench intent separate.

## Data-Flow Design

### Graph Query Flow

1. User edits filter draft in the workbench
2. User applies the filter set
3. Frontend sends a structured graph query to the API
4. API fetches a base graph using existing graph retrieval primitives
5. API applies advanced filters and view shaping
6. Frontend hydrates the response through `useLightragGraph.tsx`
7. `RawGraph` and Sigma graph are rebuilt or incrementally updated as appropriate

### Mutation Flow

Mutations remain server-authoritative:

- no optimistic writes that assume backend success
- local in-memory updates only after successful mutation responses

Update policy:

- use local partial updates for simple edits and local deletions/creations when safe
- force structured query refresh when operations can invalidate filter results or change graph identity:
  - rename
  - merge
  - source-filter-affecting edits
  - large delete operations
  - anything that may alter current result membership

All mutating flows should also propagate concurrency-token validation failures back to the frontend so the workbench can prompt the user to refresh rather than silently overwriting newer changes.

## Merge and Disambiguation Design

### Manual Merge

User flow:

1. Select one or more source entities
2. Select a target entity
3. Review retained values for the target
4. Confirm merge
5. Receive post-merge navigation options

### Suggested Merge

User flow:

1. Request suggestions for the current graph result or visible selection
2. Review candidate groups and evidence
3. Choose a candidate group
4. Open the same merge confirmation panel used by manual merge
5. Execute merge

### Shared Merge Constraints

- source and target entities must be explicit in the final confirmation
- the UI must show why the suggestion exists
- users must be able to reject or skip candidates
- merge failure states must be classified and rendered clearly

### Alias Preservation Requirement

Merge must preserve recall for historical and alternate names.

Problem:

- source entity names may still be the lexical form used in indexed source documents
- hard-removing the source label after merge can reduce future retrieval and graph lookup quality

Design requirement:

- each merged source entity name should be retained on the target entity as alias data
- aliases should live in a structured field such as `aliases`, or another canonical multi-value representation that does not lose the original names
- entity detail and merge-review UIs should surface retained aliases

This requirement applies to both manual merge and suggested merge flows.

## Internationalization and Multi-Language Requirements

This work must treat internationalization as a first-class requirement, not as follow-up polish.

### UI Strings

All new graph workbench UI strings must be added through `react-i18next` and stored in locale resources under:

- `lightrag_webui/src/locales/en.json`
- `lightrag_webui/src/locales/zh.json`
- `lightrag_webui/src/locales/zh_TW.json`
- `lightrag_webui/src/locales/fr.json`
- `lightrag_webui/src/locales/ar.json`
- `lightrag_webui/src/locales/ru.json`
- `lightrag_webui/src/locales/ja.json`
- `lightrag_webui/src/locales/de.json`
- `lightrag_webui/src/locales/uk.json`
- `lightrag_webui/src/locales/ko.json`
- `lightrag_webui/src/locales/vi.json`

No new graph workbench label, action text, or dialog copy should be hardcoded in components.

### Structured API Messages

New backend routes should return stable machine-readable codes for frontend localization, especially for:

- merge suggestion reason types
- validation failures
- delete confirmation summaries
- merge failure classes

Recommended pattern:

```json
{
  "code": "merge_target_missing",
  "message": "Target entity does not exist",
  "details": {...}
}
```

The frontend should localize based on `code`, using `message` as a fallback.

### Suggestion Reason Localization

Merge suggestion reasons must not be returned as pre-localized prose only.

Instead, return reason codes such as:

- `name_similarity`
- `shared_neighbors`
- `shared_sources`
- `shared_file_paths`
- `same_entity_type`

The frontend maps these to localized human-readable descriptions.

### Locale-Sensitive Formatting

Where the workbench displays numbers, counts, dates, or time ranges, formatting should follow the current UI locale instead of assuming a single language or regional style.

### Locale Loading Strategy

The current frontend i18n bootstrap eagerly imports all locale JSON files into the main bundle. The graph workbench should not make that cost worse without an optimization path.

Design requirement:

- graph workbench implementation must remain compatible with the current i18n setup
- the design should preserve a follow-up path to lazy-load locale payloads instead of always bundling every language into the initial load
- if lazy-loading is implemented during this work, it should be done at the shared `i18n.ts` layer rather than as a graph-only special case

### Translation Completeness Rule

Before merging implementation, all new graph workbench translation keys must exist in every currently supported locale file. Fallback behavior is acceptable for imperfect wording, but missing-key UI should not ship.

## Error Handling and Consistency Rules

### Consistency

- backend success is the source of truth
- frontend partial graph updates happen only after backend success
- operations that may invalidate result membership trigger a query refresh

### Error Handling

#### Create

- preserve form input on failure
- show field-specific or domain-specific errors when available

#### Delete

- do not remove anything from the graph view if the backend delete fails
- keep the confirmation state visible with an actionable error message

#### Merge

Merge failures should distinguish:

- suggestion generation failure
- merge validation failure
- merge execution failure
- post-merge refresh failure

These should not collapse into a single generic error.

### Destructive Operation Safeguards

The first version still excludes true undo/rollback, but destructive actions need a clear safety envelope:

- delete and merge flows must have explicit confirmation steps
- the API should return enough structured detail for the UI to preview impact before final submission
- implementation should preserve operational recovery paths such as export/log evidence and forward compatibility with tombstone-style recovery

Soft delete across graph storage and vector storage is a reasonable future direction, but it is intentionally deferred because it would require broader schema and query-contract changes across multiple storage backends.

## Testing Design

### Backend

Add a dedicated route-level test module:

- `tests/test_graph_routes.py`

Cover at least:

- structured graph query route
- graph-domain delete routes
- merge suggestion route
- legacy compatibility route behavior
- conflict detection for stale concurrency tokens
- hard-limit and truncation metadata behavior

Also expand graph-operation coverage around `utils_graph.py`:

- manual merge validation
- merge suggestion scoring logic
- delete response structure
- merge response structure
- alias preservation after merge

### Frontend

Add graph workbench tests covering at least:

- filter payload assembly
- AND/OR semantics projection for flat filter groups
- workbench tab switching
- merge suggestion selection into merge form
- delete confirmation behavior
- create form prefill behavior
- graph refetch trigger behavior after invalidating mutations
- i18n key coverage for new workbench UI
- stale-write conflict handling in edit/delete/merge flows

## Proposed File Impact

Expected primary change areas:

- `lightrag/api/routers/graph_routes.py`
- `lightrag/api/routers/document_routes.py`
- `lightrag/utils_graph.py`
- `lightrag/lightrag.py`
- `lightrag_webui/src/features/GraphViewer.tsx`
- `lightrag_webui/src/components/graph/PropertiesView.tsx`
- `lightrag_webui/src/components/graph/EditablePropertyRow.tsx`
- `lightrag_webui/src/hooks/useLightragGraph.tsx`
- `lightrag_webui/src/stores/graph.ts`
- `lightrag_webui/src/stores/settings.ts`
- new graph workbench components/stores under `lightrag_webui/src/components/graph/` and `lightrag_webui/src/stores/`
- locale files under `lightrag_webui/src/locales/`
- backend tests under `tests/`
- frontend tests under `lightrag_webui/src/**/*.test.ts*`

## Design Summary

This design intentionally keeps the graph storage contract stable, lifts advanced filtering into the API/workbench layer, and reshapes the graph page into a graph-governance workbench rather than a pure viewer. It treats disambiguation as a first-class user workflow and explicitly bakes multilingual requirements into both the UI and the API contract so the implementation does not regress into hardcoded single-language behavior.
