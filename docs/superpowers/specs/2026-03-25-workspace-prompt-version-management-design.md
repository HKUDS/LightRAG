# LightRAG Workspace Prompt Version Management Design

**Date:** 2026-03-25

**Status:** Approved for implementation planning

## Goal

Add workspace-scoped, server-persisted prompt version management to LightRAG so users can manage formal prompt versions from WebUI, separate indexing-related configuration from retrieval-related configuration, and reuse saved retrieval versions as request-scoped temporary overrides during retrieval testing.

## Scope

This design covers:

- Workspace-scoped prompt version persistence on the server
- Two independent configuration groups:
  - indexing configuration
  - retrieval configuration
- WebUI prompt management as a top-level page between Knowledge Graph and Retrieval
- Version metadata, activation, copy, diff, deletion, and comments
- Retrieval-page temporary version selection built on existing `prompt_overrides`
- Versioned management of `ENTITY_TYPES`
- Versioned management of indexing prompt families and retrieval prompt families
- Safe fallback to existing LightRAG behavior when no version is activated

This design does not cover:

- Multi-user permissions or approval workflows
- Audit logs or review histories
- Automatic re-indexing when indexing configuration changes
- Database-backed persistence for prompt versions in the first iteration

## Current State

Today, prompt behavior is split across two runtime layers:

1. `lightrag/prompt.py` defines the built-in prompt schema and default prompt values
2. Query-time prompt overrides are limited to `query` and `keywords`

The current runtime boundary is already explicit:

- Query-time override families:
  - `query`
  - `keywords`
- Indexing-time families:
  - `shared`
  - `entity_extraction`
  - `summary`

`ENTITY_TYPES` is not part of `PROMPT_SCHEMA`, but it is passed through `addon_params` and directly affects entity extraction. It therefore belongs with indexing-time configuration from a user mental-model perspective.

The current WebUI only exposes query-time prompt override editing inside Retrieval settings. That is useful for debugging, but it is not sufficient for:

- persistent prompt profiles
- workspace-level formal configuration
- version switching
- indexing prompt management
- managing `ENTITY_TYPES`

## Design Principles

- Keep the runtime boundary honest: retrieval testing must not pretend to override indexing-only behavior
- Separate formal configuration management from one-off query experimentation
- Make workspace the isolation boundary for prompt versions
- Preserve current LightRAG behavior whenever no version is activated
- Keep initial persistence simple and file-based
- Use full snapshots for each saved version, not patch-only records
- Treat version activation and version saving as separate actions

## Configuration Groups

Prompt management will be split into two independent groups, each with its own version chain and active version pointer.

### Indexing Configuration

This group controls values that affect document ingestion, entity extraction, and summarization.

Included fields:

- `ENTITY_TYPES`
- `SUMMARY_LANGUAGE`
- `shared.tuple_delimiter`
- `shared.completion_delimiter`
- `entity_extraction.system_prompt`
- `entity_extraction.user_prompt`
- `entity_extraction.continue_prompt`
- `entity_extraction.examples`
- `summary.summarize_entity_descriptions`

Behavioral notes:

- Activation affects future indexing and summarization work
- Activation does not rewrite already indexed data
- Retrieval-page temporary override does not apply to this group

### Retrieval Configuration

This group controls values that affect query answering and keyword extraction.

Included fields:

- `query.rag_response`
- `query.naive_rag_response`
- `query.kg_query_context`
- `query.naive_query_context`
- `keywords.keywords_extraction`
- `keywords.keywords_extraction_examples`

Behavioral notes:

- Activation affects future retrieval calls for the workspace
- Retrieval-page temporary override may select a saved retrieval version for a single request
- Temporary selection must project only this group into `prompt_overrides`

## Version Model

Each configuration group stores versions independently.

Suggested version record:

```python
{
    "version_id": "uuid-or-stable-id",
    "group_type": "indexing" | "retrieval",
    "version_name": "Human readable name",
    "version_number": 1,
    "comment": "Optional description of what changed",
    "source_version_id": "optional parent version id",
    "created_at": "ISO8601 timestamp",
    "payload": {
        # indexing or retrieval config snapshot
    }
}
```

Suggested group registry:

```python
{
    "workspace": "default",
    "group_type": "indexing" | "retrieval",
    "active_version_id": "optional version id",
    "versions": [...]
}
```

### Why full snapshots

Each version should store the full effective payload for its group, not just a diff, because that makes:

- activation simple
- diff generation deterministic
- copying cheap
- future migration safer

## Fallback Rules

Fallback behavior is mandatory and must preserve current LightRAG semantics.

### No active version

If a configuration group has no active version, LightRAG must continue to use the existing built-in path:

- built-in defaults from `lightrag/prompt.py`
- existing env-driven `ENTITY_TYPES`
- existing env-driven `SUMMARY_LANGUAGE`
- any existing instance-level `prompt_config` behavior that remains in scope

Prompt version management must not become a hard prerequisite for normal operation.

### Version registry exists but active pointer is empty

Even if saved versions exist, if `active_version_id` is empty for a group:

- that group falls back to the original runtime behavior
- saved versions are treated as available drafts/presets
- nothing changes until the user explicitly activates one

### Retrieval temporary override

Retrieval-page temporary version selection must never mutate the active retrieval version pointer.

## Chinese Seed Versions

When a workspace first opens Prompt Management and no version registry exists yet, the server should generate two seed version chains:

- one indexing seed version
- one retrieval seed version

These seed versions must be populated from curated Chinese prompt templates, not from the English built-in defaults shown directly to the user.

Requirements:

- Placeholder structure must remain compatible with the current schema
- List fields must remain list fields
- The seed versions are saved but not automatically activated
- If the user never activates any version, runtime still uses original LightRAG behavior

Suggested seed names:

- `indexing-zh-default`
- `retrieval-zh-default`

Suggested seed comments:

- `中文初始版本`

## Persistence Strategy

First iteration should use file persistence under the workspace runtime directory.

Suggested shape:

- one file per workspace per group
- or one workspace file containing both groups

Either of the following is acceptable:

```text
<working_dir>/<workspace>/prompt_versions/indexing.json
<working_dir>/<workspace>/prompt_versions/retrieval.json
```

or

```text
<working_dir>/<workspace>/prompt_versions/registry.json
```

Selection criteria:

- easy atomic writes
- easy backup
- easy manual inspection
- minimal coupling to a specific storage backend

## Runtime Resolution

### Indexing runtime

Indexing runtime should resolve configuration in this order:

1. Current built-in defaults
2. Current env-backed `ENTITY_TYPES` / `SUMMARY_LANGUAGE`
3. Active indexing version payload, if any

The indexing version payload should override:

- `shared`
- `entity_extraction`
- `summary`
- `ENTITY_TYPES`
- `SUMMARY_LANGUAGE`

If no indexing version is active, runtime remains unchanged.

### Retrieval runtime

Retrieval runtime should resolve configuration in this order:

1. Current built-in defaults
2. Active retrieval version payload, if any
3. Request-scoped temporary retrieval override, if any

The temporary retrieval override is derived from a selected saved retrieval version and projected into the existing query request `prompt_overrides` model.

Because `prompt_overrides` only supports `query` and `keywords`, Retrieval-page temporary selection must only send:

- `query.*`
- `keywords.*`

## API Design

Formal version management should use dedicated APIs rather than extending the existing query route.

### Read group summary

`GET /prompt-config/groups`

Returns, per workspace:

- available groups
- active version ids
- version counts
- whether seed versions exist

### List versions for a group

`GET /prompt-config/{group_type}/versions`

Returns:

- `active_version_id`
- version list with metadata

### Read a version

`GET /prompt-config/{group_type}/versions/{version_id}`

Returns:

- metadata
- full payload

### Create a version

`POST /prompt-config/{group_type}/versions`

Request body:

- `version_name`
- `comment`
- `payload`
- optional `source_version_id`

Behavior:

- validate payload against the allowed fields for the group
- assign next `version_number`
- save as a new version
- do not activate automatically

### Activate a version

`POST /prompt-config/{group_type}/versions/{version_id}/activate`

Behavior:

- set active version pointer for that workspace and group
- return resulting active metadata

### Delete a version

`DELETE /prompt-config/{group_type}/versions/{version_id}`

Behavior:

- reject deletion of the active version
- allow deletion of inactive versions

### Diff a version

`GET /prompt-config/{group_type}/versions/{version_id}/diff?base_version_id=...`

Returns grouped differences by field.

### Seed initialization

`POST /prompt-config/initialize`

Behavior:

- create Chinese seed versions if no registry exists yet
- no-op if already initialized

This endpoint can also be triggered lazily by the first read request.

## Validation Rules

### Shared prompt validation

Continue reusing `validate_prompt_config()` for prompt families already defined in `PROMPT_SCHEMA`.

### Group-specific validation

- Retrieval group accepts only:
  - `query`
  - `keywords`
- Indexing group accepts only:
  - `shared`
  - `entity_extraction`
  - `summary`

### Additional non-schema validation

Because `ENTITY_TYPES` and `SUMMARY_LANGUAGE` live outside `PROMPT_SCHEMA`, add explicit validation:

- `ENTITY_TYPES` must be `list[str]`
- empty list should be rejected
- blank items should be rejected
- `SUMMARY_LANGUAGE` must be non-empty `str`

## WebUI Information Architecture

Top navigation order becomes:

- `Documents`
- `Knowledge Graph`
- `Prompt Management`
- `Retrieval`
- `API`

### Prompt Management page

The page uses the selected A-layout: control-console style two-column layout.

Left column:

- workspace indicator
- group switcher:
  - indexing configuration
  - retrieval configuration
- version list
- active version badge
- create new version
- copy from version

Right column:

- version metadata header
- fields:
  - version name
  - version number
  - optional comment
  - source version
- actions:
  - save as new version
  - view diff
  - activate version
- grouped editor sections

### Field rendering

Render fields by actual type:

- single-line input:
  - `version_name`
  - `SUMMARY_LANGUAGE`
  - `shared.*`
- multi-line editor:
  - prompt templates
- list editor:
  - `ENTITY_TYPES`
  - `keywords.keywords_extraction_examples`
  - `entity_extraction.examples`
- optional multi-line note:
  - `comment`

### Diff UI

Diff opens in a drawer or modal and compares:

- selected version
- active version by default

## Retrieval Page Design

Retrieval page no longer directly edits prompt templates.

Instead it exposes a retrieval-version temporary selector:

- default option: use active retrieval version
- saved options: retrieval versions for current workspace
- label must clearly state:
  - `Only applies to this request`

Behavior:

- selecting a saved retrieval version does not activate it
- the selected version is projected to request `prompt_overrides`
- only `query` and `keywords` are sent

The page should also keep a text link or subtle entry point to Prompt Management for formal editing, but it is not the primary control.

## User Messaging

The UI must clearly distinguish these meanings:

- save version
- activate version
- temporary retrieval override

Required guidance:

- indexing activation affects future indexing only
- retrieval activation affects subsequent retrievals for the workspace
- retrieval temporary override affects only the current request
- if no active version exists, the system is using original built-in behavior

## Compatibility and Migration

This feature should be additive.

- Existing SDK/API users continue to work unchanged
- Existing `prompt_overrides` stays intact
- Existing `.env` values still matter when no active indexing version exists
- Existing built-in prompt defaults still matter when no active version exists

No mandatory migration of current data is required.

## Testing Strategy

Backend tests should cover:

- seed initialization
- empty registry fallback
- activation per group
- indexing payload validation
- retrieval payload validation
- `ENTITY_TYPES` validation
- diff generation
- inactive-only deletion
- retrieval temporary override projection from saved version
- no mutation of active version during retrieval temporary selection

Frontend tests should cover:

- navigation tab insertion order
- group switching
- version list rendering
- comment optionality
- list-field editors
- retrieval temporary selector behavior
- disabled or empty-state messaging when no versions exist

## Open Implementation Note

This design intentionally keeps version management file-based in iteration one. If future work wants database-backed prompt registries, the API contract and WebUI flow should remain stable while only the repository layer changes.
