# LightRAG Prompt Customization Design

**Date:** 2026-03-25

**Status:** Approved for implementation planning

## Goal

Upgrade LightRAG's prompt system so prompt behavior is fully customizable across Python SDK, API, and WebUI, while preserving backward compatibility with the current `PROMPTS` dictionary and existing `user_prompt` / `system_prompt` usage.

## Scope

This design covers:

- Default prompt management in `lightrag/prompt.py`
- Instance-level prompt defaults via `LightRAG(...)`
- Per-request query-time prompt overrides via `QueryParam(...)` and API requests
- WebUI support for editing and sending query-time prompt overrides
- Prompt-aware validation and cache invalidation

This design does not cover:

- Persistent server-side storage of prompt presets
- Workspace-level prompt profile management
- Multi-user prompt permission controls
- WebUI control of indexing-time prompt behavior

## Current State

Today, prompt behavior is split across two different models:

1. `lightrag/prompt.py` exposes a global `PROMPTS` dictionary containing built-in templates for:
   - entity extraction
   - keyword extraction
   - query answering
   - query context formatting
   - summary generation
   - delimiter constants
2. Query-time customization is limited to:
   - `system_prompt`: ad hoc override for answer generation
   - `QueryParam.user_prompt`: extra answer-formatting instructions appended to the response prompt

This creates several problems:

- Core prompts are effectively hard-coded
- Different prompt families are customized through inconsistent mechanisms
- The API and WebUI only expose `user_prompt`, which is too narrow for advanced customization
- Cache keys do not consistently reflect prompt changes
- Prompt customization is not typed, validated, or scoped by lifecycle

## Design Principles

- Preserve existing behavior for users who do not adopt the new feature
- Use one structured prompt model instead of many ad hoc fields
- Separate indexing-time prompt configuration from query-time overrides
- Reject invalid prompt overrides early and explicitly
- Differentiate hard runtime failures from softer prompt quality regressions
- Keep WebUI focused on query-time customization only
- Treat API-exposed prompt overrides as a privileged capability, not a default-safe public feature
- Leave a clean extension path for future preset/profile support

## Prompt Taxonomy

The prompt system will be organized into the following families.

### Shared

Used across multiple flows:

- `tuple_delimiter`
- `completion_delimiter`

### Query

Used only while answering a query:

- `rag_response`
- `naive_rag_response`
- `kg_query_context`
- `naive_query_context`

### Keywords

Used only while extracting query keywords:

- `keywords_extraction`
- `keywords_extraction_examples`

### Entity Extraction

Used during indexing / document ingestion:

- `system_prompt`
- `user_prompt`
- `continue_prompt`
- `examples`

### Summary

Used while merging and summarizing entity / relationship descriptions:

- `summarize_entity_descriptions`

## Configuration Model

### Default Prompt Source

`lightrag/prompt.py` will remain the canonical home of built-in default templates. The existing `PROMPTS` dictionary will continue to exist for compatibility, but it will no longer be the only runtime interface.

The file will be upgraded to expose:

- default prompt config generation
- structured merge utilities
- prompt validation utilities
- prompt fingerprint helpers for cache keys

### Instance-Level Prompt Defaults

`LightRAG` will gain a new field:

```python
prompt_config: dict[str, Any] = field(default_factory=dict)
```

This config represents instance-wide prompt defaults and may override any built-in prompt family, including indexing-time prompts.

Typical usage:

```python
rag = LightRAG(
    ...,
    prompt_config={
        "query": {
            "rag_response": "...custom template..."
        },
        "entity_extraction": {
            "system_prompt": "...custom extractor system prompt..."
        }
    }
)
```

### Per-Request Query Overrides

`QueryParam` will gain a new field:

```python
prompt_overrides: dict[str, Any] | None = None
```

This field is intentionally narrower than `LightRAG.prompt_config`.

Allowed families:

- `query`
- `keywords`

Disallowed families:

- `entity_extraction`
- `summary`
- `shared`

Reason:

- Query requests should only override behavior that actually executes inside the current query lifecycle
- Allowing indexing-time prompt overrides at request time would create misleading UX because existing indexed data would not be reprocessed

### Effective Prompt Resolution

Runtime prompt resolution will follow this order:

1. Built-in defaults from `lightrag/prompt.py`
2. `LightRAG.prompt_config`
3. `QueryParam.prompt_overrides`
4. Legacy `system_prompt` shortcut, mapped only onto the final answer template

Legacy `system_prompt` behavior:

- `kg_query` maps it to `query.rag_response`
- `naive_query` maps it to `query.naive_rag_response`

Legacy `user_prompt` behavior:

- stays unchanged
- remains an answer-stage additive instruction injected into the answer template
- if a custom answer template omits `{user_prompt}` and `user_prompt` is non-empty, the runtime appends a standard additional-instructions block to preserve compatibility

## Prompt Validation

Structured prompt overrides must be validated before execution.

### Validation Levels

Validation is split into two layers.

#### Structural Validation

These checks protect runtime correctness and must reject invalid input:

- Unknown top-level family: reject
- Unknown prompt key inside a family: reject
- Wrong value type: reject
- Invalid examples type for list-based templates: reject
- Empty delimiter values: reject

#### Semantic Validation

These checks protect prompt quality and feature preservation:

- Missing strictly required placeholders: reject
- Missing recommended placeholders: warn

Warnings should be surfaced through logs and, for API requests, returned as structured warning metadata where practical.

### Placeholder Rules

#### Strictly Required Placeholders

These placeholders are required because omitting them would remove essential runtime input for the operation:

- `query.rag_response`
  - `{context_data}`
- `query.naive_rag_response`
  - `{content_data}`
- `keywords.keywords_extraction`
  - `{query}`
- `entity_extraction.system_prompt`
  - `{tuple_delimiter}`
  - `{completion_delimiter}`
- `entity_extraction.user_prompt`
  - `{input_text}`
- `summary.summarize_entity_descriptions`
  - `{description_list}`

For `query.kg_query_context`, the template must include at least one context-bearing placeholder from:

- `{entities_str}`
- `{relations_str}`
- `{text_chunks_str}`
- `{reference_list_str}`

#### Recommended Placeholders

These placeholders are recommended because they preserve useful behavior, but their absence should not block advanced customization:

- `query.rag_response`
  - `{response_type}`
  - `{user_prompt}`
- `query.naive_rag_response`
  - `{response_type}`
  - `{user_prompt}`
- `query.kg_query_context`
  - `{entities_str}`
  - `{relations_str}`
  - `{text_chunks_str}`
  - `{reference_list_str}`
- `keywords.keywords_extraction`
  - `{examples}`
  - `{language}`
- `entity_extraction.system_prompt`
  - `{entity_types}`
  - `{examples}`
  - `{language}`
- `entity_extraction.user_prompt`
  - `{entity_types}`
  - `{completion_delimiter}`
  - `{language}`
- `summary.summarize_entity_descriptions`
  - `{description_type}`
  - `{description_name}`
  - `{summary_length}`
  - `{language}`

### Error Reporting

Validation failures should produce precise errors that are easy to debug:

- unknown family / key
- missing placeholders
- invalid type
- invalid empty value
- missing recommended placeholders

At the API layer:

- structural failures and strictly required placeholder failures should surface as 4xx errors with machine-readable detail
- missing recommended placeholders should surface as warnings, not request rejection

## Runtime Integration

### Query Answering

The following paths must resolve prompts from the effective structured config instead of directly indexing `PROMPTS`:

- `kg_query`
- `naive_query`
- `_build_final_query_context`

### Keyword Extraction

`extract_keywords_only` must use:

- effective `keywords_extraction`
- effective `keywords_extraction_examples`

### Entity Extraction

`extract_entities` must use instance-level effective config only:

- `entity_extraction.system_prompt`
- `entity_extraction.user_prompt`
- `entity_extraction.continue_prompt`
- `entity_extraction.examples`
- shared delimiters

### Summary Generation

The summary generation path must use effective `summary.summarize_entity_descriptions`.

## Cache Strategy

Prompt changes must affect cache identity.

### Why

Today the same query can incorrectly reuse cached results even after prompt behavior changes. This is unsafe once prompts become customizable.

### Required Change

Cache keys for all prompt-sensitive operations must include a prompt fingerprint derived from the effective prompt config relevant to that operation.

Recommended granularity:

- query answer cache:
  - query family fingerprint
  - query-time `user_prompt`
- keyword extraction cache:
  - keywords family fingerprint
- entity extraction cache:
  - entity extraction family fingerprint
  - shared delimiter fingerprint
- summary cache:
  - summary family fingerprint

The fingerprint should be deterministic and normalized so logically identical configs produce the same hash.

### Indexing Cost Note

Prompt fingerprinting affects indexing-time caches too.

Implications:

- changing `LightRAG.prompt_config` for `entity_extraction` or `summary` changes the effective cache identity for future indexing operations
- re-inserting or reprocessing the same documents under a new indexing prompt config may trigger fresh LLM work and additional API cost
- existing indexed graph data is not automatically rewritten when prompt config changes

This phase does not add a selective cache-reuse policy. Instead, the system should:

- document this behavior clearly
- log it clearly when indexing-time prompt config differs from prior runs

Future work may add an explicit policy switch such as "reuse old indexing cache" vs "force prompt-consistent reprocessing", but that is out of scope for this iteration.

## API Design

`lightrag/api/routers/query_routes.py` will add:

```python
prompt_overrides: Optional[Dict[str, Any]] = Field(
    default=None,
    description="Structured prompt overrides for query-time prompt families."
)
```

Accepted families:

- `query`
- `keywords`

The request model should convert this directly into `QueryParam.prompt_overrides`, but only when the API is explicitly allowed to accept prompt overrides.

### API Safety Gate

Because prompt overrides can weaken or bypass system-level prompting constraints, API support must be protected by a server-side capability flag:

```python
allow_prompt_overrides_via_api: bool = False
```

Behavior:

- default: disabled
- when disabled:
  - API rejects `prompt_overrides`
  - SDK usage remains unaffected
  - WebUI should hide or disable the prompt override editor
- when enabled:
  - API accepts `prompt_overrides` for trusted deployments

Recommended error when disabled:

- HTTP 403 or equivalent client-visible configuration error
- clear message that request-level prompt overrides are not enabled on this server

The server should also expose this capability to trusted clients, for example through an existing status/config endpoint, so WebUI can determine whether prompt override editing should be interactive.

### JSON Schema Notes

List-style prompt fields use `List[str]`.

Example request:

```json
{
  "query": "请总结知识库的关键主题",
  "mode": "mix",
  "user_prompt": "请输出为中文要点",
  "prompt_overrides": {
    "query": {
      "rag_response": "...custom answer template..."
    },
    "keywords": {
      "keywords_extraction": "...custom keyword extractor...",
      "keywords_extraction_examples": [
        "Example 1: ...",
        "Example 2: ..."
      ]
    }
  }
}
```

## WebUI Design

### Scope Boundary

The WebUI will support query-time prompt customization only.

It will not edit:

- entity extraction prompts
- summary prompts
- shared delimiters
- instance-level server defaults

Those remain controlled through SDK usage and service startup configuration.

### State Model

`lightrag_webui/src/api/lightrag.ts`

- extend `QueryRequest` with `prompt_overrides`

`lightrag_webui/src/stores/settings.ts`

- persist `querySettings.prompt_overrides`
- keep existing `user_prompt`
- add migration for older stored settings
- preserve local draft overrides even when the server later disables API-side prompt overrides

### UI Model

Keep the existing `user_prompt` field as the lightweight answer-formatting control.

Add a new collapsible section in retrieval query settings:

- section title: Prompt Customization
- enabled by default only when non-empty overrides exist
- only interactive when the server advertises that API prompt overrides are enabled

The section should support two levels:

#### Basic Mode

Expose the most common fields:

- `query.rag_response`
- `query.naive_rag_response`
- `keywords.keywords_extraction`

#### Advanced Mode

Expose all query-time prompt override fields:

- `query.rag_response`
- `query.naive_rag_response`
- `query.kg_query_context`
- `query.naive_query_context`
- `keywords.keywords_extraction`
- `keywords.keywords_extraction_examples`

### Editing Behavior

Each editable prompt item should provide:

- enable / disable control
- editor area
- restore default action
- required placeholders hint

For list-style templates such as `keywords_extraction_examples`, the UI should use multi-entry editing instead of forcing a single large text blob.

### Request Behavior

Both `/query` and `/query/stream` request paths must send `querySettings.prompt_overrides` when present.

If the server disables API prompt overrides:

- the editor should be hidden or read-only
- locally persisted draft data may remain in storage
- the UI should not attempt to submit those overrides

### Debug Experience

When `only_need_prompt` is enabled, the response should represent the fully resolved prompt after merging defaults, instance config, request overrides, and legacy fields. This allows the WebUI to act as a reliable prompt-debug surface.

## Backward Compatibility

The following behaviors must remain unchanged for existing users:

- no prompt config supplied -> current runtime behavior remains the same
- `QueryParam.user_prompt` continues to work
- Python callers may still pass `system_prompt` to `query()` / `aquery()` / `aquery_llm()`
- `PROMPTS` continues to exist as the built-in prompt source
- when an answer template omits `{user_prompt}`, non-empty `user_prompt` still takes effect via runtime append behavior

Compatibility interpretation:

- old code keeps working
- new code gets a structured and validated prompt layer

## Error Handling

### Python SDK

Invalid `prompt_config` or `prompt_overrides` should raise explicit value errors before LLM execution begins.

### API

Invalid request prompt overrides should return a client error response with detailed messages.

If API prompt overrides are disabled by server policy, requests containing `prompt_overrides` should be rejected before prompt merge and validation proceeds.

### WebUI

If the backend rejects a prompt override:

- keep the user's editor content intact
- show the returned validation error
- do not silently clear fields

## File Impact

Expected primary file touch points:

- `lightrag/prompt.py`
- `lightrag/base.py`
- `lightrag/lightrag.py`
- `lightrag/operate.py`
- `lightrag/api/routers/query_routes.py`
- `lightrag_webui/src/api/lightrag.ts`
- `lightrag_webui/src/stores/settings.ts`
- `lightrag_webui/src/components/retrieval/QuerySettings.tsx`
- WebUI locale files for new UI text
- tests for backend and frontend prompt customization behavior

## Testing Strategy

### Backend

- prompt config merge unit tests
- prompt validation unit tests
- placeholder validation tests
- query prompt override tests for `kg_query`
- query prompt override tests for `naive_query`
- keyword extraction prompt override tests
- cache key changes when prompt fingerprints change
- `only_need_prompt` returns fully resolved prompt
- invalid override API response tests

### Frontend

- request typing tests for `prompt_overrides`
- Zustand migration tests
- Query settings interaction tests
- request payload tests for `/query` and `/query/stream`
- validation error rendering tests
- restore default behavior tests

## Rollout Plan

1. Introduce structured prompt config and validation in `lightrag/prompt.py`
2. Add `LightRAG.prompt_config`
3. Add `QueryParam.prompt_overrides`
4. Refactor prompt consumers in `operate.py`
5. Update cache key calculation to include prompt fingerprints
6. Extend API request model
7. Extend WebUI types, store, and query settings panel
8. Add regression tests
9. Update documentation and examples

## Future Extension Path

This design intentionally leaves room for a future preset/profile system by treating prompt configuration as a structured document rather than a collection of ad hoc fields.

Possible future work:

- named prompt presets
- import/export prompt bundles
- workspace-level prompt persistence
- prompt version comparison
- admin-controlled prompt policy

## Approved Decisions

- Prompt customization must cover Python SDK, API, and WebUI
- Instance-level defaults may customize all prompt families
- Request-level overrides are limited to query-time families
- WebUI supports query-time prompt overrides only
- Existing `user_prompt` remains as an additive answer instruction
- Existing `system_prompt` remains as a legacy compatibility shortcut
