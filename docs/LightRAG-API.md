# LightRAG HTTP API

This document describes the REST API exposed by the LightRAG server, including
workspace isolation, document ingestion, retrieval, graph operations and the
Ollama-compatible endpoints.

## 1. Connection and conventions

Set the server URL once and use it in the examples below:

```bash
export LIGHTRAG_URL="http://localhost:9621"
```

The port is configurable. Swagger UI and the machine-readable contract are
available at `$LIGHTRAG_URL/docs`, `$LIGHTRAG_URL/redoc` and
`$LIGHTRAG_URL/openapi.json`.

### Authentication

When authentication is enabled, send either an API key or a JWT on every
protected request:

```http
X-API-Key: <LIGHTRAG_API_KEY>
```

or:

```http
Authorization: Bearer <access_token>
```

Obtain a JWT with a form-encoded login request:

```bash
curl -sS -X POST "$LIGHTRAG_URL/login" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'username=<username>' \
  --data-urlencode 'password=<password>'
```

The response contains `access_token`, `token_type` (`bearer`), and server
version metadata. If authentication is disabled, `/auth-status` and `/login`
return a guest token; protected APIs can also be called without a credential.
Do not expose API keys or passwords in client-side code.

### Workspace selection

All document, query, graph and Ollama data-plane endpoints accept this optional
header:

```http
X-LightRAG-Workspace-ID: <workspace UUID>
```

If it is omitted, the configured default workspace is used. Workspace CRUD
endpoints under `/v1/workspaces` do not use this header. A workspace is an
isolation boundary: its files, document status, KV data, vectors, graph and
query results are independent from every other workspace.

The examples use these shell variables:

```bash
export AUTH_HEADER="X-API-Key: $LIGHTRAG_API_KEY" # omit when auth is disabled
export WORKSPACE_ID="<workspace UUID>"
```

## 2. Workspace management

### Create a workspace

`POST /v1/workspaces` (201)

Request JSON:

```json
{
  "display_name": "Finance 2025",
  "description": "Annual reports and audit documents"
}
```

`display_name` is required (1–255 characters). `description` is optional and
may be null (maximum 10,000 characters).

Response:

```json
{
  "id": "7b2e7ef9-7a25-4f7d-9d4b-b3c0e15d4b47",
  "display_name": "Finance 2025",
  "description": "Annual reports and audit documents",
  "status": "active",
  "is_default": false,
  "created_at": "2026-07-19T14:00:00+00:00",
  "updated_at": "2026-07-19T14:00:00+00:00",
  "deleted_at": null
}
```

### List workspaces

`GET /v1/workspaces?include_deleted=false`

The default excludes workspaces whose deletion has completed. Pass
`include_deleted=true` to include retained registry records. The response is an
array of `WorkspaceResponse` objects.

### Get a workspace

`GET /v1/workspaces/{workspace_id}`

Returns one `WorkspaceResponse` or `404` if the workspace does not exist. The
default workspace cannot be deleted, but its display metadata can be updated.

### Update a workspace

`PATCH /v1/workspaces/{workspace_id}`

Send at least one field:

```json
{
  "display_name": "Finance - audited",
  "description": null
}
```

The response is the updated `WorkspaceResponse`.

### Delete a workspace

`DELETE /v1/workspaces/{workspace_id}` (202)

Deletion is asynchronous and removes the workspace's documents, files,
document status, vectors, graph and other storage data:

```json
{
  "status": "deleting",
  "workspace_id": "7b2e7ef9-7a25-4f7d-9d4b-b3c0e15d4b47",
  "job_id": "e5bfdd7c-c28e-45e7-9df1-36fbbbece9f6",
  "message": "Workspace deletion has been scheduled."
}
```

Do not send new data-plane requests to a workspace after deletion is
scheduled. The default workspace is protected (`409`).

## 3. Documents and ingestion

Ingestion is asynchronous. A successful upload or text insertion returns a
`track_id`; poll `/documents/track_status/{track_id}` or use the paginated
status endpoint before querying.

### Upload a file

`POST /documents/upload` (multipart form-data)

```bash
curl -sS -X POST "$LIGHTRAG_URL/documents/upload" \
  -H "$AUTH_HEADER" \
  -H "X-LightRAG-Workspace-ID: $WORKSPACE_ID" \
  -F 'file=@./annual-report.pdf'
```

The server chooses the configured parser by extension. In the standard setup,
PDF and other MinerU-supported complex documents use the MinerU service with
quality-focused options, while DOCX/PPTX/XLSX and simple Markdown/TXT/HTML use
native/legacy parsing. The response is:

```json
{
  "status": "success",
  "message": "Document uploaded and queued for processing",
  "track_id": "a8ad1d72-5fe2-4d8a-8c3d-d7fa5bb12c22"
}
```

The original filename is used for deduplication within the selected workspace.
Duplicate names return `409`; unsupported extensions return `400`; files over
the configured limit return `413`.

### Insert one text document

`POST /documents/text` (JSON)

```json
{
  "text": "The company reported revenue of ...",
  "file_source": "manual-note-2025.txt",
  "chunking": {
    "strategy": "recursive_character",
    "params": {
      "chunk_token_size": 1200,
      "chunk_overlap_token_size": 100,
      "separators": ["\n\n", "\n", ". ", " "]
    }
  }
}
```

`text` and `file_source` must be non-empty. `chunking` is optional. Supported
strategies are `fixed_token`, `recursive_character`, `semantic_vector` and
`paragraph_semantic`. Fixed/recursive/paragraph strategies accept token size
and overlap; semantic vector additionally accepts a breakpoint threshold type
(`percentile`, `standard_deviation`, `interquartile` or `gradient`), threshold
amount, buffer size and sentence regex. Invalid combinations return `422`.

### Insert multiple text documents

`POST /documents/texts` (JSON)

```json
{
  "texts": ["First document", "Second document"],
  "file_sources": ["first.txt", "second.txt"],
  "chunking": {"strategy": "fixed_token", "params": {"chunk_token_size": 800}}
}
```

`texts` must be non-empty. `file_sources`, when supplied, must have the same
length as `texts` and unique names. The response shape is the same as a single
insertion (`status`, `message`, `track_id`).

### Scan, retry and cancel processing

| Method and path | Purpose | Typical response |
| --- | --- | --- |
| `POST /documents/scan` | Discover files in the workspace input directory and enqueue them | `{"status":"scanning_started","message":"...","track_id":"..."}` |
| `POST /documents/reprocess_failed` | Retry failed documents | `{"status":"reprocessing_started","message":"...","track_id":""}` |
| `POST /documents/cancel_pipeline` | Request cancellation of the active pipeline | `{"status":"cancellation_requested","message":"..."}` or `not_busy` |
| `POST /documents/clear_cache` | Clear the current workspace's LLM cache | `{"status":"success","message":"..."}` |

`/documents/scan` can return `scanning_skipped_pipeline_busy` when another
exclusive operation is active.

### Poll a track

`GET /documents/track_status/{track_id}`

```json
{
  "track_id": "a8ad1d72-5fe2-4d8a-8c3d-d7fa5bb12c22",
  "documents": [
    {
      "id": "doc-123",
      "content_summary": "Annual financial report ...",
      "content_length": 182340,
      "status": "PROCESSED",
      "created_at": "2026-07-19T14:01:00+00:00",
      "updated_at": "2026-07-19T14:03:10+00:00",
      "track_id": "a8ad1d72-5fe2-4d8a-8c3d-d7fa5bb12c22",
      "chunks_count": 96,
      "error_msg": null,
      "metadata": {},
      "file_path": "annual-report.pdf"
    }
  ],
  "total_count": 1,
  "status_summary": {"PROCESSED": 1}
}
```

Document status values include `PENDING`, `PARSING`, `ANALYZING`,
`PROCESSING`, `PREPROCESSED`, `PROCESSED` and `FAILED`. Query only after the
required documents are `PROCESSED`; inspect `error_msg` for failures.

### Browse document status

Preferred endpoint: `POST /documents/paginated`

```json
{
  "status_filters": ["PROCESSED", "FAILED"],
  "page": 1,
  "page_size": 50,
  "sort_field": "updated_at",
  "sort_direction": "desc"
}
```

`page_size` is 10–200. The response contains `documents`, a `pagination`
object (`page`, `page_size`, `total_count`, `total_pages`, `has_next`,
`has_prev`) and `status_counts`.

Other status endpoints:

- `GET /documents/status_counts` returns `{"status_counts": {"PROCESSED": 12, ...}}`.
- `GET /documents/pipeline_status` returns operational fields such as `busy`,
  `job_name`, `docs`, `batchs`, `cur_batch`, `request_pending`,
  `latest_message` and `update_status`.
- `GET /documents` is a legacy all-status response grouped under `statuses`;
  use the paginated endpoint for new integrations.

### Delete documents

Delete selected documents asynchronously with `DELETE /documents/delete_document`:

```json
{
  "doc_ids": ["doc-123", "doc-456"],
  "delete_file": true,
  "delete_llm_cache": false
}
```

The response is `{ "status": "deletion_started", "message": "...", "doc_id": "doc-123" }`.
The status can be `busy` or `not_allowed` when the pipeline state does not
permit deletion. To clear every document in the current workspace, use
`DELETE /documents` (the operation is destructive and may return `busy` while
another pipeline is running).

## 4. RAG query and retrieval

### Complete JSON response

`POST /query`

```json
{
  "query": "What was the revenue in 2025?",
  "mode": "mix",
  "top_k": 60,
  "chunk_top_k": 20,
  "enable_rerank": true,
  "include_references": true,
  "include_chunk_content": false,
  "include_content_blocks": true,
  "include_graph_context": false,
  "conversation_history": [
    {"role": "user", "content": "Summarize the report."},
    {"role": "assistant", "content": "It covers fiscal year 2025."}
  ]
}
```

`query` must contain at least three characters. `mode` is one of `local`,
`global`, `hybrid`, `naive`, `mix` or `bypass`; `mix` is the usual combined
retrieval mode. All token/top-k fields are positive integers. `response_type`
and `user_prompt` can customize generation. `hl_keywords` and `ll_keywords`
are optional keyword lists.

Response:

```json
{
  "response": "Revenue in 2025 was ...",
  "references": [
    {"reference_id": "ref-1", "file_path": "annual-report.pdf"}
  ]
}
```

Set `include_chunk_content=true` to include a `content` array in each
reference. Set `only_need_context` or `only_need_prompt` when an application
needs retrieval material instead of a generated answer.

`include_content_blocks` and `include_graph_context` default to `false`. When
enabled, the response keeps the existing keys and adds normalized structures:

```json
{
  "content_blocks": [
    {"type": "markdown", "markdown": "The answer ..."},
    {"type": "image", "asset_id": "ast_01J...", "caption": "Model architecture", "page": 4},
    {"type": "table", "html": "<table>...</table>", "caption": "Results"},
    {"type": "equation", "latex": "A = \\sigma(Wx+b)"}
  ],
  "graph_context": {"nodes": ["ClinicalModernBERT"], "edges": ["rel_758"]}
}
```

References preserve `reference_id` and `file_path`; when document metadata is
available they additionally contain `document_id`, `display_name` and `page`.

### Streaming response

`POST /query/stream` accepts the same JSON request and returns
`application/x-ndjson`. Each line is an independent JSON object. The first line
may contain `references`, `content_blocks` and `graph_context`; following lines
contain `response` fragments, and a failure is represented by
`{"error":"..."}`. Use `stream:false` if a single complete JSON line is
preferred.

### Structured retrieval data

`POST /query/data` accepts the same query body and returns:

```json
{
  "status": "success",
  "message": "Query executed successfully",
  "data": {"entities": [], "relationships": [], "chunks": [], "references": []},
  "metadata": {"mode": "mix", "keywords": [], "...": "..."}
}
```

The contents of `data` and `metadata` depend on the selected mode and storage
backend. This endpoint always includes references.

## 5. Knowledge graph

All graph operations are scoped by `X-LightRAG-Workspace-ID`.

### Read graph labels and topology

| Endpoint | Parameters | Response |
| --- | --- | --- |
| `GET /graph/label/list` | none | JSON array of labels |
| `GET /graph/label/popular?limit=300` | `limit`: 1–1000 | JSON array |
| `GET /graph/label/search?q=Tesla&limit=50` | required `q`; `limit`: 1–100 | JSON array |
| `GET /graphs?label=Tesla&max_depth=3&max_nodes=1000` | optional label, depth and node limits | Graph object containing nodes/edges |
| `GET /graph/entity/exists?name=Tesla` | required `name` | `{"exists": true}` |

### Normalized graph projection

`GET /graphs/projection`

This endpoint converts the active graph backend's raw shape to one stable
contract for applications. It is isolated by the workspace header.

| Parameter | Default | Allowed/meaning |
| --- | ---: | --- |
| `view` | `pruned` | `full` or `pruned` |
| `max_nodes` | `300` | Maximum returned nodes (1–10,000) |
| `max_edges` | `800` | Maximum returned edges (1–50,000) |
| `strategy` | `importance` | `importance`, `community` or `degree` |
| `seed` | empty | Entity ID/label for a local projection |
| `max_depth` | `2` | BFS depth when `seed` is supplied |
| `include_media` | `true` | Attach asset references to nodes |
| `cursor` | empty | Opaque cursor returned for a large graph |

Example:

```bash
curl -sS "$LIGHTRAG_URL/graphs/projection?view=pruned&max_nodes=300&max_edges=800&strategy=importance" \
  -H "$AUTH_HEADER" \
  -H "X-LightRAG-Workspace-ID: $WORKSPACE_ID"
```

Response nodes use `entity_type`, `labels`, `source_document_ids`, `display`,
`metadata` and (when enabled) `media`. Edges use `source`, `target`,
`relation`, `keywords`, `weight` and `source_document_ids`:

```json
{
  "workspace_id": "7b2e7ef9-7a25-4f7d-9d4b-b3c0e15d4b47",
  "view": "pruned",
  "revision": "2026-07-19T16:40:10Z",
  "nodes": [{
    "id": "ClinicalModernBERT",
    "name": "ClinicalModernBERT",
    "entity_type": "artifact",
    "labels": ["ClinicalModernBERT"],
    "description": "ClinicalModernBERT is an encoder-only transformer...",
    "source_document_ids": ["doc-3266fbe1bed18306ea8ed3cb06c1a6f5"],
    "display": {"kind": "note", "title": "ClinicalModernBERT"},
    "media": [{"asset_id": "ast_01J...", "kind": "image", "mime_type": "image/png", "page": 4, "caption": "Model architecture"}],+
    "metadata": {"degree": 12}
  }],
  "edges": [{
    "id": "rel_758",
    "source": "ClinicalModernBERT",
    "target": "FlashAttention",
    "relation": "integrates",
    "description": "ClinicalModernBERT integrates FlashAttention...",
    "keywords": ["efficiency", "integration"],
    "weight": 1.0,
    "source_document_ids": ["doc-3266fbe1bed18306ea8ed3cb06c1a6f5"]
  }],
  "page": {"next_cursor": null, "truncated": false},
  "stats": {"total_nodes": 1260, "total_edges": 3481, "returned_nodes": 300, "returned_edges": 800}
}
```

`source_document_ids` are resolved from graph source/chunk IDs when the active
KV backend contains the corresponding chunks. `media[].asset_id` is opaque;
clients must use `/assets/{asset_id}` rather than parsing a sidecar or a
filesystem path.

### Create or edit entities

`POST /graph/entity/create`

```json
{
  "entity_name": "Tesla",
  "entity_data": {"description": "Electric vehicle company", "entity_type": "ORGANIZATION"}
}
```

`POST /graph/entity/edit`

```json
{
  "entity_name": "Tesla",
  "updated_data": {"description": "Updated description"},
  "allow_rename": false,
  "allow_merge": false
}
```

`POST /graph/entities/merge`

```json
{
  "entities_to_change": ["Ellon Musk", "Elon Msk"],
  "entity_to_change_into": "Elon Musk"
}
```

Create/edit/merge responses contain `status`, `message` and operation-specific
`data` fields. Clients should preserve unknown data fields for forward
compatibility.

### Create or edit relations

`POST /graph/relation/create`

```json
{
  "source_entity": "Elon Musk",
  "target_entity": "Tesla",
  "relation_data": {
    "description": "Founder and CEO relationship",
    "keywords": "CEO, founder",
    "weight": 1.0
  }
}
```

`POST /graph/relation/edit`

```json
{
  "source_id": "Elon Musk",
  "target_id": "Tesla",
  "updated_data": {"weight": 2.0}
}
```

### Delete entities or relations

```http
DELETE /graph/entity/delete
DELETE /graph/relation/delete
```

Entity deletion body: `{"entity_name":"Tesla"}`. Relation deletion body:
`{"source_entity":"Elon Musk","target_entity":"Tesla"}`. Responses use
`status` (`success`, `not_found`, `not_allowed` or `fail`), `message` and may
include `file_path`, `doc_id` and `status_code`.

## 6. Multimodal assets

### Download an asset

`GET /assets/{asset_id}`

Query parameters:

- `variant`: `original` (default) or `thumbnail`.
- `download`: `false` (default) serves inline; `true` sets an attachment
  disposition for download.

```bash
curl -L "$LIGHTRAG_URL/assets/ast_01JH3JNG4YQKQY2F5VJ86YQ4K1?variant=thumbnail" \
  -H "$AUTH_HEADER" \
  -H "X-LightRAG-Workspace-ID: $WORKSPACE_ID" \
  -o figure.png
```

Success is binary, not JSON. The response includes the detected `Content-Type`,
`Content-Length`, `Content-Disposition`, an `ETag` in the form
`"sha256-..."`, `Cache-Control: private, max-age=3600` and byte-range support.
The server validates that the opaque ID belongs to the workspace selected by
the header. Missing or cross-workspace assets always return `404`.

If a parser has not produced a physical thumbnail, the service generates a
bounded image thumbnail when the Pillow image extra is available; otherwise it
safely falls back to the original bytes. No filesystem path is exposed.

## 7. Ollama-compatible endpoints

These routes make LightRAG usable by clients that speak the Ollama API. They
also honor `X-LightRAG-Workspace-ID`.

- `GET /api/version` → `{"version":"0.9.3"}` (version may change).
- `GET /api/tags` → Ollama model list (`models` array).
- `GET /api/ps` → currently loaded model/process list (`models` array).

### Generate

`POST /api/generate`

```json
{
  "model": "lightrag:latest",
  "prompt": "Summarize the annual report",
  "system": null,
  "stream": false,
  "options": {}
}
```

With `stream:false`, the response is an Ollama-style object containing
`model`, `created_at`, `response` and `done`. With `stream:true`, the response
is NDJSON with a final `done:true` line. This endpoint performs direct LLM
generation and does not automatically run LightRAG retrieval.

### Chat

`POST /api/chat`

```json
{
  "model": "lightrag:latest",
  "messages": [{"role": "user", "content": "What was revenue in 2025?"}],
  "stream": true,
  "options": {}
}
```

`messages` must be non-empty and the last message must have role `user`.
Chat supports the mode prefixes `/local`, `/global`, `/hybrid`, `/naive`,
`/mix`, `/bypass` and `/context`; for example, `/hybrid[Answer briefly] What
was revenue in 2025?`. Responses follow the Ollama chat shape, with NDJSON for
streaming and a single JSON object for `stream:false`.

## 8. Service and UI endpoints

- `GET /health` is a liveness endpoint. Unauthenticated calls return basic
  status/version/pipeline signals; authenticated calls include operational
  configuration and metrics.
- `GET /` redirects to the WebUI (normally `/webui/`) or Swagger when the UI
  bundle is unavailable.
- `GET /auth-status` reports whether authentication is enabled and returns
  version/UI metadata. In guest mode it also returns a short-lived guest token.
- `GET /docs`, `GET /redoc`, `GET /openapi.json` expose interactive and machine
  readable API documentation.

## 9. Typical integration flow

```text
1. POST /v1/workspaces                         -> workspace UUID
2. Upload files or POST /documents/text        -> track_id
3. GET /documents/track_status/{track_id}      -> wait for PROCESSED
4. POST /documents/paginated                    -> inspect status/errors
5. POST /query or /query/stream                 -> answer with references
6. GET /graphs and /graph/label/*               -> visualize/search the KG
```

Every request in steps 2–6 should include the same
`X-LightRAG-Workspace-ID`. To remove all associated data, call
`DELETE /v1/workspaces/{workspace_id}` and wait for the deletion job to finish
before reusing that identifier.

## 10. Error handling

Common statuses are:

| HTTP | Meaning |
| --- | --- |
| `400` | Invalid file type, malformed request or unsupported operation |
| `401` | Missing/invalid/expired JWT or API key |
| `403` | Credential is valid but operation is not permitted |
| `404` | Workspace, document, graph item or route not found |
| `409` | Duplicate document, workspace/pipeline busy, or protected default workspace |
| `413` | Uploaded file exceeds the configured size limit |
| `422` | Pydantic validation error; response includes a `detail` array with field locations |
| `500` | Unexpected server or dependency failure; inspect server logs and `error_msg` |

Validation errors follow FastAPI's shape, for example:

```json
{
  "detail": [
    {"loc": ["body", "query"], "msg": "String should have at least 3 characters", "type": "string_too_short"}
  ]
}
```

For production integrations, treat ingestion and workspace deletion as
asynchronous jobs, retry only idempotent reads, and log the returned
`track_id`/`job_id` for support diagnostics.
