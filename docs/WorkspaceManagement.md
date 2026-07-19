# Workspace management API

LightRAG's API supports multiple isolated knowledge bases in one server. The
existing document, query, graph, retrieval, and Ollama-compatible URLs remain
unchanged. Select the knowledge base with the optional request header:

```http
X-LightRAG-Workspace-ID: 3f03e039-7f15-4984-bec2-2e1fd4c11af2
```

When the header is omitted, the server uses the default workspace. Swagger UI
shows this header on every workspace-scoped endpoint and lets clients supply it
from the normal **Try it out** form.

## Workspace lifecycle

| Operation | Endpoint |
| --- | --- |
| Create | `POST /v1/workspaces` |
| List | `GET /v1/workspaces` |
| Read | `GET /v1/workspaces/{workspace_id}` |
| Update metadata | `PATCH /v1/workspaces/{workspace_id}` |
| Delete all workspace data | `DELETE /v1/workspaces/{workspace_id}` |

Example creation request:

```json
{
  "display_name": "Finance knowledge base",
  "description": "FY2025 reports and policies"
}
```

The returned `id` is the UUID supplied in `X-LightRAG-Workspace-ID`. It is a
public API identifier; it is not a storage namespace.

## Storage model

Workspace control-plane metadata is persisted in the same durable database
family configured for the deployment. With the current PostgreSQL/pgvector
configuration, these tables are created in `POSTGRES_DATABASE`:

```text
lightrag_workspaces
lightrag_workspace_jobs
```

With `Mongo*` storage configuration, the registry uses MongoDB collections with
the same names. `LIGHTRAG_WORKSPACE_REGISTRY_BACKEND=auto` is the default and
selects PostgreSQL for `PG*` storage or MongoDB for `Mongo*` storage; it may be
set explicitly to `postgres` or `mongodb` when a deployment has both families.
There is no SQLite or file-based registry fallback. A configuration without a
durable PostgreSQL or MongoDB registry fails fast rather than silently splitting
workspace metadata into a local file.

Each registry record owns an immutable internal `storage_key`, such as
`ws_8cf...`. LightRAG receives that key as its normal `workspace` argument:

| Data plane | Isolation mechanism |
| --- | --- |
| PostgreSQL / pgvector | Existing `workspace` column on LightRAG tables and vector records |
| Neo4j | Existing workspace-specific node label and indexes |
| MongoDB, Redis, Milvus, Qdrant, OpenSearch | Existing backend workspace partition/filter support |
| JSON, NanoVectorDB, NetworkX, Faiss | Existing workspace-prefixed files/namespaces |
| Uploaded source and parser artifacts | `INPUT_DIR/<storage_key>/` |

Do not set a backend-specific override such as `POSTGRES_WORKSPACE`,
`NEO4J_WORKSPACE`, `MONGODB_WORKSPACE`, `MILVUS_WORKSPACE`,
`QDRANT_WORKSPACE`, `REDIS_WORKSPACE`, `OPENSEARCH_WORKSPACE`, or
`MEMGRAPH_WORKSPACE`. LightRAG refuses startup in multi-workspace mode if one
is set, because it would collapse all workspaces into one backend namespace.

The server preserves existing data without an automatic migration: the
registry's default workspace uses the pre-existing `WORKSPACE` value (normally
empty, which maps to the legacy/default namespaces of the selected backends).
New workspaces always receive isolated storage keys.

## Deletion behavior

`DELETE /v1/workspaces/{workspace_id}` returns `202 Accepted`. The server sets
the workspace to `deleting`, blocks new data-plane requests for it, then removes
all LightRAG storage rows/records, graph data, response cache, uploaded files,
and parser artifacts. The operation is tracked in the registry job table and
can safely be retried after an operational failure.

The legacy default workspace is protected from deletion. Migrate its data to a
new workspace first, then choose a new default in a dedicated migration step.

## Security

Workspace isolation is not authorization. With one global `LIGHTRAG_API_KEY`,
any holder of that key can select any workspace UUID. If different users or
tenants must be restricted from each other, add authenticated principals and a
`workspace_members` authorization table before exposing the API outside the
trusted application boundary.
