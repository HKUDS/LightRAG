# PostgreSQL with separately provisioned schemas

`POSTGRES_AUTO_MIGRATE=true` is the default. It retains automatic extension,
table, index and legacy-data initialization. Set `POSTGRES_AUTO_MIGRATE=false`
when administrators manage the schema separately and application workers use
restricted, non-owner credentials.

Supported backends are `PGKVStorage`, `PGDocStatusStorage`, `PGVectorStorage`
and `PGTableGraphStorage`. AGE-based `PGGraphStorage` explicitly refuses this
mode. Other storage backends keep their own initialization policies.

The equivalent `config.ini` setting is:

```ini
[postgres]
auto_migrate = false
```

The environment overrides `config.ini`. Values are case insensitive and trimmed:
`true/false`, `yes/no`, `on/off`, `t/f`, `1/0`. A blank value means false.
An unrecognized explicit value raises a configuration error. All PostgreSQL
storages in one process must use the same migration policy and vector settings;
a conflicting request cannot reuse the shared pool. Reconnecting preserves the
policy, registers the pgvector codecs and restores existing session settings.

## Administrator provisioning

1. Stop application workers before performing a schema upgrade. Back up the
   database and retain the deployment's configuration anchor and storage data.
2. Use the **same LightRAG version, database, search path, vector index type,
   embedding model name and dimension** as the runtime. Use administrative
   credentials and `POSTGRES_AUTO_MIGRATE=true`. Initialize and finalize a
   LightRAG instance with the intended storage backends using the normal API:

   ```python
   await rag.initialize_storages()
   await rag.finalize_storages()
   ```

   This uses the existing provisioning and migration code, including graph
   normalization and model-specific vector table creation. It does not require
   a new migration tool. Repeat for workspaces with pending legacy migrations,
   and for each embedding model/dimension the deployment will use. Inspect any
   migration errors: automatic mode retains its existing best-effort behavior
   for some migrations and performance indexes.
3. Grant runtime access to the resolved application tables and extension types.
   Use a login role that does not own the tables, with database `CONNECT`,
   schema `USAGE`, and `SELECT`, `INSERT`, `UPDATE`, `DELETE` on the application
   tables. No schema `CREATE` or table ownership is needed. Include
   `LIGHTRAG_CONFIG` and all configured model-specific vector tables.
   Retained legacy vector tables and visible sibling model tables also need
   `SELECT` on `workspace` and table `DELETE`: normal workspace cleanup removes
   this workspace's rows from them, even when migration has already finished.
   Startup checks those cleanup grants using the same sibling discovery as
   deletion. No `INSERT` or `UPDATE` grant is needed on these retained tables.
   `LIGHTRAG_VDB_MIGRATION_STATE`, when consulted, needs only `SELECT` on
   `target_table` and `workspace`; No-DDL mode does not write migration markers.
4. Start workers with runtime credentials and `POSTGRES_AUTO_MIGRATE=false`.
   Keep `WORKING_DIR` persistent: configuration identity, anchor and embedding
   baseline checks still run. No-DDL mode allows normal data writes, including
   configuration claims; it is not a read-only database mode.

When using the server, select backends through the usual `LIGHTRAG_*_STORAGE`
settings. For direct library use, select them on the `LightRAG` instance.
Credentials belong in the deployment's secret configuration, not in source.

## Startup compatibility checks

Startup uses read-only PostgreSQL catalogs and queries. It never attempts
extension, table or index creation, alteration, deletion, automatic legacy
vector copies, or graph edge normalization. Validation failures propagate
before document processing; there is no fallback to automatic migration.

The shared client validates these common tables, as automatic initialization
already provisions them together:

- `LIGHTRAG_DOC_FULL`, `LIGHTRAG_DOC_CHUNKS`, `LIGHTRAG_LLM_CACHE`, `LIGHTRAG_CONFIG`
- `LIGHTRAG_DOC_STATUS`, `LIGHTRAG_FULL_ENTITIES`, `LIGHTRAG_FULL_RELATIONS`
- `LIGHTRAG_ENTITY_CHUNKS`, `LIGHTRAG_RELATION_CHUNKS`

Each name is resolved through the runtime `search_path` using `to_regclass`.
Column, key, index and privilege checks use that relation's OID. An unrelated
same-named object elsewhere cannot satisfy the checks. Ordinary and partitioned
tables are accepted; views and foreign tables are not. Required column types
must be compatible with the current backend SQL. Wider text columns are
accepted, and additional nullable/defaulted/generated columns are allowed.
Columns that receive explicit SQL NULL (such as `parse_engine`) must allow it;
a default does not replace an explicitly supplied NULL. Columns written by
runtime SQL cannot be generated columns or `GENERATED ALWAYS` identity columns.
An omitted NOT NULL column, including `meta`/timestamps omitted by KV inserts,
needs a default or generation. Nullable omitted columns need no default, and
non-nullable columns with guaranteed runtime values are accepted. Defaults do
not have to match the provisioning DDL exactly. Primary keys
must be usable and nondeferrable, with exactly the required key columns.
Constraint and index names need not match the automatically generated names.

Index policy:

- The common `(workspace, id)` primary keys supply the required uniqueness
  indexes for upserts and workspace isolation. Redundant standalone `id` and
  `(workspace, id)` indexes are not required.
- Vector tables require a usable, nonpartial index on `content_vector` using
  the configured HNSW, HNSW_HALFVEC, IVFFLAT or VCHORDRQ method and cosine
  operator class. Index build parameters need not match exactly. The declared
  column type and dimension are checked even when the table is empty.
- Graph tables require their composite primary keys and both validated
  composite foreign keys to the **resolved** node table with `ON DELETE CASCADE`.
  A usable btree index beginning with `(workspace, namespace, tgt_id)` is also
  required for reverse endpoint access. The primary keys cover source access.
- Other indexes, including document-status pagination/scheduling and the
  graph's noncanonical-edge partial index, are performance optimizations.
  Their absence does not prevent correct operations and does not refuse startup.
  A missing or incompatible scheduling index emits a warning, as in automatic mode.
  Provision them through the normal administrator path to preserve query plans.

Usable required indexes must be valid, ready and live. Partial or expression
indexes cannot substitute for the required full column indexes. No-DDL mode
does not require pgvector when the selected combination does not use
`PGVectorStorage`. When it does, pgvector must already be installed; halfvec
still requires pgvector >= 0.7.0, dimension limits still apply, and iterative
scan capability detection still runs. VCHORDRQ additionally requires the
VectorChord (`vchord`) extension and retains its existing session configuration.

## Missing objects and migrations

Errors name the affected object and the expected definition. Have an
administrator provision or repair it with the current LightRAG version, then
restart the workers. Do not grant the runtime DDL privileges to work around a
validation failure while leaving this mode enabled.

A populated legacy vector source with an empty target, or with an active
`LIGHTRAG_VDB_MIGRATION_STATE` record for this target/workspace, requires an
administrator to complete migration. Retained legacy tables alone do not refuse
startup. A nonempty target without a marker retains the existing migration
contract: it may reflect intentional deletions, so rows are not copied back.
An inert marker with no remaining legacy source rows is accepted. See
[legacy migration residues](design/PostgresLegacyVectorMigration.md). Older
unmarked partial migrations remain indistinguishable from completed migrations
followed by deletions; audit/rebuild them administratively when needed.

Noncanonical graph edges are checked only in the effective workspace and
namespace, using the same C-collation condition as automatic normalization.
They cause an actionable refusal rather than a runtime data migration.
Legacy LLM cache keys likewise require administrative migration.

Vector-space baseline, coverage, configuration identity and anchor checks are
unchanged. Physical vector type/dimension mismatches raise a schema validation
error, not the recoverable `VectorSpaceMismatchError`: the rebuild tool must
not offer destructive drop-and-retry recovery for a column definition that
deleting rows cannot repair. Repairing a column's dimension or switching its
vector/halfvec type requires administrator schema work. Legitimate embedding
provenance refusals elsewhere retain their existing recovery behavior.

To return to automatic mode, stop workers, set `POSTGRES_AUTO_MIGRATE=true`
(or remove the setting when no `config.ini` override remains), and restart with
credentials authorized to perform the existing initialization/migrations.

## Boundaries

Validation is a startup check, not a lock against later administrator DDL.
Keep the schema and search path stable while workers run. Table privileges do
not establish arbitrary row-level security policy compatibility; this mode
makes no universal RLS guarantee. The runtime still needs DML permissions for
all normal data and configuration operations. Existing multi-store consistency,
workspace precedence (including `POSTGRES_WORKSPACE`) and configuration
partition rules are unchanged.

Opt-in tests in `tests/kg/postgres_impl/test_postgres_no_ddl_integration.py`
create disposable databases and non-owner login roles using `POSTGRES_*`
administrator credentials with `CREATEDB` and `CREATEROLE`. They trace runtime
SQL independently of privileges and remove their own databases/roles.
The existing plain-PostgreSQL CI job selects `-m pg_smoke --run-integration`:
it runs the No-DDL common/graph startup case, column-compatibility cases and the
existing graph/anchor smoke tests without pgvector. Vector-dependent cases are
not selected by that job. Run the complete new file with `--run-integration`
against a pgvector-enabled server to exercise physical vector schemas,
model-table consistency, legacy/sibling cleanup grants, codec registration and
similarity search. VectorChord still needs its own extension-enabled environment.
