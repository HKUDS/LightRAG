# Offline Configuration Migration Tool

`lightrag-migrate-config` moves a deployment's **configuration container**
(the embedding baselines and every other row of the `config` namespace) from
one backend **type** to another — for example from `PGKVStorage` to
`MongoKVStorage` — without a running server. It moves the configuration
storage anchor only after the copy has been verified.

The design contract is *The anchor and the container identity* in
[`docs/design/ConfigurationStorageContract.md`](../../docs/design/ConfigurationStorageContract.md).

## When do I need this?

Every deployment is anchored to one configuration container:
`<WORKING_DIR>/_lightrag_config/storage_anchor.json` records the backend type
and the container's UUID, and every start checks both before reading any
baseline. When `LIGHTRAG_CONFIG_STORAGE` is unset, the configuration backend
follows `LIGHTRAG_KV_STORAGE`. Changing the KV backend therefore moves the
configuration candidate, and the start is refused:

```
Refusing to start: the configuration storage selected now is MongoKVStorage
(_lightrag_config), but the anchor ... binds this deployment to PGKVStorage ...
```

You have two ways out:

1. **Keep the configuration where it is** (no migration). Set
   `LIGHTRAG_CONFIG_STORAGE=PGKVStorage`, the anchored backend, explicitly.
   Business KV data can live on Mongo while the configuration stays on
   PostgreSQL.
2. **Move the configuration** to the new backend type with this tool.

This tool only moves a container **across types**. For a same-type move, such
as PostgreSQL to another PostgreSQL, Mongo to Mongo, an OpenSearch
snapshot/reindex, or copying the JSON file, use the backend's own dump and
restore. The identity row travels with the data, and a container of the same
type with the same UUID passes the start-up check.

## Usage

```bash
# Stop every LightRAG server, SDK process and maintenance tool first!

# See what would happen. This reads everything and writes nothing.
lightrag-migrate-config --target-backend MongoKVStorage --dry-run

# Migrate.
lightrag-migrate-config --target-backend MongoKVStorage

# Source and target connections in separate env files:
lightrag-migrate-config --target-backend OpenSearchKVStorage \
    --source-env ./old.env --target-env ./new.env
```

| option | meaning |
| --- | --- |
| `--target-backend` | `JsonKVStorage`, `PGKVStorage`, `MongoKVStorage` or `OpenSearchKVStorage`; must differ from the anchored backend |
| `--source-env` | env file with the **source** connection (default: the current environment) |
| `--target-env` | env file with the **target** connection (default: the current environment) |
| `--dry-run` | report the anchor, the source (row count, workspace scopes), the target and the verdict on it; write no row (opening the target still provisions a missing table, collection or index, as any start does) |
| `--assume-exclusive` | proceed where the anchor lock cannot be taken (see *Locking*) |
| `--yes` | skip the confirmation prompt |

**Connections.** The two backends are of different types, so they read
different variables: `POSTGRES_*`, `MONGO_*` / `MONGODB_*`, `OPENSEARCH_*`, and
`LIGHTRAG_CONFIG_DIR` for JSON. The tool refuses when the two env files set
different values for a variable that either selected backend reads. It never
writes an env file and never logs a credential.

**Working directory.** The anchor and its lock are resolved from the
**current** environment's `WORKING_DIR`. The tool refuses when either env file
names a different `WORKING_DIR`. A JSON source takes its `config_dir` from
`--source-env`, and a JSON target takes its `config_dir` from `--target-env`.

## What it does

1. Takes the anchor lock **exclusively** and strict-reads the anchor. With no
   anchor, nothing is bound, and the tool refuses: start the server once to
   bind it. The target type must differ from the anchored one.
2. Opens the source as the anchored type. Its identity must equal the
   anchor's UUID. An unreadable source refuses: restore or reconnect it and
   re-run. A target refuses any source row carrying a field it reserves
   for itself: `id` on `PGKVStorage` (it returns the key as `id`) and
   `__mirrored_id` on `OpenSearchKVStorage` (it writes the key there).
3. Opens the target and classifies it:
   - **empty**: the tool claims it;
   - **already holds this identity**: the target is left over from an earlier
     attempt of this migration, and the tool resumes;
   - **anything else** (another identity, or rows without one): the tool
     refuses. It never overwrites or merges into such a target.
4. Writes the target's identity row, with the **same UUID**, first, then
   flushes strictly and reads it back. This row is the ownership marker that
   makes a re-run safe.
5. Copies every row: server-scope rows and every workspace's rows, with the
   key and the row envelope verbatim. A row that is not a well-formed row (an
   integer `schema_version`, a string `workspace` and a mapping `value`) is
   refused and listed, never skipped; `updated_at` / `updated_by` are only
   diagnostic and are copied as they are. Target rows that the source does not
   hold, or holds differently, are deleted. The tool is allowed to delete them
   only because step 3 proved ownership.
6. Flushes strictly. It then compares every target row with the source (key
   set and content, ignoring backend-owned `_id` / `create_time` /
   `update_time`).
7. Replaces the anchor atomically with `{target type, same UUID}`. **This is
   the commit point.**

After step 7, set `LIGHTRAG_CONFIG_STORAGE=<target>` explicitly, carry over
every target connection setting that `--target-env` supplied (the tool lists
them by name), and start the server. The tool does not edit the environment:
without those settings the server opens a different target container and is
refused.

## Failure and recovery

- **Before step 7** the anchor is unchanged. The old environment keeps
  working on the source, and the new one is refused on the type mismatch.
  Re-run the tool: it resumes through the ownership marker and reconciles the
  target to the source as it is **now**, even if the source changed in the
  meantime. Anything that goes wrong after the target was claimed is reported
  as a failure, never as a refusal, even a source row damaged mid-copy (the
  re-run then refuses and names it).
- **After step 7** the migration is complete, and the tool reports
  `Switched`.
- **Outcome unknown.** When the anchor replace raises and the anchor then
  cannot be read back, the replace may or may not have landed, and the tool
  says so instead of guessing. Start nothing until the anchor reads back: if
  it names the target, the migration is complete; if it still names the
  source, re-run.
- **No source row is ever written or deleted.** Removing the old copy is a
  separate, explicit action. Opening either side is an ordinary backend
  `initialize()`, which may provision a missing table or index or bring an
  existing container's schema to this version, as any start does.
- **A shared source container.** Every workspace's rows are copied, and the
  dry run lists the scopes. Deployments that stay on the source are
  unaffected, but their rows in the target are stale copies. The target must
  be dedicated to this deployment. Consolidating into an already-shared
  container is not supported.

## Locking

Every starter (server, Gunicorn master, SDK, `lightrag-rebuild-vdb`,
`lightrag-clear-storage`) takes the anchor lock **shared**. This tool takes it
**exclusively**, so it is refused while any of them runs on this
`WORKING_DIR`, and they are refused while it runs.

Some filesystems and platforms cannot lock: NFSv3 without lockd, SMB/CIFS, a
read-only directory, or Windows, where `msvcrt` has no shared lock. There,
starters warn and proceed, and this tool refuses unless you pass
`--assume-exclusive`. That flag records that **you** stopped every reader and
writer.

The lock is local to one `WORKING_DIR`. Deployments that use other working
directories but share the same remote configuration container must all be
stopped and coordinated by hand. There is no distributed lock and no rolling
migration.
