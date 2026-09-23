# Server instance contract: several servers on one host

Status: **the supported deployment shape**, stated in one place. Before this
document, the rule existed only as scattered remarks, several of which said
"separate process trees are unsupported" without saying *on what*.

Read this before changing anything that one server could share with another
server on the same host. That covers files under `WORKING_DIR` or
`INPUT_DIR`, anything keyed without the workspace, and any lock taken at
startup.

## The rule

On one host, several LightRAG Server instances **may run at the same time
and share one `WORKING_DIR`** (and one `INPUT_DIR`) when all five hold:

1. **Each instance serves its own business workspace, down to the physical
   names every backend derives from it.** `WORKSPACE`, after the server's
   sanitization (every character outside `[A-Za-z0-9_]` becomes `_`, so
   `a-b` and `a_b` are ONE workspace), differs between the instances. The
   default workspace (empty) counts as one workspace like any other. Several
   backends map the workspace further, and lossily, onto a label, a
   collection, an index, a graph or a column value; two workspaces that land
   on one physical partition ARE one workspace for this rule. *Workspace
   names every backend keeps apart* below lists them and gives a naming rule
   that avoids all of them.
2. **No `*_WORKSPACE` override collapses them.** A backend override such as
   `POSTGRES_WORKSPACE` or `MILVUS_WORKSPACE` maps different logical
   workspaces onto one physical container. That is legacy compatibility
   (*`*_WORKSPACE` is legacy compatibility* in `ConfigurationStorageContract.md`), and
   two instances collapsed that way ARE the same workspace for this rule.
3. **Every instance uses the same configuration container, and it is not the
   local JSON file.** `LIGHTRAG_CONFIG_STORAGE` (or the `LIGHTRAG_KV_STORAGE`
   it follows when unset) must be `PGKVStorage`, `MongoKVStorage` or
   `OpenSearchKVStorage`, and every instance must point it at the same
   database, cluster or server. The anchor at
   `WORKING_DIR/_lightrag_config/storage_anchor.json` is one per `WORKING_DIR`
   and records one backend and one container identity, so an instance on a
   different backend, or on another container of the same backend, is
   refused at startup (*Rules at a glance* in
   `ConfigurationStorageContract.md`).
4. **Each instance listens on its own port.**
5. **The first start is serialized.** A start that finds no anchor takes the
   bind lock (`.lightrag_anchor_bind.lock`) so that only one instance creates
   the container identity. Where that lock cannot be taken (Windows, whose
   `msvcrt` has no shared lock mode, or a filesystem without locks) it fails
   open with a warning. Two instances starting at once there can then write
   different identities: the identity row is an upsert, and only the anchor
   publish is no-clobber, so the anchor can end up naming a UUID the
   container no longer holds and every later start is refused. On such a
   host, start one instance alone until the anchor exists, then start the
   others.

An "instance" is one process tree. That is either one uvicorn process
(`lightrag-server`) or one Gunicorn master with its forked workers
(`lightrag-gunicorn`). Workers of one master are one instance, not several.

Everything else is **unsupported**:

- two instances on the same workspace, whatever their directories;
- instances on different hosts that share one `WORKING_DIR` over a network
  filesystem.

## Workspace names every backend keeps apart

Each row is a pair of different workspaces that one backend stores in one
partition. Only OpenSearch notices; everywhere else the two instances read,
write and delete each other's data without an error.

| backend | workspaces that share a partition | detected |
| --- | --- | --- |
| `Neo4JStorage`, `MemgraphStorage` | the empty workspace and `base` (both become the graph label `base`). `Neo4JStorage` also derives its full-text index name by stripping leading and trailing `_` and prefixing `ws_` to a name that starts with a digit, so `foo` / `_foo_`, `_` / `base` and `1abc` / `ws_1abc` share an index, and label search in the second of each pair silently finds nothing | no |
| `QdrantVectorDBStorage` | the empty workspace and `_` (the payload value it stores for the empty one) | no |
| `PGKVStorage`, `PGVectorStorage`, `PGDocStatusStorage`, `PGTableGraphStorage` | the empty workspace and `default` (the column value stored for the empty one) | no |
| `PGGraphStorage` | the empty workspace and `default` **in any letter case** (so `Default` shares only the graph with the empty workspace, and its other PostgreSQL rows stay apart); two workspaces whose `<workspace>_chunk_entity_relation` agree in the first 63 bytes, which PostgreSQL keeps of a graph name (any two longer than 41 bytes that share their first 41); and a mixed-case workspace with its lowercase twin, on the raw SQL paths that name the graph unquoted | no |
| `MongoKVStorage` / `MongoVectorDBStorage` and the other Mongo storages | names are `<workspace>_<namespace>`, so `W` with namespace `text_chunks` and `W_text` with namespace `chunks` are one collection; likewise `entity_chunks` / `_entity`, `relation_chunks` / `_relation` and `full_entities` / `_full`. With an empty workspace this means workspace `text` (or `entity`, `relation`, `full`) collides with the default workspace | no |
| OpenSearch storages | case (`TeamA` / `teama`); a leading `_`, which becomes `x_` (`_foo` / `x_foo`); and the same four joins as MongoDB | **refused** at startup by the index's workspace ownership marker (`WorkspaceIndexCollisionError`), except on an index created before the marker existed that could not be marked |
| file-backed storages, `RedisKVStorage`, `RedisDocStatusStorage`, `MilvusVectorDBStorage` | none in code. On a case-insensitive filesystem (default macOS and Windows) the file-backed storages share `TeamA/` and `teama/` | – |

**A naming rule that avoids every row:** lowercase letters and digits only,
starting with a letter, at most 41 characters, and not one of `base`,
`default`, `text`, `entity`, `relation` or `full`. On Windows, also avoid the
reserved device names (`con`, `prn`, `aux`, `nul`, `com1`–`com9`,
`lpt1`–`lpt9`), which cannot be directory names there. The empty (default)
workspace is fine alongside any name that follows it.

## Why the rule is sound

Every piece of per-workspace state is disjoint by construction, so two
instances on different workspaces never write the same thing:

| state | where it lives | why two workspaces cannot collide |
| --- | --- | --- |
| file-backed business storages (`JsonKVStorage`, `JsonDocStatusStorage`, `NetworkXStorage`, `NanoVectorDBStorage`, `FaissVectorDBStorage`) | `WORKING_DIR/<workspace>/`, or `WORKING_DIR/` for the default workspace | different files; the default workspace's files sit at the top level, the others in their own subdirectories |
| server-backed business storages | a workspace column, collection/index prefix or payload partition | partitioned by workspace in every backend |
| uploads and parsed artifacts | `INPUT_DIR/<workspace>/`, or `INPUT_DIR/` for the default workspace | the scan and the clear read only files at their own directory's top level (`iter_new_files`, `/documents/clear`), so the default workspace never reaches into a named workspace's subdirectory. The parser's source lookup stays in the workspace's own subdirectory too; see *The parser's source lookup* below |
| embedding baselines | the configuration container, key `<workspace>/embedding/<target>` | the key's scope is the workspace (*What discriminates two deployments on one server* in `ConfigurationStorageContract.md`) |
| `pipeline_status`, keyed locks, the ingress mailbox | `shared_storage`, per process tree | each instance coordinates only its own workspace; no cross-instance coordination is needed because no two instances touch one workspace |

### The parser's source lookup

When a document is parsed, `_resolve_source_file_for_parser`
(`lightrag/pipeline.py`) finds its source file from the stored basename.
Once `INPUT_DIR/<workspace>/` exists, which the API server's
`DocumentManager` creates at start, a named workspace looks only in its own
directories (`INPUT_DIR/<workspace>/` and `./inputs/<workspace>/`, each with
its `__parsed__/`). It never falls back to `INPUT_DIR/` or `./inputs/`,
which are the default workspace's, so a workspace whose own source file is
gone gets it reported missing rather than parsing another workspace's file
of the same name.

Without that directory, the base directories are still searched. That is
the SDK case: a caller with a workspace keeps its files directly in
`INPUT_DIR`. It is never a multi-instance deployment, because every server
creates its workspace's directory.

Each server holds one `LightRAG` instance on one workspace. The
`LIGHTRAG-WORKSPACE` request header selects which workspace `/health`
reports on; it never lets one instance write another workspace's data.

## What the instances DO share, and how each is guarded

A few things are not per-workspace. Each one either has its own guard or is
the reason for one of the rule's conditions:

| shared thing | guard |
| --- | --- |
| the JSON configuration file `WORKING_DIR/_lightrag_config/kv_server_config.json` | **condition 3.** `JsonKVStorage` publishes the whole namespace by rewriting the file from a per-process-tree copy, so two instances would overwrite each other's baselines. That is why a file-backed configuration storage claims `config_dir` exclusively (`lightrag/kg/working_dir_lock.py`) and a second instance is **refused** with `WorkingDirectoryInUseError` |
| a server-backed configuration container (the `LIGHTRAG_CONFIG` table, the `_lightrag_config_config` collection or index) | disjoint row keys per workspace; nothing else is shared today |
| the host's ports | condition 4; a clash fails at bind time |

A new server-wide or container-wide record breaks the rule unless it gets a
guard. Examples are a row not scoped by workspace, or a file directly under
`WORKING_DIR` that every instance writes. The in-tree keyed locks cannot see
another instance, so "it runs under the keyed lock" is not a guard here.
Such a change must state which of these it uses:

- write it **once**, under a lock the instances share (an OS file lock in
  `WORKING_DIR`, which every instance on this host can see), and re-read it
  inside that lock;
- or scope it by workspace;
- or make every writer write the **same** value, so the race is harmless in
  direction.

## What is enforced and what is not

| situation | outcome |
| --- | --- |
| two instances, JSON configuration storage, one `config_dir` | **refused**: the second fails with `WorkingDirectoryInUseError` |
| two instances, different workspaces, one server-backed configuration container | runs; supported |
| two instances on one `WORKING_DIR`, different configuration backends or containers | **refused**: the second instance fails the anchor check at startup |
| two instances, **same** workspace | **not detected**. They race on every write of that workspace: pipeline state, file-backed storages (whole-file rewrites) and baseline claims. Unsupported, not an accepted residue |
| two instances collapsed onto one container by a `*_WORKSPACE` override | not detected; same as the row above. `warn_about_workspace_overrides()` announces every override at server start |
| two workspaces a backend maps to one physical partition (condition 1) | **refused** on OpenSearch by the index ownership marker; **not detected** on every other backend, where it is the same-workspace row above |
| two first starts at once where the bind lock fails open (condition 5) | **not detected** at bind time; the next start is refused when the anchor and the container identity disagree |
| instances on different hosts sharing `WORKING_DIR` over NFS / SMB | not supported. File locks there may fail open (a warning, then proceed), and nothing else crosses hosts |

The undetected rows are left undetected on purpose. Any detection would have
to be a claim keyed by workspace that every instance takes. Taken on the
workspace directory it would miss server-backed data, which has no
directory; taken in a server backend it would need a lease that outlives a
crash. Neither is part of this contract. An operator runs one instance per
workspace.

## Maintenance tools

`lightrag-rebuild-vdb` and `lightrag-clear-storage` act on **one**
workspace. Stop the instance serving that workspace first; instances serving
other workspaces may keep running. Both tools still take the `config_dir`
claim when the configuration storage is JSON, and so refuse while any
instance runs on it, which is consistent with condition 3.

## Where the other contracts defer to this one

- `ConfigurationStorageContract.md`: *One server at a time on a file-backed
  configuration*, *What the lock does and does not span*, and *What the
  category does not retire*.
- `FileBackedSnapshotContract.md`: *One file per namespace per process tree*.
- `lightrag/kg/working_dir_lock.py`: the business-data residue.
