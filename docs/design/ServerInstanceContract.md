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
and share one `WORKING_DIR`** (and one `INPUT_DIR`) when all four hold:

1. **Each instance serves its own business workspace.** `WORKSPACE`, after
   the server's sanitization (every character outside `[A-Za-z0-9_]` becomes
   `_`, so `a-b` and `a_b` are ONE workspace), differs between the instances.
   The default workspace (empty) counts as one workspace like any other.
2. **No `*_WORKSPACE` override collapses them.** A backend override such as
   `POSTGRES_WORKSPACE` or `MILVUS_WORKSPACE` maps different logical
   workspaces onto one physical container. That is legacy compatibility
   (*`*_WORKSPACE` is legacy compatibility* in `ConfigurationStorage.md`), and
   two instances collapsed that way ARE the same workspace for this rule.
3. **The configuration storage is not the local JSON file.**
   `LIGHTRAG_CONFIG_STORAGE` (or the `LIGHTRAG_KV_STORAGE` it follows when
   unset) must be `PGKVStorage`, `MongoKVStorage` or `OpenSearchKVStorage`.
4. **Each instance listens on its own port.**

An "instance" is one process tree. That is either one uvicorn process
(`lightrag-server`) or one Gunicorn master with its forked workers
(`lightrag-gunicorn`). Workers of one master are one instance, not several.

Everything else is **unsupported**:

- two instances on the same workspace, whatever their directories;
- instances on different hosts that share one `WORKING_DIR` over a network
  filesystem.

## Why the rule is sound

Every piece of per-workspace state is disjoint by construction, so two
instances on different workspaces never write the same thing:

| state | where it lives | why two workspaces cannot collide |
| --- | --- | --- |
| file-backed business storages (`JsonKVStorage`, `JsonDocStatusStorage`, `NetworkXStorage`, `NanoVectorDBStorage`, `FaissVectorDBStorage`) | `WORKING_DIR/<workspace>/`, or `WORKING_DIR/` for the default workspace | different files; the default workspace's files sit at the top level, the others in their own subdirectories |
| server-backed business storages | a workspace column, collection/index prefix or payload partition | partitioned by workspace in every backend |
| uploads and parsed artifacts | `INPUT_DIR/<workspace>/`, or `INPUT_DIR/` for the default workspace | the scan and the clear read only files at their own directory's top level (`iter_new_files`, `/documents/clear`), so the default workspace never reaches into a named workspace's subdirectory |
| embedding baselines | the configuration container, key `<workspace>/embedding/<target>` | the key's scope is the workspace (*What discriminates two deployments on one server* in `ConfigurationStorage.md`) |
| `pipeline_status`, keyed locks, the ingress mailbox | `shared_storage`, per process tree | each instance coordinates only its own workspace; no cross-instance coordination is needed because no two instances touch one workspace |

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
| two instances, different workspaces, server-backed configuration | runs; supported |
| two instances, **same** workspace | **not detected**. They race on every write of that workspace: pipeline state, file-backed storages (whole-file rewrites) and baseline claims. Unsupported, not an accepted residue |
| two instances collapsed onto one container by a `*_WORKSPACE` override | not detected; same as the row above. `warn_about_workspace_overrides()` announces every override at server start |
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

- `ConfigurationStorage.md`: *One server at a time on a file-backed
  configuration*, *What the lock does and does not span*, and *What the
  category does not retire*.
- `FileBackedSnapshotContract.md`: *One file per namespace per process tree*.
- `lightrag/kg/working_dir_lock.py`: the business-data residue.
