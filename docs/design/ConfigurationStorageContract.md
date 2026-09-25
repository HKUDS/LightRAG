# Configuration storage contract

Status: **implemented.** This is the contract the code keeps. It is not a
plan: every rule below is enforced by the code named next to it and pinned by
the *Acceptance scenarios* at the end.

| | |
| --- | --- |
| code | `lightrag/config_store.py`, `lightrag/config_anchor.py`, `lightrag/kg/anchor_lock.py`, `lightrag/kg/working_dir_lock.py`, `LightRAG.initialize_storages()` / `finalize_storages()`, the `config` namespace branch in each admitted KV backend |
| tools | `lightrag-rebuild-vdb`, `lightrag-clear-storage`, `lightrag-migrate-config`, and the setup wizard (`make env-storage`, `make env-validate`) |
| tests | `tests/config_store/`, beside each backend under `tests/kg/<backend>_impl/`, `tests/tools/`, `tests/workspace/test_anchor_lock.py`, `tests/setup/test_config_anchor.py` |
| read with | `VectorSpaceProvenance.md` (the verdict rules and the blind spots the baselines close) and `ServerInstanceContract.md` (several servers on one host) |

Read this before touching any of:
- `lightrag/config_store.py`, `lightrag/config_anchor.py` or `lightrag/kg/anchor_lock.py`;
- `LightRAG.initialize_storages` / `finalize_storages`;
- the `config` KV namespace on any backend;
- `config_storage` / `config_dir`;
- the baseline writes in `lightrag-rebuild-vdb`, the configuration cleanup in `/documents/clear` and `lightrag-clear-storage`, or `lightrag-migrate-config`.

Also read it before adding a key or moving a setting out of an environment variable.

## Rules at a glance

Each rule is expanded in the section named after it.

1. **One namespace, its own category.** The KV namespace `config` holds the
   server's settings and every workspace's. It is selected by
   `config_storage` / `LIGHTRAG_CONFIG_STORAGE`, which admits `JsonKVStorage`,
   `MongoKVStorage`, `PGKVStorage` and `OpenSearchKVStorage`. Anything else
   is refused at construction, by name, Redis and every vector storage
   included. Unset, it follows `kv_storage`. *(The category.)*
2. **No workspace addresses the container.** Its name is written in code: a
   fixed table, collection or index, or a file in `config_dir`, whose default
   `<working_dir>/_lightrag_config` must never move.
   `create_configuration_storage()` is the single way in, and no
   `*_WORKSPACE` variable reaches it. *(The container.)*
3. **Keys are scoped and never reparsed.** A key is `<workspace>/<suffix>` or
   `_lightrag_server/<suffix>`. The separator is `/`, never `.`. The scope is
   carried as a row field, and every suffix is declared in
   `CONFIG_KEY_REGISTRY` before anything writes it. *(Keys, Key registry.)*
4. **Reads are strict.** A read that could not complete is a failure, never
   "absent". *(Reads are strict.)*
5. **The container has an identity and each deployment an anchor.**
   `_lightrag_server/storage_identity` holds a UUID.
   `<working_dir>/_lightrag_config/storage_anchor.json` records
   `{backend, storage_uuid}`, and never follows `config_dir`. Every start
   checks both before reading any baseline. Deleting the anchor is the
   sanctioned rebind. *(The anchor and the container identity.)*
6. **Startup has a fixed order.** Steps 0a–0c come before step 1 and are not
   sticky. Steps 1 to 9 are sticky, cancellation included.
   - Only the precheck of records that exist runs before the vector
     storages initialize.
   - `INITIALIZED` means the resources exist and must be released; it does
     not mean the checks passed.
   - A failure before `INITIALIZED` rolls back everything it opened.
   - No backend acquires a process-wide resource in its constructor.

   *(Startup sequence.)*
7. **A missing baseline is established only on evidence.** That evidence is
   a positive probe or a confirmed-empty container. The claim runs under a
   keyed lock, with a strict flush and a strict read-back inside it. A flush
   that left anything buffered is a failed flush. A probe that could not run
   writes nothing. *(Establishing a baseline, Claiming a baseline atomically.)*
8. **Data first, configuration last.** A rebuild records a target's baseline
   only after that target is durable and verified. A drop deletes the records
   only after every data storage dropped. *(Rebuild, Workspace drop.)*
9. **Exclusion is per process tree, per host.** A file-backed configuration
   claims `config_dir` exclusively. The anchor lock is shared by starters and
   exclusive only for the migration. The bind lock serializes a first bind
   across the servers that share one `working_dir`. Nothing here spans
   hosts. *(One server at a time, What the lock does and does not span.)*
10. **The baselines never suppress the other two checks.** They are ANDed
    with the coverage gate and the per-container markers. *(What the
    category does not retire.)*

## Why the configuration storage exists

Two needs meet in one place.

- **Configuration must be reachable independently of the active workspace.**
  Every business storage is bound to one workspace, which is right for
  knowledge-base data and wrong for the server's own settings. The server
  reads those before any workspace is open.
- **A container name is an assertion, not a record.** Milvus, Qdrant and
  PostgreSQL name their vector containers `{folded_model}_{dim}d` from the
  *current* configuration (`_generate_collection_suffix()`), so a container
  name can never disagree with that configuration and proves nothing about
  what the vectors inside were written in. Closing the blind spots that
  `VectorSpaceProvenance.md` lists needs the adopted embedding space to be
  **recorded**, per workspace and per target, outside the vector containers.

## The container

| | |
| --- | --- |
| KV namespace | `config` |
| selected by | `config_storage` / `LIGHTRAG_CONFIG_STORAGE` — its own category |
| `JsonKVStorage` | `config_dir/kv_server_config.json` (`CONFIG_JSON_FILE_NAME`), `config_dir` defaulting to `<working_dir>/_lightrag_config` |
| `PGKVStorage` | table `LIGHTRAG_CONFIG (workspace, id, value JSONB, create_time, update_time)`, partition constant `_lightrag_config` |
| `MongoKVStorage` | collection `_lightrag_config_config` |
| `OpenSearchKVStorage` | index `x_lightrag_config_config` (the backend's sanitizer prepends `x`) |
| identity | row `_lightrag_server/storage_identity`, `value = {"uuid": <UUIDv4>}` |
| anchor | `<working_dir>/_lightrag_config/storage_anchor.json`, fixed, never following `config_dir`, and resolved to an absolute path once at construction, as `config_dir` is |

**It is a KV namespace on purpose.** KV container names carry no model
suffix, so a record kept here does not move when the embedding model changes.
That is the property the vector container name lacks, and it is why this
record can serve as evidence where the name cannot.

**One fixed container holds every row**: the server's own and every
workspace's. Configuration does not follow the knowledge base it configures.

**No workspace addresses it.**
- Every name in the table is written in code. The four backends branch on
  the `config` namespace, which nothing else is ever opened on and which no
  caller can ask for.
- `CONFIG_CONTAINER_TAG` is the constant the names are composed from. It is
  not a workspace, nothing validates it as one, and no `*_WORKSPACE` variable
  reaches it.
- `create_configuration_storage()` is the single way in. After construction
  it refuses a backend that re-bound the container elsewhere.
- A tenant may legally be called `_lightrag_config`. On each backend that is a
  different file, table, collection or index, so nothing is shared.

**The PostgreSQL `workspace` column is a partition constant, not a
workspace.** The table keeps the `(workspace, id)` primary key every other
table has, and writes the container tag into it. The workspace a row is about
lives in the payload, and in `id`.

**The JSON file name is fixed.** It is named for what it holds, not derived
from the namespace. Renaming it, like moving `config_dir`, makes every
recorded baseline read as absent.

### The category, and the four it admits

| backend | note |
| --- | --- |
| `JsonKVStorage` | the only file-backed member; see *One server at a time on a file-backed configuration* |
| `PGKVStorage` | fixed table |
| `MongoKVStorage` | fixed collection |
| `OpenSearchKVStorage` | fixed index |

- **Refused by name, at construction.** `resolve_configuration_storage()`
  refuses any other selection in `LightRAG.__post_init__`, so it never fails
  later on a missing method.
- **No vector storage may serve here.** Its container name derives from the
  embedding configuration, which is exactly what the baselines must be
  independent of. Configuration also has to be readable before any vector
  storage initializes.
- **`RedisKVStorage` is excluded.** Redis is being retired from business
  storage, and the configuration path must not be what keeps it alive.
- **Unset follows `kv_storage`.** An existing deployment's records are
  already in that backend. When the inherited selection is not admitted (for
  example Redis), it is refused with a message naming it and asking for an
  explicit one of the four.
- **One resolver for every caller.** `configuration_selection_from_env()`
  resolves `(config_storage, config_dir)` for `LightRAG`, the Gunicorn master
  and the maintenance tools alike. A second reading of the environment would
  claim or open a different container than the workers do.

### `config_dir`, and why its default is load bearing

The JSON backend keeps its file in `config_dir`, not under
`working_dir/<workspace>/`. The default is `<working_dir>/_lightrag_config`,
which is where the file has always been. Pointing the default anywhere else
makes every recorded baseline read as **absent** on the next start, and
absent is the one answer that lets a start bootstrap. A test writes that
layout by hand and asserts the recorded model comes back unchanged.

### The setup wizard

The wizard applies the same rules, so it never approves an `.env` that the
server then refuses:

- **`make env-storage`**
  - asks for a configuration backend whenever the KV selection is not one of
    the four;
  - collects the chosen backend's connection settings;
  - reports the anchor, and warns when the selection it just made resolves
    to another backend type;
  - reports only once the runtime target of the `.env` being written is
    settled, since switching between host and Compose also switches the
    directory the next server reads.
- **`make env-validate`**
  - refuses an unadmitted configuration backend, whether it is set
    explicitly or inherited;
  - refuses a backend type the readable anchor does not bind;
  - only warns about an anchor its narrow parser cannot confirm, since the
    server reads that one strictly and says why.
- **"Readable" means what the server's parser accepts.** The whole file
  must be one JSON object with exactly the three members and nothing around
  it, and a repeated key keeps its last value, as in Python's `json`. A
  per-field match would pass a file with an extra member, which the server
  refuses, and could read another backend out of a repeated key.
- **An empty host `WORKING_DIR=` is the server's start directory**, as
  `os.path.abspath("")` makes it; only an unset key gets `./rag_storage`.
- **A host `WORKING_DIR` using `${...}` is not resolved.** The server's
  python-dotenv expands it; a second expansion here would be a second answer
  to drift from the first. Both flows warn that the anchor was not checked.
- **The wizard reads the `.env` it writes: plain `KEY=value` lines.** Other
  python-dotenv forms that only a hand edit produces are neither parsed nor
  flagged: `export KEY=`, quoted or bare keys, spaces around `=`, inline
  comments, backslash escapes, multi-line quoted values, a key bound twice
  in different forms, Unicode whitespace, a lone CR as a line ending.
  Modelling dotenv's grammar in bash would be a second parser to drift from
  the server's. A hand-edited `.env` is the operator's to keep consistent:
  at worst the check runs on the plain value, and the server still refuses
  a mismatched anchor at startup with nothing written. Regeneration
  rewrites the keys the wizard manages as plain `KEY=value`. What the wizard
  writes itself it reads as dotenv does, including a doubled backslash in a
  single-quoted value, which dotenv decodes to one.
- **`.env` is the wizard's only source of values.** It is a static `.env`
  tool: the shell it runs in says nothing about the environment a server
  will start from, so exported variables are never consulted, even though a
  host server loads `.env` with `override=False`. An exported value that
  contradicts `.env` is the operator's to reconcile.
- **The path is normalized lexically before the lookup**, as the server's
  `os.path.abspath` does: `./missing/../actual` is `./actual`.
- **A lookup that finds nothing is absence only below a searchable
  directory.** An ancestor that cannot be searched or is not a directory
  makes the server's `open()` fail and refuse, so the wizard reports that
  anchor as unreadable, never absent. So does a symlink that does not
  resolve: it may be a loop (`ELOOP`) as well as dangling.
- **The migration command the wizard recommends names `WORKING_DIR`**
  whenever the anchor's directory is not the one the tool resolves from the
  host `.env`, as for every Compose deployment.
- **`make env-base` and `make env-server`** run the same admitted check
  before writing. They ask for a backend only when the file would otherwise
  be unstartable, and leave a sound `.env` byte-identical. They can switch
  between host and Compose too, so they report the anchor the same way once
  their runtime target is settled.
- **The wizard never moves the container as a side effect.**
  `select_config_storage` computes *where the records are* once, from the
  previous `.env`, and every branch reads that one answer:
  - an explicit `LIGHTRAG_CONFIG_STORAGE` is never dropped;
  - a changed KV backend prompts, defaulting to the backend the records are
    in, instead of taking the container with it.
- **The wizard reads the anchor and never writes, moves or deletes it.** It
  reads it at the `WORKING_DIR` the server will use on this host, which is
  `./data/rag_storage` for the compose runtime.
- **Operator edits to the compose file are kept, not interpreted.**
  Regeneration preserves what an operator adds to the lightrag service,
  such as another mount or an `environment:` entry like
  `LIGHTRAG_KV_STORAGE` (which then outranks `.env`). The wizard does not
  read any of it; following arbitrary edits would make it a second Compose
  implementation. Keeping such an edit consistent with the anchor is the
  operator's responsibility, and a finding that the wizard misjudges an
  edited compose file is outside its scope.

### `*_WORKSPACE` is legacy compatibility, and baselines do not follow it

A baseline is keyed by the workspace the **caller** named. Each backend
applies its own `*_WORKSPACE` override inside its constructor, below the
layer that computes that key. An override therefore deliberately collapses
distinct logical workspaces onto one physical container, and the record and
the container it describes can sit under different names.

The key is not re-scoped by the effective workspace. The overrides exist to
keep legacy data reachable, and they are meant to be invisible above the
storage layer. Keying records by them would build them into the record format
and bless the unsupported act of using one to *move* data.

So the rule is announced, not enforced. `warn_about_workspace_overrides()`
runs once per server start, from `lightrag_server.main` and the Gunicorn
master's `on_starting`, and names every override in effect. Recovery if one
was used to move data: point it back, or rebuild the moved target with
`lightrag-rebuild-vdb`. The configuration container itself is never affected,
because no override applies to a name written in code.

### What discriminates two deployments on one server

The container's name is the same for everyone, so two deployments pointed at
one PostgreSQL, MongoDB or OpenSearch share its table, collection or index.
The **row key's scope**, the business workspace the row is about, keeps their
records apart, and different deployments are required to use different
business workspaces (`ServerInstanceContract.md`). Two deployments that share
a backend **and** a workspace name overwrite each other's baselines. That is
the unsupported case listed under *Accepted residues*.

**The container identity is not a deployment id.** The UUID identifies the
whole container and is never a row discriminator. It lives inside the
container and travels with it (a dump/restore keeps it). A start without an
anchor adopts the UUID the container already has, and an ordinary redeploy
that keeps `WORKING_DIR` and the container passes unchanged. Two deployments
sharing one container correctly share its identity. The deployment id this
section rejects is in *Rejected alternatives*.

### One server at a time on a file-backed configuration

`JsonKVStorage` shares its in-memory copy only inside one process tree and
publishes by rewriting the whole file. Two process trees on one `config_dir`
would each rewrite the file over the other. An overwritten baseline reads
back as **absent**, which lets the next start bootstrap the configured model
over vectors nobody probed, with nothing in any log. The in-process guard
(*One file per namespace per process tree* in `FileBackedSnapshotContract.md`)
cannot see another process tree.

So a file-backed configuration storage **claims `config_dir`** for the life
of its process tree (`acquire_working_dir_lock`), and a second process tree
is refused with `WorkingDirectoryInUseError`.

- **It is an OS lock, not a PID file.** The kernel releases it when the
  holder dies, so nothing stale is left to reap.
- **`fork` shares it.** The Gunicorn master takes it in `on_starting`, before
  forking. The workers inherit it and count themselves in. The master
  resolves the directory with `configuration_selection_from_env()`, fed the
  **parsed** working directory: `--working-dir` is never written back to the
  environment, so `run_with_gunicorn` hands the parsed value to the config
  module, and `resolved_working_dir()` prefers it.
- **It fails open.** Where the filesystem cannot lock (NFSv3 without lockd,
  SMB/CIFS), it logs a warning and proceeds.
- **Only a file-backed configuration storage claims.** A server-backed one
  claims nothing.
- **`lightrag-rebuild-vdb` and `lightrag-clear-storage` take the claim too.**
  Each is a second process tree by construction.

Three lock files can sit in `<working_dir>/_lightrag_config/`, and each one
makes a separate statement:

| file | mode | taken by | purpose |
| --- | --- | --- | --- |
| `.lightrag_storage.lock` | exclusive | a JSON configuration storage's process tree | one server per JSON configuration file |
| `.lightrag_anchor.lock` | **shared** for every starter, **exclusive** for `lightrag-migrate-config` | server, Gunicorn master before fork, SDK, rebuild and clear tools; the migration | no migration while anything runs on this `working_dir` |
| `.lightrag_anchor_bind.lock` | exclusive, polled, bounded wait | only a start that finds **no** anchor | one first bind across the servers sharing this `working_dir` |

The order is: the anchor lock, the `config_dir` claim, then the bind lock
inside step 1b. Where locking is unavailable (a filesystem without locks, a
read-only directory, or Windows, whose `msvcrt.locking` has no shared mode),
every starter lock fails open with a warning. The migration then refuses
unless the operator passes `--assume-exclusive`.

#### The claim goes back last, and only after the teardown

Releasing the claim while this process still holds shared-namespace holds or
unflushed writes would let the next server rewrite the same file. So
`finalize_storages()`:

- keeps the queue drains interruptible, and absorbs a cancellation there;
- runs the storage teardown as a shielded task, drained to completion;
- releases the `config_dir` claim, then the anchor lock, in a `finally`,
  after that task;
- re-raises the absorbed cancellation.

The startup rollback shields each release. It drains every release that a
cancellation detached before it hands the claim back, because
`asyncio.shield` keeps a release alive but does not finish it.

## Keys

```
<workspace>/<suffix>          a per-workspace setting
_lightrag_server/<suffix>     a server-global setting
```

- **The separator is `/`, never `.`.** `validate_workspace()` forbids `/` but
  allows dots (`"v1.0"` is a legitimate workspace name), so a dotted key
  could not be split back apart unambiguously.
- **Keys are never reparsed anyway.** The row carries `workspace` as a field,
  and every reader classifies by the field. The separator rule is a second
  lock on a door the row shape already closes.
- **The server scope is an object, not a string.** A tenant may legally be
  called `_lightrag_server`. `SERVER_SCOPE` is a sentinel: `config_key()`
  compares by identity and renders the prefix afterwards. A suffix is
  registered with exactly one scope, so a tenant key and a server key can
  never be the same key.
- **OpenSearch's lossy sanitization cannot reach the container.** For the
  `config` namespace, `_resolve_workspace` consults no workspace at all.
  `_build_index_name` refuses, before a client opens, any *other* namespace
  whose index name would normalize onto the container's.

## Row shape

Every row has the same shape:

```json
{
  "schema_version": 1,
  "workspace": "<workspace or _lightrag_server>",
  "updated_at": "<iso8601>",
  "updated_by": "<component that wrote it>",
  "value": { }
}
```

`schema_version` is per key, not global. A reader requires an integer equal
to the key's registered version before it interprets `value`. A missing,
unsupported or wrongly typed version (booleans included) is an unreadable
record: it raises `ConfigurationStorageError`, and is never treated as absent
or replaced at startup.

## Key registry

`CONFIG_KEY_REGISTRY` is the complete list of keys. Every entry declares its
suffix, scope, schema and `schema_version`, readers, writers and `sensitive`
flag. `make_config_row` refuses an unregistered suffix. Nothing enforces the
`readers` / `writers` tuples, so a new caller declares itself there before it
touches a row, and `tests/config_store/test_config_store.py` pins the lists.

| suffix | scope | readers | writers |
| --- | --- | --- | --- |
| `storage_identity` | server | startup, rebuild tool, clear tool, migration | startup (bind), migration (into its target) |
| `embedding/entities`, `embedding/relationships`, `embedding/chunks` | workspace | startup, rebuild tool, clear tool, migration | startup (claim), rebuild tool (record), `/documents/clear` and clear tool (delete), migration (copy) |

No key is `sensitive` yet. What that flag must *do* is an open item (*Not
implemented yet*).

## The anchor and the container identity

When `LIGHTRAG_CONFIG_STORAGE` is unset, the configuration backend is
re-resolved from `kv_storage` on every start. Without an anchor, a changed KV
selection would silently move the container. The recorded baselines would be
replaced by fresh probe verdicts, and records that probing cannot re-derive
would be lost. The anchor turns that drift into a refusal.

**It has two pieces.**

- **The identity.** `_lightrag_server/storage_identity`, registered with
  `SERVER_SCOPE`, holds `{"uuid": <UUIDv4>}` and identifies the whole
  container. Every workspace in the container shares it.
  - It is generated once, by `new_storage_uuid()` (`uuid.uuid4()`, drawn from
    the OS CSPRNG), and only when the row is confirmed absent.
  - It is never overwritten once valid.
  - A workspace clear, a rebuild and row maintenance never touch it:
    `delete_workspace_configuration` deletes only registered per-workspace
    suffixes.
- **The anchor.** `<working_dir>/_lightrag_config/storage_anchor.json`
  (`lightrag/config_anchor.py`) holds exactly `schema_version`, `backend` and
  `storage_uuid`. It holds no host, port, credential or connection string.
  Its path depends only on `working_dir` and does **not** follow
  `config_dir`: moving the anchor with the data would move the check along
  with the thing it checks.

A UUID is used rather than only the backend type because a type-only anchor
catches KV drift and nothing else. The UUID also catches a same-type change:
another database or an empty one, a changed `LIGHTRAG_CONFIG_DIR`, or a
restore from a backup older than the identity.

### What is compared

| change | result |
| --- | --- |
| IP, port, DNS or credentials change; type and UUID unchanged | passes; a connection failure still fails normally |
| same backend type, different UUID | refused |
| anchor present, container UUID missing | refused; no UUID is created |
| backend type changed, even with the UUID copied over | refused |
| JSON `config_dir` changed | the fixed anchor is still read, and the verdict is the target container's UUID |
| identity matches, an embedding baseline does not | refused by the baselines, unchanged |
| anchor deleted by the operator | the no-anchor branch: adopt the container's UUID, or create one |

The UUID identifies a logical container. It is not a credential and does not
prove that a database is unique or complete: a clone or an old backup carries
the same UUID. Clone, rollback and tamper detection are out of scope.

### Reads and writes

- **The anchor is read strictly.** Its structure, the exact `schema_version`,
  an admitted backend and a canonical UUID are all validated. Only "file does
  not exist" is the no-anchor branch. A permission error, a directory in its
  place, a truncated or corrupt file or an unknown version refuses
  (`ConfigurationIdentityError`, cause `anchor_unreadable`).
- **The identity row is read strictly.** A backend error, a wrong schema
  version or scope, or a non-canonical UUID is an error, never absent.
- **The identity is written through `flush_configuration_storage`**, with a
  strict read-back. A write still retained in OpenSearch's process-local
  buffer is therefore a failure (*A flush that retained anything is a failed
  flush here*).
- **The anchor is written durably, in two modes.** Both write a temp file in
  the same directory, `fsync` it, publish it, and then `fsync` the directory
  where the platform can. Every failure raises with a definite message; none
  is reported as success.
  - The **bind** publishes no-clobber: `link` then `unlink` on POSIX,
    `rename` on Windows. Where hard links are unavailable it claims the name
    with `O_CREAT | O_EXCL` and then replaces that empty claim with the temp
    file, so it rests on no lock (the bind lock fails open, and the keyed
    lock sees one process tree). The residue: until the replace lands, the
    anchor is an empty file that every reader refuses as unreadable. A crash
    in that window leaves the empty file for the operator to delete, which
    is the sanctioned rebind.
  - The **migration** publishes by atomic replace. Nothing else ever
    replaces an anchor, and a normal start never overwrites or deletes one.
- **`WORKING_DIR` must persist,** whichever backends are selected, and the
  anchor belongs in backup and restore. An ephemeral `WORKING_DIR` loses the
  anchor on every restart and makes the check vacuous, which is why the
  no-anchor bind logs at WARNING. A read-only `WORKING_DIR` fails the first
  bind loudly.

### Startup: steps 0a–0c and 1b

```
0a. take the shared anchor lock                            (not sticky)
0b. strict-read the anchor
0c. anchor present and the candidate backend type differs
      -> refuse before the configuration storage initializes
    (then the JSON config_dir claim)
1.  initialize the configuration storage
1b. under keyed lock "configuration_identity"             (sticky; rolled
      re-read the anchor                                     back like step 2)
      anchored:     identity equal -> continue
                    missing / different / invalid / error -> refuse
      not anchored: take the bind lock, re-read the anchor
                      (another server may have bound: verify against it)
                    identity present -> adopt it
                    confirmed absent -> create it: upsert, strict flush,
                                        strict read-back, compare
                    publish the anchor {candidate backend, uuid}, no-clobber
                    WARNING: container, uuid, adopted or created
2-9. see *Startup sequence*
```

- **Steps 0a–0c open nothing**, so a failure there is ordinary, not sticky,
  and hands the lock back (*Where stickiness starts*).
- **The Gunicorn master runs 0a–0c in `on_starting`** before forking, so a
  type mismatch refuses the master rather than every worker. Workers run
  0b–1b themselves.
- **The anchor is published as soon as the identity is confirmed**, before
  any business storage initializes. It records the binding, not whether any
  baseline is recorded.
- **The keyed lock serializes the workers of one master.** The bind lock
  serializes a first bind across the servers sharing a `working_dir`: those
  servers all bind the same container-wide row, and the keyed lock cannot
  see them. An anchored start never takes the bind lock, so running servers
  never serialize on it.

**Why there is no `pending` state.** The identity is written first and the
anchor second, so every interruption heals by adoption:

| interrupted after | next start | outcome |
| --- | --- | --- |
| identity durable, anchor not published | no anchor, identity present → adopt | heals |
| anchor publish failed | start fails loudly; the retry adopts the durable identity | heals |
| any of steps 2–9 | anchor and identity already consistent | the sticky-failure rules |

### Refusals, and the rebind

**Deleting `storage_anchor.json` is the sanctioned rebind.** The next start
binds to whatever container the current configuration selects, and logs a
WARNING naming the container and the UUID. For that one start, drift
detection is off and protection falls back to the baselines, the coverage
gate and the per-container markers.

It is the recovery for a container intentionally emptied or replaced, or
restored from a backup older than its identity. Never delete the anchor while
servers run: running processes do not re-read it, and a respawned Gunicorn
worker would rebind.

Every refusal names the expected and actual backend type and UUID, and the
anchor's path, and hides credentials. The advice depends on the cause:

| refusal | first advice | deleting the anchor |
| --- | --- | --- |
| container UUID missing | if the container was intentionally emptied, replaced or restored from an old backup, delete the anchor (path given) | primary recovery |
| backend type differs | set `LIGHTRAG_CONFIG_STORAGE` to the anchored backend explicitly, or run `lightrag-migrate-config` | listed last, marked as abandoning every record in the old container |
| same type, different UUID | check that the connection settings point at the intended database | listed last, same warning |

### Maintenance tools

`lightrag-rebuild-vdb` and `lightrag-clear-storage`:
- take the shared anchor lock;
- resolve the backend through the same selection code as a start;
- refuse a backend-type mismatch before anything opens;
- **verify** the identity (`verify_configuration_identity`) after the
  configuration storage opens and before any data storage does.

They never create or rewrite the anchor or the identity. With no anchor they
say that nothing was verified, and proceed.

### Offline migration: `lightrag-migrate-config`

`lightrag/tools/migrate_config.py`; the operator guide is
`lightrag/tools/README_MIGRATE_CONFIG.md`.

- **It moves the container across backend types only.** A same-type move is
  the backend's own dump/restore: the identity travels with the data, and
  "same type, same UUID" passes.
- **It keeps the UUID.** The backend type already tells the source and the
  target apart.
- **It resolves both sides' connections from separate env files.** Two
  backends of different types read disjoint variables. The tool refuses
  conflicting values for a variable either backend reads, and refuses an env
  file that names a different `WORKING_DIR`. It never writes an env file or
  logs a credential.

```
1. take the anchor lock EXCLUSIVELY; strict-read the anchor
     no anchor, target unadmitted, or target type == anchored -> refuse
2. open the source as the anchored type; its identity must equal the anchor's
   enumerate it; a malformed row is refused and listed
3. classify the target
     empty                         -> claim it
     identity == the anchor's UUID -> an earlier attempt's residue: resume
     anything else                 -> refuse; no overwrite, no merge
4. write the target's identity (same UUID) FIRST: the ownership marker
5. copy every row, paged, key and envelope verbatim; delete and flush the
   target rows the source does not hold or holds differently BEFORE upserting,
   so a merging upsert keeps no stale field
6. strict flush; compare every row with the source, backend metadata excluded
7. replace the anchor {target type, same UUID} atomically   <- COMMIT
```

- **`--dry-run`** takes the lock shared, runs steps 1–3 and writes no row.
  Opening the target at step 3 is an ordinary backend `initialize()`, which
  provisions a missing container as any start on that backend would. The
  identity is the first **row** the migration writes, not the first byte on
  the target.
- **Before step 7 the anchor is unchanged**, and a re-run resumes, converging
  the target onto the *current* source. A source lost during the copy is such
  a failure too, never a raw backend error. So is a claim whose identity
  write errored: the server may have committed the marker before the client
  saw the error, so it is never reported as a refusal, and the re-run's
  classification finds the marker. The rule is general: from the claim on
  (steps 5–6), nothing is reported as a refusal. A source row that turned
  bad after step 2, or a target identity that reads back malformed during
  verification, fails the attempt, and the re-run refuses by name.
- **"Well-formed" means the fields a reader interprets:** an integer
  `schema_version`, a string `workspace` and a mapping `value`. The
  diagnostic `updated_at` / `updated_by` are copied verbatim and verified by
  digest; they are never grounds to refuse. A backend's typed corruption
  (`CorruptStorageRecordError`) is one damaged row, refused by name, not a
  store failure.
- **The tool gives back its `config_dir` claims before the anchor lock,**
  the order every starter uses. A storage whose `initialize()` fails is
  finalized before the failure is reported.
- **The anchor read under the lock must be the one the command started
  from.** The command chose which connection settings are the source's from
  the anchor it read before the lock. If another migration moved the anchor
  in between, the run is refused before anything opens.
- **A source that cannot be read at step 2 is a refusal,** never a
  resumable failure. Nothing is claimed yet: restore or reconnect the source.
- **A target refuses any source row carrying a field it reserves,** before
  the claim: `id` on PostgreSQL (returned as the key on every read) and
  `__mirrored_id` on OpenSearch (written as the key and dropped from every
  read). Such a field could neither be read back nor told apart from the
  backend's own value on a later migration out of it.
- **A replace that reports failure is re-checked against the file,** so a
  directory `fsync` failing after the replace landed is reported as switched.
  When the anchor then cannot be read back, the outcome is reported as
  **indeterminate**, neither switched nor unchanged, and the operator
  inspects the anchor before starting anything.
- **The target's connection settings must be persisted.** What
  `--target-env` supplied exists only in the tool's process. The success
  message lists those settings by name, never by value; without them the
  server opens a different target container and is refused.
- **No source row is ever written or deleted.**
- **Only a key mirror is metadata.** A row's `_id` and backend timestamps
  are the backend's own on every backend. `id` is dropped only for rows read
  from `PGKVStorage`, the one admitted backend that returns the key as both
  `id` and `_id`. No admitted backend yields `__mirrored_id`: OpenSearch
  writes it and drops it from every read. Any other field, including an `id`
  equal to its key or a row's own `__mirrored_id`, is envelope content: it is
  copied and verified, never silently dropped from both sides.
- **A damaged record in this migration's own target is not converged
  automatically.** The enumeration dies at that record, so there is no
  complete listing to converge from. The refusal says the target is this
  migration's residue, and that removing the row, or discarding the target
  container, is safe.

## Embedding baselines: one per vector target

```
<workspace>/embedding/entities
<workspace>/embedding/relationships
<workspace>/embedding/chunks
```

`value` is `{model, dim, origin}`. The model is stored **unfolded**
(`declared_model_name()`), as the per-container marker stores it.

There are three records, not one per workspace, because
`lightrag-rebuild-vdb` rebuilds the three targets as three separate steps
against three containers. After an interrupted or partial rebuild they
legitimately sit in different spaces, and a single record could not describe
that.

### What the record means

> **The embedding space adopted as this target's active baseline.**

It is not "the space these vectors were written in", which the store cannot
know for a container it never probed. `origin` records how the claim was
established. It is **diagnostic only and never enters a verdict**:

| `origin` | established by |
| --- | --- |
| `probe` | the homogeneous cosine probe reproduced stored vectors under this model |
| `empty` | the source and the index were both empty |
| `rebuild` | a successful `lightrag-rebuild-vdb` of this target |

### Verdicts

Each target is compared on its own against the configured embedding
function:

| record | verdict |
| --- | --- |
| present, `model` or `dim` differs | **refuse to start**, naming the target and both spaces, and pointing at `lightrag-rebuild-vdb` |
| present, equal | proceed |
| confirmed absent | a bootstrap target, established after the storages are up |
| unreadable | startup failure (*Reads are strict*) |

When several targets mismatch, one refusal names **all** of them
(`EmbeddingBaselineMismatchError`). The dimension is compared only when both
sides declare one.

### Establishing a baseline: every target on its own evidence

Each target's probe samples from that target's own source:

| target | sample |
| --- | --- |
| `entities` | the graph's most-connected labels (`get_popular_labels`), mapped to entity vector ids |
| `relationships` | the first batch of `iter_edges`, mapped to **both** candidate relation vector ids (`make_relation_vdb_ids`: canonical, then the legacy reverse order) |
| `chunks` | the first page of `text_chunks.iter_rows()`: one bounded round trip, and the row id is the chunk vector id |

- **The source verdict comes from a read that raises when it fails.**
  "Empty" becomes a durable write, so an outage must never read as empty.
  `BaseKVStorage.is_empty()` swallows errors on the server backends, so the
  chunk source is read through the first page of `iter_rows()` instead. A KV
  backend without enumeration answers "unknown", and nothing is recorded on
  it.
- **Only a target with a baseline to establish pays for that read.** The
  strict read, the probe and the container's own `is_empty()` are all keyed
  on `baseline_targets`, the targets the precheck found absent.
- **A verdict is about one container only.** Nothing is recorded, and no
  marker adopted, on a sibling's verdict.

Per target:

| source empty | source populated |
| --- | --- |
| Record `origin=empty` only if that target's vector container is also confirmed empty. If vectors survive behind an empty source, or either read failed: leave absent, warn, and point at the rebuild. | Run that target's probe. Negative: **refuse**, and write nothing. Positive: record `origin=probe`. Could not run (embedder down, unreadable, nothing to sample): leave absent and retry on the next start. |

## Startup sequence

`LightRAG.initialize_storages()`:

```
0a-0c. the anchor lock, the anchor, the backend-type check; the config_dir claim
1.  initialize the configuration storage
1b. verify the container's identity against the anchor, or bind it
2.  strict-read this workspace's three baselines
3.  PRECHECK the records that exist:
      any mismatch -> refuse; no vector storage is initialized
      absent       -> remember as a bootstrap target
4.  initialize the business storages (KV, graph, doc status, three vector)
5.  mark INITIALIZED
6.  the coverage gate and the entity adoption probe
7.  establish the baselines remembered in step 3
8.  flush the configuration storage (through flush_configuration_storage)
9.  return
```

- **Only step 3 must precede step 4.** That order puts the mismatch refusal
  ahead of the legacy-container migration that Milvus, Qdrant and PostgreSQL
  run inside `initialize()`.
- **When a record is absent, step 4 runs before anything has judged the
  data.** That migration may therefore copy rows first. This is an accepted
  residue.
- **The precheck cannot cover everything.** The adoption probe reads
  `entities_vdb`, so it can only run after step 4.

### `INITIALIZED` means the resources are up, not that the service may serve

| | meaning |
| --- | --- |
| `INITIALIZED` | the resources exist, and `finalize_storages()` must release them |
| a successful return from `initialize_storages()` | the checks passed, and the instance may serve |

`finalize_storages()` releases nothing unless the status is `INITIALIZED`,
so step 5 sits before the checks that can refuse. The configuration storage
is an ordinary member of the teardown list: initialized first, finalized
last, exactly once.

### Cleanup before `INITIALIZED` exists

Steps 1 to 4 run while the status is still `CREATED`, so any failure there
releases, in `_release_after_early_failure`, everything that was started:

- every storage the loop **started** is tracked, in order;
- on failure, each is finalized in reverse order: the one that raised (it
  may have allocated), then those that succeeded, then the configuration
  storage;
- a teardown failure is logged and never replaces the original exception;
- a storage the loop never reached is left alone.

#### Construction takes nothing

**No backend acquires a process-wide resource in its constructor.** A
storage the loop never reached is therefore holding nothing. This is
load-bearing and easy to break: a constructor has no teardown path, and a
refusal in `LightRAG.__post_init__` leaves nothing to call
`finalize_storages()` on. A backend must take pools and clients in
`initialize()`. The configuration storage is still constructed as the last
statement of `__post_init__` that can raise, and a new check in
`__post_init__` goes above it.

### Where stickiness starts

- **Steps 1 to 9, 1b included, are sticky.** Any failure there, before or
  after `INITIALIZED`, is retained (`_retain_startup_failure`) and re-raised
  by every later call. Otherwise a retry would early-return on the status as
  ready, or re-run steps against storages a rollback closed. **A new
  instance is the retry.**
- **The preamble is not sticky.** It opens nothing and moves no status, so a
  failure there is ordinary and the next call re-runs every step. The
  preamble is: binding the event loop, the default workspace,
  `pipeline_status`, the anchor steps 0a–0c, and the directory claim.
- **Cancellation is a failure too.** `BaseException` is retained, and an
  interruption that cannot itself be re-raised later is retained as a
  `RuntimeError` naming it.

## Claiming a baseline atomically

`claim_embedding_baseline` runs under a keyed lock per `(workspace, target)`:

```python
async with get_storage_keyed_lock(key, namespace="configuration_embedding_claim"):
    row = await read_config_row_strict(config, key)
    if row is None:
        await config.upsert({key: candidate})
        await flush_configuration_storage(config, ...)   # INSIDE the lock
        row = await read_config_row_strict(config, key)
    validate(row)                                       # against THIS process's config
```

The flush completes inside the lock, because `OpenSearchKVStorage.upsert`
buffers in process memory until it is flushed. The strict read-back
validates what is **actually stored** against this process's configuration,
so a record another worker got in first with is judged, not assumed.

### A flush that retained anything is a failed flush here

`flush_configuration_storage()` is the only flush configuration code uses:
for the claims, the identity, the rebuild record, the drop, the migration and
step 8. It **asks the buffer, not the return value**:

- Nothing is retained (`has_pending_index_ops(include_deletes=True)` is
  false): the write landed, and the strict read-back confirms it. This holds
  even when the flush **raised** `ReferencesIntactFlushError`, because the
  bulk landed and only the refresh after it failed. Treating that as a
  failure would report a durable write as one that did not happen.
- Something is retained: the buffer is dropped, so what the caller reports is
  what is true, and `ConfigurationStorageError` is raised. OpenSearch keeps
  per-item retryable failures (408, 429, 5xx) buffered and returns normally,
  and its strict read would confirm them from the buffer.
- A backend that cannot be asked keeps the conservative reading of its own
  raise. A backend without a buffer answers "nothing retained" and pays
  nothing.

### What the lock does and does not span

| scope | shared |
| --- | --- |
| coroutines in one process | yes |
| workers forked from one Gunicorn master | yes |
| two independent masters, containers, hosts, or SDK processes | **no** |

A read-back does not extend that boundary: two masters can each read absent,
each write, and each read back their own write. So concurrent initialization
of **one workspace** by separate process trees is **unsupported, not a
handled residue**. Instances on different workspaces claim different keys and
never contend (`ServerInstanceContract.md`). The container-wide identity is
serialized across servers on one `working_dir` by the bind lock. A first bind
from different working directories or hosts against one container is
unsupported.

## Reads are strict

```
confirmed absent  -> bootstrap (or, for the identity, adopt/create)
read failure      -> startup failure
present, differs  -> refusal
present, equal    -> proceed
```

- **Every decision read goes through `read_config_row_strict`.** It uses
  `get_by_id_strict()`, and checks the backend's
  `supports_strict_point_reads` ClassVar, whose default on `BaseKVStorage` is
  `False`.
- **A deliberately emptied store refuses while an anchor exists.** Its
  identity is gone, so step 1b stops the start before any baseline is read.
  Deleting the anchor is the recovery.
- **A fetched record that is not a row raises
  `ConfigurationRecordMalformedError`**, a subclass of
  `ConfigurationStorageError`, so everything that must stop still stops.
  `lightrag-clear-storage` alone may go on: it shows such a record as
  UNREADABLE and drops the workspace, because deletion is by key and never
  reads the value. `JsonKVStorage` raises `CorruptStorageRecordError` for a
  non-mapping payload, so that case is typed rather than read as an outage.
- **Strictness survives the layer below the read.** The JSON backend loads
  its file once per process tree under `namespace_init_claim`. A load that
  fails, cancellation included, hands the claim back, so the next instance
  reads the file again rather than an empty namespace.

## Rebuild: one target at a time, configuration last

Per target:

```
rebuild the target's vectors
-> flush the vector storage
-> verify: nothing retained (has_pending_index_ops(include_deletes=True))
-> record THAT target's baseline (record_embedding_baseline)
-> flush the configuration storage
```

- A rebuild of `entities` updates `embedding/entities` and nothing else.
- **A returning flush is not the verification.** A retained operation left
  by the last flush, or an answer that could not be read, fails the rebuild:
  the baseline would otherwise claim an incomplete index, and neither the
  precheck nor the coverage gate would catch it. `OpenSearchVectorDBStorage`
  implements `has_pending_index_ops` for that reason.
- A failed rebuild advances nothing. A rebuild whose baseline record fails
  makes the **tool exit non-zero**; the stale record keeps refusing, and a
  re-run converges.
- The tool bypasses the mismatch refusal only for the targets it rebuilds.
  Its sources (the graph and `text_chunks`) open through the strict path.

## Workspace drop: data first, configuration last

```
drop every data storage of the workspace
-> confirm every drop succeeded
-> delete the three baseline records (delete_workspace_configuration)
-> flush, and strict-read that every one is gone
```

| residue | consequence |
| --- | --- |
| data gone, configuration remains | a workspace recreated under the same name may be refused until the rows are cleaned. **Accepted** (loud, recoverable) |
| configuration gone, data remains | the next start bootstraps over surviving vectors. **Never acceptable** |

- If **any** drop fails, all three records stay. A partial drop never
  deletes the records "for the parts that dropped".
- **A cancelled drop is a drop that did not happen.** The drop results are
  classified on `BaseException`, because `asyncio.gather(...,
  return_exceptions=True)` returns a `CancelledError` *object*.
- The two callers are `/documents/clear` and `lightrag-clear-storage`.
  `/documents/clear` counts its opt-in LLM-cache drop as a data drop.
  `lightrag-clear-storage` never drops the cache.
- The identity row is never deleted by either caller.

## Enumeration, and what the inventory really costs

`BaseKVStorage.iter_rows(page_size=...)` is the enumeration surface. All five
KV backends implement it, and four of them can serve as the configuration
storage.

- **It is paged or streaming,** never "load every row".
- **It raises when the container cannot be read,** and never ends the stream
  early. A missing index or a closed connection must not read as an empty
  listing. `OpenSearchKVStorage` refuses in the same index-missing states
  its strict point read refuses in.
- **Classification is by the row's `workspace` field**
  (`iter_configuration_rows`), and never by reparsing the key.
- **A page is a best-effort snapshot, not a set.** Redis `SCAN` may repeat a
  key. The chunk sampler keeps distinct ids and spends its budget on rows
  examined.

**Startup reads at most the first page**, and only for a target whose
baseline is absent. `JsonKVStorage` snapshots its key list before the first
page, so the cost is paid once, not on every start. On Redis, proving
emptiness walks the keyspace, but so does the `is_empty()` it replaces.

## What the category does not retire

- **The working-directory claim stays.** Two servers on one `working_dir`
  and one **workspace** still overwrite each other's file-backed business
  data. On different workspaces they share only the configuration container
  (`ServerInstanceContract.md`). The claim follows the configuration file
  onto `config_dir`.
- **The baselines answer one question: identity** — is the configured space
  the one adopted for this target? Two mechanisms answer different questions
  and stay:
  - **coverage** (`BaseVectorStorage.is_empty()` and the source/index
    pairing): does the index cover the data it indexes?
  - **the per-container markers,** which travel with the data and catch a
    container copied in from elsewhere.

  All three are ANDed, and a baseline **never suppresses** a refusal from
  either of the others. Every refusal names the invariant that failed, the
  target, and the command that fixes it.

## Standing obligations

- **The store never holds what is needed to reach the store.** The backend
  selection and its connection settings stay in the environment.
- **Secrets need a policy before the first secret goes in.** The registry's
  `sensitive` flag exists. What it does is undecided: refuse to echo, encrypt
  at rest, or refuse storage entirely. It must be decided before any key is
  marked sensitive.
- **Every new caller of a key declares itself** in the registry's `readers`
  / `writers`.

## Accepted residues

Each is a decision with a recovery path (*Consistency without transactions*
in `AGENTS.md`).

- **A legacy-container copy is made before any judgement.** With records
  absent, the Milvus, Qdrant and PostgreSQL legacy migration inside
  `initialize()` may copy rows into a `{model}_{dim}d` container before the
  probe runs. The copy is never served if the probe refuses. *Recovery:*
  `lightrag-rebuild-vdb`.
- **An orphan configuration row remains after a failed drop cleanup.** It
  outlives its workspace. *Recovery:* a maintenance pass that deletes rows
  whose scope names a workspace with no data.
- **Without an anchor, a replaced or cleared store loses every baseline.**
  With an anchor, step 1b refuses. Without one, reads confirm absent and the
  start bootstraps; this is announced by `warn_about_unrecorded_baselines()`,
  not enforced, because it is indistinguishable from a first start. Even
  then, nothing is adopted on the configured model alone.
- **The no-anchor boundary.** A deleted anchor, a replaced volume and an
  ephemeral `WORKING_DIR` all look like a first start and bind to whatever
  is selected, with a WARNING. *Recovery when unintended:* restore the
  anchor, or point the configuration back and delete the anchor the wrong
  start wrote.
- **An orphan identity row can remain after an interrupted bind.** If the
  configuration is repointed between the identity write and the anchor
  publish, the first container keeps an identity no anchor names. A later
  bind adopts it, and an anchored start against it refuses. Harmless in both
  directions.
- **The migration's ownership marker cannot tell its own residue from an
  older copy of the same identity.** It does not need to: both belong to
  this identity, and step 5 converges either. A target shared with another
  deployment is out of scope.
- **Either side may have been provisioned** by its `initialize()`: the
  source (even one the identity check then refuses as the wrong container)
  and a target that is refused or only dry-run. That means an empty table or
  index created, or an existing container's schema and container marker
  brought to this version, as any start of this version against it would.
  No row is ever written or deleted on the source, nor on the target before
  the verdict. A non-mutating probe per backend would remove this, at the
  cost of a second open path in every admitted backend. That cost is not
  worth it for a container that only a LightRAG configuration storage uses.
- **Two deployments sharing one backend AND one workspace name overwrite
  each other's baselines.** This is unsupported. *Recovery:* give them
  different workspaces, and rebuild.
- **The JSON backend publishes its whole namespace on flush,** so one file
  is the write point for every workspace in the process. This is accepted
  under the "file-backed storages are for small-scale testing" limit.
- **Every workspace's configuration is colocated in one container.**
  Configuration is server-owned, so combining it is the correct semantics.

## Rejected alternatives

Each of these has been decided. Reopening one needs evidence that its
reason no longer holds.

| alternative | why not |
| --- | --- |
| a reserved workspace name for the container (`_lightrag_config` as a tenant name) | a reserved name must be defended everywhere a name can be chosen (case-insensitively for OpenSearch, and at every `*_WORKSPACE` override). A container nobody can address by workspace needs no defence |
| a deployment id as a row discriminator | an operator-set id can be copied into a clone, and a derived id moves when the container does; either way records read as absent after a redeploy |
| keying baselines by the effective (`*_WORKSPACE`) workspace | it would bake a legacy override into the record format, and bless using one to move data |
| a default `config_dir` anywhere but `<working_dir>/_lightrag_config` | every existing baseline would read as absent |
| moving `INITIALIZED` ahead of the storage loop, or an `INITIALIZING` status | teardown would call `finalize()` on storages that never initialized, which no backend contract offers |
| a retry on the same instance after a step 1–9 failure | not every `finalize()` can be undone (OpenSearch flushes its buffer before its client guard); a new instance is the retry |
| a type-only anchor | it catches KV drift only; same-type container changes need the UUID |
| a `pending` anchor state or a bind journal | identity-first, anchor-second ordering heals every crash by adoption |
| a fresh UUID on migration | the backend type already discriminates; a second identity would only be one more thing to verify |
| a same-type migration tool | native dump/restore keeps the UUID. The PostgreSQL, MongoDB and OpenSearch client managers are process-wide singletons reading `os.environ`, so two containers of one type cannot be open in one process |
| an exclusive anchor lock for every starter | it would refuse starters that coexist today; only the migration needs exclusion |
| a `rebind` command | deleting the anchor covers it; one is added only if misuse shows up |
| a distributed lock or create-if-absent/CAS in each backend, for concurrent initialization across hosts | a project of its own, not required by the supported deployment shape |

## Not implemented yet

- **The server-level pair `_lightrag_server/embedding.current` /
  `.previous`, and the startup inventory** naming which workspaces still
  need a rebuild, both built on `iter_rows()`. The pair must stay
  **diagnostic and never gate**: it flaps when differently configured
  servers start one after another, while a per-target baseline moves only on
  a successful rebuild.
- **Moving environment settings into the store, key by key**, each one
  declared in the registry and subject to *Standing obligations*.
- **A display-name → UUID mapping for workspaces.** It would reduce the
  drop residue to orphan rows. The name-reuse obligation remains, because
  `LightRAG(workspace=...)` is a library call too.
- **The secrets policy.**

## Acceptance scenarios

Each scenario is a regression test. Test docstrings cite these numbers, so
the numbering is stable: new scenarios are appended, and none is renumbered.

1. All three records match → normal startup.
2. `entities` mismatches → refused **before** any vector storage initializes.
3. `relationships` mismatches → refused, naming `relationships`.
4. `chunks` mismatches → refused, naming `chunks`.
5. Several targets mismatch → one refusal listing all of them.
6. Legacy workspace, first start: each probe succeeds on its own sample and
   each target records `origin=probe`; a target whose probe could not run
   stays absent while the other two are recorded.
7. Entity probe returns negative → refused, and **no** record is written.
8. Rebuilding `chunks` alone updates only `embedding/chunks`.
9. Rebuild succeeds, the configuration write fails → the tool exits non-zero
   and the next startup still refuses.
10. On OpenSearch, the claim flushes and strict-reads back before releasing
    the keyed lock; a retained retryable failure fails the claim, the rebuild
    record and the drop.
11. Two workers of one Gunicorn master claim concurrently → exactly one
    baseline.
12. A tenant called `_lightrag_config` starts normally and shares no file,
    table, collection or index with the container.
13. Any workspace storage fails to drop → all three records are kept.
14. Every drop succeeds → the three records are deleted, and a workspace of
    the same name can be recreated and started.
15. A configuration read fails at transport level → startup fails; it is not
    treated as absent.
16. The inventory pages through configuration rows on each of the five KV
    backends.
17. A strict read fails at step 2 → the configuration storage is fully
    finalized and no other storage was initialized.
18. The coverage gate or the entity probe refuses → `finalize_storages()`
    releases the configuration storage **and** every business storage.
19. A claim, read-back or configuration flush fails → a second
    `initialize_storages()` fails again instead of early-returning.
20. A normal shutdown finalizes the configuration storage together with the
    others, exactly once.
21. Concurrent claims are covered only for workers of one Gunicorn master;
    nothing asserts anything about two independent masters on one workspace.
22. A deployment whose baselines are recorded has every one read by the next
    start, unchanged; a mismatching model still refuses; the same deployment
    with `config_dir` pointed elsewhere reads absence.
23. A configuration backend outside the four is refused at construction by
    name — a vector storage class, and `RedisKVStorage` both named and
    inherited from `kv_storage`.
24. A start with no baseline on record announces the container it looked in;
    a start with records, or missing only some, stays quiet.
25. Two deployments sharing one container hold disjoint rows, per backend,
    with keys that differ by scope.
26. The single-server claim is taken on `config_dir`, not `working_dir`, and
    only when the configuration storage is file-backed.
27. A storage in step 4 raises (first, middle, last member of the loop): the
    configuration storage, the storage that raised and every storage before
    it are released exactly once; storages never reached are untouched; the
    original exception propagates.
28. A first start with no anchor creates the identity, then the anchor, on
    each of the four backends; an inherited Redis selection is refused by
    name.
29. A deployment with configuration rows but no identity binds by creating
    it; every baseline row is read unchanged.
30. An anchor naming another backend type is refused at step 0c: nothing
    initialized, nothing written, the anchor untouched, and not sticky; an
    explicit `LIGHTRAG_CONFIG_STORAGE` naming the anchored backend passes.
31. Same type and UUID passes; a different UUID, a missing UUID, an invalid
    row or a read error each refuse and create nothing.
32. A no-anchor bind whose identity read fails at transport level creates
    neither a UUID nor an anchor.
33. A Gunicorn master refuses a type mismatch before forking.
34. A changed `config_dir` still reads the fixed anchor and is judged by the
    target's UUID; a moved directory that keeps its identity passes.
35. An anchor that cannot be read is never absent; a failed write, publish or
    directory fsync is loud; a start never overwrites an anchor.
36. A crash or cancellation between the identity write and the anchor
    publish heals by adoption; a 1b failure is sticky and releases every
    resource and the anchor lock.
37. Concurrent binds in one process tree create exactly one identity and
    publish the anchor once; two process trees on one `working_dir` binding
    at once do too (one creates, the other waits on the bind lock and
    verifies); an anchored start never waits on the bind lock; a bind lock
    held past its timeout refuses with nothing written.
38. Deleting the anchor rebinds with a WARNING naming the container and the
    UUID; each refusal carries its per-cause advice.
39. Starters share the anchor lock; an exclusive hold refuses them and any
    starter refuses the exclusive hold; on a filesystem that cannot lock,
    starters warn and proceed and the exclusive hold needs
    `--assume-exclusive`.
40. A workspace clear never deletes the identity; the rebuild and clear tools
    refuse on an identity or type mismatch before any data storage opens,
    and never create an anchor.
41. The identity's strict read, flush and read-back run on all four
    backends, and an OpenSearch write still retained in the buffer binds
    nothing.
42. A migration copies every workspace's rows and the server-scope rows,
    keeps the UUID and the source, and moves the anchor last; each refusal
    (same-type or unadmitted target, no anchor, a source that is not the
    anchored container, a malformed row, a foreign or non-empty target)
    writes nothing.
43. A pagination, write, flush or verification failure, or a failed anchor
    replace whose anchor reads back, leaves the anchor on the source; the
    re-run resumes and converges a target whose source changed between
    attempts.
44. The migration is refused while any starter holds the anchor lock, and
    without `--assume-exclusive` where locking is unavailable; conflicting
    env files and a different `WORKING_DIR` are refused.
45. The wizard calls an anchor readable only when the server's parser accepts
    it, reads the backend a repeated key leaves to the server, reports on the
    directory of the runtime target any of its flows is switching to, reads
    an empty `WORKING_DIR` as the start directory, and warns instead of
    guessing for a `WORKING_DIR` that uses `${...}`; a path it cannot search
    is unreadable, not absent; a Compose deployment's migration advice
    names its `./data/rag_storage`; operator edits to the compose file are
    kept by regeneration and not interpreted, and so are hand-written dotenv
    forms other than `KEY=value`. `lightrag-migrate-config`
    resolves an empty
    `WORKING_DIR` as the start directory, as the server does.

## History

Built by [#4006](https://github.com/HKUDS/LightRAG/issues/4006) (the
facility), [#4020](https://github.com/HKUDS/LightRAG/issues/4020) (its own
category, fixed containers) and
[#4059](https://github.com/HKUDS/LightRAG/issues/4059) (identity, anchor and
migration). It builds on the embedding-space work of
[#3978](https://github.com/HKUDS/LightRAG/issues/3978).

This file replaces `docs/design/ConfigurationStorage.md`, the design guide
written before implementation. Everything that guide asked for is either
implemented and stated above, or listed under *Not implemented yet*.
