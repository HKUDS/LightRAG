# Configuration storage contract

Status: **slices 1 and 1b implemented** — slice 1
([#4006](https://github.com/HKUDS/LightRAG/issues/4006)) built the facility
(`lightrag/config_store.py`, the `config` KV namespace, the nine-step startup in
`LightRAG.initialize_storages()`, the rebuild and drop commit protocols, the
enumeration surface `BaseKVStorage.iter_rows()`); slice 1b
([#4020](https://github.com/HKUDS/LightRAG/issues/4020)) made configuration its
own storage **category** and retired the reserved workspace name it used to
live under. Slice 2 and everything under *later* in the rollout table are still
planned. The acceptance scenarios are regression tests under
`tests/config_store/`, with the per-backend enumeration tests beside each
backend under `tests/kg/`.

It builds on the embedding-space work in
[#3978](https://github.com/HKUDS/LightRAG/issues/3978): the per-container
markers, the coverage gate and the homogeneous adoption probe this document
treats as existing landed in
[#4007](https://github.com/HKUDS/LightRAG/pull/4007). Read
`VectorSpaceProvenance.md` alongside this — it holds the verdict rules this
document reuses, and the four blind spots this facility exists to close.

Read this before adding a key, before binding a storage to the configuration
workspace, and before moving anything out of an environment variable.

## Why this exists

Two needs meet in the same place.

**Configuration must be reachable independently of the active workspace.** Every
storage LightRAG opens today is bound to `self.workspace` — PostgreSQL by a
`workspace` column, Redis, MongoDB and OpenSearch by a name prefix, the JSON
backends by a subdirectory under `working_dir`. That is right for
knowledge-base data and wrong for configuration: the server reads its own
settings while switching between workspaces, and reads them before any
workspace is open. Configuration is the server's property, not a tenant's.

**A container name is an assertion, not a record.**
`BaseVectorStorage._generate_collection_suffix()` builds `{folded_model}_{dim}d`
from the *current* configuration, so it can never disagree with the current
configuration. Milvus, Qdrant and PostgreSQL name their vector containers that
way and therefore cannot tell whether an existing container's content came from
the model configured right now. `VectorSpaceProvenance.md` records the four
cases that get through; closing them needs an embedding space that is
*recorded*, per workspace, outside the vector containers.

One durable place answers both: a record of what a deployment is configured
with, kept apart from the data that configuration produced.

## The container

| | |
| --- | --- |
| KV namespace | `config` |
| Selected by | `config_storage` / `LIGHTRAG_CONFIG_STORAGE` — its own category |
| `JsonKVStorage` | `config_dir/kv_store_config.json`, default `<working_dir>/_lightrag_config` |
| `PGKVStorage` | table `LIGHTRAG_CONFIG (workspace, id, value JSONB, create_time, update_time)`, partition constant `_lightrag_config` |
| `MongoKVStorage` | collection `_lightrag_config_config` |
| `OpenSearchKVStorage` | index `x_lightrag_config_config` (the backend's own sanitizer prepends `x`) |

The namespace is a KV namespace on purpose: **KV container names carry no model
suffix**, so a record kept here does not move when the embedding model changes.
That is exactly the property the vector container name lacks, and it is the
reason this record can be evidence where the name cannot.

One fixed container holds every row — the server's own configuration *and* every
workspace's. Configuration does not follow the knowledge base it configures.

**No workspace addresses it.** Every name above is written in code. The four
backends key on the `config` NAMESPACE, which nothing else is ever opened on,
and `CONFIG_CONTAINER_TAG` is the constant they compose it from — it is not a
workspace, nothing validates it as one, and no `*_WORKSPACE` variable reaches
it.

### The category, and the four it admits

Configuration is selected independently of the four business storages:

| backend | why |
| --- | --- |
| `JsonKVStorage` | the default, and the only file-backed member |
| `MongoKVStorage` | fixed collection |
| `PGKVStorage` | fixed table, transactional, trivially backed up |
| `OpenSearchKVStorage` | fixed index |

`RedisKVStorage` is **deliberately excluded**: Redis is being retired from
business storage, and the configuration path must not be the reason its
enumeration surface is kept alive. **No vector storage may serve here**, and
that is a rule rather than an accident of today's registry — a vector
container's name is derived from the embedding configuration
(`_generate_collection_suffix()` builds `{folded_model}_{dim}d`), which is
exactly the assertion the baselines exist to be independent of, and
configuration has to be readable *before* any vector storage initializes
(step 3 of the startup sequence).

A selection outside the four is refused at construction, **by name**, rather
than failing later on a missing method. The setup wizard follows the same
rule at both ends: `make env-storage` asks for a configuration backend
whenever the KV selection is not one of the four and collects that backend's
database requirements, and `make env-validate` refuses an `.env` whose
configuration backend — explicit or inherited — is outside them, because
validation that approves a file the server then refuses is worse than no
validation.

An **explicit** `LIGHTRAG_CONFIG_STORAGE` already in an `.env` is never
dropped by a later wizard run, even when the KV backend would be admitted.
Dropping it moves the container to another backend without migrating the rows,
and they then read as absent — the same silent bootstrap the `config_dir`
default exists to prevent, and it would additionally let the wizard's marker
cleanup tear down the managed service that backend runs on. Moving
configuration is an explicit edit plus a rebuild, never a side effect.

**Unset, the selection follows `kv_storage`.** That is the only default that
does not orphan an existing deployment's records: they are already in that
backend's container. A `kv_storage` the category does not admit is refused with
a message that names it and says the records it holds are not migrated.

Two combinations this buys that were impossible before: business data on
Milvus/Qdrant with configuration on PostgreSQL, and file-backed business data
with configuration on a server backend.

### `config_dir`, and why its default is load bearing

The JSON backend puts its file in `config_dir`, not under
`working_dir/<workspace>/`. `config_dir` defaults to
`<working_dir>/_lightrag_config` — **exactly** where the reserved workspace put
it before — and that is the whole reason for the spelling. Point the default
anywhere new and every baseline of an existing deployment reads as *absent* on
the first start after an upgrade, which is the one answer that lets a start
bootstrap; the deployment would then record the currently configured model over
vectors nobody probed, with nothing in any log to say so. The migration is
therefore the absence of one, pinned by a test that writes the old layout by
hand and asserts the recorded model comes back unchanged.

### What retiring the reserved name retired

Slice 1 bought "configuration is not a tenant" with a reserved *name*, and a
reserved name has to be defended everywhere a name can be chosen. It needed all
of:

- `validate_workspace()` refusing the whole `_lightrag*` family
  **case-insensitively**, because OpenSearch lowercases index names, so
  `_LightRAG_config` and `_lightrag_config` are one index there;
- a context-variable grant spanning exactly one construction, so the factory
  could bind the name every backend's `__post_init__` otherwise rejects;
- `validate_workspace_override()` refusing the family at six backends, so no
  `*_WORKSPACE` variable could point tenant data *into* it;
- a standing rule that no public `allow_reserved`-style flag may ever be added.

**None of that exists any more.** Nobody can stray into the configuration
container because it is not reachable by naming a workspace at all, and no
override can redirect into it because there is no name to collide with. A
tenant may now legally be called `_lightrag_config`: on the JSON backend that
is a different *file* in the same directory, on PostgreSQL a different *table*,
on MongoDB and OpenSearch a different *collection* and *index*. Nothing is
shared, so nothing has to be defended.

What stays, and why:

- **`create_configuration_storage()` is still the single door in.** It is where
  the namespace, the container tag and the post-construction check that no
  backend re-bound the container live.
- **`validate_workspace_override()` stays**, with its reserved check removed and
  a real one in its place: a `*_WORKSPACE` value is applied *inside* a backend's
  constructor, after `validate_workspace()` has already passed the constructor
  argument, so this is the only place that value is ever checked at all. It now
  holds the override to the same rules as a constructor argument.
- **The `config` namespace is the marker.** Each of the four branches on it
  rather than on a name. Nothing else is ever opened on that namespace, and a
  caller cannot ask for it: namespaces are fixed by LightRAG's own construction.

#### `*_WORKSPACE` is legacy compatibility, and baselines do not follow it

A baseline is keyed by the workspace the CALLER named, and each backend applies
its own `*_WORKSPACE` override inside its constructor, below the layer that
computes that key. So when an override is in effect the record and the
container it describes can sit under different names — and the override
deliberately **collapses distinct logical workspaces onto one physical
container** (`MilvusVectorDBStorage` says so in its own comments), so two
instances on different models can each hold a matching baseline over the same
container.

This is **not** re-keyed by the effective workspace, and the reason is what the
variables are for: they exist to keep LEGACY data reachable, and their presence
is meant to be transparent to everything above the storage layer. Keying
configuration by them would make that presence visible in the record format and
would bless what is actually the unsupported act — using one to MOVE data.
Setting, changing or clearing an override on a deployment that already has data
points the instance at another container while everything keyed by the caller's
workspace stays behind, and no check below can tell that apart from an ordinary
start.

So the rule is announced rather than enforced:
`warn_about_workspace_overrides()` names every override in effect, states that
they are deprecated, that a storage's workspace should follow the server's, and
that switching data location by editing one is unsupported. It is called from
the **application's** startup and not from `LightRAG` — the uvicorn entry point
(`lightrag_server.main`) and the Gunicorn master's `on_starting` hook, which
runs before it forks — so an operator hears it once per server start rather
than once per worker or once per instance. Recovery, if one was
moved: point the override back, or rebuild the moved target with
`lightrag-rebuild-vdb`, which re-embeds from the authoritative sources and
records the baseline afresh.

**The configuration container never reaches any of this.** Its name is written
in code, so no override applies to it, and the announcement above is about
tenant data only.

PostgreSQL is the only backend needing schema work: `NAMESPACE_TABLE_MAP` gains
an entry, `TABLES` gains the DDL, and `PGKVStorage` needs the SQL templates it
dispatches per namespace (`get_by_id_config`, `get_by_ids_config`,
`upsert_config`). See *Enumeration* for the one thing the other three do owe.

#### The `workspace` column on `LIGHTRAG_CONFIG` is not a workspace

The table keeps the shape every other one has — `(workspace, id)` primary key —
and writes the container tag into that column as a **partition constant**. It
was never a tenant's name even in slice 1; the workspace a row is *about* is a
field inside the payload, and `id` carries it too. Keeping the column means an
existing deployment's rows are read where they already are, which is the same
choice the `config_dir` default makes for the same reason. Dropping it would be
a DDL migration bought for cosmetics.

#### What discriminates two deployments on one server

A fixed container name is, by construction, the **same name for everyone**: two
independent deployments pointed at one PostgreSQL, MongoDB or OpenSearch land
in the same table, collection or index. What keeps their baselines apart is the
row **key**, whose scope is the business workspace the row is about — and the
contract already requires different LightRAG instances, and different server
instances, to use different business workspaces. That was already true in slice
1, where the container's workspace was a constant shared by every deployment
too; nothing about this change makes it more or less true.

Two deployments that share a server backend **and** a business workspace name
do overwrite each other's baselines. That is the same unsupported case as two
servers sharing a `working_dir`, one layer up, and it is listed under *Accepted
residues*. It is not closed by a deployment id: an id an operator can set is an
id an operator can copy into a cloned deployment, and one derived from the
environment moves when the container does — either way the records would read
as *absent* after an ordinary redeploy, which is the failure this whole
facility exists to prevent. The requirement that business workspaces differ is
the cheaper and already-load-bearing rule.

### One server at a time on a file-backed configuration

The in-process guard (*One file per namespace per process tree* in
`FileBackedSnapshotContract.md`) cannot see another SERVER. Each process tree
has its own in-memory copy, so two of them started on one `working_dir` each
load the configuration file, each accumulate a private view, and each rewrite
the whole thing — the later flush dropping whatever the other recorded since.

That is fatal here in a way it is not elsewhere: an overwritten baseline reads
back as **absent**, and absent is the one answer that lets a start bootstrap.
The next start does not refuse the model change the baseline existed to refuse;
it records the configured model over vectors nobody probed, and the protection
is gone with nothing in any log.

So a file-backed configuration storage **claims its `config_dir`** for the life
of its process tree (`lightrag/kg/working_dir_lock.py`), and a second tree is
refused with `WorkingDirectoryInUseError`. With `config_dir` at its default the
two name the same deployment either way; where it is set, the claim follows the
file it exists to protect. Five properties matter:

- **An OS lock, not a PID file.** The kernel releases it when the holder dies,
  so a `SIGKILL`, an OOM kill or a power cut leaves nothing stale to reap and
  there is no read-PID-then-probe-liveness race.
- **`fork` shares it.** The Gunicorn master takes it in `on_starting`, *before*
  forking, and the workers inherit that claim and count themselves in. Taken
  after the fork, each worker would open its own descriptor and all but one
  would be refused. The master must therefore claim **the directory its
  workers will ask for**: it resolves `config_storage` and `config_dir`
  through the same `configuration_selection_from_env()` the workers reach via
  `LightRAG`, and `lightrag-rebuild-vdb` uses it too. A master reading the
  environment its own way is the same failure as taking the claim late —
  nothing inheritable, and every worker after the first refused at startup.
- **It fails open.** Locking is unreliable on NFSv3 without lockd and on
  SMB/CIFS, and a `working_dir` on a network volume is ordinary in container
  deployments. A backend that cannot lock gets a warning and proceeds; refusing
  would break deployments that work, to protect against a rarer failure.
- **It is asked of the configuration storage**, not of the four business ones.
  A deployment that moves configuration to a server backend stops claiming
  anything at all, and one that keeps JSON configuration claims the directory
  that file is in.
- **`lightrag-rebuild-vdb` takes it too**, around its own configuration-storage
  lifecycle. The tool is a second process tree by construction, and the record
  it writes to say the rebuild happened is exactly the one a running server
  would overwrite. The confirmation prompt is not a substitute: it asks the
  operator, the claim asks the filesystem.

#### The claim goes back last, and only after the teardown

Handing the claim back is not the only thing shutdown owes: handing it back
**while this process is still up** is the exact combination the claim exists to
prevent. The next server opens the same files while this one still holds the
shared-namespace holds and whatever it has not flushed, and the two rewrite
each other's namespaces — the failure the claim was added to close, now caused
by the release rather than by its absence.

So `finalize_storages()` orders its own cancellation:

- **the queue drains stay interruptible.** `_shutdown_model_queues()` waits on
  whatever is in flight, so a shutdown timeout escalating to a cancel lands
  there more often than anywhere else, and a wedged queue must not wedge the
  shutdown. The cancellation is **absorbed**, not propagated;
- **the storage teardown does not.** Once it has begun it runs as a shielded
  task that is drained to completion, so no cancellation can leave it half-done
  — every await inside it, the cache-pair commit included, would otherwise be
  its own exit;
- **the claim is released after that task has completed**, in a `finally`
  because it is the one release nothing else can perform: a retry returns early
  on the status, so a claim left held by a process on its way out refuses the
  next server for nothing;
- **the cancellation is re-raised** once all of that is done, so the caller
  still sees a cancelled shutdown.

The startup rollback owes the same ordering for the same reason, and pays it
differently: it shields each release so a cancel cannot stop it moving to the
next storage. But `asyncio.shield` only keeps a release **alive** — awaiting a
shielded task returns the moment the awaiting task is cancelled, which
*detaches* the release rather than finishing it. A detached release races the
event loop's own shutdown and loses exactly what it was called to hand back. So
every release a cancellation detached is **drained before the claim goes back**.

**Accepted residue.** A deployment whose configuration is on a server backend
(or in a `config_dir` of its own) but whose business data is file-backed is
*not* protected: two servers there
still overwrite each other's `full_docs`, `doc_status`, graph and vectors, and
lose more than baselines doing it. That is the long-standing "separate process
trees are unsupported" position, unchanged. This claim narrows the blast radius
rather than closing it, because the baseline is the case whose failure is
silent. Recovery is unchanged — one server per directory, or server backends —
and widening the claim to any file-backed storage is a deliberate follow-up,
since it would refuse deployments that work today.

## Keys

```
<workspace>/<key>            a per-workspace setting
_lightrag_server/<key>       a server-global setting
```

**The separator is `/`, never `.`** — this is load-bearing.
`validate_workspace()` rejects only `/`, `\`, `.` and `..`, and its docstring
names `"v1.0"` as a legitimate workspace. A dotted key `v1.0.embedding` cannot
be reparsed: splitting on the first dot and splitting on the last dot give
different answers. `/` cannot appear in a workspace name at all.

**Keys are never reparsed anyway.** The row carries `workspace` as an explicit
field and enumeration reads the field. The separator rule is the second lock on
a door the row schema already closes; both stay, because reparsing a key is the
kind of shortcut that gets reintroduced by a later patch.

**The scope is the container's only discriminator.** The container name is
fixed, so two deployments sharing one server backend share it; what keeps their
rows disjoint is this scope. See *What discriminates two deployments on one
server*.

## Row shape

Uniform, so this namespace stays a configuration store rather than a place to
drop keys:

```json
{
  "schema_version": 1,
  "workspace": "<workspace or _lightrag_server>",
  "updated_at": "<iso8601>",
  "updated_by": "<component that wrote it>",
  "value": { }
}
```

`schema_version` is per key, not global: keys evolve independently.

## Key registry

One module owns the list of keys. Each entry declares:

| field | why |
| --- | --- |
| key suffix | the name after the scope |
| scope | per-workspace or server-global |
| schema | what `value` holds, and its current `schema_version` |
| readers / writers | who may read, who may write |
| sensitive | whether the value must never be echoed |

A general namespace without a registry becomes a junk drawer. The `sensitive`
flag exists from the first entry even though nothing sensitive is stored yet —
adding the column later means auditing every key that already exists.

## The first keys: one baseline per vector target

**Three keys, from the first slice. Not one.**

```
<workspace>/embedding/entities
<workspace>/embedding/relationships
<workspace>/embedding/chunks
```

`value` is `{model, dim, origin}`. The model name is stored **unfolded**, via
`lightrag.kg.vector_space.declared_model_name()`, for the same reason the
per-container marker stores it unfolded.

One record per workspace would be wrong, and could not be split later.
`lightrag-rebuild-vdb` rebuilds `entities`, `relationships` and `chunks` as
three separate steps against three separate containers
(`RebuildTool.vector_targets()` in `rebuild_vdb.py`), so after an interrupted or deliberately partial
rebuild the three legitimately sit in different embedding spaces. A single
record cannot be advanced by a partial rebuild without either lying about the
targets that were not rebuilt or refusing to record the one that was.

### What the record means

Not *the space these vectors were written in* — the store cannot know that for a
container it has never probed. It is:

> **the embedding space adopted as this target's active baseline.**

Everything below follows from that being an adopted claim rather than an
observation.

`origin` records how the claim was established. It is **diagnostic only and
never participates in a verdict**:

| `origin` | how the baseline was established |
| --- | --- |
| `probe` | the homogeneous cosine probe reproduced stored vectors under this model |
| `empty` | both the source and the index were empty, so there was nothing to contradict |
| `rebuild` | written by a successful `lightrag-rebuild-vdb` of this target |

### Verdicts

Each of the three is compared independently against the configured embedding
function:

| record | verdict |
| --- | --- |
| present, `model` or `dim` differs | **refuse to start**, naming *which target* and both spaces, pointing at `lightrag-rebuild-vdb` |
| present, equal | proceed |
| absent (confirmed) | remember it as a bootstrap target; establish it after the storages are up |
| unreadable | **startup failure** — see *Reads are strict* |

When more than one target mismatches, the refusal names **all** of them in one
message. An operator planning a rebuild needs the whole list, not the first one.

### Establishing a baseline: every target on its own evidence

Each target has a probe, and each probe samples from that target's own source:

| target | sample |
| --- | --- |
| `entities` | the graph's most-connected labels (`get_popular_labels`), mapped to entity vector ids |
| `relationships` | the first batch of `iter_edges`, mapped to **both** candidate relation vector ids (`make_relation_vdb_ids`: the canonical one, then the legacy reverse-order one a historical custom-KG import may have hashed under) -- an all-legacy store sampled by the canonical id alone would never be examined |
| `chunks` | the first page of `text_chunks.iter_rows()` -- one bounded round trip, not a scan; the row id is the chunk vector id |

The chunk probe is what the enumeration surface makes possible. Before
`BaseKVStorage.iter_rows()` existed the only KV enumeration was
`rebuild_vdb.enumerate_kv_keys()`, a backend-specific full scan that has no
place on a startup path, and `doc_status` could not stand in for it
(`ainsert_custom_kg` writes chunks and no doc-status row). A first page of a
paged reader is a different thing from a scan: it costs one round trip whatever
the namespace holds.

**The source verdict must come from a read that raises when it fails.** A
verdict of "empty" is now a durable write, not merely a skipped check, so an
outage reported as emptiness would stamp the configured model over vectors
nobody probed -- and the next start, finding a record, would never probe them.
`BaseKVStorage.is_empty()` catches its errors and answers `True` on the four
server backends, so the chunk source is read through the first page of
`iter_rows()` instead (the same bounded read the chunk probe samples from),
which the base contract requires to raise on failure. A KV backend without
enumeration falls back to `is_empty()` and its "empty" is read as *unknown*:
the coverage check it had is unchanged, and no baseline is recorded on it. The
graph readers behind the other two verdicts propagate their failures already.

**Only a target with a baseline to establish pays for that read.** The strict
read, the probe and the container's own `is_empty()` are all keyed on the same
list -- `baseline_targets`, the targets the precheck found absent. A target
whose baseline is already recorded claims nothing from this start, so its
source is read exactly as the coverage check has always read it
(`is_empty()`), and nothing enumerates. This is not an optimization detail:
the enumeration contract says a startup path must never scan a namespace, and
`JsonKVStorage` -- whose rows live in a `Manager().dict()` -- snapshots its key
list before it can yield a first page. Charging that to every start would
contradict the contract on the one backend that cannot page lazily; charging
it to the single start that claims the baseline does not.

So, per target and independently:

| source empty | source populated |
| --- | --- |
| record now, `origin=empty` -- **only if that target's vector container is empty too** (fail-loud `is_empty()`). Surviving vectors behind an empty source: nothing can vouch for them and there is nothing to sample, so leave absent, warn, and point at the rebuild; container unreadable, or source unreadable: leave absent | run that target's probe. Negative → **refuse**, and write nothing. Positive → record, `origin=probe`. Could not run (embedder down, timeout, unreadable source or index, no sampleable row) → leave absent and retry next start |

**A verdict is about one container only.** The three targets share an
`embedding_func` but not a history: `lightrag-rebuild-vdb` rebuilds them as
three separate steps, so an interrupted rebuild after a same-dimension model
change can leave `entities` in the current space while `relationships` or
`chunks` still hold the previous model's vectors. Nothing is recorded, and no
container marker is adopted, on a sibling's verdict. An earlier revision of
this document trusted `relationships` and `chunks` on first use because they
had no probe; that trade-off no longer exists and is gone.

`lightrag-rebuild-vdb` rewrites a target's record after rebuilding it. Without
that, a deliberate model change would have no way through, and a gate with no
way through is a gate operators disable.

## Startup sequence

The precheck and the adoption are **two different phases**, and only the first
can precede vector initialization. An earlier draft of this document said the
whole check runs before the vector storages initialize; that is impossible — the
entity probe reads `entities_vdb`.

```
1.  initialize the configuration storage
2.  strict-read the three records for this workspace
3.  PRECHECK, on records that exist:
      any mismatch  -> refuse; no vector storage is initialized
      all equal     -> continue
      absent        -> remember as a bootstrap target
4.  initialize the remaining KV, graph, doc-status and the three vector storages
5.  mark INITIALIZED
6.  run the existing coverage gate and the entity adoption probe
7.  establish the baselines remembered in step 3
8.  flush the configuration storage -- through the same flush the claims
    use, never `index_done_callback()` directly, so the rules below apply
    to it too
9.  return successfully
```

**Only step 3 must precede step 4.** That is what puts the mismatch refusal
ahead of the legacy-container migration on Milvus, Qdrant and PostgreSQL, which
runs inside `initialize()`.

When a record is **absent**, the precheck has nothing to compare and step 4 runs
first — so that migration may copy rows into a `{model}_{dim}d` container before
anything has judged them. The direction is still safe, because a subsequent
negative probe refuses to serve; but the copy has happened. That is an accepted
residue, recorded below. This document does **not** claim that every legacy
relabel is prevented before it occurs.

### `INITIALIZED` means the resources are up, not that the service may serve

Step 5 sits where it does deliberately, and moving it to the end would break a
contract the existing code states in a comment: every storage above it holds
clients, pools and locks, and `finalize_storages()` skips the whole teardown
unless the status says `INITIALIZED` (`LightRAG.finalize_storages`). Marking it last would
mean that a refusal from the coverage gate, a failed probe, a failed claim or a
failed flush leaves every storage up with the status still `CREATED`, and the
caller's `finalize_storages()` silently releases nothing.

So the two ideas are kept apart:

| | meaning |
| --- | --- |
| `INITIALIZED` | the resources exist and `finalize_storages()` must release them |
| a successful return from `initialize_storages()` | the checks passed and the instance may serve |

Consequences the implementation owes:

- **The configuration storage is an ordinary member of the teardown list.** It is
  initialized first and finalized with the rest, exactly once.
- **Everything after step 5 is sticky.** A post-`INITIALIZED` failure — coverage
  gate, probe, claim, read-back, flush — must be retained and re-raised by the
  next `initialize_storages()` call, which would otherwise take the
  `status != CREATED` early return and come back successful without re-running
  anything. The merged code already does this for the embedding-space verdict
  (`LightRAG._startup_refusal`): a stored refusal is re-raised at the top of the
  method, and its comment gives the reason — "turning a fail-closed gate into a
  one-shot one". Every failure introduced here joins that mechanism rather than
  inventing a second one.
- **Steps 1-4 run outside it**, and need explicit cleanup instead (below).

### Cleanup before `INITIALIZED` exists

Everything up to and including step 4 runs while the status is still `CREATED`,
so none of it is covered by the teardown list. Three failure points live there,
and all three must release what they opened:

**Step 2, a strict read that could not complete**, and **step 3, a precheck
refusal.** The configuration storage is up and nothing else is. Both paths close
it explicitly (client, pool, refcount) before raising. A refusal that leaks a
connection pool turns a safety feature into an operational one.

**Step 4, a storage that fails partway through the loop.** This is the case an
earlier revision of this document got wrong by claiming only steps 1-3 were
exposed. The loop initializes twelve storages one at a time; if the eighth
raises, seven are up, the configuration storage is up, the status is still
`CREATED`, and the caller's `finalize_storages()` releases none of them.

So step 4 owns a rollback protocol:

- track every storage the loop has **started**, in order;
- on a failure, best-effort finalize in reverse order — the storage that raised
  (it may have allocated before raising), then the ones that succeeded, then the
  configuration storage;
- a teardown failure is logged and never replaces the original exception, which
  is what propagates;
- a storage the loop never reached is left alone: no backend acquires anything
  before its `initialize()` runs, so it is holding nothing. See *Construction
  takes nothing* below.

**Do not simply move `INITIALIZED` ahead of the loop instead.** Teardown would
then call `finalize()` on storages that never ran `initialize()`, and that is
not a contract the backends currently offer: `PGKVStorage.finalize` and
`MongoKVStorage.finalize` guard on `self.db is not None` and are safe, Redis
documents `close()` as idempotent — but `OpenSearchKVStorage.finalize` awaits
`_flush_pending_kv_ops()` *before* its `self.client is not None` guard, so it
does not obviously tolerate an instance that was never initialized. Introducing
an `INITIALIZING` status that teardown understands is a legitimate alternative,
but it needs that per-backend contract established first and stated here; the
tracked-list rollback needs nothing new from any backend.

Partial-initialization leakage predates this design — the loop has always been
able to fail midway. What this slice adds is the configuration storage in the
same chain, and it must not be the reason the gap goes on being undocumented.

#### Construction takes nothing

**No backend acquires a process-wide resource in its constructor**, so a
storage the loop never reached is holding nothing and the rollback owes it
nothing. That invariant is load-bearing here and easy to break silently: a
constructor has no teardown path, `LightRAG.__post_init__` is synchronous and
builds twelve storages before validating anything, and a refusal anywhere in
that sequence drops every storage already built on the floor — no `finalize()`
is reachable, and there is no half-built `LightRAG` to call
`finalize_storages()` on. The Redis backends held their shared-pool reference
that way until [#4017](https://github.com/HKUDS/LightRAG/pull/4017) moved the
acquisition into `initialize()`, where every other backend already did it, and
gated `close()` on a reference actually held. A new backend that acquires in
`__post_init__` reopens the leak, and the rollback cannot cover it: `finalize()`
on an instance that never initialized is not a contract any backend offers —
the same argument that keeps `INITIALIZED` behind the loop.

One ordering survives from when this was not true: the configuration storage is
constructed as the **last statement of `__post_init__` that can raise**, after
every business storage and after the validations that follow them
(`llm_model_func` present, `role_llm_configs` well-formed). It is no longer
load-bearing — there is nothing of the configuration storage's to leak — but it
is kept, because building something a refusal is about to discard is pointless
either way. A new check added to `__post_init__` goes **above** that
construction.

**A failed step 4 does not heal by retrying.** Either the failure is retained
the way post-`INITIALIZED` failures are, or a full retry is supported and
re-runs every step from 1; what is not acceptable is a second
`initialize_storages()` returning successfully because some state was left
behind by the first. The implementation takes the first option, for every
failure before `INITIALIZED` and not only step 4's: not every backend's
`finalize()` is safe to undo (`OpenSearchKVStorage.finalize()` flushes its
pending buffer ahead of its own client guard), so a retry on the same object
could neither succeed honestly nor re-run from step 1. A new instance is the
retry.

**Where stickiness starts.** Steps 1 through 9 are the guarded phase; the
preamble before them -- binding the event loop, the default workspace,
`pipeline_status` -- is not, and deliberately so. Stickiness exists to stop two
things: a later call early-returning on a status that says `INITIALIZED`, and
re-running steps against storages a rollback has closed. Neither is reachable
before step 1, where nothing has been opened and the status has not moved, so a
failure there is an ordinary failure and the retry re-runs every check for real.
Making it sticky would strand an instance over a transient shared-storage
failure and buy no safety. A regression test pins the boundary rather than
leaving it to be re-derived.

**Cancellation is a failure too.** `asyncio.CancelledError` is not an
`Exception`, and a handler that catches only `Exception` after `INITIALIZED`
lets a cancelled probe, claim or flush leave the status `INITIALIZED` with
nothing retained -- the next call early-returns as ready with the checks never
completed. Every failure is retained, `BaseException` included; an interruption
that cannot itself be re-raised later is retained as a `RuntimeError` naming
it.

## Claiming a baseline atomically

`read absent → upsert` is not atomic. Two workers starting together can both
read absent and both write.

Within one process tree the claim is made under a keyed lock, per workspace and
target:

```python
async with get_storage_keyed_lock(
    target_key,
    namespace="configuration_embedding_claim",
):
    record = await config.get_by_id_strict(target_key)
    if record is None:
        await config.upsert({target_key: candidate})
        await config.index_done_callback()      # BEFORE releasing the lock
        record = await config.get_by_id_strict(target_key)
    validate(record)                            # against THIS process's config
```

`index_done_callback()` must complete inside the lock. `OpenSearchKVStorage.upsert`
buffers in process memory and its own docstring says the buffer is
process-local until the flush; releasing the lock first lets another worker read
absent and claim again.

**A flush that retained anything is a failed flush here.** The same backend
keeps per-item *retryable* failures (408 / 429 / 5xx) buffered and returns from
`index_done_callback()` normally — the residue heals on the next flush, which is
fine for the pipeline — and its strict point read answers from that buffer: a
buffered upsert reads as present, a buffered tombstone as gone. Flush then
read-back would therefore confirm a claim, a rebuild record or a drop the server
never saw. So every configuration flush asks
`has_pending_index_ops(include_deletes=True)` afterwards; a retained operation
drops the buffer (what the caller reports is then what is true) and raises
`ConfigurationStorageError`. **Every** flush means every one: the claims, the
rebuild record, the drop, and step 8's final guard, which is why none of them
calls `index_done_callback()` directly. Only after that does the strict read-back confirm
anything, and it is then a read of the server. Backends without a buffer answer
`False` and pay nothing.

**And a flush that RAISED is not automatically a failed one.** The mirror case:
the bulk landed and only `indices.refresh()` afterwards failed. The backend can
prove that raise lost nothing and says so by raising
`ReferencesIntactFlushError` — but that type covers two situations its own
docstring separates, and they need opposite answers: every operation still
buffered (a transport error from the bulk call), or the commit landed and only a
step after it failed. Catching both as a failed flush reports a **durable write
as one that did not happen**, which is the asymmetry *Consistency without
transactions* forbids outright, and it costs a refused startup on a record the
server already has, or a whole re-embed reported as failed by the rebuild tool.

So the same question decides both directions: **ask the buffer, not the return
value.** Nothing retained after a raise means the write landed, and the strict
read-back — which every caller here performs — is what confirms it. Something
retained means the write is not durable, whether the flush returned or raised. A
backend that cannot be asked keeps the conservative reading of its own raise,
because there is then no way to tell the two situations apart.

The strict read-back is not a formality, but its job is narrower than it looks.
It confirms that the write is visible and durable, and it validates whatever is
*actually stored* against this process's configuration rather than against what
this process believed it wrote. Inside the lock's scope that closes the
buffered-write hole above. It does **not** provide exclusion, and the section
below says exactly where that ends.

### What the lock does and does not span

| scope | shared |
| --- | --- |
| coroutines in one process | yes |
| workers forked from one Gunicorn master | yes |
| two independent Gunicorn masters | **no** |
| two containers / pods / hosts | **no** |
| separate SDK processes | **no** |

`shared_storage` locks do not cross a process tree, so the deployment model is
part of this contract rather than an assumption under it. **The atomic claim
covers workers sharing one `shared_storage` instance — one Gunicorn master —
and nothing wider.** Two independent masters, containers, hosts or SDK process
groups must not initialize the same configuration store concurrently, and
overlapping rolling deployments are not supported. That is the same constraint
the embedding-space work already operates under: LightRAG propagates no
configuration between worker processes and supports no rolling update, so a
model change is always stop → `lightrag-rebuild-vdb` → start, one deployment at
a time.

**The read-back does not extend that boundary, and this document previously
claimed it did.** It does not, and the counter-example is simple:

```
master A  read absent
master B  read absent
master A  write A, flush, read back A, validate -> proceeds
master B  write B, flush, read back B, validate -> proceeds
```

Both sides read back their own write and both start; the store ends up holding
B while A is serving on a baseline that is no longer recorded. A read-back can
only report the record that exists at the moment it runs. Exclusion needs
mutual exclusion or a compare-and-set, and neither is present here.

Concurrent initialization by separate process trees is therefore **unsupported,
not a handled residue** — nothing in this slice makes it safe, and it must not
be listed among the states this design accepts. Supporting it later needs one
of: a distributed `shared_storage` lock, or genuine create-if-absent/CAS in each
backend. Both are projects of their own, and neither is required for the
supported deployment shape.

## Reads are strict

These records decide whether to serve, so a read that could not be completed
must never be mistaken for a record that does not exist:

```
confirmed absent  -> bootstrap
read failure      -> startup failure
present, differs  -> vector-space refusal
present, equal    -> proceed
```

Use `get_by_id_strict()`. All five KV backends already declare
`supports_strict_point_reads = True` (a `ClassVar` on each of the five KV
classes), so this
costs no new backend work — but the caller must still check the ClassVar rather
than assume it, since `BaseKVStorage` defaults it to `False`.

A configuration store that was deliberately emptied reads as confirmed absent
and bootstraps again, which is correct. A store that cannot be reached does not.

**Strictness has to survive the layer below the read, too.** On the JSON
backend the file is read once per process tree and shared: the first instance
to ask wins a claim, loads the file into the shared namespace dict, and every
later instance reads that dict instead of the file. The flag recording the
claim says "loaded" from the moment it is taken, so a load that FAILS — a
momentary `PermissionError`, a full disk — used to leave the namespace marked
loaded and EMPTY. The instance that hit the failure refused correctly; the next
one in the same process tree skipped the file, read absence as confirmed
absence, bootstrapped its own model as a first start, and published the whole
namespace over the record that should have refused it, taking every other
workspace's rows in that file with it (a commit publishes the whole namespace).
So a claim is a promise to finish: leaving the load by exception, cancellation
included, hands the claim back and the next instance reads the file again
(`namespace_init_claim` in `lightrag/kg/shared_storage.py`). A transient
failure is therefore recoverable and a persistent one simply fails again,
loudly, in the next claimer.

## Rebuild: one target at a time, configuration last

Per target, in this order:

```
rebuild the target VDB
→ flush / index_done_callback the target VDB
→ verify the rebuild succeeded, the retained buffer included
→ update THAT target's configuration key
→ flush / index_done_callback the configuration storage
```

- rebuilding `entities` updates `embedding/entities` and nothing else; likewise
  for the other two;
- the configuration write happens **after** the target is durable and verified,
  never before;
- **a returning `index_done_callback` is not the verification.** A per-item
  backend keeps its retryable failures (408/429/5xx) buffered and returns
  normally, so the last flush of a rebuild can leave vectors that never reached
  the server with nothing left to retry them. Mid-rebuild that residue heals —
  the next flush retries it — and is accepted; a residue left by the LAST flush
  is not, because the baseline about to be written would claim the target was
  adopted in the configured space while its index is incomplete, and no later
  check catches it: the startup precheck sees a matching record, and the
  coverage gate only refuses an EMPTY index. So the tool asks the vector
  storage directly (`has_pending_index_ops(include_deletes=True)`) before
  recording, and a retained operation — or an answer that could not be read —
  is a failed rebuild. This is the same rule the configuration flush follows
  (*A flush that retained anything is a failed flush*), applied to the data
  side, and it is why `OpenSearchVectorDBStorage` implements that method
  rather than inheriting the base `False`;
- a failed rebuild advances nothing;
- a rebuild that succeeds while the configuration write fails makes the **tool
  exit non-zero**. The stale record keeps refusing startup, which is the safe
  direction; re-running the tool converges;
- targets that were not rebuilt keep their old records, so the next startup
  refuses again and lists exactly what remains.

The tool may bypass the ordinary mismatch refusal for the targets it is
rebuilding — it is the way out of that refusal — but it must not treat the other
targets' mismatches as resolved. Its source stores (the graph and `text_chunks`)
open through the ordinary strict path.

## Workspace drop: data first, configuration last

```
stop writes to the workspace
→ drop every data storage for that workspace
→ confirm every drop succeeded
→ delete the three configuration records
→ flush the configuration storage
```

If **any** data storage fails to drop, all three records stay. The two possible
residues are not symmetric:

| residue | consequence |
| --- | --- |
| data gone, configuration remains | a workspace recreated under the same name may be refused until the stale rows are cleaned. Recoverable, loud. **Accepted.** |
| configuration gone, data remains | the next startup treats surviving vectors as a first bootstrap and may adopt a wrong baseline over them. **Never acceptable.** |

That asymmetry is the whole reason for the ordering, and it is why a partial
drop must not opportunistically delete "the records for the parts that did
drop".

**A cancelled drop is a drop that did not happen.** The endpoint runs the drops
through `asyncio.gather(..., return_exceptions=True)`, which hands a cancelled
child back as a `CancelledError` **object in the results list** — and that
inherits from `BaseException`, not `Exception`. Classified on `Exception` alone
it reads as a success, "every drop landed" becomes true, and the records go with
the second residue above. The results are therefore classified on
`BaseException`.

"Every data storage" includes the opt-in LLM cache drop that `/documents/clear`
runs after the storage drops: the records are deleted after it, and a failed
cache drop keeps them exactly as a failed storage drop does. The cache rows
that survive are workspace data too, and the endpoint's history entry names
which drop kept the records.

Workspace names becoming UUIDs later reduces the accepted residue to orphan rows
and a misleading inventory rather than a wrong refusal, but does not remove the
obligation: `LightRAG(workspace=...)` is a library call too, and the
never-reuse-a-name guarantee would be a property of the server's creation path
only.

## Enumeration, and what the inventory really costs

The second slice reports which workspaces still need a rebuild, and listing the
configuration container is that inventory — but `BaseKVStorage` has **no
enumeration API**, so adding a namespace does not make the listing free.
PostgreSQL can answer it with a query; Redis, MongoDB, OpenSearch and the JSON
backends each need their own traversal.

The first slice therefore also defines the enumeration surface the second slice
consumes:

- a paged or streaming read over the configuration namespace — never "load every
  row into memory", since it grows with the number of workspaces times the
  number of keys;
- classification by the row's explicit `workspace` field, never by reparsing the
  key;
- per-backend implementations counted as real work, not as a free consequence of
  the namespace;
- **a container that cannot be read raises; it never ends the stream.** A
  missing index, a dropped collection or a closed connection is not an empty
  listing, and the callers cannot tell the two apart from a clean end: the
  inventory would under-report, and the chunk source verdict would turn a lost
  container into a durable `origin=empty` baseline. `OpenSearchKVStorage`
  refuses in the same index-missing states its `get_by_id_strict` refuses in
  (index not ready, index gone mid-scan), for the same reason: after `initialize()` the
  index exists, so its absence is indistinguishable from data loss.

Whether the five implementations land in slice 1 or slice 2 is a scheduling
choice; what is not a choice is pretending PostgreSQL is the only backend with
work to do.

Only **four** of those five can serve as the configuration storage today; the
fifth, `RedisKVStorage`, keeps `iter_rows` for business KV until Redis is
removed, and the configuration path is not the reason it stays.

Startup uses the same surface, and stays inside the same rule, by bounding both
*what* it reads and *when*: never more than the first page, and only for a
target whose baseline is absent (*Establishing a baseline*). A backend that
cannot page its first page cheaply therefore pays once, not on every start.

That bound is on the ROWS, not on the round trips, and on a backend that finds
its namespace by scanning a key prefix the two come apart: proving a namespace
EMPTY means reaching the end of the keyspace however many batches that takes,
because `MATCH` is applied after each batch and a batch that matches nothing
looks exactly like the end. Redis is the case. It is not a cost the strict read
introduces -- `is_empty()`, the read every other start uses, is
`scan_iter(match=..., count=1)` over the same keyspace -- so what the strict
read buys (an outage that cannot be recorded as `origin=empty`) is bought at no
extra walk.

A page is also not a set: `iter_rows` is a best-effort snapshot, and Redis says
in its own docstring that `SCAN` may return a key twice while the keyspace
rehashes, leaving de-duplication to the caller that needs uniqueness. The chunk
sampler is that caller, because the probe weighs each sampled row as an
independent record -- so it keeps distinct ids only, and spends its budget on
rows examined rather than ids kept. A duplicated source shrinks the sample; it
never pads it with copies that would spend the comparison slots without adding
evidence.

## What the category does not retire

**The working-directory claim stays.** Even with configuration on a server
backend, two servers sharing one `working_dir` still overwrite each other's
`full_docs`, `doc_status`, graph and vectors — and lose more than baselines
doing it. The claim now follows the configuration storage onto `config_dir`,
which is where the file it protects actually is.

## What this does not retire

The baselines answer **identity**: is the configured space the one adopted for
this target. Two existing mechanisms answer different questions and stay:

- **`BaseVectorStorage.is_empty()` and the source/index pairing rule** answer
  **coverage**: does the index cover the data it indexes. They are the only
  evidence for a dropped container, an unmounted volume, a rebuild stopped
  between steps, a vector backend switched without migrating, a workspace prefix
  that does not match, and every deployment whose records are not established
  yet.
- **The per-container markers** travel with the data. They survive the
  configuration store being replaced or cleared, and they are the only evidence
  that can catch a container copied in from elsewhere or holding two embedding
  spaces at once.

All three are ANDed, and a configuration baseline **never suppresses** a refusal
from either of the others. Each refuses on evidence the others cannot obtain,
and every refusal must name which invariant failed, which target it concerns,
and the single command that fixes it — three ways to refuse startup is
acceptable only if an operator never has to guess which one fired.

## Obligations this design takes on

**Dropping a workspace deletes its configuration rows, data first.** See above;
needs a delete-then-recreate-then-start regression test and a partial-drop test.

**The store can never hold what is needed to reach the store.** The KV backend
selection and its connection string stay in the environment. "All configuration
in the database" means everything after the connection, not everything.

**Secrets need a policy before the first secret goes in.** API keys in the
knowledge-base database enter backups and any endpoint that echoes
configuration. The registry carries the `sensitive` flag from day one; what that
flag *does* — refuse to echo, encrypt at rest, or refuse storage entirely — is
not decided here and must be decided before a key marked sensitive exists.

## Accepted residues

Per *Consistency without transactions* in `AGENTS.md`, each is a decision with a
recovery path.

**A legacy-container copy made before any judgement.** With records absent, the
Milvus / Qdrant / PostgreSQL legacy migration runs inside `initialize()` and may
copy rows into a `{model}_{dim}d` container before the probe judges anything.
The copy is never served if the probe then refuses, and the next
`lightrag-rebuild-vdb` overwrites it. Recovery: the rebuild. This is the one
place where the ordering does not get ahead of the migration.

**An orphan configuration row after a failed drop cleanup.** The row outlives
its workspace. Recovery: a maintenance command that deletes rows whose scope
names a workspace with no data. Severity drops to reporting noise once workspace
names are UUIDs.

**A replaced or cleared configuration store loses every baseline.** Reads
confirm absent, which bootstraps again — the same class as a wiped marker, and
the same recovery. A separately selected backend adds a second way to reach
this state: point `config_storage` or `config_dir` at an empty or different
store and the start looks exactly like a first one. Announced rather than
enforced (`warn_about_unrecorded_baselines()`, the same posture as
`warn_about_workspace_overrides()`), because the two are indistinguishable from
inside the process. What is *not* residue: nothing is adopted on the configured
model alone even then — a baseline is established only for a target whose
container is confirmed empty or whose stored vectors an adoption probe vouched
for.

**Two deployments sharing one server backend AND one business workspace name
overwrite each other's baselines.** The container name is fixed for everyone,
so the row key's scope is the discriminator, and identical scopes collide. Same
unsupported case as two servers on one `working_dir`, one layer up; see *What
discriminates two deployments on one server* for why a deployment id is not the
fix. Recovery: give them different workspaces and rebuild with
`lightrag-rebuild-vdb`.

**A `RedisKVStorage` deployment's slice-1 records are not migrated.** Redis is
not in the configuration category, so such a deployment names one of the four
and its previous records stay where they are, unread. It is refused at startup
by name rather than starting on an empty store, and the baselines are
re-established on the next start — on probe evidence, never on the configured
model alone. Recovery, if the probe cannot vouch for them:
`lightrag-rebuild-vdb`.

**Whole-namespace publication on the JSON backend.** `JsonKVStorage` rewrites
the whole namespace file on flush, so `_lightrag_config` becomes a single write
point shared by every workspace in the process. Visibility is unaffected (its
data is a shared `Manager().dict()`), only publication granularity. Accepted
under the existing "file-backed storages are for small-scale testing and
validation only" limit.

**Colocation of every workspace's configuration in one container.** PostgreSQL
already colocates all workspaces in one table per namespace, so this is no
change there; MongoDB, OpenSearch and the JSON backends go from separated
to combined. Accepted because configuration is server-owned rather than
tenant-owned, which makes combining it the more correct semantics, not merely
the more convenient one.

## Rollout

| slice | contents |
| --- | --- |
| 1 | the `config` namespace across the five KV backends (PostgreSQL DDL + SQL templates; the internal reserved-workspace factory; the enumeration surface), the reserved `_lightrag*` name family, **three** `<workspace>/embedding/<target>` records with their verdicts, the split startup sequence with cleanup on early refusal, the atomic claim, per-target `rebuild_vdb` commits, data-first drop cleanup, key registry |
| 1b | configuration as its own **category** (`config_storage`, four admitted backends, refusal by name), `config_dir` for the JSON backend with the migration-free default, fixed container names on PostgreSQL / MongoDB / OpenSearch, the single-server claim moved onto `config_dir`, the unrecorded-baseline announcement, and the retirement of the whole reserved-name machinery |
| 2 | `_lightrag_server/embedding.current` / `.previous` and the startup inventory naming which workspaces still need a rebuild, over the enumeration surface from slice 1 |
| later | migrating existing environment variables into the store, key by key; a display-name → UUID mapping once workspace names become UUIDs; the secrets policy |

The server-level pair in slice 2 is **diagnostic and never gates**. It flaps
when differently configured servers start one after another against the same
store — sequentially, since concurrent initialization is unsupported — while a
per-target baseline moves only on a successful rebuild. A value that looks authoritative
and is not will otherwise be wired into a refusal by someone reading it later.

Slice 1 is a safety property and lands alone. Slice 1b changes where the
records live and nothing about what they mean, which is why it can follow
immediately: every verdict rule, the claim protocol, the rebuild and drop
orderings are untouched.

## Acceptance scenarios

The implementation is not complete until these are regression tests.

1. All three records match → normal startup.
2. `entities` mismatches → refused **before** any vector storage initializes.
3. `relationships` mismatches → refused, naming `relationships`.
4. `chunks` mismatches → refused, naming `chunks`.
5. Several targets mismatch → one refusal listing all of them, not one per start.
6. Legacy workspace, first start: each of the three probes succeeds on its own
   sample and each target records `origin=probe`; a target whose probe could
   not run stays absent while the other two are recorded.
7. Entity probe returns negative → refused, and **no** record is written.
8. Rebuilding `chunks` alone updates only `embedding/chunks`.
9. Rebuild succeeds, the configuration write fails → the tool exits non-zero and
   the next startup still refuses.
10. On OpenSearch, the claim flushes and strict-reads back before releasing the
    keyed lock; a flush the backend answered with a retryable per-item failure
    (the operation still buffered) fails the claim, the rebuild record and the
    drop rather than being confirmed from the buffer.
11. Two workers of one Gunicorn master claim concurrently → exactly one baseline.
12. The configuration container is unreachable by naming a workspace: a tenant
    called `_lightrag_config` starts normally and shares no file, table,
    collection or index with it.
13. Any workspace storage fails to drop → all three records are kept.
14. Every drop succeeds → the three records are deleted, and a workspace of the
    same name can be recreated and started.
15. A configuration read fails at transport level → startup fails; it is not
    treated as absent.
16. The inventory pages through configuration rows on each of the five KV
    backends.
17. A configuration strict read fails at step 2 → the configuration storage is
    fully finalized and no other storage was initialized.
18. The coverage gate or the entity probe refuses → `finalize_storages()`
    releases the configuration storage **and** every business storage.
19. A claim, read-back or configuration flush fails → a second
    `initialize_storages()` fails again instead of early-returning on
    `INITIALIZED`.
20. A normal shutdown finalizes the configuration storage together with the
    others, exactly once.
21. Concurrent claims are covered only for workers of one Gunicorn master;
    nothing asserts anything about two independent masters, which the contract
    does not support.
22. A deployment on slice 1's layout upgrades with **every** baseline still
    read: the recorded model is unchanged, its `origin` is not rewritten, and a
    mismatching model still refuses. The counterexample is pinned too — the
    same deployment with `config_dir` pointed elsewhere reads absence.
23. A configuration backend outside the four is refused at construction with a
    message naming it; a vector storage class in particular, and
    `RedisKVStorage` both when named and when it would be inherited from
    `kv_storage`.
24. A start on which no baseline is on record announces the container it
    looked in; a start with records, and a start missing only some of them,
    stay quiet.
25. Two deployments sharing one container hold disjoint rows, pinned per
    backend: one collection on MongoDB, one index on OpenSearch, one partition
    constant on PostgreSQL, one file on JSON — and keys that differ by scope.
26. The single-server claim is taken on `config_dir`, not `working_dir`, and
    only when the configuration storage is file-backed.
27. A storage in step 4 raises — injected at the first, a middle and the last
    member of the loop. The configuration storage, the storage that raised and
    every storage initialized before it are each released exactly once;
    storages the loop never reached are not touched; the original exception is
    what propagates.
