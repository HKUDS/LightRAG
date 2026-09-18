# Configuration storage contract

Status: **planned**. Nothing in this document is implemented yet. It states the
rules the implementation must follow, so that the first slice does not have to
be re-cut when the second arrives. Tracked in
[#4006](https://github.com/HKUDS/LightRAG/issues/4006).

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
`workspace` column, Redis and MongoDB by a name prefix, the JSON backends by a
subdirectory under `working_dir`. That is right for knowledge-base data and
wrong for configuration: the server reads its own settings while switching
between workspaces, and reads them before any workspace is open. Configuration
is the server's property, not a tenant's.

**A container name is an assertion, not a record.**
`BaseVectorStorage._generate_collection_suffix()` builds `{folded_model}_{dim}d`
from the *current* configuration, so it can never disagree with the current
configuration. Milvus, Qdrant and PostgreSQL name their vector containers that
way and therefore cannot tell whether an existing container's content came from
the model configured right now. `docs/design/VectorSpaceProvenance.md` records
the four cases that get through; closing them needs an embedding space that is
*recorded*, per workspace, outside the vector containers.

One durable place answers both: a record of what a deployment is configured
with, kept apart from the data that configuration produced.

## The container

| | |
| --- | --- |
| KV namespace | `config` |
| Workspace | `_lightrag_config` (fixed, reserved) |
| PostgreSQL | new table `LIGHTRAG_CONFIG (workspace, id, value JSONB, create_time, update_time)` |
| MongoDB / Redis / JSON | nothing new — a collection, a key prefix, a file named after the namespace |

The namespace is a KV namespace on purpose: **KV container names carry no model
suffix**, so a record kept here does not move when the embedding model changes.
That is exactly the property the vector container name lacks, and it is the
reason this record can be evidence where the name cannot.

One fixed workspace holds every row — the server's own configuration *and* every
workspace's. Configuration does not follow the knowledge base it configures.

PostgreSQL is the only backend with real work: `NAMESPACE_TABLE_MAP` gains an
entry, `TABLES` gains the DDL, and `PGKVStorage` needs the three SQL templates
it dispatches per namespace (`get_by_id_config`, `get_by_ids_config`,
`upsert_config`).

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

The whole `_lightrag*` workspace-name family is reserved, enforced in
`validate_workspace()`. One rule rather than a new reservation every time a
scope appears — and reserving it must happen in the **first** slice, before any
deployment can create a workspace with such a name, because a reservation made
later cannot reclaim a name already in use. Cost: a deployment whose workspace
is already named that way fails to start, loudly, with a message that names the
reason.

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

## The first key: `<workspace>/embedding`

`value` is `{model, dim}` — the embedding space this workspace's vectors were
written in. The model name is stored **unfolded**, via
`lightrag.kg.vector_space.declared_model_name()`, for the same reason the
per-container marker stores it unfolded.

Verdicts, following *absent evidence never refuses* from
`docs/design/VectorSpaceProvenance.md`:

| record | configured | verdict |
| --- | --- | --- |
| present, differs in `model` or `dim` | any | **refuse to start**, naming both spaces and `lightrag-rebuild-vdb` |
| present, equal | any | proceed — one KV read, no embedding call |
| absent, sources empty | any | write the record; a fresh deployment defines its own space |
| absent, sources not empty | any | adopt **only** if the homogeneous cosine probe confirms the containers reproduce under the configured model; otherwise leave it absent and behave as before |

The absent-and-populated case is the one that must not be shortcut. Writing the
current configuration unconditionally would, for a deployment that upgraded and
changed models in the same step, stamp the wrong model permanently — and from
then on every check agrees with itself. The probe already exists for exactly
this decision on the per-container markers; this reuses it rather than
inventing a second rule.

`lightrag-rebuild-vdb` rewrites the record after a successful rebuild. Without
that, a deliberate model change would have no way through, and a gate with no
way through is a gate operators disable.

### Ordering: before the vector storages initialize

The check runs **before** any vector storage's `initialize()`, not after.

`initialize_storages()` currently initializes every storage and then runs the
cross-storage gate. That is too late for this record: the legacy-container
migration on Milvus, Qdrant and PostgreSQL runs *inside* `initialize()` and
copies rows from an un-suffixed container into `{model}_{dim}d` guarded only by
dimension. By the time the existing gate can refuse, a same-dimension model
change has already relabelled foreign vectors.

So the sequence becomes: initialize the KV storage that holds the configuration
→ read `<workspace>/embedding` → decide → initialize everything else. This is
the only structural change the first slice makes to startup.

The check belongs in core, not in `lightrag/api/`. A library user who never
runs the server needs the same refusal.

## The server-level pair, and why it never gates

A later slice adds `_lightrag_server/embedding.current` and
`_lightrag_server/embedding.previous`, advanced at startup by
`configured != stored.current → previous = stored.current; current = configured`.

It answers *did this server change model, and from what to what*. It does not
answer *which workspaces need rebuilding*: rebuilds complete one workspace at a
time, so at any moment some are done, some are not, some were created after the
change and some have not been touched in months. One pair of values cannot hold
those four states. The per-workspace record can, and it is the authority.

**The server pair is diagnostic and must never participate in a refusal.** It
flaps: two servers with different configurations starting alternately push it
back and forth, while the per-workspace record moves only on a successful
rebuild. Writing this down is the point — a value that looks authoritative and
is not will otherwise be wired into a gate by someone reading it later.

During the transition, while the embedding configuration still comes from the
environment, the server rows are a mirror of it at each start. The difference
between `current` and `previous` is just as usable.

## What this does not retire

The per-workspace record answers **identity**: is the configured space the one
this workspace's vectors were written in. Two existing mechanisms answer
different questions and stay:

- **`BaseVectorStorage.is_empty()` and the source/index pairing rule** answer
  **coverage**: does the index cover the data it indexes. They are the only
  evidence for a dropped container, an unmounted volume, a rebuild stopped
  between steps, a vector backend switched without migrating, a workspace
  prefix that does not match, and every deployment whose record has not been
  established yet.
- **The per-container markers** travel with the data. They survive the
  configuration store being replaced or cleared, and they are the only evidence
  that can catch a container copied in from elsewhere or holding two embedding
  spaces at once.

All three are ANDed. Each refuses on evidence the others cannot obtain, and
every refusal must name which invariant failed and the single command that
fixes it — three ways to refuse startup is acceptable only if an operator never
has to guess which one fired.

## Obligations this design takes on

**Dropping a workspace must delete its configuration rows.** When the record
lived in the workspace's own container this was free: deleting the workspace
deleted the record. Here it is an explicit step on the drop path, and it needs
a delete-then-recreate-then-start regression test. Skipping it leaves an orphan
row that a recreated workspace of the same name inherits, claiming its vectors
were written by a model they were not — a wrong refusal, or a wrong pass.

Workspace names becoming UUIDs later reduces this to orphan rows and a
misleading inventory rather than a wrong verdict, but does not remove the
obligation: `LightRAG(workspace=...)` is a library call too, and the
never-reuse-a-name guarantee would be a property of the server's creation path
only.

**The store can never hold what is needed to reach the store.** The KV backend
selection and its connection string stay in the environment. "All configuration
in the database" means everything after the connection, not everything.

**Secrets need a policy before the first secret goes in.** API keys in the
knowledge-base database enter backups and any endpoint that echoes
configuration. The registry carries the `sensitive` flag from day one; what
that flag *does* — refuse to echo, encrypt at rest, or refuse storage entirely —
is not decided here and must be decided before a key marked sensitive exists.

## Accepted residues

Per *Consistency without transactions* in `AGENTS.md`, each is a decision with a
recovery path.

**An orphan configuration row after a failed drop cleanup.** The row outlives
its workspace. Recovery: a maintenance command that deletes rows whose scope
names a workspace with no data. Severity drops to reporting noise once workspace
names are UUIDs.

**A replaced or cleared configuration store loses every record.** The verdict
falls back to absent, which never refuses — the same class as a wiped marker,
and the same recovery: the next start adopts on probe evidence, or an operator
runs `lightrag-rebuild-vdb`.

**Whole-namespace publication on the JSON backend.** `JsonKVStorage` rewrites
the whole namespace file on flush, so `_lightrag_config` becomes a single write
point shared by every workspace in the process. Visibility is not affected (its
data is a shared `Manager().dict()`), only publication granularity. Accepted
under the existing "file-backed storages are for small-scale testing and
validation only" limit.

**Colocation of every workspace's configuration in one container.** PostgreSQL
already colocates all workspaces in one table per namespace, so this is no
change there; MongoDB, Redis and the JSON backends go from separated to
combined. Accepted because configuration is server-owned rather than
tenant-owned, which makes combining it the more correct semantics, not merely
the more convenient one.

## Rollout

| slice | contents |
| --- | --- |
| 1 | the `config` namespace across the five KV backends (PostgreSQL DDL + SQL templates; nothing for the rest), the reserved `_lightrag*` name family, the `<workspace>/embedding` record and its verdicts, the ordering change in `initialize_storages()`, the `rebuild_vdb` write, drop cleanup, key registry with one entry |
| 2 | `_lightrag_server/embedding.current` / `.previous`, and the startup inventory naming which workspaces still need a rebuild. Listing the configuration container *is* the workspace inventory — the codebase has no other way to enumerate workspaces today, which is why this slice is cheap only after slice 1 |
| later | migrating existing environment variables into the store, key by key; a display-name → UUID mapping once workspace names become UUIDs; the secrets policy |

Slice 1 is a safety property and lands alone. Slice 2 is operator-facing
reporting built on top of it, and must not be folded in: one commit carrying one
idea reviews better than one commit carrying two.
