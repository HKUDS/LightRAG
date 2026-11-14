# Configuration storage: design and implementation

Status: **describes the implementation as it stands**, after the
per-workspace JSON snapshot layout replaced the single aggregate file. It is
the design companion of [ConfigurationStorageContract.md](ConfigurationStorageContract.md):

| document | answers | authority |
| --- | --- | --- |
| this file | what the parts are, why they are shaped this way, where the original plan was changed and why, what is still weak, and how to change it safely | explanatory |
| `ConfigurationStorageContract.md` | every rule the code keeps, the refusal table, the accepted residues, the rejected alternatives, the acceptance scenarios | normative |
| `VectorSpaceProvenance.md` | the verdict rules the baselines feed, the adoption probe, the coverage gate | normative |
| `ServerInstanceContract.md` | what several servers on one host may share, and what is refused or undetected | normative |

When this file and a contract disagree, the contract wins and this file is
the one to fix. Rules are summarized here only as far as the design needs
them; do not copy a rule here instead of reading it there.

## 1. The problem

Two needs meet in one place.

- **A vector container name is an assertion, not a record.** Milvus, Qdrant
  and PostgreSQL name their vector containers after the *current* embedding
  configuration (`{model}_{dim}d`), so the name can never disagree with that
  configuration and proves nothing about what the stored vectors were
  written in. Switching the embedding model silently opens an empty
  container and serves a knowledge graph without vectors. Detecting that
  needs the adopted embedding space to be **recorded, per workspace and per
  vector target, outside the vector containers**: the *embedding
  baselines*.
- **Server-owned settings must be reachable independently of the active
  workspace.** Business storages are bound to one workspace; configuration
  is read before any workspace is open, and outlives each one.

The configuration storage is a KV namespace (`config`) that holds both. KV
container names carry no model suffix, so a record kept there does not move
when the embedding model changes. That is the property the vector container
name lacks.

## 2. Goals and non-goals

Goals, in priority order:

1. **Fail closed on evidence, never on absence.** A recorded baseline that
   disagrees with the configuration refuses the start; a record that does
   not exist yet is established only on positive evidence (an adoption probe
   or a confirmed-empty container), never from the configured model alone.
2. **Never mistake an unreachable store for an empty one.** Every read that
   decides whether the instance may serve is strict: a transport error, a
   damaged row or an unprovable answer is a failure, not "absent".
3. **Detect configuration drift**: a start that silently reads another
   configuration container (business KV changed while `LIGHTRAG_CONFIG_STORAGE`
   was unset, a different database of the same type, a replaced volume).
4. **No new preparation for an existing deployment.** Existing `WORKING_DIR`,
   workspaces and business backends keep working; the first start of this
   version binds and records on its own.
5. **Several servers, different workspaces, one `WORKING_DIR`**, including
   with JSON configuration.

Non-goals, stated so nobody designs for them by accident:

- **Multi-host concurrency.** Every lock here is per host and per process
  tree. NFS/SMB sharing of `WORKING_DIR` between hosts is unsupported.
- **Two instances on the same workspace.** Refused where a JSON claim and OS
  locks work; otherwise undetected and unsupported.
- **Transactions across stores.** The repository rule *Consistency without
  transactions* (AGENTS.md) applies: every multi-store operation has
  intermediate states; each accepted one is written down with its recovery.
- **Throughput.** The JSON backend rewrites a whole snapshot per flush and is
  for small-scale use, like every file-backed storage.
- **Compatibility with unreleased layouts.** The configuration storage never
  shipped in a release before this layout, so earlier dev-only files are not
  read, converted or diagnosed.

## 3. Architecture

### 3.1 Components

| module | role |
| --- | --- |
| `lightrag/config_store.py` | the facade: backend selection (`resolve_configuration_storage`, `default_configuration_storage`), `create_configuration_storage`, the key registry and row envelopes, strict reads, identity read/write, the bind (`bind_configuration_identity`) and the tools' verification (`verify_configuration_identity`), baseline precheck / claim / read, workspace-drop cleanup, the JSON snapshot validation (`inspect_shard_rows`, `attributable_json_snapshots`, `surviving_json_group`) |
| `lightrag/config_anchor.py` | the anchor file: strict parse, canonical UUIDs, no-clobber publish for a bind, atomic replace for a member append or a migration commit, the `IDENTITY_*` refusal causes |
| `lightrag/config_shards.py` | the JSON layout on disk: fixed paths (`json_config_dir`), workspace-name rules (`validate_config_workspace`, `validate_deployment_workspace`, `caseless_alias`), the bounded discovery scan, physical-directory alias detection |
| `lightrag/kg/anchor_lock.py` | the anchor lock (shared for starters, exclusive for the migration) and the bind lock (serializes first binds and JSON member appends) |
| `lightrag/kg/working_dir_lock.py` | the per-snapshot lifetime claim of a JSON configuration (one process tree per workspace snapshot) |
| `lightrag/namespace.py` | the reserved file names and key prefixes |
| the `config` branch of `JsonKVStorage`, `PGKVStorage`, `MongoKVStorage`, `OpenSearchKVStorage` | the four admitted backends; each maps the `config` namespace to a fixed container (file, table, collection, index) no workspace can address |
| `LightRAG.initialize_storages` / `finalize_storages` | the startup sequence and its sticky failure and rollback rules |
| `lightrag/api/gunicorn_config.py` | the master takes the anchor lock and the directory claim before forking, and runs the anchor-type check once |
| `lightrag/tools/rebuild_vdb.py`, `clear_storage.py`, `migrate_config.py` | maintenance: rebuild records baselines last, clear deletes them last, migration moves the whole container between backend types |
| `scripts/setup/` (`make env-storage`, `make env-validate`) | the wizard reads the anchor (with a parser kept in parity with `config_anchor.py`) and pins `LIGHTRAG_CONFIG_STORAGE` to the anchored backend |

### 3.2 What lives where

```text
configuration container (one per deployment, never per workspace)
  $meta/storage_identity          {"uuid": ...}            the container's identity
  $default/embedding/<target>     {model, dim, origin}     the empty workspace's baselines
  <workspace>/embedding/<target>  {model, dim, origin}     a named workspace's baselines
  $meta/json_shard                {"workspace": w}         JSON only: the snapshot's owner

WORKING_DIR/
  config_storage_anchor.json      {backend, storage_uuid[, layout, members]}
  .lightrag_anchor.lock           anchor lock (shared / exclusive)
  .lightrag_anchor_bind.lock      bind lock
  kv_workspace_config.json        JSON: the empty workspace's snapshot
  .lightrag_storage.lock          JSON: its lifetime claim
  <workspace>/
    kv_workspace_config.json      JSON: that workspace's snapshot
    .lightrag_storage.lock        JSON: its lifetime claim
    ...business files of file-backed storages
```

`<target>` is `entities`, `relationships` or `chunks`. A database backend
keeps every row in one fixed table/collection/index; JSON keeps one snapshot
per workspace, all carrying the same identity, and that set of snapshots is
**one logical container** (a *group*).

### 3.3 The three checks a start makes

```text
anchor  <->  container identity         is this the configuration container this deployment is bound to?
JSON: anchor members <-> snapshot owner is this snapshot the registered one for this workspace?
baseline <-> configured embedding       were this workspace's vectors written in the configured space?
```

The first two are about *which* records are read; only the third is about
the data. The baselines are ANDed with the coverage gate and the
per-container markers of `VectorSpaceProvenance.md`, never a substitute for
them.

## 4. Data model decisions

### 4.1 Keys

`<scope>/<suffix>`, where the scope is a named workspace, `$default` for the
empty workspace, or `$meta` for metadata. The separator is `/`, which
`validate_workspace` forbids in a name, so a key is never ambiguous and is
never re-parsed: business rows carry their real `workspace` field, metadata
rows are recognized by their registered key and carry no `workspace` at all.
Every suffix is declared in `CONFIG_KEY_REGISTRY` before anything writes it.

*Why `$` prefixes.* The first layout used `_lightrag_server/...` for
metadata and `/embedding/...` for the empty workspace. `_lightrag_server` is
a legal tenant name, so a tenant could collide with the server scope, and
the leading `/` was an accident of joining an empty scope. A `$` cannot
appear in a server-normalized name, so reserving it for internal scopes
closes the collision for every backend at once. The price is that the SDK,
the server and every backend override now reject a workspace that starts
with `$` (see 7.1).

### 4.2 The identity and the anchor

The container's identity is a random UUID in `$meta/storage_identity`. The
deployment's anchor, `WORKING_DIR/config_storage_anchor.json`, records which
backend type and which identity this `WORKING_DIR` is bound to. A start
compares both before reading any baseline.

- **Identity first, anchor second.** A bind writes (or adopts) the identity,
  then publishes the anchor no-clobber. Every crash between the two heals by
  adoption on the next start, so there is no `pending` state and no journal.
- **The anchor sits at the root for every backend**, because its scope is
  the deployment, not a workspace (the empty workspace's snapshot lives at
  the root too, and must not be the thing that defines the deployment).
- **Strict format**: integer `schema_version` 1 (a boolean is refused),
  canonical lowercase UUID, no unknown fields, `layout`/`members` only on a
  JSON anchor, distinct legal members. A symlinked or non-regular anchor file
  is refused: the atomic replace would swap the link and leave its target
  stale.
- **Deleting the anchor, with every writer stopped, is the rebind.** No
  recovery command exists; the next start binds to whatever is selected and
  says so in a WARNING.

### 4.3 JSON: one snapshot per workspace, one group per `WORKING_DIR`

The previous dev layout kept every workspace's configuration in one file
claimed exclusively by one process tree, so two servers on different
workspaces could not share a `WORKING_DIR` with JSON configuration. Each
workspace now owns a snapshot beside its business files, claimed by its own
lifetime lock; the anchor keeps the group's UUID and the member list.

- **Paths are derived, never chosen.** `json_config_dir(working_dir,
  workspace)` is the only rule. There is no directory setting and no
  constructor argument (`LIGHTRAG_CONFIG_DIR` and `LightRAG(config_dir=...)`
  were removed without a deprecation path, since they never shipped).
- **The member list is kept on purpose.** Its purpose is complete-group
  migration: a JSON-to-database migration refuses if even one registered
  member's snapshot is missing. Discovering members by scanning was
  considered and rejected: a scan cannot tell a never-created snapshot from
  a deleted one. The cost is that a first start of a workspace rewrites the
  anchor (member append, under the bind lock).
- **The owner row** names the workspace a snapshot belongs to, so a copied
  or moved snapshot is identified as such instead of being adopted.
- **In-process bookkeeping is keyed by the snapshot's real path**, because
  every configuration storage shares one namespace tag; without that, two
  workspaces' snapshots open in one process tree would alias.

### 4.4 Workspace names

`validate_workspace` holds for every backend: no path separator, not `.` or
`..`, no leading `$`. Where a configured storage keeps a workspace directory
under `WORKING_DIR` (JSON configuration or any file-backed data storage),
`validate_deployment_workspace` adds the directory rules of
`validate_config_workspace`: the five root file names and their case,
Unicode-normalization and trailing-dot/space aliases, names of only dots and
spaces, and drive-qualified names. A deployment entirely on database
backends creates no workspace directory and keeps every name it could use
before. Names are never rewritten by these checks.

Unicode normalization is handled without renormalizing anything: a
directory listing that differs from a requested workspace (or from its
snapshot's recorded owner) only in normalization, and resolves to the same
directory, is that workspace's own directory as HFS+ lists it; the other
spelling of a directory another workspace owns, which APFS resolves to the
same directory, is an alias and refused.

## 5. Lifecycle

### 5.1 Startup

The full sequence and its stickiness rules are in the contract (*Startup
sequence*). The design in one picture:

```text
0a  anchor lock (shared)                     not sticky: opens nothing
0b  strict-read the anchor
0c  anchor names another backend type -> refuse
    JSON: claim this workspace's snapshot directory
1   initialize the configuration storage
1b  verify the identity against the anchor, or bind:
      database: adopt the container's identity, or create it if confirmed absent;
                publish the anchor no-clobber
      JSON:     registered member -> verify identity + owner
                otherwise, under the bind lock: rebind from consistent surviving
                snapshots if there is no anchor; write identity + owner to this
                workspace's snapshot, flush, check the file; append the member
2   strict-read this workspace's baselines
3   precheck existing baselines -> any mismatch refuses before any vector storage opens
4   initialize the business storages
5   INITIALIZED (resources exist and must be released; not "checks passed")
6   coverage gate and adoption probe
7   establish the absent baselines on evidence, under a keyed lock, strict flush + read-back
8   flush the configuration storage
```

Two ordering choices carry the design:

- **The precheck runs before the vector storages initialize**, because
  Milvus, Qdrant and PostgreSQL run a legacy-container migration inside
  `initialize()`; a recorded mismatch must refuse before that copy happens.
- **Registration comes before the embedding checks**, so a start that is
  later refused for the model still leaves a registered workspace that
  `lightrag-rebuild-vdb` can repair.

### 5.2 Maintenance tools

- **`lightrag-rebuild-vdb`** re-embeds one workspace and records each
  target's baseline only after that target is durable and verified (data
  first, configuration last). `--workspace` selects an SDK workspace by its
  exact name; it is the tool's only option, so a server flag can never
  select one embedding configuration while the environment selects another.
- **`lightrag-clear-storage`** and `/documents/clear` drop every data storage
  first and the workspace's baselines last; the identity, the owner row and
  the membership stay, so the workspace remains registered.
- **Both verify and never bind.** They refuse an unanchored or unregistered
  JSON workspace, with advice that names a start able to register it: the
  server for a name it keeps, a library start for a name the server would
  rewrite.
- **`lightrag-migrate-config`** moves the whole container to another backend
  type under the exclusive anchor lock: the identity is kept, the target is
  claimed by writing the identity first (which makes a re-run safe without a
  journal), the rows are converged and verified, and the anchor switch is the
  commit point. A JSON side is always the whole group. Same-type moves are
  the backend's own dump/restore.

## 6. Concurrency

| lock | file | mode | purpose |
| --- | --- | --- | --- |
| anchor lock | `WORKING_DIR/.lightrag_anchor.lock` | shared for every starter and tool; exclusive for the migration | keeps starters out while the container moves |
| directory claim | `.lightrag_storage.lock` beside each JSON snapshot | exclusive per process tree, held for the instance's lifetime | one process tree per JSON snapshot; Gunicorn workers inherit the master's claim |
| bind lock | `WORKING_DIR/.lightrag_anchor_bind.lock` | exclusive, waited for up to a timeout | serializes a first bind and a JSON member append across the servers of one `WORKING_DIR`; a registered start never takes it |
| keyed lock `configuration_identity` | in-process shared storage | exclusive | serializes the workers of one master |

Order: anchor lock, then the directory claim, then the bind lock, each held
until teardown. Where file locks are unavailable the bind lock fails open
with a warning, so a single JSON server (the unset-Redis default included)
still starts; the price is that concurrent first starts can lose a member
append, detected and recovered as the contract describes.

## 7. Departures from the original plan

The implementation was planned in a design document written before the
work started (*Workspace-local JSON configuration with a shared anchor*). It
was implemented with the following departures. Each was reviewed when it
was made; the judgement column is the current assessment.

| topic | plan | implementation | why | judgement |
| --- | --- | --- | --- | --- |
| metadata keys | `_lightrag_server/storage_identity`, a server-scope `json_shard` row | `$meta/...`; the empty workspace's rows under `$default/...`; metadata rows without an outer `workspace` | `_lightrag_server` is a legal tenant name and could collide; a leading `/` for the empty scope was accidental | sound for the key format; the name reservation has a cost, see 7.1 |
| reserved names | the five root file names, exactly | also their case/Unicode/trailing-dot aliases, dot/space-only names, drive-qualified names | each closes a real collision on Windows/macOS filesystems | sound |
| where name rules apply | every effective workspace | the directory rules only where a configured storage is file-backed | a database-only deployment creates no directory and must keep its names | sound |
| discovery and symlinks | do not follow symlinks | a symlinked child **directory** is followed for its one probe; a symlinked **file** is refused; one physical directory under two names is refused | the running storage follows the link, so registration and discovery must agree, or a symlinked workspace would vanish on rebind and migration | sound |
| discovered names | never normalize again | spellings are never renormalized, but an HFS+ NFD listing is matched to the requested name or the stored owner spelling when it is the same directory | HFS+ lists NFC names in NFD; an unconditional NFC rule broke NFD workspaces on APFS and hid a real alias, so the stored spelling decides | sound |
| bind-lock fail-open | only where POSIX reports locking unsupported; Windows: every error is contention | identified contention (`EWOULDBLOCK`; every `msvcrt` failure) is never bypassed; every other way the lock cannot be taken fails open with a warning, a lock file that cannot be opened included | no errno list tells "unsupported" apart reliably across NFS and SMB, and refusing would stop a single server that runs there today; the residue is detected and recoverable | sound, now stated precisely in the code and the contracts (8, item 1) |
| no-clobber anchor publish | verify a concurrent winner | a lost race refuses with `anchor_appeared`; a database loser reads the winning anchor and the container's identity and reports whether they agree | fails loud; a retry heals the consistent case, and the split case names its recovery | sound (8, item 2) |
| rebuild tool workspace | derive one workspace path | plus `--workspace` for an exact SDK name; every other flag refused | the server rewrites `team-a` to `team_a`, so an SDK workspace was unreachable | sound |
| unregistered-workspace advice | "start the server once" | that, or a library start for a name the server would rewrite | the server advice can never succeed for such a name | sound |
| database-to-JSON claims | claim the union of directories in canonical order, then validate | existing snapshots are claimed while preparing; new snapshot directories only after the target is classified | a refused target must not leave directories behind while reporting that nothing was written; the exclusive anchor lock already keeps starters out | sound |
| JSON targets | refuse foreign UUIDs, malformed rows, wrong owners | as planned, plus server-scope rows other than the identity and keys a snapshot cannot hold refuse; a malformed row in an **owned** target is deleted as this migration's residue | no rule exists for splitting a server-wide setting across snapshots; an owned target is inactive and the source is authoritative | sound; the malformed-row deletion is now documented (8, item 6) |
| UUID mismatch on a JSON snapshot | delete the anchor and rebind | restore the snapshot, or move it aside (a member: then delete the anchor so the member list is rebuilt); delete the anchor only when every snapshot carries the other UUID | the rebind refuses snapshots that disagree, so the planned advice cannot work with one stray snapshot | sound |
| clear | retain owner, identity, membership | as planned | | as planned |
| startup refusals | not specified | anchor, lock and directory-claim refusals are one concise `Startup blocked` line, with the chained I/O cause | the messages carry their own advice; a traceback buried it | sound |

### 7.1 The `$` reservation and existing deployments

The plan promised that a deployment from the last release needs no
preparation. That holds except for one case: a server started with
`WORKSPACE=$x` used to be served as `_x` (the server rewrites every
character outside `[A-Za-z0-9_]`), and is now refused before that rewrite,
with a one-line `WORKSPACE: ...` refusal. The rewritten name could not have
collided with a `$` scope, so the refusal is a policy choice, not a
necessity.

**Decision: keep the refusal.** A name starting with `$` is reserved
everywhere, the server included, so no layer silently serves a name the
operator did not type, and the reservation reads the same in the SDK, the
server and every backend override. The change must be stated in the
release notes. The upgrade is a one-line change with no data movement: such
a deployment already stored everything under the rewritten name, so it sets
`WORKSPACE=_x` -- the name it was served under -- and starts as before.
Reopening this needs a deployment that cannot make that change.

## 8. Audit findings and their resolution

Auditing the implementation against the plan found the items below. None
lost data; each was a misleading message, an undocumented residue or a
missing test. They are kept here because each records a decision.

| # | finding | resolution |
| --- | --- | --- |
| 1 | the bind lock fails open more widely than its docs said (any non-contention `flock` error, an unopenable lock file) | kept: the rule is "identified contention is never bypassed", for the reason in 7; the module docstring and both contracts now state it, and tests pin contention, `ENOLCK` / `EOPNOTSUPP` / `EIO` and an unopenable lock file |
| 2 | a database start losing the no-clobber anchor race reported "Nothing was overwritten" although its identity upsert may have replaced the winner's | the loser reads the winning anchor and the container's identity and reports the consistent or the split state with its recovery; listed as an accepted residue in the contract |
| 3 | the no-hard-link publish fallback could leave an empty anchor while reporting that none was published | a failed replace removes the empty claim; if that fails too, the refusal says the empty file remains |
| 4 | the migration verified a JSON target through shared memory before switching the anchor | `verify_layout` reads every committed snapshot from disk and compares it with what was flushed |
| 5 | the rebind WARNING did not say retained migration sources may be stale | it does |
| 6 | the deletion of a malformed row in an owned target was undocumented | documented in the contract and `README_MIGRATE_CONFIG.md` |
| 7 | `env.example` promised a same-workspace refusal without the file-lock condition | qualified |
| 8 | `/health` reported the derived directory as `config_dir`, the removed setting's name | renamed `config_snapshot_dir` (never released) |
| 9-12 | acceptance cases covered only indirectly | each has a direct test: a start refused after registration leaves a rebuildable member; rebuild refuses other options; interrupted database-to-JSON convergence retries; mixed own/foreign target identities refuse; a damaged owned-target row converges; a wrong-owner JSON source refuses; a database bind ignores retained JSON snapshots; a lost member append heals through a rebind; an ambiguous member append is reported as it is |

**Still open (minor):** `anchor_lock.py` reimplements the Windows
`msvcrt` locking loop instead of sharing `working_dir_lock.py`'s; the
behaviour is the same, so unify them when either changes.

## 9. Changing the design

### 9.1 Invariants to keep

A change that breaks one of these needs a decision recorded in the contract
first.

- A read that could not complete is never "absent".
- A baseline is written only on evidence, under the keyed lock, with a
  strict flush and read-back; a probe that could not run writes nothing.
- Data first, configuration last: rebuild records a baseline after the data
  is durable; a drop deletes baselines after every data storage dropped.
- Identity first, anchor second; the anchor is never overwritten by a normal
  start except for a JSON member append under the bind lock.
- A registered JSON snapshot is never re-created in place; a snapshot is
  never adopted by name alone (owner and UUID must agree).
- No `*_WORKSPACE` override and no setting addresses the configuration
  container or moves a snapshot.
- A workspace-name rule may only be added where the thing it protects exists
  (a directory, a key prefix), and names are never rewritten by validation.
- Every refusal names its cause and a way out that actually works; test the
  advice by following it.

### 9.2 Recipes

- **A new configuration key**: declare it in `CONFIG_KEY_REGISTRY` with its
  scope and schema version; give it a strict reader; decide what clear,
  rebuild and migration do with it; a new **server-scope** key needs a
  sharing or split policy for JSON first (a migration into JSON refuses
  server-scope rows other than the identity today).
- **Moving an environment setting into the store**: one key at a time, each
  subject to the contract's *Standing obligations*; the environment keeps
  winning until a key's precedence is decided.
- **A new admitted backend**: a fixed container no workspace can address,
  strict point reads, `iter_rows` for enumeration, a container marker if the
  backend can alias workspaces, and tests under `tests/kg/<backend>_impl/`
  mirroring the existing `*_config_identity` / `*_config_namespace` ones.
- **A filesystem rule** (names, links, locks): emulate the filesystem in a
  test rather than reasoning about it; HFS+ and APFS behave differently on
  Unicode, and Windows behaves differently on locks and trailing dots.

### 9.3 Not implemented yet

Carried over from the contract, where each is specified:

- the diagnostic `$meta/embedding.current` / `.previous` pair and a startup
  inventory of workspaces that still need a rebuild (never a gate);
- moving environment settings into the store, key by key;
- a display-name to UUID mapping for workspaces (would reduce the drop
  residue to orphan rows);
- a secrets policy.
