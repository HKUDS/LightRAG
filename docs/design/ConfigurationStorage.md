# Configuration storage contract

Status: **slice 1 implemented** (`lightrag/config_store.py`, the `config` KV
namespace on all five backends, the nine-step startup in
`LightRAG.initialize_storages()`, the rebuild and drop commit protocols, the
enumeration surface `BaseKVStorage.iter_rows()`); slice 2 and everything under
*later* in the rollout table are still planned. Tracked in
[#4006](https://github.com/HKUDS/LightRAG/issues/4006). The acceptance
scenarios are regression tests under `tests/config_store/`, with the
per-backend enumeration tests beside each backend under `tests/kg/`.

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
| Workspace | `_lightrag_config` (fixed, reserved, internal) |
| PostgreSQL | new table `LIGHTRAG_CONFIG (workspace, id, value JSONB, create_time, update_time)` |
| MongoDB / Redis / OpenSearch / JSON | nothing new — a collection, a key prefix, an index, a file named after the namespace |

The namespace is a KV namespace on purpose: **KV container names carry no model
suffix**, so a record kept here does not move when the embedding model changes.
That is exactly the property the vector container name lacks, and it is the
reason this record can be evidence where the name cannot.

One fixed workspace holds every row — the server's own configuration *and* every
workspace's. Configuration does not follow the knowledge base it configures.

### The internal factory, and why the reservation cannot be a plain rule

The whole `_lightrag*` workspace-name family is reserved and
`validate_workspace()` rejects it — **case-insensitively**. OpenSearch
lowercases index names, so `_LightRAG_config` and `_lightrag_config` are one
index there, and a reservation that knew only one spelling would let an
ordinary storage reach the configuration container through the other. The
grant, by contrast, admits exactly one spelling. That rule, applied naively,
**rejects the configuration storage itself**: all five KV backends call
`validate_workspace(self.workspace)` in `__post_init__` (`JsonKVStorage`,
`RedisKVStorage`, `MongoKVStorage`, `PGKVStorage`, `OpenSearchKVStorage`). So
the reservation needs a private door, and the
door has to be one ordinary configuration cannot find:

- public `LightRAG(workspace="_lightrag_config")` is **refused**;
- an ordinary storage construction cannot reach a reserved name;
- only the configuration-storage factory may bind `_lightrag_config`, through an
  internal-only parameter that is not part of the public storage signature;
- **no `allow_reserved=True`-style flag on the public constructor** — a public
  bypass is the reservation with extra steps;
- `PG_WORKSPACE`, `REDIS_WORKSPACE`, `MONGODB_WORKSPACE` and
  `OPENSEARCH_WORKSPACE` must not remap the internal container onto an ordinary
  workspace. The configuration container's workspace is fixed, not configured.
- Nor may any `*_WORKSPACE` variable point tenant data INTO the family: the
  override is applied after `validate_workspace()` has passed the constructor
  argument, so every backend that honors one validates the override's value
  too (`validate_workspace_override`) and refuses a reserved name at
  construction. Neo4j and Memgraph validate after applying theirs and need no
  second check.

Reserving must happen in the **first** slice, before any deployment can create a
workspace with such a name: a reservation made later cannot reclaim a name
already in use. Cost: a deployment whose workspace is already named that way
fails to start, loudly, with a message that names the reason.

PostgreSQL is the only backend needing schema work: `NAMESPACE_TABLE_MAP` gains
an entry, `TABLES` gains the DDL, and `PGKVStorage` needs the SQL templates it
dispatches per namespace (`get_by_id_config`, `get_by_ids_config`,
`upsert_config`). See *Enumeration* for the one thing the other four do owe.

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
8.  flush the configuration storage
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
- a storage the loop never reached is not `finalize()`d — but it is not simply
  left alone either, because construction is not free everywhere. See
  *Construction is not free* below.

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

#### Construction is not free

`RedisKVStorage` and `RedisDocStatusStorage` take their shared-pool reference
in `__post_init__`, not in `initialize()` — alone among the backends. Two
consequences, and they need different answers.

**A startup that fails before a storage's turn.** Construction SUCCEEDED, so
every Redis-backed storage already holds a reference; a refusal at step 2 or 3
means the rollback list names only the configuration storage, and the other
references belong to storages whose `initialize()` never ran. `finalize()` is
not available to the rollback there — no backend promises it works on an
instance that never initialized, which is the same argument that keeps
`INITIALIZED` behind the loop. So the rollback calls `release_unstarted()` on
them instead: a surface whose default releases nothing, overridden only where
the constructor took something. The failure is sticky, so nothing it releases
can be wanted again. The alternative — moving the acquisition into
`initialize()`, where every other backend does it — is the better fix and is
tracked in [#4016](https://github.com/HKUDS/LightRAG/issues/4016); `release_unstarted()` is correct either way, and becomes a
no-op once that lands.

**A constructor that raises.** `LightRAG.__post_init__` is synchronous, so
there is no rollback to run at all: a constructor that raises after another
Redis storage was built leaks that reference with no `finalize()` reachable.
That predates this design (twelve constructors ran in sequence before it), and
what this slice owes is to add nothing to it: the configuration storage is
constructed as the **last statement of `__post_init__` that can raise** —
after every business storage and after every validation that follows them
(`llm_model_func` present, `role_llm_configs` well-formed) — so a refusal
anywhere in construction, a reserved `*_WORKSPACE` override being the one this
slice introduces, finds nothing of the configuration storage's to leak. A new
check added to `__post_init__` goes **above** that construction.

**A failed step 4 does not heal by retrying.** Either the failure is retained
the way post-`INITIALIZED` failures are, or a full retry is supported and
re-runs every step from 1; what is not acceptable is a second
`initialize_storages()` returning successfully because some state was left
behind by the first. The implementation takes the first option, for every
failure before `INITIALIZED` and not only step 4's: not every backend's
`finalize()` is reversible (`RedisKVStorage.close()` drops its client while
leaving `_initialized` set), so a retry on the same object could neither
succeed honestly nor re-run from step 1. A new instance is the retry.

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
`ConfigurationStorageError`. Only after that does the strict read-back confirm
anything, and it is then a read of the server. Backends without a buffer answer
`False` and pay nothing.

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
→ verify the rebuild succeeded
→ update THAT target's configuration key
→ flush / index_done_callback the configuration storage
```

- rebuilding `entities` updates `embedding/entities` and nothing else; likewise
  for the other two;
- the configuration write happens **after** the target is durable and verified,
  never before;
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
  the namespace.

Whether the five implementations land in slice 1 or slice 2 is a scheduling
choice; what is not a choice is pretending PostgreSQL is the only backend with
work to do.

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
the same recovery.

**Whole-namespace publication on the JSON backend.** `JsonKVStorage` rewrites
the whole namespace file on flush, so `_lightrag_config` becomes a single write
point shared by every workspace in the process. Visibility is unaffected (its
data is a shared `Manager().dict()`), only publication granularity. Accepted
under the existing "file-backed storages are for small-scale testing and
validation only" limit.

**Colocation of every workspace's configuration in one container.** PostgreSQL
already colocates all workspaces in one table per namespace, so this is no
change there; MongoDB, Redis, OpenSearch and the JSON backends go from separated
to combined. Accepted because configuration is server-owned rather than
tenant-owned, which makes combining it the more correct semantics, not merely
the more convenient one.

## Rollout

| slice | contents |
| --- | --- |
| 1 | the `config` namespace across the five KV backends (PostgreSQL DDL + SQL templates; the internal reserved-workspace factory; the enumeration surface), the reserved `_lightrag*` name family, **three** `<workspace>/embedding/<target>` records with their verdicts, the split startup sequence with cleanup on early refusal, the atomic claim, per-target `rebuild_vdb` commits, data-first drop cleanup, key registry |
| 2 | `_lightrag_server/embedding.current` / `.previous` and the startup inventory naming which workspaces still need a rebuild, over the enumeration surface from slice 1 |
| later | migrating existing environment variables into the store, key by key; a display-name → UUID mapping once workspace names become UUIDs; the secrets policy |

The server-level pair in slice 2 is **diagnostic and never gates**. It flaps
when differently configured servers start one after another against the same
store — sequentially, since concurrent initialization is unsupported — while a
per-target baseline moves only on a successful rebuild. A value that looks authoritative
and is not will otherwise be wired into a refusal by someone reading it later.

Slice 1 is a safety property and lands alone.

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
12. A reserved workspace name is refused for a public construction and accepted
    through the internal factory.
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
22. A storage in step 4 raises — injected at the first, a middle and the last
    member of the loop. The configuration storage, the storage that raised and
    every storage initialized before it are each released exactly once;
    storages the loop never reached are not touched; the original exception is
    what propagates.
