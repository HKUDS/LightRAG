# Vector-space provenance and the fail-closed gate

Read this before touching `lightrag/kg/vector_space.py`, `VectorSpaceMismatchError`,
any vector backend's attach path, or `lightrag/tools/rebuild_vdb.py`.

## The condition

An operator changes `EMBEDDING_MODEL` or `EMBEDDING_DIM`. The vectors already on
disk were produced by the *previous* model, and nothing about them changes. Two
outcomes are possible, and both used to happen depending on the backend:

- The vector container is reused, so queries return confidently wrong
  neighbours. Nothing detects this.
- A new container is provisioned (the model-isolation suffix on Milvus, Qdrant
  and PostgreSQL), so queries return nothing at all.

Both are **failing open**: the service starts and serves. The requirement is the
opposite — *refuse to serve, name what changed, and name the way out*.

Failing closed is only acceptable because a way out exists. That is the coupling
that orders this work: the recovery tool has to work before any backend is
allowed to refuse.

## Isolation taxonomy: which backends need a marker at all

A workspace may legitimately use a different embedding model from its
neighbours, so "what model wrote these vectors?" is a question about a
*container*, not about a deployment. How each backend isolates decides whether
that question is already answered by the container's **name**:

| backend | workspace isolation | model isolation | name answers "which model?" |
| --- | --- | --- | --- |
| Nano, FAISS | subdirectory under `working_dir` | none | ❌ |
| MongoDB, OpenSearch | collection / index name prefix | none | ❌ |
| Milvus | collection name prefix | collection name suffix | ✅ |
| Qdrant | point-id salting + `workspace_id` payload filter | collection name suffix | ✅ |
| PostgreSQL | `workspace` column | table name suffix | ✅ |

The split in the last column is the whole design:

- **Nano, FAISS, MongoDB, OpenSearch carry no model information in the container
  name.** A same-dimension model swap reuses the *same* container and nothing
  notices. These four need a recorded marker — they are the only backends where
  the marker is load-bearing.
- **Milvus, Qdrant and PostgreSQL already encode `{folded_model}_{dim}d` in the
  container name.** A model change lands in a different container by
  construction. **They record no separate marker**: a second copy of a fact the
  name already carries is redundant, and two copies of a fact drift apart.

Qdrant is worth stating explicitly because its collection name carries no
workspace, which reads like a gap and is not one. It uses Qdrant's own
multitenancy pattern, two independent mechanisms:

- **Point ids are workspace-salted** — `compute_mdhash_id_for_qdrant(id,
  prefix=effective_workspace)` hashes `workspace + id`, so the same chunk id in
  two workspaces yields two different point UUIDs. Two tenants cannot overwrite
  each other even if every filter were forgotten.
- **Every read and write filters on the `workspace_id` payload**, which carries
  a tenant index (`is_tenant=True`) so a tenant-filtered ANN query stays a
  tenant-local query instead of a full scan followed by a filter.

The consequence for provenance: a Qdrant collection is **multi-tenant**, so a
collection-level marker could not express "this workspace's model" even in
principle. It does not need to — the suffix guarantees every tenant in the
collection shares one model.

## What is recorded, and why a dimension is not enough

A dimension is not an identity. `text-embedding-3-small` at 1536 and a local
model at 1536 produce unrelated spaces, and a same-dimension swap is invisible to
every dimension check. So for the four backends above the **model name is
recorded next to the vectors** — never inferred.

`lightrag/kg/vector_space.py` owns the three things they need:

| helper | what it fixes |
| --- | --- |
| `declared_model_name(embedding_func)` | the single definition of "this instance's model": `str`, `strip()`ed, non-empty, otherwise `None` |
| `vector_space_marker(embedding_func)` | the payload written when a container is provisioned |
| `assert_vector_space_matches(...)` | the verdict, so the backends cannot drift apart on it |

The recorded name is **unfolded** — exactly as configured. The collection-name
suffix (`BaseVectorStorage._generate_collection_suffix`) lowercases and folds
every non-alphanumeric character to `_`, so `text-embedding-3-large` and
`text_embedding_3_large` produce the same suffix. Recording the folded name would
import that blind spot into the marker.

### Absent evidence never refuses

A container that records no model, or no dimension, predates the marker.
Silence must not read as a mismatch, or every index written before this feature
is refused on the first start after the upgrade.

The rule is symmetric: a process whose `embedding_func` carries no `model_name`
does not know what it is either, and cannot contradict a recorded name. A marker
payload that cannot be parsed reads as absent for the same reason.

The direct consequence is that silence never ends by itself, which is why an
unmarked container has to be *adopted* — see the transition below.

**Nothing but adoption may end the silence**, and that includes the write path.
A backend that records its marker when it *saves* is performing a backfill just
as surely as one that records it on attach, and a worse one: the rows it stamps
are mostly rows a previous model wrote. So a save may record the marker only
over a container this process can vouch for — one that was **empty** when this
process attached (every row since is one it wrote), or one that **already names
a model**, which the attach check has just confirmed is ours. A non-empty
container recording no model stays unmarked no matter how much is written to it,
until `drop()` empties it or the adoption probe certifies it. Getting this wrong
does not merely miss a detection: it records a false marker that every later
start believes, and that the adoption probe then sees no conflict in.

The same rule closes a second door: a process whose `embedding_func` has no
`model_name` is *accepted* against a marked container (it cannot contradict the
name), but it may not certify one either. If it did, its next save would replace
a payload naming a model with a dimension-only payload — erasing provenance that
was already established and reopening the very same-dimension swap the name was
recorded to catch. Certification needs **both** sides named; attach needs
neither.

### The server refuses to start unnamed

The library keeps working without `model_name` — and `lightrag-rebuild-vdb`
must, because it is the way out of a refusal. The **server** does not: it
refuses to start when `EMBEDDING_MODEL` is unset, empty, or whitespace.

A server with no name to record provisions containers that are unprotected for
life, and the rule above means that silence never ends on its own. There is also
no in-place way to fix it later: LightRAG propagates no configuration between
worker processes and supports no rolling update, so an embedding-model change is
always stop → `lightrag-rebuild-vdb` → start. A deployment that cannot say which
model wrote its vectors has no safe path through that sequence, and the failure
it is heading for is not an error — it is confidently wrong neighbours, returned
silently.

The refusal names `EMBEDDING_MODEL`, `lightrag-rebuild-vdb`, and why rolling is
not an option, because an operator who hits it at startup is exactly the
operator who needs all three. An `args` object that never carried the field is
refused the same way: a safety guard exempting "the attribute was never set" is
a guard with a bypass.

## Where the marker lives: never in the data plane

A marker must not be an ordinary vector record. The rule and the reason:

> A record that carries a vector lands in the ANN index, and anything in the ANN
> index can be returned by a search. A marker recalled as a search hit is a
> fabricated chunk entering an LLM's context.

| backend | marker home | why it cannot be recalled |
| --- | --- | --- |
| OpenSearch | index mapping `_meta`, beside the existing workspace identity | not a document |
| MongoDB | the collection's JSON Schema validator `description` | not a document |
| Nano | `additional_data` in the vdb JSON file | not a row in `data` / `matrix`; `query()` cannot see it |
| FAISS | a `<index>.space.json` sidecar file | not in the `.index` file, so `index.search()` cannot return it |
| Milvus, Qdrant, PostgreSQL | none — the container name is the provenance | nothing is stored |

Three approaches were considered and rejected:

- **A marker point in a Milvus or Qdrant collection.** Both require a point to
  carry a vector, so the marker would enter the ANN index. Milvus is the worse
  of the two: workspace lives in the *collection name*, so its queries carry no
  filter at all and the marker would be returned outright. Qdrant escapes only
  because every query filters on `workspace_id` — correctness resting on every
  future query remembering a filter.
- **A shared sidecar collection** for those two. It introduces a cross-workspace
  container into a design where every other container is per-workspace, adding a
  keying, lifecycle and permission surface whose mistakes are silent; it is not
  atomic with the container it describes; and it is unnecessary, because the
  name already carries the fact.
- **A marker document inside the Mongo vector collection.** It is not
  *recallable* (Atlas Vector Search only indexes documents carrying the indexed
  path, and every other read filters by `_id` / `src_id` / `tgt_id`), but it puts
  a row in the data collection and makes every future full-collection scan owe it
  an exclusion. The validator `description` is metadata and owes nothing.

FAISS is the one backend whose marker does **not** ride in a file the storage
already writes, and the reason is downgrade safety. Its `.meta.json` is
`{str(faiss_id): metadata}`, `_load_faiss_index` calls `int()` on every key, and
its `except Exception` falls back to "start with an empty index". An older
LightRAG reading a reserved key would therefore discard every metadata row, and
the next save would persist that emptiness — silent total loss of the store on a
rollback. A file an old reader never opens cannot do that.

The sidecar is deliberately **not** part of `_fingerprint_paths`: it carries no
rows, so a peer has nothing to reload because of it, and a third path would
change the two-file publication fence. It is written *before* the fenced pair,
so the metadata rename stays the last thing that happens — that rename is the
storage's commit point, and a third write after it would make a complete
publication look torn. Accepted residue: a crash between the marker write and
the pair leaves a marker describing rows that were not written. The marker only
changes when the operator changes the embedding configuration, and that is
exactly when the store is rebuilt anyway.

Nano needs none of this: `additional_data` is a key `NanoVectorDB` itself
round-trips through the same JSON object as the rows, so the marker is published
by the same atomic rename, and an older reader preserves and ignores it.

Two things the Mongo validator home requires, both easy to get wrong:

- **`drop()` must rewrite the description.** Mongo's `drop()` is
  `delete_many({})` — it removes documents, not the collection, so the validator
  survives. A `drop()` that leaves the old model recorded makes the *next*
  `initialize()` refuse again, which wedges the recovery path the tool depends
  on. (The rejected marker-document design did not have this trap; the validator
  design introduces it.)
- **A `collMod` permission failure degrades, it does not refuse.** A restricted
  role simply leaves the collection unmarked — exactly where it was before this
  feature existed. Same rule as OpenSearch's `_claim_index_for_workspace`, which
  already tolerates a failed marker write.

## `VectorSpaceMismatchError` is load-bearing

The refusal is a distinct exception type, not a `ValueError`, not an
`AssertionError`, and deliberately not `DataMigrationError` (nothing is being
migrated).

`lightrag-rebuild-vdb` responds to this condition by **dropping the container**.
A tool that reached that decision by catching `Exception` would drop data on a
cluster outage, an expired credential or a corrupt file. Nothing can tolerate
this condition safely until it is distinguishable from everything else that can
go wrong at attach time.

Two rules bind every raiser:

1. **Raise before the first storage mutation.** The refusal must leave the
   container exactly as it was.
2. **Leave the instance able to serve `drop()`.** The recovery is `drop()` then
   `initialize()` again, so a backend that raises before it has a client, a
   connection or its flush lock is *wedged*, not fail-closed — the operator then
   has to delete the container out of band, which is the defect this work
   removes.

### The named-container backends' refusal

Milvus, Qdrant and PostgreSQL record no marker, but they each already compare
the *dimension* of a container they are about to read against the one they are
configured with. That comparison is the same refusal, so it raises the same
type — and the two raiser rules apply to it unchanged.

Four things that were wrong before and are now part of the contract:

- **It is `VectorSpaceMismatchError`, not `DataMigrationError`.** Nothing is
  being migrated when it fires; the container is simply in another embedding
  space. Milvus additionally must not let `_validate_collection_and_load`
  reframe it as the generic "manual intervention required" `RuntimeError`,
  which is indistinguishable from a corrupt schema.
- **A refused instance stays droppable.** Qdrant and PostgreSQL assigned
  `_flush_lock` after the step that refuses, so `drop()` then died on
  `async with None` — the OpenSearch bug again. Both now take the lock before
  anything that can refuse, and `drop()` tolerates a container that does not
  exist, because the refusal is raised *before* the new collection or table is
  created.
- **A missing dimension on either side is a schema error, not the refusal.**
  *Absent evidence never refuses* binds the dimension comparison exactly as it
  binds the marker. `None != 768` reads as a mismatch, and the tool answers a
  mismatch by dropping the container — so a malformed `describe_collection`
  response, or an `embedding_func` that never declared a dimension, would
  authorise destroying live vectors over a fact nobody reported. Milvus checks
  both sides before comparing and raises a plain, non-droppable `ValueError` on
  either. PostgreSQL raises the same error for an undeclared dimension at the
  top of `setup_table()` rather than at the comparison, because that check sits
  inside a `try` whose `except Exception` reframes everything it catches as
  `DataMigrationError`; a *legacy* dimension it could not read is simply not
  compared, and the migration that follows fails closed at insert — the
  acceptable direction, since nothing is dropped.
- **The refusal is scoped to what THIS workspace would migrate.** Qdrant's
  legacy collection can be shared across tenants, and its gate counted every
  tenant's points. That refused a workspace with nothing to migrate, and the
  refusal could not be cleared: `drop()` only ever removes this workspace's
  legacy points, so the next `initialize()` refused again — wedged, not
  fail-closed. It now counts the legacy points this workspace would actually
  migrate (all of them when the collection is untagged, which is exactly what
  the migration below reads), so `drop()` → `initialize()` converges.

A Milvus *legacy* collection in another embedding space is deliberately NOT a
refusal: the suffixed collection does not exist yet, so nothing is being served
out of the wrong space. The legacy collection is only a migration source, and an
incompatible source is skipped. That case is the coverage gate's, not
this one's.

## The recovery protocol

`lightrag/tools/rebuild_vdb.py`:

- **Sources keep the server-identical init path.** The graph store and
  `text_chunks` are what the rebuild reads from; any failure there still aborts
  the run, migrations included. Rebuilding vectors out of a half-migrated source
  is worse than not rebuilding.
- **The three vector targets are initialized individually**, and *only*
  `VectorSpaceMismatchError` is tolerated. The refusal is recorded per target;
  anything else aborts.
- **The rebuild opens with `drop()`** on a refused target
  (`clear_vector_space_refusal`), then `initialize()` again. `drop()`
  re-provisions the container in the current embedding space and records this
  process's marker, so the second `initialize()` is the ordinary attach path and
  leaves a fully live instance — rather than a rebuild running against whichever
  half of `initialize()` completed before the refusal.
  The container is destroyed only after the operator confirms the rebuild.
- **The consistency check does not probe a refused target.** Probing it would
  report every graph record as missing: true of that container, and worthless —
  it reads as routine drift and buries the fact that the container is unusable
  and the rebuild is mandatory. The report carries the refusal under
  `incompatible` and `consistent` is `False`.

## The transition: adopting an unmarked container

Everything existing is unmarked, so the first start after the upgrade decides
the credibility of the whole gate. Three facts shape it:

1. Nothing is migrated and no data is rewritten. The marker is additive
   metadata, and an older LightRAG ignores it entirely — the change is
   downgrade-safe.
2. The dimension guards that already exist on `main` keep running, independently
   of the marker. A dimension change is still refused exactly as today.
3. The marker is absent, so no model-based refusal can fire (*absent evidence
   never refuses*) — and the container must therefore be **adopted**, or the
   silence never ends.

Adoption cannot be blind. An operator who upgrades **and** switches to a
same-dimension model in the same step would otherwise have the *new* model's
name stamped onto a container holding the *old* model's vectors — a lie recorded
permanently, after which the gate can never fire. That window is narrow and it is
exactly the failure this work exists to catch, so adoption carries evidence:

- **An empty container is adopted unconditionally.** There are no vectors to
  misdescribe.
- **A non-empty container is adopted only after a round-trip probe.** Take one
  record, re-embed its stored `content` with the current model, and compare
  against its stored vector by cosine similarity. Same model ≈ 1.0; a different
  model typically lands in 0.0–0.5, so the decision boundary is wide and neither
  quantization (`halfvec`) nor normalization moves it.

The probe runs **one layer up**, in `LightRAG.initialize_storages()`, not inside
a backend's `initialize()`. It needs a record carrying BOTH a `content` field and
its stored vector, and `BaseVectorStorage` has no enumeration API that would let
a backend find one in its own container. One layer up, the graph supplies the ids
(`get_popular_labels` → `compute_mdhash_id(name, prefix="ent-")`) and the
existing `get_by_ids` / `get_vectors_by_ids` supply the content and the vector.
That layer already needs graph access for the coverage gate below, so all
cross-storage evidence lives in one place.

`get_popular_labels` rather than `iter_labels`: it is abstract on
`BaseGraphStorage` so every backend implements it, while `iter_labels` fails
closed where it was never added. Ranking by degree also biases the sample towards
entities that certainly exist — the base contract warns that a backend MAY
surface a label with edges but no node document, and such an artifact would never
have been embedded.

**One probe per startup, and it adopts only what it probed.** The three vector
storages share an `embedding_func` but **not a history**: `lightrag-rebuild-vdb`
rebuilds entities, relationships and chunks as separate steps and offers
*entities + relationships* as a partial target, so an interrupted rebuild after a
same-dimension model change leaves entities in the current space while the others
still hold the previous model's vectors. Adopting those on the entity verdict
would record a lie that every later start believes and the probe itself sees no
conflict in — worse than never detecting anything. Relationships and chunks stay
unmarked until they can be probed with their own samples. After a marker is
successfully written, the cost is zero forever.

### When the probe cannot answer

The governing invariant:

> **Only a probe that ran and returned a negative verdict may refuse. Every form
> of "it could not run" falls back to the pre-upgrade behaviour.**

| outcome | action |
| --- | --- |
| probe ran, cosine high | write the marker; the silence ends |
| probe ran, cosine low | `VectorSpaceMismatchError` — fail closed, on the first start |
| embedding raised (provider down, auth, quota, network) | **no refusal, no marker**; log a warning, serve normally, retry next start |
| embedding timed out | same |
| no usable sample (empty graph, ids absent from the vdb, backend returned no vector) | same |
| marker write failed (read-only account, `index.blocks.write`, `collMod` denied) | same |

So **an embedding failure never prevents startup.** It only leaves the container
unmarked, which is the detection capability LightRAG has today — no worse than
before the upgrade, with the benefit deferred to a later start.

The probe must carry an **explicit timeout** (`asyncio.wait_for`). The embedding
providers wrap their calls in `@retry(stop_after_attempt(3),
wait_exponential(min=4, max=10))` (e.g. `lightrag/llm/openai.py`), so an
unreachable provider can burn 30–60s per call. Without a bound, a safety feature
becomes an availability cost; a timeout is treated as "inconclusive" like every
other non-answer.

A useful side effect: `lightrag-rebuild-vdb`'s check-only mode installs a stub
embedding function that raises deliberately, so it lands in "inconclusive" and
neither misjudges nor wrongly refuses.

## The coverage gate, and where it lives

A vector storage is an index. The invariant it owes its source is **coverage**:
if the source is not empty, the index must not be empty either. That is a
different question from the one the marker asks — the marker asks about
*identity*, which embedding space the stored vectors belong to — and neither
substitutes for the other.

The coverage question was reached from the identity side, which is why earlier
drafts of this document described it as a fallback for model changes. Milvus,
Qdrant and PostgreSQL land a model change on a *new, empty* container, so no
marker can ever fire there — the new container is genuinely theirs and correctly
named. The open question was how to detect that shape: enumerate sibling
containers per backend, or ask the cross-storage question, *the vector store is
empty while the graph is not*.

**Settled: the cross-storage form, one layer up.** Reasons, in order of weight:

- **It self-clears.** Once the rebuild populates the container the question
  answers itself. Sibling enumeration does not: the stale sibling is still there
  after a successful rebuild, so the signal has to be cancelled by something
  else — and `drop()` cannot cancel it, since a dropped-and-re-provisioned
  container is indistinguishable from a freshly created one.
- One implementation instead of three, with no per-backend enumeration
  (`list collections` / `information_schema`) and its own permissions and naming
  assumptions.
- It also catches what a marker cannot: a deleted vector file, a container
  emptied out of band, an interrupted rebuild.

### Coverage is the gate's own question, not a stand-in for identity

A model change on a named-container backend is one cause of an uncovered index,
and it is the cause that motivated the check — but it is an identity failure
surfacing as a coverage symptom, and reading the gate as "the model-change
detector for Milvus, Qdrant and PostgreSQL" understates it in one direction and
overstates it in the other.

**Understates:** the causes below are the gate's alone. No marker and no
recorded model can see any of them.

| cause | what the marker sees | what a recorded model sees |
| --- | --- | --- |
| container dropped, volume not mounted | nothing — there is no container to read | nothing — the record still matches |
| rebuild stopped between two of its three steps | nothing | nothing |
| vector backend switched without migrating | nothing — the new backend is empty and unmarked | nothing — the model did not change |
| workspace prefix does not match the vectors' | nothing | nothing |
| no record established yet (legacy deployment) | nothing to compare | absent evidence never refuses |

**Overstates:** as a model-change detector the gate is only half a check. It
fires when the change lands on an *empty* container. It cannot fire when the
change lands on a container that is populated but stale — switching from model A
to B and back to A reuses `entities_a_1024d`, which still holds the corpus as it
stood when the deployment switched away. Nor can it see a folding collision
(`_generate_collection_suffix` lowercases and replaces every non-alphanumeric
character, so `text-embedding-3-large` and `text_embedding_3_large` name one
container), a deployment with no `model_name` at all, or the legacy-container
migration relabelling foreign vectors on a same-dimension change.

Those four need an embedding space that is *recorded* per workspace, outside the
vector containers and ahead of their `initialize()`. That record is tracked in
[#4006](https://github.com/HKUDS/LightRAG/issues/4006) and does not replace this
gate; the two are ANDed, and each refuses on evidence the other cannot obtain.

### One rule, three pairings

A vector storage is an **index** over data held somewhere else, so whether its
being empty is a defect depends on that other data — which is why the question
cannot be answered inside a backend's `initialize()`. There are three such
pairings, and the rule is applied to each independently:

| source data | index | how the source is read |
| --- | --- | --- |
| `text_chunks` (KV) | `chunks_vdb` | `BaseKVStorage.is_empty()` |
| graph entities | `entities_vdb` | `get_popular_labels(limit=1)` |
| graph relations | `relationships_vdb` | first batch of `iter_edges(batch_size=1)` |

**If the indexed data is not empty, its index must not be empty either.**

Per-pairing rather than one global check, because each index is a separate
container with its own write path and its own ways of being lost, and because
**no pairing substitutes for another**:

- A workspace built entirely through `acreate_entity` / an entities-only
  `ainsert_custom_kg` has graph entities and **no text chunks**. Its empty chunk
  container is correct.
- A corpus that produces text chunks but extracts **no entities** (tables,
  numbers, a very small corpus) leaves the graph empty while `chunks_vdb` is
  populated.

The second case is what the first implementation got wrong: it checked only the
entity pairing, and its "the graph has no entities" early return ended the
*whole* function, so `naive` / `mix` retrieval silently served no chunk context
after a model change. An empty graph now exempts only the two pairings whose
source *is* the graph.

Relations add no detection coverage that entities do not already give — an edge
implies its endpoint nodes — but they are a separate container, so an
interrupted rebuild (`lightrag-rebuild-vdb` rebuilds entities, relationships and
chunks as separate steps) can leave `relationships_vdb` empty while
`entities_vdb` is populated.

### What makes an uncovered index a defect

"The vector store is empty while the graph is not" is *not* by itself a
refusal, and the first implementation that treated it that way broke two
existing pipeline tests — correctly. A graph ahead of its vector store is a
routine, self-healing residue: an ingest interrupted before its vector flush, a
batch that failed and will be retried, a graph built through the admin API.
`AGENTS.md` *Consistency without transactions* says such a state must not be
escalated, and escalating it here refuses to start the very process whose next
run repairs it.

A backend that declares `persists_vectors = False` (`NoopVectorDBStorage`) is
never judged at all: its reads are misses BY DESIGN, so every question here has
a known, meaningless answer. Graph-only ingestion is a supported configuration,
and after its first document the graph holds entities and doc-status holds a
`PROCESSED` row — without this the gate refuses every restart of it, and
`rebuilding_vector_storage` is not the answer, because such a deployment is not
rebuilding anything. Same capability `lightrag-rebuild-vdb` already reads.

A second exemption is **declared, not inferred**. An in-process rebuild BEGINS
from the state the gate refuses, and so does the supported switch from
`NoopVectorDBStorage` (graph-only ingestion, which writes no vectors by design)
to a real vector backend. `lightrag-rebuild-vdb` never reaches this check — it
drives the storages directly — but a program rebuilding through a `LightRAG`
instance does, so it says so with `rebuilding_vector_storage=True`.
Constructor-only, with no environment variable: a fail-closed check whose
bypass can be exported in a shell is one an operator silences at 3am and never
revisits, and what it silences is a deployment that answers every vector query
with nothing. Declaring it in code keeps the claim attached to the program that
makes it true.

The discriminator for everything else is **doc-status**: the gate refuses only
when **nothing in the workspace is unfinished**. Anything `PENDING`, `PARSING`,
`ANALYZING`, `PROCESSING` or `FAILED` means the graph-ahead residue has an owner
and heals by being retried. Consulted only when a pairing is about to refuse,
and at most **once** however many pairings reach that branch, so the healthy
path pays nothing for it.

Two situations satisfy that, and the vectors are missing in both by defect:
every document finished, or **there are no documents at all**. The second is not
an oversight — `acreate_entity` and `ainsert_custom_kg` write graph entities AND
their vectors while writing no doc-status row, so requiring a `PROCESSED`
document would exempt an admin-built workspace forever. It is also the one place
where "a later pipeline run repairs it" is simply false: no pipeline run will
ever recreate objects an operator created by hand.

Counting only the *unfinished* states is also what keeps a `PROCESSED` row
elsewhere from being read as evidence about *this* container. That mattered more
under the old sampling design, where an ingest that crashed after writing a batch
of well-connected nodes but before their vector upserts filled a degree-ranked
sample with rows that never had vectors. `is_empty()` narrows it a great deal —
any previously written vector makes the container non-empty, so only a first
ingest crashing mid-way produces the shape at all — but the discriminator is
kept, because that first-ingest case is real and heals by being retried.

**The count is workspace-wide; what a retry rewrites is per-object.** An
unfinished document only ever rewrites the objects *it* produces. So before the
exemption is applied to a graph-sourced pairing, the gate samples graph objects
(node payloads via `get_nodes_batch`, edge payloads from `iter_edges`) and
follows each one's own trail:

    source_id → text_chunks → full_doc_id → doc_status

If any sampled object is **not** owned by an unfinished document, no retry will
restore its vector and the refusal stands.

The trail replaced a cheaper test — "does `source_id` look like a real chunk id,
or like the `manual_creation` placeholder?" — which was wrong twice over:

- `acreate_entity` does stamp the placeholder, so that test caught it;
- but **`ainsert_custom_kg` maps its entities onto the call's own chunks**, so
  their `source_id` is a real `chunk-*` id while the chunk's `full_doc_id` names
  no doc-status row at all. The shape test read that as document-produced and
  handed it the exemption it must not get.
- and it never reached a third case: an object produced by a document that has
  **finished**, in a workspace where some *other* document is unfinished. That
  document's retry does not rewrite this object either.

Following the trail settles all three, because it asks the actual question
instead of inferring it from the shape of an id.

**The chunk pairing asks the same question from the other end.** `text_chunks`
can say it is non-empty but not *which* rows it holds — `BaseKVStorage` has no
enumeration API — so there is nothing to sample. The question is inverted
instead: ask doc-status for chunks a retry will **not** rewrite (one bounded
page of `PROCESSED` documents, hydrated for their `chunks_list` through
`get_full_docs_by_ids`, which documents exactly this page-then-hydrate pattern),
then confirm through the same trail that those chunks really are in
`text_chunks`. Both reads are strict, so an incomplete answer raises rather than
under-reporting into a refusal.

Without it that pairing kept the raw workspace-wide count while the graph
pairings had moved on, so an unrelated `PENDING` row excused an empty
`chunks_vdb` even when every chunk in the store belonged to a finished
document — and `naive` / `mix` then serve no context at all, permanently once
the pending document writes one chunk vector.

**An empty chunk read is "cannot tell", not "orphan".** `BaseKVStorage.get_by_ids`
catches its transport errors and returns an empty list, so a blip and a genuinely
absent chunk arrive as the same value. Reading it as "no document owns this"
would refuse a healthy deployment — the same defect `is_empty()` exists to avoid
on the vector side. It is read as healable instead, which costs a miss rather
than a false refusal. The custom-KG case is unaffected: those chunks *exist* and
name a `full_doc_id` that doc-status has never heard of.

**Sampling is sound here, unlike for emptiness**, and the asymmetry is the whole
reason one is sampled and the other asked. A sample that misses the documentless
object answers "no", which only declines to refuse — the behaviour without the
check at all. It can add refusals for what it finds, never remove one.
`is_empty()` had the opposite exposure: there a wrong answer *causes* a refusal,
which is why it had to be asked of a read that fails loudly. A probe that cannot
run lands on the same side as a miss.

This narrows the mixed state but does not close it: once the unfinished document
completes and writes even one vector the container is no longer empty, and any
still-missing vectors become invisible to this gate. That is the partial-loss
residue below, not a second defect.

The count comes from `count_docs_by_statuses(strict=True)`, never
`get_status_counts()`. The latter is documented to swallow its errors and return
what it managed to collect — `RedisDocStatusStorage` catches a mid-`SCAN`
failure and returns the partial counts — so it could show a finished workspace
while missing the `PENDING` row that explains everything. Strict counting raises
instead, and a raise answers "do not refuse", like every other unreadable thing
this module consults.

### How "empty" is measured: a read that fails loudly

`BaseVectorStorage.is_empty()`, implemented on all eight in-tree backends.

The first implementation **inferred** emptiness instead: 32 entity ids from
`get_popular_labels`, one batched `get_by_ids`, refuse if not one had a row.
That could not work, and the reason is worth keeping written down because it
constrains any future read the gate might use.

**Every server-backed `get_by_ids` — Milvus, Qdrant, PostgreSQL, MongoDB,
OpenSearch — catches its transport errors, logs them, and returns an empty
list.** So "the container is empty" and "the cluster blinked" arrive as the
*same value*, and no `try`/`except` around the call separates them. A confirming
second read narrowed the window but could not close it: a blip spanning both
reads still refused, and the advice a refusal gives — rebuild — is destructive.

`is_empty()` closes it by inverting the contract, and this is the whole point of
adding a method rather than reusing a reader:

> Return `True` **only** when the container was positively read and found
> empty. **Raise** on any backend failure.

That is the opposite of `BaseKVStorage.is_empty()`, which catches its errors and
answers `True` — and the asymmetry is deliberate, because the two are used in
opposite directions:

- On the **index** side, `True` can refuse a deployment, so an error reported as
  "empty" is a false outage. It must raise.
- On the **source** side, `True` only *skips* a check, so an error reported as
  "empty" costs nothing. The existing KV behaviour is therefore fine as it is,
  and the gate depends on it: an unreadable `text_chunks` lands on the skip
  branch.

The base-class default raises `StorageCapabilityError`, the fail-closed pattern
already used by `iter_labels` / `iter_edges`: a backend that has not implemented
this is one the gate cannot question, which is where every backend stood before
the gate existed. It must never be read as an answer. `NoopVectorDBStorage`
keeps that default *deliberately* — it really does hold nothing, and answering
so accurately would be a false refusal the day a caller forgets to check
`persists_vectors` first.

Each implementation counts a pending upsert as non-empty (the buffered rows are
real) and does **not** subtract pending deletes: reporting a store as non-empty
costs a check that would have found nothing, while the reverse costs a startup.

Two consequences of dropping the sample:

- The gate no longer has a sampling-probability argument to make, and the
  "a store that lost 99% of its rows is caught about half the time" partial
  detection is gone with it. `is_empty()` answers only the total case — which is
  the case a model change on a named-container backend actually produces.
- `SAMPLE_SIZE` and the positional-`None` handling (`NanoVectorDBStorage`
  returns one entry per requested id, with `None` for a miss, while most
  backends return a compacted list) now belong to the **adoption probe** alone,
  which still needs real rows carrying a `content` field and a stored vector.

### What the probe's verdict bands mean

The probe re-embeds stored `content` and compares it with the vector stored
beside it. Three outcomes, not two:

**Every compared record must agree, and one record is not enough.** A container
can hold two embedding spaces at once, by a route this design itself opens: an
upgrade that also switches to a same-dimension model starts *unmarked* when the
probe cannot run (absent evidence never refuses), keeps serving, and then takes
new writes — so the legacy rows and this model's rows share one container.
`test_writing_to_an_unverified_legacy_store_does_not_certify_it` already
constructs that shape; what it pins is only that the *save* does not stamp it.

Adopting on whichever row the sample reached first would stamp this model over
the foreign half and hide it permanently — the same lie as adopting a store
nobody probed, one level down. So the probe compares up to `PROBE_ROWS` (8)
records in **one batched embedding call** and adopts only if all of them
reproduce. A record that fails while others pass is not an ambiguity: it is
positive evidence that the container is not homogeneous, and it refuses saying
so. The governing rule is untouched — a reproduced cosine of 0.3 is a negative
verdict whoever else agrees.

Residue, since 8 records cannot prove a whole container: a container whose
foreign rows are a small enough minority can still be sampled entirely from the
majority and adopted. The degree-ranked sample leans the useful way in the
common case — the legacy entities are the well-connected, long-standing ones
and the new writes are the newcomers — but a container that is overwhelmingly
new with a few legacy stragglers can still be stamped. Proving a whole
container means re-embedding all of it, which is a rebuild, and
`lightrag-rebuild-vdb` already is that.

| cosine | verdict |
| --- | --- |
| ≥ `ADOPT_COSINE` (0.95) | adopt — the marker is recorded |
| ≤ `REFUSE_COSINE` (0.70) | refuse — `VectorSpaceMismatchError` |
| between the two | **inconclusive** — no marker, no refusal, logged |

The band exists because the governing invariant is *only a probe that returned
a **negative** verdict may refuse*, and a middling cosine is not one. The same
model re-embedding the same text lands at ~1.0 and a different model at 0.0–0.5,
so the band is empty in practice; what it protects against is the cases where
it is not — an unusually short `content`, a provider that silently truncated, a
fine-tune of the same base model. Failing a live deployment closed on that
evidence is worse than leaving the container where it already was.

Two further rules the module keeps:

- **A comparison that cannot be made is `None`, never `0.0`.** A length
  mismatch, a zero-magnitude vector or a non-finite value all mean the
  comparison says nothing — and `0.0` is the strongest possible evidence of a
  changed model, so spelling it that way would refuse on silence.
- **Nothing but the two typed refusals escapes.** An unreachable graph backend,
  a vector store that errors on a read, a marker that cannot be written: none is
  evidence about the embedding space, and none was a startup failure before this
  check existed.

### The adoption surface

Two optional methods on `BaseVectorStorage`, both defaulting to the answer the
named-container backends need (`False`), so Milvus, Qdrant and PostgreSQL are
untouched:

- `vector_space_adoption_pending()` — "this container records no model, AND
  this process can name one". Answered from state `initialize()` already
  computed, so it costs no round trip, and a process with no `model_name`
  answers `False`: it has nothing to record, so inviting a probe would buy
  evidence nobody can act on. An *empty* container never reports pending —
  Nano and FAISS mark it themselves, without evidence, because there are no
  vectors to misdescribe.
- `adopt_vector_space()` — records the marker. **MUST NOT raise.** A read-only
  account, a denied `collMod`, a full disk: all leave the container unmarked,
  which is where every container was before this feature existed. Returning
  `False` costs one more probe next start; raising would turn a safety feature
  into an outage.

`initialize_storages()` marks the storages `INITIALIZED` **before** running
either check. They have all completed `initialize()` by then and hold clients,
pools and locks, and `finalize_storages()` skips the entire teardown unless the
status says so — a caller that catches a refusal (to report it, or to go and
rebuild) would otherwise leak every one of them, and a retry on the same object
would initialize them twice.

It probes `entities_vdb` because the graph's own ids address it directly, and
adopts **only that store** — see *The transition* above for why a store nobody
probed must stay unmarked. Probing relationships and chunks needs their own
samples (graph edges, and `text_chunks` ids) and is deliberately left to a
follow-up rather than approximated. Note that this is the one place the three
pairings of the coverage gate do *not* extend to: emptiness is answerable
per container, but provenance needs a readable row from the container itself.

**A refusal is sticky, and does not mean "not initialized".** The storages are
all up when either check refuses, so `_storages_status` says `INITIALIZED` and
`finalize_storages()` — which skips its whole teardown otherwise — can release
their clients, pools and locks. The refusal is remembered on the instance and
re-raised by any later `initialize_storages()`, because that method returns
early once initialized and would otherwise come back *successful without
re-running the check*, turning a fail-closed gate into a one-shot one.
`check_lightrag_setup` reports a refused instance as not ready for the same
reason.

## Accepted residues

Per *Consistency without transactions* in `AGENTS.md`, each of these is a
decision with a recovery path, not an oversight.

**Partial vector loss is invisible to this gate.** `is_empty()` answers only the
total case: a container that lost 40% of its rows reads as non-empty and passes,
serving degraded retrieval silently. This is a real cost of asking rather than
sampling — the old 32-id sample caught a store that had lost 99% of its rows
about half the time — and it is accepted because the causes this gate is the
only evidence for (a container dropped, a volume not mounted, a rebuild stopped
between steps, a backend switched without migrating, a model change landing on
a new container) produce a *completely* empty container, never a partial one.

The right home for the partial case is the offline
`lightrag.tools.kg_integrity_repair` audit, which already enumerates the whole
graph and already reports rather than refuses. It is deliberately not a startup
check, for three reasons that compound:

- **The counts are not supposed to be equal.** *Consistency without transactions*
  in `AGENTS.md` explicitly accepts residues where one store retains an object
  another dropped, and `PurgeRecoveryContract.md` enumerates them for merge and
  rename. A count check would convert every documented accepted residue into a
  startup refusal — contradicting a design decision rather than tightening one.
- **Nothing can count.** `BaseKVStorage` has no count (only `get_by_id(s)` /
  `filter_keys` / `is_empty`), `BaseGraphStorage` has no node or edge count
  (`get_all_labels` / `get_all_edges` materialize everything), and
  `BaseVectorStorage` has none either. Supplying them means a full graph scan on
  every startup — precisely what `kg_integrity_repair` documents as
  "deliberately OFFLINE-only; the ingestion/retry/delete/scan hot paths never do
  this".
- **A count mismatch is not a negative verdict** about the embedding space, which
  is the only thing the governing invariant permits a refusal on. Three rows out
  of 200,000 is noise, and separating noise from loss needs a threshold — a
  tunable knob on a fail-closed gate is the thing that gets silenced.

An audit can afford exact counts, can attribute a shortfall to specific documents
through `source_id` → `text_chunks` → `full_doc_id`, and its output is advice
rather than an outage.

**Fold collision on Milvus / Qdrant / PostgreSQL.** The suffix lowercases and
folds punctuation, so two models whose names differ only in case or punctuation
share a container undetected. A *harmful* collision needs two genuinely
different models whose names differ only that way **and** which share a
dimension; in practice such name pairs are the same model spelled differently
by different providers or config files, which is benign. **Closed** for any
workspace with a recorded baseline: the per-target embedding baseline
(`docs/design/ConfigurationStorage.md`) stores the model name **unfolded**, so
the two spellings compare unequal at the precheck and the second refuses to
start. Still open for a workspace whose baseline is absent -- the first start
after the upgrade, or a probe that could not run -- until the record is
established. Recovery: `lightrag-rebuild-vdb`.

**No `EMBEDDING_MODEL` configured.** Milvus, Qdrant and PostgreSQL fall back to
an un-suffixed container (`qdrant_impl.py`, `milvus_impl.py`,
`postgres_impl.py` each log a warning today), which carries no model
information — so those deployments get no embedding-space gate at all, exactly as
today. On Qdrant this also means one un-suffixed collection can host tenants
running different models; point-id salting and the workspace filter still keep
them from reading or overwriting each other, so each tenant's own retrieval stays
correct. Recovery: set `EMBEDDING_MODEL` and run `lightrag-rebuild-vdb` — the
suffix then moves the workspace to a new, protected container.

**A model change that lands on a populated container on Milvus, Qdrant and
PostgreSQL** was undetected by this gate: the container name is derived from
the current configuration, so it can never contradict it, and the coverage
gate only fires when the change lands on an *empty* container. Switching from
model A to B (accepted through a rebuild) and back to A reused
`entities_a_1024d`, which is not empty, so the gate passed and retrieval was
silently stale and partial. **Closed** by the recorded per-target embedding
baseline (`docs/design/ConfigurationStorage.md`): the record moves only on a
successful rebuild, so the switch back is a mismatch, refused at step 3 of
startup -- ahead of the legacy-container migration those backends run inside
`initialize()`. What stays open is recorded in that contract as its own
residue: with no record yet (the first start after the upgrade), the
legacy-container copy may still happen before anything judges it, and a
`bootstrap_assumption` baseline on `relationships` or `chunks` can be wrong.
Recovery in both cases: `lightrag-rebuild-vdb`.

**Embedder unavailable during an adopting start, in the same upgrade as a
same-dimension model swap.** The probe cannot run, so the instance starts and
serves wrong results until a later start probes successfully. Strictly better
than today, where nothing ever detects it, but real. Recovery: the next start
with a reachable embedder refuses and names `lightrag-rebuild-vdb`.

**An interrupted rebuild** leaves a container that is empty but correctly marked,
so the marker does not refuse on the next start. The tool exits non-zero and says
so, and the coverage gate catches the state.

**A sidecar marker file diverging from its data** does not apply — no backend
uses a sidecar container (see *Where the marker lives*). FAISS's `.meta.json` and
Nano's JSON are the storage's own files, written by the same commit as the data.

## Backend rollout

Each row lands with its own change; a row is only true once that change is in.

| change | backend | work |
| --- | --- | --- |
| PR 2 | OpenSearch | marker in `_meta`; drop-capable while refused; fix the lost-`indices.create`-race attach that validates ownership but not compatibility |
| PR 3 | MongoDB | marker in the JSON Schema validator `description`; `drop()` must rewrite that description and rebuild the Atlas search index when the DIMENSION changed (the index definition records a dimension and nothing else, so a same-dimension model change leaves a usable index) |
| PR 4 / 5 | FAISS, Nano | marker in a `.space.json` sidecar / `additional_data`; move the refusal out of `__post_init__` so the object survives it and stays droppable |
| PR 6 / 7 / 8 | Milvus, Qdrant, PostgreSQL | no marker. Replace the legacy-path `DataMigrationError` with the typed refusal so the tool can tolerate it, and make a refused instance drop-capable (Qdrant and PostgreSQL assign `_flush_lock` *after* their init block, the same shape as the OpenSearch bug) — see *The named-container backends' refusal* |
| PR 9 | — | the coverage gate and the adoption probe in `LightRAG.initialize_storages()`, plus the two-method adoption surface on `BaseVectorStorage` and its implementations on the four marker backends — see *The coverage gate, and where it lives* |
