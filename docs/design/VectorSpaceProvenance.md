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
| FAISS | a reserved key in the `.meta.json` sidecar | not in the `.index` file, so `index.search()` cannot return it |
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
a backend's `initialize()`. `BaseVectorStorage` has no enumeration API — only
`get_by_ids(ids)` — so a backend cannot obtain a sample of its own records
without eight new methods. One layer up, the graph supplies the ids
(`iter_labels` → `compute_mdhash_id(name, prefix="ent-")`) and the existing
`get_by_ids` / `get_vectors_by_ids` supply the content and the vector. That layer
already needs graph access for the empty-container gate below, so all
cross-storage evidence lives in one place.

**One probe per startup.** The three vector storages share one
`embedding_func`, so probing `entities_vdb` settles the question for all three.
After a marker is successfully written, the cost is zero forever.

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

## The empty-container gate, and where it lives

Milvus, Qdrant and PostgreSQL land a model change on a *new, empty* container, so
no marker can ever fire there — the new container is genuinely theirs and
correctly named. The issue's open question was how to detect that shape:
enumerate sibling containers per backend, or ask the cross-storage question,
*the vector store is empty while the graph is not*.

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

## Accepted residues

Per *Consistency without transactions* in `AGENTS.md`, each of these is a
decision with a recovery path, not an oversight.

**Fold collision on Milvus / Qdrant / PostgreSQL.** The suffix lowercases and
folds punctuation, so two models whose names differ only in case or punctuation
share a container undetected. Accepted because a *harmful* collision needs two
genuinely different models whose names differ only that way **and** which share a
dimension; in practice such name pairs are the same model spelled differently by
different providers or config files, which is benign. Recovery:
`lightrag-rebuild-vdb`.

**No `EMBEDDING_MODEL` configured.** Milvus, Qdrant and PostgreSQL fall back to
an un-suffixed container (`qdrant_impl.py`, `milvus_impl.py`,
`postgres_impl.py` each log a warning today), which carries no model
information — so those deployments get no embedding-space gate at all, exactly as
today. On Qdrant this also means one un-suffixed collection can host tenants
running different models; point-id salting and the workspace filter still keep
them from reading or overwriting each other, so each tenant's own retrieval stays
correct. Recovery: set `EMBEDDING_MODEL` and run `lightrag-rebuild-vdb` — the
suffix then moves the workspace to a new, protected container.

**Embedder unavailable during an adopting start, in the same upgrade as a
same-dimension model swap.** The probe cannot run, so the instance starts and
serves wrong results until a later start probes successfully. Strictly better
than today, where nothing ever detects it, but real. Recovery: the next start
with a reachable embedder refuses and names `lightrag-rebuild-vdb`.

**An interrupted rebuild** leaves a container that is empty but correctly marked,
so the marker does not refuse on the next start. The tool exits non-zero and says
so, and the empty-container gate catches the state.

**A sidecar marker file diverging from its data** does not apply — no backend
uses a sidecar container (see *Where the marker lives*). FAISS's `.meta.json` and
Nano's JSON are the storage's own files, written by the same commit as the data.

## Backend rollout

Each row lands with its own change; a row is only true once that change is in.

| change | backend | work |
| --- | --- | --- |
| PR 2 | OpenSearch | marker in `_meta`; drop-capable while refused; fix the lost-`indices.create`-race attach that validates ownership but not compatibility |
| PR 3 | MongoDB | marker in the JSON Schema validator `description`; `drop()` must drop and recreate the Atlas search index, and rewrite the description |
| PR 4 / 5 | FAISS, Nano | marker in `.meta.json` / `additional_data`; move the refusal out of `__post_init__` so the object survives it and stays droppable |
| PR 6 / 7 / 8 | Milvus, Qdrant, PostgreSQL | no marker. Replace the legacy-path `DataMigrationError` with the typed refusal so the tool can tolerate it, and make a refused instance drop-capable (Qdrant assigns `_flush_lock` *after* its init block, the same shape as the OpenSearch bug) |
| gate | — | the empty-container gate and the adoption probe in `LightRAG.initialize_storages()` |
