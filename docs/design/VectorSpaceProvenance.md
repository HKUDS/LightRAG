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

## What is recorded, and why a dimension is not enough

A dimension is not an identity. `text-embedding-3-small` at 1536 and a local
model at 1536 produce unrelated spaces, and a same-dimension swap is invisible to
every dimension check. So the **model name is recorded next to the vectors** —
never inferred.

`lightrag/kg/vector_space.py` owns the three things every backend needs:

| helper | what it fixes |
| --- | --- |
| `declared_model_name(embedding_func)` | the single definition of "this instance's model": `str`, `strip()`ed, non-empty, otherwise `None` |
| `vector_space_marker(embedding_func)` | the payload written when a container is provisioned |
| `assert_vector_space_matches(...)` | the verdict, so seven backends cannot drift apart on it |

The recorded name is **unfolded** — exactly as configured. The collection-name
suffix (`BaseVectorStorage._generate_collection_suffix`) lowercases and folds
every non-alphanumeric character to `_`, so `text-embedding-3-large` and
`text_embedding_3_large` produce the same suffix and land in the same container.
Recording the folded name would import that blind spot into the marker, leaving
the suffixed backends with no way to tell those two models apart.

### Absent evidence never refuses

A container that records no model, or no dimension, predates the marker.
Silence must not read as a mismatch, or every index written before this feature
is refused on the first start after the upgrade.

The rule is symmetric: a process whose `embedding_func` carries no `model_name`
does not know what it is either, and cannot contradict a recorded name. A marker
payload that cannot be parsed reads as absent for the same reason.

The direct consequence is that **silence never ends by itself**, which is why
each backend also needs a documented decision about what to write onto existing
unmarked containers. A backfill is what converts "unknown forever" into "known
from here".

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
  half of `initialize()` happened to complete before the refusal.
  The container is destroyed only after the operator confirms the rebuild.
- **The consistency check does not probe a refused target.** Probing it would
  report every graph record as missing: true of that container, and worthless —
  it reads as routine drift and buries the fact that the container is unusable
  and the rebuild is mandatory. The report carries the refusal under
  `incompatible` and `consistent` is `False`.

Accepted residue: an interrupted rebuild leaves a container that is empty but
correctly marked, so the next start does not refuse on the marker. The tool exits
non-zero and says so; the layer gate below is what catches the empty container.

## The suffixed backends, and where the gate lives

Milvus, Qdrant and PostgreSQL already land a model change on a *new, empty*
container, so a marker alone never fires there — the new container is genuinely
theirs. The issue's open question was how to detect that shape: enumerate sibling
containers per backend, or ask the cross-storage question, *the vector store is
empty while the graph is not*.

**Settled: the cross-storage form, one layer up.** Reasons, in order of weight:

- **It self-clears.** Once the rebuild populates the container the question
  answers itself. Sibling enumeration does not: the stale sibling is still there
  after a successful rebuild, so the signal has to be cancelled by something
  else — and `drop()` cannot cancel it, since a dropped-and-re-provisioned
  container is indistinguishable from a freshly created one.
- One implementation instead of three, and no per-backend enumeration
  (`list collections` / `information_schema`) with its own permissions and
  naming assumptions.
- It also catches what the marker cannot: a deleted vector file, a container
  emptied out of band, an interrupted rebuild.

So the suffixed backends carry **provenance only**, and the empty-while-populated
gate lives above the storage layer. The marker still earns its place there — it
is what catches two models whose names fold to the same suffix.

## Backend rollout

Provenance marker homes, one per backend. Each lands with its own change; a row
is only true once that change is in.

| backend | marker home |
| --- | --- |
| OpenSearch | index mapping `_meta`, alongside the existing workspace identity |
| MongoDB | a reserved marker document in the vector collection |
| FAISS | a reserved key in the `.meta.json` sidecar |
| Nano | `additional_data` in the vdb JSON file |
| Milvus | the collection description |
| Qdrant | a reserved marker point, carrying no `workspace_id` so no query can see it |
| PostgreSQL | the table comment |
