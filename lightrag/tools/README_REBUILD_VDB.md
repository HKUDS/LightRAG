# Offline Vector Storage (VDB) Rebuild Tool

`lightrag-rebuild-vdb` restores consistency between LightRAG's authoritative
data sources and its vector storages by dropping each vector storage and
rebuilding it from scratch:

| Vector storage | Authoritative source |
|---|---|
| `entities_vdb` | graph nodes |
| `relationships_vdb` | graph edges |
| `chunks_vdb` | `text_chunks` KV store |

## When do I need this?

LightRAG performs multi-step writes (graph + vector storage). If a vector
storage write fails at runtime — embedder outage, network timeout, context
overflow on a high-degree entity — the graph and the vector storage drift
apart. The most common symptom is the edge-count drift reported in issue
[#2917](https://github.com/HKUDS/LightRAG/issues/2917): graph edges with no
vector counterpart, so `local`/`hybrid` queries miss relations that exist in
the graph.

Since v(next), `amerge_entities` raises `VectorStorageConsistencyError` when
this happens, with a pointer to this tool. No data is lost in this situation:
the graph and the `text_chunks` KV store hold everything needed to rebuild
the vectors.

A full drop + rebuild also clears *reverse* orphans (records present in the
vector storage but absent from the graph), which incremental repair cannot
reliably do.

You can also use this tool after changing the embedding model or embedding
dimension. Existing vector records were generated in the old embedding space
(and some vector backends bind collections/tables to the configured dimension),
so run the tool with the updated `.env` and choose **Rebuild ALL vector
storages** to regenerate `entities_vdb`, `relationships_vdb`, and `chunks_vdb`
from the authoritative graph/KV sources. The consistency check is not enough
for this case because it only compares which records exist, not whether an
existing vector was embedded with a previous model or dimension.

A vector storage whose container was written in a different embedding space
**refuses to attach** rather than serving an empty or foreign index. That
refusal is the condition this tool clears, so it is the one startup failure the
tool tolerates: the three vector targets are initialized individually, a typed
`VectorSpaceMismatchError` is recorded instead of aborting the run, and the
rebuild opens by dropping that container and re-provisioning it in the current
embedding space. No out-of-band step (deleting the index through the backend's
own API, `DROP TABLE`, `rm`) is needed.

A **corrupt file-backed vector snapshot** (NanoVectorDB, Faiss) is tolerated
the same way, because a rebuild is its recovery too: the target shows as
corrupt in the menu, and once you confirm the rebuild the tool first copies
every file of it to `<file>.corrupt-<token>` beside the original, then drops
and rebuilds the index. The copies are private to the user running the tool
(mode `0600`) and are never deleted automatically. They hold the same
document-derived content the index did, so remove them yourself once the
rebuilt index is confirmed. If a copy cannot be made, nothing is changed and
the message names the files to move aside by hand.

Everything else still aborts. A cluster outage, a bad credential or a corrupt
authoritative source is not a model change, and the recovery for a model change is destructive —
so it is only ever applied to a refusal that says so in its type. The
authoritative sources (graph storage and `text_chunks`) likewise keep the
server-identical startup path, migrations included: rebuilding vectors from a
half-migrated source is worse than not rebuilding.

See `docs/design/VectorSpaceProvenance.md` for the provenance marker and the
rules that decide when a backend refuses.

The tool also completes a graph-only backfill performed with
`NoopVectorDBStorage`. Stop every writer, preserve the same working directory,
workspace, graph storage, and KV storage, switch to the intended persistent
vector backend, configure the production embedding model and dimension, and
select **Rebuild ALL vector storages**. Create a new `LightRAG` instance or
restart the process after changing the vector backend. Do not start the server
until the rebuild succeeds; there is currently no persisted vector-index
readiness marker.

If the workspace's data is disposable -- a test corpus, a workspace being
decommissioned -- a rebuild is wasted embedding cost. `lightrag-clear-storage`
(see `README_CLEAR_STORAGE.md`) drops the workspace instead, which also clears
the startup refusal.

## Usage

```bash
# Stop the LightRAG Server first!
lightrag-rebuild-vdb
# or
python -m lightrag.tools.rebuild_vdb
```

The tool reads the same `.env` / environment configuration as the server
(`LIGHTRAG_GRAPH_STORAGE`, `LIGHTRAG_VECTOR_STORAGE`, `LIGHTRAG_KV_STORAGE`,
`WORKSPACE`, `WORKING_DIR`, `EMBEDDING_*`, backend connection settings) and
builds its embedding function through the exact factory the server uses —
run it with the same `.env` so rebuilt vectors live in the same embedding
space the server queries against. Without `--workspace`, `WORKSPACE` is normalized
as the server normalizes it (every character outside `[A-Za-z0-9_]` becomes `_`).
To select an SDK workspace exactly, use `lightrag-rebuild-vdb --workspace team-a`.
The explicit name is validated but never rewritten; `--workspace ""` selects the
empty/default workspace even when `WORKSPACE` is set. The selected name applies
to every source, vector target, configuration baseline and directory claim. With
JSON configuration storage the workspace must already be registered: start
the server once for a new workspace, stop it, then run the tool. The server
can register only a name its normalization leaves unchanged; an exact SDK name
such as `team-a` is registered by a library start instead (`LightRAG(...,
workspace="team-a")`, `initialize_storages()`, `finalize_storages()`).
`--workspace` is the tool's only option: every other setting, the embedding
configuration included, comes from `.env` / the environment, and any other
flag is refused.

Menu options:

1. **Consistency check (diagnose only)** — reports, per vector storage, whether
   it matches its source; see [Consistency check](#consistency-check). Run
   this first to decide whether a rebuild is worth the embedding cost. The
   check issues read queries only and does not run a rebuild (no drop +
   re-embed). It is
   **not** strictly side-effect-free, though: the tool initializes every
   storage on startup — exactly as the server does — and for some backends
   that includes schema/DDL setup and one-time legacy migrations (e.g. Qdrant
   upserts into the new collection, PostgreSQL batch-inserts into the new
   table, Milvus may create a temp collection and drop/rename the original).
   Treat running the tool — even just for a check — like starting the server:
   stop other writers first.
2. **Rebuild entities + relationships VDB** — sufficient for the #2917
   merge-failure scenario.
3. **Rebuild chunks VDB**.
4. **Rebuild ALL vector storages** — use this after changing the embedding
   model or embedding dimension.

## Important notes

- **Stop the server first.** The tool drops and rewrites vector storages;
  concurrent writers (any backend, not just file-based ones) can corrupt
  data or lose updates.
- **Embedding model/dimension changes.** Run the tool with the new embedding
  configuration and rebuild all vector storages. A consistency check can still
  pass when every vector record exists but was created with the old embedding
  model or dimension — unless the backend recorded its embedding-space
  provenance, in which case the check reports the refusal and says the rebuild
  is mandatory rather than reporting every record as missing.
- **The previous model's container is cleared once the rebuild succeeded.**
  Qdrant, PostgreSQL and Milvus name the collection / table after the embedding
  model, so a rebuild under a new model writes into a new container and leaves
  the old one behind. While a target is being rebuilt that old container is
  your fallback: if the rebuild fails, restore the previous embedding
  configuration and the server starts on it. Once the target's baseline is
  recorded the server refuses the previous model, so the tool then removes
  this workspace's data from every other model's container for that target.
  If that step cannot complete it prints a warning and the rebuild still
  counts as successful; the next rebuild or workspace clear removes the rest.
  On Milvus a rebuild also drops the legacy (unsuffixed) collection and the
  `_old` / `_temp` backups of a schema migration, as it already did on Qdrant
  and PostgreSQL: after a rebuild there is no going back to a LightRAG version
  from before model-named containers without re-embedding.
- **A refused target is dropped only after you confirm the rebuild, and only
  once its source has been read in full.** Choosing the consistency check never
  destroys anything, so a refusal can be diagnosed before it is acted on; and a
  rebuild whose source fails to read leaves the refused container in place, so
  reverting the embedding configuration still brings it back. Options 2–4
  recover exactly the targets they rebuild: picking option 2 leaves a refused
  `chunks_vdb` refused until option 3 or 4 runs.
- **Embedding cost.** A rebuild re-embeds every affected record. On large
  datasets this means real API cost and time. Use the check mode first, and
  rebuild only the storages that need it.
- **Idempotent / crash-safe.** Sources (graph, `text_chunks`) are never
  modified. If the tool crashes between drop and rewrite, just re-run it.
- **`__created_at__` reset.** Backends that store creation timestamps in
  vector records (nano, faiss) will show fresh timestamps after a rebuild.
  No query logic depends on them.
- **Custom-KG placeholder entities.** `UNKNOWN` placeholder nodes created by
  `ainsert_custom_kg` are rebuilt faithfully from the graph; they may gain a
  vector record they previously lacked (improving their retrievability).
- **Sources are read in bounded batches.** Graph nodes and edges come from
  `iter_labels()` / `iter_edges()`, chunks from `BaseKVStorage.iter_rows()`,
  so neither the rebuild nor the check holds a whole namespace in client
  memory.
- **The whole source is read before the target is dropped.** A rebuild first
  reads its source to the end and writes the prepared records to a spool file,
  then drops the target and embeds from the spool. A source that cannot be read
  therefore leaves the vector storage untouched, and no source cursor is held
  open during embedding (an OpenSearch point-in-time or a MongoDB cursor would
  expire during a slow flush). The spool needs free disk space about the size
  of the records being rebuilt (for chunks, the chunk text) in the system
  temporary directory; set `TMPDIR` to put it elsewhere. It is readable only
  by the user running the tool and is deleted when the rebuild ends, failed or
  not; if deleting it fails, the tool logs its path. A killed process can leave
  a `lightrag-rebuild-*` directory behind: it holds source text, delete it.

## Consistency check

For each target (entities, relationships, chunks) the check walks the source
in batches, probes every record a rebuild would write by its vector id, and
then compares the number of source records with the exact number of records
in the vector storage. A count above the source means the vector storage holds
records the source no longer has (reverse orphans); the check does not list
them, a rebuild clears them. Each target gets one status:

| Status | Meaning | What to do |
| --- | --- | --- |
| `consistent` | every probe hit and the counts are equal | nothing |
| `inconsistent` | a probe missed, or the counts differ | rebuild that target |
| `inconclusive` | a read or the count failed or is unsupported before any probe missed; the reason is printed | fix the cause and check again |
| `incompatible` | the storage refused to attach (other embedding space, corrupt snapshot) | rebuild; it is mandatory |
| `not_applicable` | the storage keeps no vectors | nothing |

Records a rebuild skips (a chunk without `content`, a node or edge without an
id) are skipped by the check too and counted under `skipped`, so a fresh
rebuild always checks `consistent`.

Probes use `BaseVectorStorage.get_by_ids_strict()`, which raises on a failed
read instead of answering it with a miss the way `get_by_ids()` may (OpenSearch,
for one, returns an empty slot for an item whose shard failed). A miss is
therefore a confirmed absence, and it is reported `inconsistent` even when a
later read or the count fails.

**Every bundled vector backend implements both strict reads and exact counts**
(NanoVectorDB, Faiss, PostgreSQL, Milvus, Qdrant, MongoDB, OpenSearch), scoped
to the workspace the way their reads are. A count is refused while the storage
still buffers writes, so run the check with no writers. A third-party backend
that lacks `get_by_ids_strict()` or `get_exact_count()` is
`inconclusive`, never `consistent`.

A `consistent` verdict rests on these assumptions, accepted because each is
rare and a rebuild clears whatever it hides:

- **The graph stores one relationship per pair of entities**, and `iter_edges()`
  returns it once. A backend that stored both directions would count the pair
  twice.
- **Relationship vector ids are not injective.** The id hashes the two sorted
  entity names concatenated without a delimiter, so the pairs `("ab", "c")`
  and `("a", "bc")` share one id. Two such relations count twice in the graph
  but once in the vector storage, so the target reads `inconsistent` after
  every rebuild; and together with one reverse orphan they would read
  `consistent`. Changing the id format needs a migration and is out of scope.
- **Redis `SCAN` may return a chunk key twice** (when Redis resizes its hash
  table during the scan). A repeated key inflates the chunk source count the
  same way. The check keeps no set of every id seen, by design: that would
  make its memory grow with the dataset.
- **A vector count can only overstate, never understate.** Milvus does not
  enforce primary-key uniqueness on `insert`, which its schema migration uses,
  so a retried migration can leave a duplicate row that `count(*)` counts
  twice. An overstated count reads `inconsistent` (a rebuild clears it), never
  `consistent`: that needs every source record found and the count equal to
  the source.

## Library usage

The core rebuild/check functions are plain async functions that accept your
own initialized storage instances:

Rebuild targets must persist a queryable vector index. Passing a graph-only
backend such as `NoopVectorDBStorage` raises before source records are read or
the target is dropped; configure a persistent vector backend first.

```python
from lightrag.tools.rebuild_vdb import (
    check_vdb_consistency,
    rebuild_chunks_vdb,
    rebuild_entities_vdb,
    rebuild_relationships_vdb,
)

report = await check_vdb_consistency(
    graph,
    entities_vdb,
    relationships_vdb,
    text_chunks_kv=text_chunks,  # optional: also check chunks_vdb
    chunks_vdb=chunks_vdb,
)
status = {label: t["status"] for label, t in report["targets"].items()}
needs_rebuild = {"inconsistent", "incompatible"}
# "inconclusive" is not a finding: report it and check again, never drop on it.
if status["entities"] in needs_rebuild:
    await rebuild_entities_vdb(graph, entities_vdb, global_config)
if status["relationships"] in needs_rebuild:
    await rebuild_relationships_vdb(graph, relationships_vdb, global_config)
if status.get("chunks") in needs_rebuild:
    await rebuild_chunks_vdb(text_chunks, chunks_vdb)
for label, verdict in status.items():
    if verdict == "inconclusive":
        print(f"{label}: not verified ({report['targets'][label]['reason']})")
```
