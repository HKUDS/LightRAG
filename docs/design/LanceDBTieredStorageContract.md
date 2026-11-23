# LanceDB tiered-storage contract (cold lake)

Status: **implemented for the cold-lake jobs.** This is the contract the code
keeps: the export job, the rebuild-from-snapshot promotion, and the cold-query
job. It is not a plan for a live LanceDB backend — the warm and hot tiers are
conditional future waves, and nothing here authorizes them. Every rule below
is enforced by the code named next to it; the community anchor for each
decision is in the *Anchor map* at the end.

| | |
| --- | --- |
| code | `lightrag/kg/lancedb_impl.py` (export job), `lightrag/tools/snapshot_restore.py` + the `--from-snapshot` mode of `lightrag/tools/rebuild_vdb.py` (promotion), `lightrag/kg/lancedb_cold_query.py` (cold query), `lightrag/kg/lancedb_cold_recall.py` (recall harness) |
| tools | `python -m lightrag.kg.lancedb_impl`, `lightrag-rebuild-vdb --from-snapshot`, `python -m lightrag.kg.lancedb_cold_query` |
| tests | `tests/kg/lancedb_impl/` (export + cold-query contracts, S3 integration), `tests/tools/` (rebuild idempotence + zero-re-embed proof) |
| read with | [FileBackedSnapshotContract.md](FileBackedSnapshotContract.md) (the deferred-embedding protocol the promotion seeds), [PurgeRecoveryContract.md](PurgeRecoveryContract.md) (why a restored workspace fails closed on delete), [ConfigurationStorageContract.md](ConfigurationStorageContract.md) (the closed config-backend list Lance is not on), [README_REBUILD_VDB.md](../../lightrag/tools/README_REBUILD_VDB.md) |

Read this before touching any of:

- `lightrag/kg/lancedb_impl.py`, `lightrag/kg/lancedb_cold_query.py`,
  `lightrag/kg/lancedb_cold_recall.py`;
- `lightrag/tools/snapshot_restore.py` or the `--from-snapshot` branch of
  `lightrag/tools/rebuild_vdb.py`;
- the snapshot schema (the four tables and their embedding columns);
- any proposal to register LanceDB as a storage backend, add a storage role,
  or put a Lance table on the query hot path.

This contract was promoted from the fork-local LanceDB tiered-storage
activation blueprint (a planning document, not shipped under `docs/`). The
blueprint's taxonomy and cost model stay there; the durable decisions live
here.

## Rules at a glance

Each rule is expanded in the section named after it.

1. **Derived-index tiering.** The cold snapshot is a source of truth; the hot
   vector storages are disposable derived indexes. Promotion is a rebuild
   from the snapshot, never a copy or a move, and never a re-embed.
   *(Derived-index tiering.)*
2. **At most three durable tiers.** Cold lake, artifact, hot serving. A warm
   cache and an analytics read path are optional derived layers, never a
   fourth durable copy. *(Authority layers.)*
3. **Snapshots are tagged and immutable.** Every export writes new versions
   of the four tables and tags them only when all four are complete. A tag
   name is never reused, never moved, and never pruned; a re-export is a new
   version under a new tag with the old tag intact. *(Tagged immutable
   snapshots.)*
4. **Promotion re-embeds nothing.** Vectors travel as snapshot columns and
   are seeded into the deferred-embedding vector cache; the embedding
   function is never called. Only `NanoVectorDBStorage` targets support this;
   any other backend raises `StorageCapabilityError` before anything is
   dropped. *(Zero-re-embed promotion.)*
5. **Jobs, not a backend.** The cold lake is three offline single-writer
   jobs. No registry entry, no storage role, no new `@abstractmethod`, no
   live concurrency contract, no hot-path integration. *(Single-writer job
   shape.)*
6. **Cold query is an exact scan at the latest version.** DuckDB's lance
   extension reads only the latest version and cannot pin a tag, so the
   queried tag must BE the latest version, and each snapshot is published to
   its own immutable prefix. No ANN on the cold path. *(Cold query.)*
7. **Validation precedes mutation.** Export, restore and cold query each
   refuse (typed error, nothing written / nothing dropped / no hits) before
   the first mutation when a precondition fails. *(Fail before write.)*
8. **Reserved parameters are not a config surface.** The knobs listed in
   *Reserved parameters* are design reservations for measured future waves;
   adding one requires a measurement trigger and an update to this contract.

## Derived-index tiering

LightRAG's three vector storages are rebuildable from the graph and the
`text_chunks` KV store (`lightrag-rebuild-vdb`). That makes the hot vector
layer a **derived index**: the authoritative records are the documents,
chunks and graph edges, and the snapshot stores those records together with
their embeddings as columns. The cold lake therefore holds a source of
truth, and the hot layer holds a disposable serving copy.

Promotion is **rebuild-from-source**, not copy/move:

- The snapshot's `docs` / `chunks` tables restore `full_docs` / `text_chunks`;
  `entities` / `relations` restore the graph; the three vector targets are
  then dropped and rewritten from the snapshot's embedding columns.
- Because the embeddings travel with the rows, promotion rebuilds indexes
  only. Re-embedding a large corpus on every promotion is the cost this
  design exists to avoid (a six-figure-chunk corpus re-embed is a
  four-figure-dollar event, an estimate from the blueprint's cost model).

This is the same shape as Databricks AI Search Delta Sync (a Delta source
table plus a derived serving index) and LanceDB's own index model (indices
are derived files over an authoritative manifest, rebuildable and allowed to
be incomplete). It is deliberately simpler than a vector-native tiered
store: there the durable data IS the vectors and must be moved or cached;
here the durable data is the KG and the vectors are discarded and rebuilt.

Rejected alternatives (blueprint decision D1): per-vector hot↔cold migration
(the hot backend has no automatic tiering, so a mover, dual-read and
consistency repair would all be new code); cross-tier dual-write (LightRAG
already has no cross-store transactions — a second authority multiplies the
consistency surface); query-path cross-tier federation (every query pays a
second fault domain and latency). One source of truth plus rebuild heals all
of them without new machinery.

## Authority layers

Tier by **authority** (which copy is the truth and how the other heals), not
by temperature. At most three durable tiers:

| Tier | Authority | Heals by |
| --- | --- | --- |
| cold lake | source-of-truth snapshot (docs/chunks/graph + embeddings) | re-export from a live workspace |
| artifact | multimodal objects (images/PDFs) in object storage | re-upload from the pipeline |
| hot serving | the live workspace's KV/graph/vector storages | rebuild from the cold lake |

A warm cache (object store durable, RAM/SSD cache) and an analytics read path
(DuckDB/Spark/Trino over the cold tables) are **optional derived layers**:
they hold no unique durable copy, so they are not fourth tiers. The artifact
tier is a separate wave; this contract governs the cold lake only.

The cap is the community pattern: Milvus/Zilliz "tiered storage" is one
durable object-store copy plus a cache hierarchy; Weaviate's hot/warm/cold
tenant states move one durable copy between media; Elasticsearch data tiers
and searchable snapshots keep the snapshot authoritative and read-only.
Every extra durable tier doubles the migration jobs and the consistency
surface; three is the observed ceiling of the successful systems.

## Tagged immutable snapshots

The export job (`export_snapshot` in `lightrag/kg/lancedb_impl.py`) publishes
four Lance tables — `docs`, `chunks`, `entities`, `relations` — into one
destination (a local directory, an `s3://` or an `oss://` URI). Embeddings
are fixed-size-list `float32` columns of the configured dimension, so a
later rebuild never re-embeds. Every table's schema metadata records the
exporting embedding model under `lightrag.embedding_model` (empty when the
model is unknown), so a restore can verify the vectors' query space.

Lance gives each table ACID commits and an MVCC manifest: every write is a
new version, and a **tag** names a version permanently. Tags survive
compaction and version pruning — that is what makes a tagged snapshot a
durable anchor even when the numeric versions around it are cleaned up.
(The Lance transaction spec and object-store guide define this; the
conditional-put commit protocol behind it is why the single-writer rule
below exists.)

The export rules that keep snapshots immutable:

- A tag name that already exists on ANY of the four tables raises
  `ExportError` before any write. Tags are never reused or moved.
- Each export writes a NEW version of every table; the previous tagged
  versions stay readable. Re-export is a new tag, not an update.
- Tags are created only AFTER all four tables are complete. An interrupted
  export leaves untagged versions (prunable garbage, no tagged data lost);
  a same-tag re-run publishes cleanly only while NO table carries the tag
  yet — a partial tagging refuses a same-tag retry (Accepted residue 6).
- Vectors are READ from vector storage (`get_vectors_by_ids`), never
  computed. The CLI installs a fail-loud embedding stub so an unflushed
  deferred-embedding row aborts the export instead of silently embedding.

Row identity follows the rebuild id scheme: `ent-` hashes of entity names,
`rel-` hashes of the sorted endpoint pair, and chunk ids. Relations carry
both endpoint orders as candidates (`make_relation_vdb_ids`); a legacy
reverse-order relation id is kept as stored, so the snapshot joins the
vector the hot layer actually holds. Vector rows with no source record are
disposable and are not exported.

## Zero-re-embed promotion

`lightrag/tools/snapshot_restore.py` restores a workspace from a tagged
snapshot; `lightrag-rebuild-vdb --from-snapshot URI --tag TAG` is the CLI
(`--batch-size`, default 500, and `--concurrency`, 1–3 parallel vector
targets, default 3, apply to snapshot mode only). Remote destinations
resolve credentials through `storage_options_from_env(uri)`: the URI
selects the dialect (`s3://` → `aws_*` keys, `oss://` → native OSS keys).

The mechanism, in order:

1. `open_snapshot` checks out the tag on all four tables and validates
   BEFORE any mutation: tag present, tables non-empty, embedding dimension
   equal to the configured one, no vector-eligible row without a stored
   vector (content-less chunks legitimately carry none), and the recorded
   embedding-model identity equal to the target's. Any violation raises
   `RestoreError`; nothing has been touched.
2. Sources of truth are restored first — `full_docs` / `text_chunks` KV and
   the graph — and flushed (`index_done_callback`).
3. Each vector target is dropped and rewritten. The snapshot's vectors are
   seeded into the deferred-embedding buffer's vector cache
   (`_pending_upserts[rid].vector`, the *Deferred-embedding protocol* of
   [FileBackedSnapshotContract.md](FileBackedSnapshotContract.md)): a
   pending row whose vector is already set materializes without embedding.
   The embedding function is never called.
4. The rebuild CLI installs a COUNTED fail-loud embedding stub; any embed
   attempt aborts the run, and success requires the count to be exactly 0.
   That count is the hard gate of the promotion, not a benchmark target.

The model-identity check reads the `lightrag.embedding_model` schema
metadata the export stamps on every table: vectors from a different
embedding model are an incompatible query space even at equal dimension,
so a recorded model that differs from the target's refuses the restore.
When either side is unnamed — an older snapshot without the metadata, or a
target with no configured model name — the identity cannot be verified;
the restore logs a WARNING and proceeds on the operator's assurance that
the spaces match. That warning is the accepted residue of unverifiable
identity: loud, never silent, and the dimension check still applies.

Only `NanoVectorDBStorage` targets support the vector-cache seeding; any
other backend raises `StorageCapabilityError` before anything is dropped.
This is a capability boundary, not an oversight: the other backends have no
deferred-embedding buffer to seed, and faking one would re-embed or write
vectors through a path that does not accept them.

The three targets run under an `asyncio.TaskGroup`, so a failing target
cancels and awaits its siblings — an interrupted restore never leaves
concurrent writers for the re-run. A re-run is idempotent: it re-restores the
sources and re-drops/rebuilds the indexes, healing any partial state. The
snapshot itself loses nothing. Each drop runs the caller's `before_drop`
hook first; the rebuild CLI wires it to the same private-backup recovery as
a normal rebuild, so a corrupt or refused target is backed up before it is
destroyed. Prefer an EMPTY target: over a populated workspace the source
upserts are additive while the vector targets are replaced, leaving
vector-less orphan source rows (Accepted residue 8).

The community shape: Qdrant keeps full-precision vectors as the durable
copy and reads them only to rescore a compressed candidate set — the
originals are never recomputed; Lance's when-to-index guidance treats the
index as a derived, rebuildable artifact over the data. Promotion here is
the same idea at workspace scale: the embeddings are the durable copy, the
hot index is rebuilt around them.

## Single-writer job shape

The cold lake is three offline jobs. It is deliberately NOT a storage
backend:

- No entry in `STORAGE_IMPLEMENTATIONS` / `STORAGES`, no fifth storage role,
  no new `@abstractmethod` on the base classes (each of those would tax
  every existing backend with conformance work).
- No participation in `pipeline_status`, no admin-write-gate interaction, no
  hot-path integration. The jobs run against a QUIESCED, initialized
  workspace: no writers, every flush complete.
- One export job per destination at a time; do not export while a cold
  query runs against the same prefix.

Single-writer is a correctness rule, not a throughput choice. Lance's
object-store commit protocol relies on conditional PUT, and S3-compatible
stores may silently ignore it (the object-store guide's explicit warning):
two writers can both report success and lose confirmed data. The job shape
sidesteps this entirely. Before ANY multi-writer use: verify the provider's
conditional-put enforcement, use an external manifest store
(`s3+ddb://`) or a commit lock, and run concurrent-writer tests. Until then
the fallback is local disk.

The shelved four-role live LanceDB implementation (the `dev-lancedb`
branch) is the rejected alternative: a live backend inherits every new
storage contract as conformance cost, and without a sustained maintainer
that cost is a liability — the maintainer's maturity gate on that branch
exists for exactly this reason. The job shape keeps the maintenance surface
at three modules with no live concurrency contract. See *Rejected
remedies*.

## Cold query

`lightrag/kg/lancedb_cold_query.py` answers an exact top-k over a tagged
snapshot through DuckDB's core lance extension. The query vector is
supplied by the caller; the job never embeds, never calls an LLM, and never
writes.

The shape and its constraints:

- **Exact scan, no ANN.** The SQL is
  `SELECT id, <distance>(embedding, <query>) AS distance FROM
  __lance_scan('<path>') WHERE embedding IS NOT NULL ORDER BY distance ASC,
  id ASC LIMIT k`, with `array_cosine_distance` (cosine, the default,
  matching the hot metric) or `array_distance` (l2). Smaller is closer.
  HNSW stays in RAM on the hot path: paging an HNSW graph through object
  storage is the Vespa-documented anti-pattern (graph traversal needs
  residency; paging it multiplies latency). The cold tier therefore promises
  no interactive SLO — interactive queries stay hot, and a warm cache is the
  conditional remedy if measurements demand it.
- **The tag must be the latest version.** DuckDB's lance extension
  (verified on 1.4.4 and 2.0.1) reads only the latest version and has no tag
  argument. A tag behind the latest version is REFUSED (`ColdQueryError`),
  never answered from newer rows. Publish each snapshot to its own immutable
  prefix so "latest" and "the tag I mean" coincide. Restore is different:
  `lancedb` `checkout(tag)` pins any tagged version, so promotion can read
  old tags that cold query cannot.
- **Project only `id` and `embedding`.** `lance_vector_search` projects
  every column and panics on a constant-encoded column, so the top-k is
  ordered in SQL over the two columns the answer needs. A fragment with
  exactly one row also panics on DuckDB 1.4.4 (2.0.1 reads it), so export
  with a batch size that leaves no single-row fragment. Both are accepted
  residues with a durable remedy below.
- The extension is installed once (`INSTALL lance FROM core`); the job runs
  `LOAD lance`. S3/OSS destinations additionally need a `TYPE LANCE` secret
  built from `storage_options_from_env` (the extension forwards it to Lance
  as `storage_options`; a `TYPE S3` httpfs secret is NOT read by the lance
  reader). For OSS the scan path is the `s3://` alias of the `oss://` lake
  and the secret's endpoint is the bucket-prefixed S3-compat host
  (`https://<bucket>.s3.oss-<region>-internal.aliyuncs.com`, vhost style,
  STS `SESSION_TOKEN`): the bundled lance-io uses a custom endpoint host
  as-is, so the bucket must already be prefixed. Missing table/tag, stale
  tag, wrong dimension, non-finite query values or a failed execution raise
  `ColdQueryError` before any hit; a partial ranking is never returned.

`lightrag/kg/lancedb_cold_recall.py` is the comparison harness: it takes
ranked id lists produced from the SAME stored vectors, never embeds, reads
no backend and writes nothing. `recall_at_k` divides by the de-duplicated
hot top-k (a thresholded hot baseline returning fewer than k rows does not
penalise the cold ranking); `paired_bootstrap_recall` resamples QUERIES, not
hits, and is deterministic for a given seed. Recall claims about the cold
path must come from this paired comparison, not from a synthetic-embedding
mechanism demo.

## Fail before write

All three jobs validate completely before the first mutation, and each
refusal is a typed error that leaves the world unchanged:

- Export: blank tag, reused tag, missing storages, empty source, missing or
  wrong-dimension vector → `ExportError`. The first batch of EVERY table is
  read and vector-joined before any write, so an empty source or a missing
  vector destroys nothing.
- Restore: blank tag, missing table/tag, empty table, dimension mismatch,
  missing vector, embedding-model mismatch, non-Nano target →
  `RestoreError` / `StorageCapabilityError` before anything is dropped.
- Cold query: invalid config, missing/stale tag, wrong dimension,
  non-finite values, failed execution → `ColdQueryError` before any hit.

This is the *Consistency without transactions* rule applied to jobs: a
failure must fail loud, and the documented residues below are the only
inconsistencies accepted.

## Reserved parameters

These knobs are RESERVED for future waves. They are not a config surface
today; adding one requires a measured trigger (the blueprint's
measure-before-build gate) and an update to this contract. The ones already
implemented are marked.

| Parameter | Status | Wave / purpose | Community anchor |
| --- | --- | --- | --- |
| export batch size | implemented (`--batch-size`, 500) | export throughput + fragment shape | Lance compaction guidance |
| rebuild batch size | implemented (`--batch-size`, 500) | promotion memory bound | Lance compaction guidance |
| rebuild concurrency | implemented (`--concurrency`, 1–3) | parallel vector targets | — (bounded by the three targets) |
| cold-query top-k / metric / timeout | implemented | exact-scan shape | DuckDB lance extension |
| tag retention policy | reserved | how long old tags are kept before an operator prunes them | Lance versioning / tags |
| fragment size target | reserved | export fragment sizing for scan cost | Lance compaction |
| cache watermarks / TTL / eviction | reserved | warm cache fill/evict (conditional N4) | Milvus tiered-storage eviction |
| remote-data-to-cache ratio | reserved | warm cache sizing | OpenSearch searchable snapshots |
| `read_consistency_interval` | reserved | reader refresh if a live reader ever appears | LanceDB embedded reads |
| optimize cadence | reserved | compaction scheduling for a live writer | Lance compaction |
| IVF partitions | reserved | larger-than-RAM ANN (conditional N5) | Lance IVF index guidance |
| ANN recall threshold | reserved | measured ANN-vs-exact gate (conditional N5) | Lance when-to-index |
| quantization bits | reserved | compressed candidate copy (conditional N5) | Qdrant quantization / rescore |
| promotion SLO | reserved | rebuild time budget at corpus scale | blueprint promotion gate |

The cold lake itself uses none of the reserved knobs: it is an exact scan
over immutable snapshots, so there is no cache to evict, no index to tune
and no consistency interval to set.

## Accepted residues

Each residue is accepted with its reason and recovery path, per
*Consistency without transactions*. None licenses a silent failure.

1. **Operational KV namespaces are not in the snapshot.** `doc_status`, the
   LLM response cache, and the chunk-tracking / recovery-anchor namespaces
   are not exported. A restored workspace serves queries, but delete/retry
   lack their anchors and FAIL CLOSED (`RecoveryAnchorMissingError`, see
   [PurgeRecoveryContract.md](PurgeRecoveryContract.md)) until re-ingestion
   rebuilds them. Reason: those namespaces are operational state, not
   source-of-truth content, and exporting them would couple the snapshot to
   the pipeline's scheduling contract. Recovery: re-ingest the documents;
   the snapshot loses nothing. This is the documented boundary of
   "restored workspace serves queries".
2. **Creation timestamps reset on restore.** `created_at` /
   `__created_at__` are stamped fresh when rows are re-upserted; the snapshot
   does not carry the originals. Harmless in direction: timestamps are
   metadata, the content and vectors are unchanged. Recovery: none needed;
   re-ingestion restores true times if an operator requires them.
3. **Reconstructed content may be fuller than the original.** Entity and
   relation vector-store content is rebuilt from the snapshot's full
   descriptions through the current truncation rules, so it can be fuller
   than a hot row truncated under older limits. The vector is copied
   unchanged, so ranking is unchanged; the direction is harmless (more
   context, same vector). Recovery: none needed.
4. **DuckDB 1.4.4 fragment/column panics.** A one-row fragment panics the
   1.4.4 lance reader, and `lance_vector_search` panics on a constant-encoded
   column. Avoided today by exporting with a batch size that leaves no
   single-row fragment and by projecting only `id` + `embedding` and ordering
   in SQL. Durable remedy: a newer DuckDB (2.0.1 reads one-row fragments);
   the projection discipline stays regardless because it is also the cheaper
   scan.
5. **Object-store list permission is required.** `lancedb.connect` (and the
   DuckDB httpfs read path) needs bucket-level `ListObjects` /
   `ListObjectsV2`; Put/Get/Delete alone are not enough. A destination whose
   policy denies List fails loud at connect, not silently. Recovery: grant
   the list permission (an operator action); until then the cloud jobs are
   unavailable and the local-directory destination still works.
6. **Interrupted export leaves untagged versions — or a partial tag.**
   Tags are created only after all four tables complete. A crash during
   the WRITE phase leaves untagged versions: prunable garbage, no tagged
   snapshot affected, and a re-run with the same tag publishes cleanly
   because no table carries the tag yet. A crash during the TAG phase
   (some tables tagged, others not) is different: the tag now exists, and
   the immutability preflight refuses a same-tag retry. Recovery: publish
   under a NEW tag; the created tags stay intact (tags are never moved or
   pruned) and the untagged versions remain prunable. Nothing is lost
   either way; only the recovery path differs.
7. **Interrupted restore leaves a partial hot index.** Sources are restored
   before the vector targets, and a failing target cancels its siblings, so
   a crash can leave some indexes rebuilt and others not. The snapshot is
   untouched. Recovery: re-run; the restore is idempotent and heals the
   partial state.
8. **Restore over a populated workspace leaves orphan source rows.** The
   KV/graph restore upserts additively while the vector targets are dropped
   and replaced, so pre-existing source rows absent from the tag survive
   without vectors. The direction is retention, not loss: no snapshot
   content is dropped, and a row that could have been deleted is kept
   (*Consistency without transactions*). A populated target is not refused
   because the idempotent re-run of residue 7 is itself a restore over a
   populated workspace. Recovery: prefer an EMPTY target; otherwise the
   orphans heal through a normal rebuild (which re-embeds from the restored
   sources) or re-ingestion, and an operator may clear the workspace first
   when the pre-existing content is disposable.

## Rejected remedies

These were considered and rejected; reopening one needs new evidence that
its rationale no longer holds.

1. **A four-role live LanceDB backend.** The shelved `dev-lancedb`
   implementation registered Lance for KV/vector/graph/doc-status at once.
   Every new storage contract then taxes it with conformance work, and
   without a sustained maintainer that cost is a liability — the
   maintainer's maturity gate on that branch is the record of this. The job
   shape (three offline modules, no registry entry, no live concurrency
   contract) keeps the maintenance surface minimal. A hot Lance vector
   backend is a conditional future wave requiring measured demand AND
   maintainer agreement, not a default.
2. **Re-embedding on rebuild.** Recomputing embeddings during promotion
   pays the embedding bill on every restore and couples promotion to the
   embedding service. Embeddings-as-columns plus the deferred-embedding
   vector cache make promotion index-only, proven by the counted stub
   (embed calls == 0). Rejected outright.
3. **Paged HNSW on the cold path.** Paging an HNSW graph through object
   storage multiplies latency because graph traversal needs residency (the
   Vespa paged-HNSW anti-pattern). The cold tier is an exact scan; HNSW stays
   in RAM on the hot path. ANN on cold data is a conditional hot/warm-wave
   question, not a cold-lake feature.
4. **Experimental RTE as a correctness dependency.** Lance's real-time
   engine / MemWAL is specified as experimental. The cold lake never touches
   doc-status or config, and no correctness guarantee here depends on RTE.
   A doc-status role would require RTE to be non-experimental first; that is
   a future-wave precondition, not something this contract assumes.
5. **DuckDB reading `oss://` through the extension's native OSS provider.**
   Verified on real OSS (ECS, DuckDB 2.0.1 core lance extension, bundled
   lance-io 12.0.0): the native provider fails at the first list with
   `ConfigInvalid: call HttpTransporter::install_default before using HTTP
   services` — the build never installs opendal's HTTP transport, so no
   credential configuration can fix it. The working read path is the S3
   protocol: scan the `s3://` alias with a `TYPE LANCE` secret whose endpoint
   is the bucket-prefixed `s3.` S3-compat host (the bundled lance-io uses a
   custom endpoint host as-is and does not prepend the bucket; an unprefixed
   endpoint lists an empty `_versions/` and fails as "dataset not found").
   Reopen only with a DuckDB build whose native OSS provider is fixed.

## Anchor map

Every architecture / domain / implementation / performance / parameter
decision above maps to a community source (the blueprint verified these on
its evidence date; vendor benchmark numbers are vendor-reported, not
independently audited).

| Decision | Community source |
| --- | --- |
| Derived-index tiering, promotion = rebuild | Databricks AI Search Delta Sync (docs.databricks.com/ai-search); LanceDB indices (docs.lancedb.com); blueprint D1 |
| Authority layers, ≤3 durable tiers | Milvus/Zilliz tiered storage (milvus.io/docs/tiered-storage-overview); Weaviate tenant offloading (docs.weaviate.io/tenant-offloading); Elasticsearch data tiers / searchable snapshots (elastic.co/searchable-snapshots) |
| Tagged immutable snapshots, MVCC manifest | Lance transaction spec (lance.org/format/table/transaction); Lance object-store guide (lance.org/guide/object_store) |
| Zero-re-embed promotion, embeddings-as-columns | Qdrant rescore / full-precision durable copy (qdrant.tech); Lance when-to-index (docs.lancedb.com) |
| Deferred-embedding vector cache seeding | [FileBackedSnapshotContract.md](FileBackedSnapshotContract.md), *Deferred-embedding protocol* |
| Single-writer jobs, conditional-put caution | Lance object-store guide (lance.org/guide/object_store); blueprint R1/R2 |
| Cold exact scan via DuckDB | DuckDB lance core extension (duckdb.org/docs/2.0/core_extensions/lance) |
| HNSW stays in RAM, no paged HNSW | Vespa paged-HNSW anti-pattern (blueprint §5.4 / D-decisions) |
| Reserved cache watermarks / TTL / eviction | Milvus tiered-storage eviction (milvus.io) |
| Reserved remote-data-to-cache ratio | OpenSearch searchable snapshots (docs.opensearch.org/searchable_snapshot) |
| Reserved IVF partitions / optimize cadence | Lance IVF / compaction guidance (docs.lancedb.com) |
| Reserved quantization bits | Qdrant quantization (qdrant.tech) |
| Fail-closed delete on a restored workspace | [PurgeRecoveryContract.md](PurgeRecoveryContract.md) |
| Config-backend closed list (Lance excluded) | [ConfigurationStorageContract.md](ConfigurationStorageContract.md) |