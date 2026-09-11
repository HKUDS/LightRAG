# Purge Recovery Contract

Read this before changing `_purge_kg_contributions`, `adelete_by_doc_id`, `merge_nodes_and_edges` Phase 0 anchors, the `kg_write_state` / `kg_purge` doc-status metadata, the metadata carry-over whitelists in `lightrag/utils_pipeline.py`, `compute_incremental_chunk_ids`, or the cache write ordering in `use_llm_func_with_cache` / `update_chunk_cache_list`. Summary in [AGENTS.md](../../AGENTS.md#purge-recovery-contract).

The KG is shared across documents, so "what did this document contribute?" can only be answered from the per-document **write-ahead recovery anchors** (`full_entities` / `full_relations`, written and flushed in `merge_nodes_and_edges` Phase 0 *before* the first graph mutation). The reverse lookup — graph `source_id` → `text_chunks` → `full_doc_id` — is not a fallback, because purge deletes those chunks.

The governing invariant is narrower than "every purge needs a proof":

> **A purge must never delete something that CARRIES attribution — a chunk row or an anchor row that names objects — and leave those objects behind.** An operation that removes no such carrier cannot strand anything and needs no proof.

`_purge_kg_contributions` therefore **fails closed** (`RecoveryAnchorMissingError`, surfaced as HTTP 409, nothing deleted) when it would remove a carrier without one of these proofs. Treating absent anchors as an empty candidate list was the silent-skip defect this contract exists for: graph cleanup was skipped while the chunks went anyway, stranding unattributable entities that `audit_kg_integrity` can only report as unrecoverable orphans.

| Proof | Established by |
|---|---|
| `anchors` | Both anchor ROWS present and structurally usable. **Row presence is the test, never list truthiness** — an empty row is a document that extracted no entities, and conflating the two is the original bug. |
| `pre_graph` | `doc_status.metadata.kg_write_state`. Stamped `pre_graph` at enqueue so every pre-merge failure state inherits it by carry-over; advanced to `graph_mutation_started` only by `merge_nodes_and_edges`' `on_anchors_durable` hook. **Monotonic** — nothing writes it back, because re-stamping `pre_graph` on reprocess would let the resume purge skip and orphan the previous run's contributions. Absent means UNKNOWN (a row enqueued before the marker existed), which fails closed. |
| `journal` | `doc_status.metadata.kg_purge` at a phase past `prepared`, i.e. a previous attempt got far enough to have deleted the anchors itself. |
| `empty_scope` | No chunks AND no anchor row that names anything — so the delete removes no carrier at all and the invariant is satisfied outright. This is what lets a row enqueued before the marker existed, still holding no chunks, be deleted directly (no scan, no audit). |

**`kg_write_state` must never be inferred.** `pre_graph` asserts "this document never touched the graph", which licenses deleting its chunks while *skipping the graph* — sound only because the marker is written once, at enqueue, when it is necessarily true and the document has no history to misread. A backfill keying off a momentarily-empty `chunks_list` would stamp a document that does own graph objects, and because the stamp is durable the damage lands later, when the chunks reappear: chunks deleted, graph skipped, the original silent-skip defect reproduced exactly. `empty_scope` is safe where such a backfill is not, because it is re-evaluated against live state on every call and grants nothing beyond that call. `tests/pipeline/test_purge_fail_closed.py::test_a_false_pre_graph_marker_would_reproduce_the_original_defect` pins the cost.

Anchor-driven whole-document purge is **journaled and resumable** through four ordered phases — `prepared` → `derived_committed` → `anchors_pending` → `completed` — keyed by an operation id over the document key plus its chunk SET. The journal is *required by* fail-closed rather than an optimisation: purge's last step deletes the anchors, so without it any later failure would make every retry refuse forever. A resumed purge skips exactly the phases already persisted (so it never re-runs the LLM-cache-backed rebuild); an in-flight journal for a different operation is refused (`KGPurgeOperationConflictError`), while a stale `completed` one is ignored as dead bookkeeping.

Both metadata keys are in the `_DOC_STATUS_METADATA_CARRY_OVER_KEYS` **and** `_DOC_STATUS_METADATA_DIRECTIVE_KEYS` whitelists in `lightrag/utils_pipeline.py`; dropping either at a transition or a FAILED→PENDING reset turns a resumable purge into a permanent refusal. Retiring one requires `doc_status_transition_metadata(..., drop=...)` — passing it via `extra` would persist the value, and omitting it lets carry-over restore it.

Callers: `adelete_by_doc_id` (delegates wholly to the primitive; the chunk-less branch runs it too), and the pipeline's resume path `_purge_stale_extraction_if_resuming` (which retires the journal and persists `chunks_list=[]` in one targeted write). Explicit-candidate mode — custom-chunk patch rollback — is neither journaled nor proof-checked, because its own operation journal already names the complete candidate superset; the primitive reads that journal to union in candidates no anchor row can name yet.

A document can legitimately own nothing: `skip_kg` (`process_options` `'!'`) skips extraction and the merge, so no anchor rows are ever written. Post-change those documents carry `pre_graph` and delete normally; older ones have neither proof, and anchor repair has nothing to rebuild from.

The offline remedy for a document with no proof is `audit_kg_integrity(..., apply=True)` (`lightrag/tools/kg_integrity_repair.py`): it rebuilds anchors from surviving chunk provenance, and — because it enumerates the **whole** graph, which the hot paths never do — it can additionally certify that a document appearing nowhere in that scan owns nothing, writing it the empty anchor rows that are the normal proof for such a document (`anchorless_docs` in the report). Absence is only ever concluded from the completed scan; a document that does own graph objects is repaired with its real names, never blanked.

## Chunk tracking authority

**Chunk tracking outranks graph `source_id`.** Within a surviving entity or relation, the `entity_chunks` / `relation_chunks` row is the authoritative chunk list; the graph node's `source_id` is only a truncated view of it (`apply_source_ids_limit`) and may legitimately still name chunks a previous purge already pruned — `_purge_kg_contributions` reads tracking first, falls back to `source_id` only when the row is absent, and its `graph_references_deleted_chunks` branch exists to repair exactly that lag. So code that folds a `source_id` delta back into tracking must append genuine additions only: restoring an ID that is in the graph but not in tracking writes stale attribution into the authoritative store, and a later purge would rebuild or retain KG objects from chunks that no longer exist. `compute_incremental_chunk_ids` carries this rule and `tests/utils/test_compute_incremental_chunk_ids.py` pins it. Genuinely missing attribution is repaired by `audit_kg_integrity`, never by the incremental path.

Relation chunk tracking is the authoritative chunk list, so the no-source placeholders must never be written into it.

A tracking row whose graph object is gone is not repairable one row at a time: `BaseKVStorage` has no enumeration API, so nothing can sweep for it. The operator remedy is the offline `lightrag-repair-chunk-tracking` tool, run only after every writer for the workspace has stopped. It replaces one or both namespaces from current graph keys, retains authoritative rows for live objects (including rename/merge/manual-create results), and supplements them from cached extraction results. It never seeds from graph `source_id`, whose reuse is precisely the provenance downgrade this section forbids. It is ungated, unlike `_migrate_chunk_tracking_storage`, which only fires on an empty namespace. See [ProgramingWithCore.md → Repairing chunk tracking](../ProgramingWithCore.md#repairing-chunk-tracking).

## LLM extraction cache reachability

An extraction cache row (`cache_type="extract"`) stores the prompt that produced it — which embeds the chunk text verbatim — together with the entities and relations extracted from it. Nothing indexes those rows by document: the only thing that ever reaches one again is the owning chunk's `llm_cache_list`. That list is therefore an attribution carrier in the sense of the governing invariant above, and `adelete_by_doc_id(delete_llm_cache=True)` is the promise that rests on it.

> **The reference is always recorded before the row, and committed before the row.**

The row and the reference are two writes with no transaction between them, so the ordering does not remove the intermediate state, it only chooses which one survives. Writing the row first and attaching afterwards left an unreachable row whenever the gap was cut short — a sibling chunk's exception cancelling the task through `extract_entities`' `FIRST_EXCEPTION` wait, a hard kill, or a storage failure inside the attach, which is swallowed and only logged. Attaching first can lose the reference but never the row.

The rule has to hold at **both** layers, because on a deferred KV backend they are not the same event:

| Layer | On an immediate-write backend (PG, Redis, Mongo, OpenSearch) | On a deferred one (`JsonKVStorage`, the default) |
|---|---|---|
| write | `update_chunk_cache_list` returns only after the reference is durable, so `save_to_cache` can never outlive it | `upsert` reaches shared memory only; the return orders the two writes but proves nothing about disk |
| commit | `index_done_callback` is a cheap no-op | `index_done_callback` is the commit point, so the **commit order is what makes the write order durable** |

That second row is why the commit order is enforced at **all three** places the two namespaces are committed together, not just the failure epilogue:

| Site | Ordering |
|---|---|
| `_flush_storages` (every `_insert_done`) | the pair is chained — `text_chunks`, then `llm_response_cache` — instead of gathered beside each other; every other namespace stays concurrent, so the rule costs one flush of latency, not a serialised commit. A failed chunk flush skips the cache flush. |
| `_finalize_doc_failure` | commits `text_chunks`, then `llm_response_cache`, and **defers** the second when the first did not land. When the triggering `IndexFlushError` names `text_chunks` the gate does not retry the flush at all: a per-item backend has already dropped the permanently-failed operation, so a retry would find an empty buffer and report success. |
| `_discard_pending_index_ops` | flushes `text_chunks` **before the loop** — the loop reaches it first and only *drops* its buffer — then flushes the cache regardless, per the residue below. |

Committing the cache alone — which the narrow `_persist_llm_response_cache_best_effort` did on its own — puts a row on disk whose only reference is still in memory, and the next crash strands it: the extract-stage epilogue runs on exactly the sibling-cancellation path this ordering exists for. A deferred cache commit costs a re-run of the LLM; an unreachable row holding document text is permanent.

Dropping matters as much as crashing. `OpenSearchKVStorage._flush_pending_kv_ops` keeps only *retryable* failures buffered and **removes a permanently-failed operation before raising**, so on that backend an unordered commit needs no crash at all: one permanent bulk failure on a chunk row loses the reference outright while the concurrent cache flush succeeds.

When the reference cannot be recorded the cache write is **skipped** rather than performed anyway: a lost cache entry is recomputed on the next run, while an unreachable row holding document text is permanent. Extraction caching therefore depends on `text_chunks` being writable, and that degradation is reported rather than silent — `extract_entities` publishes the first occurrence plus one end-of-stage aggregate to `pipeline_status`, on the same discipline as token-limit truncation, because an unwritable chunk store skips on every chunk of the document and one line each would evict the rest of the run from the bounded history ring.

None of this may cost the **partial cache** of a document that failed midway. A large file is many chunks, and when chunk *n* raises, chunks 1..n-1 have already paid for their LLM calls; their cache rows must reach disk so the reprocess that follows replays them instead of re-billing. That is why the epilogue commits the cache at all, and why the aborting-batch cleanup flushes it before dropping the buffer. The ordering above never withholds that cache except where `text_chunks` is itself failing — the one case where a reprocess has nothing to attach the rows to anyway — and even there the epilogue only defers: the pair stays in memory for the next all-storage commit. A reprocess does not need the old reference to benefit, because the hit is keyed on the prompt and the cache-hit branch re-attaches the key to the freshly written chunk row.

### Accepted residue

| State | Why it is accepted |
|---|---|
| A reference to a row that was never written — a crash between the attach and the write, or a `save_to_cache` that no-ops on empty content | Harmless in direction, and every reader already tolerates it: `adelete_by_doc_id` and `_rollback_one_custom_chunk_patch` pass the ids to `llm_response_cache.delete()`, where a missing id is a no-op, and `_get_cached_extraction_results` (also used by the chunk-tracking repair tool) drops `None` entries from its batch get. It goes away with the chunk row itself. |
| The cache flushed with its references dropped, on an aborting batch | `_discard_pending_index_ops` flushes the chunk rows first, then flushes `llm_response_cache` **even when that failed** — deliberately the opposite call from the failure epilogue's. There the pair survives for the next commit, so withholding the cache defers it; here the buffers are discarded immediately after, so withholding it destroys LLM work that cost real money to produce and is not regenerable from anything on disk. A dangling row ranks below that. |

Dangling references already occur independently of this ordering: two chunks whose prompts are byte-identical share one cache row, so deleting one document leaves the other's reference dangling.

### Not closed by the ordering

A resume purge deletes a document's chunk rows — and with them every reference to its cache rows — before re-chunking, and deliberately does not touch `llm_response_cache`. Re-extraction hits those rows and the cache-hit branch re-attaches them to the freshly written chunk rows, so the loop closes on its own; a run that dies in between leaves them unreferenced until the next attempt.

Reprocessing under **changed** chunking never closes it at all: the chunk text differs, so the old prompts are never reissued and no hit occurs. That is unreachable by any ordering — the prompt is gone. Reclaiming those rows needs an operator-invoked sweep that deletes cache rows whose owning chunk no longer exists, which is also the only thing that can reach `summary`, `smartheading` and multimodal analysis rows: they carry no chunk reference in the first place.

## Merge and rename failure model

Read this before reordering anything in `_merge_entities_impl` or the rename branch of `_edit_entity_impl`. Both write to three stores that have **no transaction between them** — the graph, the chunk-tracking KV, and the vector storage — so every ordering has intermediate states. The question a change has to answer is never "does an inconsistent state exist" (one always does) but **which** inconsistency it keeps.

### The rule

> Without a distributed transaction, an inconsistency is acceptable when it **heals itself later** or is **harmless in direction**. A change may only trade one accepted residue for a better one; a change that merely moves the window to the opposite direction is not an improvement.

"Better" is ranked: losing data outranks retaining an object that could have been deleted, which outranks surfacing chunks a query did not need.

The two directions are mutually exclusive by construction, which is why the choice is forced:

| Direction | Reached by | Consequence |
|---|---|---|
| rows ⊃ graph | flushing tracking **before** the graph commit | a purge subtracting from the row is more conservative: it can keep an object it could have deleted, and retrieval can surface chunks of a source whose merge did not land. **No data is lost.** |
| rows ⊂ graph | flushing tracking **after** the graph commit | a purge concludes the object has no remaining sources and deletes it, while the graph already carries the merged evidence. **Data is lost.** |

Both paths therefore flush the migrated rows before the first commit that publishes the objects they describe, and accept the first direction.

### Ordering invariants

1. **A graph commit may only publish objects whose tracking rows are already on disk.** Stated over *every* commit in the function, not just the last one — `_merge_entities_impl` commits twice, and the first one publishes the merged target and its redirected relations.
2. **A tracking row is retired only after a confirmed commit removed the object it described.** The reverse is the state the governing invariant above forbids: a live object whose attribution carrier is gone, from which a purge reads the KEEP-truncated `source_id` and can conclude "no remaining sources".
3. **The new key is written before the old key is deleted** (f86ef93c). An orphaned new-key row is dead bookkeeping a retry overwrites; a row under neither key loses the curated list outright.
4. **The removal, its commit and the retirement are one cancellation-deferring region** (`_finish_deferring_cancellation`), starting *before* `delete_node`: on an immediate-write graph backend (Neo4j, Memgraph, MongoDB, PostgreSQL — their graph `index_done_callback` is a no-op) the removal is durable as it returns, so a region that began at the commit would already be too late.
5. **A failure after a durable graph mutation is raised as `VectorStorageConsistencyError`.** `_edit_entity_impl`'s `allow_merge` branch re-raises only that type and folds everything else into a partial-success summary answering HTTP 200 with `final_entity` set to the source — a source the commit has just removed. A landed merge must never be reported as one that did not happen.

### Accepted residues

| State | Why it is accepted |
|---|---|
| Tracking rows on disk for objects the commit did not publish (failed or declined commit, or a crash) | Self-healing: the content written is what a successful merge or rename is supposed to write, so a retry reads it as its baseline, `merge_source_ids` deduplicates, and the commit brings the graph into line. Harmless in direction (rows ⊃ graph). For an **existing** target the row is not dead bookkeeping — it over-claims for a live object — which is why this is listed rather than dismissed. |
| Orphaned old-key rows after a durable removal (a retirement failure, a crash, or a direct cancellation of the region) | Dead bookkeeping until the key recurs; the failure names the keys in the log and raises typed. Nothing converges on it automatically — tracked in the follow-up issue on orphaned chunk-tracking rows. |
| Vector storage lagging the graph | The graph is authoritative; `lightrag-rebuild-vdb` restores it. |
| A vector record deleted while its graph object survives (the merge deletes source vectors before removing the nodes) | Chosen deliberately over the inverse: a missing embedding is rebuildable, a vanished node with live tracking rows is not. |
| A rename partially applied — new node and edges durable, old node intact | No carrier was deleted ahead of its object, and the new objects fall back to the `source_id` copied from the old edge. Not retryable because of the target-exists precheck; tracked with the orphaned-row follow-up. |
| A multi-source merge whose `delete_node` raises mid-loop on an immediate-write backend | Fails closed: on a deferred backend `delete_node` returning is not evidence of durability, so retiring the rows of "successfully deleted" sources would retire authoritative rows of nodes still on disk. Tracked with the orphaned-row follow-up. |

### Rejected remedies

- **Reporting a durable write as a failure** to protect unnotified workers. It fences no one — a worker that missed the reload notification is in the same state whether the commit returns `True`, returns `False`, or raises — while costing the two defects above (skipped retirement, documents marked FAILED whose graph writes are on disk). The fence needs a channel that cannot fail with the shared-storage manager, which is its own follow-up.
- **A persistent journal** of the staged cleanup. Removed from this series on purpose in 0258af56, which kept the cancellation region and dropped the journal.
- **An in-memory undo log** restoring the previous rows on a failed first commit. It covers only the process-survives path — the one that already heals on retry — and does nothing for a crash.
- **Treating a call's return as durability evidence.** `delete_node` returning normally means nothing on a deferred backend; only a confirmed `index_done_callback` does, which is what `_commit_graph_or_raise` checks (an explicit `False` is a decline; backends returning `None` are unaffected).
