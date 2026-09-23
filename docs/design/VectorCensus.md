# Offline vector census

Stop the server and all other workspace writers before auditing. Use a fresh,
initialized reader. The census itself never embeds, flushes, adopts an embedding
space, changes a marker, or repairs anything. Storage construction/initialization
still has each backend's documented startup side effects; the census is not a
replacement for initialization and does not certify that initialization is read-only.

`check_vdb_consistency` preserves the existing three-argument forward-only API.
`include_census=True` adds a separate `census` result, with optional `chunks_vdb`
and `text_chunks`. Its legacy forward keys are not a certification for an
unsupported backend. The rebuild CLI displays the census. `audit_kg_integrity`
shares its graph enumeration and adds `vector_census`; `--apply` continues to
repair recovery anchors only. Any vector rebuild remains a separately confirmed
operation in the existing rebuild tool, using its entities/relations/chunks targets.

## Certified population

Only the exact NanoVectorDBStorage implementation has a verified complete
population. The checked async `client_storage` loader supplies its materialized
metadata. IDs are copied without yielding while its live reference is held.
All four pending/unsaved buffers and the dirty flag must remain empty; the audit
never flushes them. Missing, unreadable, corrupt, incompatible, stale or malformed
snapshots are unavailable/inconclusive, never zero. Valid loaded empty files are
zero. The namespace fingerprint must match the loaded snapshot and remain stable
across the full census, including source/provenance reads. This is a race detector,
not transaction isolation: timestamp collisions and concurrent cross-store writes
remain outside the required idle-workspace operating contract.

Other vector backends report unavailable until an exact, error-transparent
population contract is verified. Non-persisting backends are not applicable.
No abstract method or storage format is added. No startup adoption gate is called.

One vector ID set V supplies total, physical matches and reverse excess V minus
all source candidate IDs. Either legacy relation orientation covers one logical
relation; both orientations are known duplicates, not reverse orphans. Counts
alone cannot detect equal-size different-ID sets. Source inventory failures prevent
coverage conclusions. Metadata reads require complete batches, and a disappeared
source chunk invalidates the chunk census.

Document attribution applies only to missing source-backed vectors. Present valid
entity/relation tracking rows outrank graph source_id, including empty rows;
source_id is a fallback only when tracking is absent. Chunk full_doc_id supplies
ownership. Unresolvable provenance is explicit; reverse IDs never imply document
ownership. Partial ingestion, purge and rebuild can leave accepted lag/residue;
a coverage gap is not proof of data loss or an automatic deletion recommendation.

## Materialization

The audit deliberately shares one full graph traversal. Memory is O(nodes + edges
+ chunk IDs/metadata + vector IDs + provenance links), plus Nano's already loaded
vectors. Source/provenance reads use bounded batches; examples are capped at20.
This is offline small-scale/default-backend tooling, not a streaming guarantee.
Graph iterator conversion and other vector backend adapters are follow-ups.
