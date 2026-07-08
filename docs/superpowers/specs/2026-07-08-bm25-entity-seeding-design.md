# BM25 + Vector Hybrid Entity Seeding — Design

Date: 2026-07-08
Related: upstream RFC [HKUDS/LightRAG#3198](https://github.com/HKUDS/LightRAG/issues/3198)
Status: approved by user, pending upstream maintainer feedback (issue comment to be posted in parallel)

## Problem

LightRAG's graph retrieval uses vector search over entities as the *only* entry
point into the knowledge graph (`_get_node_data` → `entities_vdb.query`).
Domain jargon, acronyms, and product codes are underrepresented in embedding
training data, so both the query-side and index-side vectors are degraded and
cosine similarity becomes noise. Seed entities are never found, graph traversal
never starts, and knowledge that *is* in the graph is unreachable. The RFC
reports BM25 NDCG@10 of 0.92 vs 0.80 for dense retrieval on jargon-heavy
domains.

## Scope (MVP — deliberately narrower than the RFC)

- Index **entity names only** (~10⁴–10⁵ short strings). No edge keywords, no
  chunk bodies. This targets the exact failure (local-path seeding) with the
  smallest index; corpus statistics drift, index size, and multi-process
  concerns dissolve at this scale.
- Local path (`_get_node_data`) only. The global path (`_get_edge_data`) is a
  follow-up.
- BM25 engine: the `bm25s` PyPI library (the RFC's Option C), wired as an
  **optional dependency** with lazy import. Not a hard dependency.
- Feature is **off by default** — zero behavior change unless enabled.

Rejected alternatives (recorded for the PR discussion):

- *Hand-rolled BM25*: avoids the dependency, but bm25s matches the RFC
  author's own Option C and is battle-tested; dependency risk is mitigated by
  optional-extra + lazy import following the existing Neo4j/Redis pattern.
- *PG-native full-text (RFC Options A/B)*: PG-only; the default JSON stack
  (most users) gets nothing. Left as a second-phase backend behind the same
  interface.
- *Indexing chunk bodies*: large corpus brings back index-size / IDF-drift /
  multi-worker problems and bypasses the graph — off-target for this issue.

## Architecture

### New storage role: `KEYWORD_STORAGE`

Fifth pluggable storage role beside KV / Vector / Graph / DocStatus, following
the same interface discipline (implementations never leak engine objects
across the boundary).

- `BaseKeywordStorage` (append to `lightrag/base.py`):
  - `async index_entities(names: list[str]) -> None` — (re)build from the
    full entity-name corpus
  - `async search(query: str, top_k: int) -> list[tuple[str, float]]` —
    returns `(entity_name, score)` ranked
  - `async drop() -> None`
  - (no `remove_entities`: the index is derived data rebuilt from the full
    entity-name set, so idempotent full rebuild covers deletions)
  - plus the standard lifecycle hooks (`initialize` / `finalize` /
    `index_done_callback`)
- Registry: add `KEYWORD_STORAGE` to `STORAGE_IMPLEMENTATIONS` /
  `STORAGES` in `lightrag/kg/__init__.py`; env var
  `LIGHTRAG_KEYWORD_STORAGE`, default `Bm25KeywordStorage`.
- `LightRAG` gains a `keyword_storage` field + instantiation mirroring
  `doc_status_storage`.

### Default implementation: `Bm25KeywordStorage`

File: `lightrag/kg/bm25s_keyword_impl.py`

- **Scoring**: `bm25s` library, lazy-imported. If missing, the storage
  reports itself unavailable; the feature disables with a one-time warning
  telling the user to `pip install bm25s`.
- **Tokenizer**: reuse the project's existing `TiktokenTokenizer`
  (`lightrag.utils`). Entity names and queries are encoded to BPE token ids
  and the id sequence (as strings) forms the BM25 terms — the same encoder
  on both sides guarantees matching, and byte-level BPE handles CJK without
  any new tokenizer code. Known limitation (documented in the PR): BPE
  merges at word boundaries can differ slightly between standalone names
  and names embedded in a sentence; BM25's partial-overlap scoring
  tolerates this.
- **Updates**: bm25s indexes are static → rebuild-on-change. Rebuild is
  triggered after an entity-merge batch completes (not per entity); at
  entity-name scale a full rebuild is milliseconds.
- **Persistence**: serialized under the workspace-scoped storage directory;
  loaded on `initialize`. Corrupt/missing index ⇒ treated as empty, rebuilt
  on next indexing pass.
- **Workspace isolation**: one index per workspace, same semantics as other
  file-based backends.

### Query integration (`_get_node_data` in `lightrag/operate.py`)

```
enable_bm25_seeding and index available?
 ├─ no  → existing vector-only flow, byte-for-byte unchanged (default)
 └─ yes → vector top_k  ∥  bm25 top_k (tokenizes the RAW query text —
          independent of LLM keyword extraction, so it survives even
          when keyword extraction mangles rare terms)
          → RRF fusion keyed by entity_name (k=60); dedup is inherent —
            an entity on both lists sums its rank contributions
          → truncate to top_k fused seeds
          → log each seed with source ∈ {vector, bm25, both}
          → existing downstream (get_nodes_batch, edge expansion, chunks)
            runs ONCE on the fused seed set, unchanged
```

Fusion happens at the **seed stage, before graph expansion** — never after
two full retrievals. Truncation happens **after** fusion (discarding each
list's tail before fusing would wrongly kill entities ranked low on one list
but high on the other).

### Configuration surface

- `QueryParam.enable_bm25_seeding: bool = False` (per-request override)
- Global default via env (`ENABLE_BM25_SEEDING`, read in api config like
  other query defaults) — WebUI toggle can map to the QueryParam later.
- `pyproject.toml`: `bm25s` added to an optional extra.

### Index write path

In `merge_nodes_and_edges` (operate.py), after the entity upsert batch:
rebuild once per pipeline batch from the graph's full entity-name set
(`get_all_labels()` — available on every graph backend). Deletions are
covered by the next rebuild; a stale name that BM25 still returns resolves
to a missing graph node and is dropped by the existing
"some nodes are missing" guard in `_get_node_data`, so staleness is
harmless. Indexing failures log a warning and never block document
processing — the index is derived data and can always be rebuilt.

## Error handling

1. Any exception on the BM25 path at query time (missing dep, corrupt index,
   empty index, tokenizer error) → catch, `logger.warning`, fall back to
   vector-only results. A query must never fail because of this feature.
2. Indexing failures never fail the ingestion pipeline (warning only).
3. `drop()` participates in workspace clear flows like other storages.

## Testing (per AGENTS.md)

- `tests/kg/bm25s_impl/` (new subdir + `__init__.py`): tokenizer (CJK/EN
  mixed, empty, punctuation), build/search/remove/rebuild, persistence
  round-trip, workspace isolation, missing-dependency degradation. Mock-based,
  no live services.
- Query-layer tests: RRF fusion correctness (dedup, both-lists boost,
  single-list survival), flag-off regression (behavior identical to current),
  BM25-path exception fallback.
- All green via `./scripts/test.sh`; `ruff check .` clean; code/comments/logs
  in English.

## Delivery plan

1. Post scoped-MVP comment on #3198 (parallel with development).
2. Branch `feat/bm25-entity-seeding` off `main`.
3. Commits: interface+registry → bm25s impl → query integration →
   tests → docs/env.example.
4. PR to upstream `HKUDS/LightRAG` referencing #3198: summary, motivation,
   what's changed, default-off/zero-breakage statement, test evidence.
