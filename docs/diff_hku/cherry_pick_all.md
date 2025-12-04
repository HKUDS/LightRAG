# Comprehensive premerge plan — map of all upstream features / fixes

Purpose
- This document expands the curated top-25 cherry-pick plan to a complete, review-friendly roadmap covering *all* functional areas present in upstream/main (the full commit list is saved as docs/diff_hku/unmerged_upstream_commits.txt).
- It does NOT automatically cherry-pick all 700+ commits. Instead it groups features/fixes, identifies safe integration order, test targets, migration notes, and the minimal follow-up actions needed for merging the whole upstream delta into your branch in safe waves.

How to use:
- Use the per-category sections below to plan incremental merges / PRs.
- Each group lists the types of commits present upstream, what they change, which tests you must run, and integration notes.
- For commit-level details, consult docs/diff_hku/unmerged_upstream_commits.txt which contains the raw upstream commit list (746 lines).

High-level merge waves (summary)
- Wave 0 — Safety & infra: DB migrations & RLS handling, secrets (token), CI for integration tests, and DB driver session wiring.
- Wave 1 — Stability & correctness: chunking fixes, document dedup, dedup ID handling, content parsers (DOCX/XLSX/PDF) and JSON sanitization.
- Wave 2 — Embeddings & LLMs: OpenAI/OLLAMA/Azure/Bedrock wrappers, embedding config, token limits and provider defaults; bump openai client if required.
- Wave 3 — Storage & vector DBs: Postgres vchordrq, faiss/milvus/qdrant adjustments, vector indexing and retry logic.
- Wave 4 — Workspace & lifecycle: workspace isolation tests, pipeline status, namespace locks and RAG lifecycle improvements.
- Wave 5 — Tests & CI: add upstream tests into CI, update matrix, add offline/integration jobs.
- Wave 6 — Web UI & tooling: web UI dependency updates, new components and docs; merge separately or behind flag to reduce churn.
- Wave 7 — Cleanups: dependency hygiene, docs, small refactors and non-critical enhancements.

Category-by-category (full integration plan)

1) Chunking & tokenization (critical)
- What upstream changed: top_n behavior fix, infinite-loop fix when overlap_tokens >= max_tokens, token-limit validation, recursive split tests and many test additions.
- Why it matters: chunking affects indexing, retrieval correctness and can cause hangs or wrong search results.
- Merge plan: bring all chunking fixes first (top_n + overlap fixes), add full chunking tests and regressions to CI. Validate across tokenizers used in tests.
- Tests: tests/test_chunking.py, tests/test_rerank_chunking.py, new chunking-focused tests in upstream.
- Risk: low-medium; mainly algorithmic / test coverage. Keep refactors separate if they touch storage.

2) Document ingestion & deduplication (high priority)
- What upstream changed: content dedup checks on insertion, duplicate track_id return fixes, DOCX/XLSX table handling improvements and PDF migration to pypdf, XLSX memory optimizations.
- Why it matters: correctness of ingestion and storage footprint; fixes prevent duplicate documents and structural loss for tables.
- Merge plan: merge dedup + duplicate-id fixes + DOCX/XLSX/PDF changes together. Add E2E ingestion tests using sample documents from upstream/evaluation samples.
- Tests: tests for ingestion endpoints, e2e parsing tests, sample docs under lightrag/evaluation.

3) Embeddings & LLM adapters (high impact)
- What upstream changed: provider-default rules for embeddings, configurable token limits (Azure), OpenAI client upgrade to v2+, structured output support (parsed), new supports for jina/embed wrappers and integration notes.
- Why it matters: embedding quality, API compatibility and provider stability.
- Merge plan: apply embedding config fixes + OpenAI client upgrade in a separate PR; confirm dependency bump and run embedding adapter tests.
- Tests: embedding unit tests, e2e embedding flows, cloud-provider-specific wrappers (azure/openai/batch modes).

4) Postgres & vector DB improvements (high priority for PG deployments)
- What upstream changed: vchordrq vector index support, AGE extension CASCADE, Postgres retry/instrumentation, various storage fixes and improvements.
- Why it matters: important for Postgres-backed vector DBs, migrations and RLS compatibility.
- Merge plan: ensure DB migrations are present before merging code that expects new schemas or behavior; add integration tests against a Postgres test container with vchordrq/AGE; ensure session var management for RLS.
- Tests: pytest tests/test_postgres_retry_integration.py and any DB-heavy E2E tests.

5) Workspace isolation, pipeline status & concurrency safety (high)
- What upstream changed: namespace locks (ContextVar) safety, default workspace handling, auto-initialize pipeline status, deletion concurrency fixes and robust isolation tests.
- Why it matters: prevents correctness/consistency/DoS bugs across tenants/workspaces.
- Merge plan: merge isolation tests and small fixes early; add E2E tests and load test simulation for race conditions.

6) Tests, CI & developer tooling (medium-high)
- What upstream changed: new CI workflows, offline/integration markers, tests improvements (workspaces, chunking), updated matrices (Python 3.13/3.14), and dependabot hygiene.
- Why it matters: keeps repo quality verifiable and avoids regressions.
- Merge plan: add CI changes to integration branch early to ensure subsequent PRs run tests properly; keep frontend CI separate where needed.

7) Web UI & frontend (medium)
- What upstream changed: many dependency bumps (vite/react-i18next), UI components added/changed, static swagger assets added, handle missing webui assets gracefully.
- Why it matters: high churn and large PR surface — do separately to keep backend merge risk low.
- Merge plan: open dedicated PR for webui dependency upgrades and new components; keep it independent from backend changes.

8) Tools / scripts / CLI (low-medium)
- What upstream changed: cache-clean tools, migrate_llm_cache, download_cache, CLI entrypoints and helpful helpers.
- Why it matters: operational utilities that help migrations and maintenance.
- Merge plan: merge these small tools after core infra stabilization.

9) JSON sanitization, performance & data hygiene (medium)
- What upstream changed: improved JSON sanitizers, UTF-8 fixes, memory optimization for JSON write paths.
- Why it matters: prevents corruption and memory issues on large or malformed inputs.
- Merge plan: merge with ingestion fixes and run heavy-data tests.

10) KaTeX & rendering enhancements (low)
- What upstream changed: KaTeX copy-tex extension, mhchem support, startup import fixes.
- Merge plan: safe to merge after core features as low-risk.

11) Dependency hygiene & small refactors (ongoing)
- What upstream changed: many dependabot bumps, remove "future" dependency and switch passlib->bcrypt, grooming pyproject and extras.
- Merge plan: merge progressively; keep CI green after each bump.

12) Misc / smaller improvements
- Cloud model detection, macOS Gunicorn safety, neo4j retry decorator/resilience, and other incremental robustness changes.

Automated next-step options (pick one)
- Option A (recommended): follow the merge waves above and implement small PRs per bullet (one category = several narrow PRs). This yields safe, reviewable commits and clear rollback paths.
- Option B: create an automated script to cherry-pick all commits from docs/diff_hku/unmerged_upstream_commits.txt into a `premerge/integration` branch and run tests. This is riskier — expect many conflicts and manual resolution.

Practical guidance for full merge (how to convert this map to actions):
1) Create `premerge/integration` from upstream/main.
2) Add migration PR (Wave 0) that includes DB SQL, session escrows for RLS and strict config toggles.
3) Add CI workflows to run newly-added upstream tests (Wave 5).
4) Merge chunking + ingestion fixes and corresponding tests (Wave 1).
5) Merge embed/LLM + client upgrades (Wave 2) with dependency bumps.
6) Merge Postgres/vector improvements + storage tests (Wave 3).
7) Merge workspace-isolation/pipeline fixes and add integration tests (Wave 4).
8) Merge web UI separately (Wave 6) once backend has stabilized.
9) Gradually merge remaining minor items and dependency bumps (Wave 7).

Appendix & raw commit list
- Full raw upstream commits: docs/diff_hku/unmerged_upstream_commits.txt (746 commits) — use this as the authoritative per-commit mapping.

If you want I can:
- Generate a per-commit mapping file that assigns every upstream commit to one of the categories above (CSV) so you can cherry-pick in precise order, or
- Attempt an automated run here to apply the first N commits in the prioritized list into a `premerge/integration` branch and run tests. (I recommend small batches first.)
