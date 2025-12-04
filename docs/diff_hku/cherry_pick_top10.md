# Top 10 upstream commits to cherry-pick next (curated, safe order)

Goal: small, prioritized set of high-impact fixes/features to bring from upstream/main into this branch before opening a clean PR to HKUDS. Each entry has: commit hash, short purpose, why it matters here, cherry-pick command and tests to run.

Prereq: create an integration branch from upstream/main:

  git fetch upstream
  git checkout -b premerge/integration upstream/main

Apply cherry-picks in order. Resolve conflicts locally, add tests if needed, run CI locally, then push small PRs.

1) 9009abed — Fix top_n behavior with chunking (retrieval correctness)
  Why: Retrieval results rely on top_n semantics. Fix prevents over-counting chunks vs documents.
  Command: git cherry-pick 9009abed
  Tests: pytest tests/test_rerank_chunking.py && pytest tests/test_chunking.py

2) 1d6ea0c5 — Fix chunking infinite loop when overlap_tokens >= max_tokens
  Why: Critical stability fix for document processing pipelines; prevents hangs on pathological configs.
  Command: git cherry-pick 1d6ea0c5
  Tests: pytest tests/test_chunking.py::test_overlap_edge_cases

3) 19c16bc4 — Add content deduplication check for document insertion endpoints
  Why: Prevents duplicate ingestion and storage bloat — improves data correctness.
  Command: git cherry-pick 19c16bc4
  Tests: add/ensure unit test that duplicates are deduped; run e2e insertion test.

4) 8d28b959 — Fix duplicate document responses to return original track_id
  Why: Preserves client-visible IDs for duplicated docs (compatibility and dedup tracking).
  Command: git cherry-pick 8d28b959
  Tests: replay ingestion tests and verify returned track_id is original.

5) d07023c9 — feat(postgres_impl): add vchordrq vector index support
  Why: Adds Postgres vector index support; required when Postgres is used as vector DB in production.
  Command: git cherry-pick d07023c9
  Special: Might require environment or lib changes (vchordrq) and additional tests. Run Postgres integration tests: pytest tests/test_postgres_retry_integration.py

6) d6019c82 — Add CASCADE to AGE extension creation in PostgreSQL implementation
  Why: Database initialization becomes idempotent and robust when re-creating extensions (deployment reliability).
  Command: git cherry-pick d6019c82
  Tests: Create fresh DB using docker-compose, run migrations and initialization script confirming no failure.

7) 02fdceb9 — Update OpenAI client to stable API and bump minimum version to 2.0.0
  Why: Major provider change; ensures embedding/LLM clients use stable official API and match upstream examples.
  Command: git cherry-pick 02fdceb9
  Special: Requires bumping `openai` dependency and adding to CI matrix; run tests that cover OpenAI adapters.

8) 4ab4a7ac — Allow embedding models to use provider defaults when unspecified
  Why: Makes embedding config more resilient and simpler; avoids unexpected failures when settings missing.
  Command: git cherry-pick 4ab4a7ac
  Tests: ensure embedding function wrapper tests pass and add a test where provider default is used.

9) e22ac52e — Auto-initialize pipeline status in LightRAG.initialize_storages()
  Why: Reduces manual initialization errors and avoids missing pipeline state in new deployments.
  Command: git cherry-pick e22ac52e
  Tests: integration tests calling LightRAG.initialize_storages(); check pipeline status created.

10) c434879c — Replace PyPDF2 with pypdf for PDF processing
  Why: pypdf is actively maintained and fixes several parsing bugs; improves reliability of PDF doc parsing.
  Command: git cherry-pick c434879c
  Special: Update requirements; run PDF extraction tests (if present) and any E2E doc parsing tests.

Post-cherry-pick checklist for each commit:
- Run targeted unit tests and the repo full test suite where practical (pytest -q).
- Add/adjust any missing migrations or dependency changes introduced by the cherries (e.g., openai>=2.0.0, pypdf). Commit those as separate small commits.
- If a cherry-pick touches database code, verify migrations exist and add small migration files when needed.
- Push each cherry-picked & verified branch to your fork and open a small PR referencing this upstream commit and linking back to the full planned merge.

After doing the top-10 picks, re-run the repo-level test matrix and then consider the next wave (embedding refinements, JSON sanitization, workspace isolation tests).

---

Additional picks (11–25) — next wave (high impact, medium complexity)

11) 702cfd29 — Fix document deletion concurrency control and validation logic
  Why: Prevents race conditions during deletion operations and fixes validation bugs that could leak resources or leave partial deletions.
  Command: git cherry-pick 702cfd29
  Tests: run tests/test_deletion.py and concurrency-focused deletion scenarios.

12) fec7c67f — Add comprehensive chunking tests with multi-token tokenizer edge cases
  Why: Strengthens confidence in chunking pipeline; these tests capture subtleties of tokenizer behavior.
  Command: git cherry-pick fec7c67f
  Tests: pytest tests/test_chunking.py::(all chunking tests)

13) 57332925 — Add comprehensive tests for chunking with recursive splitting
  Why: Complements previous commit — ensures coverage for recursive splitting and prevents regressions.
  Command: git cherry-pick 57332925
  Tests: pytest tests/test_chunking.py

14) 3e759f46 — Add real integration and E2E tests for workspace isolation
  Why: These tests increase confidence that tenant/workspace isolation is not regressively broken after merges.
  Command: git cherry-pick 3e759f46
  Tests: pytest tests/test_workspace_isolation.py (integration + E2E variants)

15) 436e4143 — Enhance workspace isolation test suite to 100% coverage
  Why: Improves the test suite resilience and ensures changes don't break isolation guarantees.
  Command: git cherry-pick 436e4143
  Tests: pytest tests/test_workspace_isolation.py

16) 95cd0ece — Fix DOCX table extraction by escaping special characters in cells
  Why: Improves DOCX parsing reliability for complex documents and preserves table structure.
  Command: git cherry-pick 95cd0ece
  Tests: run DOCX extraction tests under tests/ and e2e doc parsing tests.

17) 0244699d — Optimize XLSX extraction by using sheet.max_column instead of two-pass scan
  Why: Reduces memory and CPU during XLSX ingestion on large spreadsheets.
  Command: git cherry-pick 0244699d
  Tests: tests covering XLSX extraction performance/behavior.

18) 2b160163 — Optimize XLSX extraction to avoid storing all rows in memory
  Why: Major memory optimization for XLSX ingestion for large files.
  Command: git cherry-pick 2b160163
  Tests: add a large-file XLSX test or run existing XLSX tests.

19) 23cbb9c9 — Add data sanitization to JSON writing to prevent UTF-8 encoding errors
  Why: Prevents crashes/data corruption when storing untrusted metadata or malformed JSON.
  Command: git cherry-pick 23cbb9c9
  Tests: tests/test_write_json_optimization.py and new tests for edge-case JSON characters.

20) fc44f113 — Remove future dependency and replace passlib with direct bcrypt
  Why: Dependency hygiene/security improvement — reduces transitive legacy deps and uses modern bcrypt.
  Command: git cherry-pick fc44f113
  Special: Update requirements/lockfiles and run auth-related tests.

21) e5addf4d — Improve embedding config priority and add debug logging
  Why: Embedding config precedence issues can cause confusing failures — this improves observability.
  Command: git cherry-pick e5addf4d
  Tests: run embedding tests and check logs for clarity.

22) 6e2946e7 — Add max_token_size parameter to azure_openai_embed wrapper
  Why: Azure OpenAI compatibility and safety for embeddings with token limits.
  Command: git cherry-pick 6e2946e7
  Tests: tests for azure_openai wrapper and edge cases where token limits matter.

23) 4ea21240 — Add GitHub CI workflow and test markers for offline/integration tests
  Why: Brings a repeatable CI job to run offline/integration markers — needed to run new tests reliably.
  Command: git cherry-pick 4ea21240
  Tests: Validate new CI job config in a branch (dry-run); locally run tests that use markers.

24) ff8f1588 — Update env.example (important fixes/clarifications)
  Why: Env example updates reduce deployment mistakes — low complexity, important for ops.
  Command: git cherry-pick ff8f1588
  Tests: manual verification of environment examples and readme changes.

25) f72f435c — Fix chunk size handling (stability/regression prevention)
  Why: Prevents accidental misconfiguration of chunk sizes and related errors.
  Command: git cherry-pick f72f435c
  Tests: pytest tests/test_chunking.py variations and edge-case runs.

Post-cherry-pick notes: This round focuses on tests, chunking stability, workspace isolation tests, DOCX/XLSX memory improvements, JSON sanitizer, and dependency hygiene. After applying, re-run the suite and promote problematic commits into small follow-up PRs (with migration files if DB changes are required).


If you want, I can:
- Apply these cherry-picks sequentially into a `premerge/integration` branch here and run the tests. (Option B)
- Or produce shell script that runs cherry-picks and tests locally so you can run it on your machine.
