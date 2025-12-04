# Upstream (HKUDS/main @ f0d67f16) — features & fixes missing from this version

Summary: this file lists new features, bug fixes, CI/docs/tooling changes and important refactors that exist in upstream `HKUDS/LightRAG` main (commit f0d67f16 and its ancestors) but are not merged into this local branch. I grouped changes into functional areas and included short remediation notes where appropriate.

NOTE: upstream/main contains many small dependency bumps and documentation commits; this document focuses on substantive features and functional fixes that affect runtime behavior, storage, security, tooling and testing.

1) Core storage, DB & Postgres improvements
- Add PostgreSQL vchordrq vector index support and unify vector index creation logic (dev-postgres-vchordrq) — improves Postgres vector indexing semantics and config handling.
- Add CASCADE to AGE extension creation in Postgres init scripts (avoid failures when recreating extension)
- Add postgres_impl fixes, retry improvements and support for vchordrq epsilon config when probes empty
- Postgres RLS-related and storage refinements (note: local branch already added postgres_rls.sql; upstream brings complementary DB/VECTOR engine improvements and fixes). Remediation: merge upstream postgres/vchordrq changes and ensure migration scripts align.

2) Chunking, indexing and document ingestion fixes
- Fix top_n behavior: limit by documents instead of chunks to avoid over-counting. (important for retrieval ranking)
- Fix infinite loop when overlap_tokens >= max_tokens and edge-case handling for max_tokens == 1.
- Add comprehensive tests for chunking logic (multi-token tokenizer, recursive split) and chunking parameters tuning.
- Add content deduplication check for document insertion endpoints and fix duplicate document response handling to return original track_id. (prevents duplicates and preserves original IDs)

3) Embeddings & LLM / cloud provider support improvements
- Major improvements in OpenAI/OLLAMA/Azure/Bedrock embedding wrappers and clients:
  - Allow embedding provider defaults when unspecified
  - Add configurable embedding token limits and validation
  - Fix Azure OpenAI compatibility and support various deployments, fallback to AZURE_OPENAI_API_VERSION
  - Convert OpenAI client to use a stable API and bump minimum version (>=2.0.0)
  - Add support for structured OpenAI outputs via parsed field
  - Improve Bedrock error handling and add retry logic/custom exceptions
  - Additional refactors for embedding function wrapping rules, model param handling and function attribute inheritance
  - Add helper flags like configurable model parameter to jina_embed
  - Support async chunking functions for large, async chunkers
  - Add new LLM support, additions under lightrag/llm (e.g., gemini file added upstream)

4) Document / file extraction improvements
- DOCX/XLSX handling fixes (preserve table structure, whitespace, column alignment; optimize memory use)
- Replace PyPDF2 with pypdf for PDF processing (faster, more reliable parsing)

5) Workspace isolation, pipeline status, RAG lifecycle fixes
- Fix document deletion concurrency control and auto-acquire pipeline when idle.
- Auto-initialize pipeline status on LightRAG.initialize_storages() (reduces error-prone manual calls)
- Namespace, workspace handling and locking fixes: improvements to NamespaceLock (ContextVar), default workspace handling, filtering logic, consistent empty workspace handling and many concurrency bug fixes.

6) Web UI — upgrades, feature additions, fixes
- Large set of dependency upgrades for `lightrag_webui` (vite, react-i18next, plugin-react-swc, syntax highlighter, etc.). Upstream also cleaned duplicate deps and improved build tooling.
- Add new UI components / improvements (MergeDialog, graph features, translations updates, many components updated).
- Handle missing WebUI assets gracefully so server startup is not blocked.
- Add static swagger UI assets for API docs (swagger-ui files added upstream).

7) CI, testing, and developer tooling
- New/updated GitHub workflows and test runners: tests.yml, improved offline/integration CI markers, Copilot setup steps, docker-build* workflows and improved GitHub Actions versions.
- Drop older Python versions in test matrices (3.10/3.11 removed; 3.13/3.14 added) — keep CI modern.
- Add ruff to pytest extras, add pre-commit hooks and refine pytest fixtures and markers.
- Add many new tests including workspace isolation, chunking tests, overlap validation, postgres retry integration tests, rerank chunking tests, and E2E test improvements.

8) Tools & CLI
- New helper tools: clean_llm_query_cache.py, migrate_llm_cache.py, download_cache.py and related README docs for cleaning/migrating LLM caches.
- Add `lightrag-clean-llmqc` console script entrypoint.

9) Docs & deployment support
- Added docs: FrontendBuildGuide.md, OfflineDeployment.md, UV_LOCK_GUIDE.md and evaluation assets.
- Added Dockerfile.lite and docker-build-push.sh to support smaller builds and multi-format distribution.

10) KaTeX & math / feature parity
- Upstream adds KaTeX copy‑tex extension support and mhchem extension for chemistry formulas (enables better formula copying and chemistry rendering). Also fixed KaTeX loading in startup.

11) JSON, sanitization and performance
- Multiple JSON write/sanitizer enhancements (specialized sanitizers to handle tuples/dict keys/UTF8 errors, optimize sanitization performance) and fixes to avoid memory corruption on migrations.

12) Cloud model & misc improvements
- Improve cloud model detection/safety, macOS fork-safety check for Gunicorn multiworker cases; many small fixes for cloud model defaults and config.

13) Security / dependency hygiene
- Remove future dependency and replace passlib usage with direct bcrypt (adopt modern libs)

Actionable remediation checklist (priority):
- Merge and test upstream changes that affect: chunking, embeddings/LLM wrappers, doc processing, and Postgres vector indexing + RLS compatibility (High).
- Add or adapt DB migrations to incorporate any upstream schema changes required by tenant features and ensure no conflicts (High).
- Update CI matrix and tests to incorporate upstream tests (esp. workspace isolation and chunking tests) to verify no regressions (High).
- Merge Web UI updates separately behind feature flag/workflow (Medium) — major dependency churn.

If you want, I can automatically generate:
- a full commit-by-commit list (746 commits) in docs/diff_hku/unmerged_upstream_commits.txt (raw) — useful for exhaustive audit.
- cherry-pick safe/high-priority upstream commits onto this branch and prepare a candidate PR with resolved conflicts.
