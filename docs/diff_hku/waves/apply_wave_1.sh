#!/usr/bin/env bash
# Auto-generated script to apply Wave 1 commits
set -e

echo "Cherry-picking 6a8de2ed: web_ui: check node source and target"
git cherry-pick -x 6a8de2ed

echo "Cherry-picking 91387628: Add test script for aquery_data endpoint validation"
git cherry-pick -x 91387628

echo "Cherry-picking dab1c358: Optimize chat performance by reducing animations in inactive tabs"
git cherry-pick -x dab1c358

echo "Cherry-picking 9e5004e2: fix(docs): correct typo \"acivate\" → \"activate\""
git cherry-pick -x 9e5004e2

echo "Cherry-picking e2ec1cdc: Merge pull request #2258 from danielaskdd/pipeline-cancelllation"
git cherry-pick -x e2ec1cdc

echo "Cherry-picking 626b42bc: feat: add optional Langfuse observability integration"
git cherry-pick -x 626b42bc

echo "Cherry-picking 5cdb4b0e: fix: Apply ruff formatting and rename test_dataset to sample_dataset"
git cherry-pick -x 5cdb4b0e

echo "Cherry-picking b12b693a: fixed ruff format of csv path"
git cherry-pick -x b12b693a

echo "Cherry-picking 10f6e695: Improve Langfuse integration and stream response cleanup handling"
git cherry-pick -x 10f6e695

echo "Cherry-picking 2fdb5f5e: chore: trigger CI re-run 2"
git cherry-pick -x 2fdb5f5e

echo "Cherry-picking 36bffe22: chore: trigger CI re-run"
git cherry-pick -x 36bffe22

echo "Cherry-picking a172cf89: feat(evaluation): Add sample documents for reproducible RAGAS testing"
git cherry-pick -x a172cf89

echo "Cherry-picking a790f081: Refine gitignore to only exclude root-level test files"
git cherry-pick -x a790f081

echo "Cherry-picking b0d44d28: Add Langfuse observability integration documentation"
git cherry-pick -x b0d44d28

echo "Cherry-picking 2f160652: Refactor keyword_extraction from kwargs to explicit parameter"
git cherry-pick -x 2f160652

echo "Cherry-picking 5885637e: Add specialized JSON string sanitizer to prevent UTF-8 encoding errors"
git cherry-pick -x 5885637e

echo "Cherry-picking 6918a88f: Add specialized JSON string sanitizer to prevent UTF-8 encoding errors"
git cherry-pick -x 6918a88f

echo "Cherry-picking 7d394fb0: Replace asyncio.iscoroutine with inspect.isawaitable for better detection"
git cherry-pick -x 7d394fb0

echo "Cherry-picking a24d8181: Improve docling integration with macOS compatibility and CLI flag"
git cherry-pick -x a24d8181

echo "Cherry-picking c230d1a2: Replace asyncio.iscoroutine with inspect.isawaitable for better detection"
git cherry-pick -x c230d1a2

echo "Cherry-picking c246eff7: Improve docling integration with macOS compatibility and CLI flag"
git cherry-pick -x c246eff7

echo "Cherry-picking 1a183702: docs: Update test file docstring to reflect all 11 test scenarios"
git cherry-pick -x 1a183702

echo "Cherry-picking 288498cc: test: Convert test_workspace_isolation.py to pytest style"
git cherry-pick -x 288498cc

echo "Cherry-picking 3806892a: Merge pull request #2371 from BukeLy/pytest-style-conversion"
git cherry-pick -x 3806892a

echo "Cherry-picking 3e759f46: test: Add real integration and E2E tests for workspace isolation"
git cherry-pick -x 3e759f46

echo "Cherry-picking 3ec73693: test: Enhance E2E workspace isolation detection with content verification"
git cherry-pick -x 3ec73693

echo "Cherry-picking 4742fc8e: test: Add comprehensive workspace isolation test suite for PR #2366"
git cherry-pick -x 4742fc8e

echo "Cherry-picking b7b8d156: Refactor pytest dependencies into separate optional group"
git cherry-pick -x b7b8d156

echo "Cherry-picking cf73cb4d: Remove unused variables from workspace isolation test"
git cherry-pick -x cf73cb4d

echo "Cherry-picking 1fe05df2: Refactor test configuration to use pytest fixtures and CLI options"
git cherry-pick -x 1fe05df2

echo "Cherry-picking 41bf6d02: Fix test to use default workspace parameter behavior"
git cherry-pick -x 41bf6d02

echo "Cherry-picking 472b498a: Replace pytest group reference with explicit dependencies in evaluation"
git cherry-pick -x 472b498a

echo "Cherry-picking 4ea21240: Add GitHub CI workflow and test markers for offline/integration tests"
git cherry-pick -x 4ea21240

echo "Cherry-picking 4fef731f: Standardize test directory creation and remove tempfile dependency"
git cherry-pick -x 4fef731f

echo "Cherry-picking 5da82bb0: Add pre-commit to pytest dependencies and format test code"
git cherry-pick -x 5da82bb0

echo "Cherry-picking 6ae0c144: test: add concurrent execution to workspace isolation test"
git cherry-pick -x 6ae0c144

echo "Cherry-picking 7e9c8ed1: Rename test classes to prevent warning from pytest"
git cherry-pick -x 7e9c8ed1

echo "Cherry-picking 99262ada: Enhance workspace isolation test with distinct mock data and persistence"
git cherry-pick -x 99262ada

echo "Cherry-picking 0fb2925c: Remove ascii_colors dependency and fix stream handling errors"
git cherry-pick -x 0fb2925c

echo "Cherry-picking d52adb64: Merge pull request #2390 from danielaskdd/fix-pytest-logging-error"
git cherry-pick -x d52adb64

echo "Cherry-picking 5f91063c: Add ruff as dependency to pytest and evaluation extras"
git cherry-pick -x 5f91063c

echo "Cherry-picking 90e38c20: Clauss Keep GitHub Actions up to date with GitHub's Dependabot"
git cherry-pick -x 90e38c20

echo "Cherry-picking 88357675: Bump the github-actions group with 7 updates"
git cherry-pick -x 88357675

echo "Cherry-picking 0f19f80f: Configure comprehensive Dependabot for Python and frontend dependencies"
git cherry-pick -x 0f19f80f

echo "Cherry-picking 1f875122: Drop Python 3.10 and 3.11 from CI test matrix"
git cherry-pick -x 1f875122

echo "Cherry-picking 268e4ff6: Refactor dependencies and add test extra in pyproject.toml"
git cherry-pick -x 268e4ff6

echo "Cherry-picking 64760216: Configure Dependabot schedule with specific times and timezone"
git cherry-pick -x 64760216

echo "Cherry-picking d0509d6f: Merge pull request #2448 from HKUDS/dependabot/github_actions/github-actions-b6ffb444c9"
git cherry-pick -x d0509d6f

echo "Cherry-picking ecef842c: Update GitHub Actions to use latest versions (v6)"
git cherry-pick -x ecef842c

echo "Cherry-picking 561ba4e4: Fix trailing whitespace and update test mocking for rerank module"
git cherry-pick -x 561ba4e4

echo "Cherry-picking a31192dd: Update i18n file for pipeline UI text across locales"
git cherry-pick -x a31192dd

echo "Cherry-picking a5253244: Simplify skip logging and reduce pipeline status updates"
git cherry-pick -x a5253244

echo "Cherry-picking 743aefc6: Add pipeline cancellation feature for graceful processing termination"
git cherry-pick -x 743aefc6

echo "Cherry-picking 77336e50: Improve error handling and add cancellation checks in pipeline"
git cherry-pick -x 77336e50

echo "Cherry-picking f89b5ab1: Add pipeline cancellation feature with UI and i18n support"
git cherry-pick -x f89b5ab1

echo "Cherry-picking 81e3496a: Add confirmation dialog for pipeline cancellation"
git cherry-pick -x 81e3496a

echo "Cherry-picking 5f4a2804: Add Qdrant legacy collection migration with workspace support"
git cherry-pick -x 5f4a2804

echo "Cherry-picking c36afecb: Remove redundant await call in file extraction pipeline"
git cherry-pick -x c36afecb

echo "Cherry-picking 58c83f9d: Add auto-refresh of popular labels when pipeline completes"
git cherry-pick -x 58c83f9d

echo "Cherry-picking eb52ec94: feat: Add workspace isolation support for pipeline status"
git cherry-pick -x eb52ec94

echo "Cherry-picking 18a48702: fix: Add default workspace support for backward compatibility"
git cherry-pick -x 18a48702

echo "Cherry-picking 52c812b9: Fix workspace isolation for pipeline status across all operations"
git cherry-pick -x 52c812b9

echo "Cherry-picking 78689e88: Fix pipeline status namespace check to handle root case"
git cherry-pick -x 78689e88

echo "Cherry-picking 7ed0eac4: Fix workspace filtering logic in get_all_update_flags_status"
git cherry-pick -x 7ed0eac4

echo "Cherry-picking 926960e9: Refactor workspace handling to use default workspace and namespace locks"
git cherry-pick -x 926960e9

echo "Cherry-picking 9d7b7981: Add pipeline status validation before document deletion"
git cherry-pick -x 9d7b7981

echo "Cherry-picking b6a5a90e: Fix NamespaceLock concurrent coroutine safety with ContextVar"
git cherry-pick -x b6a5a90e

echo "Cherry-picking cdd53ee8: Remove manual initialize_pipeline_status() calls across codebase"
git cherry-pick -x cdd53ee8

echo "Cherry-picking ddc76f0c: Merge branch 'main' into workspace-isolation"
git cherry-pick -x ddc76f0c

echo "Cherry-picking e8383df3: Fix NamespaceLock context variable timing to prevent lock bricking"
git cherry-pick -x e8383df3

echo "Cherry-picking f1d8f18c: Merge branch 'main' into workspace-isolation"
git cherry-pick -x f1d8f18c

echo "Cherry-picking 1745b30a: Fix missing workspace parameter in update flags status call"
git cherry-pick -x 1745b30a

echo "Cherry-picking 21ad990e: Improve workspace isolation tests with better parallelism checks and cleanup"
git cherry-pick -x 21ad990e

echo "Cherry-picking 4048fc4b: Fix: auto-acquire pipeline when idle in document deletion"
git cherry-pick -x 4048fc4b

echo "Cherry-picking dfbc9736: Merge pull request #2369 from HKUDS/workspace-isolation"
git cherry-pick -x dfbc9736

echo "Cherry-picking f8dd2e07: Fix namespace parsing when workspace contains colons"
git cherry-pick -x f8dd2e07

echo "Cherry-picking 93d445df: Add pipeline status lock function for legacy compatibility"
git cherry-pick -x 93d445df

echo "Cherry-picking aac787ba: Clarify chunk tracking log message in _build_llm_context"
git cherry-pick -x aac787ba

echo "Cherry-picking 8eb0f83e: Simplify Vite build config by removing manual chunking strategy"
git cherry-pick -x 8eb0f83e

echo "Cherry-picking 35cd567c: Allow related chunks missing in knowledge graph queries"
git cherry-pick -x 35cd567c

echo "Cherry-picking dc62c78f: Add entity/relation chunk tracking with configurable source ID limits"
git cherry-pick -x dc62c78f

echo "Cherry-picking 29bf5936: Fix entity and relation chunk cleanup in deletion pipeline"
git cherry-pick -x 29bf5936

echo "Cherry-picking 3fbd704b: Enhance entity/relation editing with chunk tracking synchronization"
git cherry-pick -x 3fbd704b

echo "Cherry-picking a3370b02: Add chunk tracking cleanup to entity/relation deletion and creation"
git cherry-pick -x a3370b02

echo "Cherry-picking 2c09adb8: Add chunk tracking support to entity merge functionality"
git cherry-pick -x 2c09adb8

echo "Cherry-picking c81a56a1: Fix entity and relationship deletion when no chunk references remain"
git cherry-pick -x c81a56a1

echo "Cherry-picking 0bbef981: Optimize RAGAS evaluation with parallel execution and chunk content enrichment"
git cherry-pick -x 0bbef981

echo "Cherry-picking 963ad4c6: docs: Add documentation and examples for include_chunk_content parameter"
git cherry-pick -x 963ad4c6

echo "Cherry-picking 6b0f9795: Add workspace parameter and remove chunk-based query unit tests"
git cherry-pick -x 6b0f9795

echo "Cherry-picking 807d2461: Remove unused chunk-based node/edge retrieval methods"
git cherry-pick -x 807d2461

echo "Cherry-picking ea141e27: Fix: Remove redundant entity/relation chunk deletions"
git cherry-pick -x ea141e27

echo "Cherry-picking 77405006: Da support async chunking func to improve processing performance when a heavy `chunking_func` is passed in by user"
git cherry-pick -x 77405006

echo "Cherry-picking d137ba58: Da support async chunking func to improve processing performance when a heavy `chunking_func` is passed in by user"
git cherry-pick -x d137ba58

echo "Cherry-picking 245df75d: Da easier version: detect chunking_func result is coroutine or not"
git cherry-pick -x 245df75d

echo "Cherry-picking 50160254: Da easier version: detect chunking_func result is coroutine or not"
git cherry-pick -x 50160254

echo "Cherry-picking 940bec0b: Support async chunking functions in LightRAG processing pipeline"
git cherry-pick -x 940bec0b

echo "Cherry-picking af542391: Support async chunking functions in LightRAG processing pipeline"
git cherry-pick -x af542391

echo "Cherry-picking 1bfa1f81: Merge branch 'main' into fix_chunk_comment"
git cherry-pick -x 1bfa1f81

echo "Cherry-picking 24423c92: Merge branch 'fix_chunk_comment'"
git cherry-pick -x 24423c92

echo "Cherry-picking dacca334: refactor(chunking): rename params and improve docstring for chunking_by_token_size"
git cherry-pick -x dacca334

echo "Cherry-picking e77340d4: Adjust chunking parameters to match the default environment variable settings"
git cherry-pick -x e77340d4

echo "Cherry-picking 57332925: Add comprehensive tests for chunking with recursive splitting"
git cherry-pick -x 57332925

echo "Cherry-picking 6fea68bf: Fix ChunkTokenLimitExceededError message formatting"
git cherry-pick -x 6fea68bf

echo "Cherry-picking f72f435c: Merge pull request #2389 from danielaskdd/fix-chunk-size"
git cherry-pick -x f72f435c

echo "Cherry-picking f988a226: Add token limit validation for character-only chunking"
git cherry-pick -x f988a226

echo "Cherry-picking fec7c67f: Add comprehensive chunking tests with multi-token tokenizer edge cases"
git cherry-pick -x fec7c67f

echo "Cherry-picking a05bbf10: Add Cohere reranker config, chunking, and tests"
git cherry-pick -x a05bbf10

echo "Cherry-picking 1d6ea0c5: Fix chunking infinite loop when overlap_tokens >= max_tokens"
git cherry-pick -x 1d6ea0c5

echo "Cherry-picking 9009abed: Fix top_n behavior with chunking to limit documents not chunks"
git cherry-pick -x 9009abed

echo "Cherry-picking 54f0a7d1: Quick fix to limit source_id ballooning while inserting nodes"
git cherry-pick -x 54f0a7d1

echo "Cherry-picking 7871600d: Quick fix to limit source_id ballooning while inserting nodes"
git cherry-pick -x 7871600d

echo "Cherry-picking 1154c568: Refactor deduplication calculation and remove unused variables"
git cherry-pick -x 1154c568

echo "Cherry-picking a25003c3: Fix relation deduplication logic and standardize log message prefixes"
git cherry-pick -x a25003c3

echo "Cherry-picking 19c16bc4: Add content deduplication check for document insertion endpoints"
git cherry-pick -x 19c16bc4

echo "Cherry-picking 459e4ddc: Clean up duplicate dependencies in package.json and lock file"
git cherry-pick -x 459e4ddc

echo "Cherry-picking 8d28b959: Fix duplicate document responses to return original track_id"
git cherry-pick -x 8d28b959

