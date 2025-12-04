#!/usr/bin/env bash
# Auto-generated script to apply Wave 0 commits
set -e

echo "Cherry-picking 72db0426: Update .env loading and add API authentication to RAG evaluator"
git cherry-pick -x 72db0426

echo "Cherry-picking dd18eb5b: Merge pull request #3 from netbrah/copilot/fix-overlap-tokens-validation"
git cherry-pick -x dd18eb5b

echo "Cherry-picking fc44f113: Remove future dependency and replace passlib with direct bcrypt"
git cherry-pick -x fc44f113

echo "Cherry-picking 108cdbe1: feat: add options for PostGres connection"
git cherry-pick -x 108cdbe1

echo "Cherry-picking 457d5195: Add doc_name field to full docs storage"
git cherry-pick -x 457d5195

echo "Cherry-picking 119d2fa1: Cyran Adding support for imagePullSecrets, envFrom, and deployment strategy in Helm chart"
git cherry-pick -x 119d2fa1

echo "Cherry-picking 22a7b482: fix: renamed PostGreSQL options env variable and allowed LRU cache to be an optional env variable"
git cherry-pick -x 22a7b482

echo "Cherry-picking dd8f44e6: Vujić Fixed typo in log message when creating new graph file"
git cherry-pick -x dd8f44e6

echo "Cherry-picking f2c0b41e: Make PostgreSQL statement_cache_size configuration optional"
git cherry-pick -x f2c0b41e

echo "Cherry-picking 577b9e68: Add project intelligence files for AI agent collaboration"
git cherry-pick -x 577b9e68

echo "Cherry-picking 1bf802ee: Add AGENTS.md documentation section for AI coding agent guidance"
git cherry-pick -x 1bf802ee

echo "Cherry-picking 6e39c0c0: Rename Agments.md to AGENTS.md and standardize formatting"
git cherry-pick -x 6e39c0c0

echo "Cherry-picking 8d3b53ce: Condensed AGENTS.md to focus on essential development guidelines"
git cherry-pick -x 8d3b53ce

echo "Cherry-picking b3ed2647: Refactor PostgreSQL retry config to use centralized configuration"
git cherry-pick -x b3ed2647

echo "Cherry-picking b4d61eb8: Merge pull request #2192 from danielaskdd/postgres-network-retry"
git cherry-pick -x b4d61eb8

echo "Cherry-picking bd535e3e: Add PostgreSQL connection retry configuration options"
git cherry-pick -x bd535e3e

echo "Cherry-picking e758204a: Add PostgreSQL connection retry mechanism with comprehensive error handling"
git cherry-pick -x e758204a

echo "Cherry-picking 82397834: Merge pull request #2195 from danielaskdd/hotfix-postgres"
git cherry-pick -x 82397834

echo "Cherry-picking 9be22dd6: Preserve ordering in get_by_ids methods across all storage implementations"
git cherry-pick -x 9be22dd6

echo "Cherry-picking a5c05f1b: Add offline deployment support with cache management and layered deps"
git cherry-pick -x a5c05f1b

echo "Cherry-picking bc1a70ba: Remove explicit protobuf dependency from offline storage requirements"
git cherry-pick -x bc1a70ba

echo "Cherry-picking fbcc35bb: Merge branch 'hotfix-postgres'"
git cherry-pick -x fbcc35bb

echo "Cherry-picking e5cbc593: Optimize Docker build with multi-stage frontend compilation"
git cherry-pick -x e5cbc593

echo "Cherry-picking 19c05f9e: Add static 'offline' tag to Docker image metadata"
git cherry-pick -x 19c05f9e

echo "Cherry-picking 1fd02b18: Merge pull request #2222 from danielaskdd/offline-docker-image"
git cherry-pick -x 1fd02b18

echo "Cherry-picking 388dce2e: docs: clarify docling exclusion in offline Docker image"
git cherry-pick -x 388dce2e

echo "Cherry-picking 466de207: Migrate from pip to uv package manager for faster builds"
git cherry-pick -x 466de207

echo "Cherry-picking 65c2eb9f: Migrate Dockerfile from pip to uv package manager for faster builds"
git cherry-pick -x 65c2eb9f

echo "Cherry-picking daeca17f: Change default docker image to offline version"
git cherry-pick -x daeca17f

echo "Cherry-picking ef79821f: Add build script for multi-platform images"
git cherry-pick -x ef79821f

echo "Cherry-picking f2b6a068: Remove docling dependency and related packages from project"
git cherry-pick -x f2b6a068

echo "Cherry-picking 53240041: Park remove deprecated dotenv package."
git cherry-pick -x 53240041

echo "Cherry-picking 917e41aa: Verma Refactor SQL queries and improve input handling in PGKVStorage and PGDocStatusStorage"
git cherry-pick -x 917e41aa

echo "Cherry-picking c0f69395: Merge branch 'security/fix-sql-injection-postgres'"
git cherry-pick -x c0f69395

echo "Cherry-picking e0fd31a6: Fix logging message formatting"
git cherry-pick -x e0fd31a6

echo "Cherry-picking 019dff52: Update truncation message format in properties tooltip"
git cherry-pick -x 019dff52

echo "Cherry-picking 2f22336a: Rangana Optimize PostgreSQL initialization performance"
git cherry-pick -x 2f22336a

echo "Cherry-picking 90720471: Merge pull request #2237 from yrangana/feat/optimize-postgres-initialization"
git cherry-pick -x 90720471

echo "Cherry-picking 00aa5e53: Improve entity identifier truncation warning message format"
git cherry-pick -x 00aa5e53

echo "Cherry-picking 2476d6b7: Simplify pipeline status dialog by consolidating message sections"
git cherry-pick -x 2476d6b7

echo "Cherry-picking a97e5dad: Optimize PostgreSQL graph queries to avoid Cypher overhead and complexity"
git cherry-pick -x a97e5dad

echo "Cherry-picking a9ec15e6: Resolve lock leakage issue during user cancellation handling"
git cherry-pick -x a9ec15e6

echo "Cherry-picking 8584980e: refactor: Qdrant Multi-tenancy (Include staged)"
git cherry-pick -x 8584980e

echo "Cherry-picking 0692175c: Remove enable_logging parameter from get_data_init_lock call in MilvusVectorDBStorage"
git cherry-pick -x 0692175c

echo "Cherry-picking ec797276: Merge pull request #2279 from danielaskdd/fix-edge-merge-stage"
git cherry-pick -x ec797276

echo "Cherry-picking ee7c683f: Fix swagger docs page problem in dev mode"
git cherry-pick -x ee7c683f

echo "Cherry-picking 16d3d82a: Include static files in package distribution"
git cherry-pick -x 16d3d82a

echo "Cherry-picking 9a8742da: Improve entity merge logging by removing redundant message and fixing typo"
git cherry-pick -x 9a8742da

echo "Cherry-picking 6d4a5510: Remove redundant shutdown message from gunicorn"
git cherry-pick -x 6d4a5510

echo "Cherry-picking 98f0464a: moussa anouar Update lightrag/evaluation/eval_rag_quality.py for launguage"
git cherry-pick -x 98f0464a

echo "Cherry-picking 83715a3a: Implement two-stage pipeline for RAG evaluation with separate semaphores"
git cherry-pick -x 83715a3a

echo "Cherry-picking e5abe9dd: Restructure semaphore control to manage entire evaluation pipeline"
git cherry-pick -x e5abe9dd

echo "Cherry-picking 36501b82: Initialize shared storage for all graph storage types in graph unit test"
git cherry-pick -x 36501b82

echo "Cherry-picking f3b2ba81: Translate graph storage test from Chinese to English"
git cherry-pick -x f3b2ba81

echo "Cherry-picking f6a0ea3a: Merge pull request #2320 from danielaskdd/fix-postgres"
git cherry-pick -x f6a0ea3a

echo "Cherry-picking 5c0ced6e: Fix spelling errors in the \"使用PostgreSQL存储\" section of README-zh.md"
git cherry-pick -x 5c0ced6e

echo "Cherry-picking 1a91bcdb: Improve storage config validation and add config.ini fallback support"
git cherry-pick -x 1a91bcdb

echo "Cherry-picking 5be04263: Fix deadlock in JSON cache migration and prevent same storage selection"
git cherry-pick -x 5be04263

echo "Cherry-picking b72632e4: Add async generator lock management rule to cline extension"
git cherry-pick -x b72632e4

echo "Cherry-picking e95b02fb: Refactor storage selection UI with dynamic numbering and inline prompts"
git cherry-pick -x e95b02fb

echo "Cherry-picking 7bc6ccea: Add uv package manager support to installation docs"
git cherry-pick -x 7bc6ccea

echo "Cherry-picking 913fa1e4: Add concurrency warning for JsonKVStorage in cleanup tool"
git cherry-pick -x 913fa1e4

echo "Cherry-picking 777c9873: Optimize JSON write with fast/slow path to reduce memory usage"
git cherry-pick -x 777c9873

echo "Cherry-picking f289cf62: Optimize JSON write with fast/slow path to reduce memory usage"
git cherry-pick -x f289cf62

echo "Cherry-picking 4401f86f: Refactor exception handling in MemgraphStorage label methods"
git cherry-pick -x 4401f86f

echo "Cherry-picking 8283c86b: Refactor exception handling in MemgraphStorage label methods"
git cherry-pick -x 8283c86b

echo "Cherry-picking 393f8803: Improve LightRAG initialization checker tool with better usage docs"
git cherry-pick -x 393f8803

echo "Cherry-picking 436e4143: test: Enhance workspace isolation test suite to 100% coverage"
git cherry-pick -x 436e4143

echo "Cherry-picking 6d6716e9: Add _default_workspace to shared storage finalization"
git cherry-pick -x 6d6716e9

echo "Cherry-picking 7deb9a64: Refactor namespace lock to support reusable async context manager"
git cherry-pick -x 7deb9a64

echo "Cherry-picking d54d0d55: Standardize empty workspace handling from \"_\" to \"\" across storage"
git cherry-pick -x d54d0d55

echo "Cherry-picking e22ac52e: Auto-initialize pipeline status in LightRAG.initialize_storages()"
git cherry-pick -x e22ac52e

echo "Cherry-picking fd486bc9: Refactor storage classes to use namespace instead of final_namespace"
git cherry-pick -x fd486bc9

echo "Cherry-picking 3096f844: fix(postgres): allow vchordrq.epsilon config when probes is empty"
git cherry-pick -x 3096f844

echo "Cherry-picking 6cef8df1: Reduce log level and improve workspace mismatch message clarity"
git cherry-pick -x 6cef8df1

echo "Cherry-picking 9109509b: Merge branch 'dev-postgres-vchordrq'"
git cherry-pick -x 9109509b

echo "Cherry-picking b583b8a5: Merge branch 'feature/postgres-vchordrq-indexes' into dev-postgres-vchordrq"
git cherry-pick -x b583b8a5

echo "Cherry-picking d07023c9: feat(postgres_impl): add vchordrq vector index support and unify vector index creation logic"
git cherry-pick -x d07023c9

echo "Cherry-picking dbae327a: Merge branch 'main' into dev-postgres-vchordrq"
git cherry-pick -x dbae327a

echo "Cherry-picking f4bf5d27: fix: add logger to configure_vchordrq() and format code"
git cherry-pick -x f4bf5d27

echo "Cherry-picking 5cc91686: Expand AGENTS.md with testing controls and automation guidelines"
git cherry-pick -x 5cc91686

echo "Cherry-picking 1e415cff: Update postgreSQL docker image link"
git cherry-pick -x 1e415cff

echo "Cherry-picking 8835fc24: Improve edge case handling for max_tokens=1"
git cherry-pick -x 8835fc24

echo "Cherry-picking e136da96: Initial plan"
git cherry-pick -x e136da96

echo "Cherry-picking 8c4d7a00: Refactor: Extract retry decorator to reduce code duplication in Neo4J storage"
git cherry-pick -x 8c4d7a00

echo "Cherry-picking aeaa0b32: Add mhchem extension support for chemistry formulas in ChatMessage"
git cherry-pick -x aeaa0b32

echo "Cherry-picking 48b6a6df: Merge pull request #2446 from danielaskdd/fix-postgres"
git cherry-pick -x 48b6a6df

echo "Cherry-picking d6019c82: Add CASCADE to AGE extension creation in PostgreSQL implementation"
git cherry-pick -x d6019c82

echo "Cherry-picking 49197fbf: Update pymilvus to >=2.6.2 and add protobuf compatibility constraint"
git cherry-pick -x 49197fbf

echo "Cherry-picking baab9924: Update pymilvus dependency from 2.5.2 to >=2.6.2"
git cherry-pick -x baab9924

echo "Cherry-picking e5e16b7b: Fix Redis data migration error"
git cherry-pick -x e5e16b7b

echo "Cherry-picking 0fa9a2ee: Fix dimension type comparison in Milvus vector field validation"
git cherry-pick -x 0fa9a2ee

echo "Cherry-picking 042cbad0: Merge branch 'qdrant-multi-tenancy'"
git cherry-pick -x 042cbad0

echo "Cherry-picking 0498e80a: Merge branch 'main' into qdrant-multi-tenancy"
git cherry-pick -x 0498e80a

echo "Cherry-picking e8f5f57e: Update qdrant-client minimum version from 1.7.0 to 1.11.0"
git cherry-pick -x e8f5f57e

echo "Cherry-picking 8bb54833: Merge pull request #2368 from danielaskdd/milvus-vector-batching"
git cherry-pick -x 8bb54833

echo "Cherry-picking 7aaa51cd: Add retry decorators to Neo4j read operations for resilience"
git cherry-pick -x 7aaa51cd

echo "Cherry-picking 2832a2ca: Merge pull request #2417 from danielaskdd/neo4j-retry"
git cherry-pick -x 2832a2ca

echo "Cherry-picking 0e0b4a94: Improve Docker build workflow with automated multi-arch script and docs"
git cherry-pick -x 0e0b4a94

echo "Cherry-picking e6332ce5: Add reminder note to manual Docker build workflow"
git cherry-pick -x e6332ce5

echo "Cherry-picking 656025b7: Rename GitHub workflow from \"Tests\" to \"Offline Unit Tests\""
git cherry-pick -x 656025b7

echo "Cherry-picking a11912ff: Add testing workflow guidelines to basic development rules"
git cherry-pick -x a11912ff

echo "Cherry-picking 445adfc9: Add name to lint-and-format job in GitHub workflow"
git cherry-pick -x 445adfc9

