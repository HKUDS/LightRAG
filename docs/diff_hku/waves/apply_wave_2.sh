#!/usr/bin/env bash
# Auto-generated script to apply Wave 2 commits
set -e

echo "Cherry-picking ec40b17e: Rangana feat: Add token tracking support to openai_embed function"
git cherry-pick -x ec40b17e

echo "Cherry-picking 0f15fdc3: Merge pull request #2181 from yrangana/feat/openai-embedding-token-tracking"
git cherry-pick -x 0f15fdc3

echo "Cherry-picking 6d1ae404: Add offline Docker build support with embedded models and cache"
git cherry-pick -x 6d1ae404

echo "Cherry-picking 6a29b5da: Update Docker deployment comments for LLM and embedding hosts"
git cherry-pick -x 6a29b5da

echo "Cherry-picking 7b8223da: Update env.example with host/endpoint clarifications for LLM/embedding"
git cherry-pick -x 7b8223da

echo "Cherry-picking 9c057060: Add separate endpoint configuration for LLM and embeddings in evaluation"
git cherry-pick -x 9c057060

echo "Cherry-picking 01b07b2b: Refactor Jina embedding dimension by changing param to optional with default"
git cherry-pick -x 01b07b2b

echo "Cherry-picking 33a1482f: Add optional embedding dimension parameter control via env var"
git cherry-pick -x 33a1482f

echo "Cherry-picking 9cee5a63: Merge branch 'main' into apply-dim-to-embedding-call"
git cherry-pick -x 9cee5a63

echo "Cherry-picking ce28f30c: Add embedding_dim parameter support to embedding functions"
git cherry-pick -x ce28f30c

echo "Cherry-picking d8a6355e: Merge branch 'main' into apply-dim-to-embedding-call"
git cherry-pick -x d8a6355e

echo "Cherry-picking d94aae9c: Rangana Add dimensions parameter support to openai_embed()"
git cherry-pick -x d94aae9c

echo "Cherry-picking ffeeae42: refactor: simplify jina embedding dimension handling"
git cherry-pick -x ffeeae42

echo "Cherry-picking 03cc6262: Prohibit direct access to internal functions of EmbeddingFunc."
git cherry-pick -x 03cc6262

echo "Cherry-picking 0b2a15c4: Centralize embedding_send_dim config through args instead of env var"
git cherry-pick -x 0b2a15c4

echo "Cherry-picking 29a349f2: Merge pull request #2329 from danielaskdd/gemini-embedding"
git cherry-pick -x 29a349f2

echo "Cherry-picking a624a950: Add Gemini to APIs requiring embedding dimension parameter"
git cherry-pick -x a624a950

echo "Cherry-picking de4ed736: Add Gemini embedding support"
git cherry-pick -x de4ed736

echo "Cherry-picking f4492d48: Merge pull request #2328 from HKUDS/apply-dim-to-embedding-call"
git cherry-pick -x f4492d48

echo "Cherry-picking 05852e1a: Add max_token_size parameter to embedding function decorators"
git cherry-pick -x 05852e1a

echo "Cherry-picking 14a6c24e: Add configurable embedding token limit with validation"
git cherry-pick -x 14a6c24e

echo "Cherry-picking 2fb57e76: Fix embedding token limit initialization order"
git cherry-pick -x 2fb57e76

echo "Cherry-picking 39b49e92: Convert embedding_token_limit from property to field with __post_init__"
git cherry-pick -x 39b49e92

echo "Cherry-picking 5dec4dea: Improve embedding config priority and add debug logging"
git cherry-pick -x 5dec4dea

echo "Cherry-picking 6b2af2b5: Refactor embedding function creation with proper attribute inheritance"
git cherry-pick -x 6b2af2b5

echo "Cherry-picking 77221564: Add max_token_size parameter to embedding function decorators"
git cherry-pick -x 77221564

echo "Cherry-picking 963a0a5d: Refactor embedding function creation with proper attribute inheritance"
git cherry-pick -x 963a0a5d

echo "Cherry-picking ab4d7ac2: Add configurable embedding token limit with validation"
git cherry-pick -x ab4d7ac2

echo "Cherry-picking de4412dd: Fix embedding token limit initialization order"
git cherry-pick -x de4412dd

echo "Cherry-picking e5addf4d: Improve embedding config priority and add debug logging"
git cherry-pick -x e5addf4d

echo "Cherry-picking f0254773: Convert embedding_token_limit from property to field with __post_init__"
git cherry-pick -x f0254773

echo "Cherry-picking 3b76eea2: Merge pull request #2359 from danielaskdd/embedding-limit"
git cherry-pick -x 3b76eea2

echo "Cherry-picking b5589ce4: Merge branch 'main' into embedding-limit"
git cherry-pick -x b5589ce4

echo "Cherry-picking c13f9116: Add embedding dimension validation to EmbeddingFunc wrapper"
git cherry-pick -x c13f9116

echo "Cherry-picking 46ce6d9a: Fix Azure OpenAI embedding model parameter fallback"
git cherry-pick -x 46ce6d9a

echo "Cherry-picking 0c4cba38: Fix double decoration in azure_openai_embed and document decorator usage"
git cherry-pick -x 0c4cba38

echo "Cherry-picking 7b762110: Add fallback to AZURE_OPENAI_API_VERSION for embedding API version"
git cherry-pick -x 7b762110

echo "Cherry-picking 1b02684e: Merge pull request #2432 from danielaskdd/embedding-example"
git cherry-pick -x 1b02684e

echo "Cherry-picking 1d07ff7f: Update OpenAI and Ollama embedding func examples in README"
git cherry-pick -x 1d07ff7f

echo "Cherry-picking 4ab4a7ac: Allow embedding models to use provider defaults when unspecified"
git cherry-pick -x 4ab4a7ac

echo "Cherry-picking 56e0365c: Add configurable model parameter to jina_embed function"
git cherry-pick -x 56e0365c

echo "Cherry-picking 6e2946e7: Add max_token_size parameter to azure_openai_embed wrapper"
git cherry-pick -x 6e2946e7

echo "Cherry-picking 97a9dfca: Add important note about embedding function wrapping restrictions"
git cherry-pick -x 97a9dfca

echo "Cherry-picking b6705449: Merge pull request #2433 from danielaskdd/fix-jina-embedding"
git cherry-pick -x b6705449

echo "Cherry-picking ea8d55ab: Add documentation for embedding provider configuration rules"
git cherry-pick -x ea8d55ab

echo "Cherry-picking 37e8898c: Simplify reference formatting in LLM context generation"
git cherry-pick -x 37e8898c

echo "Cherry-picking 83d99e14: fix(OllamaAPI): Add validation to ensure last message is from user role"
git cherry-pick -x 83d99e14

echo "Cherry-picking 0b3d3150: extended to use gemini, sswitched to use gemini-flash-latest"
git cherry-pick -x 0b3d3150

echo "Cherry-picking 74694214: Update openai requirement from <2.0.0,>=1.0.0 to >=1.0.0,<3.0.0"
git cherry-pick -x 74694214

echo "Cherry-picking 175ef459: Merge pull request #2238 from HKUDS/dependabot/pip/openai-gte-1.0.0-and-lt-3.0.0"
git cherry-pick -x 175ef459

echo "Cherry-picking 162370b6: Add optional LLM cache deletion when deleting documents"
git cherry-pick -x 162370b6

echo "Cherry-picking aa916f28: docs: add generic test_dataset.json for evaluation examples Test cases with generic examples about: - LightRAG framework features and capabilities - RAG system architecture and components - Vector database support (ChromaDB, Neo4j, Milvus, etc.) - LLM provider integrations (OpenAI, Anthropic, Ollama, etc.) - RAG evaluation metrics explanation - Deployment options (Docker, FastAPI, direct integration) - Knowledge graph-based retrieval concepts"
git cherry-pick -x aa916f28

echo "Cherry-picking 994a82dc: Suppress token usage warnings for custom OpenAI-compatible endpoints"
git cherry-pick -x 994a82dc

echo "Cherry-picking 3cb4eae4: Add Chain of Thought support to Gemini LLM integration"
git cherry-pick -x 3cb4eae4

echo "Cherry-picking 6686edfd: Update Gemini LLM options: add seed and thinking config, remove MIME type"
git cherry-pick -x 6686edfd

echo "Cherry-picking 73284623: Merge pull request #2326 from danielaskdd/gemini-cot"
git cherry-pick -x 73284623

echo "Cherry-picking 8c275553: Fix Gemini response parsing to avoid warnings from non-text parts"
git cherry-pick -x 8c275553

echo "Cherry-picking 924c8cb8: Merge branch 'main' into gemini-cot"
git cherry-pick -x 924c8cb8

echo "Cherry-picking fc40a369: Add timeout support to Gemini LLM and improve parameter handling"
git cherry-pick -x fc40a369

echo "Cherry-picking 3d9de5ed: feat: improve Gemini client error handling and retry logic"
git cherry-pick -x 3d9de5ed

echo "Cherry-picking 55274dde: Add LLM cache migration tool for KV storage backends"
git cherry-pick -x 55274dde

echo "Cherry-picking 57ee7d5a: Merge branch 'main' into llm-cache-migrate"
git cherry-pick -x 57ee7d5a

echo "Cherry-picking 6b9f13c7: Enhance LLM cache migration tool with streaming and improved UX"
git cherry-pick -x 6b9f13c7

echo "Cherry-picking 6fc54d36: Move LLM cache migration tool to lightrag.tools module"
git cherry-pick -x 6fc54d36

echo "Cherry-picking 85bb98b3: Merge pull request #2331 from danielaskdd/gemini-retry"
git cherry-pick -x 85bb98b3

echo "Cherry-picking 987bc09c: Update LLM cache migration docs and improve UX prompts"
git cherry-pick -x 987bc09c

echo "Cherry-picking d0d31e92: Improve LLM cache migration tool configuration and messaging"
git cherry-pick -x d0d31e92

echo "Cherry-picking f83ea339: Add section header comment for Gemini binding options"
git cherry-pick -x f83ea339

echo "Cherry-picking 1485cb82: Add LLM query cache cleanup tool for KV storage backends"
git cherry-pick -x 1485cb82

echo "Cherry-picking 3110ca51: Merge pull request #2335 from danielaskdd/llm-cache-cleanup"
git cherry-pick -x 3110ca51

echo "Cherry-picking 754d2ad2: Add documentation for LLM cache migration between storage types"
git cherry-pick -x 754d2ad2

echo "Cherry-picking 88ab73f6: HotFix: Restore streaming response in OpenAI LLM"
git cherry-pick -x 88ab73f6

echo "Cherry-picking 8adf3180: Merge pull request #2330 from danielaskdd/llm-cache-migrate"
git cherry-pick -x 8adf3180

echo "Cherry-picking 18893015: Merge branch 'feat/add_cloud_ollama_support'"
git cherry-pick -x 18893015

echo "Cherry-picking 680e36c6: Improve Bedrock error handling with retry logic and custom exceptions"
git cherry-pick -x 680e36c6

echo "Cherry-picking f5b48587: Improve Bedrock error handling with retry logic and custom exceptions"
git cherry-pick -x f5b48587

echo "Cherry-picking 95e1fb16: Remove final_namespace attribute for in-memory storage and use namespace in clean_llm_query_cache.py"
git cherry-pick -x 95e1fb16

echo "Cherry-picking a990c1d4: fix: Correct Mock LLM output format in E2E test"
git cherry-pick -x a990c1d4

echo "Cherry-picking 021b637d: Merge pull request #2403 from danielaskdd/azure-cot-handling"
git cherry-pick -x 021b637d

echo "Cherry-picking 02fdceb9: Update OpenAI client to use stable API and bump minimum version to 2.0.0"
git cherry-pick -x 02fdceb9

echo "Cherry-picking 1e477e95: Add lightrag-clean-llmqc console script entry point"
git cherry-pick -x 1e477e95

echo "Cherry-picking 45f4f823: Refactor Azure OpenAI client creation to support client_configs merging"
git cherry-pick -x 45f4f823

echo "Cherry-picking 8777895e: Merge pull request #2401 from danielaskdd/fix-openai-keyword-extraction"
git cherry-pick -x 8777895e

echo "Cherry-picking 9f69c5bf: feat: Support structured output `parsed` from OpenAI"
git cherry-pick -x 9f69c5bf

echo "Cherry-picking ac9f2574: Improve Azure OpenAI wrapper functions with full parameter support"
git cherry-pick -x ac9f2574

echo "Cherry-picking b709f8f8: Consolidate Azure OpenAI implementation into main OpenAI module"
git cherry-pick -x b709f8f8

echo "Cherry-picking fafa1791: Fix Azure OpenAI model parameter to use deployment name consistently"
git cherry-pick -x fafa1791

echo "Cherry-picking ffd8da51: Improve Azure OpenAI compatibility and error handling"
git cherry-pick -x ffd8da51

echo "Cherry-picking 49fb11e2: Update Azure OpenAI configuration examples"
git cherry-pick -x 49fb11e2

echo "Cherry-picking 5f53de88: Fix Azure configuration examples and correct typos in env.example"
git cherry-pick -x 5f53de88

echo "Cherry-picking a898f054: Merge branch 'HKUDS:main' into cohere-rerank"
git cherry-pick -x a898f054

echo "Cherry-picking 8e50eef5: Merge branch 'main' into cohere-rerank"
git cherry-pick -x 8e50eef5

echo "Cherry-picking f0d67f16: Merge branch 'cohere-rerank'"
git cherry-pick -x f0d67f16

