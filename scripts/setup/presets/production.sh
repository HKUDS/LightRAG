# shellcheck disable=SC2034
PRESET_PRODUCTION=(
  "LIGHTRAG_KV_STORAGE=PGKVStorage"
  "LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage"
  "LIGHTRAG_GRAPH_STORAGE=Neo4JStorage"
  "LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage"
  "LLM_BINDING=openai"
  "LLM_MODEL=gpt-4o"
  "LLM_BINDING_HOST=https://api.openai.com/v1"
  "EMBEDDING_BINDING=openai"
  "EMBEDDING_MODEL=text-embedding-3-large"
  "EMBEDDING_DIM=3072"
  "EMBEDDING_BINDING_HOST=https://api.openai.com/v1"
)
