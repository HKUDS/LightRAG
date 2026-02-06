# shellcheck disable=SC2034
PRESET_DEVELOPMENT=(
  "LIGHTRAG_KV_STORAGE=JsonKVStorage"
  "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage"
  "LIGHTRAG_GRAPH_STORAGE=NetworkXStorage"
  "LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage"
  "LLM_BINDING=openai"
  "LLM_MODEL=gpt-4o"
  "LLM_BINDING_HOST=https://api.openai.com/v1"
  "EMBEDDING_BINDING=openai"
  "EMBEDDING_MODEL=text-embedding-3-large"
  "EMBEDDING_DIM=3072"
  "EMBEDDING_BINDING_HOST=https://api.openai.com/v1"
)
