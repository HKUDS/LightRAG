# Storage backend options and required environment variables.
# shellcheck disable=SC2034

declare -ag KV_STORAGE_OPTIONS=(
  "JsonKVStorage"
  "RedisKVStorage"
  "PGKVStorage"
  "MongoKVStorage"
  "OpenSearchKVStorage"
)

declare -ag GRAPH_STORAGE_OPTIONS=(
  "NetworkXStorage"
  "Neo4JStorage"
  "PGGraphStorage"
  "MongoGraphStorage"
  "MemgraphStorage"
  "NebulaGraphStorage"
  "OpenSearchGraphStorage"
)

declare -ag VECTOR_STORAGE_OPTIONS=(
  "NanoVectorDBStorage"
  "MilvusVectorDBStorage"
  "PGVectorStorage"
  "FaissVectorDBStorage"
  "QdrantVectorDBStorage"
  "MongoVectorDBStorage"
  "OpenSearchVectorDBStorage"
)

declare -ag DOC_STATUS_STORAGE_OPTIONS=(
  "JsonDocStatusStorage"
  "RedisDocStatusStorage"
  "PGDocStatusStorage"
  "MongoDocStatusStorage"
  "OpenSearchDocStatusStorage"
)

declare -Ag STORAGE_ENV_REQUIREMENTS=(
  ["JsonKVStorage"]=""
  ["MongoKVStorage"]="MONGO_URI MONGO_DATABASE"
  ["RedisKVStorage"]="REDIS_URI"
  ["PGKVStorage"]="POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
  ["OpenSearchKVStorage"]="OPENSEARCH_HOSTS OPENSEARCH_USER OPENSEARCH_PASSWORD"
  ["NetworkXStorage"]=""
  ["Neo4JStorage"]="NEO4J_URI NEO4J_USERNAME NEO4J_PASSWORD"
  ["MongoGraphStorage"]="MONGO_URI MONGO_DATABASE"
  ["MemgraphStorage"]="MEMGRAPH_URI"
  ["NebulaGraphStorage"]="NEBULA_HOSTS NEBULA_USER"
  ["PGGraphStorage"]="POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
  ["OpenSearchGraphStorage"]="OPENSEARCH_HOSTS OPENSEARCH_USER OPENSEARCH_PASSWORD"
  ["NanoVectorDBStorage"]=""
  ["MilvusVectorDBStorage"]="MILVUS_URI MILVUS_DB_NAME"
  ["PGVectorStorage"]="POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
  ["FaissVectorDBStorage"]=""
  ["QdrantVectorDBStorage"]="QDRANT_URL"
  ["MongoVectorDBStorage"]="MONGO_URI MONGO_DATABASE"
  ["OpenSearchVectorDBStorage"]="OPENSEARCH_HOSTS OPENSEARCH_USER OPENSEARCH_PASSWORD"
  ["JsonDocStatusStorage"]=""
  ["RedisDocStatusStorage"]="REDIS_URI"
  ["PGDocStatusStorage"]="POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
  ["MongoDocStatusStorage"]="MONGO_URI MONGO_DATABASE"
  ["OpenSearchDocStatusStorage"]="OPENSEARCH_HOSTS OPENSEARCH_USER OPENSEARCH_PASSWORD"
)

declare -Ag STORAGE_DB_TYPES=(
  ["MongoKVStorage"]="mongodb"
  ["MongoGraphStorage"]="mongodb"
  ["MongoVectorDBStorage"]="mongodb"
  ["MongoDocStatusStorage"]="mongodb"
  ["RedisKVStorage"]="redis"
  ["RedisDocStatusStorage"]="redis"
  ["PGKVStorage"]="postgresql"
  ["PGGraphStorage"]="postgresql"
  ["PGVectorStorage"]="postgresql"
  ["PGDocStatusStorage"]="postgresql"
  ["Neo4JStorage"]="neo4j"
  ["MemgraphStorage"]="memgraph"
  ["NebulaGraphStorage"]="nebula"
  ["MilvusVectorDBStorage"]="milvus"
  ["QdrantVectorDBStorage"]="qdrant"
  ["OpenSearchKVStorage"]="opensearch"
  ["OpenSearchGraphStorage"]="opensearch"
  ["OpenSearchVectorDBStorage"]="opensearch"
  ["OpenSearchDocStatusStorage"]="opensearch"
)
