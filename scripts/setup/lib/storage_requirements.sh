# Storage backend options and required environment variables.
# shellcheck disable=SC2034

declare -ag KV_STORAGE_OPTIONS=(
  "JsonKVStorage"
  "PGKVStorage"
  "MongoKVStorage"
  "OpenSearchKVStorage"
  "RedisKVStorage"
)

declare -ag GRAPH_STORAGE_OPTIONS=(
  "NetworkXStorage"
  "PGGraphStorage"
  "PGTableGraphStorage"
  "MongoGraphStorage"
  "OpenSearchGraphStorage"
  "MemgraphStorage"
  "Neo4JStorage"
)

declare -ag VECTOR_STORAGE_OPTIONS=(
  "NanoVectorDBStorage"
  "PGVectorStorage"
  "MongoVectorDBStorage"
  "OpenSearchVectorDBStorage"
  "MilvusVectorDBStorage"
  "FaissVectorDBStorage"
  "QdrantVectorDBStorage"
)

declare -ag DOC_STATUS_STORAGE_OPTIONS=(
  "JsonDocStatusStorage"
  "PGDocStatusStorage"
  "MongoDocStatusStorage"
  "OpenSearchDocStatusStorage"
  "RedisDocStatusStorage"
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
  ["PGGraphStorage"]="POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
  ["PGTableGraphStorage"]="POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
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
  ["PGTableGraphStorage"]="postgresql"
  ["PGVectorStorage"]="postgresql"
  ["PGDocStatusStorage"]="postgresql"
  ["Neo4JStorage"]="neo4j"
  ["MemgraphStorage"]="memgraph"
  ["MilvusVectorDBStorage"]="milvus"
  ["QdrantVectorDBStorage"]="qdrant"
  ["OpenSearchKVStorage"]="opensearch"
  ["OpenSearchGraphStorage"]="opensearch"
  ["OpenSearchVectorDBStorage"]="opensearch"
  ["OpenSearchDocStatusStorage"]="opensearch"
)
