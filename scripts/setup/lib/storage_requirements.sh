# Storage backend options and required environment variables.
# shellcheck disable=SC2034

declare -ag KV_STORAGE_OPTIONS=(
  "JsonKVStorage"
  "RedisKVStorage"
  "PGKVStorage"
  "MongoKVStorage"
)

declare -ag GRAPH_STORAGE_OPTIONS=(
  "NetworkXStorage"
  "Neo4JStorage"
  "PGGraphStorage"
  "MongoGraphStorage"
  "MemgraphStorage"
)

declare -ag VECTOR_STORAGE_OPTIONS=(
  "NanoVectorDBStorage"
  "MilvusVectorDBStorage"
  "PGVectorStorage"
  "FaissVectorDBStorage"
  "QdrantVectorDBStorage"
  "MongoVectorDBStorage"
)

declare -ag DOC_STATUS_STORAGE_OPTIONS=(
  "JsonDocStatusStorage"
  "RedisDocStatusStorage"
  "PGDocStatusStorage"
  "MongoDocStatusStorage"
)

declare -Ag STORAGE_ENV_REQUIREMENTS=(
  ["JsonKVStorage"]=""
  ["MongoKVStorage"]="MONGO_URI MONGO_DATABASE"
  ["RedisKVStorage"]="REDIS_URI"
  ["PGKVStorage"]="POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
  ["NetworkXStorage"]=""
  ["Neo4JStorage"]="NEO4J_URI NEO4J_USERNAME NEO4J_PASSWORD"
  ["MongoGraphStorage"]="MONGO_URI MONGO_DATABASE"
  ["MemgraphStorage"]="MEMGRAPH_URI"
  ["PGGraphStorage"]="POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
  ["NanoVectorDBStorage"]=""
  ["MilvusVectorDBStorage"]="MILVUS_URI MILVUS_DB_NAME"
  ["PGVectorStorage"]="POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
  ["FaissVectorDBStorage"]=""
  ["QdrantVectorDBStorage"]="QDRANT_URL"
  ["MongoVectorDBStorage"]="MONGO_URI MONGO_DATABASE"
  ["JsonDocStatusStorage"]=""
  ["RedisDocStatusStorage"]="REDIS_URI"
  ["PGDocStatusStorage"]="POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DATABASE"
  ["MongoDocStatusStorage"]="MONGO_URI MONGO_DATABASE"
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
  ["MilvusVectorDBStorage"]="milvus"
  ["QdrantVectorDBStorage"]="qdrant"
)
