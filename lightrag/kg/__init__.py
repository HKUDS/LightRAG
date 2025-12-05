STORAGE_IMPLEMENTATIONS = {
    'KV_STORAGE': {
        'implementations': [
            'JsonKVStorage',
            'RedisKVStorage',
            'PGKVStorage',
            'MongoKVStorage',
        ],
        'required_methods': ['get_by_id', 'upsert'],
    },
    'GRAPH_STORAGE': {
        'implementations': [
            'NetworkXStorage',
            'Neo4JStorage',
            'PGGraphStorage',
            'MongoGraphStorage',
            'MemgraphStorage',
        ],
        'required_methods': ['upsert_node', 'upsert_edge'],
    },
    'VECTOR_STORAGE': {
        'implementations': [
            'NanoVectorDBStorage',
            'MilvusVectorDBStorage',
            'PGVectorStorage',
            'FaissVectorDBStorage',
            'QdrantVectorDBStorage',
            'MongoVectorDBStorage',
            # "ChromaVectorDBStorage",
        ],
        'required_methods': ['query', 'upsert'],
    },
    'DOC_STATUS_STORAGE': {
        'implementations': [
            'JsonDocStatusStorage',
            'RedisDocStatusStorage',
            'PGDocStatusStorage',
            'MongoDocStatusStorage',
        ],
        'required_methods': ['get_docs_by_status'],
    },
}

# Storage implementation environment variable without default value
STORAGE_ENV_REQUIREMENTS: dict[str, list[str]] = {
    # KV Storage Implementations
    'JsonKVStorage': [],
    'MongoKVStorage': [
        'MONGO_URI',
        'MONGO_DATABASE',
    ],
    'RedisKVStorage': ['REDIS_URI'],
    'PGKVStorage': ['POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DATABASE'],
    # Graph Storage Implementations
    'NetworkXStorage': [],
    'Neo4JStorage': ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD'],
    'MongoGraphStorage': [
        'MONGO_URI',
        'MONGO_DATABASE',
    ],
    'MemgraphStorage': ['MEMGRAPH_URI'],
    'AGEStorage': [
        'AGE_POSTGRES_DB',
        'AGE_POSTGRES_USER',
        'AGE_POSTGRES_PASSWORD',
    ],
    'PGGraphStorage': [
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'POSTGRES_DATABASE',
    ],
    # Vector Storage Implementations
    'NanoVectorDBStorage': [],
    'MilvusVectorDBStorage': [
        'MILVUS_URI',
        'MILVUS_DB_NAME',
    ],
    # "ChromaVectorDBStorage": [],
    'PGVectorStorage': ['POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DATABASE'],
    'FaissVectorDBStorage': [],
    'QdrantVectorDBStorage': ['QDRANT_URL'],  # QDRANT_API_KEY has default value None
    'MongoVectorDBStorage': [
        'MONGO_URI',
        'MONGO_DATABASE',
    ],
    # Document Status Storage Implementations
    'JsonDocStatusStorage': [],
    'RedisDocStatusStorage': ['REDIS_URI'],
    'PGDocStatusStorage': ['POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DATABASE'],
    'MongoDocStatusStorage': [
        'MONGO_URI',
        'MONGO_DATABASE',
    ],
}

# Storage implementation module mapping
STORAGES = {
    'NetworkXStorage': '.kg.networkx_impl',
    'JsonKVStorage': '.kg.json_kv_impl',
    'NanoVectorDBStorage': '.kg.nano_vector_db_impl',
    'JsonDocStatusStorage': '.kg.json_doc_status_impl',
    'Neo4JStorage': '.kg.neo4j_impl',
    'MilvusVectorDBStorage': '.kg.milvus_impl',
    'MongoKVStorage': '.kg.mongo_impl',
    'MongoDocStatusStorage': '.kg.mongo_impl',
    'MongoGraphStorage': '.kg.mongo_impl',
    'MongoVectorDBStorage': '.kg.mongo_impl',
    'RedisKVStorage': '.kg.redis_impl',
    'RedisDocStatusStorage': '.kg.redis_impl',
    'ChromaVectorDBStorage': '.kg.chroma_impl',
    'PGKVStorage': '.kg.postgres_impl',
    'PGVectorStorage': '.kg.postgres_impl',
    'AGEStorage': '.kg.age_impl',
    'PGGraphStorage': '.kg.postgres_impl',
    'PGDocStatusStorage': '.kg.postgres_impl',
    'FaissVectorDBStorage': '.kg.faiss_impl',
    'QdrantVectorDBStorage': '.kg.qdrant_impl',
    'MemgraphStorage': '.kg.memgraph_impl',
}


def verify_storage_implementation(storage_type: str, storage_name: str) -> None:
    """Verify if storage implementation is compatible with specified storage type

    Args:
        storage_type: Storage type (KV_STORAGE, GRAPH_STORAGE etc.)
        storage_name: Storage implementation name

    Raises:
        ValueError: If storage implementation is incompatible or missing required methods
    """
    if storage_type not in STORAGE_IMPLEMENTATIONS:
        raise ValueError(f'Unknown storage type: {storage_type}')

    storage_info = STORAGE_IMPLEMENTATIONS[storage_type]
    if storage_name not in storage_info['implementations']:
        raise ValueError(
            f"Storage implementation '{storage_name}' is not compatible with {storage_type}. "
            f'Compatible implementations are: {", ".join(storage_info["implementations"])}'
        )
