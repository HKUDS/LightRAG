from dataclasses import dataclass
from lightrag import LightRAG as BaseLightRAG
from lightrag.lightrag import lazy_external_import

storage_lookup = {
    "MongoKVStorage": {
        "import_path": "new.new_kg.new_mongo_impl",
        "storage_name": "NewMongoKVStorage",
    },
    "Neo4JStorage": {
        "import_path": "new.new_kg.new_neo4j_impl",
        "storage_name": "NewNeo4JStorage",
    },
    "NanoVectorDBStorage": {
        "import_path": "new.new_kg.new_nano_vector_db_impl",
        "storage_name": "NewNanoVectorDBStorage",
    },
    "NetworkXStorage": {
        "import_path": "new.new_kg.new_networkx_impl",
        "storage_name": "NewNetworkXStorage",
    },
    "JsonDocStatusStorage": {
        "import_path": "new.new_kg.new_json_doc_status_impl",
        "storage_name": "NewJsonDocStatusStorage",
    },
}


@dataclass
class NewLightRAG(BaseLightRAG):
    def __post_init__(self):
        super().__post_init__()

    def _get_storage_class(self, storage_name: str) -> dict:
        if storage_name not in storage_lookup:
            storage_class = super()._get_storage_class(storage_name)
        else:
            storage_class = lazy_external_import(
                storage_lookup[storage_name]["import_path"],
                storage_lookup[storage_name]["storage_name"],
            )
        return storage_class
