from dataclasses import dataclass
import numpy as np
from sqlalchemy import Float
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage as BaseVectorStorage

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)


@dataclass
class NewNanoVectorDBStorage(BaseVectorStorage):
    def __post_init__(self):
        logger.info("Initializing New NanoVectorDBStorage")
        super().__post_init__()

    async def delete_entity_relation_by_nodes(self, src_entity_name: str,tgt_entity_name: str):
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == src_entity_name and dp["tgt_id"] == tgt_entity_name
            ]
            logger.debug(f"Found {len(relations)} relations for entity {src_entity_name} and {tgt_entity_name}")
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                await self.delete(ids_to_delete)
                logger.debug(
                    f"Deleted {len(ids_to_delete)} relations for entity {src_entity_name} and {tgt_entity_name}"
                )
            else:
                logger.debug(f"No relations found for entity for entity {src_entity_name} and {tgt_entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for entity {src_entity_name} and {tgt_entity_name}: {e}")

    async def delete_all(self):
        try:
            self.client_storage["data"] = []
            self.client_storage["matrix"] = np.array([], dtype=Float).reshape(
                0, self._client.embedding_dim
            )
        except Exception as e:
            logger.error(f"Error while deleting all collections: {e}")