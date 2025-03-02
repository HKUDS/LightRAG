from dataclasses import dataclass
from typing import Dict
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage as DocStatusStorage


from lightrag.utils import (
    logger
)

@dataclass
class NewJsonDocStatusStorage(DocStatusStorage):
    def __post_init__(self):
        logger.info("Initializing New JsonDocStatusStorage")
        super().__post_init__()

    async def get_all_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all documents"""
        return {k: v for k, v in self._data.items()}

    async def drop(self):
        self._data = {}