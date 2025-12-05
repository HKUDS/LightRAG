import asyncio
import os
from dataclasses import dataclass
from typing import Any, final

import numpy as np

from lightrag.base import BaseVectorStorage
from lightrag.utils import logger

from chromadb import HttpClient, PersistentClient  # type: ignore
from chromadb.config import Settings  # type: ignore


@final
@dataclass
class ChromaVectorDBStorage(BaseVectorStorage):
    """ChromaDB vector storage implementation."""

    def __post_init__(self):
        try:
            config = self.global_config.get('vector_db_storage_cls_kwargs', {})
            cosine_threshold = config.get('cosine_better_than_threshold')
            if cosine_threshold is None:
                raise ValueError('cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs')
            self.cosine_better_than_threshold = cosine_threshold

            user_collection_settings = config.get('collection_settings', {})
            # Default HNSW index settings for ChromaDB
            default_collection_settings = {
                # Distance metric used for similarity search (cosine similarity)
                'hnsw:space': 'cosine',
                # Number of nearest neighbors to explore during index construction
                # Higher values = better recall but slower indexing
                'hnsw:construction_ef': 128,
                # Number of nearest neighbors to explore during search
                # Higher values = better recall but slower search
                'hnsw:search_ef': 128,
                # Number of connections per node in the HNSW graph
                # Higher values = better recall but more memory usage
                'hnsw:M': 16,
                # Number of vectors to process in one batch during indexing
                'hnsw:batch_size': 100,
                # Number of updates before forcing index synchronization
                # Lower values = more frequent syncs but slower indexing
                'hnsw:sync_threshold': 1000,
            }
            collection_settings = {
                **default_collection_settings,
                **user_collection_settings,
            }

            local_path = config.get('local_path', None)
            if local_path:
                self._client = PersistentClient(
                    path=local_path,
                    settings=Settings(
                        allow_reset=True,
                        anonymized_telemetry=False,
                    ),
                )
            else:
                auth_provider = config.get('auth_provider', 'chromadb.auth.token_authn.TokenAuthClientProvider')
                auth_credentials = config.get('auth_token', 'secret-token')
                headers = {}

                if 'token_authn' in auth_provider:
                    headers = {config.get('auth_header_name', 'X-Chroma-Token'): auth_credentials}
                elif 'basic_authn' in auth_provider:
                    auth_credentials = config.get('auth_credentials', 'admin:admin')

                self._client = HttpClient(
                    host=config.get('host', 'localhost'),
                    port=config.get('port', 8000),
                    headers=headers,
                    settings=Settings(
                        chroma_api_impl='rest',
                        chroma_client_auth_provider=auth_provider,
                        chroma_client_auth_credentials=auth_credentials,
                        allow_reset=True,
                        anonymized_telemetry=False,
                    ),
                )

            self._collection = self._client.get_or_create_collection(
                name=self.namespace,
                metadata={
                    **collection_settings,
                    'dimension': self.embedding_func.embedding_dim,
                },
            )
            # Use batch size from collection settings if specified
            self._max_batch_size = self.global_config.get(
                'embedding_batch_num', collection_settings.get('hnsw:batch_size', 32)
            )
        except Exception as e:
            logger.error(f'ChromaDB initialization failed: {e!s}')
            raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(f'Inserting {len(data)} to {self.namespace}')
        if not data:
            return

        try:
            import time

            current_time = int(time.time())

            ids = list(data.keys())
            documents = [v['content'] for v in data.values()]
            metadatas = [
                {**{k: v for k, v in item.items() if k in self.meta_fields}, 'created_at': current_time}
                for item in data.values()
            ]

            # Process in batches
            batches = [documents[i : i + self._max_batch_size] for i in range(0, len(documents), self._max_batch_size)]

            embedding_tasks = [self.embedding_func(batch) for batch in batches]
            embeddings_list = []

            # Pre-allocate embeddings_list with known size
            embeddings_list = [None] * len(embedding_tasks)

            # Use asyncio.gather instead of as_completed if order doesn't matter
            embeddings_results = await asyncio.gather(*embedding_tasks)
            embeddings_list = list(embeddings_results)

            embeddings = np.concatenate(embeddings_list)

            # Upsert in batches
            for i in range(0, len(ids), self._max_batch_size):
                batch_slice = slice(i, i + self._max_batch_size)

                self._collection.upsert(
                    ids=ids[batch_slice],
                    embeddings=embeddings[batch_slice].tolist(),
                    documents=documents[batch_slice],
                    metadatas=metadatas[batch_slice],
                )

            return ids

        except Exception as e:
            logger.error(f'Error during ChromaDB upsert: {e!s}')
            raise

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        try:
            embedding = await self.embedding_func([query], _priority=5)  # higher priority for query

            results = self._collection.query(
                query_embeddings=embedding.tolist() if not isinstance(embedding, list) else embedding,
                n_results=top_k * 2,  # Request more results to allow for filtering
                include=['metadatas', 'distances', 'documents'],
            )

            # Filter results by cosine similarity threshold and take top k
            # We request 2x results initially to have enough after filtering
            # ChromaDB returns cosine similarity (1 = identical, 0 = orthogonal)
            # We convert to distance (0 = identical, 1 = orthogonal) via (1 - similarity)
            # Only keep results with distance below threshold, then take top k
            return [
                {
                    'id': results['ids'][0][i],
                    'distance': 1 - results['distances'][0][i],
                    'content': results['documents'][0][i],
                    'created_at': results['metadatas'][0][i].get('created_at'),
                    **results['metadatas'][0][i],
                }
                for i in range(len(results['ids'][0]))
                if (1 - results['distances'][0][i]) >= self.cosine_better_than_threshold
            ][:top_k]

        except Exception as e:
            logger.error(f'Error during ChromaDB query: {e!s}')
            raise

    async def index_done_callback(self) -> None:
        # ChromaDB handles persistence automatically
        pass

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by its ID.

        Args:
            entity_name: The ID of the entity to delete
        """
        try:
            logger.info(f'Deleting entity with ID {entity_name} from {self.namespace}')
            self._collection.delete(ids=[entity_name])
        except Exception as e:
            logger.error(f'Error during entity deletion: {e!s}')
            raise

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete an entity and its relations by ID.
        In vector DB context, this is equivalent to delete_entity.

        Args:
            entity_name: The ID of the entity to delete
        """
        await self.delete_entity(entity_name)

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            self._collection.delete(ids=ids)
            logger.debug(f'Successfully deleted {len(ids)} vectors from {self.namespace}')
        except Exception as e:
            logger.error(f'Error while deleting vectors from {self.namespace}: {e!s}')
            raise

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        try:
            # Query the collection for a single vector by ID
            result = self._collection.get(ids=[id], include=['metadatas', 'embeddings', 'documents'])

            if not result or not result['ids'] or len(result['ids']) == 0:
                return None

            # Format the result to match the expected structure
            return {
                'id': result['ids'][0],
                'vector': result['embeddings'][0],
                'content': result['documents'][0],
                'created_at': result['metadatas'][0].get('created_at'),
                **result['metadatas'][0],
            }
        except Exception as e:
            logger.error(f'Error retrieving vector data for ID {id}: {e!s}')
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        try:
            # Query the collection for multiple vectors by IDs
            result = self._collection.get(ids=ids, include=['metadatas', 'embeddings', 'documents'])

            if not result or not result['ids'] or len(result['ids']) == 0:
                return []

            # Format the results to match the expected structure and preserve ordering
            formatted_map: dict[str, dict[str, Any]] = {}
            for i, result_id in enumerate(result['ids']):
                record = {
                    'id': result_id,
                    'vector': result['embeddings'][i],
                    'content': result['documents'][i],
                    'created_at': result['metadatas'][i].get('created_at'),
                    **result['metadatas'][i],
                }
                formatted_map[str(result_id)] = record

            ordered_results: list[dict[str, Any] | None] = []
            for requested_id in ids:
                ordered_results.append(formatted_map.get(str(requested_id)))

            return ordered_results
        except Exception as e:
            logger.error(f'Error retrieving vector data for IDs {ids}: {e}')
            return []

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources

        This method will delete all documents from the ChromaDB collection.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        try:
            # Get all IDs in the collection
            result = self._collection.get(include=[])
            if result and result['ids'] and len(result['ids']) > 0:
                # Delete all documents
                self._collection.delete(ids=result['ids'])

            logger.info(f'Process {os.getpid()} drop ChromaDB collection {self.namespace}')
            return {'status': 'success', 'message': 'data dropped'}
        except Exception as e:
            logger.error(f'Error dropping ChromaDB collection {self.namespace}: {e}')
            return {'status': 'error', 'message': str(e)}
