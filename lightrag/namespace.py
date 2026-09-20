from __future__ import annotations

from typing import Iterable


# All namespace should not be changed
class NameSpace:
    KV_STORE_FULL_DOCS = "full_docs"
    KV_STORE_TEXT_CHUNKS = "text_chunks"
    KV_STORE_LLM_RESPONSE_CACHE = "llm_response_cache"
    KV_STORE_FULL_ENTITIES = "full_entities"
    KV_STORE_FULL_RELATIONS = "full_relations"
    KV_STORE_ENTITY_CHUNKS = "entity_chunks"
    KV_STORE_RELATION_CHUNKS = "relation_chunks"

    VECTOR_STORE_ENTITIES = "entities"
    VECTOR_STORE_RELATIONSHIPS = "relationships"
    VECTOR_STORE_CHUNKS = "chunks"

    GRAPH_STORE_CHUNK_ENTITY_RELATION = "chunk_entity_relation"

    DOC_STATUS = "doc_status"

    # The server's own configuration, and every workspace's. Held in ONE fixed,
    # reserved workspace rather than following the knowledge base it
    # configures -- see docs/design/ConfigurationStorage.md.
    KV_STORE_CONFIG = "config"


# The workspace-name family LightRAG keeps for itself. ``validate_workspace``
# refuses every name starting with it unless the configuration-storage factory
# is the one binding it, so a tenant can never collide with an internal
# container. Reserved from the first release that has the container: a
# reservation made later cannot reclaim a name already in use.
RESERVED_WORKSPACE_PREFIX = "_lightrag"

# The one reserved workspace that exists today: the container every
# configuration row lives in, whatever workspace that row is ABOUT.
CONFIG_WORKSPACE = "_lightrag_config"

# The scope a server-global configuration key is filed under, so its rows sort
# beside the per-workspace ones without being mistaken for a tenant's.
SERVER_CONFIG_SCOPE = "_lightrag_server"


def is_namespace(namespace: str, base_namespace: str | Iterable[str]):
    if isinstance(base_namespace, str):
        return namespace.endswith(base_namespace)
    return any(is_namespace(namespace, ns) for ns in base_namespace)
