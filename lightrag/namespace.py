from __future__ import annotations

import os
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

    # The server's own configuration, and every workspace's. Held in ONE fixed
    # container of its own rather than following the knowledge base it
    # configures. This namespace is ALSO the marker every backend keys its
    # fixed-container rule on: nothing else is ever opened on it.
    # See docs/design/ConfigurationStorage.md.
    KV_STORE_CONFIG = "config"


# The fixed tag every backend names the configuration container after.
#
# **It is not a workspace.** Nothing validates it as one, no ``*_WORKSPACE``
# variable may remap it, and no caller may choose it: the configuration
# storage is its own category, and its container is named in code. The four
# backends compose it into their own container name -- PostgreSQL writes it
# into ``LIGHTRAG_CONFIG``'s partition column, MongoDB and OpenSearch prefix
# their collection / index with it, and the JSON backend ignores it entirely
# in favour of ``config_dir``.
#
# The spelling is slice 1's reserved workspace name on purpose: keeping it
# means an existing deployment's rows are read where they already are rather
# than reading as ABSENT, which is the one answer that lets a start bootstrap.
# See docs/design/ConfigurationStorage.md.
CONFIG_CONTAINER_TAG = "_lightrag_config"


def default_config_dir(working_dir: str) -> str:
    """Where a file-backed configuration storage keeps its file by default.

    ``<working_dir>/_lightrag_config`` -- **exactly** where slice 1's reserved
    workspace put it. The default is chosen for that and nothing else: point
    it anywhere new and every baseline of an existing deployment reads as
    ABSENT on the first start after an upgrade, which is the one answer that
    lets a start bootstrap. Nothing in any log would say so.
    """
    return os.path.join(working_dir, CONFIG_CONTAINER_TAG)


# The scope a server-global configuration key is filed under, so its rows sort
# beside the per-workspace ones without being mistaken for a tenant's.
SERVER_CONFIG_SCOPE = "_lightrag_server"


def is_namespace(namespace: str, base_namespace: str | Iterable[str]):
    if isinstance(base_namespace, str):
        return namespace.endswith(base_namespace)
    return any(is_namespace(namespace, ns) for ns in base_namespace)
