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

    # The server's own configuration, and every workspace's. Held in ONE fixed
    # container of its own rather than following the knowledge base it
    # configures. This namespace is ALSO the marker every backend keys its
    # fixed-container rule on: nothing else is ever opened on it.
    # See docs/design/ConfigurationStorageContract.md.
    KV_STORE_CONFIG = "config"


# The fixed tag every backend names the configuration container after.
#
# **It is not a workspace.** Nothing validates it as one, no ``*_WORKSPACE``
# variable may remap it, and no caller may choose it: the configuration
# storage is its own category, and its container is named in code. The
# database backends compose it into their own container name -- PostgreSQL
# writes it into ``LIGHTRAG_CONFIG``'s partition column, MongoDB and
# OpenSearch prefix their collection / index with it. The JSON backend keeps
# one snapshot per workspace (``lightrag/config_shards.py``) and uses the tag
# only as the prefix of its in-process bookkeeping key.
#
# The leading underscore keeps it out of the space of names a tenant would
# be given, so a container is recognisable as ours at a glance in a shared
# PostgreSQL, MongoDB or OpenSearch. See docs/design/ConfigurationStorageContract.md.
CONFIG_CONTAINER_TAG = "_lightrag_config"

# The file a JSON-backed configuration storage keeps in each workspace's
# directory (``WORKING_DIR/<workspace>``, or ``WORKING_DIR`` itself for the
# empty workspace). Named for its role rather than derived as
# ``kv_store_<namespace>.json`` like every data namespace.
CONFIG_JSON_FILE_NAME = "kv_workspace_config.json"

# The deployment-wide files directly under ``WORKING_DIR``: the anchor, the
# shared runtime/migration lock and the first-bind / member-registration
# lock. The JSON configuration's lifetime claim sits beside each workspace
# snapshot under the last name. A workspace named like any of these would
# make its directory collide with one of them (``config_shards``).
ANCHOR_FILE_NAME = "config_storage_anchor.json"
ANCHOR_LOCK_FILE_NAME = ".lightrag_anchor.lock"
ANCHOR_BIND_LOCK_FILE_NAME = ".lightrag_anchor_bind.lock"
CONFIG_CLAIM_FILE_NAME = ".lightrag_storage.lock"


# The prefix a server-global configuration key is RENDERED with, so its rows
# sort beside the per-workspace ones. It is a spelling, not an identity: what
# marks a key as server-global is ``SERVER_SCOPE`` below.
SERVER_CONFIG_SCOPE = "_lightrag_server"


class _ServerScope:
    """The scope of a server-global configuration row.

    An object rather than a magic string, because a workspace name IS a
    string: any string sentinel is a name a tenant can also be given, and with
    the reserved name family retired nothing stops one from being. A tenant so
    named would otherwise be refused its own baseline -- its key would look
    like the server's scope -- which is a startup failure over a legal name.

    Two rows can still RENDER under the same prefix, and that is harmless:
    a suffix is registered with exactly one scope, so a tenant's key and a
    server-global one can never be the same key.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<server scope {SERVER_CONFIG_SCOPE!r}>"


SERVER_SCOPE = _ServerScope()
"""Pass this as a scope to address a server-global key; no workspace can."""


def is_namespace(namespace: str, base_namespace: str | Iterable[str]):
    if isinstance(base_namespace, str):
        return namespace.endswith(base_namespace)
    return any(is_namespace(namespace, ns) for ns in base_namespace)
