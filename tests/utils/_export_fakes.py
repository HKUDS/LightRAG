"""Storage stand-ins shared by the ``aexport_data`` tests."""

from __future__ import annotations

import networkx as nx

from lightrag.kg.networkx_impl import NetworkXStorage


def make_networkx_storage(graph: nx.Graph) -> NetworkXStorage:
    """A ``NetworkXStorage`` reading ``graph`` with no file or lock behind it."""
    storage = NetworkXStorage.__new__(NetworkXStorage)
    storage.workspace = "test"
    storage.namespace = "chunk_entity_relation"

    async def _get_graph():
        return graph

    storage._get_graph = _get_graph
    return storage


class NullVdb:
    """A vector storage with no records and no ``client_storage``."""

    async def get_by_id(self, _id: str) -> None:
        return None
