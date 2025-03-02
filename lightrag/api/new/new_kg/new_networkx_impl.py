from dataclasses import dataclass
from lightrag.kg.networkx_impl import NetworkXStorage as BaseGraphStorage
from lightrag.utils import (
    logger
)

@dataclass
class NewNetworkXStorage(BaseGraphStorage):
    def __post_init__(self):
        logger.info("Initializing New NetworkXStorage")
        super().__post_init__()

    async def query_all(self):
        try:
            nodes = list(self._graph.nodes(data=True))
            edges = list(self._graph.edges(data=True))
            nodes_list = []
            edges_list = []
            for node in nodes:
                node_data = {
                    "id": node[0],
                    "label": node[0],
                    "entity_type": node[1].get("entity_type", None),
                    "description": node[1].get("description", None),
                    "source_id": node[1].get("source_id", None),
                }
                nodes_list.append(node_data)
            for edge in edges:
                edge_data = {
                    "id": edge[0] + "_" + edge[1],
                    "source": edge[0],
                    "target": edge[1],
                    "weight": edge[2].get("weight", 0.0),
                    "description": edge[2].get("description", None),
                    "keywords": edge[2].get("keywords", None),
                    "source_id": edge[2].get("source_id", None),
                }
                edges_list.append(edge_data)
            return {"nodes": nodes_list, "edges": edges_list}
        except Exception as e:
            logger.error(f"Error occurred while querying all nodes: {e}")

    async def delete_all(self):
        try:
            self._graph.clear()
        except Exception as e:
            logger.error(f"Error occurred while deleting all nodes: {e}")
