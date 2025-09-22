"""
Undirected graph property test module for graph storage.
Tests the undirected behavior of the graph storage implementation.
"""

import asyncio
from ..core.translation_engine import t_enhanced as t
from ascii_colors import ASCIIColors


async def test_graph_undirected_property(storage):
    """
    Test undirected graph property of storage (bilingual).
    """
    try:
        # 1. Insert test data
        node1_id = t("computer_science")
        node1_data = {
            "entity_id": node1_id,
            "description": t("cs_desc"),
            "keywords": t("cs_keywords"),
            "entity_type": t("subject"),
        }
        print(f"{t('insert_node_1')}: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        node2_id = t("data_structure")
        node2_data = {
            "entity_id": node2_id,
            "description": t("ds_desc"),
            "keywords": t("ds_keywords"),
            "entity_type": t("concept"),
        }
        print(f"{t('insert_node_2')}: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        node3_id = t("algorithm")
        node3_data = {
            "entity_id": node3_id,
            "description": t("algo_desc"),
            "keywords": t("algo_keywords"),
            "entity_type": t("concept"),
        }
        print(f"{t('insert_node_3')}: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 2. Test undirected property after inserting edge
        print(t("test_insert_edge_undirected_property"))
        edge1_data = {
            "relationship": t("contains"),
            "weight": 1.0,
            "description": t("cs_contains_ds"),
        }
        print(f"{t('insert_edge_1')}: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        forward_edge = await storage.get_edge(node1_id, node2_id)
        print(f"{t('forward_edge_props')}: {forward_edge}")
        assert (
            forward_edge is not None
        ), f"{t('unable_read_edge')}: {node1_id} -> {node2_id}"

        reverse_edge = await storage.get_edge(node2_id, node1_id)
        print(f"{t('reverse_edge_props')}: {reverse_edge}")
        assert (
            reverse_edge is not None
        ), f"{t('unable_read_reverse_edge')}: {node2_id} -> {node1_id}"

        assert forward_edge == reverse_edge, t("forward_reverse_inconsistent")
        print(t("undirected_verification_success"))

        # 3. Test edge degree undirected property
        print(t("test_edge_degree_undirected_property"))
        edge2_data = {
            "relationship": t("contains"),
            "weight": 1.0,
            "description": t("cs_contains_algo"),
        }
        print(f"{t('insert_edge_2')}: {node1_id} -> {node3_id}")
        await storage.upsert_edge(node1_id, node3_id, edge2_data)

        forward_degree = await storage.edge_degree(node1_id, node2_id)
        reverse_degree = await storage.edge_degree(node2_id, node1_id)
        print(f"{t('forward_edge_degree')}: {node1_id} -> {node2_id}: {forward_degree}")
        print(f"{t('reverse_edge_degree')}: {node2_id} -> {node1_id}: {reverse_degree}")
        assert forward_degree == reverse_degree, t("forward_reverse_inconsistent")
        print(t("undirected_edge_degree_verification_success"))

        # 4. Test undirected property after deleting edge
        print(t("test_delete_edge_undirected_property"))
        print(f"{t('delete_edge')}: {node1_id} -> {node2_id}")
        await storage.remove_edges([(node1_id, node2_id)])

        forward_edge = await storage.get_edge(node1_id, node2_id)
        print(
            f"{t('query_forward_edge_after_delete')}: {node1_id} -> {node2_id}: {forward_edge}"
        )
        assert (
            forward_edge is None
        ), f"{t('edge_should_be_deleted')}: {node1_id} -> {node2_id}"

        reverse_edge = await storage.get_edge(node2_id, node1_id)
        print(
            f"{t('query_reverse_edge_after_delete')}: {node2_id} -> {node1_id}: {reverse_edge}"
        )
        assert reverse_edge is None, t("reverse_edge_should_be_deleted")
        print(t("undirected_delete_verification_success"))

        # 5. Test batch undirected property
        print(t("test_batch_undirected_property"))
        await storage.upsert_edge(node1_id, node2_id, edge1_data)
        edge_dicts = [
            {"src": node1_id, "tgt": node2_id},
            {"src": node1_id, "tgt": node3_id},
        ]
        reverse_edge_dicts = [
            {"src": node2_id, "tgt": node1_id},
            {"src": node3_id, "tgt": node1_id},
        ]
        edges_dict = await storage.get_edges_batch(edge_dicts)
        reverse_edges_dict = await storage.get_edges_batch(reverse_edge_dicts)
        print(f"{t('batch_get_edges_result')}: {edges_dict.keys()}")
        print(f"{t('batch_get_reverse_edges_result')}: {reverse_edges_dict.keys()}")
        for (src, tgt), props in edges_dict.items():
            assert (
                (
                    tgt,
                    src,
                )
                in reverse_edges_dict
            ), f"{t('reverse_edge_should_be_in_result')}: {tgt} -> {src}"
            assert props == reverse_edges_dict[(tgt, src)], t(
                "forward_reverse_inconsistent"
            )
        print(t("undirected_batch_verification_success"))

        # 6. Test batch get node edges undirected property
        print(t("test_batch_get_node_edges_undirected_property"))
        nodes_edges = await storage.get_nodes_edges_batch([node1_id, node2_id])
        print(f"{t('batch_get_nodes_edges_result')}: {nodes_edges.keys()}")
        node1_edges = nodes_edges[node1_id]
        node2_edges = nodes_edges[node2_id]
        has_edge_to_node2 = any(
            (src == node1_id and tgt == node2_id) for src, tgt in node1_edges
        )
        has_edge_to_node3 = any(
            (src == node1_id and tgt == node3_id) for src, tgt in node1_edges
        )
        assert (
            has_edge_to_node2
        ), f"{t('node_edge_should_contain')}: {node1_id} -> {node2_id}"
        assert (
            has_edge_to_node3
        ), f"{t('node_edge_should_contain')}: {node1_id} -> {node3_id}"
        has_edge_to_node1 = any(
            (src == node2_id and tgt == node1_id)
            or (src == node1_id and tgt == node2_id)
            for src, tgt in node2_edges
        )
        assert (
            has_edge_to_node1
        ), f"{t('node_edge_should_contain_connection')}: {node2_id} <-> {node1_id}"
        print(t("undirected_nodes_edges_verification_success"))

        ASCIIColors.green(t("undirected_test_complete"))
        return True

    except Exception as e:
        ASCIIColors.red(f"{t('test_error')}: {str(e)}")
        return False


# For direct execution/testing
async def main():
    """Test function for direct execution"""
    from ..core.storage_setup import initialize_graph_test_storage

    print(t("starting_undirected_graph_test"))
    storage = await initialize_graph_test_storage()

    if storage:
        result = await test_graph_undirected_property(storage)
        print(f"Test result: {result}")
    else:
        print("Storage initialization failed")


if __name__ == "__main__":
    asyncio.run(main())
