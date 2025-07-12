"""
Advanced graph storage test module
"""

from ascii_colors import ASCIIColors
from lightrag.types import KnowledgeGraph
from ..core.translation_engine import t_enhanced as t


async def test_graph_advanced(storage):
    """
    测试图数据库的高级操作:
    1. 使用 node_degree 获取节点的度数
    2. 使用 edge_degree 获取边的度数
    3. 使用 get_node_edges 获取节点的所有边
    4. 使用 get_all_labels 获取所有标签
    5. 使用 get_knowledge_graph 获取知识图谱
    6. 使用 delete_node 删除节点
    7. 使用 remove_nodes 批量删除节点
    8. 使用 remove_edges 删除边
    9. 使用 drop 清理数据
    """
    try:
        # 1. 插入测试数据
        # 插入节点1: 人工智能
        node1_id = t("artificial_intelligence")
        node1_data = {
            "entity_id": node1_id,
            "description": t("ai_desc"),
            "keywords": t("ai_keywords"),
            "entity_type": t("tech_field"),
        }
        print(f"{t('insert_node')} 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # Insert node 2: Machine Learning / 插入节点2: 机器学习
        node2_id = t("machine_learning")
        node2_data = {
            "entity_id": node2_id,
            "description": t("ml_desc"),
            "keywords": t("ml_keywords"),
            "entity_type": t("tech_field"),
        }
        print(f"{t('insert_node')} 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # Insert node 3: Deep Learning / 插入节点3: 深度学习
        node3_id = t("deep_learning")
        node3_data = {
            "entity_id": node3_id,
            "description": t("dl_desc"),
            "keywords": t("dl_keywords"),
            "entity_type": t("tech_field"),
        }
        print(f"{t('insert_node')} 3: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # Insert edge 1: AI -> ML / 插入边1: 人工智能 -> 机器学习
        edge1_data = {
            "relationship": t("contains"),
            "weight": 1.0,
            "description": t("ai_contains_ml"),
        }
        print(f"{t('insert_edge')} 1: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # Insert edge 2: ML -> DL / 插入边2: 机器学习 -> 深度学习
        edge2_data = {
            "relationship": t("contains"),
            "weight": 1.0,
            "description": t("ml_contains_dl"),
        }
        print(f"{t('insert_edge')} 2: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 2. Test node_degree - get node degree / 测试 node_degree - 获取节点的度数
        print(f"== {t('test_node_degree')}: {node1_id} ==")
        node1_degree = await storage.node_degree(node1_id)
        print(t("node_degree_display") % (node1_id, node1_degree))
        assert node1_degree == 1, t("node_degree_should_be") % (
            node1_id,
            1,
            node1_degree,
        )

        # 2.1 Test all node degrees / 测试所有节点的度数
        print(f"== {t('test_all_node_degrees')}")
        node2_degree = await storage.node_degree(node2_id)
        node3_degree = await storage.node_degree(node3_id)
        print(t("node_degree_display") % (node2_id, node2_degree))
        print(t("node_degree_display") % (node3_id, node3_degree))
        assert node2_degree == 2, t("node_degree_should_be") % (
            node2_id,
            2,
            node2_degree,
        )
        assert node3_degree == 1, t("node_degree_should_be") % (
            node3_id,
            1,
            node3_degree,
        )

        # 3. Test edge_degree - get edge degree / 测试 edge_degree - 获取边的度数
        print(f"== {t('test_edge_degree')}: {node1_id} -> {node2_id} ==")
        edge_degree = await storage.edge_degree(node1_id, node2_id)
        print(t("edge_degree_display") % (node1_id, node2_id, edge_degree))
        assert edge_degree == 3, t("edge_degree_should_be") % (
            node1_id,
            node2_id,
            3,
            edge_degree,
        )

        # 3.1 Test reverse edge degree - verify undirected graph property / 测试反向边的度数 - 验证无向图特性
        print(f"== {t('test_reverse_edge_degree')}: {node2_id} -> {node1_id}")
        reverse_edge_degree = await storage.edge_degree(node2_id, node1_id)
        print(t("edge_degree_display") % (node2_id, node1_id, reverse_edge_degree))
        assert edge_degree == reverse_edge_degree, t(
            "forward_reverse_edge_inconsistent"
        )
        print(t("undirected_verification_success"))

        # 4. Test get_node_edges - get all edges of node / 测试 get_node_edges - 获取节点的所有边
        print(f"== {t('test_get_node_edges')}: {node2_id} ==")
        node2_edges = await storage.get_node_edges(node2_id)
        print(f"{t('node_degree')} {node2_id} {t('all_edges')}: {node2_edges}")

        assert len(node2_edges) == 2, t("node_should_have_edges") % (
            node2_id,
            2,
            len(node2_edges),
        )

        # 4.1 Verify undirected graph property of node edges / 验证节点边的无向图特性
        print(f"== {t('verify_node_edges_undirected')}")
        # Check if contains connections with node1 and node3 (regardless of direction) / 检查是否包含与node1和node3的连接关系（无论方向）
        has_connection_with_node1 = False
        has_connection_with_node3 = False
        for edge in node2_edges:
            # Check if has connection with node1 (regardless of direction) / 检查是否有与node1的连接（无论方向）
            if (edge[0] == node1_id and edge[1] == node2_id) or (
                edge[0] == node2_id and edge[1] == node1_id
            ):
                has_connection_with_node1 = True
            # Check if has connection with node3 (regardless of direction) / 检查是否有与node3的连接（无论方向）
            if (edge[0] == node2_id and edge[1] == node3_id) or (
                edge[0] == node3_id and edge[1] == node2_id
            ):
                has_connection_with_node3 = True

        assert has_connection_with_node1, t("node_edge_should_contain_connection") % (
            node2_id,
            node1_id,
        )
        assert has_connection_with_node3, t("node_edge_should_contain_connection") % (
            node2_id,
            node3_id,
        )
        print(t("undirected_node_edges_success") % node2_id)

        # 5. Test get_all_labels - get all labels / 测试 get_all_labels - 获取所有标签
        print(t("test_get_all_labels"))
        all_labels = await storage.get_all_labels()
        print(f"{t('all_labels')}: {all_labels}")
        assert len(all_labels) == 3, t("should_have_labels") % (3, len(all_labels))
        assert node1_id in all_labels, t("should_be_in_label_list") % node1_id
        assert node2_id in all_labels, t("should_be_in_label_list") % node2_id
        assert node3_id in all_labels, t("should_be_in_label_list") % node3_id

        # 6. Test get_knowledge_graph - get knowledge graph / 测试 get_knowledge_graph - 获取知识图谱
        print(t("test_get_knowledge_graph"))
        kg = await storage.get_knowledge_graph("*", max_depth=2, max_nodes=10)
        print(f"{t('knowledge_graph_nodes')}: {len(kg.nodes)}")
        print(f"{t('knowledge_graph_edges')}: {len(kg.edges)}")
        assert isinstance(kg, KnowledgeGraph), t("result_should_be_kg_type")
        assert len(kg.nodes) == 3, t("kg_should_have_nodes") % (3, len(kg.nodes))
        assert len(kg.edges) == 2, t("kg_should_have_edges") % (2, len(kg.edges))

        # 7. Test delete_node - delete node / 测试 delete_node - 删除节点
        print(f"== {t('test_delete_node')}: {node3_id} == ")
        await storage.delete_node(node3_id)
        node3_props = await storage.get_node(node3_id)
        print(t("query_after_deletion_display") % (node3_id, node3_props))
        assert node3_props is None, t("node_should_be_deleted") % node3_id

        # Re-insert node3 for subsequent testing / 重新插入节点3用于后续测试
        await storage.upsert_node(node3_id, node3_data)
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 8. Test remove_edges - delete edges / 测试 remove_edges - 删除边
        print(f"== {t('test_remove_edges')}: {node2_id} -> {node3_id} == ")
        await storage.remove_edges([(node2_id, node3_id)])
        edge_props = await storage.get_edge(node2_id, node3_id)
        print(t("query_edge_after_deletion_display") % (node2_id, node3_id, edge_props))
        assert edge_props is None, t("edge_should_be_deleted") % (node2_id, node3_id)

        # 8.1 Verify undirected graph property for edge deletion / 验证删除边的无向图特性
        print(f"== {t('verify_undirected_property')}: {node3_id} -> {node2_id} == ")
        reverse_edge_props = await storage.get_edge(node3_id, node2_id)
        print(
            t("query_reverse_edge_after_deletion")
            % (node3_id, node2_id, reverse_edge_props)
        )
        assert reverse_edge_props is None, t("reverse_edge_should_be_deleted") % (
            node3_id,
            node2_id,
        )
        print(t("undirected_deletion_success"))

        # 9. Test remove_nodes - batch delete nodes / 测试 remove_nodes - 批量删除节点
        print(f"== {t('test_remove_nodes')}: [{node2_id}, {node3_id}] == ")
        await storage.remove_nodes([node2_id, node3_id])
        node2_props = await storage.get_node(node2_id)
        node3_props = await storage.get_node(node3_id)
        print(t("query_after_deletion_display") % (node2_id, node2_props))
        print(t("query_after_deletion_display") % (node3_id, node3_props))
        assert node2_props is None, t("node_should_be_deleted") % node2_id
        assert node3_props is None, t("node_should_be_deleted") % node3_id

        ASCIIColors.green(t("advanced_test_complete"))
        return True

    except Exception as e:
        ASCIIColors.red(f"{t('test_error')}: {str(e)}")
        return False


async def run_advanced_test():
    """Run the advanced test standalone"""
    from ..core.storage_setup import initialize_graph_test_storage

    storage = await initialize_graph_test_storage()
    if storage is None:
        ASCIIColors.red(t("init_storage_failed"))
        return False

    try:
        ASCIIColors.blue(t("starting_advanced_test"))
        result = await test_graph_advanced(storage)
        return result
    finally:
        if hasattr(storage, "close"):
            await storage.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_advanced_test())
