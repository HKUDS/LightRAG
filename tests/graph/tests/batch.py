"""
Batch operations test module for graph storage.
Tests all batch operations including get_nodes_batch, node_degrees_batch,
edge_degrees_batch, get_edges_batch, get_nodes_edges_batch, and chunk-based operations.
"""

import asyncio
from lightrag.constants import GRAPH_FIELD_SEP
from ..core.translation_engine import t_enhanced as t
from ascii_colors import ASCIIColors


async def test_graph_batch_operations(storage):
    """
    测试图数据库的批量操作:
    1. 使用 get_nodes_batch 批量获取多个节点的属性
    2. 使用 node_degrees_batch 批量获取多个节点的度数
    3. 使用 edge_degrees_batch 批量获取多个边的度数
    4. 使用 get_edges_batch 批量获取多个边的属性
    5. 使用 get_nodes_edges_batch 批量获取多个节点的所有边
    """
    try:
        chunk1_id = "1"
        chunk2_id = "2"
        chunk3_id = "3"
        # 1. 插入测试数据
        # 插入节点1: 人工智能
        node1_id = t("artificial_intelligence")
        node1_data = {
            "entity_id": node1_id,
            "description": t("ai_desc"),
            "keywords": t("ai_keywords"),
            "entity_type": t("tech_field"),
            "source_id": GRAPH_FIELD_SEP.join([chunk1_id, chunk2_id]),
        }
        print(f"{t('insert_node_1')}: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 插入节点2: 机器学习
        node2_id = t("machine_learning")
        node2_data = {
            "entity_id": node2_id,
            "description": t("ml_desc"),
            "keywords": t("ml_keywords"),
            "entity_type": t("tech_field"),
            "source_id": GRAPH_FIELD_SEP.join([chunk2_id, chunk3_id]),
        }
        print(f"{t('insert_node_2')}: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 插入节点3: 深度学习
        node3_id = t("deep_learning")
        node3_data = {
            "entity_id": node3_id,
            "description": t("dl_desc"),
            "keywords": t("dl_keywords"),
            "entity_type": t("tech_field"),
            "source_id": GRAPH_FIELD_SEP.join([chunk3_id]),
        }
        print(f"{t('insert_node_3')}: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 插入节点4: 自然语言处理
        node4_id = t("natural_language_processing")
        node4_data = {
            "entity_id": node4_id,
            "description": t("nlp_desc"),
            "keywords": t("nlp_keywords"),
            "entity_type": t("tech_field"),
        }
        print(f"{t('insert_node_4')}: {node4_id}")
        await storage.upsert_node(node4_id, node4_data)

        # 插入节点5: 计算机视觉
        node5_id = t("computer_vision")
        node5_data = {
            "entity_id": node5_id,
            "description": t("cv_desc"),
            "keywords": t("cv_keywords"),
            "entity_type": t("tech_field"),
        }
        print(f"{t('insert_node_5')}: {node5_id}")
        await storage.upsert_node(node5_id, node5_data)

        # 插入边1: 人工智能 -> 机器学习
        edge1_data = {
            "relationship": t("contains"),
            "weight": 1.0,
            "description": t("ai_contains_ml"),
            "source_id": GRAPH_FIELD_SEP.join([chunk1_id, chunk2_id]),
        }
        print(f"{t('insert_edge_1')}: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # 插入边2: 机器学习 -> 深度学习
        edge2_data = {
            "relationship": t("contains"),
            "weight": 1.0,
            "description": t("ml_contains_dl"),
            "source_id": GRAPH_FIELD_SEP.join([chunk2_id, chunk3_id]),
        }
        print(f"{t('insert_edge_2')}: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 插入边3: 人工智能 -> 自然语言处理
        edge3_data = {
            "relationship": t("contains"),
            "weight": 1.0,
            "description": t("ai_contains_nlp"),
            "source_id": GRAPH_FIELD_SEP.join([chunk3_id]),
        }
        print(f"{t('insert_edge_3')}: {node1_id} -> {node4_id}")
        await storage.upsert_edge(node1_id, node4_id, edge3_data)

        # 插入边4: 人工智能 -> 计算机视觉
        edge4_data = {
            "relationship": t("contains"),
            "weight": 1.0,
            "description": t("ai_contains_cv_desc"),
        }
        print(f"{t('insert_edge_4')}: {node1_id} -> {node5_id}")
        await storage.upsert_edge(node1_id, node5_id, edge4_data)

        # 插入边5: 深度学习 -> 自然语言处理
        edge5_data = {
            "relationship": t("dl_applied_nlp_relationship"),
            "weight": 0.8,
            "description": t("dl_applied_nlp_desc"),
        }
        print(f"{t('insert_edge_5')}: {node3_id} -> {node4_id}")
        await storage.upsert_edge(node3_id, node4_id, edge5_data)

        # 插入边6: 深度学习 -> 计算机视觉
        edge6_data = {
            "relationship": t("dl_applied_cv_relationship"),
            "weight": 0.8,
            "description": t("dl_applied_cv_desc"),
        }
        print(f"{t('insert_edge_6')}: {node3_id} -> {node5_id}")
        await storage.upsert_edge(node3_id, node5_id, edge6_data)

        # 2. 测试 get_nodes_batch - 批量获取多个节点的属性
        print(t("batch_get_nodes"))
        node_ids = [node1_id, node2_id, node3_id]
        nodes_dict = await storage.get_nodes_batch(node_ids)
        print(f"{t('batch_get_nodes_result')}: {nodes_dict.keys()}")
        assert len(nodes_dict) == 3, t("should_return_nodes") % (3, len(nodes_dict))
        assert node1_id in nodes_dict, t("should_be_in_result") % node1_id
        assert node2_id in nodes_dict, t("should_be_in_result") % node2_id
        assert node3_id in nodes_dict, t("should_be_in_result") % node3_id
        assert nodes_dict[node1_id]["description"] == node1_data["description"], (
            t("description_mismatch") % node1_id
        )
        assert nodes_dict[node2_id]["description"] == node2_data["description"], (
            t("description_mismatch") % node2_id
        )
        assert nodes_dict[node3_id]["description"] == node3_data["description"], (
            t("description_mismatch") % node3_id
        )

        # 3. 测试 node_degrees_batch - 批量获取多个节点的度数
        print(t("batch_node_degrees"))
        node_degrees = await storage.node_degrees_batch(node_ids)
        print(f"{t('batch_node_degrees_result')}: {node_degrees}")
        assert len(node_degrees) == 3, t("should_return_node_degrees") % (
            3,
            len(node_degrees),
        )
        assert node1_id in node_degrees, t("should_be_in_result") % node1_id
        assert node2_id in node_degrees, t("should_be_in_result") % node2_id
        assert node3_id in node_degrees, t("should_be_in_result") % node3_id
        assert node_degrees[node1_id] == 3, t("node_degree_should_be") % (
            node1_id,
            3,
            node_degrees[node1_id],
        )
        assert node_degrees[node2_id] == 2, t("node_degree_should_be") % (
            node2_id,
            2,
            node_degrees[node2_id],
        )
        assert node_degrees[node3_id] == 3, t("node_degree_should_be") % (
            node3_id,
            3,
            node_degrees[node3_id],
        )

        # 4. 测试 edge_degrees_batch - 批量获取多个边的度数
        print(t("batch_edge_degrees"))
        edges = [(node1_id, node2_id), (node2_id, node3_id), (node3_id, node4_id)]
        edge_degrees = await storage.edge_degrees_batch(edges)
        print(f"{t('batch_edge_degrees_result')}: {edge_degrees}")
        assert len(edge_degrees) == 3, t("should_return_edge_degrees") % (
            3,
            len(edge_degrees),
        )
        assert (
            node1_id,
            node2_id,
        ) in edge_degrees, t(
            "edge_should_be_in_result"
        ) % (node1_id, node2_id)
        assert (
            node2_id,
            node3_id,
        ) in edge_degrees, t(
            "edge_should_be_in_result"
        ) % (node2_id, node3_id)
        assert (
            node3_id,
            node4_id,
        ) in edge_degrees, t(
            "edge_should_be_in_result"
        ) % (node3_id, node4_id)

        # 验证边的度数是否正确（源节点度数 + 目标节点度数）
        assert (
            edge_degrees[(node1_id, node2_id)] == 5
        ), f"边 {node1_id} -> {node2_id} 度数应为5，实际为 {edge_degrees[(node1_id, node2_id)]}"
        assert (
            edge_degrees[(node2_id, node3_id)] == 5
        ), f"边 {node2_id} -> {node3_id} 度数应为5，实际为 {edge_degrees[(node2_id, node3_id)]}"
        assert (
            edge_degrees[(node3_id, node4_id)] == 5
        ), f"边 {node3_id} -> {node4_id} 度数应为5，实际为 {edge_degrees[(node3_id, node4_id)]}"

        # 5. 测试 get_edges_batch - 批量获取多个边的属性
        print(t("batch_get_edges"))
        # 将元组列表转换为Neo4j风格的字典列表
        edge_dicts = [{"src": src, "tgt": tgt} for src, tgt in edges]
        edges_dict = await storage.get_edges_batch(edge_dicts)
        print(f"{t('batch_get_edges_result')}: {edges_dict.keys()}")
        assert len(edges_dict) == 3, t("should_return_edge_properties") % (
            3,
            len(edges_dict),
        )
        assert (
            node1_id,
            node2_id,
        ) in edges_dict, t(
            "edge_should_be_in_result"
        ) % (node1_id, node2_id)
        assert (
            node2_id,
            node3_id,
        ) in edges_dict, t(
            "edge_should_be_in_result"
        ) % (node2_id, node3_id)
        assert (
            node3_id,
            node4_id,
        ) in edges_dict, t(
            "edge_should_be_in_result"
        ) % (node3_id, node4_id)
        assert (
            edges_dict[(node1_id, node2_id)]["relationship"]
            == edge1_data["relationship"]
        ), f"边 {node1_id} -> {node2_id} 关系不匹配"
        assert (
            edges_dict[(node2_id, node3_id)]["relationship"]
            == edge2_data["relationship"]
        ), f"边 {node2_id} -> {node3_id} 关系不匹配"
        assert (
            edges_dict[(node3_id, node4_id)]["relationship"]
            == edge5_data["relationship"]
        ), f"边 {node3_id} -> {node4_id} 关系不匹配"

        # 5.1 测试反向边的批量获取 - 验证无向图特性
        print(t("test_reverse_edges_batch"))
        # 创建反向边的字典列表
        reverse_edge_dicts = [{"src": tgt, "tgt": src} for src, tgt in edges]
        reverse_edges_dict = await storage.get_edges_batch(reverse_edge_dicts)
        print(f"{t('batch_get_reverse_edges_result')}: {reverse_edges_dict.keys()}")
        assert len(reverse_edges_dict) == 3, t(
            "should_return_reverse_edge_properties"
        ) % (3, len(reverse_edges_dict))

        # 验证正向和反向边的属性是否一致
        for (src, tgt), props in edges_dict.items():
            assert (
                tgt,
                src,
            ) in reverse_edges_dict, t(
                "reverse_edge_should_be_in_result"
            ) % (tgt, src)
            assert (
                props == reverse_edges_dict[(tgt, src)]
            ), f"边 {src} -> {tgt} 和反向边 {tgt} -> {src} 的属性不一致"

        print(t("undirected_batch_verification_success"))

        # 6. 测试 get_nodes_edges_batch - 批量获取多个节点的所有边
        print(t("test_get_nodes_edges_batch"))
        nodes_edges = await storage.get_nodes_edges_batch([node1_id, node3_id])
        print(f"{t('batch_get_nodes_edges_result')}: {nodes_edges.keys()}")
        assert len(nodes_edges) == 2, t("should_return_node_edges") % (
            2,
            len(nodes_edges),
        )
        assert node1_id in nodes_edges, t("should_be_in_result") % node1_id
        assert node3_id in nodes_edges, t("should_be_in_result") % node3_id
        assert len(nodes_edges[node1_id]) == 3, t("node_should_have_edges_count") % (
            node1_id,
            3,
            len(nodes_edges[node1_id]),
        )
        assert len(nodes_edges[node3_id]) == 3, t("node_should_have_edges_count") % (
            node3_id,
            3,
            len(nodes_edges[node3_id]),
        )

        # 6.1 验证批量获取节点边的无向图特性
        print(t("verify_batch_nodes_edges_undirected"))

        # 检查节点1的边是否包含所有相关的边（无论方向）
        node1_outgoing_edges = [
            (src, tgt) for src, tgt in nodes_edges[node1_id] if src == node1_id
        ]
        node1_incoming_edges = [
            (src, tgt) for src, tgt in nodes_edges[node1_id] if tgt == node1_id
        ]
        print(
            f"{t('node')} {node1_id} {t('node_outgoing_edges')}: {node1_outgoing_edges}"
        )
        print(
            f"{t('node')} {node1_id} {t('node_incoming_edges')}: {node1_incoming_edges}"
        )

        # 检查是否包含到机器学习、自然语言处理和计算机视觉的边
        has_edge_to_node2 = any(tgt == node2_id for _, tgt in node1_outgoing_edges)
        has_edge_to_node4 = any(tgt == node4_id for _, tgt in node1_outgoing_edges)
        has_edge_to_node5 = any(tgt == node5_id for _, tgt in node1_outgoing_edges)

        assert has_edge_to_node2, t("node_edge_list_should_contain_edge_to") % (
            node1_id,
            node2_id,
        )
        assert has_edge_to_node4, t("node_edge_list_should_contain_edge_to") % (
            node1_id,
            node4_id,
        )
        assert has_edge_to_node5, t("node_edge_list_should_contain_edge_to") % (
            node1_id,
            node5_id,
        )

        # 检查节点3的边是否包含所有相关的边（无论方向）
        node3_outgoing_edges = [
            (src, tgt) for src, tgt in nodes_edges[node3_id] if src == node3_id
        ]
        node3_incoming_edges = [
            (src, tgt) for src, tgt in nodes_edges[node3_id] if tgt == node3_id
        ]
        print(
            f"{t('node')} {node3_id} {t('node_outgoing_edges')}: {node3_outgoing_edges}"
        )
        print(
            f"{t('node')} {node3_id} {t('node_incoming_edges')}: {node3_incoming_edges}"
        )

        # 检查是否包含与机器学习、自然语言处理和计算机视觉的连接（忽略方向）
        has_connection_with_node2 = any(
            (src == node2_id and tgt == node3_id)
            or (src == node3_id and tgt == node2_id)
            for src, tgt in nodes_edges[node3_id]
        )
        has_connection_with_node4 = any(
            (src == node3_id and tgt == node4_id)
            or (src == node4_id and tgt == node3_id)
            for src, tgt in nodes_edges[node3_id]
        )
        has_connection_with_node5 = any(
            (src == node3_id and tgt == node5_id)
            or (src == node5_id and tgt == node3_id)
            for src, tgt in nodes_edges[node3_id]
        )

        assert has_connection_with_node2, t(
            "node_edge_list_should_contain_connection"
        ) % (node3_id, node2_id)
        assert has_connection_with_node4, t(
            "node_edge_list_should_contain_connection"
        ) % (node3_id, node4_id)
        assert has_connection_with_node5, t(
            "node_edge_list_should_contain_connection"
        ) % (node3_id, node5_id)

        print(t("undirected_nodes_edges_verification_success"))

        # 7. 测试 get_nodes_by_chunk_ids - 批量根据 chunk_ids 获取多个节点

        print(t("test_get_nodes_by_chunk_ids"))
        nodes = await storage.get_nodes_by_chunk_ids([chunk2_id])
        assert len(nodes) == 2, t("chunk_should_have_nodes") % (
            chunk2_id,
            2,
            len(nodes),
        )

        has_node1 = any(node["entity_id"] == node1_id for node in nodes)
        has_node2 = any(node["entity_id"] == node2_id for node in nodes)

        print(t("test_single_chunk_id_multiple_nodes"))
        assert has_node1, t("node_should_be_in_result") % node1_id
        assert has_node2, t("node_should_be_in_result") % node2_id

        print(t("test_multiple_chunk_ids_partial_match"))
        nodes = await storage.get_nodes_by_chunk_ids([chunk2_id, chunk3_id])
        assert len(nodes) == 3, t("chunks_should_have_nodes") % (
            chunk2_id,
            chunk3_id,
            3,
            len(nodes),
        )

        has_node1 = any(node["entity_id"] == node1_id for node in nodes)
        has_node2 = any(node["entity_id"] == node2_id for node in nodes)
        has_node3 = any(node["entity_id"] == node3_id for node in nodes)

        assert has_node1, t("node_should_be_in_result") % node1_id
        assert has_node2, t("node_should_be_in_result") % node2_id
        assert has_node3, t("node_should_be_in_result") % node3_id

        # 8. 测试 get_edges_by_chunk_ids - 批量根据 chunk_ids 获取多条边
        print(t("test_get_edges_by_chunk_ids"))

        edges = await storage.get_edges_by_chunk_ids([chunk2_id])
        assert len(edges) == 2, t("chunk_should_have_edges") % (
            chunk2_id,
            2,
            len(edges),
        )
        print(t("test_single_chunk_id_multiple_edges"))

        has_edge_node1_node2 = any(
            edge["source"] == node1_id and edge["target"] == node2_id for edge in edges
        )
        has_edge_node2_node3 = any(
            edge["source"] == node2_id and edge["target"] == node3_id for edge in edges
        )

        assert has_edge_node1_node2, t("chunk_should_contain_edge") % (
            chunk2_id,
            node1_id,
            node2_id,
        )
        assert has_edge_node2_node3, t("chunk_should_contain_edge") % (
            chunk2_id,
            node2_id,
            node3_id,
        )

        print(t("test_multiple_chunk_ids_partial_edges"))
        edges = await storage.get_edges_by_chunk_ids([chunk2_id, chunk3_id])
        assert len(edges) == 3, t("chunks_should_have_edges") % (
            chunk2_id,
            chunk3_id,
            3,
            len(edges),
        )

        has_edge_node1_node2 = any(
            edge["source"] == node1_id and edge["target"] == node2_id for edge in edges
        )
        has_edge_node2_node3 = any(
            edge["source"] == node2_id and edge["target"] == node3_id for edge in edges
        )
        has_edge_node1_node4 = any(
            edge["source"] == node1_id and edge["target"] == node4_id for edge in edges
        )

        assert has_edge_node1_node2, t("chunks_should_contain_edge") % (
            chunk2_id,
            chunk3_id,
            node1_id,
            node2_id,
        )
        assert has_edge_node2_node3, t("chunks_should_contain_edge") % (
            chunk2_id,
            chunk3_id,
            node2_id,
            node3_id,
        )
        assert has_edge_node1_node4, t("chunks_should_contain_edge") % (
            chunk2_id,
            chunk3_id,
            node1_id,
            node4_id,
        )

        ASCIIColors.green(t("batch_operations_test_complete"))
        return True

    except Exception as e:
        ASCIIColors.red(f"{t('test_error')}: {str(e)}")
        return False


# For direct execution/testing
async def main():
    """Test function for direct execution"""
    from ..core.storage_setup import (
        setup_kuzu_test_environment,
        initialize_graph_test_storage,
    )

    print(t("starting_batch_operations_test"))
    storage = await initialize_graph_test_storage()

    if storage:
        result = await test_graph_batch_operations(storage)
        print(f"Test result: {result}")
    else:
        print("Storage initialization failed")


if __name__ == "__main__":
    asyncio.run(main())
