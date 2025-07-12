"""
Translations specific to batch operations testing
"""

BATCH_TEST_TRANSLATIONS = {
    # Batch operation tests
    "batch_get_nodes": ["== 测试 get_nodes_batch ==", "== Test get_nodes_batch =="],
    "batch_node_degrees": [
        "== 测试 node_degrees_batch ==",
        "== Test node_degrees_batch ==",
    ],
    "batch_edge_degrees": [
        "== 测试 edge_degrees_batch ==",
        "== Test edge_degrees_batch ==",
    ],
    "batch_get_edges": ["== 测试 get_edges_batch ==", "== Test get_edges_batch =="],
    "test_reverse_edges_batch": [
        "== 测试反向边的批量获取 ==",
        "== Test reverse edges batch get ==",
    ],
    "test_get_nodes_edges_batch": [
        "=== 测试 get_nodes_edges_batch ===",
        "=== Test get_nodes_edges_batch ===",
    ],
    "verify_batch_nodes_edges_undirected": [
        "=== 验证批量获取节点边的无向图特性 ===",
        "=== Verify batch get node edges undirected graph property ===",
    ],
    "test_get_nodes_by_chunk_ids": [
        "== 测试 get_nodes_by_chunk_ids ==",
        "== Test get_nodes_by_chunk_ids ==",
    ],
    "test_single_chunk_id_multiple_nodes": [
        "== 测试单个 chunk_id，匹配多个节点 ==",
        "== Test single chunk_id, matching multiple nodes ==",
    ],
    "test_multiple_chunk_ids_partial_match": [
        "== 测试多个 chunk_id，部分匹配多个节点 ==",
        "== Test multiple chunk_ids, partial matching multiple nodes ==",
    ],
    "test_get_edges_by_chunk_ids": [
        "== 测试 get_edges_by_chunk_ids ==",
        "== Test get_edges_by_chunk_ids ==",
    ],
    "test_single_chunk_id_multiple_edges": [
        "== 测试单个 chunk_id，匹配多条边 ==",
        "== Test single chunk_id, matching multiple edges ==",
    ],
    "test_multiple_chunk_ids_partial_edges": [
        "== 测试多个 chunk_id，部分匹配多条边 ==",
        "== Test multiple chunk_ids, partial matching multiple edges ==",
    ],
    # Results
    "batch_get_nodes_result": [
        "批量获取节点属性结果",
        "Batch get node properties result",
    ],
    "batch_node_degrees_result": [
        "批量获取节点度数结果",
        "Batch get node degrees result",
    ],
    "batch_edge_degrees_result": [
        "批量获取边度数结果",
        "Batch get edge degrees result",
    ],
    "batch_get_edges_result": [
        "批量获取边属性结果",
        "Batch get edge properties result",
    ],
    "batch_get_reverse_edges_result": [
        "批量获取反向边属性结果",
        "Batch get reverse edge properties result",
    ],
    "batch_get_nodes_edges_result": [
        "批量获取节点边结果",
        "Batch get node edges result",
    ],
    # Insert messages
    "insert_node_1": ["插入节点1", "Insert node 1"],
    "insert_node_2": ["插入节点2", "Insert node 2"],
    "insert_node_3": ["插入节点3", "Insert node 3"],
    "insert_node_4": ["插入节点4", "Insert node 4"],
    "insert_node_5": ["插入节点5", "Insert node 5"],
    "insert_edge_1": ["插入边1", "Insert edge 1"],
    "insert_edge_2": ["插入边2", "Insert edge 2"],
    "insert_edge_3": ["插入边3", "Insert edge 3"],
    "insert_edge_4": ["插入边4", "Insert edge 4"],
    "insert_edge_5": ["插入边5", "Insert edge 5"],
    "insert_edge_6": ["插入边6", "Insert edge 6"],
    # Node/Edge Display
    "node_outgoing_edges": ["的出边", "Node outgoing edges"],
    "node_incoming_edges": ["的入边", "Node incoming edges"],
    "node": ["节点", "Node"],
    # Assert messages
    "should_return_nodes": [
        "应返回%d个节点，实际返回 %d 个",
        "Should return %d nodes, actual %d",
    ],
    "should_return_node_degrees": [
        "应返回%d个节点的度数，实际返回 %d 个",
        "Should return %d node degrees, actual %d",
    ],
    "should_return_edge_degrees": [
        "应返回%d条边的度数，实际返回 %d 条",
        "Should return %d edge degrees, actual %d",
    ],
    "should_return_edge_properties": [
        "应返回%d条边的属性，实际返回 %d 条",
        "Should return %d edge properties, actual %d",
    ],
    "should_return_reverse_edge_properties": [
        "应返回%d条反向边的属性，实际返回 %d 条",
        "Should return %d reverse edge properties, actual %d",
    ],
    "should_return_node_edges": [
        "应返回%d个节点的边，实际返回 %d 个",
        "Should return %d node edges, actual %d",
    ],
    "should_be_in_result": [
        "%s 应在返回结果中",
        "%s should be in result",
    ],
    "edge_should_be_in_result": [
        "边 %s -> %s 应在返回结果中",
        "Edge %s -> %s should be in result",
    ],
    "reverse_edge_should_be_in_result": [
        "反向边 %s -> %s 应在返回结果中",
        "Reverse edge %s -> %s should be in result",
    ],
    "node_should_be_in_result": [
        "节点 %s 应在返回结果中",
        "Node %s should be in result",
    ],
    "node_edge_list_should_contain_edge_to": [
        "节点 %s 的边列表中应包含到 %s 的边",
        "Node %s edge list should contain edge to %s",
    ],
    "node_should_have_edges_count": [
        "%s 应有%d条边，实际有 %d 条",
        "%s should have %d edges, actual %d",
    ],
    "chunk_should_have_nodes": [
        "%s 应有%d个节点，实际有 %d 个",
        "%s should have %d nodes, actual %d",
    ],
    "chunk_should_have_edges": [
        "%s 应有%d条边，实际有 %d 条",
        "%s should have %d edges, actual %d",
    ],
    "chunks_should_have_nodes": [
        "%s, %s 应有%d个节点，实际有 %d 个",
        "%s, %s should have %d nodes, actual %d",
    ],
    "chunks_should_have_edges": [
        "%s, %s 应有%d条边，实际有 %d 条",
        "%s, %s should have %d edges, actual %d",
    ],
    "chunk_should_contain_edge": [
        "%s 应包含 %s 到 %s 的边",
        "%s should contain edge from %s to %s",
    ],
    "chunks_should_contain_edge": [
        "%s, %s 应包含 %s 到 %s 的边",
        "%s, %s should contain edge from %s to %s",
    ],
    "node_edge_list_should_contain_connection": [
        "节点 %s 的边列表中应包含与 %s 的连接",
        "Node %s edge list should contain connection with %s",
    ],
    "description_mismatch": [
        "%s 描述不匹配",
        "%s description mismatch",
    ],
    # Success messages
    "undirected_batch_verification_success": [
        "无向图特性验证成功：批量获取的正向和反向边属性一致",
        "Undirected graph property verification successful: batch obtained forward and reverse edge properties are consistent",
    ],
    "undirected_nodes_edges_verification_success": [
        "无向图特性验证成功：批量获取的节点边包含所有相关的边（无论方向）",
        "Undirected graph property verification successful: batch obtained node edges contain all related edges (regardless of direction)",
    ],
    # Test completion
    "batch_operations_test_complete": [
        "\n批量操作测试完成",
        "\nBatch operations test completed",
    ],
    "starting_batch_operations_test": [
        "\n=== 开始批量操作测试 ===",
        "\n=== Starting Batch Operations Test ===",
    ],
}
