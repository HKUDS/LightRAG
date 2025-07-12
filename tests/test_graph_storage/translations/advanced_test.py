"""
Translations specific to advanced graph testing
"""

ADVANCED_TEST_TRANSLATIONS = {
    # Node data for advanced tests
    "ai_keywords": [
        "人工智能,机器学习,深度学习,神经网络,算法",
        "artificial intelligence,machine learning,deep learning,neural networks,algorithms",
    ],
    "ml_desc": [
        "机器学习是人工智能的一个子领域，使计算机能够在没有明确编程的情况下学习和改进。",
        "Machine Learning is a subset of artificial intelligence that enables computers to learn and improve without being explicitly programmed.",
    ],
    "ml_keywords": [
        "机器学习,监督学习,无监督学习,强化学习,算法",
        "machine learning,supervised learning,unsupervised learning,reinforcement learning,algorithms",
    ],
    "dl_desc": [
        "深度学习是机器学习的一个子领域，使用具有多层的神经网络来模拟人脑的学习过程。",
        "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to mimic the human brain's learning process.",
    ],
    "dl_keywords": [
        "深度学习,神经网络,卷积神经网络,循环神经网络,人工神经网络",
        "deep learning,neural networks,convolutional neural networks,recurrent neural networks,artificial neural networks",
    ],
    "contains": [
        "包含",
        "contains",
    ],
    "ai_contains_ml": [
        "人工智能领域包含机器学习这个重要的子领域",
        "The field of artificial intelligence contains machine learning as an important subfield",
    ],
    "ml_contains_dl": [
        "机器学习领域包含深度学习这个重要的子领域",
        "The field of machine learning contains deep learning as an important subfield",
    ],
    # Advanced test operations
    "test_node_degree": ["测试 node_degree", "Test node_degree"],
    "test_all_node_degrees": ["测试所有节点的度数", "Test all node degrees"],
    "test_edge_degree": ["测试 edge_degree", "Test edge_degree"],
    "test_reverse_edge_degree": ["测试反向边的度数", "Test reverse edge degree"],
    "test_get_node_edges": ["测试 get_node_edges", "Test get_node_edges"],
    "test_get_all_labels": ["== 测试 get_all_labels ==", "== Test get_all_labels =="],
    "test_get_knowledge_graph": [
        "== 测试 get_knowledge_graph ==",
        "== Test get_knowledge_graph ==",
    ],
    "test_delete_node": ["测试 delete_node", "Test delete_node"],
    "test_remove_edges": ["测试 remove_edges", "Test remove_edges"],
    "test_remove_nodes": ["测试 remove_nodes", "Test remove_nodes"],
    # Verification operations
    "verify_undirected_property": [
        "验证无向图特性",
        "Verify undirected graph property",
    ],
    "verify_node_edges_undirected": [
        "验证节点边的无向图特性",
        "Verify node edges undirected property",
    ],
    # Display messages
    "node_degree": ["节点度数", "Node degree"],
    "edge_degree": ["边度数", "Edge degree"],
    "reverse_edge_degree": ["反向边度数", "Reverse edge degree"],
    "all_edges": ["所有边", "All edges"],
    "all_labels": ["所有标签", "All labels"],
    "knowledge_graph_nodes": ["知识图谱节点数", "Knowledge graph nodes"],
    "knowledge_graph_edges": ["知识图谱边数", "Knowledge graph edges"],
    "query_after_deletion": ["删除后查询", "Query after deletion"],
    "re_insert_for_test": ["重新插入用于后续测试", "Re-insert for subsequent testing"],
    # Assertion messages using %-style formatting
    "edge_degree_should_be": [
        "边 %s -> %s 的度数应为%d，实际为 %d",
        "Edge %s -> %s degree should be %d, actual %d",
    ],
    "forward_reverse_edge_inconsistent": [
        "正向和反向边属性不一致",
        "Forward and reverse edge properties are inconsistent",
    ],
    "node_edge_should_contain_connection": [
        "节点 %s 的边列表中应包含与 %s 的连接",
        "Node %s edge list should contain connection with %s",
    ],
    "node_should_be_deleted": [
        "节点 %s 应被删除",
        "Node %s should be deleted",
    ],
    "edge_should_be_deleted": [
        "边 %s -> %s 应被删除",
        "Edge %s -> %s should be deleted",
    ],
    "reverse_edge_should_be_deleted": [
        "反向边 %s -> %s 应被删除",
        "Reverse edge %s -> %s should be deleted",
    ],
    "should_have_labels": [
        "应有%d个标签，实际有 %d",
        "Should have %d labels, actual %d",
    ],
    "should_be_in_label_list": [
        "%s 应在标签列表中",
        "%s should be in label list",
    ],
    "result_should_be_kg_type": [
        "返回结果应为 KnowledgeGraph 类型",
        "Result should be KnowledgeGraph type",
    ],
    "kg_should_have_nodes": [
        "知识图谱应有%d个节点，实际有 %d",
        "Knowledge graph should have %d nodes, actual %d",
    ],
    "kg_should_have_edges": [
        "知识图谱应有%d条边，实际有 %d",
        "Knowledge graph should have %d edges, actual %d",
    ],
    "node_degree_display": [
        "节点 %s 的度数: %d",
        "Node %s degree: %d",
    ],
    "edge_degree_display": [
        "边 %s -> %s 的度数: %d",
        "Edge %s -> %s degree: %d",
    ],
    "undirected_node_edges_success": [
        "无向图特性验证成功：节点 %s 的边列表包含所有相关的边",
        "Undirected graph property verification successful: node %s edge list contains all related edges",
    ],
    "query_after_deletion_display": [
        "删除后查询节点属性 %s: %s",
        "Query after deletion %s: %s",
    ],
    "query_edge_after_deletion_display": [
        "删除后查询边属性 %s -> %s: %s",
        "Query after deletion %s -> %s: %s",
    ],
    # Test completion
    "advanced_test_complete": [
        "\n高级测试完成",
        "\nAdvanced test completed",
    ],
    "starting_advanced_test": [
        "\n=== 开始高级测试 ===",
        "\n=== Starting Advanced Test ===",
    ],
    "test_error": [
        "测试错误",
        "Test error",
    ],
    "query_reverse_edge_after_deletion": [
        "删除后查询反向边属性 %s -> %s: %s",
        "Query after deletion reverse edge %s -> %s: %s",
    ],
}
