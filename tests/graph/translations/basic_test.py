"""
Translations specific to basic graph testing
"""

BASIC_TEST_TRANSLATIONS = {
    # Basic test specific messages
    "success_read_node": ["成功读取节点属性", "Successfully read node properties"],
    "success_read_edge": ["成功读取边属性", "Successfully read edge properties"],
    "success_read_reverse": [
        "成功读取反向边属性",
        "Successfully read reverse edge properties",
    ],
    "failed_read_node": ["读取节点属性失败", "Failed to read node properties"],
    "failed_read_edge": ["读取边属性失败", "Failed to read edge properties"],
    "failed_read_reverse_edge": [
        "读取反向边属性失败",
        "Failed to read reverse edge properties",
    ],
    "unable_read_node": ["未能读取节点属性", "Unable to read node properties"],
    "unable_read_edge": ["未能读取边属性", "Unable to read edge properties"],
    "unable_read_reverse_edge": [
        "未能读取反向边属性",
        "Unable to read reverse edge properties",
    ],
    # Property descriptions
    "node_desc": ["节点描述", "Node description"],
    "node_type": ["节点类型", "Node type"],
    "node_keywords": ["节点关键词", "Node keywords"],
    "edge_relation": ["边关系", "Edge relationship"],
    "edge_desc": ["边描述", "Edge description"],
    "edge_weight": ["边权重", "Edge weight"],
    "reverse_edge_relation": ["反向边关系", "Reverse edge relationship"],
    "reverse_edge_desc": ["反向边描述", "Reverse edge description"],
    "reverse_edge_weight": ["反向边权重", "Reverse edge weight"],
    # Validation messages
    "node_id_mismatch": ["节点ID不匹配: 期望", "Node ID mismatch: expected"],
    "node_desc_mismatch": ["节点描述不匹配", "Node description mismatch"],
    "node_type_mismatch": ["节点类型不匹配", "Node type mismatch"],
    "edge_relation_mismatch": ["边关系不匹配", "Edge relationship mismatch"],
    "edge_desc_mismatch": ["边描述不匹配", "Edge description mismatch"],
    "edge_weight_mismatch": ["边权重不匹配", "Edge weight mismatch"],
    # Undirected graph verification
    "forward_reverse_inconsistent": [
        "正向和反向边属性不一致，无向图特性验证失败",
        "Forward and reverse edge properties inconsistent, undirected graph property verification failed",
    ],
    "undirected_verification_failed": [
        "无向图特性验证失败",
        "undirected graph property verification failed",
    ],
    # Test completion
    "basic_test_complete": [
        "\n基本测试完成，数据已保留在数据库中",
        "\nBasic test completed, data retained in database",
    ],
    "starting_basic_test": [
        "\n=== 开始基本测试 ===",
        "\n=== Starting Basic Test ===",
    ],
}
