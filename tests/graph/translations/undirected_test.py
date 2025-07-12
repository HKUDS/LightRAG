"""
Translations specific to undirected graph property testing
"""

UNDIRECTED_TEST_TRANSLATIONS = {
    # Node descriptions
    "cs_desc": [
        "计算机科学是研究计算理论、算法设计和计算机系统设计的学科",
        "Computer Science is the study of computation theory, algorithm design, and computer system design",
    ],
    "ds_desc": [
        "数据结构是计算机中存储、组织数据的方式",
        "Data structure is the way to store and organize data in computers",
    ],
    "algo_desc": [
        "算法是解决问题的一系列有序的计算步骤",
        "Algorithm is a series of ordered computational steps to solve problems",
    ],
    # Node keywords
    "cs_keywords": [
        "计算机,科学,编程",
        "computer,science,programming",
    ],
    "ds_keywords": [
        "数据,结构,存储",
        "data,structure,storage",
    ],
    "algo_keywords": [
        "算法,计算,步骤",
        "algorithm,computation,steps",
    ],
    # Entity types
    "subject": ["学科", "Subject"],
    "concept": ["概念", "Concept"],
    # Edge descriptions
    "cs_contains_ds": [
        "计算机科学领域包含数据结构概念",
        "Computer Science field contains Data Structure concepts",
    ],
    "cs_contains_algo": [
        "计算机科学领域包含算法概念",
        "Computer Science field contains Algorithm concepts",
    ],
    # Test operations
    "test_insert_edge_undirected_property": [
        "测试插入边的无向图特性",
        "Test insert edge undirected property",
    ],
    "test_edge_degree_undirected_property": [
        "测试边度数的无向图特性",
        "Test edge degree undirected property",
    ],
    "test_delete_edge_undirected_property": [
        "测试删除边的无向图特性",
        "Test delete edge undirected property",
    ],
    "test_batch_undirected_property": [
        "测试批量操作的无向图特性",
        "Test batch operations undirected property",
    ],
    "test_batch_get_node_edges_undirected_property": [
        "测试批量获取节点边的无向图特性",
        "Test batch get node edges undirected property",
    ],
    # Edge operations
    "forward_edge_props": [
        "正向边属性",
        "Forward edge properties",
    ],
    "reverse_edge_props": [
        "反向边属性",
        "Reverse edge properties",
    ],
    "forward_edge_degree": [
        "正向边度数",
        "Forward edge degree",
    ],
    "reverse_edge_degree": [
        "反向边度数",
        "Reverse edge degree",
    ],
    "delete_edge": [
        "删除边",
        "Delete edge",
    ],
    "query_forward_edge_after_delete": [
        "删除后查询正向边",
        "Query forward edge after delete",
    ],
    "query_reverse_edge_after_delete": [
        "删除后查询反向边",
        "Query reverse edge after delete",
    ],
    # Error messages
    "unable_read_edge": [
        "无法读取边",
        "Unable to read edge",
    ],
    "unable_read_reverse_edge": [
        "无法读取反向边",
        "Unable to read reverse edge",
    ],
    "forward_reverse_inconsistent": [
        "正向和反向边不一致",
        "Forward and reverse edges are inconsistent",
    ],
    "reverse_edge_should_be_deleted": [
        "反向边应该被删除",
        "Reverse edge should be deleted",
    ],
    "reverse_edge_should_be_in_result": [
        "反向边应该在结果中",
        "Reverse edge should be in result",
    ],
    "node_edge_should_contain": [
        "节点边应该包含",
        "Node edge should contain",
    ],
    "node_edge_should_contain_connection": [
        "节点边应该包含连接",
        "Node edge should contain connection",
    ],
    # Verification messages
    "undirected_edge_degree_verification_success": [
        "无向图边度数验证成功",
        "Undirected graph edge degree verification successful",
    ],
    "undirected_delete_verification_success": [
        "无向图删除验证成功",
        "Undirected graph deletion verification successful",
    ],
    "undirected_batch_verification_success": [
        "无向图批量操作验证成功",
        "Undirected graph batch operations verification successful",
    ],
    # Test completion
    "undirected_test_complete": [
        "无向图特性测试完成",
        "Undirected graph property test completed",
    ],
    "starting_undirected_graph_test": [
        "\n=== 开始无向图特性测试 ===",
        "\n=== Starting Undirected Graph Property Test ===",
    ],
}
