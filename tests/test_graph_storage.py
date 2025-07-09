#!/usr/bin/env python
"""
Universal Graph Storage Test Program / 通用图存储测试程序

This program selects the graph storage type to use based on the LIGHTRAG_GRAPH_STORAGE configuration in .env,
and tests its basic and advanced operations.

该程序根据.env中的LIGHTRAG_GRAPH_STORAGE配置选择使用的图存储类型，
并对其进行基本操作和高级操作的测试。

Language Support / 语言支持:
Set the TEST_LANGUAGE environment variable to 'english' or 'chinese' to change the language.
Default is 'chinese' for backwards compatibility.

设置 TEST_LANGUAGE 环境变量为 'english' 或 'chinese' 来更改语言。
默认为 'chinese' 以保持向后兼容性。

Example usage / 使用示例:
    # Run in English / 英文运行
    TEST_LANGUAGE=english python test_graph_storage.py

    # Run in Chinese (default) / 中文运行（默认）
    python test_graph_storage.py

    # Or set in .env file / 或在 .env 文件中设置
    TEST_LANGUAGE=english

Supported graph storage types include / 支持的图存储类型包括：
- NetworkXStorage
- Neo4JStorage
- MongoDBStorage
- PGGraphStorage
- MemgraphStorage
"""

import asyncio
import os
import sys
import importlib
import numpy as np
import tempfile
import shutil
import argparse
from dotenv import load_dotenv
from ascii_colors import ASCIIColors

# Load environment variables first to get language setting / 首先加载环境变量以获取语言设置
load_dotenv(dotenv_path=".env", override=False)

# Language configuration / 语言配置
# Default to chinese for backwards compatibility / 默认为中文以保持向后兼容性
_DEFAULT_LANGUAGE = os.getenv("TEST_LANGUAGE", "chinese").lower()
if _DEFAULT_LANGUAGE not in ["english", "chinese"]:
    _DEFAULT_LANGUAGE = "chinese"

# Global language setting that can be modified / 可以修改的全局语言设置
LANGUAGE = _DEFAULT_LANGUAGE

# Translation dictionaries / 翻译字典
TRANSLATIONS = {
    "english": {
        # Messages / 消息
        "warning_no_env": "Warning: No .env file found in the current directory, which may affect storage configuration loading.",
        "continue_execution": "Continue execution? (yes/no): ",
        "test_cancelled": "Test program cancelled",
        "error": "Error",
        "warning": "Warning",
        "current_graph_storage": "Current configured graph storage type",
        "supported_graph_storage": "Supported graph storage types",
        "init_storage_failed": "Failed to initialize storage instance, test program exiting",
        "select_test_type": "Please select test type:",
        "basic_test": "1. Basic test (node and edge insertion, reading)",
        "advanced_test": "2. Advanced test (degree, labels, knowledge graph, delete operations, etc.)",
        "batch_test": "3. Batch operation test (batch get node, edge attributes and degrees, etc.)",
        "undirected_test": "4. Undirected graph property test (verify undirected graph properties of storage)",
        "special_char_test": "5. Special character test (verify single quotes, double quotes and backslashes, etc.)",
        "all_tests": "6. All tests",
        "select_option": "Please enter option (1/2/3/4/5/6): ",
        "cleaning_data": "Cleaning data before executing tests...",
        "data_cleaned": "Data cleaning completed",
        "invalid_option": "Invalid option",
        "connection_closed": "Storage connection closed",
        "kuzu_temp_cleaned": "KuzuDB temporary directory cleaned",
        # Additional test messages / 额外的测试消息
        "test_completed": "Test completed",
        "batch_test_complete": "Batch operations test completed",
        "undirected_test_complete": "Undirected graph property test completed",
        "all_tests_completed": "All tests completed successfully",
        "no_description": "No description",
        "no_type": "No type",
        "no_keywords": "No keywords",
        "no_relationship": "No relationship",
        "no_weight": "No weight",
        # Test progress messages / 测试进度消息
        "test_node_degree": "Test node_degree",
        "test_all_node_degrees": "Test all node degrees",
        "test_edge_degree": "Test edge_degree",
        "test_reverse_edge_degree": "Test reverse edge degree",
        "test_get_node_edges": "Test get_node_edges",
        "test_get_all_labels": "Test get_all_labels",
        "test_get_knowledge_graph": "Test get_knowledge_graph",
        "test_delete_node": "Test delete_node",
        "test_remove_edges": "Test remove_edges",
        "test_remove_nodes": "Test remove_nodes",
        "verify_undirected_property": "Verify undirected graph property",
        "verify_node_edges_undirected": "Verify node edges undirected property",
        "node_degree": "Node degree",
        "edge_degree": "Edge degree",
        "reverse_edge_degree": "Reverse edge degree",
        "all_edges": "All edges",
        "all_labels": "All labels",
        "knowledge_graph_nodes": "Knowledge graph nodes",
        "knowledge_graph_edges": "Knowledge graph edges",
        "query_after_deletion": "Query after deletion",
        "re_insert_for_test": "Re-insert for subsequent testing",
        "advanced_test_complete": "Advanced test completed",
        # Additional missing translations / 额外的缺失翻译
        "failed_read_node": "Failed to read node properties",
        "failed_read_edge": "Failed to read edge properties",
        "failed_read_reverse_edge": "Failed to read reverse edge properties",
        "unable_read_node": "Unable to read node properties",
        "unable_read_edge": "Unable to read edge properties",
        "unable_read_reverse_edge": "Unable to read reverse edge properties",
        "node_id_mismatch": "Node ID mismatch: expected",
        "actual": "actual",
        "node_desc_mismatch": "Node description mismatch",
        "node_type_mismatch": "Node type mismatch",
        "edge_relation_mismatch": "Edge relationship mismatch",
        "edge_desc_mismatch": "Edge description mismatch",
        "edge_weight_mismatch": "Edge weight mismatch",
        "forward_reverse_inconsistent": "Forward and reverse edge properties inconsistent, undirected graph property verification failed",
        "undirected_verification_failed": "undirected graph property verification failed",
        # Test data / 测试数据
        "artificial_intelligence": "Artificial Intelligence",
        "machine_learning": "Machine Learning",
        "deep_learning": "Deep Learning",
        "natural_language_processing": "Natural Language Processing",
        "computer_vision": "Computer Vision",
        "computer_science": "Computer Science",
        "data_structure": "Data Structure",
        "algorithm": "Algorithm",
        # Descriptions / 描述
        "ai_desc": "Artificial Intelligence is a branch of computer science that attempts to understand the essence of intelligence and produce a new kind of intelligent machine that can respond in a way similar to human intelligence.",
        "ml_desc": "Machine Learning is a branch of artificial intelligence that uses statistical methods to enable computer systems to learn without being explicitly programmed.",
        "dl_desc": "Deep Learning is a branch of machine learning that uses multi-layer neural networks to simulate the human brain's learning process.",
        "nlp_desc": "Natural Language Processing is a branch of artificial intelligence that focuses on enabling computers to understand and process human language.",
        "cv_desc": "Computer Vision is a branch of artificial intelligence that focuses on enabling computers to obtain information from images or videos.",
        "cs_desc": "Computer Science is the science that studies computers and their applications.",
        "ds_desc": "Data Structure is a fundamental concept in computer science, used to organize and store data.",
        "algo_desc": "Algorithm is the steps and methods for solving problems.",
        # Keywords / 关键词
        "ai_keywords": "AI,machine learning,deep learning",
        "ml_keywords": "supervised learning,unsupervised learning,reinforcement learning",
        "dl_keywords": "neural networks,CNN,RNN",
        "nlp_keywords": "NLP,text analysis,language models",
        "cv_keywords": "CV,image recognition,object detection",
        "cs_keywords": "computer,science,technology",
        "ds_keywords": "data,structure,organization",
        "algo_keywords": "algorithm,steps,methods",
        # Entity types / 实体类型
        "tech_field": "Technology Field",
        "concept": "Concept",
        "subject": "Subject",
        "test_node": "Test Node",
        # Relationships / 关系
        "contains": "contains",
        "applied_to": "applied to",
        "special_relation": "special 'relation'",
        "complex_relation": 'complex "relation" \\type',
        # Relationship descriptions / 关系描述
        "ai_contains_ml": "The field of artificial intelligence contains machine learning as a subfield",
        "ml_contains_dl": "The field of machine learning contains deep learning as a subfield",
        "ai_contains_nlp": "The field of artificial intelligence contains natural language processing as a subfield",
        "ai_contains_cv": "The field of artificial intelligence contains computer vision as a subfield",
        "dl_applied_nlp": "Deep learning technology is applied to the field of natural language processing",
        "dl_applied_cv": "Deep learning technology is applied to the field of computer vision",
        "cs_contains_ds": "Computer science contains data structures as a concept",
        "cs_contains_algo": "Computer science contains algorithms as a concept",
        # Test messages / 测试消息
        "insert_node": "Insert node",
        "insert_edge": "Insert edge",
        "read_node_props": "Read node properties",
        "read_edge_props": "Read edge properties",
        "read_reverse_edge": "Read reverse edge properties",
        "success_read_node": "Successfully read node properties",
        "success_read_edge": "Successfully read edge properties",
        "success_read_reverse": "Successfully read reverse edge properties",
        "node_desc": "Node description",
        "node_type": "Node type",
        "node_keywords": "Node keywords",
        "edge_relation": "Edge relationship",
        "edge_desc": "Edge description",
        "edge_weight": "Edge weight",
        "reverse_edge_relation": "Reverse edge relationship",
        "reverse_edge_desc": "Reverse edge description",
        "reverse_edge_weight": "Reverse edge weight",
        "undirected_verification_success": "Undirected graph property verification successful: forward and reverse edge properties are consistent",
        "basic_test_complete": "Basic test completed, data retained in database",
        "test_error": "Error occurred during testing",
        # Special characters test / 特殊字符测试
        "node_with_quotes": "Node with 'single quotes'",
        "node_with_double_quotes": 'Node with "double quotes"',
        "node_with_backslash": "Node with \\backslash\\",
        "desc_with_special": "This description contains 'single quotes', \"double quotes\" and \\backslash",
        "desc_with_complex": "This description contains 'single quotes' and \"double quotes\" as well as \\backslash\\path",
        "desc_with_windows_path": "This description contains Windows path C:\\Program Files\\ and escape characters \\n\\t",
        "keywords_special": "special characters,quotes,escape",
        "keywords_backslash": "backslash,path,escape",
        "keywords_json": "special characters,quotes,JSON",
        "edge_desc_special": "This edge description contains 'single quotes', \"double quotes\" and \\backslash",
        "edge_desc_sql": "Contains SQL injection attempt: SELECT * FROM users WHERE name='admin'--",
        "verify_node_special": "Verify node special characters",
        "verify_edge_special": "Verify edge special characters",
        "special_char_verification_success": "Special character verification successful",
        "special_char_test_complete": "Special character test completed, data retained in database",
        # Additional batch test translations
        "batch_get_nodes": "Test get_nodes_batch",
        "batch_get_nodes_result": "Batch get node properties result",
        "batch_node_degrees": "Test node_degrees_batch",
        "batch_node_degrees_result": "Batch get node degrees result",
        "batch_edge_degrees": "Test edge_degrees_batch",
        "batch_edge_degrees_result": "Batch get edge degrees result",
        "batch_get_edges": "Test get_edges_batch",
        "batch_get_edges_result": "Batch get edge properties result",
        "test_reverse_edges_batch": "Test reverse edges batch get",
        "insert_node_1": "Insert node 1",
        "insert_node_2": "Insert node 2",
        "insert_node_3": "Insert node 3",
        "insert_node_4": "Insert node 4",
        "insert_node_5": "Insert node 5",
        "insert_edge_1": "Insert edge 1",
        "insert_edge_2": "Insert edge 2",
        "insert_edge_3": "Insert edge 3",
        "insert_edge_4": "Insert edge 4",
        "insert_edge_5": "Insert edge 5",
        "insert_edge_6": "Insert edge 6",
        # Additional description translations
        "ai_contains_cv_desc": "The field of artificial intelligence contains computer vision as a subfield",
        "dl_applied_nlp_relationship": "applied to",
        "dl_applied_nlp_desc": "Deep learning technology is applied to the field of natural language processing",
        "dl_applied_cv_relationship": "applied to",
        "dl_applied_cv_desc": "Deep learning technology is applied to the field of computer vision",
        # Additional batch operation translations
        "batch_get_reverse_edges_result": "Batch get reverse edge properties result",
        "undirected_batch_verification_success": "Undirected graph property verification successful: batch obtained forward and reverse edge properties are consistent",
        "test_get_nodes_edges_batch": "Test get_nodes_edges_batch",
        "batch_get_nodes_edges_result": "Batch get node edges result",
        "verify_batch_nodes_edges_undirected": "Verify batch get node edges undirected graph property",
        "node_outgoing_edges": "Node outgoing edges",
        "node_incoming_edges": "Node incoming edges",
        "undirected_nodes_edges_verification_success": "Undirected graph property verification successful: batch obtained node edges contain all related edges (regardless of direction)",
        "test_get_nodes_by_chunk_ids": "Test get_nodes_by_chunk_ids",
        "test_single_chunk_id_multiple_nodes": "Test single chunk_id, matching multiple nodes",
        "test_multiple_chunk_ids_partial_match": "Test multiple chunk_ids, partial matching multiple nodes",
        "test_get_edges_by_chunk_ids": "Test get_edges_by_chunk_ids",
        "test_single_chunk_id_multiple_edges": "Test single chunk_id, matching multiple edges",
        "test_multiple_chunk_ids_partial_edges": "Test multiple chunk_ids, partial matching multiple edges",
        "batch_operations_test_complete": "Batch operations test completed",
    },
    "chinese": {
        # Keep all existing Chinese text as fallback
        "warning_no_env": "警告: 当前目录中没有找到.env文件，这可能会影响存储配置的加载。",
        "continue_execution": "是否继续执行? (yes/no): ",
        "test_cancelled": "测试程序已取消",
        "error": "错误",
        "warning": "警告",
        "current_graph_storage": "当前配置的图存储类型",
        "supported_graph_storage": "支持的图存储类型",
        "init_storage_failed": "初始化存储实例失败，测试程序退出",
        "select_test_type": "请选择测试类型:",
        "basic_test": "1. 基本测试 (节点和边的插入、读取)",
        "advanced_test": "2. 高级测试 (度数、标签、知识图谱、删除操作等)",
        "batch_test": "3. 批量操作测试 (批量获取节点、边属性和度数等)",
        "undirected_test": "4. 无向图特性测试 (验证存储的无向图特性)",
        "special_char_test": "5. 特殊字符测试 (验证单引号、双引号和反斜杠等特殊字符)",
        "all_tests": "6. 全部测试",
        "select_option": "请输入选项 (1/2/3/4/5/6): ",
        "cleaning_data": "执行测试前清理数据...",
        "data_cleaned": "数据清理完成",
        "invalid_option": "无效的选项",
        "connection_closed": "存储连接已关闭",
        "kuzu_temp_cleaned": "KuzuDB临时目录已清理",
        # Additional test messages - Chinese / 额外的测试消息 - 中文
        "test_completed": "测试完成",
        "batch_test_complete": "批量操作测试完成",
        "undirected_test_complete": "无向图特性测试完成",
        "all_tests_completed": "所有测试成功完成",
        "no_description": "无描述",
        "no_type": "无类型",
        "no_keywords": "无关键词",
        "no_relationship": "无关系",
        "no_weight": "无权重",
        # Test progress messages - Chinese / 测试进度消息 - 中文
        "test_node_degree": "测试 node_degree",
        "test_all_node_degrees": "测试所有节点的度数",
        "test_edge_degree": "测试 edge_degree",
        "test_reverse_edge_degree": "测试反向边的度数",
        "test_get_node_edges": "测试 get_node_edges",
        "test_get_all_labels": "测试 get_all_labels",
        "test_get_knowledge_graph": "测试 get_knowledge_graph",
        "test_delete_node": "测试 delete_node",
        "test_remove_edges": "测试 remove_edges",
        "test_remove_nodes": "测试 remove_nodes",
        "verify_undirected_property": "验证无向图特性",
        "verify_node_edges_undirected": "验证节点边的无向图特性",
        "node_degree": "节点度数",
        "edge_degree": "边度数",
        "reverse_edge_degree": "反向边度数",
        "all_edges": "所有边",
        "all_labels": "所有标签",
        "knowledge_graph_nodes": "知识图谱节点数",
        "knowledge_graph_edges": "知识图谱边数",
        "query_after_deletion": "删除后查询",
        "re_insert_for_test": "重新插入用于后续测试",
        "advanced_test_complete": "高级测试完成",
        # Additional missing translations - Chinese / 额外的缺失翻译 - 中文
        "failed_read_node": "读取节点属性失败",
        "failed_read_edge": "读取边属性失败",
        "failed_read_reverse_edge": "读取反向边属性失败",
        "unable_read_node": "未能读取节点属性",
        "unable_read_edge": "未能读取边属性",
        "unable_read_reverse_edge": "未能读取反向边属性",
        "node_id_mismatch": "节点ID不匹配: 期望",
        "actual": "实际",
        "node_desc_mismatch": "节点描述不匹配",
        "node_type_mismatch": "节点类型不匹配",
        "edge_relation_mismatch": "边关系不匹配",
        "edge_desc_mismatch": "边描述不匹配",
        "edge_weight_mismatch": "边权重不匹配",
        "forward_reverse_inconsistent": "正向和反向边属性不一致，无向图特性验证失败",
        "undirected_verification_failed": "无向图特性验证失败",
        # Test data - Chinese
        "artificial_intelligence": "人工智能",
        "machine_learning": "机器学习",
        "deep_learning": "深度学习",
        "natural_language_processing": "自然语言处理",
        "computer_vision": "计算机视觉",
        "computer_science": "计算机科学",
        "data_structure": "数据结构",
        "algorithm": "算法",
        # Descriptions - Chinese
        "ai_desc": "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "ml_desc": "机器学习是人工智能的一个分支，它使用统计学方法让计算机系统在不被明确编程的情况下也能够学习。",
        "dl_desc": "深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。",
        "nlp_desc": "自然语言处理是人工智能的一个分支，专注于使计算机理解和处理人类语言。",
        "cv_desc": "计算机视觉是人工智能的一个分支，专注于使计算机能够从图像或视频中获取信息。",
        "cs_desc": "计算机科学是研究计算机及其应用的科学。",
        "ds_desc": "数据结构是计算机科学中的一个基础概念，用于组织和存储数据。",
        "algo_desc": "算法是解决问题的步骤和方法。",
        # Keywords - Chinese
        "ai_keywords": "AI,机器学习,深度学习",
        "ml_keywords": "监督学习,无监督学习,强化学习",
        "dl_keywords": "神经网络,CNN,RNN",
        "nlp_keywords": "NLP,文本分析,语言模型",
        "cv_keywords": "CV,图像识别,目标检测",
        "cs_keywords": "计算机,科学,技术",
        "ds_keywords": "数据,结构,组织",
        "algo_keywords": "算法,步骤,方法",
        # Entity types - Chinese
        "tech_field": "技术领域",
        "concept": "概念",
        "subject": "学科",
        "test_node": "测试节点",
        # Relationships - Chinese
        "contains": "包含",
        "applied_to": "应用于",
        "special_relation": "特殊'关系'",
        "complex_relation": '复杂"关系"\\类型',
        # Relationship descriptions - Chinese
        "ai_contains_ml": "人工智能领域包含机器学习这个子领域",
        "ml_contains_dl": "机器学习领域包含深度学习这个子领域",
        "ai_contains_nlp": "人工智能领域包含自然语言处理这个子领域",
        "ai_contains_cv": "人工智能领域包含计算机视觉这个子领域",
        "dl_applied_nlp": "深度学习技术应用于自然语言处理领域",
        "dl_applied_cv": "深度学习技术应用于计算机视觉领域",
        "cs_contains_ds": "计算机科学包含数据结构这个概念",
        "cs_contains_algo": "计算机科学包含算法这个概念",
        # Test messages - Chinese
        "insert_node": "插入节点",
        "insert_edge": "插入边",
        "read_node_props": "读取节点属性",
        "read_edge_props": "读取边属性",
        "read_reverse_edge": "读取反向边属性",
        "success_read_node": "成功读取节点属性",
        "success_read_edge": "成功读取边属性",
        "success_read_reverse": "成功读取反向边属性",
        "node_desc": "节点描述",
        "node_type": "节点类型",
        "node_keywords": "节点关键词",
        "edge_relation": "边关系",
        "edge_desc": "边描述",
        "edge_weight": "边权重",
        "reverse_edge_relation": "反向边关系",
        "reverse_edge_desc": "反向边描述",
        "reverse_edge_weight": "反向边权重",
        "undirected_verification_success": "无向图特性验证成功：正向和反向边属性一致",
        "basic_test_complete": "基本测试完成，数据已保留在数据库中",
        "test_error": "测试过程中发生错误",
        # Special characters test - Chinese
        "node_with_quotes": "包含'单引号'的节点",
        "node_with_double_quotes": '包含"双引号"的节点',
        "node_with_backslash": "包含\\反斜杠\\的节点",
        "desc_with_special": "这个描述包含'单引号'、\"双引号\"和\\反斜杠",
        "desc_with_complex": "这个描述同时包含'单引号'和\"双引号\"以及\\反斜杠\\路径",
        "desc_with_windows_path": "这个描述包含Windows路径C:\\Program Files\\和转义字符\\n\\t",
        "keywords_special": "特殊字符,引号,转义",
        "keywords_backslash": "反斜杠,路径,转义",
        "keywords_json": "特殊字符,引号,JSON",
        "edge_desc_special": "这个边描述包含'单引号'、\"双引号\"和\\反斜杠",
        "edge_desc_sql": "包含SQL注入尝试: SELECT * FROM users WHERE name='admin'--",
        "verify_node_special": "验证节点特殊字符",
        "verify_edge_special": "验证边特殊字符",
        "special_char_verification_success": "特殊字符验证成功",
        "special_char_test_complete": "特殊字符测试完成，数据已保留在数据库中",
        # Additional batch test translations
        "batch_get_nodes": "== 测试 get_nodes_batch",
        "batch_get_nodes_result": "批量获取节点属性结果",
        "batch_node_degrees": "== 测试 node_degrees_batch",
        "batch_node_degrees_result": "批量获取节点度数结果",
        "batch_edge_degrees": "== 测试 edge_degrees_batch",
        "batch_edge_degrees_result": "批量获取边度数结果",
        "batch_get_edges": "== 测试 get_edges_batch",
        "batch_get_edges_result": "批量获取边属性结果",
        "test_reverse_edges_batch": "== 测试反向边的批量获取",
        "insert_node_1": "插入节点1",
        "insert_node_2": "插入节点2",
        "insert_node_3": "插入节点3",
        "insert_node_4": "插入节点4",
        "insert_node_5": "插入节点5",
        "insert_edge_1": "插入边1",
        "insert_edge_2": "插入边2",
        "insert_edge_3": "插入边3",
        "insert_edge_4": "插入边4",
        "insert_edge_5": "插入边5",
        "insert_edge_6": "插入边6",
        # Additional description translations
        "ai_contains_cv_desc": "人工智能领域包含计算机视觉这个子领域",
        "dl_applied_nlp_relationship": "应用于",
        "dl_applied_nlp_desc": "深度学习技术应用于自然语言处理领域",
        "dl_applied_cv_relationship": "应用于",
        "dl_applied_cv_desc": "深度学习技术应用于计算机视觉领域",
        # Additional batch operation translations
        "batch_get_reverse_edges_result": "批量获取反向边属性结果",
        "undirected_batch_verification_success": "无向图特性验证成功：批量获取的正向和反向边属性一致",
        "test_get_nodes_edges_batch": "== 测试 get_nodes_edges_batch",
        "batch_get_nodes_edges_result": "批量获取节点边结果",
        "verify_batch_nodes_edges_undirected": "== 验证批量获取节点边的无向图特性",
        "node_outgoing_edges": "的出边",
        "node_incoming_edges": "的入边",
        "undirected_nodes_edges_verification_success": "无向图特性验证成功：批量获取的节点边包含所有相关的边（无论方向）",
        "test_get_nodes_by_chunk_ids": "== 测试 get_nodes_by_chunk_ids",
        "test_single_chunk_id_multiple_nodes": "== 测试单个 chunk_id，匹配多个节点",
        "test_multiple_chunk_ids_partial_match": "== 测试多个 chunk_id，部分匹配多个节点",
        "test_get_edges_by_chunk_ids": "== 测试 get_edges_by_chunk_ids",
        "test_single_chunk_id_multiple_edges": "== 测试单个 chunk_id，匹配多条边",
        "test_multiple_chunk_ids_partial_edges": "== 测试多个 chunk_id，部分匹配多条边",
        "batch_operations_test_complete": "\n批量操作测试完成",
    },
}


def t(key):
    """Translation function / 翻译函数"""
    return TRANSLATIONS[LANGUAGE].get(key, key) or key


# Add to project root directory to Python path / 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.types import KnowledgeGraph
from lightrag.kg import (
    STORAGE_IMPLEMENTATIONS,
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
    verify_storage_implementation,
)
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.constants import GRAPH_FIELD_SEP


# 模拟的嵌入函数，返回随机向量
async def mock_embedding_func(texts):
    return np.random.rand(len(texts), 10)  # 返回10维随机向量


def check_env_file():
    """
    检查.env文件是否存在，如果不存在则发出警告
    返回True表示应该继续执行，False表示应该退出
    """
    if not os.path.exists(".env"):
        warning_msg = t("warning_no_env")
        ASCIIColors.yellow(warning_msg)

        # 检查是否在交互式终端中运行
        if sys.stdin.isatty():
            response = input(t("continue_execution"))
            if response.lower() != "yes":
                ASCIIColors.red(t("test_cancelled"))
                return False
    return True


def setup_kuzu_environment():
    """
    设置KuzuDB测试环境
    """
    # 创建临时目录用于KuzuDB测试
    test_dir = tempfile.mkdtemp(prefix="kuzu_test_")
    kuzu_db_path = os.path.join(test_dir, "test_kuzu.db")

    # 设置环境变量
    os.environ["KUZU_DB_PATH"] = kuzu_db_path
    os.environ["KUZU_WORKSPACE"] = "test_workspace"

    return test_dir, kuzu_db_path


def cleanup_kuzu_environment(test_dir):
    """
    清理KuzuDB测试环境
    """
    try:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
    except Exception as e:
        ASCIIColors.yellow(f"警告: 清理临时目录失败: {str(e)}")


async def initialize_graph_storage():
    """
    根据环境变量初始化相应的图存储实例
    返回初始化的存储实例
    """
    # 从环境变量中获取图存储类型
    graph_storage_type = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")

    # 验证存储类型是否有效
    try:
        verify_storage_implementation("GRAPH_STORAGE", graph_storage_type)
    except ValueError as e:
        ASCIIColors.red(f"错误: {str(e)}")
        ASCIIColors.yellow(
            f"支持的图存储类型: {', '.join(STORAGE_IMPLEMENTATIONS['GRAPH_STORAGE']['implementations'])}"
        )
        return None

    # 检查所需的环境变量
    required_env_vars = STORAGE_ENV_REQUIREMENTS.get(graph_storage_type, [])
    missing_env_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_env_vars:
        ASCIIColors.red(
            f"错误: {graph_storage_type} 需要以下环境变量，但未设置: {', '.join(missing_env_vars)}"
        )
        return None

    # KuzuDB特殊处理：自动设置测试环境
    temp_dir = None
    if graph_storage_type == "KuzuDBStorage":
        temp_dir, kuzu_db_path = setup_kuzu_environment()
        ASCIIColors.cyan(f"KuzuDB测试环境已设置: {kuzu_db_path}")

    # 动态导入相应的模块
    module_path = STORAGES.get(graph_storage_type)
    if not module_path:
        ASCIIColors.red(f"错误: 未找到 {graph_storage_type} 的模块路径")
        if temp_dir:
            cleanup_kuzu_environment(temp_dir)
        return None

    try:
        module = importlib.import_module(module_path, package="lightrag")
        storage_class = getattr(module, graph_storage_type)
    except (ImportError, AttributeError) as e:
        ASCIIColors.red(f"错误: 导入 {graph_storage_type} 失败: {str(e)}")
        if temp_dir:
            cleanup_kuzu_environment(temp_dir)
        return None

    # 初始化存储实例
    global_config = {
        "embedding_batch_num": 10,  # 批处理大小
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.5  # 余弦相似度阈值
        },
        "working_dir": os.environ.get("WORKING_DIR", "./rag_storage"),  # 工作目录
        "max_graph_nodes": 1000,  # KuzuDB需要的配置
    }

    # 如果使用 NetworkXStorage，需要先初始化 shared_storage
    if graph_storage_type == "NetworkXStorage":
        initialize_share_data()  # 使用单进程模式

    try:
        # KuzuDB需要特殊的初始化参数
        if graph_storage_type == "KuzuDBStorage":
            storage = storage_class(
                namespace="test_graph",
                global_config=global_config,
                embedding_func=mock_embedding_func,
                workspace="test_workspace",
            )
        else:
            storage = storage_class(
                namespace="test_graph",
                global_config=global_config,
                embedding_func=mock_embedding_func,
            )

        # 初始化连接
        await storage.initialize()

        # 将临时目录信息存储到storage对象中，以便后续清理
        if temp_dir:
            storage._temp_dir = temp_dir

        return storage
    except Exception as e:
        ASCIIColors.red(f"错误: 初始化 {graph_storage_type} 失败: {str(e)}")
        if temp_dir:
            cleanup_kuzu_environment(temp_dir)
        return None


async def test_graph_basic(storage):
    """
    测试图数据库的基本操作:
    1. 使用 upsert_node 插入两个节点
    2. 使用 upsert_edge 插入一条连接两个节点的边
    3. 使用 get_node 读取一个节点
    4. 使用 get_edge 读取一条边
    """
    try:
        # 1. 插入第一个节点
        node1_id = t("artificial_intelligence")
        node1_data = {
            "entity_id": node1_id,
            "description": t("ai_desc"),
            "keywords": t("ai_keywords"),
            "entity_type": t("tech_field"),
        }
        print(f"{t('insert_node')} 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 2. Insert second node / 插入第二个节点
        node2_id = t("machine_learning")
        node2_data = {
            "entity_id": node2_id,
            "description": t("ml_desc"),
            "keywords": t("ml_keywords"),
            "entity_type": t("tech_field"),
        }
        print(f"{t('insert_node')} 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 3. Insert connecting edge / 插入连接边
        edge_data = {
            "relationship": t("contains"),
            "weight": 1.0,
            "description": t("ai_contains_ml"),
        }
        print(f"{t('insert_edge')}: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge_data)

        # 4. Read node properties / 读取节点属性
        print(f"{t('read_node_props')}: {node1_id}")
        node1_props = await storage.get_node(node1_id)
        if node1_props:
            print(f"{t('success_read_node')}: {node1_id}")
            print(
                f"{t('node_desc')}: {node1_props.get('description', t('no_description'))}"
            )
            print(f"{t('node_type')}: {node1_props.get('entity_type', t('no_type'))}")
            print(
                f"{t('node_keywords')}: {node1_props.get('keywords', t('no_keywords'))}"
            )
            # Verify returned properties are correct / 验证返回的属性是否正确
            assert (
                node1_props.get("entity_id") == node1_id
            ), f"{t('node_id_mismatch')} {node1_id}, {t('actual')} {node1_props.get('entity_id')}"
            assert node1_props.get("description") == node1_data["description"], t(
                "node_desc_mismatch"
            )
            assert node1_props.get("entity_type") == node1_data["entity_type"], t(
                "node_type_mismatch"
            )
        else:
            print(f"{t('failed_read_node')}: {node1_id}")
            assert False, f"{t('unable_read_node')}: {node1_id}"

        # 5. Read edge properties / 读取边属性
        print(f"{t('read_edge_props')}: {node1_id} -> {node2_id}")
        edge_props = await storage.get_edge(node1_id, node2_id)
        if edge_props:
            print(f"{t('success_read_edge')}: {node1_id} -> {node2_id}")
            print(
                f"{t('edge_relation')}: {edge_props.get('relationship', t('no_relationship'))}"
            )
            print(
                f"{t('edge_desc')}: {edge_props.get('description', t('no_description'))}"
            )
            print(f"{t('edge_weight')}: {edge_props.get('weight', t('no_weight'))}")
            # Verify returned properties are correct / 验证返回的属性是否正确
            assert edge_props.get("relationship") == edge_data["relationship"], t(
                "edge_relation_mismatch"
            )
            assert edge_props.get("description") == edge_data["description"], t(
                "edge_desc_mismatch"
            )
            assert edge_props.get("weight") == edge_data["weight"], t(
                "edge_weight_mismatch"
            )
        else:
            print(f"{t('failed_read_edge')}: {node1_id} -> {node2_id}")
            assert False, f"{t('unable_read_edge')}: {node1_id} -> {node2_id}"

        # 5.1 Verify undirected graph property - read reverse edge properties / 验证无向图特性 - 读取反向边属性
        print(f"{t('read_reverse_edge')}: {node2_id} -> {node1_id}")
        reverse_edge_props = await storage.get_edge(node2_id, node1_id)
        if reverse_edge_props:
            print(f"{t('success_read_reverse')}: {node2_id} -> {node1_id}")
            print(
                f"{t('reverse_edge_relation')}: {reverse_edge_props.get('relationship', t('no_relationship'))}"
            )
            print(
                f"{t('reverse_edge_desc')}: {reverse_edge_props.get('description', t('no_description'))}"
            )
            print(
                f"{t('reverse_edge_weight')}: {reverse_edge_props.get('weight', t('no_weight'))}"
            )
            # Verify forward and reverse edge properties are the same / 验证正向和反向边属性是否相同
            assert edge_props == reverse_edge_props, t("forward_reverse_inconsistent")
            print(t("undirected_verification_success"))
        else:
            print(f"{t('failed_read_reverse_edge')}: {node2_id} -> {node1_id}")
            assert False, f"{t('unable_read_reverse_edge')}: {node2_id} -> {node1_id}, {t('undirected_verification_failed')}"

        print(t("basic_test_complete"))
        return True

    except Exception as e:
        ASCIIColors.red(f"{t('test_error')}: {str(e)}")
        return False


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
        print(f"== {t('test_node_degree')}: {node1_id}")
        node1_degree = await storage.node_degree(node1_id)
        print(
            f"{t('node_degree')} {node1_id}: {node1_degree}"
            if LANGUAGE == "english"
            else f"节点 {node1_id} 的度数: {node1_degree}"
        )
        assert node1_degree == 1, (
            f"Node {node1_id} degree should be 1, actual {node1_degree}"
            if LANGUAGE == "english"
            else f"节点 {node1_id} 的度数应为1，实际为 {node1_degree}"
        )

        # 2.1 Test all node degrees / 测试所有节点的度数
        print(f"== {t('test_all_node_degrees')}")
        node2_degree = await storage.node_degree(node2_id)
        node3_degree = await storage.node_degree(node3_id)
        print(
            f"{t('node_degree')} {node2_id}: {node2_degree}"
            if LANGUAGE == "english"
            else f"节点 {node2_id} 的度数: {node2_degree}"
        )
        print(
            f"{t('node_degree')} {node3_id}: {node3_degree}"
            if LANGUAGE == "english"
            else f"节点 {node3_id} 的度数: {node3_degree}"
        )
        assert node2_degree == 2, (
            f"Node {node2_id} degree should be 2, actual {node2_degree}"
            if LANGUAGE == "english"
            else f"节点 {node2_id} 的度数应为2，实际为 {node2_degree}"
        )
        assert node3_degree == 1, (
            f"Node {node3_id} degree should be 1, actual {node3_degree}"
            if LANGUAGE == "english"
            else f"节点 {node3_id} 的度数应为1，实际为 {node3_degree}"
        )

        # 3. Test edge_degree - get edge degree / 测试 edge_degree - 获取边的度数
        print(f"== {t('test_edge_degree')}: {node1_id} -> {node2_id}")
        edge_degree = await storage.edge_degree(node1_id, node2_id)
        print(
            f"{t('edge_degree')} {node1_id} -> {node2_id}: {edge_degree}"
            if LANGUAGE == "english"
            else f"边 {node1_id} -> {node2_id} 的度数: {edge_degree}"
        )
        assert edge_degree == 3, (
            f"Edge {node1_id} -> {node2_id} degree should be 3, actual {edge_degree}"
            if LANGUAGE == "english"
            else f"边 {node1_id} -> {node2_id} 的度数应为3，实际为 {edge_degree}"
        )

        # 3.1 Test reverse edge degree - verify undirected graph property / 测试反向边的度数 - 验证无向图特性
        print(f"== {t('test_reverse_edge_degree')}: {node2_id} -> {node1_id}")
        reverse_edge_degree = await storage.edge_degree(node2_id, node1_id)
        print(
            f"{t('reverse_edge_degree')} {node2_id} -> {node1_id}: {reverse_edge_degree}"
            if LANGUAGE == "english"
            else f"反向边 {node2_id} -> {node1_id} 的度数: {reverse_edge_degree}"
        )
        assert edge_degree == reverse_edge_degree, (
            "Forward and reverse edge degrees inconsistent, undirected graph property verification failed"
            if LANGUAGE == "english"
            else "正向边和反向边的度数不一致，无向图特性验证失败"
        )
        print(t("undirected_verification_success"))

        # 4. Test get_node_edges - get all edges of node / 测试 get_node_edges - 获取节点的所有边
        print(f"== {t('test_get_node_edges')}: {node2_id}")
        node2_edges = await storage.get_node_edges(node2_id)
        print(
            f"{t('node_degree')} {node2_id} {t('all_edges')}: {node2_edges}"
            if LANGUAGE == "english"
            else f"节点 {node2_id} 的所有边: {node2_edges}"
        )
        assert len(node2_edges) == 2, (
            f"Node {node2_id} should have 2 edges, actual {len(node2_edges)}"
            if LANGUAGE == "english"
            else f"节点 {node2_id} 应有2条边，实际有 {len(node2_edges)}"
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

        assert has_connection_with_node1, (
            f"Node {node2_id} edge list should contain connection with {node1_id}"
            if LANGUAGE == "english"
            else f"节点 {node2_id} 的边列表中应包含与 {node1_id} 的连接"
        )
        assert has_connection_with_node3, (
            f"Node {node2_id} edge list should contain connection with {node3_id}"
            if LANGUAGE == "english"
            else f"节点 {node2_id} 的边列表中应包含与 {node3_id} 的连接"
        )
        print(
            f"Undirected graph property verification successful: node {node2_id} edge list contains all related edges"
            if LANGUAGE == "english"
            else f"无向图特性验证成功：节点 {node2_id} 的边列表包含所有相关的边"
        )

        # 5. Test get_all_labels - get all labels / 测试 get_all_labels - 获取所有标签
        print(f"== {t('test_get_all_labels')}")
        all_labels = await storage.get_all_labels()
        print(f"{t('all_labels')}: {all_labels}")
        assert len(all_labels) == 3, (
            f"Should have 3 labels, actual {len(all_labels)}"
            if LANGUAGE == "english"
            else f"应有3个标签，实际有 {len(all_labels)}"
        )
        assert node1_id in all_labels, (
            f"{node1_id} should be in label list"
            if LANGUAGE == "english"
            else f"{node1_id} 应在标签列表中"
        )
        assert node2_id in all_labels, (
            f"{node2_id} should be in label list"
            if LANGUAGE == "english"
            else f"{node2_id} 应在标签列表中"
        )
        assert node3_id in all_labels, (
            f"{node3_id} should be in label list"
            if LANGUAGE == "english"
            else f"{node3_id} 应在标签列表中"
        )

        # 6. Test get_knowledge_graph - get knowledge graph / 测试 get_knowledge_graph - 获取知识图谱
        print(f"== {t('test_get_knowledge_graph')}")
        kg = await storage.get_knowledge_graph("*", max_depth=2, max_nodes=10)
        print(f"{t('knowledge_graph_nodes')}: {len(kg.nodes)}")
        print(f"{t('knowledge_graph_edges')}: {len(kg.edges)}")
        assert isinstance(kg, KnowledgeGraph), (
            "Result should be KnowledgeGraph type"
            if LANGUAGE == "english"
            else "返回结果应为 KnowledgeGraph 类型"
        )
        assert len(kg.nodes) == 3, (
            f"Knowledge graph should have 3 nodes, actual {len(kg.nodes)}"
            if LANGUAGE == "english"
            else f"知识图谱应有3个节点，实际有 {len(kg.nodes)}"
        )
        assert len(kg.edges) == 2, (
            f"Knowledge graph should have 2 edges, actual {len(kg.edges)}"
            if LANGUAGE == "english"
            else f"知识图谱应有2条边，实际有 {len(kg.edges)}"
        )

        # 7. Test delete_node - delete node / 测试 delete_node - 删除节点
        print(f"== {t('test_delete_node')}: {node3_id}")
        await storage.delete_node(node3_id)
        node3_props = await storage.get_node(node3_id)
        print(
            f"{t('query_after_deletion')} {node3_id}: {node3_props}"
            if LANGUAGE == "english"
            else f"删除后查询节点属性 {node3_id}: {node3_props}"
        )
        assert node3_props is None, (
            f"Node {node3_id} should have been deleted"
            if LANGUAGE == "english"
            else f"节点 {node3_id} 应已被删除"
        )

        # Re-insert node3 for subsequent testing / 重新插入节点3用于后续测试
        await storage.upsert_node(node3_id, node3_data)
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 8. Test remove_edges - delete edges / 测试 remove_edges - 删除边
        print(f"== {t('test_remove_edges')}: {node2_id} -> {node3_id}")
        await storage.remove_edges([(node2_id, node3_id)])
        edge_props = await storage.get_edge(node2_id, node3_id)
        print(
            f"{t('query_after_deletion')} {node2_id} -> {node3_id}: {edge_props}"
            if LANGUAGE == "english"
            else f"删除后查询边属性 {node2_id} -> {node3_id}: {edge_props}"
        )
        assert edge_props is None, (
            f"Edge {node2_id} -> {node3_id} should have been deleted"
            if LANGUAGE == "english"
            else f"边 {node2_id} -> {node3_id} 应已被删除"
        )

        # 8.1 Verify undirected graph property for edge deletion / 验证删除边的无向图特性
        print(f"== {t('verify_undirected_property')}: {node3_id} -> {node2_id}")
        reverse_edge_props = await storage.get_edge(node3_id, node2_id)
        print(
            f"{t('query_after_deletion')} reverse edge {node3_id} -> {node2_id}: {reverse_edge_props}"
            if LANGUAGE == "english"
            else f"删除后查询反向边属性 {node3_id} -> {node2_id}: {reverse_edge_props}"
        )
        assert reverse_edge_props is None, (
            f"Reverse edge {node3_id} -> {node2_id} should also be deleted, undirected graph property verification failed"
            if LANGUAGE == "english"
            else f"反向边 {node3_id} -> {node2_id} 也应被删除，无向图特性验证失败"
        )
        print(
            "Undirected graph property verification successful: after deleting edge in one direction, reverse edge is also deleted"
            if LANGUAGE == "english"
            else "无向图特性验证成功：删除一个方向的边后，反向边也被删除"
        )

        # 9. Test remove_nodes - batch delete nodes / 测试 remove_nodes - 批量删除节点
        print(f"== {t('test_remove_nodes')}: [{node2_id}, {node3_id}]")
        await storage.remove_nodes([node2_id, node3_id])
        node2_props = await storage.get_node(node2_id)
        node3_props = await storage.get_node(node3_id)
        print(
            f"{t('query_after_deletion')} {node2_id}: {node2_props}"
            if LANGUAGE == "english"
            else f"删除后查询节点属性 {node2_id}: {node2_props}"
        )
        print(
            f"{t('query_after_deletion')} {node3_id}: {node3_props}"
            if LANGUAGE == "english"
            else f"删除后查询节点属性 {node3_id}: {node3_props}"
        )
        assert node2_props is None, (
            f"Node {node2_id} should have been deleted"
            if LANGUAGE == "english"
            else f"节点 {node2_id} 应已被删除"
        )
        assert node3_props is None, (
            f"Node {node3_id} should have been deleted"
            if LANGUAGE == "english"
            else f"节点 {node3_id} 应已被删除"
        )

        print(f"\n{t('advanced_test_complete')}")
        return True

    except Exception as e:
        ASCIIColors.red(f"测试过程中发生错误: {str(e)}")
        return False


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
        assert len(nodes_dict) == 3, f"应返回3个节点，实际返回 {len(nodes_dict)} 个"
        assert node1_id in nodes_dict, f"{node1_id} 应在返回结果中"
        assert node2_id in nodes_dict, f"{node2_id} 应在返回结果中"
        assert node3_id in nodes_dict, f"{node3_id} 应在返回结果中"
        assert (
            nodes_dict[node1_id]["description"] == node1_data["description"]
        ), f"{node1_id} 描述不匹配"
        assert (
            nodes_dict[node2_id]["description"] == node2_data["description"]
        ), f"{node2_id} 描述不匹配"
        assert (
            nodes_dict[node3_id]["description"] == node3_data["description"]
        ), f"{node3_id} 描述不匹配"

        # 3. 测试 node_degrees_batch - 批量获取多个节点的度数
        print(t("batch_node_degrees"))
        node_degrees = await storage.node_degrees_batch(node_ids)
        print(f"{t('batch_node_degrees_result')}: {node_degrees}")
        assert (
            len(node_degrees) == 3
        ), f"应返回3个节点的度数，实际返回 {len(node_degrees)} 个"
        assert node1_id in node_degrees, f"{node1_id} 应在返回结果中"
        assert node2_id in node_degrees, f"{node2_id} 应在返回结果中"
        assert node3_id in node_degrees, f"{node3_id} 应在返回结果中"
        assert (
            node_degrees[node1_id] == 3
        ), f"{node1_id} 度数应为3，实际为 {node_degrees[node1_id]}"
        assert (
            node_degrees[node2_id] == 2
        ), f"{node2_id} 度数应为2，实际为 {node_degrees[node2_id]}"
        assert (
            node_degrees[node3_id] == 3
        ), f"{node3_id} 度数应为3，实际为 {node_degrees[node3_id]}"

        # 4. 测试 edge_degrees_batch - 批量获取多个边的度数
        print(t("batch_edge_degrees"))
        edges = [(node1_id, node2_id), (node2_id, node3_id), (node3_id, node4_id)]
        edge_degrees = await storage.edge_degrees_batch(edges)
        print(f"{t('batch_edge_degrees_result')}: {edge_degrees}")
        assert (
            len(edge_degrees) == 3
        ), f"应返回3条边的度数，实际返回 {len(edge_degrees)} 条"
        assert (
            node1_id,
            node2_id,
        ) in edge_degrees, f"边 {node1_id} -> {node2_id} 应在返回结果中"
        assert (
            node2_id,
            node3_id,
        ) in edge_degrees, f"边 {node2_id} -> {node3_id} 应在返回结果中"
        assert (
            node3_id,
            node4_id,
        ) in edge_degrees, f"边 {node3_id} -> {node4_id} 应在返回结果中"
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
        assert len(edges_dict) == 3, f"应返回3条边的属性，实际返回 {len(edges_dict)} 条"
        assert (
            node1_id,
            node2_id,
        ) in edges_dict, f"边 {node1_id} -> {node2_id} 应在返回结果中"
        assert (
            node2_id,
            node3_id,
        ) in edges_dict, f"边 {node2_id} -> {node3_id} 应在返回结果中"
        assert (
            node3_id,
            node4_id,
        ) in edges_dict, f"边 {node3_id} -> {node4_id} 应在返回结果中"
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
        assert (
            len(reverse_edges_dict) == 3
        ), f"应返回3条反向边的属性，实际返回 {len(reverse_edges_dict)} 条"

        # 验证正向和反向边的属性是否一致
        for (src, tgt), props in edges_dict.items():
            assert (
                tgt,
                src,
            ) in reverse_edges_dict, f"反向边 {tgt} -> {src} 应在返回结果中"
            assert (
                props == reverse_edges_dict[(tgt, src)]
            ), f"边 {src} -> {tgt} 和反向边 {tgt} -> {src} 的属性不一致"

        print(t("undirected_batch_verification_success"))

        # 6. 测试 get_nodes_edges_batch - 批量获取多个节点的所有边
        print(t("test_get_nodes_edges_batch"))
        nodes_edges = await storage.get_nodes_edges_batch([node1_id, node3_id])
        print(f"{t('batch_get_nodes_edges_result')}: {nodes_edges.keys()}")
        assert (
            len(nodes_edges) == 2
        ), f"应返回2个节点的边，实际返回 {len(nodes_edges)} 个"
        assert node1_id in nodes_edges, f"{node1_id} 应在返回结果中"
        assert node3_id in nodes_edges, f"{node3_id} 应在返回结果中"
        assert (
            len(nodes_edges[node1_id]) == 3
        ), f"{node1_id} 应有3条边，实际有 {len(nodes_edges[node1_id])} 条"
        assert (
            len(nodes_edges[node3_id]) == 3
        ), f"{node3_id} 应有3条边，实际有 {len(nodes_edges[node3_id])} 条"

        # 6.1 验证批量获取节点边的无向图特性
        print(t("verify_batch_nodes_edges_undirected"))

        # 检查节点1的边是否包含所有相关的边（无论方向）
        node1_outgoing_edges = [
            (src, tgt) for src, tgt in nodes_edges[node1_id] if src == node1_id
        ]
        node1_incoming_edges = [
            (src, tgt) for src, tgt in nodes_edges[node1_id] if tgt == node1_id
        ]
        print(f"节点 {node1_id} {t('node_outgoing_edges')}: {node1_outgoing_edges}")
        print(f"节点 {node1_id} {t('node_incoming_edges')}: {node1_incoming_edges}")

        # 检查是否包含到机器学习、自然语言处理和计算机视觉的边
        has_edge_to_node2 = any(tgt == node2_id for _, tgt in node1_outgoing_edges)
        has_edge_to_node4 = any(tgt == node4_id for _, tgt in node1_outgoing_edges)
        has_edge_to_node5 = any(tgt == node5_id for _, tgt in node1_outgoing_edges)

        assert has_edge_to_node2, f"节点 {node1_id} 的边列表中应包含到 {node2_id} 的边"
        assert has_edge_to_node4, f"节点 {node1_id} 的边列表中应包含到 {node4_id} 的边"
        assert has_edge_to_node5, f"节点 {node1_id} 的边列表中应包含到 {node5_id} 的边"

        # 检查节点3的边是否包含所有相关的边（无论方向）
        node3_outgoing_edges = [
            (src, tgt) for src, tgt in nodes_edges[node3_id] if src == node3_id
        ]
        node3_incoming_edges = [
            (src, tgt) for src, tgt in nodes_edges[node3_id] if tgt == node3_id
        ]
        print(f"节点 {node3_id} {t('node_outgoing_edges')}: {node3_outgoing_edges}")
        print(f"节点 {node3_id} {t('node_incoming_edges')}: {node3_incoming_edges}")

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

        assert (
            has_connection_with_node2
        ), f"节点 {node3_id} 的边列表中应包含与 {node2_id} 的连接"
        assert (
            has_connection_with_node4
        ), f"节点 {node3_id} 的边列表中应包含与 {node4_id} 的连接"
        assert (
            has_connection_with_node5
        ), f"节点 {node3_id} 的边列表中应包含与 {node5_id} 的连接"

        print(t("undirected_nodes_edges_verification_success"))

        # 7. 测试 get_nodes_by_chunk_ids - 批量根据 chunk_ids 获取多个节点
        print(t("test_get_nodes_by_chunk_ids"))

        print(t("test_single_chunk_id_multiple_nodes"))
        nodes = await storage.get_nodes_by_chunk_ids([chunk2_id])
        assert len(nodes) == 2, f"{chunk1_id} 应有2个节点，实际有 {len(nodes)} 个"

        has_node1 = any(node["entity_id"] == node1_id for node in nodes)
        has_node2 = any(node["entity_id"] == node2_id for node in nodes)

        assert has_node1, f"节点 {node1_id} 应在返回结果中"
        assert has_node2, f"节点 {node2_id} 应在返回结果中"

        print(t("test_multiple_chunk_ids_partial_match"))
        nodes = await storage.get_nodes_by_chunk_ids([chunk2_id, chunk3_id])
        assert (
            len(nodes) == 3
        ), f"{chunk2_id}, {chunk3_id} 应有3个节点，实际有 {len(nodes)} 个"

        has_node1 = any(node["entity_id"] == node1_id for node in nodes)
        has_node2 = any(node["entity_id"] == node2_id for node in nodes)
        has_node3 = any(node["entity_id"] == node3_id for node in nodes)

        assert has_node1, f"节点 {node1_id} 应在返回结果中"
        assert has_node2, f"节点 {node2_id} 应在返回结果中"
        assert has_node3, f"节点 {node3_id} 应在返回结果中"

        # 8. 测试 get_edges_by_chunk_ids - 批量根据 chunk_ids 获取多条边
        print(t("test_get_edges_by_chunk_ids"))

        print(t("test_single_chunk_id_multiple_edges"))
        edges = await storage.get_edges_by_chunk_ids([chunk2_id])
        assert len(edges) == 2, f"{chunk2_id} 应有2条边，实际有 {len(edges)} 条"

        has_edge_node1_node2 = any(
            edge["source"] == node1_id and edge["target"] == node2_id for edge in edges
        )
        has_edge_node2_node3 = any(
            edge["source"] == node2_id and edge["target"] == node3_id for edge in edges
        )

        assert has_edge_node1_node2, f"{chunk2_id} 应包含 {node1_id} 到 {node2_id} 的边"
        assert has_edge_node2_node3, f"{chunk2_id} 应包含 {node2_id} 到 {node3_id} 的边"

        print(t("test_multiple_chunk_ids_partial_edges"))
        edges = await storage.get_edges_by_chunk_ids([chunk2_id, chunk3_id])
        assert (
            len(edges) == 3
        ), f"{chunk2_id}, {chunk3_id} 应有3条边，实际有 {len(edges)} 条"

        has_edge_node1_node2 = any(
            edge["source"] == node1_id and edge["target"] == node2_id for edge in edges
        )
        has_edge_node2_node3 = any(
            edge["source"] == node2_id and edge["target"] == node3_id for edge in edges
        )
        has_edge_node1_node4 = any(
            edge["source"] == node1_id and edge["target"] == node4_id for edge in edges
        )

        assert (
            has_edge_node1_node2
        ), f"{chunk2_id}, {chunk3_id} 应包含 {node1_id} 到 {node2_id} 的边"
        assert (
            has_edge_node2_node3
        ), f"{chunk2_id}, {chunk3_id} 应包含 {node2_id} 到 {node3_id} 的边"
        assert (
            has_edge_node1_node4
        ), f"{chunk2_id}, {chunk3_id} 应包含 {node1_id} 到 {node4_id} 的边"

        print(t("batch_operations_test_complete"))
        return True

    except Exception as e:
        ASCIIColors.red(f"测试过程中发生错误: {str(e)}")
        return False


async def test_graph_special_characters(storage):
    """
    Test graph database handling of special characters / 测试图数据库对特殊字符的处理:
    1. Test node names and descriptions containing single quotes, double quotes and backslashes / 测试节点名称和描述中包含单引号、双引号和反斜杠
    2. Test edge descriptions containing single quotes, double quotes and backslashes / 测试边的描述中包含单引号、双引号和反斜杠
    3. Verify special characters are correctly saved and retrieved / 验证特殊字符是否被正确保存和检索
    """
    try:
        # 1. Test special characters in node names / 测试节点名称中的特殊字符
        node1_id = t("node_with_quotes")
        node1_data = {
            "entity_id": node1_id,
            "description": t("desc_with_special"),
            "keywords": t("keywords_special"),
            "entity_type": t("test_node"),
        }
        print(
            f"{t('insert_node')} 1 with special characters: {node1_id}"
            if LANGUAGE == "english"
            else f"插入包含特殊字符的节点1: {node1_id}"
        )
        await storage.upsert_node(node1_id, node1_data)

        # 2. Test double quotes in node names / 测试节点名称中的双引号
        node2_id = t("node_with_double_quotes")
        node2_data = {
            "entity_id": node2_id,
            "description": t("desc_with_complex"),
            "keywords": t("keywords_json"),
            "entity_type": t("test_node"),
        }
        print(
            f"{t('insert_node')} 2 with special characters: {node2_id}"
            if LANGUAGE == "english"
            else f"插入包含特殊字符的节点2: {node2_id}"
        )
        await storage.upsert_node(node2_id, node2_data)

        # 3. Test backslashes in node names / 测试节点名称中的反斜杠
        node3_id = t("node_with_backslash")
        node3_data = {
            "entity_id": node3_id,
            "description": t("desc_with_windows_path"),
            "keywords": t("keywords_backslash"),
            "entity_type": t("test_node"),
        }
        print(
            f"{t('insert_node')} 3 with special characters: {node3_id}"
            if LANGUAGE == "english"
            else f"插入包含特殊字符的节点3: {node3_id}"
        )
        await storage.upsert_node(node3_id, node3_data)

        # 4. Test special characters in edge descriptions / 测试边描述中的特殊字符
        edge1_data = {
            "relationship": t("special_relation"),
            "weight": 1.0,
            "description": t("edge_desc_special"),
        }
        print(
            f"{t('insert_edge')} with special characters: {node1_id} -> {node2_id}"
            if LANGUAGE == "english"
            else f"插入包含特殊字符的边: {node1_id} -> {node2_id}"
        )
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # 5. Test more complex special character combinations in edges / 测试边描述中的更复杂特殊字符组合
        edge2_data = {
            "relationship": t("complex_relation"),
            "weight": 0.8,
            "description": t("edge_desc_sql"),
        }
        print(
            f"{t('insert_edge')} with complex special characters: {node2_id} -> {node3_id}"
            if LANGUAGE == "english"
            else f"插入包含复杂特殊字符的边: {node2_id} -> {node3_id}"
        )
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 6. Verify node special characters are correctly saved / 验证节点特殊字符是否正确保存
        print(f"\n== {t('verify_node_special')}")
        for node_id, original_data in [
            (node1_id, node1_data),
            (node2_id, node2_data),
            (node3_id, node3_data),
        ]:
            node_props = await storage.get_node(node_id)
            if node_props:
                print(
                    f"Successfully read node: {node_id}"
                    if LANGUAGE == "english"
                    else f"成功读取节点: {node_id}"
                )
                print(
                    f"Node description: {node_props.get('description', 'No description')}"
                    if LANGUAGE == "english"
                    else f"节点描述: {node_props.get('description', '无描述')}"
                )

                # Verify node ID is correctly saved / 验证节点ID是否正确保存
                assert node_props.get("entity_id") == node_id, (
                    f"Node ID mismatch: expected {node_id}, actual {node_props.get('entity_id')}"
                    if LANGUAGE == "english"
                    else f"节点ID不匹配: 期望 {node_id}, 实际 {node_props.get('entity_id')}"
                )

                # Verify description is correctly saved / 验证描述是否正确保存
                assert node_props.get("description") == original_data["description"], (
                    f"Node description mismatch: expected {original_data['description']}, actual {node_props.get('description')}"
                    if LANGUAGE == "english"
                    else f"节点描述不匹配: 期望 {original_data['description']}, 实际 {node_props.get('description')}"
                )

                print(
                    f"Node {node_id} special character verification successful"
                    if LANGUAGE == "english"
                    else f"节点 {node_id} 特殊字符验证成功"
                )
            else:
                print(
                    f"Failed to read node properties: {node_id}"
                    if LANGUAGE == "english"
                    else f"读取节点属性失败: {node_id}"
                )
                assert False, (
                    f"Unable to read node properties: {node_id}"
                    if LANGUAGE == "english"
                    else f"未能读取节点属性: {node_id}"
                )

        # 7. Verify edge special characters are correctly saved / 验证边特殊字符是否正确保存
        print(f"\n== {t('verify_edge_special')}")
        edge1_props = await storage.get_edge(node1_id, node2_id)
        if edge1_props:
            print(
                f"Successfully read edge: {node1_id} -> {node2_id}"
                if LANGUAGE == "english"
                else f"成功读取边: {node1_id} -> {node2_id}"
            )
            print(
                f"Edge relationship: {edge1_props.get('relationship', 'No relationship')}"
                if LANGUAGE == "english"
                else f"边关系: {edge1_props.get('relationship', '无关系')}"
            )
            print(
                f"Edge description: {edge1_props.get('description', 'No description')}"
                if LANGUAGE == "english"
                else f"边描述: {edge1_props.get('description', '无描述')}"
            )

            # Verify edge relationship is correctly saved / 验证边关系是否正确保存
            assert edge1_props.get("relationship") == edge1_data["relationship"], (
                f"Edge relationship mismatch: expected {edge1_data['relationship']}, actual {edge1_props.get('relationship')}"
                if LANGUAGE == "english"
                else f"边关系不匹配: 期望 {edge1_data['relationship']}, 实际 {edge1_props.get('relationship')}"
            )

            # Verify edge description is correctly saved / 验证边描述是否正确保存
            assert edge1_props.get("description") == edge1_data["description"], (
                f"Edge description mismatch: expected {edge1_data['description']}, actual {edge1_props.get('description')}"
                if LANGUAGE == "english"
                else f"边描述不匹配: 期望 {edge1_data['description']}, 实际 {edge1_props.get('description')}"
            )

            print(
                f"Edge {node1_id} -> {node2_id} special character verification successful"
                if LANGUAGE == "english"
                else f"边 {node1_id} -> {node2_id} 特殊字符验证成功"
            )
        else:
            print(
                f"Failed to read edge properties: {node1_id} -> {node2_id}"
                if LANGUAGE == "english"
                else f"读取边属性失败: {node1_id} -> {node2_id}"
            )
            assert False, (
                f"Unable to read edge properties: {node1_id} -> {node2_id}"
                if LANGUAGE == "english"
                else f"未能读取边属性: {node1_id} -> {node2_id}"
            )

        edge2_props = await storage.get_edge(node2_id, node3_id)
        if edge2_props:
            print(
                f"Successfully read edge: {node2_id} -> {node3_id}"
                if LANGUAGE == "english"
                else f"成功读取边: {node2_id} -> {node3_id}"
            )
            print(
                f"Edge relationship: {edge2_props.get('relationship', 'No relationship')}"
                if LANGUAGE == "english"
                else f"边关系: {edge2_props.get('relationship', '无关系')}"
            )
            print(
                f"Edge description: {edge2_props.get('description', 'No description')}"
                if LANGUAGE == "english"
                else f"边描述: {edge2_props.get('description', '无描述')}"
            )

            # Verify edge relationship is correctly saved / 验证边关系是否正确保存
            assert edge2_props.get("relationship") == edge2_data["relationship"], (
                f"Edge relationship mismatch: expected {edge2_data['relationship']}, actual {edge2_props.get('relationship')}"
                if LANGUAGE == "english"
                else f"边关系不匹配: 期望 {edge2_data['relationship']}, 实际 {edge2_props.get('relationship')}"
            )

            # Verify edge description is correctly saved / 验证边描述是否正确保存
            assert edge2_props.get("description") == edge2_data["description"], (
                f"Edge description mismatch: expected {edge2_data['description']}, actual {edge2_props.get('description')}"
                if LANGUAGE == "english"
                else f"边描述不匹配: 期望 {edge2_data['description']}, 实际 {edge2_props.get('description')}"
            )

            print(
                f"Edge {node2_id} -> {node3_id} special character verification successful"
                if LANGUAGE == "english"
                else f"边 {node2_id} -> {node3_id} 特殊字符验证成功"
            )
        else:
            print(
                f"Failed to read edge properties: {node2_id} -> {node3_id}"
                if LANGUAGE == "english"
                else f"读取边属性失败: {node2_id} -> {node3_id}"
            )
            assert False, (
                f"Unable to read edge properties: {node2_id} -> {node3_id}"
                if LANGUAGE == "english"
                else f"未能读取边属性: {node2_id} -> {node3_id}"
            )

        print(f"\n{t('special_char_test_complete')}")
        return True

    except Exception as e:
        ASCIIColors.red(f"{t('test_error')}: {str(e)}")
        return False


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
        print(f"\n== {t('test_insert_edge_undirected_property')}")
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
        print(f"\n== {t('test_edge_degree_undirected_property')}")
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
        print(f"\n== {t('test_delete_edge_undirected_property')}")
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
        print(f"\n== {t('test_batch_undirected_property')}")
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
        print(f"\n== {t('test_batch_get_node_edges_undirected_property')}")
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

        print(f"\n{t('undirected_test_complete')}")
        return True

    except Exception as e:
        ASCIIColors.red(f"{t('test_error')}: {str(e)}")
        return False


async def main():
    """Main function / 主函数"""
    # Display program title / 显示程序标题
    if LANGUAGE == "english":
        ASCIIColors.cyan(
            """
    ╔══════════════════════════════════════════════════════════════╗
    ║                 Universal Graph Storage Test                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """
        )
    else:
        ASCIIColors.cyan(
            """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🍇  通用图存储测试程序  🍇                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """
        )

    # Check .env file / 检查.env文件
    if not check_env_file():
        return

    # Load environment variables / 加载环境变量
    load_dotenv(dotenv_path=".env", override=False)

    # Get graph storage type / 获取图存储类型
    graph_storage_type = os.getenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    ASCIIColors.magenta(f"\n{t('current_graph_storage')}: {graph_storage_type}")
    ASCIIColors.white(
        f"{t('supported_graph_storage')}: {', '.join(STORAGE_IMPLEMENTATIONS['GRAPH_STORAGE']['implementations'])}"
    )

    # Initialize storage instance / 初始化存储实例
    storage = await initialize_graph_storage()
    if not storage:
        ASCIIColors.red(t("init_storage_failed"))
        return

    try:
        # Display test options / 显示测试选项
        ASCIIColors.yellow(f"\n{t('select_test_type')}")
        ASCIIColors.white(t("basic_test"))
        ASCIIColors.white(t("advanced_test"))
        ASCIIColors.white(t("batch_test"))
        ASCIIColors.white(t("undirected_test"))
        ASCIIColors.white(t("special_char_test"))
        ASCIIColors.white(t("all_tests"))

        choice = input(f"\n{t('select_option')}")

        # Clean data before executing tests / 在执行测试前清理数据
        if choice in ["1", "2", "3", "4", "5", "6"]:
            ASCIIColors.yellow(f"\n{t('cleaning_data')}")
            await storage.drop()
            ASCIIColors.green(f"{t('data_cleaned')}\n")

        if choice == "1":
            await test_graph_basic(storage)
        elif choice == "2":
            await test_graph_advanced(storage)
        elif choice == "3":
            await test_graph_batch_operations(storage)
        elif choice == "4":
            await test_graph_undirected_property(storage)
        elif choice == "5":
            await test_graph_special_characters(storage)
        elif choice == "6":
            if LANGUAGE == "english":
                ASCIIColors.cyan("\n=== Starting Basic Test ===")
            else:
                ASCIIColors.cyan("\n=== 开始基本测试 ===")
            basic_result = await test_graph_basic(storage)

            if basic_result:
                if LANGUAGE == "english":
                    ASCIIColors.cyan("\n=== Starting Advanced Test ===")
                else:
                    ASCIIColors.cyan("\n=== 开始高级测试 ===")
                advanced_result = await test_graph_advanced(storage)

                if advanced_result:
                    if LANGUAGE == "english":
                        ASCIIColors.cyan("\n=== Starting Batch Operations Test ===")
                    else:
                        ASCIIColors.cyan("\n=== 开始批量操作测试 ===")
                    batch_result = await test_graph_batch_operations(storage)

                    if batch_result:
                        if LANGUAGE == "english":
                            ASCIIColors.cyan(
                                "\n=== Starting Undirected Graph Property Test ==="
                            )
                        else:
                            ASCIIColors.cyan("\n=== 开始无向图特性测试 ===")
                        undirected_result = await test_graph_undirected_property(
                            storage
                        )

                        if undirected_result:
                            if LANGUAGE == "english":
                                ASCIIColors.cyan(
                                    "\n=== Starting Special Character Test ==="
                                )
                            else:
                                ASCIIColors.cyan("\n=== 开始特殊字符测试 ===")
                            await test_graph_special_characters(storage)
        else:
            ASCIIColors.red(t("invalid_option"))

    finally:
        # Close connection / 关闭连接
        if storage:
            await storage.finalize()
            # Clean KuzuDB temporary directory / 清理KuzuDB临时目录
            if hasattr(storage, "_temp_dir") and storage._temp_dir:
                cleanup_kuzu_environment(storage._temp_dir)
                ASCIIColors.green(f"\n{t('kuzu_temp_cleaned')}")
            ASCIIColors.green(f"\n{t('connection_closed')}")


if __name__ == "__main__":
    # Parse command line arguments / 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Universal Graph Storage Test Program / 通用图存储测试程序"
    )
    parser.add_argument(
        "--language",
        "-l",
        choices=["english", "chinese"],
        default=LANGUAGE,
        help="Test language (default: from TEST_LANGUAGE env var or 'chinese') / 测试语言（默认：从 TEST_LANGUAGE 环境变量或 'chinese'）",
    )

    args = parser.parse_args()

    # Override language setting from command line / 从命令行覆盖语言设置
    if args.language != LANGUAGE:
        globals()["LANGUAGE"] = args.language

    # Print language info / 打印语言信息
    if LANGUAGE == "english":
        print(
            "Running tests in English. Use --language chinese or set TEST_LANGUAGE=chinese for Chinese."
        )
    else:
        print(
            "使用中文运行测试。使用 --language english 或设置 TEST_LANGUAGE=english 来使用英文。"
        )

    asyncio.run(main())
