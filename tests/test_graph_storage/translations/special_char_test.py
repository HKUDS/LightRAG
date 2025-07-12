"""
Translations specific to special character testing
"""

SPECIAL_CHAR_TEST_TRANSLATIONS = {
    # Special character nodes
    "node_with_quotes": ["包含'单引号'的节点", "Node with 'single quotes'"],
    "node_with_double_quotes": ['包含"双引号"的节点', 'Node with "double quotes"'],
    "node_with_backslash": ["包含\\反斜杠\\的节点", "Node with \\backslash\\"],
    # Special character descriptions
    "desc_with_special": [
        "这个描述包含'单引号'、\"双引号\"和\\反斜杠",
        "This description contains 'single quotes', \"double quotes\" and \\backslash",
    ],
    "desc_with_complex": [
        "这个描述同时包含'单引号'和\"双引号\"以及\\反斜杠\\路径",
        "This description contains 'single quotes' and \"double quotes\" as well as \\backslash\\path",
    ],
    "desc_with_windows_path": [
        "这个描述包含Windows路径C:\\Program Files\\和转义字符\\n\\t",
        "This description contains Windows path C:\\Program Files\\ and escape characters \\n\\t",
    ],
    # Keywords with special characters
    "keywords_special": [
        "特殊字符,引号,转义",
        "special characters,quotes,escape",
    ],
    "keywords_backslash": [
        "反斜杠,路径,转义",
        "backslash,path,escape",
    ],
    "keywords_json": [
        "特殊字符,引号,JSON",
        "special characters,quotes,JSON",
    ],
    # Edge descriptions with special characters
    "edge_desc_special": [
        "这个边描述包含'单引号'、\"双引号\"和\\反斜杠",
        "This edge description contains 'single quotes', \"double quotes\" and \\backslash",
    ],
    "edge_desc_sql": [
        "包含SQL注入尝试: SELECT * FROM users WHERE name='admin'--",
        "Contains SQL injection attempt: SELECT * FROM users WHERE name='admin'--",
    ],
    # Relations with special characters
    "complex_relation": [
        "复杂'关系'包含\\转义",
        "complex 'relation' with \\escape",
    ],
    # Test operations
    "insert_node_with_special_1": [
        "插入包含特殊字符的节点1",
        "Insert node 1 with special characters",
    ],
    "insert_node_with_special_2": [
        "插入包含特殊字符的节点2",
        "Insert node 2 with special characters",
    ],
    "insert_node_with_special_3": [
        "插入包含特殊字符的节点3",
        "Insert node 3 with special characters",
    ],
    "insert_edge_with_special": [
        "插入包含特殊字符的边",
        "Insert edge with special characters",
    ],
    "insert_edge_with_complex_special": [
        "插入包含复杂特殊字符的边",
        "Insert edge with complex special characters",
    ],
    # Read operations
    "read_node_success": [
        "成功读取节点",
        "Successfully read node",
    ],
    "read_edge_success": [
        "成功读取边",
        "Successfully read edge",
    ],
    "read_node_props_failed": [
        "读取节点属性失败",
        "Failed to read node properties",
    ],
    "read_edge_props_failed": [
        "读取边属性失败",
        "Failed to read edge properties",
    ],
    # Property display
    "node_description": [
        "节点描述",
        "Node description",
    ],
    "edge_relationship": [
        "边关系",
        "Edge relationship",
    ],
    # Verification
    "verify_node_special": ["验证节点特殊字符", "Verify node special characters"],
    "verify_edge_special": ["验证边特殊字符", "Verify edge special characters"],
    "special_char_verification_success": [
        "特殊字符验证成功",
        "Special character verification successful",
    ],
    # Verification messages
    "node_special_char_verification_success": [
        "节点 {} 特殊字符验证成功",
        "Node {} special character verification successful",
    ],
    "edge_special_char_verification_success": [
        "边 {} -> {} 特殊字符验证成功",
        "Edge {} -> {} special character verification successful",
    ],
    # Error messages
    "node_id_mismatch_f": [
        "节点ID不匹配: 期望 {}, 实际 {}",
        "Node ID mismatch: expected {}, actual {}",
    ],
    "node_description_mismatch": [
        "节点描述不匹配: 期望 {}, 实际 {}",
        "Node description mismatch: expected {}, actual {}",
    ],
    "edge_relationship_mismatch": [
        "边关系不匹配: 期望 {}, 实际 {}",
        "Edge relationship mismatch: expected {}, actual {}",
    ],
    "edge_description_mismatch": [
        "边描述不匹配: 期望 {}, 实际 {}",
        "Edge description mismatch: expected {}, actual {}",
    ],
    "unable_to_read_node_props": [
        "未能读取节点属性: {}",
        "Unable to read node properties: {}",
    ],
    "unable_to_read_edge_props": [
        "未能读取边属性: {} -> {}",
        "Unable to read edge properties: {} -> {}",
    ],
    # Test completion
    "special_char_test_complete": [
        "特殊字符测试完成",
        "Special character test completed",
    ],
    "starting_special_character_test": [
        "\n=== 开始特殊字符测试 ===",
        "\n=== Starting Special Character Test ===",
    ],
}
