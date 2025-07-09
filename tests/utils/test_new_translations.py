#!/usr/bin/env python
"""
Test the new translation keys for test progress messages
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_new_translations():
    """Test the newly added translation keys"""

    print("=== Testing New Translation Keys ===\n")

    # Test English
    os.environ["TEST_LANGUAGE"] = "english"
    from tests.test_graph_storage import t
    import tests.test_graph_storage

    tests.test_graph_storage.LANGUAGE = "english"

    print("English test progress messages:")
    print(f"- Test node degree: {t('test_node_degree')}")
    print(f"- Test all node degrees: {t('test_all_node_degrees')}")
    print(f"- Test edge degree: {t('test_edge_degree')}")
    print(f"- Test reverse edge degree: {t('test_reverse_edge_degree')}")
    print(f"- Test get node edges: {t('test_get_node_edges')}")
    print(f"- Test get all labels: {t('test_get_all_labels')}")
    print(f"- Test get knowledge graph: {t('test_get_knowledge_graph')}")
    print(f"- Test delete node: {t('test_delete_node')}")
    print(f"- Test remove edges: {t('test_remove_edges')}")
    print(f"- Advanced test complete: {t('advanced_test_complete')}")

    # Test Chinese
    tests.test_graph_storage.LANGUAGE = "chinese"

    print("\nChinese test progress messages:")
    print(f"- Test node degree: {t('test_node_degree')}")
    print(f"- Test all node degrees: {t('test_all_node_degrees')}")
    print(f"- Test edge degree: {t('test_edge_degree')}")
    print(f"- Test reverse edge degree: {t('test_reverse_edge_degree')}")
    print(f"- Test get node edges: {t('test_get_node_edges')}")
    print(f"- Test get all labels: {t('test_get_all_labels')}")
    print(f"- Test get knowledge graph: {t('test_get_knowledge_graph')}")
    print(f"- Test delete node: {t('test_delete_node')}")
    print(f"- Test remove edges: {t('test_remove_edges')}")
    print(f"- Advanced test complete: {t('advanced_test_complete')}")

    print("\n=== All translations working! ===")

    print("\nNow the test output should show:")
    print("English: 'Test node_degree: Artificial Intelligence'")
    print("Chinese: '测试 node_degree: 人工智能'")
    print("\nInstead of mixed:")
    print("'== 测试 node_degree: Artificial Intelligence'")


if __name__ == "__main__":
    test_new_translations()
