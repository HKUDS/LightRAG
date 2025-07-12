"""
Special characters test module for graph storage.
Tests handling of special characters in node names, descriptions, and edge properties.
"""

import asyncio
from ..core.translation_engine import t_enhanced as t
from ascii_colors import ASCIIColors


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
        print(f"{t('insert_node_with_special_1')}: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 2. Test double quotes in node names / 测试节点名称中的双引号
        node2_id = t("node_with_double_quotes")
        node2_data = {
            "entity_id": node2_id,
            "description": t("desc_with_complex"),
            "keywords": t("keywords_json"),
            "entity_type": t("test_node"),
        }
        print(f"{t('insert_node_with_special_2')}: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 3. Test backslashes in node names / 测试节点名称中的反斜杠
        node3_id = t("node_with_backslash")
        node3_data = {
            "entity_id": node3_id,
            "description": t("desc_with_windows_path"),
            "keywords": t("keywords_backslash"),
            "entity_type": t("test_node"),
        }
        print(f"{t('insert_node_with_special_3')}: {node3_id}")
        await storage.upsert_node(node3_id, node3_data)

        # 4. Test special characters in edge descriptions / 测试边描述中的特殊字符
        edge1_data = {
            "relationship": t("special_relation"),
            "weight": 1.0,
            "description": t("edge_desc_special"),
        }
        print(f"{t('insert_edge_with_special')}: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge1_data)

        # 5. Test more complex special character combinations in edges / 测试边描述中的更复杂特殊字符组合
        edge2_data = {
            "relationship": t("complex_relation"),
            "weight": 0.8,
            "description": t("edge_desc_sql"),
        }
        print(f"{t('insert_edge_with_complex_special')}: {node2_id} -> {node3_id}")
        await storage.upsert_edge(node2_id, node3_id, edge2_data)

        # 6. Verify node special characters are correctly saved / 验证节点特殊字符是否正确保存
        print(f"\n== {t('verify_node_special')} ==")
        for node_id, original_data in [
            (node1_id, node1_data),
            (node2_id, node2_data),
            (node3_id, node3_data),
        ]:
            node_props = await storage.get_node(node_id)
            if node_props:
                print(f"{t('read_node_success')}: {node_id}")
                print(
                    f"{t('node_description')}: {node_props.get('description', t('no_description'))}"
                )

                # Verify node ID is correctly saved / 验证节点ID是否正确保存
                assert node_props.get("entity_id") == node_id, t(
                    "node_id_mismatch_f"
                ).format(node_id, node_props.get("entity_id"))

                # Verify description is correctly saved / 验证描述是否正确保存
                assert node_props.get("description") == original_data["description"], t(
                    "node_description_mismatch"
                ).format(original_data["description"], node_props.get("description"))

                print(t("node_special_char_verification_success").format(node_id))
            else:
                print(f"{t('read_node_props_failed')}: {node_id}")
                assert False, t("unable_to_read_node_props").format(node_id)

        # 7. Verify edge special characters are correctly saved / 验证边特殊字符是否正确保存
        print(f"\n== {t('verify_edge_special')} ==")
        edge1_props = await storage.get_edge(node1_id, node2_id)
        if edge1_props:
            print(f"{t('read_edge_success')}: {node1_id} -> {node2_id}")
            print(
                f"{t('edge_relationship')}: {edge1_props.get('relationship', t('no_relationship'))}"
            )
            print(
                f"{t('edge_desc')}: {edge1_props.get('description', t('no_description'))}"
            )

            # Verify edge relationship is correctly saved / 验证边关系是否正确保存
            assert edge1_props.get("relationship") == edge1_data["relationship"], t(
                "edge_relationship_mismatch"
            ).format(edge1_data["relationship"], edge1_props.get("relationship"))

            # Verify edge description is correctly saved / 验证边描述是否正确保存
            assert edge1_props.get("description") == edge1_data["description"], t(
                "edge_description_mismatch"
            ).format(edge1_data["description"], edge1_props.get("description"))

            print(
                t("edge_special_char_verification_success").format(node1_id, node2_id)
            )
        else:
            print(f"{t('read_edge_props_failed')}: {node1_id} -> {node2_id}")
            assert False, t("unable_to_read_edge_props").format(node1_id, node2_id)

        edge2_props = await storage.get_edge(node2_id, node3_id)
        if edge2_props:
            print(f"{t('read_edge_success')}: {node2_id} -> {node3_id}")
            print(
                f"{t('edge_relationship')}: {edge2_props.get('relationship', t('no_relationship'))}"
            )
            print(
                f"{t('edge_desc')}: {edge2_props.get('description', t('no_description'))}"
            )

            # Verify edge relationship is correctly saved / 验证边关系是否正确保存
            assert edge2_props.get("relationship") == edge2_data["relationship"], t(
                "edge_relationship_mismatch"
            ).format(edge2_data["relationship"], edge2_props.get("relationship"))

            # Verify edge description is correctly saved / 验证边描述是否正确保存
            assert edge2_props.get("description") == edge2_data["description"], t(
                "edge_description_mismatch"
            ).format(edge2_data["description"], edge2_props.get("description"))

            print(
                t("edge_special_char_verification_success").format(node2_id, node3_id)
            )
        else:
            print(f"{t('read_edge_props_failed')}: {node2_id} -> {node3_id}")
            assert False, t("unable_to_read_edge_props").format(node2_id, node3_id)

        ASCIIColors.green(t("special_char_test_complete"))
        return True

    except Exception as e:
        ASCIIColors.red(f"{t('test_error')}: {str(e)}")
        return False


# For direct execution/testing
async def main():
    """Test function for direct execution"""
    from ..core.storage_setup import initialize_graph_test_storage

    print(t("starting_special_character_test"))
    storage = await initialize_graph_test_storage()

    if storage:
        result = await test_graph_special_characters(storage)
        print(f"Test result: {result}")
    else:
        print("Storage initialization failed")


if __name__ == "__main__":
    asyncio.run(main())
