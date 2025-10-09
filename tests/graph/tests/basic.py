"""
Basic graph storage test module
"""

from ..core.translation_engine import t
from ascii_colors import ASCIIColors


async def test_graph_basic(storage):
    """
    Test basic graph operations:
    1. Insert nodes using upsert_node
    2. Insert edges using upsert_edge
    3. Read node properties using get_node
    4. Read edge properties using get_edge
    """
    try:
        ASCIIColors.cyan(t("starting_basic_test"))

        # 1. Insert first node
        node1_id = t("artificial_intelligence")
        node1_data = {
            "entity_id": node1_id,
            "description": t("ai_desc"),
            "keywords": t("ai_keywords"),
            "entity_type": t("tech_field"),
        }
        print(f"{t('insert_node')} 1: {node1_id}")
        await storage.upsert_node(node1_id, node1_data)

        # 2. Insert second node
        node2_id = t("machine_learning")
        node2_data = {
            "entity_id": node2_id,
            "description": t("ml_desc"),
            "keywords": t("ml_keywords"),
            "entity_type": t("tech_field"),
        }
        print(f"{t('insert_node')} 2: {node2_id}")
        await storage.upsert_node(node2_id, node2_data)

        # 3. Insert connecting edge
        edge_data = {
            "relationship": t("contains"),
            "weight": 1.0,
            "description": t("ai_contains_ml"),
        }
        print(f"{t('insert_edge')}: {node1_id} -> {node2_id}")
        await storage.upsert_edge(node1_id, node2_id, edge_data)

        # 4. Read node properties
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
            # Verify returned properties are correct
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

        # 5. Read edge properties
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
            # Verify returned properties are correct
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

        # 5.1 Verify undirected graph property - read reverse edge properties
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
            # Verify forward and reverse edge properties are the same
            assert edge_props == reverse_edge_props, t("forward_reverse_inconsistent")
            print(t("undirected_verification_success"))
        else:
            print(f"{t('failed_read_reverse_edge')}: {node2_id} -> {node1_id}")
            assert False, f"{t('unable_read_reverse_edge')}: {node2_id} -> {node1_id}, {t('undirected_verification_failed')}"

        ASCIIColors.green(t("basic_test_complete"))
        return True

    except Exception as e:
        ASCIIColors.red(f"{t('test_error')}: {str(e)}")
        return False
