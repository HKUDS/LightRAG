import networkx as nx
import os
import json
import networkx as nx
from pyvis.network import Network
import random


G = nx.read_graphml("/Users/llp/opensource/LightRAG/tests/TechnicalDemo/graph_chunk_entity_relation.graphml")


def get_all_edges_and_nodes(G):
    # Get all edges and their properties
    edges_with_properties = []
    for u, v, data in G.edges(data=True):
        edges_with_properties.append(
            {
                "start": u,
                "end": v,
                "label": data.get(
                    "label", ""
                ),  # Assuming 'label' is used for edge type
                "properties": data,
                "start_node_properties": G.nodes[u],
                "end_node_properties": G.nodes[v],
            }
        )

    return edges_with_properties


def get_all_entities(G):
    """获取图中所有实体（节点）及其属性"""
    entities = []
    for node, data in G.nodes(data=True):
        entities.append({
            "id": node,
            "properties": data
        })
    return entities


def get_isolated_entities(G):
    """获取图中所有孤立的实体（没有任何连接的节点）"""
    isolated_entities = []
    for node in nx.isolates(G):
        isolated_entities.append({
            "id": node,
            "properties": G.nodes[node]
        })
    return isolated_entities


def remove_isolated_entities(G):
    """清除图中所有孤立的实体"""
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    return len(isolated_nodes)


def save_graph(G, output_path):
    """保存图到文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nx.write_graphml(G, output_path)
    print(f"图已保存到: {output_path}")


def save_entities_to_json(entities, output_path="/Users/llp/opensource/LightRAG/tests/test_results/entities.json"):
    """将实体信息保存为JSON格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建包含所有实体的字典
    entities_dict = {
        "entities": [
            {
                "id": entity["id"],
                "properties": entity["properties"]
            }
            for entity in entities
        ],
        "total_count": len(entities)
    }
    
    # 写入JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entities_dict, f, indent=2, ensure_ascii=False)
    
    print(f"实体信息已保存到: {output_path}")

def export_html(graphml_path):
    # Load the GraphML file
    G = nx.read_graphml(graphml_path)

    # Create a Pyvis network
    net = Network(height="100vh", notebook=True)

    # Convert NetworkX graph to Pyvis network
    net.from_nx(G)


    # Add colors and title to nodes
    for node in net.nodes:
        node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if "description" in node:
            node["title"] = node["description"]

    # Add title to edges
    for edge in net.edges:
        if "description" in edge:
            edge["title"] = edge["description"]

    # Save and display the network
    net.show("/Users/llp/opensource/LightRAG/tests/knowledge_graph.html")

# Example usage
if __name__ == "__main__":
    # Assume G is your NetworkX graph loaded from Neo4j

    # 获取并打印所有边
    # all_edges = get_all_edges_and_nodes(G)
    # print("===== 边信息 =====")
    # for edge in all_edges:
    #     print(f"Edge Label: {edge['label']}")
    #     print(f"Edge Properties: {edge['properties']}")
    #     print(f"Start Node: {edge['start']}")
    #     print(f"Start Node Properties: {edge['start_node_properties']}")
    #     print(f"End Node: {edge['end']}")
    #     print(f"End Node Properties: {edge['end_node_properties']}")
    #     print("---")
    
    # # 获取并打印所有实体
    # print("\n\n===== 实体信息 =====")
    # all_entities = get_all_entities(G)
    # for entity in all_entities:
    #     print(f"Entity ID: {entity['id']}")
    #     print(f"Entity Properties: {entity['properties']}")
    #     print("---")
    
    # 获取并打印所有孤立实体
    print("\n\n===== 孤立实体信息 =====")
    isolated_entities = get_isolated_entities(G)
    print(f"孤立实体总数: {len(isolated_entities)}")
    for entity in isolated_entities:
        print(f"Entity ID: {entity['id']}")
        print(f"Entity Properties: {entity['properties']}")
        print("---")
    
    # 清除孤立实体
    print("\n\n===== 清除孤立实体 =====")
    removed_count = remove_isolated_entities(G)
    print(f"已清除 {removed_count} 个孤立实体")
    
    # 验证是否还有孤立实体
    remaining_isolated = get_isolated_entities(G)
    print(f"剩余孤立实体数量: {len(remaining_isolated)}")
    
    # 保存清除孤立实体后的图
    print("\n\n===== 保存处理后的图 =====")
    output_path = "/Users/llp/opensource/LightRAG/tests/TechnicalDemo/graph_chunk_entity_relation_no_isolates.graphml"
    save_graph(G, output_path)

    export_html(output_path)
    
    # # 将实体信息保存为JSON格式
    # save_entities_to_json(all_entities)
