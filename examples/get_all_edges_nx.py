import networkx as nx

G = nx.read_graphml("./dickensTestEmbedcall/graph_chunk_entity_relation.graphml")


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


# Example usage
if __name__ == "__main__":
    # Assume G is your NetworkX graph loaded from Neo4j

    all_edges = get_all_edges_and_nodes(G)

    # Print all edges and node properties
    for edge in all_edges:
        print(f"Edge Label: {edge['label']}")
        print(f"Edge Properties: {edge['properties']}")
        print(f"Start Node: {edge['start']}")
        print(f"Start Node Properties: {edge['start_node_properties']}")
        print(f"End Node: {edge['end']}")
        print(f"End Node Properties: {edge['end_node_properties']}")
        print("---")
