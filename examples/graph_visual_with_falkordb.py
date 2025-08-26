import os
import xml.etree.ElementTree as ET
import falkordb

# Constants
WORKING_DIR = "./dickens"
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100

# FalkorDB connection credentials
FALKORDB_HOST = "localhost"
FALKORDB_PORT = 6379
FALKORDB_GRAPH_NAME = "dickens_graph"


def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d1']", namespace).text.strip('"')
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d3']", namespace).text
                if node.find("./data[@key='d3']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d5']", namespace).text)
                if edge.find("./data[@key='d5']", namespace) is not None
                else 1.0,
                "description": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d7']", namespace).text
                if edge.find("./data[@key='d7']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d8']", namespace).text
                if edge.find("./data[@key='d8']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        return data

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def insert_nodes_and_edges_to_falkordb(data):
    """Insert graph data into FalkorDB"""
    try:
        # Connect to FalkorDB
        db = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        graph = db.select_graph(FALKORDB_GRAPH_NAME)

        print(f"Connected to FalkorDB at {FALKORDB_HOST}:{FALKORDB_PORT}")
        print(f"Using graph: {FALKORDB_GRAPH_NAME}")

        nodes = data["nodes"]
        edges = data["edges"]

        print(f"Total nodes to insert: {len(nodes)}")
        print(f"Total edges to insert: {len(edges)}")

        # Insert nodes in batches
        for i in range(0, len(nodes), BATCH_SIZE_NODES):
            batch_nodes = nodes[i : i + BATCH_SIZE_NODES]

            # Build UNWIND query for batch insert
            query = """
            UNWIND $nodes AS node
            CREATE (n:Entity {
                entity_id: node.id,
                entity_type: node.entity_type,
                description: node.description,
                source_id: node.source_id
            })
            """

            graph.query(query, {"nodes": batch_nodes})
            print(f"Inserted nodes {i+1} to {min(i + BATCH_SIZE_NODES, len(nodes))}")

        # Insert edges in batches
        for i in range(0, len(edges), BATCH_SIZE_EDGES):
            batch_edges = edges[i : i + BATCH_SIZE_EDGES]

            # Build UNWIND query for batch insert
            query = """
            UNWIND $edges AS edge
            MATCH (source:Entity {entity_id: edge.source})
            MATCH (target:Entity {entity_id: edge.target})
            CREATE (source)-[r:DIRECTED {
                weight: edge.weight,
                description: edge.description,
                keywords: edge.keywords,
                source_id: edge.source_id
            }]-(target)
            """

            graph.query(query, {"edges": batch_edges})
            print(f"Inserted edges {i+1} to {min(i + BATCH_SIZE_EDGES, len(edges))}")

        print("Data insertion completed successfully!")

        # Print some statistics
        node_count_result = graph.query("MATCH (n:Entity) RETURN count(n) AS count")
        edge_count_result = graph.query(
            "MATCH ()-[r:DIRECTED]-() RETURN count(r) AS count"
        )

        node_count = (
            node_count_result.result_set[0][0] if node_count_result.result_set else 0
        )
        edge_count = (
            edge_count_result.result_set[0][0] if edge_count_result.result_set else 0
        )

        print("Final statistics:")
        print(f"- Nodes in database: {node_count}")
        print(f"- Edges in database: {edge_count}")

    except Exception as e:
        print(f"Error inserting data into FalkorDB: {e}")


def query_graph_data():
    """Query and display some sample data from FalkorDB"""
    try:
        # Connect to FalkorDB
        db = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        graph = db.select_graph(FALKORDB_GRAPH_NAME)

        print("\n=== Sample Graph Data ===")

        # Get some sample nodes
        query = (
            "MATCH (n:Entity) RETURN n.entity_id, n.entity_type, n.description LIMIT 5"
        )
        result = graph.query(query)

        print("\nSample Nodes:")
        if result.result_set:
            for record in result.result_set:
                print(f"- {record[0]} ({record[1]}): {record[2][:100]}...")

        # Get some sample edges
        query = """
        MATCH (a:Entity)-[r:DIRECTED]-(b:Entity)
        RETURN a.entity_id, b.entity_id, r.weight, r.description
        LIMIT 5
        """
        result = graph.query(query)

        print("\nSample Edges:")
        if result.result_set:
            for record in result.result_set:
                print(
                    f"- {record[0]} -> {record[1]} (weight: {record[2]}): {record[3][:100]}..."
                )

        # Get node degree statistics
        query = """
        MATCH (n:Entity)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) AS degree
        RETURN min(degree) AS min_degree, max(degree) AS max_degree, avg(degree) AS avg_degree
        """
        result = graph.query(query)

        print("\nNode Degree Statistics:")
        if result.result_set:
            record = result.result_set[0]
            print(f"- Min degree: {record[0]}")
            print(f"- Max degree: {record[1]}")
            print(f"- Avg degree: {record[2]:.2f}")

    except Exception as e:
        print(f"Error querying FalkorDB: {e}")


def clear_graph():
    """Clear all data from the FalkorDB graph"""
    try:
        db = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
        graph = db.select_graph(FALKORDB_GRAPH_NAME)

        # Delete all nodes and relationships
        graph.query("MATCH (n) DETACH DELETE n")
        print("Graph cleared successfully!")

    except Exception as e:
        print(f"Error clearing graph: {e}")


def main():
    xml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")

    if not os.path.exists(xml_file):
        print(
            f"Error: File {xml_file} not found. Please ensure the GraphML file exists."
        )
        print(
            "This file is typically generated by LightRAG after processing documents."
        )
        return

    print("FalkorDB Graph Visualization Example")
    print("====================================")
    print(f"Processing file: {xml_file}")
    print(f"FalkorDB connection: {FALKORDB_HOST}:{FALKORDB_PORT}")
    print(f"Graph name: {FALKORDB_GRAPH_NAME}")
    print()

    # Parse XML to JSON
    print("1. Parsing GraphML file...")
    data = xml_to_json(xml_file)
    if data is None:
        print("Failed to parse XML file.")
        return

    print(f"   Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

    # Ask user what to do
    while True:
        print("\nOptions:")
        print("1. Clear existing graph data")
        print("2. Insert data into FalkorDB")
        print("3. Query sample data")
        print("4. Exit")

        choice = input("\nSelect an option (1-4): ").strip()

        if choice == "1":
            print("\n2. Clearing existing graph data...")
            clear_graph()

        elif choice == "2":
            print("\n2. Inserting data into FalkorDB...")
            insert_nodes_and_edges_to_falkordb(data)

        elif choice == "3":
            print("\n3. Querying sample data...")
            query_graph_data()

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
