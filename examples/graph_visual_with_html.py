import pipmaster as pm

if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")

import networkx as nx
from pyvis.network import Network
import random

# Load the GraphML file
G = nx.read_graphml("./dickens/graph_chunk_entity_relation.graphml")

# Create a Pyvis network
net = Network(height="100vh", notebook=True)

# Convert NetworkX graph to Pyvis network
net.from_nx(G)


# Add colors and title to nodes
for node in net.nodes:
    node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    if "description" in node:
        node["title"] = node["description"]

# Style edges based on whether they're inferred
for edge in net.edges:
    # Check if this edge is from inference
    edge_data = G.get_edge_data(edge["from"], edge["to"])

    # Build descriptive title
    title_parts = []
    if edge_data and "description" in edge_data:
        title_parts.append(edge_data["description"])

    is_inferred = False
    inference_method = None
    confidence = None
    relationship_type = None

    if edge_data:
        is_inferred = edge_data.get("inferred", "false").lower() == "true"
        inference_method = edge_data.get("inference_method", "")
        confidence = edge_data.get("confidence", "")
        relationship_type = edge_data.get("relationship_type", "")

    # Style inferred edges differently
    if is_inferred:
        # Determine color based on relationship type
        if relationship_type == "competitor":
            edge["color"] = "rgba(255, 69, 58, 0.4)"  # Red, faded
            edge["label"] = "🥊 Competitor"
        elif relationship_type == "partnership":
            edge["color"] = "rgba(52, 199, 89, 0.4)"  # Green, faded
            edge["label"] = "🤝 Partner"
        elif relationship_type == "supply_chain":
            edge["color"] = "rgba(90, 200, 250, 0.4)"  # Blue, faded
            edge["label"] = "📦 Supply Chain"
        else:
            edge["color"] = "rgba(150, 150, 150, 0.4)"  # Gray, faded
            edge["label"] = "inferred"

        # Dashed line for inferred relationships
        edge["dashes"] = [5, 5]  # Dash pattern
        edge["width"] = 1  # Thinner line

        # Add inference info to title
        if confidence:
            title_parts.append(f"Confidence: {confidence}")
        if inference_method:
            title_parts.append(f"Method: {inference_method}")
        title_parts.append("(Inferred Relationship)")
    else:
        # Solid line for explicitly extracted relationships
        edge["color"] = "#666666"
        edge["width"] = 2
        title_parts.append("(Explicitly Extracted)")

    if title_parts:
        edge["title"] = "\n".join(title_parts)

# Save and display the network
net.show("knowledge_graph.html")

print("\n" + "="*70)
print("Knowledge Graph Visualization Generated")
print("="*70)
print("\nLegend:")
print("  • Solid lines (thick) = Explicitly extracted relationships")
print("  • Dashed lines (thin, faded) = Inferred relationships")
print("    - 🥊 Red = Competitor relationships")
print("    - 🤝 Green = Partnership relationships")
print("    - 📦 Blue = Supply chain relationships")
print("    - Gray = Generic inferred relationships")
print("\nHover over edges to see details including confidence scores.")
print("="*70)
