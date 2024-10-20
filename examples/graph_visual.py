import networkx as nx
from pyvis.network import Network

# Load the GraphML file
G = nx.read_graphml('./dickens/graph_chunk_entity_relation.graphml')

# Create a Pyvis network
net = Network(notebook=True)

# Convert NetworkX graph to Pyvis network
net.from_nx(G)

# Save and display the network
net.show('knowledge_graph.html')