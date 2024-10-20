import networkx as nx
from pyvis.network import Network
import random

# Load the GraphML file
G = nx.read_graphml('./dickens/graph_chunk_entity_relation.graphml')

# Create a Pyvis network
net = Network(notebook=True)

# Convert NetworkX graph to Pyvis network
net.from_nx(G)

# Add colors to nodes
for node in net.nodes:
    node['color'] = "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Save and display the network
net.show('knowledge_graph.html')