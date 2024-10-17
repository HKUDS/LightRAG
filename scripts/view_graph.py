import networkx as nx
import os
import plotly.graph_objects as go
WORKING_DIR = "./ragtest"
# Find all GraphML files in the directory and sort them alphabetically
graphml_files = [f for f in os.listdir(WORKING_DIR) if f.endswith('.graphml')]
graphml_files.sort()
def create_hover_text(attributes):
    return '<br>'.join([f'{k}: {v}' for k, v in attributes.items() if k != 'id'])
# Loop through each GraphML file
for graphml_file in graphml_files:
    # Construct the full path to the GraphML file
    graphml_path = os.path.join(WORKING_DIR, graphml_file)
    
    # Load the graph
    G = nx.read_graphml(graphml_path)
    
    # Create a layout for the graph
    pos = nx.spring_layout(G)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(create_hover_text(G.nodes[node]))
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in G.nodes()],  # Use node IDs as labels
        hovertext=node_text,
        marker=dict(size=20, color='lightblue'),
        textposition='top center'
    )
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    edge_hover_x = []
    edge_hover_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_hover_x.append((x0 + x1) / 2)
        edge_hover_y.append((y0 + y1) / 2)
        edge_text.append(create_hover_text(G.edges[edge]))
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        mode='lines'
    )
    edge_hover_trace = go.Scatter(
        x=edge_hover_x, y=edge_hover_y,
        mode='markers',
        marker=dict(size=0.5, color='rgba(0,0,0,0)'),
        hoverinfo='text',
        hovertext=edge_text,
        hoverlabel=dict(bgcolor='white'),
    )
    # Create the figure
    fig = go.Figure(data=[edge_trace, edge_hover_trace, node_trace],
                    layout=go.Layout(
                        title=graphml_file,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    # Save the figure as an interactive HTML file
    output_file = f'{os.path.splitext(graphml_file)[0]}_interactive.html'
    fig.write_html(os.path.join(WORKING_DIR, output_file))
    print(f"Interactive graph saved as {output_file}")