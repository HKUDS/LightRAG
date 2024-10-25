import os
import streamlit as st
import streamlit.components.v1 as components
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from pyvis.network import Network
import tempfile
import networkx as nx
import pandas as pd
import PyPDF2
import io

# Page configuration
st.set_page_config(page_title="LightRAG Demo", page_icon="🔍", layout="wide")

# Define working directory at the top level
WORKING_DIR = ".idea/ragcache"
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete
    )
    st.session_state.messages = []
    st.session_state.documents = []

# Sidebar
with st.sidebar:
    st.title("🔍 LightRAG")
    
    # Configuration section
    st.header("Configuration")
    
    # OpenAI API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to use the chat interface"
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    model_name = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0,  # Set default to first option (gpt-4o-mini)
        help="Select the model for chat responses"
    )
    
    # Update model if changed
    MODEL_OPTIONS = {
        "gpt-4o-mini": gpt_4o_mini_complete,
        "gpt-4": gpt_4o_complete,
        # Add other model mappings as needed
    }
    
    if model_name in MODEL_OPTIONS:
        st.session_state.rag.llm_model_func = MODEL_OPTIONS[model_name]
    
    # Chat Controls moved under Configuration
    st.subheader("Chat Controls")
    
    # Query mode selection
    query_mode = st.selectbox(
        "Query Mode",
        ["hybrid", "naive", "local", "global"],
        help="Select the search mode for document retrieval"
    )
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Chat info
    with st.expander("ℹ️ Chat Information"):
        st.markdown("""
        - Use natural language to ask questions
        - Reference specific documents
        - Ask for comparisons or summaries
        - Request specific details
        """)
    
    st.markdown("---")
        
    # Knowledge Graph section
    st.header("Knowledge Base")
    
    # Document input
    with st.expander("Add Document", expanded=True):
        tab1, tab2 = st.tabs(["Upload", "Paste"])
        
        with tab1:
            uploaded_file = st.file_uploader("Upload file", type=['txt', 'md', 'pdf'])
            if uploaded_file:
                try:
                    if uploaded_file.type == "application/pdf":
                        # Process PDF file
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                        content = ""
                        for page in pdf_reader.pages:
                            content += page.extract_text() + "\n"
                    else:
                        # Process text files as before
                        content = uploaded_file.read().decode()
                    
                    with st.spinner("Processing..."):
                        st.session_state.rag.insert(content)
                        if uploaded_file.name not in st.session_state.documents:
                            st.session_state.documents.append(uploaded_file.name)
                    st.success("Processed!")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        with tab2:
            text_input = st.text_area("Paste text:", height=100)
            if st.button("Process") and text_input:
                with st.spinner("Processing..."):
                    st.session_state.rag.insert(text_input)
                    new_doc_name = f"Text {len(st.session_state.documents) + 1}"
                    if new_doc_name not in st.session_state.documents:
                        st.session_state.documents.append(new_doc_name)
                st.success("Processed!")
    
    # Table of Contents moved into expander
    with st.expander("📚 Documents", expanded=False):
        if st.session_state.documents:
            for idx, doc in enumerate(st.session_state.documents, 1):
                st.markdown(f"{idx}. {doc}")
        else:
            st.info("No documents added yet. Use 'Add Document' above to get started.")
    
    # Add Knowledge Graph Statistics
    graphml_path = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    if os.path.exists(graphml_path):
        st.markdown("---")
        st.subheader("KGraph Statistics")
        
        try:
            # Load GraphML using networkx
            G = nx.read_graphml(graphml_path)
            
            # Create metrics container
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                entity_count = G.number_of_nodes()
                st.metric(
                    label="Entities",
                    value=entity_count,
                    help="Total number of nodes in the graph"
                )
            
            with stats_col2:
                relation_count = G.number_of_edges()
                st.metric(
                    label="Relations",
                    value=relation_count,
                    help="Total number of connections in the graph"
                )
                            
        except Exception as e:
            st.error(f"Error loading graph statistics: {str(e)}")

# Main container
# Knowledge Graph Visualization at the top
with st.expander("View Knowledge Graph", expanded=True):
    refresh_graph = st.button("🔄 Refresh Graph")
    
    # Check for existing graph data
    graphml_path = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    has_graph_data = os.path.exists(graphml_path)
    
    if has_graph_data or st.session_state.documents:
        if refresh_graph:
            with st.spinner("Updating graph..."):
                try:
                    if os.path.exists(graphml_path):
                        # Create network
                        net = Network(
                            height="400px",
                            width="100%",
                            bgcolor="#ffffff",
                            font_color="black",
                            directed=True
                        )
                        
                        try:
                            # Load GraphML using networkx
                            G = nx.read_graphml(graphml_path)
                            
                            # Add nodes with different colors based on type
                            node_colors = {
                                'chunk': '#97C2FC',  # Light blue
                                'entity': '#FFD700',  # Gold
                                'relation': '#FF6B6B'  # Coral
                            }
                            
                            node_shapes = {
                                'chunk': 'dot',
                                'entity': 'diamond',
                                'relation': 'triangle'
                            }
                            
                            # Add nodes
                            for node, data in G.nodes(data=True):
                                try:
                                    node_type = data.get('type', 'entity')
                                    net.add_node(
                                        node,
                                        label=str(node)[:20],  # Truncate long labels
                                        color=node_colors.get(node_type, '#97C2FC'),
                                        title=f"Type: {node_type}\nID: {node}",
                                        shape=node_shapes.get(node_type, 'dot'),
                                        size=20
                                    )
                                except Exception as e:
                                    st.warning(f"Error adding node: {str(e)}")
                                    continue
                            
                            # Add edges
                            for source, target, data in G.edges(data=True):
                                try:
                                    net.add_edge(
                                        source, 
                                        target,
                                        title=data.get('type', 'connects'),
                                        arrows='to'  # Add arrows for direction
                                    )
                                except Exception as e:
                                    st.warning(f"Error adding edge: {str(e)}")
                                    continue
                            
                            # Configure physics with more stable settings
                            net.set_options("""{
                                "physics": {
                                    "enabled": true,
                                    "stabilization": {
                                        "enabled": true,
                                        "iterations": 100,
                                        "updateInterval": 50
                                    },
                                    "barnesHut": {
                                        "gravitationalConstant": -2000,
                                        "springLength": 150,
                                        "springConstant": 0.04,
                                        "damping": 0.09
                                    }
                                },
                                "interaction": {
                                    "navigationButtons": true,
                                    "tooltipDelay": 100,
                                    "hover": true,
                                    "multiselect": true,
                                    "dragNodes": true
                                },
                                "edges": {
                                    "smooth": {
                                        "type": "continuous",
                                        "forceDirection": "none"
                                    }
                                }
                            }""")
                            
                            # Save and display
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
                                net.save_graph(tmp.name)
                                with open(tmp.name, 'r', encoding='utf-8') as f:
                                    components.html(f.read(), height=400)
                                os.unlink(tmp.name)
                            
                            # Update statistics based on actual graph
                            st.session_state.entity_count = G.number_of_nodes()
                            st.session_state.relation_count = G.number_of_edges()
                            st.session_state.node_types = {
                                node_type: sum(1 for _, data in G.nodes(data=True) 
                                             if data.get('type') == node_type)
                                for node_type in node_colors.keys()
                            }
                            st.success("Graph updated!")
                            
                        except Exception as e:
                            st.error(f"Error loading graph: {str(e)}")
                    else:
                        st.warning("No graph data found. Try processing some documents first.")
                except Exception as e:
                    st.error(f"Error refreshing graph: {str(e)}")
        else:
            st.info("Click 'Refresh Graph' to visualize the knowledge graph")
    else:
        st.info("No graph data available. Add documents to create a knowledge graph.")

# Chat columns below graph - modified to single column
st.header("LightRAG Chat Interface")

# Enhanced Chat Interface - now full width
# Custom CSS for the chat container
st.markdown("""
    <style>
        .stChatFloatingInputContainer {
            bottom: 20px;
        }
        div[data-testid="stChatMessageContainer"] {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 5px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Display chat history with timestamps
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.markdown(message["content"])
        with col2:
            st.caption(message.get("timestamp", ""))
    
# Chat input
if prompt := st.chat_input("Ask about your documents...", key="chat_input"):
    # Add timestamp to message with timezone
    timestamp = pd.Timestamp.now().strftime("%H:%M:%S %Z")
    
    # Add user message with timestamp
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.markdown(prompt)
        with col2:
            st.caption(timestamp)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_timestamp = pd.Timestamp.now().strftime("%H:%M:%S %Z")
            response = st.session_state.rag.query(
                prompt,
                param=QueryParam(mode=query_mode)
            )
            
            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.markdown(response)
            with col2:
                st.caption(response_timestamp)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": response_timestamp
            })
