import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from pprint import pprint
#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio 
# nest_asyncio.apply() 
#########

WORKING_DIR = "./dickensTestEmbedcall"


# G = nx.read_graphml('./dickensTestEmbedcall/graph_chunk_entity_relation.graphml')
# nx.write_gexf(G, "graph_chunk_entity_relation.gefx")

import networkx as nx
from networkx_query import search_nodes, search_edges
G = nx.read_graphml('./dickensTestEmbedcall/graph_chunk_entity_relation.graphml')
query = {}  # Empty query matches all nodes
result = search_nodes(G, query)

# Extract node IDs from the result
node_ids = sorted([node for node in result])

print("All node IDs in the graph:")
pprint(node_ids)
raise Exception


# raise Exception

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

with open("./book.txt") as f:
    rag.insert(f.read())

# Perform naive search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# Perform local search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# Perform global search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# Perform hybrid search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))