"""
1.
This demo uses structured output from Ollama's LLM, which requires Ollama's version >= 0.9.0. 
If you haven't installed Ollama, please refer to the official documentation for installation.

2.
If your server can't connect to HuggingFace, you can set the environment variable HF_ENDPOINT to a mirror site as follows:
    export HF_ENDPOINT=https://hf-mirror.com  # Linux/macOS
    set HF_ENDPOINT=https://hf-mirror.com  # Windows
"""
from pyvis.network import Network
from openai import OpenAI
from fuzzywuzzy import fuzz
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from collections import defaultdict
from collections import deque
from lightrag.prompt import PROMPTS
import requests, json, sys, logging, os
import networkx as nx

############################    Settings   #####################################
Set = {
    "WORK_DIC": "Your working directory",  # Working directory for graph files and output
    "LLM_Model": "Your LLM model",  # Language model used for processing
    
    # Batch processing parameters
    "max_common_list_size": 5,  # Max isolated nodes to process together in edge creation
    "split_size_create_edge": 40,  # Target batch size for edge creation clustering
    "split_size_merge_delete": 30,  # Target batch size for merge/delete operations
    "max_create_edge_times": 3,  # Max retries for edge creation when LLM output is inconsistent
    "format_storage_lenth": 5,  # Number of merged/summary nodes to store from LLM response
    "need_clean_isolated_nodes": True,  # Whether to clean isolated nodes after processing
    "ollama_host": "http://localhost:11434"  # Local Ollama service URL
}

rag = LightRAG(
    working_dir = Set["WORK_DIC"],
    llm_model_func = ollama_model_complete,
    # llm_model_name="deepseek-r1:70b",
    llm_model_name = Set["LLM_Model"],
    llm_model_max_async = 4,
    llm_model_max_token_size = 32768,
    llm_model_kwargs={"host": Set["ollama_host"], "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="bge-m3", host=Set["ollama_host"]
        ),
    ),
    chunk_token_size=8192,           # 分块大小
    chunk_overlap_token_size=4096,   # 重叠避免信息断裂
)

merge_keys = [f"merge_{i}" for i in range(1, Set["format_storage_lenth"] + 1)]
summary_keys = [f"summary_{i}" for i in range(1, Set["format_storage_lenth"] + 1)]
goal_merge = """
---Objective---
You are a knowledge point filtering and merging assistant. You must strictly follow the given output format.
Merge duplicate knowledge points. For example, "Greedy Algorithm" and "GREEDY ALGORITHM" should be merged as 'Greedy Algorithm'.
Merge similar knowledge points under the same category into a summarized knowledge point, making it suitable as a major node in a knowledge graph.

---Format Requirements---
Use 'merge_1': ['Point_1', 'Point_2', 'Point_3', ...] to indicate merged knowledge points. Here, 'Summary' is the summarized knowledge point, and 'Point_1', 'Point_2', 'Point_3', ... are the selected input points being merged.
Use 'summary_1': 'sum_merge_1' to denote the summarized knowledge point, where 'sum_merge_1' is your summarized point for 'merge_1'.

---Important Notes---
The number of summary_x must match the number of merge_x, and they must correspond one-to-one.
Nodes within merge_x must not be duplicated. Nodes across different merge_x must not be duplicated.
Nodes in merge_x must be selected from the input only; do not create new nodes.
Summarized knowledge points should be concise and must not end with punctuation. For example, 'Linear Time Algorithm' and 'Equilibrium Condition' are concise and correct.
If no points require merging later, output an empty set. For example, 'merge_8': [] indicates no points were merged.
Only merge identical concepts. Do not merge different concepts. For instance, 'Space Complexity', 'Time Complexity', and 'Computational Complexity' are distinct concepts and cannot be merged into 'Complexity'.
merge_x must contain at least 2 output elements; never output a single point alone.
"""
Format_Merge = {
    "model": Set["LLM_Model"],
    "messages": [
        {"role":"system","content": ""},
        {"role": "user","content": ""}
    ],
    "stream": False,
    "format": {
        "type": "object",
        "properties": {
            **{key: {
                "type": "array",
                "items": {
                    "type": "string"
                }
            } for key in merge_keys
            },

            **{sum: {
                "type": "string",
            } for sum in summary_keys
            },
        },
        "required": merge_keys + summary_keys
    }
}
goal_delete = """
---Objective---
You are a non-knowledge point content deletion assistant, and you must strictly follow the given output format.
Delete non-knowledge point content, such as 'Lemma XXX', 'Theorem XXX', 'stdio.h', "*", "/=", "y", etc.

---Format Requirements---
Use 'delete': ['Point_1', 'Point_2', 'Point_3', ...] to indicate the non-knowledge point content you have deleted.

---Important Notes---
If there is no non-knowledge point content to delete, output an empty set: 'delete': [].
Do not delete knowledge points: For example, "amortized weight-balanced trees", "Lowest Common Ancestor (LCA)", "binary search trees" are knowledge points and must not be deleted.
"""
Format_Delete = {
    "model": Set["LLM_Model"],
    "messages": [
        {"role":"system","content": ""},
        {"role": "user","content": ""}
    ],
    "stream": False,
    "format": {
        "type": "object",
        "properties": {
            "delete": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            }
        },
        "required": ["delete"]
    }
}   
goal_create = """
---Objective---
You are a Knowledge Graph Synthesis Assistant tasked with identifying relationships between specified knowledge points. You must strictly adhere to the given output format.

---Format Requirements---
The elements in "node_1" and "node_2" must have identical counts.
"node_1": List of isolated knowledge points selected for association.
"node_2": List of set knowledge points selected for association.
"relation": Description of the relationship between corresponding node_1 and node_2 elements.
"keywords": Concise summary of the relationship (maximum 3 terms).
"weight": Numeric strength of the relationship (integer 5-30).

---Important Notes---
ONLY establish relationships between isolated knowledge points and set knowledge points.
DO NOT create relationships within isolated knowledge points.
DO NOT create relationships within set knowledge points.
Ensure every isolated knowledge point is included in relationships.
Output strictly in the specified format without additional explanations.
Elements in node_1, node_2, relation, keywords, and weight must be one-to-one correlated and equal in list length.

---Example---
Input:
Isolated Points: ['Node Count', 'Equivalence Problem']  
Set Points: ['Disjoint Set', 'Dynamic Equivalence Problem', 'Find Operation', 'Union Operation', 'Equivalence Class', 'Path Compression', 'Union by Rank', 'Rank', 'Root', 'Union by Size', 'Union-Find Heuristic']  
Output:
{  
  "node_1": ["Node Count", "Node Count", "Node Count", "Equivalence Problem", "Equivalence Problem", "Equivalence Problem"],  
  "node_2": ["Disjoint Set", "Dynamic Equivalence Problem", "Find Operation", "Equivalence Class", "Path Compression", "Union by Rank"],  
  "relation": [  
    "Node Count is a key parameter for disjoint set size, optimizing union-find operations",  
    "Tracking Node Count in dynamic equivalence problems evaluates algorithmic complexity",  
    "Find Operation uses Node Count to accelerate path compression depth reduction",  
    "Equivalence Problem defines equivalence classes via relation-satisfying elements",  
    "Path Compression boosts efficiency by optimizing Find operations for equivalence",  
    "Union by Rank leverages Node Count to merge smaller sets into larger ones"  
  ],  
  "keywords": [  
    "Set size optimization",   
    "Complexity evaluation",   
    "Path compression acceleration",
    "Equivalence class definition",   
    "Find operation efficiency",   
    "Merge strategy"  
  ],  
  "weight": [20, 18, 15, 25, 17, 14]  
}    



"""
Format_Create_Edges = {
    "model": Set["LLM_Model"],
    "messages": [
        {"role":"system","content": ""},
        {"role": "user","content": ""}
    ],
    "stream": False,
    "format": {
        "type": "object",
        "properties": {
            "node_1": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "node_2": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "relation": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "keywords": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "weight": {
                "type": "array",
                "items": {
                    "type": "integer"
                }
            }
        },
        "required": ["node_1", "node_2", "relation", "keywords", "weight"]
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
global create_edge_time
###############################################################################

#############################    Functions   ##################################
def case_sensitive_ratio(s1, s2):
    return fuzz.ratio(s1, s2)

# Clustering algorithm
class SemanticClusterBatcher:
    def __init__(self, TARGET_SIZE, MAX_SIZE, MIN_SIZE):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.font_path = 'simhei.ttf'
        self.TARGET_SIZE = TARGET_SIZE  # Target batch size
        self.MAX_SIZE = MAX_SIZE        # Maximum batch size
        self.MIN_SIZE = MIN_SIZE        # Minimum batch size

    def _validate_input(self, nodes):
        if len(nodes) == 0:
            raise ValueError("Input node list cannot be empty")
        if not all(isinstance(node, str) for node in nodes):
            raise TypeError("All nodes must be strings")

    def _hierarchical_clustering(self, embeddings, target_size):
        """Hierarchical clustering with dynamic cluster count adjustment"""
        try:
            Z = linkage(embeddings, method='ward', metric='euclidean')
            target_clusters = max(1, len(embeddings) // target_size)
            return fcluster(Z, t=target_clusters, criterion='maxclust')
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {str(e)}")
            raise

    def _split_large_clusters(self, clusters, embeddings, original_nodes, MAX_CLUSTER_SIZE, MIN_BATCH_SIZE):
        """Split oversized clusters with element integrity verification"""
        final_clusters = []
        for cluster in clusters:
            original_count = len(cluster)
            if original_count <= MAX_CLUSTER_SIZE:
                final_clusters.append(cluster)
                continue

            try:
                # Use node indices instead of direct lookup (avoid duplicate node issues)
                indices = [i for i, n in enumerate(original_nodes) if n in cluster]
                sub_embeddings = embeddings[indices]
                
                # Ensure indices match cluster elements
                assert len(indices) == original_count, "Indices don't match cluster elements"
                
                n_sub = max(2, original_count//MIN_BATCH_SIZE + 1)
                kmeans = KMeans(n_clusters=n_sub, n_init=10)
                sub_labels = kmeans.fit_predict(sub_embeddings)
                
                # Rebuild subclusters in original order
                sub_clusters = defaultdict(list)
                for node, label in zip(cluster, sub_labels):
                    sub_clusters[label].append(node)
                
                # Verify split integrity
                split_total = sum(len(v) for v in sub_clusters.values())
                if split_total != original_count:
                    raise ValueError(f"Element loss: original {original_count} after split {split_total}")
                
                final_clusters.extend(sub_clusters.values())
            except Exception as e:
                logger.warning(f"Sub-clustering failed, keeping original cluster: {str(e)}")
                final_clusters.append(cluster)
        return final_clusters

    def _optimize_batches(self, clusters, MAX_CLUSTER_SIZE, MIN_BATCH_SIZE):
        """Optimize batch grouping using dynamic queue"""
        batches = []
        current_batch = []
        cluster_queue = deque(sorted(clusters, key=len, reverse=True))  # Sort by size
        
        while cluster_queue:
            cluster = cluster_queue.popleft()
            
            # Direct addition condition
            if len(current_batch) + len(cluster) <= self.TARGET_SIZE:
                current_batch.extend(cluster)
                continue
                
            # Split needed case
            remaining_space = self.TARGET_SIZE - len(current_batch)
            
            if remaining_space >= self.MIN_SIZE and len(cluster) > remaining_space:
                # Cut and requeue remainder
                current_batch.extend(cluster[:remaining_space])
                cluster_queue.appendleft(cluster[remaining_space:])
            else:
                # Start new batch
                batches.append(current_batch)
                current_batch = list(cluster)
        
        # Add final batch
        if current_batch:
            batches.append(current_batch)
        
        # Final integrity check
        input_count = sum(len(c) for c in clusters)
        output_count = sum(len(b) for b in batches)
        if input_count != output_count:
            error_msg = f"Critical error: element count changed ({input_count}→{output_count})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return batches

    def cluster_and_batch(self, nodes):
        """Main processing flow"""
        self._validate_input(nodes)
        
        # Semantic embedding
        logger.info("Generating semantic embeddings...")
        embeddings = self.model.encode(nodes)
        
        # Hierarchical clustering
        logger.info("Performing hierarchical clustering...")
        cluster_ids = self._hierarchical_clustering(embeddings, self.TARGET_SIZE)
        
        # Organize clustering results
        cluster_dict = defaultdict(list)
        for node, cid in zip(nodes, cluster_ids):
            cluster_dict[cid].append(node)
        initial_clusters = list(cluster_dict.values())
        
        # Split oversized clusters
        logger.info("Optimizing cluster sizes...")
        optimized_clusters = self._split_large_clusters(initial_clusters, embeddings, nodes, self.MAX_SIZE, self.MIN_SIZE)
        
        # Generate final batches
        logger.info("Creating final batches...")
        batches = self._optimize_batches(optimized_clusters, self.MAX_SIZE, self.MIN_SIZE)
        
        return batches

# Cluster nodes by label
def classify_nodes_labels(TARGET_SIZE, MAX_SIZE, MIN_SIZE):
    print("################################################################################")
    print("                   ####  CLASSIFY NODES - LABEL CLUSTERING   ####")
    print("################################################################################")
    # Initialize cluster processor
    batcher_labels = SemanticClusterBatcher(TARGET_SIZE, MAX_SIZE, MIN_SIZE)
    nodes_batches = []
    G = nx.read_graphml(Set["WORK_DIC"] + "/" + "graph_chunk_entity_relation.graphml")
    net = Network(height="100vh", notebook=True, select_menu=True)
    net.from_nx(G)
    classified_data = {}
    for node in net.nodes:
        if 'entity_type' in node:
            entity_type = node['entity_type']
            # Initialize list if entity_type not present
            if entity_type not in classified_data:
                classified_data[entity_type] = []
            # Add current dict to entity_type list
            classified_data[entity_type].append(node['label'])
        else:
            print("Node without entity_type:", node)

    short_KP = []
    for entity_type, items in classified_data.items():
        if(len(items) < MAX_SIZE):
            if(len(items) > 10):
                nodes_batches.append(items)
            else:
                short_KP.append(items)
        else:
            splited_long = batcher_labels.cluster_and_batch(items)
            print("Oversized cluster:\n" + str(items) + "\n")
            for splited in splited_long:
                nodes_batches.append(splited)
                print("       —— Split cluster:\n" + str(splited) + "\n")

    combined_short_KP = [item for sublist in short_KP for item in sublist]
    if(len(combined_short_KP) < MAX_SIZE):
        nodes_batches.append(combined_short_KP)
        print("Scattered knowledge points < max cluster size, added directly")
    else:
        print("Scattered knowledge points > max cluster size, splitting")
        batcher_labels_isolated = SemanticClusterBatcher(TARGET_SIZE, MAX_SIZE, MIN_SIZE)
        splited_long_isolated = batcher_labels_isolated.cluster_and_batch(combined_short_KP)
        print("        Oversized cluster:\n" + str(splited_long_isolated) + "\n")
        for splited_isolated in splited_long_isolated:
            print("             —— Split cluster:\n" + str(splited_isolated) + "\n")
            nodes_batches.append(splited_isolated)
    
    print("Batch size distribution:")
    for batches_index in range(len(nodes_batches)):
        print(str(len(nodes_batches[batches_index])) + "[ " + str(batches_index + 1) + " / " + str(len(nodes_batches)) + " ]")
    return nodes_batches, G

# Cluster nodes by similarity
def classify_nodes_whole(TARGET_SIZE, MAX_SIZE, MIN_SIZE):
    print("################################################################################")
    print("                       ####  CLASSIFY NODES - SIMILARITY CLUSTERING   ####")
    print("################################################################################")
    # Collections
    nodes_batches = []
    original_nodes = []
    # Load GraphML file
    G = nx.read_graphml(Set["WORK_DIC"] + "/" + "graph_chunk_entity_relation.graphml")
    for node in G.nodes:
        mapping = { node : node.strip('"')}
        G = nx.relabel_nodes(G, mapping)
    net = Network(height="100vh", notebook=True, select_menu=True)
    net.from_nx(G)
    for node in net.nodes:
        original_nodes.append(node["label"])
    
    # Initialize processor
    batcher_whole = SemanticClusterBatcher(TARGET_SIZE, MAX_SIZE, MIN_SIZE)
    try:
        # Perform clustering and batching
        nodes_batches = batcher_whole.cluster_and_batch(original_nodes)
        # Output results
        print(f"\nTotal batches: {len(nodes_batches)}")
        print("Batch size distribution:")
        for batches_index in range(len(nodes_batches)):
            print(str(len(nodes_batches[batches_index])) + "[ " + str(batches_index + 1) + " / " + str(len(nodes_batches)) + " ]")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
    return nodes_batches, G

# Merge/delete node executor
def merge_delete_nodes(merged_nodes_each_batch, summmaried_nodes_each_batch, deleted_nodes_each_batch, this_batch):
    # Merge nodes in KG-RAG
    if merged_nodes_each_batch:
        merge_lenth = min(len(merged_nodes_each_batch), len(summmaried_nodes_each_batch))
        for merge_index in range(0, merge_lenth):
            if(len(merged_nodes_each_batch[merge_index]) == 1):
                print(f"    —— Single output, skip merge: {str((merged_nodes_each_batch[merge_index]))}")
                continue
            else:
                find_node_this_slice = []
                this_slice = merged_nodes_each_batch[merge_index]
                for quote_index in range(0,len(this_slice)):
                    merged_node_each_slice = this_slice[quote_index]
                    found_node = process.extractOne(merged_node_each_slice, this_batch)
                    if found_node is None:
                        print(f"    —— LLM hallucination: {str(merged_node_each_slice)}")
                        continue
                    elif(found_node[1] >= 95):
                        print(f"          —— Original: {merged_node_each_slice} Found: {str(found_node[0])} Similarity: {str(found_node[1])}")
                        find_node_this_slice.append(found_node[0])
                        this_batch.remove(found_node[0])
                    else:
                        print(f"    —— No matching knowledge point (merge): {str(merged_node_each_slice)} — {str(found_node[0])} Similarity: {str(found_node[1])}")
                if(len(find_node_this_slice) > 1):
                    input_source_entities = find_node_this_slice
                    input_target_entity = summmaried_nodes_each_batch[merge_index]
                    print(f"    Merging nodes: {input_target_entity} : {str(input_source_entities)} \n")
                    rag.merge_entities(source_entities = input_source_entities, target_entity = input_target_entity)
                else:
                    print(f"    —— Insufficient merge nodes, skipping: {str(merged_node_each_slice)}")
        print("-----------------------------------------------------------------------------")

    else:
        print("No nodes to merge")
        print("-----------------------------------------------------------------------------")

    # Delete nodes in KG-RAG
    if deleted_nodes_each_batch:
        for deleted_index in range(len(deleted_nodes_each_batch)):
            found_deleted_node = process.extractOne(deleted_nodes_each_batch[deleted_index], this_batch)
            if found_deleted_node is None:
                print(f"    —— LLM hallucination: {str(deleted_nodes_each_batch[deleted_index])}")
                continue
            elif(found_deleted_node[1] >= 95):
                input_deleted_entity = found_deleted_node[0]
                rag.delete_by_entity(input_deleted_entity)
                print(f"    Deleted node: {input_deleted_entity}")
                this_batch.remove(input_deleted_entity)
            else:
                print(f"    —— No matching knowledge point (delete): {str(deleted_nodes_each_batch[deleted_index])} — {str(found_deleted_node[0])} Similarity: {str(found_deleted_node[1])}")
        print("-----------------------------------------------------------------------------")
    else:
        print("No nodes to delete")
        print("-----------------------------------------------------------------------------")
    
    return this_batch
    
# Merge/delete node controller
def merge_delete_control(nodes_batches, mode):
    for i in range(0, len(nodes_batches)):
        print("#############################################################################")
        print(f"      ####  PRUNING NODES [ {str(i + 1)}/{str(len(nodes_batches))} ] SIZE [ {str(len(nodes_batches[i]))} ] [ {str(mode)} ]  ####")
        print("#############################################################################")
        input_text = str(nodes_batches[i])
        print("Input batch:\n" + input_text)
        print("-----------------------------------------------------------------------------")

        # Merge nodes via ollama
        for message in Format_Merge["messages"]:
            if message["role"] == "system":
                message["content"] = goal_merge
            if message["role"] == "user":
                message["content"] = input_text

        json_data_merge = json.dumps(Format_Merge)
        response_merge = requests.post(Set["ollama_host"] + "/api/chat", data=json_data_merge, headers={'Content-Type': 'application/json'})
        print("Original response (merge/json):\n" + str(response_merge.text))
        response_merge_str = json.loads(response_merge.text)
        res_merge = json.loads(response_merge_str["message"]["content"])
        print("Original response (merge/content):\n" + str(res_merge))
        print("-----------------------------------------------------------------------------")
        res_merged = list(res_merge.items())[:Set["format_storage_lenth"]]
        res_summary = list(res_merge.items())[Set["format_storage_lenth"]:Set["format_storage_lenth"]*2]
        res_merge_nonempty = [v for k, v in res_merged if v]
        res_summmary_nonempty = [v for k, v in res_summary if v]
        while(len(res_merge_nonempty) != len(res_merge_nonempty)):
            print("Merged/summary length mismatch, retrying!")
            json_data_merge = json.dumps(Format_Merge)
            response_merge = requests.post(Set["ollama_host"] + "/api/chat", data=json_data_merge, headers={'Content-Type': 'application/json'})
            response_merge_str = json.loads(response_merge.text)
            res_merge = json.loads(response_merge_str["message"]["content"])
            print("Original response (merge):\n" + str(res_merge))
            print("-----------------------------------------------------------------------------")
            res_merged = list(res_merge.items())[:Set["format_storage_lenth"]]
            res_summary = list(res_merge.items())[Set["format_storage_lenth"]:Set["format_storage_lenth"]*2]
            res_merge_nonempty = [v for k, v in res_merged if v]
            res_summmary_nonempty = [v for k, v in res_summary if v]

        # Delete nodes via ollama
        for message in Format_Delete["messages"]:
            if message["role"] == "system":
                message["content"] = goal_delete
            if message["role"] == "user":
                message["content"] = input_text

        json_data_delete = json.dumps(Format_Delete)
        response_delete = requests.post(Set["ollama_host"] + "/api/chat", data=json_data_delete, headers={'Content-Type': 'application/json'})
        print("Original response (delete/json):\n" + str(response_delete.text))
        response_delete_str = json.loads(response_delete.text)
        res_delete = json.loads(response_delete_str["message"]["content"])
        print("Original response (delete/content):\n" + str(res_delete))
        print("-----------------------------------------------------------------------------")
        res_deleted = res_delete["delete"]

        # Execute merge/delete operations
        merge_delete_nodes(res_merge_nonempty, res_summmary_nonempty, res_deleted, nodes_batches[i])
        

    print("################################################################################")
    print(f"                 ####  NODE PRUNING COMPLETE [ {str(mode)} ]  ####")
    print("################################################################################")

# Check for isolated nodes
def check_isolated_nodes(this_batch, G):
    single_nodes =[]
    filtered_nodes = []
    for node in this_batch:
        if(G.degree[node] == 1):
            neighbor = G.neighbors(node)
            neighbor_degree = [item[1] for item in G.degree(neighbor)]
            if(neighbor_degree[0] <= 1):
                filtered_nodes.append([node])
        elif(G.degree[node] == 0):
            single_nodes.append(node)
    single_nodes = [item for sublist in single_nodes for item in sublist]
    filtered_nodes.append(single_nodes)
    isolated_nodes = [item for sublist in filtered_nodes for item in sublist]

    common_elements = set(isolated_nodes).intersection(this_batch)
    common_elements_list = list(common_elements)

    return common_elements_list

# Create edges for isolated nodes
def create_edge(items, common_elements_list):
    create_edge_time = 0
    print(f"This batch: \n{items}")
    common_elements = set(common_elements_list)
    handled_items = []
    
    handled_items.append(common_elements_list)
    handled_items.append([item for item in items if item not in common_elements])
    print("----------------------------------------------------------------")

    input_text = "Isolated knowledge points:" + str(handled_items[0]) + "\n" + "Group knowledge points:" + str(handled_items[1])
    print("Input text:\n" + input_text)
    print("-----------------------------------------------------------------------------")
    # Use ollama to process batch
    for message in Format_Create_Edges["messages"]:
        if message["role"] == "system":
            message["content"] = goal_create
        if message["role"] == "user":
            message["content"] = input_text

    json_data = json.dumps(Format_Create_Edges)
    response = requests.post(Set["ollama_host"] + "/api/chat", data=json_data, headers={'Content-Type': 'application/json'})
    response_str = json.loads(response.text)
    res = json.loads(response_str["message"]["content"])
    print("Original response:\n" + str(res))
    print("-----------------------------------------------------------------------------")
    all_equal = all(len(lst) == len(res["node_1"]) for lst in [res["node_2"], res["relation"], res["keywords"], res["weight"]])

    while(all_equal == False and create_edge_time < 5):
        print("LLM output length mismatch, retrying!")
        json_data = json.dumps(Format_Create_Edges)
        response = requests.post(Set["ollama_host"] + "/api/chat", data=json_data, headers={'Content-Type': 'application/json'})
        response_str = json.loads(response.text)
        res = json.loads(response_str["message"]["content"])
        print("Original response:\n" + str(res))
        print("-----------------------------------------------------------------------------")
        all_equal = all(len(lst) == len(res["node_1"]) for lst in [res["node_2"], res["relation"], res["keywords"], res["weight"]])
        create_edge_time +=1

    # Create relations between entities
    for i in range(len(res["node_1"])):
        rag.create_relation(res["node_1"][i], res["node_2"][i], {
            "description": res["relation"][i],
            "keywords": res["keywords"][i],
            "weight": res["weight"][i]
        })


# Edge creation controller
def create_edge_control(nodes_batches, max_common_list_size, G):
    for batches_index in range(0, len(nodes_batches)):
        print("################################################################################")
        print(f"                ####  CREATE CONNECTIONS [ {str(batches_index + 1)} / {str(len(nodes_batches))} ]  ####")
        print("################################################################################")
        this_batch = nodes_batches[batches_index]

        # First remove non-knowledge content
        for message in Format_Delete["messages"]:
            if message["role"] == "system":
                message["content"] = goal_delete
            if message["role"] == "user":
                message["content"] = str(this_batch)

        json_data_delete = json.dumps(Format_Delete)
        response_delete = requests.post(Set["ollama_host"] + "/api/chat", data=json_data_delete, headers={'Content-Type': 'application/json'})
        response_delete_str = json.loads(response_delete.text)
        res_delete = json.loads(response_delete_str["message"]["content"])
        print("Original response (delete):\n" + str(res_delete))
        print("-----------------------------------------------------------------------------")
        res_deleted = res_delete["delete"]
        this_batch_new = merge_delete_nodes([], [], res_deleted, this_batch)

        # Then create relations
        common_elements_list = check_isolated_nodes(this_batch_new, G)
        if(common_elements_list == []):
            print("---------------  NO ISOLATED NODES  ---------------")
            continue
        elif(len(common_elements_list) > max_common_list_size):
            print("-----------------------------------------------------------------------------")
            print("---------------  SPLITTING LONG ISOLATED NODE LIST  ---------------")
            print(f"Original isolated nodes:\n{str(common_elements_list)}")
            print("-----------------------------------------------------------------------------")
            for long_common_index in range(0, len(common_elements_list), max_common_list_size):
                print(f"Split isolated nodes:\n{str(common_elements_list[long_common_index:long_common_index + max_common_list_size])}")
                splited_long_common_elements_list = common_elements_list[long_common_index:long_common_index + max_common_list_size]
                create_edge(this_batch_new, splited_long_common_elements_list)
            print("#################################")
        else:
            create_edge(this_batch_new, common_elements_list)
            print("#################################")
        
        G_new = nx.read_graphml(Set["WORK_DIC"] + "/" + "graph_chunk_entity_relation.graphml")
        isolated_nodes = check_isolated_nodes(this_batch_new, G_new)
        while(isolated_nodes != []):
            print(f"-------  Still have isolated nodes, try again:  {isolated_nodes}  -------")
            create_edge(this_batch_new, isolated_nodes)
            G_newest = nx.read_graphml(Set["WORK_DIC"] + "/" + "graph_chunk_entity_relation.graphml")
            isolated_nodes = check_isolated_nodes(this_batch_new, G_newest)

# 清洗孤立节点
def clean_isolated_nodes():
    print("#############################################################################")
    print("                          ####  CLEAN ISOLATED NODES  ####")
    print("#############################################################################")
    original_nodes = []
    G = nx.read_graphml(Set["WORK_DIC"] + "/" + "graph_chunk_entity_relation.graphml")
    net = Network(height="100vh", notebook=True, select_menu=True)
    net.from_nx(G)
    for node in net.nodes:
        original_nodes.append(node["label"])
    for node in original_nodes:
        if(G.degree[node] == 1):
            neighbor = G.neighbors(node)
            neighbor_degree = [item[1] for item in G.degree(neighbor)]
            if(neighbor_degree[0] <= 1):
                rag.delete_by_entity(node)
                print("Cleaned isolated nodes:  " + str(node))
        elif(G.degree[node] == 0):
            rag.delete_by_entity(node)
            print("Cleaned isolated nodes:  " + str(node))

#logger
class Logger(object):
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

sys.stdout = Logger(Set["WORK_DIC"] + "/output.txt")

def merge_delete_create_control():
    TARGET_SIZE_MD = Set["split_size_merge_delete"]
    MAX_SIZE_MD = int(Set["split_size_merge_delete"] * 1.25)
    MIN_SIZE_MD = int(Set["split_size_merge_delete"] * 0.75)
    nodes_batches_MD, G_merge_delete = classify_nodes_labels(TARGET_SIZE_MD, MAX_SIZE_MD, MIN_SIZE_MD)
    merge_delete_control(nodes_batches_MD, "Acordding to labels")

    nodes_batches_MD_whole, G_MD_whole = classify_nodes_whole(TARGET_SIZE_MD, MAX_SIZE_MD, MIN_SIZE_MD)
    merge_delete_control(nodes_batches_MD_whole, "Acordding to similarity")

    TARGET_SIZE_CREATE = Set["split_size_create_edge"]
    MAX_SIZE_CREATE = int(Set["split_size_create_edge"] * 1.25)
    MIN_SIZE_CREATE = int(Set["split_size_create_edge"] * 0.75)
    nodes_batches_create, G_create = classify_nodes_labels(TARGET_SIZE_CREATE, MAX_SIZE_CREATE, MIN_SIZE_CREATE)
    create_edge_control(nodes_batches_create, Set["max_common_list_size"], G_create)

    if(Set["need_clean_isolated_nodes"] == True):
        clean_isolated_nodes()

if __name__ == "__main__":
    merge_delete_create_control()
    print("################################################################################")
    print("                         ####  FULL PROCESS COMPLETED  ####")
    print("################################################################################")