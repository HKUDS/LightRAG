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
import requests, json, sys, logging, random
import networkx as nx
from lightrag.prompt import PROMPTS

############################    Settings   #####################################
Set = {
    "WORK_DIC": "./Ollama_Qwen_Labels_Three_Books_",  # Working directory for graph files and output
    
    "llm_model": "qwen2.5-coder:32b",  # Language model used for processing
    "ollama_host": "http://localhost:11434",  # Local Ollama service URL
    "embedding_dim": 1024,  # Embedding dimension for text
    "embedding_max_token_size": 8192,  # Max token size for embeddings
    "embedding_model": "bge-m3",  # Model used for embeddings
    
    # Batch processing parameters
    "split_size_create_edge": 20,  # Target batch size for edge creation clustering
    "split_size_clean": 30,  # Target batch size for merge/delete operations
    
    "need_clean_isolated_nodes": False,  # Whether to clean isolated nodes after processing
}

rag = LightRAG(
    working_dir = Set["WORK_DIC"],
    llm_model_func = ollama_model_complete,
    llm_model_name = Set["llm_model"],
    llm_model_max_async = 4,
    llm_model_max_token_size = 32768,
    llm_model_kwargs={"host": Set["ollama_host"], "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=Set["embedding_dim"],
        max_token_size=Set["embedding_max_token_size"],
        func=lambda texts: ollama_embed(
            texts, embed_model=Set["embedding_model"], host=Set["ollama_host"]
        ),
    ),
)

goal_clean = PROMPTS["goal_clean"]
format_clean = PROMPTS["format_clean"]

goal_create = PROMPTS["goal_create"]
format_create = PROMPTS["format_create"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

###############################################################################

#############################    Functions   ##################################
def case_sensitive_ratio(s1, s2):
    return fuzz.ratio(s1, s2)

# Use LLM to process input text and return structured output
def use_LLM(goal, input_text, format):
    for message in format["messages"]:
        if message["role"] == "system":
            message["content"] = goal
        if message["role"] == "user":
            message["content"] = input_text
        format["model"] = Set["llm_model"]
        
    json_data = json.dumps(format)
    response = requests.post(Set["ollama_host"] + "/api/chat", data=json_data, headers={'Content-Type': 'application/json'})
    
    try:
        response_dict = json.loads(response.text)
        # print(f"response_dict is:\n{response_dict}\n\n")
        
        # repair the content string
        content_str = response_dict["message"]["content"].replace("\n", "")
        content_str.replace("\\", "")
        content_str.replace(" ", "")
        # print(content_str)

        if content_str.startswith('{\n{') or content_str.startswith('{{'):
            content_str = content_str[1:]

        content_dict = json.loads(content_str)
        return content_dict
    
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
    
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

# Cluster nodes by similarity
def classify_nodes_by_similarity(TARGET_SIZE, MAX_SIZE, MIN_SIZE):
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
            # print("Oversized cluster:\n" + str(items) + "\n")
            for splited in splited_long:
                nodes_batches.append(splited)
                # print("       —— Split cluster:\n" + str(splited) + "\n")

    combined_short_KP = [item for sublist in short_KP for item in sublist]
    if(len(combined_short_KP) < MAX_SIZE):
        nodes_batches.append(combined_short_KP)
        print("Scattered knowledge points < max cluster size, added directly")
    else:
        print("Scattered knowledge points > max cluster size, splitting")
        batcher_labels_isolated = SemanticClusterBatcher(TARGET_SIZE, MAX_SIZE, MIN_SIZE)
        splited_long_isolated = batcher_labels_isolated.cluster_and_batch(combined_short_KP)
        # print("        Oversized cluster:\n" + str(splited_long_isolated) + "\n")
        for splited_isolated in splited_long_isolated:
            # print("             —— Split cluster:\n" + str(splited_isolated) + "\n")
            nodes_batches.append(splited_isolated)
    
    # print("Batch size distribution:")
    # for batches_index in range(len(nodes_batches)):
    #     print(str(len(nodes_batches[batches_index])) + "[ " + str(batches_index + 1) + " / " + str(len(nodes_batches)) + " ]")
    return nodes_batches, G

# Clean node
def clean_nodes(nodes_batches):
    for i in range(0, len(nodes_batches)):
        print("#############################################################################")
        print(f"                  ####  CLEAN NODES[{str(i + 1)}/{str(len(nodes_batches))}]  Lenth [{str(len(nodes_batches[i]))}]  ####")
        print("#############################################################################")
        this_batch = nodes_batches[i]
        # print("The input batch is  :\n" + input_text)
        # print("-----------------------------------------------------------------------------")
        res = use_LLM(goal_clean, str(this_batch), format_clean)
        merge_operations = res.get('merge', [])
        delete_operations = res.get('delete', [])
        # Merge similar nodes
        if merge_operations:
            for op in merge_operations:
                nodes_to_merge = op['keywords']
                summarized_node = op['summary']
                # Single output handling
                if len(nodes_to_merge) == 1:
                    print(f"    ———— Single output, skipping merge: {nodes_to_merge}")
                    continue
                found_nodes = []
                for node in nodes_to_merge:
                    # Search for matching nodes in this_batch
                    match = process.extractOne(node, this_batch)
                    if not match:
                        continue
                    matched_node, score = match
                    if score >= 95:
                        found_nodes.append(matched_node)
                
                # Merge nodes if at least two are found
                if len(found_nodes) >= 2 and summarized_node != found_nodes:
                    # Merge nodes
                    this_batch.append(summarized_node)
                    print(f"    Merged nodes: {summarized_node} ← {found_nodes}")
                    rag.merge_entities(source_entities = found_nodes, target_entity = summarized_node)
                    # Remove merged nodes from this_batch
                    for n in found_nodes:
                        if n in this_batch:
                            this_batch.remove(n)
                else:
                    print(f"    ———— Single output, skipping merge: {found_nodes}")
        else:
            print("    No merge operations found")
        print("-----------------------------------------------------------------------------")
        
        # Delete useless nodes
        if delete_operations:
            deleted_nodes = []
            for node in delete_operations:
                # Search for matching nodes in this_batch
                match = process.extractOne(node, this_batch)
                if not match:
                    continue
                    
                matched_node, score = match
                if score >= 95:
                    this_batch.remove(matched_node)
                    rag.delete_by_entity(matched_node)
                    deleted_nodes.append(matched_node)
                else:
                    print(f"    ———— Matched node score too low: {node} → {matched_node} ({score})")
            print(f"    Deleted node: {deleted_nodes}")
        else:
            print("    No delete operations found")
        
        print("-----------------------------------------------------------------------------")

    print("################################################################################")
    print(f"                     ####  NODE PRUNING COMPLETE  ####")
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
def create_edge(nodes_batches, G):
    for batches_index in range(0, len(nodes_batches)):
        print("################################################################################")
        print(f"                ####  CREATE CONNECTIONS [ {str(batches_index + 1)} / {str(len(nodes_batches))} ]  ####")
        print("################################################################################")
        this_batch = nodes_batches[batches_index]
        isolated_nodes = check_isolated_nodes(this_batch, G)
        if isolated_nodes != []:
            for i in isolated_nodes:
                this_batch.remove(i)
            create_str = "isolated nodes: " + str(isolated_nodes) + "\n" + "related nodes: " + str(this_batch)
            res = use_LLM(goal_create, create_str, format_create)

            all_possible_nodes = this_batch + isolated_nodes
            
            for create in res["relations"]:
                try:
                    if not all(key in create for key in ["source", "target", "description", "keywords", "weight"]):
                        print(f"    —— Missing required fields in relation: {create}")
                        continue
                    
                    # Use FuzzyWuzzy to match the node
                    source_options = process.extract(create["source"], all_possible_nodes, limit=5)
                    best_source, source_score = source_options[0] if source_options else (None, 0)
                    
                    target_options = process.extract(create["target"], all_possible_nodes, limit=5)
                    best_target, target_score = target_options[0] if target_options else (None, 0)
                    
                    # Log the match details
                    match_details = (
                        f"Original source: {create['source']} → Best match: {best_source} (score={source_score})"
                        f" | Original target: {create['target']} → Best match: {best_target} (score={target_score})"
                    )
                    
                    # Check if the best matches are valid
                    if source_score < 85 or target_score < 85:
                        print(f"    —— Low match scores: {match_details}")
                        continue
                    
                    # Skip the connection if both nodes are isolated
                    if best_source in isolated_nodes and best_target in isolated_nodes:
                        print(f"    —— Skipping isolated-isolated connection: {match_details}")
                        continue
                    
                    print(f"    Creating relation: {best_source} → {best_target}")
                    rag.create_relation(best_source, best_target, {
                        "description": create["description"],
                        "keywords": ', '.join(map(repr, create["keywords"])),
                        "weight": create["weight"]
                    })
                    
                except KeyError as e:
                    print(f"    —— KeyError: {e}")
                    continue
                except Exception as e:
                    print(f"    —— Unexpected error: {str(e)}")
                    continue

# Clean isolated nodes
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

def clean_create_control():
    TARGET_SIZE_MD = Set["split_size_clean"]
    MAX_SIZE_MD = int(Set["split_size_clean"] * 1.25)
    MIN_SIZE_MD = int(Set["split_size_clean"] * 0.75)
    nodes_batches_clean, G_clean = classify_nodes_by_similarity(TARGET_SIZE_MD, MAX_SIZE_MD, MIN_SIZE_MD)
    clean_nodes(nodes_batches_clean)

    TARGET_SIZE_CREATE = Set["split_size_create_edge"]
    MAX_SIZE_CREATE = int(Set["split_size_create_edge"] * 1.25)
    MIN_SIZE_CREATE = int(Set["split_size_create_edge"] * 0.75)
    nodes_batches_create, G_create = classify_nodes_by_similarity(TARGET_SIZE_CREATE, MAX_SIZE_CREATE, MIN_SIZE_CREATE)
    create_edge(nodes_batches_create, G_create)

    if(Set["need_clean_isolated_nodes"] == True):
        clean_isolated_nodes()

if __name__ == "__main__":
    # Generate original knowledge graph
    G = nx.read_graphml(f"./{Set["WORK_DIC"]}/graph_chunk_entity_relation.graphml")
    net = Network(height="100vh", notebook=True)
    net.from_nx(G)
    for node in net.nodes:
        node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if "description" in node:
            node["title"] = node["description"]
    for edge in net.edges:
        if "description" in edge:
            edge["title"] = edge["description"]
    net.show(f"./{Set["WORK_DIC"]}/original_knowledge_graph.html")
    
    # Start cleaning process
    clean_create_control()
    
    # Generate cleaned knowledge graph
    G = nx.read_graphml(f"./{Set["WORK_DIC"]}/graph_chunk_entity_relation.graphml")
    net = Network(height="100vh", notebook=True)
    net.from_nx(G)
    for node in net.nodes:
        node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if "description" in node:
            node["title"] = node["description"]
    for edge in net.edges:
        if "description" in edge:
            edge["title"] = edge["description"]
    net.show(f"./{Set["WORK_DIC"]}/cleaned_knowledge_graph.html")
    print("################################################################################")
    print("                         ####  FULL PROCESS COMPLETED  ####")
    print("################################################################################")