import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from .utils import logger
from fuzzywuzzy import process
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from collections import defaultdict
from collections import deque
from .prompt import PROMPTS
from abc import ABC, abstractmethod
from functools import partial
from json_repair import repair_json


# ======================= Base Strategy Class =======================
class BaseDeduplicationStrategy(ABC):
    """Base class for deduplication strategies, defines standard interfaces"""

    def __init__(self, node_data: list, param: dict = None):
        self.node_data = node_data
        self.param = param

    @abstractmethod
    async def classify_nodes_by_similarity(self) -> list:
        """Asynchronously classify nodes by similarity"""
        pass

    @abstractmethod
    async def clean_nodes(self, nodes_batches: list) -> None:
        """Main method for asynchronously cleaning nodes"""
        pass


# ======================= Clustering Processor =======================
class SemanticClusterBatcher:
    """Semantic clustering batch processor"""

    def __init__(self, TARGET_SIZE, MAX_SIZE, MIN_SIZE, ModelName):
        self.model = SentenceTransformer(ModelName)
        self.TARGET_SIZE = TARGET_SIZE
        self.MAX_SIZE = MAX_SIZE
        self.MIN_SIZE = MIN_SIZE

    def _validate_input(self, nodes):
        if len(nodes) == 0:
            raise ValueError("Input node list cannot be empty")
        if not all(isinstance(node, str) for node in nodes):
            raise TypeError("All nodes must be strings")

    def _hierarchical_clustering(self, embeddings, target_size):
        """Hierarchical clustering algorithm"""
        try:
            Z = linkage(embeddings, method="ward", metric="euclidean")
            target_clusters = max(1, len(embeddings) // target_size)
            return fcluster(Z, t=target_clusters, criterion="maxclust")
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {str(e)}")
            raise

    def _split_large_clusters(
        self, clusters, embeddings, original_nodes, MAX_CLUSTER_SIZE, MIN_BATCH_SIZE
    ):
        """Split oversized clusters"""
        final_clusters = []
        for cluster in clusters:
            original_count = len(cluster)
            if original_count <= MAX_CLUSTER_SIZE:
                final_clusters.append(cluster)
                continue

            try:
                indices = [i for i, n in enumerate(original_nodes) if n in cluster]
                sub_embeddings = embeddings[indices]

                assert (
                    len(indices) == original_count
                ), "Indices don't match cluster elements"

                n_sub = max(2, original_count // MIN_BATCH_SIZE + 1)
                kmeans = KMeans(n_clusters=n_sub, n_init=10)
                sub_labels = kmeans.fit_predict(sub_embeddings)

                sub_clusters = defaultdict(list)
                for node, label in zip(cluster, sub_labels):
                    sub_clusters[label].append(node)

                split_total = sum(len(v) for v in sub_clusters.values())
                if split_total != original_count:
                    raise ValueError(
                        f"Element loss: original {original_count} after split {split_total}"
                    )

                final_clusters.extend(sub_clusters.values())
            except Exception as e:
                logger.warning(
                    f"Sub-clustering failed, keeping original cluster: {str(e)}"
                )
                final_clusters.append(cluster)
        return final_clusters

    def _optimize_batches(self, clusters, MAX_CLUSTER_SIZE, MIN_BATCH_SIZE):
        """Optimize batch grouping"""
        batches = []
        current_batch = []
        cluster_queue = deque(sorted(clusters, key=len, reverse=True))

        while cluster_queue:
            cluster = cluster_queue.popleft()

            if len(current_batch) + len(cluster) <= self.TARGET_SIZE:
                current_batch.extend(cluster)
                continue

            remaining_space = self.TARGET_SIZE - len(current_batch)

            if remaining_space >= self.MIN_SIZE and len(cluster) > remaining_space:
                current_batch.extend(cluster[:remaining_space])
                cluster_queue.appendleft(cluster[remaining_space:])
            else:
                batches.append(current_batch)
                current_batch = list(cluster)

        if current_batch:
            batches.append(current_batch)

        input_count = sum(len(c) for c in clusters)
        output_count = sum(len(b) for b in batches)
        if input_count != output_count:
            error_msg = (
                f"Critical error: element count changed ({input_count}→{output_count})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        return batches

    def cluster_and_batch(self, nodes):
        """Main processing pipeline"""
        self._validate_input(nodes)

        logger.info("Generating semantic embeddings...")
        embeddings = self.model.encode(nodes)

        logger.info("Performing hierarchical clustering...")
        cluster_ids = self._hierarchical_clustering(embeddings, self.TARGET_SIZE)

        cluster_dict = defaultdict(list)
        for node, cid in zip(nodes, cluster_ids):
            cluster_dict[cid].append(node)
        initial_clusters = list(cluster_dict.values())

        logger.info("Optimizing cluster sizes...")
        optimized_clusters = self._split_large_clusters(
            initial_clusters, embeddings, nodes, self.MAX_SIZE, self.MIN_SIZE
        )

        logger.info("Creating final batches...")
        batches = self._optimize_batches(
            optimized_clusters, self.MAX_SIZE, self.MIN_SIZE
        )

        return batches


# ======================= LLM-based Cleaning Strategy =======================
class LLMBasedCleaning(BaseDeduplicationStrategy):
    """LLM-based node cleaning strategy"""

    def __init__(self, rag, node_data: list, param: dict = None):
        super().__init__(node_data, param)
        self.rag = rag

    async def classify_nodes_by_similarity(self, node_data) -> list:
        """Asynchronously classify nodes"""
        logger.info(
            f"Classifying nodes by similarity with batch size: {self.param.get('clean_split_batch_size', 30)}"
        )
        TARGET_SIZE = self.param.get("clean_split_batch_size", 30)
        MAX_SIZE = int(TARGET_SIZE * 1.25)
        MIN_SIZE = int(TARGET_SIZE * 0.75)

        batcher_labels = SemanticClusterBatcher(
            TARGET_SIZE,
            MAX_SIZE,
            MIN_SIZE,
            self.param.get(
                "clean_split_batch_model", "paraphrase-multilingual-MiniLM-L12-v2"
            ),
        )
        # Classify node data
        classified_data = defaultdict(list)
        for node in node_data:
            if "entity_type" in node:
                classified_data[node["entity_type"]].append(node["entity_id"])
            else:
                logger.warning(f"Node without entity_type: {node}")

        # Process node batches
        nodes_batches = []
        short_KP = []
        for entity_type, items in classified_data.items():
            if len(items) < MAX_SIZE:
                if len(items) > 10:
                    nodes_batches.append(items)
                else:
                    short_KP.append(items)
            else:
                splited_long = batcher_labels.cluster_and_batch(items)
                for splited in splited_long:
                    nodes_batches.append(splited)

        # Process short knowledge points
        combined_short_KP = [item for sublist in short_KP for item in sublist]
        if len(combined_short_KP) < MAX_SIZE:
            nodes_batches.append(combined_short_KP)
            logger.info("Scattered knowledge points < max cluster size, added directly")
        else:
            batcher_labels_isolated = SemanticClusterBatcher(
                TARGET_SIZE, MAX_SIZE, MIN_SIZE
            )
            splited_long_isolated = batcher_labels_isolated.cluster_and_batch(
                combined_short_KP
            )
            for splited_isolated in splited_long_isolated:
                nodes_batches.append(splited_isolated)

        return nodes_batches

    async def clean_nodes(
        self,
        nodes_batches: list,
    ) -> list:
        """Main method for asynchronously cleaning nodes, preserving original structure"""
        for i in range(0, len(nodes_batches)):
            logger.info(
                "#############################################################################"
            )
            logger.info(
                f"                  ####  CLEANING NODES[{str(i + 1)}/{str(len(nodes_batches))}]  Length [{str(len(nodes_batches[i]))}]  ####"
            )
            logger.info(
                "#############################################################################"
            )

            this_batch = nodes_batches[i]
            use_model_func: callable = self.rag.llm_model_func
            use_model_func = partial(use_model_func, _priority=5)
            clean_system_prompt = (
                self.param.get("clean_system_prompt") or PROMPTS["goal_clean"]
            )
            clean_system_prompt = (
                str(clean_system_prompt) + "\n" + PROMPTS["goal_clean_examples"]
            )
            response = await use_model_func(
                str(this_batch),
                system_prompt=clean_system_prompt,
            )
            # logger.info(f"    clean_system_prompt: {clean_system_prompt}")
            # logger.info(f"    Response: {response}")
            if response is None or response.strip() in [
                "null",
                "",
                "[]",
                "Null",
                "NULL",
            ]:
                logger.info("    No cleaning needed, skipping this batch")
                continue
            # Process merge operations
            try:
                repaired = repair_json(response)
                data = json.loads(repaired)
            except Exception as e:
                logger.info(f"Failed to repair JSON during node cleaning: {e}")
            merge_operations = data.get("merge", [])

            # Merge nodes
            if merge_operations:
                for op in merge_operations:
                    nodes_to_merge = op["keywords"]
                    summarized_node = op["summary"]

                    # Skip single-node merges
                    if len(nodes_to_merge) == 1:
                        logger.info(
                            f"    ———— Single output, skipping merge: {nodes_to_merge}"
                        )
                        continue

                    found_nodes = []
                    for node in nodes_to_merge:
                        match = process.extractOne(node, this_batch)
                        if match and match[1] >= 95:
                            found_nodes.append(match[0])

                    # Execute merge
                    if len(found_nodes) >= 2 and summarized_node != found_nodes:
                        this_batch.append(summarized_node)
                        logger.info(
                            f"    Merged nodes: {summarized_node} ← {found_nodes}"
                        )
                        await self.rag.amerge_entities(
                            source_entities=found_nodes, target_entity=summarized_node
                        )
                        # Remove merged nodes
                        for n in found_nodes:
                            if n in this_batch:
                                this_batch.remove(n)
                    else:
                        logger.info(
                            f"    ———— Single output, skipping merge: {found_nodes}"
                        )
            else:
                logger.info("    No merge operations found")

        logger.info(
            "################################################################################"
        )
        logger.info("                     ####  NODE CLEANING COMPLETE  ####")
        logger.info(
            "################################################################################"
        )
