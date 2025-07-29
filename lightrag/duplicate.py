import json
from typing import Protocol, Optional, List, Dict, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import partial

try:
    from fuzzywuzzy import process

    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    process = None
    FUZZYWUZZY_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import KMeans

    CLUSTERING_AVAILABLE = True
except ImportError:
    linkage = fcluster = KMeans = None
    CLUSTERING_AVAILABLE = False

try:
    from json_repair import repair_json

    JSON_REPAIR_AVAILABLE = True
except ImportError:
    repair_json = json.loads  # Fallback to standard json.loads
    JSON_REPAIR_AVAILABLE = False

from .utils import logger
from collections import defaultdict
from collections import deque
from .prompt import PROMPTS


# ======================= Data Structures =======================
@dataclass
class EntityInfo:
    """Standard entity information structure"""

    entity_id: str
    entity_type: str
    description: Optional[str] = None


# ======================= Service Interfaces =======================
class DeduplicationService(Protocol):
    """Protocol for deduplication service that provides access to RAG functionality"""

    @property
    def rag_instance(self):
        """Get the RAG instance"""
        ...

    async def process_with_llm(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> Optional[str]:
        """Process text using RAG's LLM function"""
        ...

    async def merge_entities(
        self, source_entities: List[str], target_entity: str
    ) -> None:
        """Merge entities using RAG's merge function"""
        ...

    async def get_embeddings(self, texts: List[str]) -> Any:
        """Get embeddings using RAG's embedding function"""
        ...


class LightRAGDeduplicationService:
    """Concrete implementation of DeduplicationService for LightRAG"""

    def __init__(self, rag_instance):
        self._rag = rag_instance

    @property
    def rag_instance(self):
        """Get the RAG instance"""
        return self._rag

    async def process_with_llm(
        self, prompt: str, system_prompt: str = "", **kwargs
    ) -> Optional[str]:
        """Process text using RAG's LLM function"""
        use_model_func = partial(self._rag.llm_model_func, _priority=5)
        return await use_model_func(prompt, system_prompt=system_prompt, **kwargs)

    async def merge_entities(
        self, source_entities: List[str], target_entity: str
    ) -> None:
        """Merge entities using RAG's merge function"""
        return await self._rag.amerge_entities(
            source_entities=source_entities, target_entity=target_entity
        )

    async def get_embeddings(self, texts: List[str]) -> Any:
        """Get embeddings using RAG's embedding function"""
        if not self._rag.embedding_func:
            raise ValueError("RAG instance does not have embedding function configured")
        return await self._rag.embedding_func(texts)


# ======================= Configuration System =======================
@dataclass
class BaseDeduplicationConfig:
    """Base configuration for all deduplication strategies"""

    strategy_name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDeduplicationConfig":
        """Create config instance from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LLMBasedConfig(BaseDeduplicationConfig):
    """Configuration specific to LLM-based deduplication strategy"""

    strategy_name: str = "llm_based"
    target_batch_size: int = 30
    max_batch_size: Optional[int] = None
    min_batch_size: Optional[int] = None
    similarity_threshold: float = 0.85
    system_prompt: Optional[str] = None

    def __post_init__(self):
        if self.max_batch_size is None:
            self.max_batch_size = int(self.target_batch_size * 1.25)
        if self.min_batch_size is None:
            self.min_batch_size = int(self.target_batch_size * 0.75)


# Configuration factory
class ConfigFactory:
    """Factory for creating strategy-specific configurations"""

    _config_classes = {
        "llm_based": LLMBasedConfig,
    }

    @classmethod
    def register_config(cls, strategy_name: str, config_class: type):
        """Register a new configuration class"""
        cls._config_classes[strategy_name] = config_class

    @classmethod
    def create_config(
        cls, strategy_name: str, config_data: Dict[str, Any]
    ) -> BaseDeduplicationConfig:
        """Create strategy-specific configuration"""
        if strategy_name not in cls._config_classes:
            available_configs = list(cls._config_classes.keys())
            raise ValueError(
                f"Unknown configuration for strategy '{strategy_name}'. "
                f"Available configurations: {available_configs}"
            )

        config_class = cls._config_classes[strategy_name]
        return config_class(**config_data)


# ======================= Clustering Processor =======================
class SemanticClusterBatcher:
    """Semantic clustering batch processor using RAG's embedding function"""

    def __init__(
        self,
        config: BaseDeduplicationConfig,
        deduplication_service: DeduplicationService,
    ):
        self.config = config
        self.deduplication_service = deduplication_service

    def _validate_input(self, nodes: List[str]):
        """Validate input nodes"""
        if not nodes:
            raise ValueError("Input node list cannot be empty")

    async def _get_embeddings(self, nodes: List[str]) -> Any:
        """Get embeddings using RAG's embedding function"""
        try:
            embeddings = await self.deduplication_service.get_embeddings(nodes)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings from RAG: {e}")
            raise

    def _hierarchical_clustering(self, embeddings, target_size: int):
        """Hierarchical clustering algorithm"""
        if not CLUSTERING_AVAILABLE:
            raise ImportError("scipy and scikit-learn are required for clustering")

        try:
            Z = linkage(embeddings, method="ward", metric="euclidean")
            target_clusters = max(1, len(embeddings) // target_size)
            return fcluster(Z, t=target_clusters, criterion="maxclust")
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {str(e)}")
            raise

    def _split_large_clusters(
        self, clusters: List[List[str]], embeddings, original_nodes: List[str]
    ):
        """Split oversized clusters using KMeans"""
        if not hasattr(self.config, "max_batch_size") or not hasattr(
            self.config, "min_batch_size"
        ):
            # For strategies without batch size limits, return as-is
            return clusters

        final_clusters = []
        for cluster in clusters:
            original_count = len(cluster)
            if original_count <= self.config.max_batch_size:
                final_clusters.append(cluster)
                continue

            try:
                indices = [i for i, n in enumerate(original_nodes) if n in cluster]
                sub_embeddings = embeddings[indices]

                assert (
                    len(indices) == original_count
                ), "Indices don't match cluster elements"

                n_sub = max(2, original_count // self.config.min_batch_size + 1)
                kmeans = KMeans(n_clusters=n_sub, n_init=10, random_state=42)
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
                logger.info(
                    f"Split cluster of size {original_count} into {len(sub_clusters)} sub-clusters"
                )
            except Exception as e:
                logger.warning(
                    f"Sub-clustering failed, keeping original cluster: {str(e)}"
                )
                final_clusters.append(cluster)
        return final_clusters

    def _optimize_batches(self, clusters: List[List[str]]) -> List[List[str]]:
        """Optimize batch grouping using greedy algorithm"""
        if not hasattr(self.config, "target_batch_size"):
            # For strategies without batch optimization, return as-is
            return clusters

        batches = []
        current_batch = []
        cluster_queue = deque(sorted(clusters, key=len, reverse=True))

        while cluster_queue:
            cluster = cluster_queue.popleft()

            if len(current_batch) + len(cluster) <= self.config.target_batch_size:
                current_batch.extend(cluster)
                continue

            remaining_space = self.config.target_batch_size - len(current_batch)

            if (
                remaining_space >= self.config.min_batch_size
                and len(cluster) > remaining_space
            ):
                current_batch.extend(cluster[:remaining_space])
                cluster_queue.appendleft(cluster[remaining_space:])
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = list(cluster)

        if current_batch:
            batches.append(current_batch)

        # Validate element count consistency
        input_count = sum(len(c) for c in clusters)
        output_count = sum(len(b) for b in batches)
        if input_count != output_count:
            error_msg = (
                f"Critical error: element count changed ({input_count}→{output_count})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        return batches

    async def cluster_and_batch(self, nodes: List[str]) -> List[List[str]]:
        """Main processing pipeline for clustering and batching using RAG's embedding function"""
        self._validate_input(nodes)

        logger.info("Generating semantic embeddings using RAG's embedding function...")
        embeddings = await self._get_embeddings(nodes)

        logger.info("Performing hierarchical clustering...")
        target_size = getattr(self.config, "target_batch_size", 30)
        cluster_ids = self._hierarchical_clustering(embeddings, target_size)

        cluster_dict = defaultdict(list)
        for node, cid in zip(nodes, cluster_ids):
            cluster_dict[cid].append(node)
        initial_clusters = list(cluster_dict.values())

        logger.info("Optimizing cluster sizes...")
        optimized_clusters = self._split_large_clusters(
            initial_clusters, embeddings, nodes
        )

        logger.info("Creating final batches...")
        batches = self._optimize_batches(optimized_clusters)

        return batches


# ======================= Base Strategy Class =======================
class BaseDeduplicationStrategy(ABC):
    """Base class for deduplication strategies"""

    def __init__(self, service: DeduplicationService, config: BaseDeduplicationConfig):
        self.service = service
        self.config = config
        self._check_dependencies()

    @abstractmethod
    def _check_dependencies(self) -> None:
        """Check strategy-specific dependencies"""
        pass

    @abstractmethod
    async def classify_nodes_by_similarity(
        self, node_data: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """Classify nodes by similarity and return batches"""
        pass

    @abstractmethod
    async def clean_nodes(self, nodes_batches: List[List[str]]) -> None:
        """Clean nodes by removing duplicates"""
        pass

    def _normalize_node_data(self, node_data: List[Dict[str, Any]]) -> List[EntityInfo]:
        """Normalize node data to standard EntityInfo structure"""
        normalized = []
        for node in node_data:
            if isinstance(node, dict):
                if "entity_type" in node and "entity_id" in node:
                    entity_info = EntityInfo(
                        entity_id=node["entity_id"],
                        entity_type=node["entity_type"],
                        description=node.get("description"),
                    )
                    normalized.append(entity_info)
                else:
                    logger.warning(f"Node missing required fields: {node}")
            else:
                logger.warning(f"Invalid node format: {node}")
        return normalized


# ======================= Strategy Factory =======================
class DeduplicationStrategyFactory:
    """Factory for creating deduplication strategies"""

    _strategies = {}

    @classmethod
    def register_strategy(cls, strategy_name: str, strategy_class: type):
        """Register a new deduplication strategy"""
        cls._strategies[strategy_name] = strategy_class

    @classmethod
    def create_strategy(
        cls,
        strategy_name: str,
        service: DeduplicationService,
        config: Union[BaseDeduplicationConfig, Dict[str, Any]],
    ) -> BaseDeduplicationStrategy:
        """Create a deduplication strategy instance"""
        if strategy_name not in cls._strategies:
            available_strategies = list(cls._strategies.keys())
            raise ValueError(
                f"Unknown deduplication strategy '{strategy_name}'. "
                f"Available strategies: {available_strategies}"
            )

        # Convert dict config to proper config object if needed
        if isinstance(config, dict):
            config = ConfigFactory.create_config(strategy_name, config)

        strategy_class = cls._strategies[strategy_name]
        return strategy_class(service, config)

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names"""
        return list(cls._strategies.keys())


# ======================= LLM-based Cleaning Strategy =======================
class LLMBasedCleaning(BaseDeduplicationStrategy):
    """LLM-based node cleaning strategy"""

    def __init__(self, service: DeduplicationService, config: LLMBasedConfig):
        super().__init__(service, config)

    def _check_dependencies(self) -> None:
        """Check LLM-based strategy specific dependencies"""
        missing_deps = []
        if not FUZZYWUZZY_AVAILABLE:
            missing_deps.append("fuzzywuzzy")
        if not CLUSTERING_AVAILABLE:
            missing_deps.append("scipy and scikit-learn")
        if not JSON_REPAIR_AVAILABLE:
            missing_deps.append("json-repair")

        if missing_deps:
            raise ImportError(
                f"Missing dependencies for LLM-based deduplication: {', '.join(missing_deps)}. "
                f"Install with: pip install lightrag-hku[deduplication]"
            )

    async def classify_nodes_by_similarity(
        self, node_data: List[Dict[str, Any]]
    ) -> List[List[str]]:
        """Classify nodes by similarity and return batches"""
        logger.info(
            f"Classifying nodes by similarity with batch size: {self.config.target_batch_size}"
        )

        # Normalize input data
        entities = self._normalize_node_data(node_data)

        # Group by entity type
        classified_data = defaultdict(list)
        for entity in entities:
            classified_data[entity.entity_type].append(entity.entity_id)

        # Process node batches
        nodes_batches = []
        short_batches = []

        # Use improved SemanticClusterBatcher, pass in deduplication_service
        batcher = SemanticClusterBatcher(self.config, self.service)
        # logger.info(f"classified_data: {classified_data}")
        for entity_type, items in classified_data.items():
            if len(items) <= self.config.max_batch_size:
                if len(items) >= self.config.min_batch_size:
                    nodes_batches.append(items)
                else:
                    short_batches.append(items)
            else:
                # Use semantic clustering for large groups
                split_batches = await batcher.cluster_and_batch(items)
                nodes_batches.extend(split_batches)

        # Handle small batches
        if short_batches:
            combined_short = [item for sublist in short_batches for item in sublist]
            if len(combined_short) >= self.config.min_batch_size:
                if len(combined_short) <= self.config.max_batch_size:
                    nodes_batches.append(combined_short)
                else:
                    # Apply clustering to combined short batches
                    split_batches = await batcher.cluster_and_batch(combined_short)
                    nodes_batches.extend(split_batches)
                logger.info(
                    f"Processed {len(short_batches)} small batches into {len(combined_short)} items"
                )

        logger.info(f"Created {len(nodes_batches)} batches for processing")
        # logger.info(f"nodes_batches: {nodes_batches}")
        return nodes_batches

    async def clean_nodes(self, nodes_batches: List[List[str]]) -> None:
        """Main method for cleaning nodes with improved error handling"""
        failed_batches = []

        for i, batch in enumerate(nodes_batches):
            logger.info("=" * 80)
            logger.info(
                f"CLEANING BATCH [{i + 1}/{len(nodes_batches)}] - Size: {len(batch)}"
            )
            logger.info("=" * 80)

            try:
                success = await self._process_single_batch(batch)
                if not success:
                    failed_batches.append((i, batch))
            except Exception as e:
                logger.error(f"Failed to process batch {i + 1}: {str(e)}")
                failed_batches.append((i, batch))

        if failed_batches:
            logger.warning(f"Failed to process {len(failed_batches)} batches")

        logger.info("=" * 80)
        logger.info("NODE CLEANING COMPLETE")
        logger.info("=" * 80)

    async def _process_single_batch(self, batch: List[str]) -> bool:
        """Process a single batch with proper error handling"""
        try:
            # Prepare system prompt
            system_prompt = self.config.system_prompt or PROMPTS["goal_clean"]
            system_prompt = str(system_prompt) + "\n" + PROMPTS["goal_clean_examples"]

            # Call LLM
            response = await self.service.process_with_llm(
                str(batch), system_prompt=system_prompt
            )

            if not response or response.strip().lower() in ["null", "", "[]", "none"]:
                logger.info("No cleaning needed for this batch")
                return True

            # Parse response
            try:
                repaired = repair_json(response)
                data = json.loads(repaired)
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {response}")
                return False

            # Process merge operations
            merge_operations = data.get("merge", [])
            if not merge_operations:
                logger.info("No merge operations found")
                return True

            return await self._execute_merge_operations(merge_operations, batch)

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return False

    async def _execute_merge_operations(
        self, merge_operations: List[Dict], batch: List[str]
    ) -> bool:
        """Execute merge operations with atomic transaction support"""
        successful_merges = []

        try:
            for op in merge_operations:
                nodes_to_merge = op.get("keywords", [])
                summarized_node = op.get("summary", "")

                if len(nodes_to_merge) <= 1:
                    logger.info(f"Skipping single-node merge: {nodes_to_merge}")
                    continue

                # Find matching nodes with configurable threshold
                found_nodes = []
                for node in nodes_to_merge:
                    match = process.extractOne(node, batch)
                    if match and match[1] >= (self.config.similarity_threshold * 100):
                        found_nodes.append(match[0])

                if (
                    len(found_nodes) >= 2 and summarized_node
                    # and summarized_node not in found_nodes
                ):
                    try:
                        # Execute merge operation
                        await self.service.merge_entities(found_nodes, summarized_node)

                        # Update batch atomically
                        for node in found_nodes:
                            if node in batch:
                                batch.remove(node)
                        batch.append(summarized_node)

                        successful_merges.append((found_nodes, summarized_node))
                        logger.info(
                            f"Successfully merged: {summarized_node} ← {found_nodes}"
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to merge entities {found_nodes} → {summarized_node}: {e}"
                        )
                        # Note: We don't rollback here as the batch update hasn't happened yet
                        continue
                else:
                    logger.info(
                        f"Insufficient nodes for merge: found_nodes: {found_nodes}, summarized_node: {summarized_node}"
                    )

            return True

        except Exception as e:
            logger.error(f"Error during merge operations: {str(e)}")
            return False


# Register LLM-based strategy
DeduplicationStrategyFactory.register_strategy("llm_based", LLMBasedCleaning)
