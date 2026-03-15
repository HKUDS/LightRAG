import json
import asyncio
import numpy as np
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
    target_batch_size: int = 20  # Reduce batch size to ease ollama burden
    max_batch_size: Optional[int] = None
    min_batch_size: Optional[int] = None
    similarity_threshold: float = 0.75  # Relax embedding threshold
    system_prompt: Optional[str] = None
    strictness_level: str = "strict"

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
        """Get embeddings using RAG's embedding function with intelligent batch processing"""
        try:
            # For ollama, limit batch size to prevent decode errors
            max_embedding_batch_size = 1  # Conservative batch size for Ollama stability

            if len(nodes) <= max_embedding_batch_size:
                # Small batch, process directly
                embeddings = await self.deduplication_service.get_embeddings(nodes)
                return embeddings
            else:
                # Large batch, split into smaller chunks
                logger.info(
                    f"Splitting {len(nodes)} nodes into smaller batches for embedding"
                )
                all_embeddings = []

                for i in range(0, len(nodes), max_embedding_batch_size):
                    batch = nodes[i : i + max_embedding_batch_size]
                    logger.debug(
                        f"Processing embedding batch {i//max_embedding_batch_size + 1}/{(len(nodes) + max_embedding_batch_size - 1)//max_embedding_batch_size}"
                    )

                    try:
                        batch_embeddings = (
                            await self.deduplication_service.get_embeddings(batch)
                        )
                        all_embeddings.append(batch_embeddings)

                        # Add delay between batches to prevent overwhelming Ollama
                        if i + max_embedding_batch_size < len(nodes):
                            await asyncio.sleep(0.2)

                    except Exception as batch_error:
                        logger.error(
                            f"Failed to process embedding batch: {batch_error}"
                        )
                        # Try individual processing as fallback
                        logger.info(
                            "Attempting individual text embedding for failed batch"
                        )
                        individual_embeddings = []
                        for node in batch:
                            try:
                                single_embedding = (
                                    await self.deduplication_service.get_embeddings(
                                        [node]
                                    )
                                )
                                individual_embeddings.append(single_embedding[0])
                                await asyncio.sleep(0.1)
                            except Exception as single_error:
                                logger.error(
                                    f"Failed to embed individual node '{node}': {single_error}"
                                )
                                # Create zero embedding as fallback
                                if individual_embeddings:
                                    zero_embedding = np.zeros_like(
                                        individual_embeddings[0]
                                    )
                                else:
                                    # Default dimension, adjust if needed for your embedding model
                                    zero_embedding = np.zeros(1024)
                                individual_embeddings.append(zero_embedding)

                        if individual_embeddings:
                            all_embeddings.append(np.array(individual_embeddings))

                if not all_embeddings:
                    raise RuntimeError("Failed to generate any embeddings")

                # Concatenate all embeddings
                return np.vstack(all_embeddings)

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
                # Check different possible field names
                entity_id = node.get("entity_id") or node.get("entity_name")
                entity_type = node.get("entity_type")

                if entity_type and entity_id:
                    entity_info = EntityInfo(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        description=node.get("description"),
                    )
                    normalized.append(entity_info)
                else:
                    logger.warning(
                        f"Node missing required fields (entity_id/entity_name and entity_type): {node}"
                    )
            else:
                logger.warning(
                    f"Invalid node format (expected dict, got {type(node)}): {node}"
                )
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
            logger.info("\n" + "-" * 100)
            logger.info(
                f"CLEANING BATCH [{i + 1}/{len(nodes_batches)}] - Size: {len(batch)}"
            )

            try:
                success = await self._process_single_batch(batch)
                if not success:
                    failed_batches.append((i, batch))
            except Exception as e:
                logger.error(f"Failed to process batch {i + 1}: {str(e)}")
                failed_batches.append((i, batch))

        if failed_batches:
            logger.warning(f"Failed to process {len(failed_batches)} batches")

        logger.info(
            f"NODE CLEANING COMPLETE - Total: {len(nodes_batches)}, "
            f"Success: {len(nodes_batches) - len(failed_batches)}, "
            f"Failed: {len(failed_batches)}"
        )

    async def _get_entity_descriptions(self, entity_names: List[str]) -> Dict[str, str]:
        """Get entity description information"""
        descriptions = {}
        try:
            # Get entity information from knowledge graph
            for entity_name in entity_names:
                entity_data = await self.service.rag_instance.chunk_entity_relation_graph.get_node(
                    entity_name
                )

                if entity_data:
                    # Check if entity_data is a dictionary
                    if isinstance(entity_data, dict):
                        descriptions[entity_name] = entity_data.get(
                            "description", "No description"
                        )
                    elif isinstance(entity_data, list) and len(entity_data) > 0:
                        # If it's a list, take the first element
                        first_item = (
                            entity_data[0] if isinstance(entity_data[0], dict) else {}
                        )
                        descriptions[entity_name] = first_item.get(
                            "description", "No description"
                        )
                    else:
                        descriptions[entity_name] = "No description"
                else:
                    descriptions[entity_name] = "No description"
        except Exception as e:
            logger.warning(f"Failed to get entity descriptions: {e}")
            # If failed to get, use default description
            for entity_name in entity_names:
                descriptions[entity_name] = "No description"

        return descriptions

    async def _process_single_batch(
        self, batch: List[str], max_retries: int = 2
    ) -> bool:
        """Process a single batch with proper error handling and retry mechanism"""

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retrying attempt {attempt}...")

                # First analysis uses only entity names, without descriptions

                # Prepare system prompt based on strictness level
                if self.config.system_prompt:
                    system_prompt = self.config.system_prompt
                else:
                    # Choose prompt based on strictness level
                    strictness_prompts = {
                        "strict": PROMPTS["goal_clean_strict"],
                        "medium": PROMPTS["goal_clean_medium"],
                        "loose": PROMPTS["goal_clean_loose"],
                    }
                    system_prompt = strictness_prompts.get(
                        self.config.strictness_level,
                        PROMPTS[
                            "goal_clean_medium"
                        ],  # Default to medium if invalid level
                    )
                system_prompt = (
                    str(system_prompt) + "\n" + PROMPTS["goal_clean_examples"]
                )

                # Add specific analysis instruction for name-only analysis
                analysis_instruction = PROMPTS["name_only_analysis_instruction"]

                full_system_prompt = system_prompt + analysis_instruction

                # Call LLM with entity names only (first analysis)

                response = await self.service.process_with_llm(
                    str(batch), system_prompt=full_system_prompt
                )

                # Add delay for ollama to recover context
                await asyncio.sleep(0.5)

                if not response or response.strip().lower() in [
                    "null",
                    "",
                    "[]",
                    "none",
                ]:
                    logger.info("No cleaning needed for this batch")
                    return True

                # Parse response with improved error handling
                try:
                    repaired = repair_json(response)
                    data = json.loads(repaired)

                    # Handle both dict and list formats
                    if isinstance(data, dict):
                        merge_operations = data.get("merge", [])
                    elif isinstance(data, list):
                        # If LLM returns a list directly, treat it as merge operations
                        merge_operations = data
                    else:
                        logger.error(
                            f"Unexpected data format: {type(data)}, data: {data}"
                        )
                        if attempt < max_retries:
                            continue
                        return False

                except Exception as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Raw response: {response}")
                    if attempt < max_retries:
                        continue
                    return False

                # Process merge operations
                if not merge_operations:
                    logger.info("No merge operations found")
                    return True

                logger.info(f"Found {len(merge_operations)} merge suggestions")
                for i, op in enumerate(merge_operations, 1):
                    if not isinstance(op, dict):
                        logger.warning(f"Invalid merge operation format: {op}")

                return await self._execute_merge_operations(merge_operations, batch)

            except Exception as e:
                logger.error(
                    f"Error processing batch (attempt {attempt + 1}): {str(e)}"
                )
                if attempt < max_retries:
                    continue
                return False

        return False

    async def _verify_merge_with_descriptions(
        self, entities_to_merge: List[str], summary: str
    ) -> List[Dict]:
        """
        Perform secondary verification using entity descriptions to return refined merge suggestions
        """
        try:
            logger.info(f"Verifying merge: {entities_to_merge} → {summary}")

            # Get entity descriptions for all entities to be merged
            entity_descriptions = await self._get_entity_descriptions(entities_to_merge)

            # Format entities with descriptions for LLM analysis
            entities_text_list = []
            for i, entity_name in enumerate(entities_to_merge, 1):
                description = entity_descriptions.get(entity_name, "No description")
                entities_text_list.append(f"{i}. {entity_name}")
                entities_text_list.append(f"   Description: {description}")

            entities_with_descriptions = "\n".join(entities_text_list)

            # Build verification prompt
            verification_prompt = PROMPTS["secondary_merge_verification"].replace(
                "{entities_with_descriptions}", entities_with_descriptions
            )

            # Add examples to the system prompt
            full_system_prompt = PROMPTS["secondary_verification_examples"]

            # Call LLM for verification
            response = await self.service.process_with_llm(
                verification_prompt, system_prompt=full_system_prompt
            )

            # Add delay for ollama to recover context
            await asyncio.sleep(0.3)

            if not response:
                logger.warning("No response from LLM for merge verification")
                return []

            # Parse the verification response with improved error handling
            try:
                # Clean the response first
                cleaned_response = response.strip()

                # Try to extract JSON from the response if it's wrapped in markdown
                if "```json" in cleaned_response:
                    start = cleaned_response.find("```json") + 7
                    end = cleaned_response.find("```", start)
                    if end != -1:
                        cleaned_response = cleaned_response[start:end].strip()
                elif "```" in cleaned_response:
                    start = cleaned_response.find("```") + 3
                    end = cleaned_response.rfind("```")
                    if end != -1 and end > start:
                        cleaned_response = cleaned_response[start:end].strip()

                # Remove any leading/trailing whitespace or newlines
                cleaned_response = cleaned_response.strip()

                # Try json-repair first
                if JSON_REPAIR_AVAILABLE:
                    try:
                        repaired = repair_json(cleaned_response)
                        verification_result = json.loads(repaired)
                    except Exception as repair_error:
                        logger.warning(
                            f"JSON repair failed, trying direct parse: {repair_error}"
                        )
                        verification_result = json.loads(cleaned_response)
                else:
                    verification_result = json.loads(cleaned_response)

                merge_operations = verification_result.get("merge", [])

                return merge_operations

            except Exception as e:
                logger.error(f"Failed to parse verification response: {e}")
                logger.debug(f"Raw verification response: {response}")
                return []

        except Exception as e:
            logger.error(f"Error during merge verification: {e}")
            return []

    async def _execute_merge_operations(
        self, merge_operations: List[Dict], batch: List[str]
    ) -> bool:
        """Execute merge operations with secondary verification and atomic transaction support"""
        successful_merges = []

        try:
            for i, op in enumerate(merge_operations, 1):
                # Validate operation format
                if not isinstance(op, dict):
                    continue

                nodes_to_merge = op.get("keywords", [])
                summarized_node = op.get("summary", "")

                if not nodes_to_merge or not summarized_node:
                    continue

                if len(nodes_to_merge) <= 1:
                    continue

                # Validate that summary is one of the keywords (fuzzywuzzy validation)
                summary_match = process.extractOne(summarized_node, nodes_to_merge)
                if not summary_match or summary_match[1] < (
                    self.config.similarity_threshold * 100
                ):
                    logger.info(
                        f"Skipping illegal merge operation {i}: summary '{summarized_node}' not in keywords list {nodes_to_merge}, best match: {summary_match}"
                    )
                    continue

                # Use the matched keyword as the actual summary to ensure exact matching
                actual_summary = summary_match[0]
                logger.debug(
                    f"Summary validation passed: '{summarized_node}' → '{actual_summary}'"
                )

                # Find matching nodes with configurable threshold
                found_nodes = []
                for node in nodes_to_merge:
                    if not node:  # Skip empty nodes
                        continue
                    match = process.extractOne(node, batch)
                    if match and match[1] >= (self.config.similarity_threshold * 100):
                        found_nodes.append(match[0])

                if len(found_nodes) < 2:
                    logger.info(
                        f"Insufficient matched nodes for merge: found_nodes: {found_nodes}, summarized_node: {actual_summary}"
                    )
                    continue

                # Update summarized_node to use the validated actual summary
                summarized_node = actual_summary

                try:
                    verified_merge_operations = (
                        await self._verify_merge_with_descriptions(
                            found_nodes, summarized_node
                        )
                    )

                    if not verified_merge_operations:
                        logger.info(
                            f" -> Skipping merge due to failed verification: {found_nodes} → {summarized_node}"
                        )
                        continue

                    # Execute verified merge operations
                    for j, verified_op in enumerate(verified_merge_operations, 1):
                        if not isinstance(verified_op, dict):
                            continue

                        verified_nodes = verified_op.get("keywords", [])
                        verified_summary = verified_op.get("summary", "")

                        if len(verified_nodes) <= 1 or not verified_summary:
                            logger.warning(
                                f"Skipping invalid verification result {j}: nodes={verified_nodes}, summary={verified_summary}"
                            )
                            continue

                        # Validate that verified summary is one of the verified keywords (fuzzywuzzy validation)
                        verified_summary_match = process.extractOne(
                            verified_summary, verified_nodes
                        )
                        if not verified_summary_match or verified_summary_match[1] < (
                            self.config.similarity_threshold * 100
                        ):
                            logger.warning(
                                f"Skipping verification result {j}: summary '{verified_summary}' not in keywords list {verified_nodes}, best match: {verified_summary_match}"
                            )
                            continue

                        # Use the matched keyword as the actual verified summary
                        actual_verified_summary = verified_summary_match[0]
                        logger.debug(
                            f"Verified summary validation passed: '{verified_summary}' → '{actual_verified_summary}'"
                        )

                        # Find matching nodes from the verified list
                        verified_found_nodes = []
                        for node in verified_nodes:
                            if not node:  # Skip empty nodes
                                continue
                            match = process.extractOne(node, batch)
                            if match and match[1] >= (
                                self.config.similarity_threshold * 100
                            ):
                                verified_found_nodes.append(match[0])

                        if len(verified_found_nodes) >= 2:
                            await self.service.merge_entities(
                                verified_found_nodes, actual_verified_summary
                            )

                            # Update batch atomically
                            for node in verified_found_nodes:
                                if node in batch:
                                    batch.remove(node)
                            if actual_verified_summary not in batch:
                                batch.append(actual_verified_summary)

                            successful_merges.append(
                                (verified_found_nodes, actual_verified_summary)
                            )
                            logger.info(
                                f" √ Successfully merged: {actual_verified_summary} ← {verified_found_nodes}\n"
                            )

                except Exception as e:
                    logger.error(
                        f" X Failed to merge entities {found_nodes} → {summarized_node}: {e}"
                    )
                    # Continue with next operation instead of failing the whole batch
                    continue

            logger.info(
                f"Merge operations completed: {len(successful_merges)} successful"
            )
            return True

        except Exception as e:
            logger.error(f"Critical error during merge operations: {str(e)}")
            return False


# Register LLM-based strategy
DeduplicationStrategyFactory.register_strategy("llm_based", LLMBasedCleaning)
