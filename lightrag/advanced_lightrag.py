from __future__ import annotations

import asyncio
import traceback
from datetime import datetime
from typing import Any, AsyncIterator, Dict, Optional, Tuple
from dataclasses import asdict

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.utils import logger
from lightrag.query_logger import get_query_logger, LogLevel, QueryLogger
from lightrag.advanced_operate import advanced_semantic_chunking

try:
    from lightrag.constants import (
        DEFAULT_ENABLE_ENHANCED_RELATIONSHIP_FILTER,
        DEFAULT_LOG_RELATIONSHIP_CLASSIFICATION,
        DEFAULT_RELATIONSHIP_FILTER_PERFORMANCE_TRACKING,
        DEFAULT_ENHANCED_FILTER_CONSOLE_LOGGING,
        DEFAULT_ENHANCED_FILTER_MONITORING_MODE,
    )
except ImportError:
    # Fallback defaults if constants not available
    DEFAULT_ENABLE_ENHANCED_RELATIONSHIP_FILTER = True
    DEFAULT_LOG_RELATIONSHIP_CLASSIFICATION = False
    DEFAULT_RELATIONSHIP_FILTER_PERFORMANCE_TRACKING = True
    DEFAULT_ENHANCED_FILTER_CONSOLE_LOGGING = False
    DEFAULT_ENHANCED_FILTER_MONITORING_MODE = False


class AdvancedLightRAG(LightRAG):
    """
    Advanced LightRAG implementation with enhanced features for production use.

    This class extends the base LightRAG with:
    - Comprehensive query logging and metrics tracking
    - Detailed retrieval information for debugging and optimization
    - Support for typed relationships (83 predefined types)
    - Hybrid mix mode combining knowledge graph and vector search
    - Advanced semantic chunking with markdown header awareness
    - Semantic weight calculation and dynamic thresholds
    - Enhanced error handling and recovery

    Architecture follows clean separation of concerns with no monkey-patching.
    All advanced features are properly initialized and encapsulated.

    Example:
        ```python
        # Basic usage
        rag = AdvancedLightRAG(
            working_dir="./storage",
            query_log_file_path="./logs/queries.log",
            enable_mix_mode=True,
            enable_relationship_types=True,
            use_advanced_chunking=True
        )

        # Query with retrieval details
        response, details = await rag.aquery("What is machine learning?")
        print(f"Retrieved {details.get('retrieved_entities_count', 0)} entities")
        ```

    Migration from standard LightRAG:
        - Replace `LightRAG` with `AdvancedLightRAG`
        - Update query calls to handle the new tuple return format
        - Optionally configure advanced features as needed
    """

    def __init__(
        self,
        # Query Logging Parameters
        query_log_file_path: str = "lightrag_queries.log",
        query_log_max_file_size_bytes: int = 10 * 1024 * 1024,  # 10 MB
        query_log_backup_count: int = 5,
        query_log_level: LogLevel = LogLevel.STANDARD,
        query_log_archive_dir: Optional[str] = None,
        query_log_retention_days: Optional[int] = None,
        # Advanced Features
        enable_mix_mode: bool = True,
        enable_relationship_types: bool = True,
        relationship_types_count: int = 83,
        enable_semantic_weights: bool = True,
        enable_retrieval_details: bool = True,
        use_advanced_chunking: bool = True,
        **kwargs,
    ):
        """
        Initialize AdvancedLightRAG with enhanced capabilities.

        Args:
            query_log_file_path: Path for query log files
            query_log_max_file_size_bytes: Max log file size before rotation (default: 10MB)
            query_log_backup_count: Number of backup log files to keep (default: 5)
            query_log_level: Detail level for query logging (VERBOSE/STANDARD/MINIMAL)
            query_log_archive_dir: Optional directory for compressed log archives
            query_log_retention_days: Days to retain archived logs (None = forever)
            enable_mix_mode: Enable hybrid KG + vector search capabilities
            enable_relationship_types: Use 83 typed relationships vs generic "related"
            relationship_types_count: Number of relationship types in registry
            enable_semantic_weights: Enable semantic similarity weighting
            enable_retrieval_details: Return detailed retrieval information
            use_advanced_chunking: Use advanced semantic chunking with markdown awareness
            **kwargs: Additional parameters for base LightRAG
        """
        # Store configuration
        self.query_log_file_path = query_log_file_path
        self.query_log_max_file_size_bytes = query_log_max_file_size_bytes
        self.query_log_backup_count = query_log_backup_count
        self.query_log_level = query_log_level
        self.query_log_archive_dir = query_log_archive_dir
        self.query_log_retention_days = query_log_retention_days

        self.enable_mix_mode = enable_mix_mode
        self.enable_relationship_types = enable_relationship_types
        self.relationship_types_count = relationship_types_count
        self.enable_semantic_weights = enable_semantic_weights
        self.enable_retrieval_details = enable_retrieval_details
        self.use_advanced_chunking = use_advanced_chunking

        # Set advanced chunking function if enabled
        if use_advanced_chunking:
            kwargs["chunking_func"] = advanced_semantic_chunking
            logger.info("🧠 Advanced semantic chunking enabled")

        # Initialize parent class first
        super().__init__(**kwargs)

        # Read enhanced relationship filter configuration from environment
        import os as os_module  # Avoid any potential name conflicts

        self.enable_enhanced_relationship_filter = os_module.getenv(
            "ENABLE_ENHANCED_RELATIONSHIP_FILTER", "false"
        ).lower() in ("true", "1", "yes", "on")

        self.log_relationship_classification = os_module.getenv(
            "LOG_RELATIONSHIP_CLASSIFICATION", "false"
        ).lower() in ("true", "1", "yes", "on")

        self.relationship_filter_performance_tracking = os_module.getenv(
            "RELATIONSHIP_FILTER_PERFORMANCE_TRACKING", "true"
        ).lower() in ("true", "1", "yes", "on")

        self.enhanced_filter_console_logging = os_module.getenv(
            "ENHANCED_FILTER_CONSOLE_LOGGING", "false"
        ).lower() in ("true", "1", "yes", "on")

        self.enhanced_filter_monitoring_mode = os_module.getenv(
            "ENHANCED_FILTER_MONITORING_MODE", "false"
        ).lower() in ("true", "1", "yes", "on")

        # Log post-processing configuration status (after initialization)
        chunk_processing_enabled = getattr(self, "enable_chunk_post_processing", False)
        llm_processing_enabled = getattr(self, "enable_llm_post_processing", True)

        if chunk_processing_enabled:
            logger.info("✅ Chunk-level relationship post-processing enabled")
        else:
            logger.info("❌ Chunk-level relationship post-processing disabled")

        if llm_processing_enabled:
            logger.info("✅ Document-level LLM post-processing enabled")
        else:
            logger.info("❌ Document-level LLM post-processing disabled")

        # Log enhanced relationship filter configuration status
        enhanced_filter_enabled = self.enable_enhanced_relationship_filter
        log_classification = self.log_relationship_classification
        track_performance = self.relationship_filter_performance_tracking
        console_logging = self.enhanced_filter_console_logging

        if enhanced_filter_enabled:
            logger.info("🎯 Enhanced Relationship Filter: ENABLED")
            logger.info(
                f"   📊 Performance Tracking: {'ENABLED' if track_performance else 'DISABLED'}"
            )
            logger.info(
                f"   🔍 Classification Logging: {'ENABLED' if log_classification else 'DISABLED'}"
            )

            # Show the data-driven categories
            logger.info("   🏷️  Data-Driven Categories (based on your Neo4j patterns):")
            logger.info("      • technical_core (USES, INTEGRATES_WITH, RUNS_ON, ...)")
            logger.info(
                "      • development_operations (CREATES, CONFIGURES, DEVELOPS, ...)"
            )
            logger.info(
                "      • troubleshooting_support (TROUBLESHOOTS, DEBUGS, SOLVES, ...)"
            )
            logger.info("      • system_interactions (HOSTS, MANAGES, PROCESSES, ...)")
            logger.info("      • abstract_conceptual (RELATED, AFFECTS, SUPPORTS, ...)")
            logger.info(
                "      • data_flow (READS_FROM, WRITES_TO, EXTRACTS_DATA_FROM, ...)"
            )

            # Initialize enhanced filter logging with full configuration
            try:
                from lightrag.kg.utils.enhanced_filter_logger import (
                    log_enhanced_filter_initialization,
                )

                filter_config = {
                    "enable_enhanced_relationship_filter": enhanced_filter_enabled,
                    "log_relationship_classification": log_classification,
                    "relationship_filter_performance_tracking": track_performance,
                    "enhanced_filter_console_logging": console_logging,
                    # Include other relevant config
                    "enable_chunk_post_processing": chunk_processing_enabled,
                    "enable_llm_post_processing": llm_processing_enabled,
                }
                if console_logging:
                    log_enhanced_filter_initialization(filter_config)
                else:
                    logger.debug(
                        "Enhanced filter logging configured (console output disabled)"
                    )
            except ImportError:
                logger.warning("Enhanced filter logger not available")
        else:
            logger.info(
                "❌ Enhanced Relationship Filter: DISABLED (using basic filtering)"
            )

        # Log logs directory information
        logs_dir = "logs"
        try:
            import os

            if os.path.exists(logs_dir):
                logger.info(f"📁 Enhanced filter logs will be saved to: ./{logs_dir}/")
            else:
                logger.debug(f"Logs directory will be created at: ./{logs_dir}/")
        except Exception:
            pass

        # Initialize components
        self._query_logger: QueryLogger | None = None

        # Initialize relationship registry if enabled
        if self.enable_relationship_types:
            try:
                from lightrag.kg.utils.relationship_registry import (
                    RelationshipTypeRegistry,
                )

                self._relationship_registry = RelationshipTypeRegistry()
                logger.info(
                    f"Initialized relationship registry with {len(self._relationship_registry.registry)} types"
                )
            except ImportError:
                logger.warning(
                    "Relationship registry not available, using default relationships"
                )
                self._relationship_registry = None

        # Initialize semantic utilities if enabled
        if self.enable_semantic_weights:
            try:
                from lightrag.kg.utils.semantic_utils import (
                    calculate_semantic_weight,
                    process_relationship_weight,
                )

                self._calculate_semantic_weight = calculate_semantic_weight
                self._process_relationship_weight = process_relationship_weight
                logger.info("Semantic weight utilities initialized")
            except ImportError:
                logger.warning("Semantic utilities not available")
                self._calculate_semantic_weight = None
                self._process_relationship_weight = None

    async def _init_query_logger(self):
        """Initialize the query logger instance."""
        if self._query_logger is None:
            self._query_logger = await get_query_logger(
                log_file_path=self.query_log_file_path,
                max_file_size_bytes=self.query_log_max_file_size_bytes,
                backup_count=self.query_log_backup_count,
                log_level=self.query_log_level,
                archive_dir=self.query_log_archive_dir,
                retention_days=self.query_log_retention_days,
            )

    async def get_query_logger_instance(self) -> QueryLogger:
        """Get the query logger instance for this LightRAG instance."""
        if not hasattr(self, "_query_logger") or self._query_logger is None:
            await self._init_query_logger()
        return self._query_logger

    def _get_enhanced_config(self) -> dict:
        """Get configuration dict including enhanced filter settings."""
        # Start with base config from dataclass
        config = asdict(self)

        # Add enhanced filter configurations
        config["enable_enhanced_relationship_filter"] = (
            self.enable_enhanced_relationship_filter
        )
        config["log_relationship_classification"] = self.log_relationship_classification
        config["relationship_filter_performance_tracking"] = (
            self.relationship_filter_performance_tracking
        )
        config["enhanced_filter_console_logging"] = self.enhanced_filter_console_logging
        config["enhanced_filter_monitoring_mode"] = self.enhanced_filter_monitoring_mode

        return config

    async def _process_entity_relation_graph(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
        """
        Override the base method to use advanced entity extraction with relationship types.
        """
        try:
            # Use the advanced extraction instead of base extraction
            from lightrag.advanced_operate import extract_entities_with_types

            chunk_results = await extract_entities_with_types(
                chunk,
                global_config=self._get_enhanced_config(),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
            )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships with advanced features: {str(e)}"
            logger.error(error_msg)
            if pipeline_status_lock:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = error_msg
                    pipeline_status["history_messages"].append(error_msg)
            raise e

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> Tuple[str | AsyncIterator[str] | None, Dict[str, Any]]:
        """
        Perform an async query with retrieval details tracking.

        Args:
            query: The query to be executed
            param: Configuration parameters for query execution
            system_prompt: Custom system prompt

        Returns:
            Tuple of (response, retrieval_details) where retrieval_details contains
            comprehensive information about the retrieval process
        """
        # Initialize retrieval details
        start_time = datetime.now()
        error_message = None
        response_obj = None
        retrieval_details_for_log: Dict[str, Any] = {}

        try:
            # Import enhanced query functions
            if self.enable_retrieval_details:
                from lightrag.advanced_operate import (
                    kg_query_with_details,
                    naive_query_with_details,
                    mix_kg_vector_query,
                )

                # Use enhanced query functions that return details
                if param.mode in ["local", "global", "hybrid"]:
                    response_obj, retrieval_details_for_log = (
                        await kg_query_with_details(
                            query.strip(),
                            self.chunk_entity_relation_graph,
                            self.entities_vdb,
                            self.relationships_vdb,
                            self.text_chunks,
                            param,
                            self._get_enhanced_config(),
                            hashing_kv=self.llm_response_cache,
                            system_prompt=system_prompt,
                            chunks_vdb=(
                                self.chunks_vdb if self.enable_mix_mode else None
                            ),
                        )
                    )
                elif param.mode == "naive":
                    response_obj, retrieval_details_for_log = (
                        await naive_query_with_details(
                            query.strip(),
                            self.chunks_vdb,
                            self.text_chunks,
                            param,
                            self._get_enhanced_config(),
                            hashing_kv=self.llm_response_cache,
                            system_prompt=system_prompt,
                        )
                    )
                elif param.mode == "mix" and self.enable_mix_mode:
                    response_obj, retrieval_details_for_log = await mix_kg_vector_query(
                        query.strip(),
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        self.relationships_vdb,
                        self.chunks_vdb,
                        self.text_chunks,
                        param,
                        self._get_enhanced_config(),
                        hashing_kv=self.llm_response_cache,
                        system_prompt=system_prompt,
                    )
                elif param.mode == "bypass":
                    # Direct LLM query without knowledge retrieval
                    use_llm_func = (
                        param.model_func
                        or self._get_enhanced_config()["llm_model_func"]
                    )
                    param.stream = True if param.stream is None else param.stream
                    response_obj = await use_llm_func(
                        query.strip(),
                        system_prompt=system_prompt,
                        history_messages=param.conversation_history,
                        stream=param.stream,
                    )
                    retrieval_details_for_log = {"mode": "bypass", "direct_llm": True}
                else:
                    raise ValueError(f"Unknown mode {param.mode}")
            else:
                # Fallback to standard query functions without details
                from lightrag.operate import kg_query, naive_query

                if param.mode in ["local", "global", "hybrid"]:
                    response_obj = await kg_query(
                        query.strip(),
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        self.relationships_vdb,
                        self.text_chunks,
                        param,
                        self.global_config,
                        hashing_kv=self.llm_response_cache,
                        system_prompt=system_prompt,
                        chunks_vdb=self.chunks_vdb if self.enable_mix_mode else None,
                    )
                    retrieval_details_for_log = {
                        "mode": param.mode,
                        "top_k": param.top_k,
                        "enable_mix_mode": self.enable_mix_mode,
                    }
                elif param.mode == "naive":
                    response_obj = await naive_query(
                        query.strip(),
                        self.chunks_vdb,
                        self.text_chunks,
                        param,
                        self.global_config,
                        hashing_kv=self.llm_response_cache,
                        system_prompt=system_prompt,
                    )
                    retrieval_details_for_log = {"mode": "naive", "top_k": param.top_k}
                elif param.mode == "bypass":
                    # Direct LLM query without knowledge retrieval
                    use_llm_func = (
                        param.model_func
                        or self._get_enhanced_config()["llm_model_func"]
                    )
                    param.stream = True if param.stream is None else param.stream
                    response_obj = await use_llm_func(
                        query.strip(),
                        system_prompt=system_prompt,
                        history_messages=param.conversation_history,
                        stream=param.stream,
                    )
                    retrieval_details_for_log = {"mode": "bypass", "direct_llm": True}
                else:
                    raise ValueError(f"Unknown mode {param.mode}")

            # Handle response formatting for logging
            final_response_text = ""
            if param.stream and hasattr(response_obj, "__aiter__"):
                final_response_text = "[Streaming Response]"
            elif isinstance(response_obj, str):
                final_response_text = response_obj
            else:
                final_response_text = (
                    str(response_obj) if response_obj is not None else ""
                )
                if response_obj is not None:
                    logger.info(f"Response type: {type(response_obj)}")

        except Exception as e:
            error_message = traceback.format_exc()
            logger.error(f"Error during query execution: {error_message}")
            final_response_text = f"Error: {str(e)}"
            response_obj = final_response_text
            retrieval_details_for_log = {"error": str(e)}

        finally:
            end_time = datetime.now()
            response_time_ms = (end_time - start_time).total_seconds() * 1000

            # Log query if logger is enabled
            q_logger = await self.get_query_logger_instance()
            if q_logger:
                await q_logger.log_query(
                    query_text=query.strip(),
                    response_text=final_response_text,
                    user_id=param.user_id if hasattr(param, "user_id") else None,
                    session_id=(
                        param.session_id if hasattr(param, "session_id") else None
                    ),
                    query_parameters=(
                        param.to_dict() if hasattr(param, "to_dict") else vars(param)
                    ),
                    response_time_ms=response_time_ms,
                    tokens_processed=retrieval_details_for_log.get("tokens_processed"),
                    error_message=error_message,
                    retrieval_details=retrieval_details_for_log,
                )

            await self._query_done()

        return response_obj, retrieval_details_for_log

    def query(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> Tuple[str | AsyncIterator[str] | None, Dict[str, Any]]:
        """
        Sync version of query that returns (response, retrieval_details).
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aquery(query, param, system_prompt))

    async def aquery_with_separate_keyword_extraction(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ) -> Tuple[str | AsyncIterator[str], Dict[str, Any]]:
        """
        Async query with separate keyword extraction that returns retrieval details.
        """
        try:
            if self.enable_retrieval_details:
                from lightrag.advanced_operate import query_with_keywords_and_details

                response, retrieval_details = await query_with_keywords_and_details(
                    query=query,
                    prompt=prompt,
                    param=param,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entities_vdb=self.entities_vdb,
                    relationships_vdb=self.relationships_vdb,
                    chunks_vdb=self.chunks_vdb,
                    text_chunks_db=self.text_chunks,
                    global_config=self._get_enhanced_config(),
                    hashing_kv=self.llm_response_cache,
                )
            else:
                from lightrag.operate import query_with_keywords

                response = await query_with_keywords(
                    query=query,
                    prompt=prompt,
                    param=param,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entities_vdb=self.entities_vdb,
                    relationships_vdb=self.relationships_vdb,
                    chunks_vdb=self.chunks_vdb,
                    text_chunks_db=self.text_chunks,
                    global_config=self._get_enhanced_config(),
                    hashing_kv=self.llm_response_cache,
                )

                # Create basic retrieval details for logging
                retrieval_details = {
                    "mode": param.mode,
                    "top_k": param.top_k,
                    "separate_keyword_extraction": True,
                }
        except Exception as e:
            logger.error(f"Error in keyword extraction query: {str(e)}")
            response = f"Error: {str(e)}"
            retrieval_details = {"error": str(e), "separate_keyword_extraction": True}

        # Log query
        q_logger = await self.get_query_logger_instance()
        if q_logger:
            response_str_for_log = ""
            if isinstance(response, str):
                response_str_for_log = response

            await q_logger.log_query(
                query_text=query.strip(),
                response_text=response_str_for_log,
                user_id=param.user_id if hasattr(param, "user_id") else None,
                session_id=param.session_id if hasattr(param, "session_id") else None,
                query_parameters=(
                    param.to_dict() if hasattr(param, "to_dict") else vars(param)
                ),
                retrieval_details=retrieval_details,
            )

        await self._query_done()
        return response, retrieval_details

    async def aquery_with_separate_keyword_extraction_enhanced(
        self, query: str, prompt: str, param: QueryParam = QueryParam()
    ) -> Tuple[str | AsyncIterator[str], Dict[str, Any]]:
        """
        Async query with separate keyword extraction that returns retrieval details.
        """
        try:
            if self.enable_retrieval_details:
                from lightrag.advanced_operate import query_with_keywords_and_details

                response, retrieval_details = await query_with_keywords_and_details(
                    query=query,
                    prompt=prompt,
                    param=param,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entities_vdb=self.entities_vdb,
                    relationships_vdb=self.relationships_vdb,
                    chunks_vdb=self.chunks_vdb,
                    text_chunks_db=self.text_chunks,
                    global_config=self._get_enhanced_config(),
                    hashing_kv=self.llm_response_cache,
                )
            else:
                from lightrag.operate import query_with_keywords

                response = await query_with_keywords(
                    query=query,
                    prompt=prompt,
                    param=param,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entities_vdb=self.entities_vdb,
                    relationships_vdb=self.relationships_vdb,
                    chunks_vdb=self.chunks_vdb,
                    text_chunks_db=self.text_chunks,
                    global_config=self._get_enhanced_config(),
                    hashing_kv=self.llm_response_cache,
                )

                # Create basic retrieval details for logging
                retrieval_details = {
                    "mode": param.mode,
                    "top_k": param.top_k,
                    "separate_keyword_extraction": True,
                }
        except Exception as e:
            logger.error(f"Error in keyword extraction query: {str(e)}")
            response = f"Error: {str(e)}"
            retrieval_details = {"error": str(e), "separate_keyword_extraction": True}

        # Log query
        q_logger = await self.get_query_logger_instance()
        if q_logger:
            response_str_for_log = ""
            if isinstance(response, str):
                response_str_for_log = response

            await q_logger.log_query(
                query_text=query.strip(),
                response_text=response_str_for_log,
                user_id=param.user_id if hasattr(param, "user_id") else None,
                session_id=param.session_id if hasattr(param, "session_id") else None,
                query_parameters=(
                    param.to_dict() if hasattr(param, "to_dict") else vars(param)
                ),
                retrieval_details=retrieval_details,
            )

        await self._query_done()
        return response, retrieval_details


def create_advanced_lightrag(
    working_dir: str,
    enable_all_features: bool = True,
    query_log_level: LogLevel = LogLevel.STANDARD,
    **kwargs,
) -> AdvancedLightRAG:
    """
    Factory function to create AdvancedLightRAG with sensible defaults.

    Args:
        working_dir: Directory for storage and logs
        enable_all_features: Enable all advanced features by default
        query_log_level: Logging detail level
        **kwargs: Additional configuration parameters

    Returns:
        Configured AdvancedLightRAG instance

    Example:
        ```python
        rag = create_advanced_lightrag(
            working_dir="./storage",
            query_log_level=LogLevel.VERBOSE
        )
        ```
    """
    return AdvancedLightRAG(
        working_dir=working_dir,
        query_log_file_path=f"{working_dir}/queries.log",
        query_log_level=query_log_level,
        enable_mix_mode=enable_all_features,
        enable_relationship_types=enable_all_features,
        enable_semantic_weights=enable_all_features,
        enable_retrieval_details=enable_all_features,
        use_advanced_chunking=enable_all_features,
        **kwargs,
    )
