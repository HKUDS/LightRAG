"""
Advanced Query Operations for LightRAG

This module provides enhanced query functions that return detailed retrieval information
alongside responses. It's part of the migration strategy to move away from monkey-patching
and provide clean, maintainable advanced features.

Key Features:
- All query functions return (response, retrieval_details) tuples
- Comprehensive timing and metrics tracking
- Support for typed relationships via registry system
- Hybrid mix mode for combining KG and vector search
- Advanced semantic chunking with markdown header awareness
- Backward compatibility with base LightRAG operations

Architecture:
- Imports base functions from lightrag.operate
- Enhances them with detailed tracking and metrics
- Uses relationship registry when available
- Maintains clean separation of concerns

Usage:
    ```python
    from lightrag.advanced_operate import kg_query_with_details, advanced_semantic_chunking

    response, details = await kg_query_with_details(
        query="What is machine learning?",
        knowledge_graph_inst=kg,
        entities_vdb=entities_vdb,
        # ... other parameters
    )

    print(f"Retrieved {details['retrieved_entities_count']} entities")
    print(f"Query took {details['timings']['total_ms']}ms")
    ```

Migration Notes:
- This module replaces the old monkey-patching approach
- All functions are async and follow consistent patterns
- Error handling is comprehensive and provides detailed feedback
- Performance is optimized with parallel execution where possible
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from lightrag.operate import (
    _build_query_context,
    _get_node_data,
    _get_edge_data,
    get_keywords_from_query,
    extract_entities as base_extract_entities,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    get_conversation_turns,
    use_llm_func_with_cache,
    logger,
    truncate_list_by_token_size,
)
from lightrag.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, QueryParam
from lightrag.prompt import PROMPTS, GRAPH_FIELD_SEP
from lightrag.utils import Tokenizer
from lightrag.kg.utils.relationship_registry import standardize_relationship_type


def advanced_semantic_chunking(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> list[dict[str, Any]]:
    """
    Advanced semantic chunking function that intelligently splits text based on
    markdown headers with sophisticated fallback handling.

    This enhanced version provides superior RAG performance by:
    1. Preserving semantic boundaries via markdown header analysis
    2. Using token-aware splitting for optimal embedding model performance
    3. Adding enhanced context for sub-chunks when sections are oversized
    4. Graceful fallback to character-based splitting when dependencies unavailable
    5. Comprehensive error handling and logging

    Args:
        tokenizer: Tokenizer instance for token counting
        content: Text content to be chunked
        split_by_character: Optional character to split on (used in fallback)
        split_by_character_only: Whether to only split on the specified character
        overlap_token_size: Token overlap between chunks (for fallback methods)
        max_token_size: Maximum tokens per chunk

    Returns:
        List of chunk dictionaries with text content and metadata

    Example:
        ```python
        tokenizer = TiktokenTokenizer()
        chunks = advanced_semantic_chunking(
            tokenizer=tokenizer,
            content=markdown_text,
            max_token_size=512,
            overlap_token_size=64
        )

        for chunk in chunks:
            print(f"Chunk: {chunk['content'][:100]}...")
            print(f"Metadata: {chunk.get('metadata', {})}")
        ```
    """
    if not content or not content.strip():
        logger.warning("Empty content provided to advanced_semantic_chunking")
        return []

    logger.info(
        f"[advanced_semantic_chunking] Processing content of length: {len(content)}"
    )

    try:
        # Try to import required libraries for advanced chunking
        try:
            from langchain.text_splitter import (
                MarkdownHeaderTextSplitter,
                RecursiveCharacterTextSplitter,
            )

            has_langchain = True
            logger.debug("✓ LangChain available for advanced semantic chunking")
        except ImportError:
            logger.warning(
                "⚠️ LangChain not available! Advanced semantic chunking disabled. "
                "Install with: pip install langchain langchain-text-splitters"
            )
            has_langchain = False

        # Check for tiktoken for optimal token counting
        try:
            import tiktoken

            has_tiktoken = True
            encoding = tiktoken.get_encoding("cl100k_base")
            logger.debug("✓ tiktoken available for precise token counting")

            def token_counter(text: str) -> int:
                return len(encoding.encode(text))

        except ImportError:
            logger.warning(
                "⚠️ tiktoken not available! Using tokenizer fallback. "
                "Install with: pip install tiktoken"
            )
            has_tiktoken = False

            def token_counter(text: str) -> int:
                return len(tokenizer.encode(text))

        # If advanced libraries unavailable, use fallback
        if not has_langchain:
            logger.info("📄 Using fallback chunking method")
            return _fallback_chunking(
                tokenizer, content, max_token_size, overlap_token_size
            )

        # Configure markdown header splitting
        headers_to_split_on = [
            ("#", "header_1"),  # Primary sections (H1)
            ("##", "header_2"),  # Secondary sections (H2)
            ("###", "header_3"),  # Tertiary sections (H3)
        ]

        # Create markdown splitter
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,  # Keep headers for context
        )

        # Split content by headers
        logger.debug("🔍 Analyzing markdown structure...")
        header_splits = markdown_splitter.split_text(content)
        logger.info(f"📑 Found {len(header_splits)} header-based sections")

        # Configure recursive splitter for oversized sections
        if has_tiktoken:
            recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=max_token_size,
                chunk_overlap=overlap_token_size,
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
            )
        else:
            # Convert token sizes to approximate character sizes (rough estimate: 1 token ≈ 4 chars)
            char_chunk_size = max_token_size * 4
            char_overlap_size = overlap_token_size * 4

            recursive_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", " ", ""],
                chunk_size=char_chunk_size,
                chunk_overlap=char_overlap_size,
                length_function=len,
            )

        # Process each header section
        final_chunks = []
        for chunk_doc in header_splits:
            section_content = chunk_doc.page_content.strip()
            metadata = chunk_doc.metadata

            if not section_content:
                continue

            # Extract header hierarchy
            header_1 = metadata.get("header_1", "")
            header_2 = metadata.get("header_2", "")
            header_3 = metadata.get("header_3", "")

            # Determine primary section identifier
            section_title = header_3 or header_2 or header_1 or "Content Section"
            header_level = 3 if header_3 else (2 if header_2 else 1)

            # Check if section needs secondary splitting
            section_tokens = token_counter(section_content)
            logger.debug(f"📊 Section '{section_title}': {section_tokens} tokens")

            if section_tokens <= max_token_size:
                # Section fits in one chunk - keep intact
                final_chunks.append(
                    {
                        "content": section_content,
                        "metadata": {
                            "header_1": header_1,
                            "header_2": header_2,
                            "header_3": header_3,
                            "section_title": section_title,
                            "header_level": header_level,
                            "is_subsection": False,
                            "token_count": section_tokens,
                            "chunk_method": "semantic_header",
                        },
                    }
                )
                logger.debug(f"✓ Kept section '{section_title}' as single chunk")

            else:
                # Section too large - apply intelligent sub-splitting
                logger.info(
                    f"🔄 Sub-splitting large section '{section_title}' ({section_tokens} tokens)"
                )

                # Create context prefix for better standalone comprehension
                section_prefix = section_content[
                    : min(200, len(section_content))
                ].strip()
                if len(section_prefix) < len(section_content):
                    # Find a good breaking point
                    if ". " in section_prefix[-50:]:
                        section_prefix = section_prefix[
                            : section_prefix.rfind(". ") + 1
                        ]
                    section_prefix += "..."

                # Split the content
                sub_chunks = recursive_splitter.split_text(section_content)
                logger.debug(f"📄 Split into {len(sub_chunks)} sub-chunks")

                for i, sub_chunk in enumerate(sub_chunks):
                    # Enhanced context for better standalone understanding
                    contextual_content = sub_chunk
                    if i > 0 and section_prefix:
                        # Add section context to non-first chunks
                        contextual_content = (
                            f"[Section context: {section_prefix}]\n\n{sub_chunk}"
                        )

                    sub_chunk_tokens = token_counter(contextual_content)

                    final_chunks.append(
                        {
                            "content": contextual_content,
                            "metadata": {
                                "header_1": header_1,
                                "header_2": header_2,
                                "header_3": header_3,
                                "section_title": section_title,
                                "header_level": header_level,
                                "is_subsection": True,
                                "subsection_index": i + 1,
                                "subsection_total": len(sub_chunks),
                                "section_prefix": section_prefix,
                                "token_count": sub_chunk_tokens,
                                "chunk_method": "semantic_recursive",
                            },
                        }
                    )

        # Filter empty chunks and validate
        final_chunks = [chunk for chunk in final_chunks if chunk["content"].strip()]

        # Convert to LightRAG expected format
        lightrag_chunks = []
        for i, chunk in enumerate(final_chunks):
            chunk_tokens = chunk["metadata"].get(
                "token_count", token_counter(chunk["content"])
            )

            # Use the exact same format as chunking_by_token_size
            lightrag_chunks.append(
                {
                    "content": chunk["content"],
                    "tokens": chunk_tokens,
                }
            )

        logger.info(
            f"✅ Advanced semantic chunking complete: {len(lightrag_chunks)} chunks created"
        )

        # Log chunk statistics
        total_tokens = sum(chunk["tokens"] for chunk in lightrag_chunks)
        avg_tokens = total_tokens / len(lightrag_chunks) if lightrag_chunks else 0
        logger.debug(
            f"📈 Chunk stats - Total: {total_tokens} tokens, Average: {avg_tokens:.1f} tokens/chunk"
        )

        return lightrag_chunks

    except Exception as e:
        logger.error(f"❌ Error in advanced_semantic_chunking: {str(e)}")
        logger.error(traceback.format_exc())
        logger.warning("🔄 Falling back to simple chunking due to error")
        return _fallback_chunking(
            tokenizer, content, max_token_size, overlap_token_size
        )


def _fallback_chunking(
    tokenizer: Tokenizer, content: str, max_token_size: int, overlap_token_size: int
) -> list[dict[str, Any]]:
    """
    Fallback chunking method when advanced libraries are unavailable.
    Uses simple token-based splitting with overlap.
    """
    logger.info("📝 Using simple token-based fallback chunking")

    # Use the original chunking_by_token_size logic as fallback
    from lightrag.operate import chunking_by_token_size

    fallback_chunks = chunking_by_token_size(
        tokenizer=tokenizer,
        content=content,
        split_by_character=None,
        split_by_character_only=False,
        overlap_token_size=overlap_token_size,
        max_token_size=max_token_size,
    )

    # The original chunking_by_token_size already returns the correct format
    # Just ensure we're not adding any extra fields
    clean_chunks = []
    for chunk in fallback_chunks:
        clean_chunks.append(
            {
                "content": chunk["content"],
                "tokens": chunk["tokens"],
            }
        )

    logger.info(f"📋 Fallback chunking complete: {len(clean_chunks)} chunks")
    return clean_chunks


def combine_contexts(entities_lists, relations_lists, text_units_lists):
    """
    Simple function to combine multiple context lists.

    Args:
        entities_lists: List of entity context lists
        relations_lists: List of relation context lists
        text_units_lists: List of text unit context lists

    Returns:
        Tuple of (combined_entities, combined_relations, combined_text_units)
    """
    # Combine entities (remove duplicates by entity name)
    combined_entities = []
    seen_entities = set()
    for entity_list in entities_lists:
        if isinstance(entity_list, list):
            for entity in entity_list:
                entity_name = entity.get("entity", entity.get("entity_name", ""))
                if entity_name and entity_name not in seen_entities:
                    seen_entities.add(entity_name)
                    combined_entities.append(entity)

    # Combine relations (remove duplicates by src-tgt pair)
    combined_relations = []
    seen_relations = set()
    for relation_list in relations_lists:
        if isinstance(relation_list, list):
            for relation in relation_list:
                src = relation.get("entity1", relation.get("src_id", ""))
                tgt = relation.get("entity2", relation.get("tgt_id", ""))
                relation_key = f"{src}-{tgt}"
                if relation_key and relation_key not in seen_relations:
                    seen_relations.add(relation_key)
                    combined_relations.append(relation)

    # Combine text units (remove duplicates by id)
    combined_text_units = []
    seen_text_units = set()
    for text_unit_list in text_units_lists:
        if isinstance(text_unit_list, list):
            for text_unit in text_unit_list:
                text_id = text_unit.get("id", str(text_unit.get("chunk_id", "")))
                if text_id and text_id not in seen_text_units:
                    seen_text_units.add(text_id)
                    combined_text_units.append(text_unit)

    return combined_entities, combined_relations, combined_text_units


def get_relationship_types() -> List[str]:
    """
    Get relationship types from the registry if available, otherwise return default types.

    This provides backward compatibility while supporting the registry system.
    When the relationship registry is available, it uses the configured types.
    Otherwise, it falls back to a hardcoded list of 83 relationship types.

    Returns:
        List of available relationship type strings

    Example:
        ```python
        types = get_relationship_types()
        print(f"Available types: {len(types)}")  # 83 or registry count
        ```
    """
    try:
        from lightrag.kg.utils.relationship_registry import RelationshipTypeRegistry

        registry = RelationshipTypeRegistry()
        return list(registry.registry.keys())
    except ImportError:
        # Fallback to hardcoded types for backward compatibility
        return [
            "related",
            "associated",
            "influences",
            "uses",
            "part of",
            "depends on",
            "creates",
            "generates",
            "impacts",
            "derived from",
            "requires",
            "provides",
            "connects",
            "enables",
            "facilitates",
            "improves",
            "supports",
            "maintains",
            "implements",
            "extends",
            "inherits",
            "contains",
            "includes",
            "consists of",
            "belongs to",
            "owned by",
            "managed by",
            "controlled by",
            "supervises",
            "reports to",
            "collaborates",
            "interacts",
            "triggers",
            "causes",
            "prevents",
            "conflicts",
            "contradicts",
            "opposes",
            "competes",
            "differs",
            "resembles",
            "similar to",
            "equivalent",
            "represents",
            "models",
            "simulates",
            "emulates",
            "transcends",
            "encompasses",
            "categorizes",
            "characterizes",
            "defines",
            "describes",
            "explains",
            "demonstrates",
            "validates",
            "verifies",
            "confirms",
            "denies",
            "refutes",
            "challenges",
            "questions",
            "analyzes",
            "evaluates",
            "measures",
            "quantifies",
            "converts",
            "transforms",
            "translates",
            "adapts",
            "modifies",
            "customizes",
            "configures",
            "optimizes",
            "enhances",
            "upgrades",
            "replaces",
            "succeeds",
            "precedes",
            "follows",
            "leads to",
            "results in",
            "contributes",
            "participates",
            "engages",
            "involves",
            "attracts",
            "recommends",
            "suggests",
        ]


async def kg_query_with_details(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
    chunks_vdb: BaseVectorStorage | None = None,
) -> Tuple[str | AsyncIterator[str] | None, Dict[str, Any]]:
    """
    Enhanced kg_query that returns both response and retrieval details.
    """
    logger.info("[kg_query_with_details] Starting KG query with details tracking")

    # Initialize retrieval details
    retrieval_details = {"timings": {}}

    # Get model function
    use_model_func = (
        query_param.model_func
        if query_param.model_func
        else global_config["llm_model_func"]
    )

    # Handle cache
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response, {"cached": True}

    # Extract keywords with timing
    t_start_keywords = time.perf_counter()
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv
    )
    retrieval_details["timings"]["keyword_extraction_ms"] = (
        time.perf_counter() - t_start_keywords
    ) * 1000
    retrieval_details["effective_hl_keywords"] = hl_keywords
    retrieval_details["effective_ll_keywords"] = ll_keywords

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level keywords: {ll_keywords}")

    # Handle empty keywords
    if not hl_keywords and not ll_keywords:
        logger.warning("No keywords extracted")
        return PROMPTS["fail_response"], retrieval_details

    # Adjust mode based on available keywords
    if not ll_keywords and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            f"No low-level keywords, switching from {query_param.mode} to global mode"
        )
        query_param.mode = "global"
    if not hl_keywords and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            f"No high-level keywords, switching from {query_param.mode} to local mode"
        )
        query_param.mode = "local"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build context with timing
    t_start_context = time.perf_counter()
    context_str, context_retrieval_details = await _build_query_context_with_details(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )
    retrieval_details["timings"]["context_build_ms"] = (
        time.perf_counter() - t_start_context
    ) * 1000
    retrieval_details.update(context_retrieval_details)

    if query_param.only_need_context:
        return context_str, retrieval_details

    if context_str is None:
        logger.warning("Context building failed")
        return PROMPTS["fail_response"], retrieval_details

    # Build system prompt
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # Build user prompt
    user_prompt = (
        query_param.user_prompt
        if query_param.user_prompt
        else PROMPTS["DEFAULT_USER_PROMPT"]
    )

    sys_prompt = sys_prompt_temp.format(
        context_data=context_str,
        response_type=query_param.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    if query_param.only_need_prompt:
        return sys_prompt, retrieval_details

    # Generate response with timing
    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(f"[kg_query_with_details] Prompt Tokens: {len_of_prompts}")

    t_start_llm = time.perf_counter()
    llm_response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    retrieval_details["timings"]["llm_call_ms"] = (
        time.perf_counter() - t_start_llm
    ) * 1000
    retrieval_details["prompt_tokens"] = len_of_prompts

    # Handle streaming response
    if query_param.stream:

        async def logging_wrapper_generator(original_generator):
            chunk_count = 0
            try:
                async for chunk in original_generator:
                    chunk_count += 1
                    yield chunk
                logger.info(f"Streaming completed with {chunk_count} chunks")
            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                raise

        return logging_wrapper_generator(llm_response), retrieval_details

    # Handle non-streaming response
    final_response = llm_response
    if isinstance(llm_response, str):
        # Clean response
        processed_string = llm_response
        if len(processed_string) > len(sys_prompt):
            processed_string = (
                processed_string.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )
        final_response = processed_string

        # Cache response
        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=final_response,
                    prompt=query,
                    quantized=quantized,
                    min_val=min_val,
                    max_val=max_val,
                    mode=query_param.mode,
                    cache_type="query",
                ),
            )

    return final_response, retrieval_details


async def naive_query_with_details(
    query: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> Tuple[str | AsyncIterator[str], Dict[str, Any]]:
    """
    Enhanced naive_query that returns retrieval details.
    """
    retrieval_details = {
        "timings": {},
        "retrieved_chunks_initial_count": 0,
        "retrieved_chunks_after_truncation_count": 0,
        "retrieved_chunks_summary": [],
    }

    # Get model function
    use_model_func = (
        query_param.model_func
        if query_param.model_func
        else global_config["llm_model_func"]
    )

    # Handle cache
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response, {"cached": True}

    # Vector search
    t_start_search = time.perf_counter()
    results = await chunks_vdb.query(
        query, top_k=query_param.top_k, ids=query_param.ids
    )
    retrieval_details["timings"]["vector_search_ms"] = (
        time.perf_counter() - t_start_search
    ) * 1000
    retrieval_details["retrieved_chunks_initial_count"] = len(results)

    if not results:
        return PROMPTS["fail_response"], retrieval_details

    # Get chunk content
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter valid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found")
        return PROMPTS["fail_response"], retrieval_details

    # Truncate chunks
    tokenizer: Tokenizer = global_config["tokenizer"]
    truncated_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
        tokenizer=tokenizer,
    )

    retrieval_details["retrieved_chunks_after_truncation_count"] = len(truncated_chunks)
    retrieval_details["retrieved_chunks_summary"] = [
        {
            "id": c.get("id", r.get("id", "unknown")),
            "score": r.get("score", 0.0),
            "content_summary": (
                (c.get("content", "")[:50] + "...") if c.get("content") else ""
            ),
            "file_path": c.get("file_path", "unknown_source"),
        }
        for c, r in zip(truncated_chunks, results[: len(truncated_chunks)])
    ]

    logger.info(
        f"Naive query: {len(truncated_chunks)} chunks, top_k: {query_param.top_k}"
    )

    # Format chunks
    section = "\n--New Chunk--\n".join(
        [
            f"File path: {c.get('file_path', 'unknown_source')}\n{c['content']}"
            for c in truncated_chunks
        ]
    )

    if query_param.only_need_context:
        return section, retrieval_details

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # Build prompt
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt, retrieval_details

    # Generate response
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    t_start_llm = time.perf_counter()

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    retrieval_details["timings"]["llm_call_ms"] = (
        time.perf_counter() - t_start_llm
    ) * 1000
    retrieval_details["prompt_tokens"] = len_of_prompts

    # Clean and cache response
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    quantized=quantized,
                    min_val=min_val,
                    max_val=max_val,
                    mode=query_param.mode,
                    cache_type="query",
                ),
            )

    return response, retrieval_details


async def mix_kg_vector_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> Tuple[str | AsyncIterator[str] | None, Dict[str, Any]]:
    """
    Hybrid retrieval combining knowledge graph and vector search.
    """
    tokenizer: Tokenizer = global_config["tokenizer"]

    # Initialize retrieval details
    combined_retrieval_details = {
        "timings": {},
        "kg_retrieval_details": {},
        "vector_retrieval_details": {},
        "effective_hl_keywords": [],
        "effective_ll_keywords": [],
    }

    # Get model function
    use_model_func = (
        query_param.model_func
        if query_param.model_func
        else global_config["llm_model_func"]
    )

    # Handle cache
    args_hash = compute_args_hash("mix", query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, "mix", cache_type="query"
    )
    if cached_response is not None:
        return cached_response, {"cached": True}

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # Execute KG and vector searches in parallel
    async def get_kg_context():
        try:
            hl_keywords, ll_keywords = await get_keywords_from_query(
                query, query_param, global_config, hashing_kv
            )

            if not hl_keywords and not ll_keywords:
                logger.warning("No keywords extracted for KG search")
                return None, {}

            # Set query mode based on keywords
            ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
            hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

            if not ll_keywords_str:
                query_param.mode = "global"
            elif not hl_keywords_str:
                query_param.mode = "local"
            else:
                query_param.mode = "hybrid"

            # Build KG context
            context_str, kg_details = await _build_query_context_with_details(
                ll_keywords_str,
                hl_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )

            kg_details["effective_hl_keywords"] = hl_keywords
            kg_details["effective_ll_keywords"] = ll_keywords

            return context_str, kg_details

        except Exception as e:
            logger.error(f"Error in KG context retrieval: {str(e)}")
            traceback.print_exc()
            return None, {"error": str(e)}

    async def get_vector_context() -> Tuple[str | None, Dict[str, Any]]:
        vector_details = {
            "retrieved_chunks_initial_count": 0,
            "retrieved_chunks_after_truncation_count": 0,
            "retrieved_chunks_summary": [],
        }

        try:
            # Include conversation history in vector search
            augmented_query = query
            if history_context:
                augmented_query = f"{history_context}\n{query}"

            # Reduce top_k for hybrid mode
            mix_topk = min(10, query_param.top_k)
            results = await chunks_vdb.query(
                augmented_query, top_k=mix_topk, ids=query_param.ids
            )
            vector_details["retrieved_chunks_initial_count"] = len(results)

            if not results:
                return None, vector_details

            chunks_ids = [r["id"] for r in results]
            chunks = await text_chunks_db.get_by_ids(chunks_ids)

            valid_chunks = []
            for chunk, result in zip(chunks, results):
                if chunk is not None and "content" in chunk:
                    chunk_with_metadata = {
                        "content": chunk["content"],
                        "created_at": result.get("created_at", None),
                        "file_path": result.get(
                            "file_path", chunk.get("file_path", "unknown_source")
                        ),
                    }
                    valid_chunks.append(chunk_with_metadata)

            if not valid_chunks:
                return None, vector_details

            # Truncate chunks
            truncated_chunks = truncate_list_by_token_size(
                valid_chunks,
                key=lambda x: x["content"],
                max_token_size=query_param.max_token_for_text_unit,
                tokenizer=tokenizer,
            )

            vector_details["retrieved_chunks_after_truncation_count"] = len(
                truncated_chunks
            )
            vector_details["retrieved_chunks_summary"] = [
                {
                    "id": c.get("id", res.get("id", "unknown")),
                    "score": res.get("score", 0.0),
                    "content_summary": (
                        (c.get("content", "")[:50] + "...") if c.get("content") else ""
                    ),
                    "file_path": c.get("file_path", "unknown_source"),
                }
                for c, res in zip(truncated_chunks, results[: len(truncated_chunks)])
            ]

            # Format chunks with metadata
            formatted_chunks = []
            for c in truncated_chunks:
                chunk_text = f"File path: {c['file_path']}\n{c['content']}"
                if c.get("created_at"):
                    chunk_text = f"[Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c['created_at']))}]\n{chunk_text}"
                formatted_chunks.append(chunk_text)

            return "\n--New Chunk--\n".join(formatted_chunks), vector_details

        except Exception as e:
            logger.error(f"Error in vector context retrieval: {e}")
            vector_details["error"] = str(e)
            return None, vector_details

    # Execute both retrievals in parallel
    t_start_hybrid = time.perf_counter()
    (kg_context_str, kg_ret_details), (vector_context_str, vec_ret_details) = (
        await asyncio.gather(get_kg_context(), get_vector_context())
    )
    combined_retrieval_details["timings"]["hybrid_retrieval_ms"] = (
        time.perf_counter() - t_start_hybrid
    ) * 1000

    # Populate retrieval details
    combined_retrieval_details["kg_retrieval_details"] = kg_ret_details
    combined_retrieval_details["vector_retrieval_details"] = vec_ret_details

    if kg_ret_details and isinstance(kg_ret_details, dict):
        combined_retrieval_details["effective_hl_keywords"] = kg_ret_details.get(
            "effective_hl_keywords", []
        )
        combined_retrieval_details["effective_ll_keywords"] = kg_ret_details.get(
            "effective_ll_keywords", []
        )

    # Check if we have any context
    if kg_context_str is None and vector_context_str is None:
        return PROMPTS["fail_response"], combined_retrieval_details

    # Return context if requested
    if query_param.only_need_context:
        final_context = f"""
-----Knowledge Graph Context-----
{kg_context_str if kg_context_str else "No relevant knowledge graph information found"}

-----Vector Context-----
{vector_context_str if vector_context_str else "No relevant text information found"}
""".strip()
        return final_context, combined_retrieval_details

    # Build hybrid prompt
    sys_prompt = (
        system_prompt
        if system_prompt
        else PROMPTS.get("mix_rag_response", PROMPTS["rag_response"])
    ).format(
        kg_context=(
            kg_context_str
            if kg_context_str
            else "No relevant knowledge graph information found"
        ),
        vector_context=(
            vector_context_str
            if vector_context_str
            else "No relevant text information found"
        ),
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt, combined_retrieval_details

    # Generate response
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(f"[mix_kg_vector_query] Prompt Tokens: {len_of_prompts}")

    t_start_llm = time.perf_counter()
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    combined_retrieval_details["timings"]["llm_call_ms"] = (
        time.perf_counter() - t_start_llm
    ) * 1000
    combined_retrieval_details["prompt_tokens"] = len_of_prompts

    # Clean and cache response
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        if hashing_kv and hashing_kv.global_config.get("enable_llm_cache"):
            await save_to_cache(
                hashing_kv,
                CacheData(
                    args_hash=args_hash,
                    content=response,
                    prompt=query,
                    quantized=quantized,
                    min_val=min_val,
                    max_val=max_val,
                    mode="mix",
                    cache_type="query",
                ),
            )

    return response, combined_retrieval_details


async def _build_query_context_with_details(
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
) -> Tuple[str | None, Dict[str, Any]]:
    """
    Enhanced context building with retrieval details.
    """
    retrieval_details = {}

    if query_param.mode == "local":
        entities_context, relations_context, text_units_context, details = (
            await _get_node_data_with_details(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            )
        )
        retrieval_details = details
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context, details = (
            await _get_edge_data_with_details(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )
        )
        retrieval_details = details
    else:  # hybrid mode
        ll_data = await _get_node_data_with_details(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
        hl_data = await _get_edge_data_with_details(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

        ll_entities, ll_relations, ll_text_units, ll_details = ll_data
        hl_entities, hl_relations, hl_text_units, hl_details = hl_data

        # Combine contexts
        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities, ll_entities],
            [hl_relations, ll_relations],
            [hl_text_units, ll_text_units],
        )

        # Merge retrieval details
        retrieval_details = {**ll_details, **hl_details}
        retrieval_details["effective_hl_keywords"] = (
            hl_keywords.split(", ") if hl_keywords else []
        )
        retrieval_details["effective_ll_keywords"] = (
            ll_keywords.split(", ") if ll_keywords else []
        )

    # Check if we have any context
    if not entities_context and not relations_context:
        return None, retrieval_details

    # Format as JSON
    entities_str = json.dumps(entities_context, ensure_ascii=False)
    relations_str = json.dumps(relations_context, ensure_ascii=False)
    text_units_str = json.dumps(text_units_context, ensure_ascii=False)

    result = f"""-----Entities-----

```json
{entities_str}
```

-----Relationships-----

```json
{relations_str}
```

-----Sources-----

```json
{text_units_str}
```

"""
    return result, retrieval_details


async def _get_node_data_with_details(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Enhanced _get_node_data with retrieval details tracking.
    """
    retrieval_details = {
        "retrieved_entities_initial_count": 0,
        "retrieved_entities_after_truncation_count": 0,
        "retrieved_entities_summary": [],
        "retrieved_relationships_count": 0,
        "retrieved_relationships_summary": [],
        "retrieved_chunks_count": 0,
        "retrieved_chunks_summary": [],
        "effective_ll_keywords": query.split(", ") if query else [],
    }

    # Get entities from vector DB
    logger.info(f"Query nodes: {query}, top_k: {query_param.top_k}")
    results = await entities_vdb.query(
        query, top_k=query_param.top_k, ids=query_param.ids
    )
    retrieval_details["retrieved_entities_initial_count"] = len(results)

    if not results:
        return "", "", "", retrieval_details

    # Use existing _get_node_data logic
    from lightrag.operate import _get_node_data

    entities_context, relations_context, text_units_context = await _get_node_data(
        query,
        knowledge_graph_inst,
        entities_vdb,
        text_chunks_db,
        query_param,
    )

    # Extract counts from contexts
    if isinstance(entities_context, list):
        retrieval_details["retrieved_entities_after_truncation_count"] = len(
            entities_context
        )
        retrieval_details["retrieved_entities_summary"] = [
            {
                "name": e.get("entity", ""),
                "type": e.get("type", "UNKNOWN"),
                "rank": e.get("rank", 0),
                "description_summary": (
                    (e.get("description", "")[:50] + "...")
                    if e.get("description")
                    else ""
                ),
            }
            for e in entities_context
        ]

    if isinstance(relations_context, list):
        retrieval_details["retrieved_relationships_count"] = len(relations_context)
        retrieval_details["retrieved_relationships_summary"] = [
            {
                "source": r.get("entity1", ""),
                "target": r.get("entity2", ""),
                "weight": r.get("weight", 0.0),
                "description_summary": (
                    (r.get("description", "")[:50] + "...")
                    if r.get("description")
                    else ""
                ),
            }
            for r in relations_context
        ]

    if isinstance(text_units_context, list):
        retrieval_details["retrieved_chunks_count"] = len(text_units_context)
        retrieval_details["retrieved_chunks_summary"] = [
            {
                "id": str(c.get("id", "unknown")),
                "content_summary": (
                    (c.get("content", "")[:50] + "...") if c.get("content") else ""
                ),
                "file_path": c.get("file_path", "unknown_source"),
            }
            for c in text_units_context
        ]

    return entities_context, relations_context, text_units_context, retrieval_details


async def _get_edge_data_with_details(
    keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Enhanced _get_edge_data with retrieval details tracking.
    """
    retrieval_details = {
        "retrieved_entities_initial_count": 0,
        "retrieved_entities_after_truncation_count": 0,
        "retrieved_entities_summary": [],
        "retrieved_relationships_initial_count": 0,
        "retrieved_relationships_after_truncation_count": 0,
        "retrieved_relationships_summary": [],
        "retrieved_chunks_count": 0,
        "retrieved_chunks_summary": [],
        "effective_hl_keywords": keywords.split(", ") if keywords else [],
    }

    # Get relationships from vector DB
    logger.info(f"Query edges: {keywords}, top_k: {query_param.top_k}")
    results = await relationships_vdb.query(
        keywords, top_k=query_param.top_k, ids=query_param.ids
    )
    retrieval_details["retrieved_relationships_initial_count"] = len(results)

    if not results:
        return "", "", "", retrieval_details

    # Use existing _get_edge_data logic
    from lightrag.operate import _get_edge_data

    entities_context, relations_context, text_units_context = await _get_edge_data(
        keywords,
        knowledge_graph_inst,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )

    # Extract details from contexts
    if isinstance(entities_context, list):
        retrieval_details["retrieved_entities_after_truncation_count"] = len(
            entities_context
        )
        retrieval_details["retrieved_entities_summary"] = [
            {
                "name": e.get("entity", ""),
                "type": e.get("type", "UNKNOWN"),
                "rank": e.get("rank", 0),
                "description_summary": (
                    (e.get("description", "")[:50] + "...")
                    if e.get("description")
                    else ""
                ),
            }
            for e in entities_context
        ]

    if isinstance(relations_context, list):
        retrieval_details["retrieved_relationships_after_truncation_count"] = len(
            relations_context
        )
        retrieval_details["retrieved_relationships_summary"] = [
            {
                "source": r.get("entity1", ""),
                "target": r.get("entity2", ""),
                "weight": r.get("weight", 0.0),
                "description_summary": (
                    (r.get("description", "")[:50] + "...")
                    if r.get("description")
                    else ""
                ),
            }
            for r in relations_context
        ]

    if isinstance(text_units_context, list):
        retrieval_details["retrieved_chunks_count"] = len(text_units_context)
        retrieval_details["retrieved_chunks_summary"] = [
            {
                "id": str(c.get("id", "unknown")),
                "content_summary": (
                    (c.get("content", "")[:50] + "...") if c.get("content") else ""
                ),
                "file_path": c.get("file_path", "unknown_source"),
            }
            for c in text_units_context
        ]

    return entities_context, relations_context, text_units_context, retrieval_details


async def query_with_keywords_and_details(
    query: str,
    prompt: str,
    param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> Tuple[str | AsyncIterator[str], Dict[str, Any]]:
    """
    Enhanced query_with_keywords that returns retrieval details.
    """
    # Extract keywords
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query=query,
        query_param=param,
        global_config=global_config,
        hashing_kv=hashing_kv,
    )

    # Format query with keywords
    ll_keywords_str = ", ".join(ll_keywords)
    hl_keywords_str = ", ".join(hl_keywords)
    formatted_question = f"{prompt}\n\n### Keywords:\nHigh-level: {hl_keywords_str}\nLow-level: {ll_keywords_str}\n\n### Query:\n{query}"

    # Store original query for mix mode
    if hasattr(param, "original_query"):
        param.original_query = query

    # Route to appropriate query function
    if param.mode in ["local", "global", "hybrid"]:
        return await kg_query_with_details(
            formatted_question,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    elif param.mode == "naive":
        return await naive_query_with_details(
            formatted_question,
            chunks_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    elif param.mode == "mix":
        return await mix_kg_vector_query(
            formatted_question,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            chunks_vdb,
            text_chunks_db,
            param,
            global_config,
            hashing_kv=hashing_kv,
        )
    else:
        raise ValueError(f"Unknown mode {param.mode}")


async def extract_entities_with_types(
    chunks: dict[str, Any],
    global_config: dict[str, str],
    pipeline_status: dict = None,
    pipeline_status_lock=None,
    llm_response_cache: BaseKVStorage | None = None,
) -> list:
    """
    Enhanced entity extraction that uses the robust relationship type standardization
    from the relationship registry for consistent formatting.
    """
    logger.info("Using enhanced relationship type standardization from registry")

    # Get base extraction results without domain-specific enhancement
    chunk_results = await base_extract_entities(
        chunks,
        global_config,
        pipeline_status,
        pipeline_status_lock,
        llm_response_cache,
    )

    # Post-process to standardize relationship types for Neo4j
    for maybe_nodes, maybe_edges in chunk_results:
        # Process edges to format relationship types for Neo4j
        for edge_key, edges in maybe_edges.items():
            for edge in edges:
                # Get original relationship type from LLM
                original_rel_type = edge.get("relationship_type", "related")

                # Apply enhanced Neo4j standardization from registry
                neo4j_type = standardize_relationship_type(original_rel_type)

                # Log the transformation for transparency
                if original_rel_type != neo4j_type.lower().replace("_", " "):
                    logger.debug(
                        f"Standardized relationship: '{original_rel_type}' -> Neo4j: '{neo4j_type}'"
                    )

                # Update edge with formatted types
                edge["relationship_type"] = neo4j_type.lower().replace(
                    "_", " "
                )  # Human-readable for compatibility
                edge["original_type"] = (
                    original_rel_type  # Preserve original LLM output
                )
                edge["neo4j_type"] = neo4j_type  # Neo4j label format
                edge["formatting_confidence"] = (
                    1.0  # Always high confidence for enhanced standardization
                )

                logger.debug(
                    f"Enhanced standardization: '{original_rel_type}' -> Neo4j: '{neo4j_type}'"
                )

    return chunk_results


def find_closest_relationship_type(rel_type: str) -> str:
    """
    Find the closest matching relationship type using fuzzy matching.
    """
    rel_type_lower = rel_type.lower().strip()

    # Direct match
    for valid_type in get_relationship_types():
        if valid_type.lower() == rel_type_lower:
            return valid_type

    # Partial match
    for valid_type in get_relationship_types():
        if rel_type_lower in valid_type.lower() or valid_type.lower() in rel_type_lower:
            return valid_type

    # Character similarity
    max_similarity = 0
    best_match = "related"

    for valid_type in get_relationship_types():
        # Simple Jaccard similarity
        set1 = set(rel_type_lower)
        set2 = set(valid_type.lower())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        similarity = intersection / union if union > 0 else 0

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = valid_type

    return best_match
