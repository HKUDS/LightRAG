import os
import inspect
from lightrag.utils import logger, get_env_value, EmbeddingFunc
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.constants import DEFAULT_LLM_TIMEOUT, DEFAULT_EMBEDDING_TIMEOUT

class LLMConfigCache:
    """Smart LLM and Embedding configuration cache class"""

    def __init__(self, args):
        self.args = args

        # Initialize configurations based on binding conditions
        self.openai_llm_options = None
        self.gemini_llm_options = None
        self.gemini_embedding_options = None
        self.ollama_llm_options = None
        self.ollama_embedding_options = None

        # Only initialize and log OpenAI options when using OpenAI-related bindings
        if args.llm_binding in ["openai", "azure_openai"]:
            from lightrag.llm.binding_options import OpenAILLMOptions

            self.openai_llm_options = OpenAILLMOptions.options_dict(args)
            logger.info(f"OpenAI LLM Options: {self.openai_llm_options}")

        if args.llm_binding == "gemini":
            from lightrag.llm.binding_options import GeminiLLMOptions

            self.gemini_llm_options = GeminiLLMOptions.options_dict(args)
            logger.info(f"Gemini LLM Options: {self.gemini_llm_options}")

        # Only initialize and log Ollama LLM options when using Ollama LLM binding
        if args.llm_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaLLMOptions

                self.ollama_llm_options = OllamaLLMOptions.options_dict(args)
                logger.info(f"Ollama LLM Options: {self.ollama_llm_options}")
            except ImportError:
                logger.warning(
                    "OllamaLLMOptions not available, using default configuration"
                )
                self.ollama_llm_options = {}

        # Only initialize and log Ollama Embedding options when using Ollama Embedding binding
        if args.embedding_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaEmbeddingOptions

                self.ollama_embedding_options = OllamaEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Ollama Embedding Options: {self.ollama_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "OllamaEmbeddingOptions not available, using default configuration"
                )
                self.ollama_embedding_options = {}

        # Only initialize and log Gemini Embedding options when using Gemini Embedding binding
        if args.embedding_binding == "gemini":
            try:
                from lightrag.llm.binding_options import GeminiEmbeddingOptions

                self.gemini_embedding_options = GeminiEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Gemini Embedding Options: {self.gemini_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "GeminiEmbeddingOptions not available, using default configuration"
                )
                self.gemini_embedding_options = {}

def create_optimized_openai_llm_func(config_cache: LLMConfigCache, args, llm_timeout: int):
    """Create optimized OpenAI LLM function with pre-processed configuration"""

    async def optimized_openai_alike_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        from lightrag.llm.openai import openai_complete_if_cache

        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []

        # Use pre-processed configuration to avoid repeated parsing
        kwargs["timeout"] = llm_timeout
        if config_cache.openai_llm_options:
            kwargs.update(config_cache.openai_llm_options)

        return await openai_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=args.llm_binding_host,
            api_key=args.llm_binding_api_key,
            **kwargs,
        )

    return optimized_openai_alike_model_complete

def create_optimized_azure_openai_llm_func(config_cache: LLMConfigCache, args, llm_timeout: int):
    """Create optimized Azure OpenAI LLM function with pre-processed configuration"""

    async def optimized_azure_openai_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        from lightrag.llm.azure_openai import azure_openai_complete_if_cache

        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []

        # Use pre-processed configuration to avoid repeated parsing
        kwargs["timeout"] = llm_timeout
        if config_cache.openai_llm_options:
            kwargs.update(config_cache.openai_llm_options)

        return await azure_openai_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=args.llm_binding_host,
            api_key=os.getenv("AZURE_OPENAI_API_KEY", args.llm_binding_api_key),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            **kwargs,
        )

    return optimized_azure_openai_model_complete

def create_optimized_gemini_llm_func(config_cache: LLMConfigCache, args, llm_timeout: int):
    """Create optimized Gemini LLM function with cached configuration"""

    async def optimized_gemini_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        from lightrag.llm.gemini import gemini_complete_if_cache

        if history_messages is None:
            history_messages = []

        # Use pre-processed configuration to avoid repeated parsing
        kwargs["timeout"] = llm_timeout
        if (
            config_cache.gemini_llm_options is not None
            and "generation_config" not in kwargs
        ):
            kwargs["generation_config"] = dict(config_cache.gemini_llm_options)

        return await gemini_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=args.llm_binding_api_key,
            base_url=args.llm_binding_host,
            keyword_extraction=keyword_extraction,
            **kwargs,
        )

    return optimized_gemini_model_complete

async def bedrock_model_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    # Lazy import
    from lightrag.llm.bedrock import bedrock_complete_if_cache

    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    if history_messages is None:
        history_messages = []

    # Use global temperature for Bedrock
    kwargs["temperature"] = get_env_value("BEDROCK_LLM_TEMPERATURE", 1.0, float)

    # Need args? No, args not available here easily unless passed?
    # Wait, original code used 'args' from outer scope.
    # We must pass args to this function or make strict args.
    # Actually, bedrock_complete_if_cache takes model_name.
    # Let's see original code:
    # args.llm_model is used.
    # So create_llm_model_func MUST close over args?
    # Or we pass args to create_llm_model_func.
    # Original: `def create_llm_model_func(binding: str):` inside `create_app` which has `args`.
    # I need to change signature of `create_llm_model_func`.
    
    # Wait, `bedrock_model_complete` uses `args.llm_model`.
    # I will modify `create_llm_model_func` to accept `args`.
    pass 
    # Placeholder: see create_llm_model_func below.

def create_llm_model_func(binding: str, args, config_cache: LLMConfigCache, llm_timeout: int):
    """
    Create LLM model function based on binding type.
    """
    try:
        if binding == "lollms":
            from lightrag.llm.lollms import lollms_model_complete
            return lollms_model_complete
        elif binding == "ollama":
            from lightrag.llm.ollama import ollama_model_complete
            return ollama_model_complete
        elif binding == "aws_bedrock":
             # Bedrock needs args.llm_model
             async def _bedrock_wrapper(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs):
                from lightrag.llm.bedrock import bedrock_complete_if_cache
                keyword_extraction = kwargs.pop("keyword_extraction", None)
                if keyword_extraction:
                    kwargs["response_format"] = GPTKeywordExtractionFormat
                if history_messages is None:
                    history_messages = []
                kwargs["temperature"] = get_env_value("BEDROCK_LLM_TEMPERATURE", 1.0, float)
                return await bedrock_complete_if_cache(
                    args.llm_model,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    **kwargs,
                )
             return _bedrock_wrapper
        elif binding == "azure_openai":
            return create_optimized_azure_openai_llm_func(
                config_cache, args, llm_timeout
            )
        elif binding == "gemini":
            return create_optimized_gemini_llm_func(config_cache, args, llm_timeout)
        else:  # openai and compatible
            return create_optimized_openai_llm_func(config_cache, args, llm_timeout)
    except ImportError as e:
        raise Exception(f"Failed to import {binding} LLM binding: {e}")

def create_llm_model_kwargs(binding: str, args, llm_timeout: int) -> dict:
    if binding in ["lollms", "ollama"]:
        try:
            from lightrag.llm.binding_options import OllamaLLMOptions

            return {
                "host": args.llm_binding_host,
                "timeout": llm_timeout,
                "options": OllamaLLMOptions.options_dict(args),
                "api_key": args.llm_binding_api_key,
            }
        except ImportError as e:
            raise Exception(f"Failed to import {binding} options: {e}")
    return {}

def create_optimized_embedding_function(
    config_cache: LLMConfigCache, binding, model, host, api_key, args
) -> EmbeddingFunc:
    # Step 1: Import provider function and extract default attributes
    provider_func = None
    provider_max_token_size = None
    provider_embedding_dim = None

    try:
        if binding == "openai":
            from lightrag.llm.openai import openai_embed
            provider_func = openai_embed
        elif binding == "ollama":
            from lightrag.llm.ollama import ollama_embed
            provider_func = ollama_embed
        elif binding == "gemini":
            from lightrag.llm.gemini import gemini_embed
            provider_func = gemini_embed
        elif binding == "jina":
            from lightrag.llm.jina import jina_embed
            provider_func = jina_embed
        elif binding == "azure_openai":
            from lightrag.llm.azure_openai import azure_openai_embed
            provider_func = azure_openai_embed
        elif binding == "aws_bedrock":
            from lightrag.llm.bedrock import bedrock_embed
            provider_func = bedrock_embed
        elif binding == "lollms":
            from lightrag.llm.lollms import lollms_embed
            provider_func = lollms_embed

        # Extract attributes if provider is an EmbeddingFunc
        if provider_func and isinstance(provider_func, EmbeddingFunc):
            provider_max_token_size = provider_func.max_token_size
            provider_embedding_dim = provider_func.embedding_dim
            logger.debug(
                f"Extracted from {binding} provider: "
                f"max_token_size={provider_max_token_size}, "
                f"embedding_dim={provider_embedding_dim}"
            )
    except ImportError as e:
        logger.warning(f"Could not import provider function for {binding}: {e}")

    # Step 2: Apply priority
    final_max_token_size = args.embedding_token_limit or provider_max_token_size
    final_embedding_dim = (
        args.embedding_dim if args.embedding_dim else provider_embedding_dim
    )

    # Step 3: Create optimized embedding function
    async def optimized_embedding_function(texts, embedding_dim=None):
        try:
            if binding == "lollms":
                from lightrag.llm.lollms import lollms_embed
                actual_func = (
                    lollms_embed.func
                    if isinstance(lollms_embed, EmbeddingFunc)
                    else lollms_embed
                )
                return await actual_func(texts, base_url=host, api_key=api_key)
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_embed
                actual_func = (
                    ollama_embed.func
                    if isinstance(ollama_embed, EmbeddingFunc)
                    else ollama_embed
                )
                if config_cache.ollama_embedding_options is not None:
                    ollama_options = config_cache.ollama_embedding_options
                else:
                    from lightrag.llm.binding_options import OllamaEmbeddingOptions
                    ollama_options = OllamaEmbeddingOptions.options_dict(args)

                kwargs = {
                    "texts": texts,
                    "host": host,
                    "api_key": api_key,
                    "options": ollama_options,
                }
                if model:
                    kwargs["embed_model"] = model
                return await actual_func(**kwargs)
            elif binding == "azure_openai":
                from lightrag.llm.azure_openai import azure_openai_embed
                actual_func = (
                    azure_openai_embed.func
                    if isinstance(azure_openai_embed, EmbeddingFunc)
                    else azure_openai_embed
                )
                kwargs = {"texts": texts, "api_key": api_key}
                if model:
                    kwargs["model"] = model
                return await actual_func(**kwargs)
            elif binding == "aws_bedrock":
                from lightrag.llm.bedrock import bedrock_embed
                actual_func = (
                    bedrock_embed.func
                    if isinstance(bedrock_embed, EmbeddingFunc)
                    else bedrock_embed
                )
                kwargs = {"texts": texts}
                if model:
                    kwargs["model"] = model
                return await actual_func(**kwargs)
            elif binding == "jina":
                from lightrag.llm.jina import jina_embed
                actual_func = (
                    jina_embed.func
                    if isinstance(jina_embed, EmbeddingFunc)
                    else jina_embed
                )
                kwargs = {
                    "texts": texts,
                    "embedding_dim": embedding_dim,
                    "base_url": host,
                    "api_key": api_key,
                }
                if model:
                    kwargs["model"] = model
                return await actual_func(**kwargs)
            elif binding == "gemini":
                from lightrag.llm.gemini import gemini_embed
                actual_func = (
                    gemini_embed.func
                    if isinstance(gemini_embed, EmbeddingFunc)
                    else gemini_embed
                )
                if config_cache.gemini_embedding_options is not None:
                    gemini_options = config_cache.gemini_embedding_options
                else:
                    from lightrag.llm.binding_options import GeminiEmbeddingOptions
                    gemini_options = GeminiEmbeddingOptions.options_dict(args)

                kwargs = {
                    "texts": texts,
                    "base_url": host,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                    "task_type": gemini_options.get(
                        "task_type", "RETRIEVAL_DOCUMENT"
                    ),
                }
                if model:
                    kwargs["model"] = model
                return await actual_func(**kwargs)
            else:  # openai and compatible
                from lightrag.llm.openai import openai_embed
                actual_func = (
                    openai_embed.func
                    if isinstance(openai_embed, EmbeddingFunc)
                    else openai_embed
                )
                kwargs = {
                    "texts": texts,
                    "base_url": host,
                    "api_key": api_key,
                    "embedding_dim": embedding_dim,
                }
                if model:
                    kwargs["model"] = model
                return await actual_func(**kwargs)
        except ImportError as e:
            raise Exception(f"Failed to import {binding} embedding: {e}")

    # Step 4: Wrap in EmbeddingFunc and return
    embedding_func_instance = EmbeddingFunc(
        embedding_dim=final_embedding_dim,
        func=optimized_embedding_function,
        max_token_size=final_max_token_size,
        send_dimensions=False,  # Will be set later
        model_name=model,
    )
    
    # Configure Send Dimensions (Logic moved here or handled by caller? Logic moved here)
    # But checking sig requires func...
    sig = inspect.signature(embedding_func_instance.func)
    has_embedding_dim_param = "embedding_dim" in sig.parameters
    
    embedding_send_dim = args.embedding_send_dim
    
    if args.embedding_binding in ["jina", "gemini"]:
        send_dimensions = has_embedding_dim_param
    else:
        send_dimensions = embedding_send_dim and has_embedding_dim_param

    embedding_func_instance.send_dimensions = send_dimensions
    
    logger.info(
        f"Embedding config: binding={binding} model={model} "
        f"embedding_dim={final_embedding_dim} max_token_size={final_max_token_size} "
        f"send_dimensions={send_dimensions}"
    )

    return embedding_func_instance

def create_server_rerank_func(args):
    # Retrieve functions and return logic
    rerank_model_func = None
    if args.rerank_binding != "null":
        from lightrag.rerank import cohere_rerank, jina_rerank, ali_rerank

        rerank_functions = {
            "cohere": cohere_rerank,
            "jina": jina_rerank,
            "aliyun": ali_rerank,
        }

        selected_rerank_func = rerank_functions.get(args.rerank_binding)
        if not selected_rerank_func:
            logger.error(f"Unsupported rerank binding: {args.rerank_binding}")
            raise ValueError(f"Unsupported rerank binding: {args.rerank_binding}")

        # Defaults logic
        if args.rerank_model is None or args.rerank_binding_host is None:
            sig = inspect.signature(selected_rerank_func)
            if args.rerank_model is None and "model" in sig.parameters:
                default_model = sig.parameters["model"].default
                if default_model != inspect.Parameter.empty:
                    args.rerank_model = default_model
            if args.rerank_binding_host is None and "base_url" in sig.parameters:
                default_base_url = sig.parameters["base_url"].default
                if default_base_url != inspect.Parameter.empty:
                    args.rerank_binding_host = default_base_url

        async def server_rerank_func(
            query: str, documents: list, top_n: int = None, extra_body: dict = None
        ):
            kwargs = {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "api_key": args.rerank_binding_api_key,
                "model": args.rerank_model,
                "base_url": args.rerank_binding_host,
            }
            if args.rerank_binding == "cohere":
                kwargs["enable_chunking"] = (
                    os.getenv("RERANK_ENABLE_CHUNKING", "false").lower() == "true"
                )
                kwargs["max_tokens_per_doc"] = int(
                    os.getenv("RERANK_MAX_TOKENS_PER_DOC", "4096")
                )

            return await selected_rerank_func(**kwargs, extra_body=extra_body)

        logger.info(
            f"Reranking is enabled: {args.rerank_model or 'default model'} using {args.rerank_binding} provider"
        )
        return server_rerank_func
    else:
        logger.info("Reranking is disabled")
        return None
