import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug, EmbeddingFunc
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables from .env file
# from dotenv import load_dotenv
# load_dotenv()

# Configuration
WORKING_DIR = "/home/js/LightRAG/jrs/work/mod_linx/_mod_linx_work_dir"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 3072))
API_KEY = os.getenv("EMBEDDING_BINDING_API_KEY")
MAX_TOKEN_SIZE = int(os.getenv("MAX_TOKEN_SIZE", 8192))


def configure_logging():
    """Configure logging with console and rotating file handlers."""
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_query.log"))
    print(f"\nLightRAG query log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )
    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "true").lower() == "true")

async def initialize_rag():
    """Initialize LightRAG with custom embedding function."""
    print("Initializing LightRAG for querying...")
    
    # Initialize embedding model
    embed_model = OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
        dimensions=EMBEDDING_DIM
    )
    
    # Define async embedding function
    async def async_embedding_func(texts):
        return embed_model.get_text_embedding_batch(texts)
    
    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=MAX_TOKEN_SIZE,               
        func=async_embedding_func
    )
    
    # Initialize LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=embedding_func,
        llm_model_func=gpt_4o_mini_complete
    )
    
    await rag.initialize_storages()
    await initialize_pipeline_status()
    await rag.aclear_cache()
    return rag

async def main():
    """Main function to query the LightRAG index."""
    rag = None
    try:
        if not os.getenv("OPENAI_API_KEY") and not API_KEY:
            raise ValueError("OPENAI_API_KEY or EMBEDDING_BINDING_API_KEY environment variable not set")
        rag = await initialize_rag()
        
        # Check if index exists
        if not os.path.exists(os.path.join(WORKING_DIR, "kv_store_full_docs.json")):
            raise FileNotFoundError(f"No index found in {WORKING_DIR}. Run the indexing script first.")
        
        # Perform query

        query = (
            "What does wire 130 do and what is each end of the wire connected to?"
        )
      
         
        for mode in ["naive", "local", "global", "hybrid", "mix"]:  # "naive", "local", "global", "hybrid", "mix"
            print("\n=====================")
            print(f"Query mode: {mode}")
            print("=====================")
            response = await rag.aquery(
                query,
                param=QueryParam(mode=mode, top_k=70),  # top_k=70, only_need_context=True, only_need_prompt=True
            )
            print(response)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            print("Finalizing storages...")
            await rag.finalize_storages()

if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
    print("\nQuerying Done!")


    '''
    print("--- All Loaded Environment Variables ---")
    # os.environ is a dictionary-like object
    # You can iterate over its items (key-value pairs)
    for key, value in os.environ.items():
        print(f"{key}={value}")

    print("--------------------------------------")
    '''



