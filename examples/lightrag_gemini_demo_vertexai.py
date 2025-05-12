# pip install -q -U google-genai sentencepiece

import os
from typing import Optional
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.google import google_complete, google_embed, google_embed_insert
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import GemmaTokenizer, logger, set_verbose_debug

from dotenv import load_dotenv

load_dotenv()
GOOGLE_GEMINI_MODEL = os.environ.get("LLM_MODEL")
if GOOGLE_GEMINI_MODEL is None:
    os.environ["LLM_MODEL"] = "gemini-2.0-flash-001"

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.environ.get(
    "GOOGLE_GENAI_USE_VERTEXAI", True
)
os.environ["GOOGLE_CLOUD_PROJECT"] = os.environ.get(
    "GOOGLE_CLOUD_PROJECT", "your-project-id"
)
os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get(
    "GOOGLE_CLOUD_LOCATION", "us-central1"
)
# or
# os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "your-api-key")

GOOGLE_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
if GOOGLE_EMBEDDING_MODEL is None:
    os.environ["EMBEDDING_MODEL"] = "text-embedding-004"

WORKING_DIR = "./dickens"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
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

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag(task_type: Optional[str] = None) -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_name=GOOGLE_GEMINI_MODEL,
        llm_model_func=google_complete,
        # llm_model_kwargs={"temperature": 0.0},
        # Use GemmaTokenizer for Google Gemini tokenization
        tokenizer=GemmaTokenizer(
            # tokenizer_dir="./tokenizer",  # Path to the tokenizer directory or automatic download from https://github.com/google/gemma_pytorch/blob/main/tokenizer/gemma3_cleaned_262144_v2.spiece.model
        ),
        embedding_func=google_embed_insert
        if task_type and task_type == "RETRIEVAL_DOCUMENT"
        else google_embed,  # google embeddings can be task specific
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # Initialize RAG instance for inserting data
        rag = await initialize_rag(task_type="RETRIEVAL_DOCUMENT")

        with open("./book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Initialize RAG instance for querying
        rag = await initialize_rag()

        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print("\n=====================")
        print("Query mode: local")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
