import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status
import json
from typing import Optional
from lightrag.utils import logger, set_verbose_debug

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

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


async def initialize_rag(addon_params: Optional[dict] = None):
    rag_kwargs = {
        "working_dir": WORKING_DIR,
        "llm_model_func": gpt_4o_mini_complete,
        "addon_params": addon_params,
    }
    rag = LightRAG(**rag_kwargs)

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


# create file based example, based on following proposed directory structure:
# my_docs/
#  └── books/
#      ├── book1.txt
#      ├── book2.txt
#  └── articles/
#      ├── article1.txt
#      ├── article2.txt
#      ├── insert_prompt_template.json
# my_queries/
#  └── articles/
#      └── query_prompt_template.json
#
# prompt templates must follow default .utils.prompt.py template_key nomenclature and include same placeholders:
# arg             template_key                       type        expected_placeholder_keys in {}
# --------------------------------------------------------------------------------------------------
# global_config   "language"                         str         -
# global_config   "tuple_delimiter"                  str         -
# global_config   "record_delimiter"                 str         -
# global_config   "completion_delimiter"             str         -
# global_config   "similarity_check"                 str         original_prompt,cached_prompt
# --
# global_config   "summarize_entity_descriptions"    str         language,entity_name,description_list
# global_config   "entity_extraction_examples"       str         tuple_delimiter,record_delimiter,completion_delimiter
# global_config   "entity_types"                     list[str]   -
# global_config   "entity_extraction"                str         language,entity_types,tuple_delimiter,record_delimiter,completion_delimiter,examples,input_text
# global_config   "entity_continue_extraction"       str         entity_types,tuple_delimiter,language,record_delimiter,completion_delimiter
# global_config   "entity_if_loop_extraction"        str         -
# global_config   "keywords_extraction"              str         examples,history,query,language
# global_config   "keywords_extraction_examples"     str         -
# --
# query_param     "rag_response"                     str         history,content_data,response_type
# query_param     "naive_rag_response"               str         history,content_data,response_type
# query_param     "mix_rag_response"                 str         history,kg_context,vector_context,response_type
# query_param     "fail_rag_response"                str         -


json.dump(
    {"entity_extraction_examples": ["device", "make", "model", "publication", "date"]},
    open("./my_docs/articles/insert_template_prompts.json", "w"),
)
json.dump(
    {"rag_response": "System prompt specific to articles..."},
    open("./my_queries/articles/query_template_prompts.json", "w"),
)

docs = {
    "books": {
        "file_paths": ["./books/book1.txt", "./books/book2.txt"],
        "addon_params": {
            "entity_extraction_examples": ["organization", "person", "location"],
        },
        "system_prompts": {
            "rag_response": "KG mode system prompt specific to books...",
            "naive_rag_response": "Naive mode system prompt specific to books...",
            "mix_rag_response": "Mix mode system prompt specific to books...",
        },
    },
    "articles": {
        "file_paths": ["./articles/article1.txt", "./articles/article2.txt"],
        "addon_params": json.load(
            open("./my_docs/articles/insert_template_prompts.json", "r")
        ),
        "system_prompts": json.load(
            open("./my_queries/articles/query_template_prompts.json", "r")
        ),
    },
}


def get_content(file_paths):
    contents = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            contents.append(f.read())
    return contents


async def main():
    rag = None
    for doc_type, doc_info in docs.items():
        # Insert differently per doc type
        file_paths = doc_info["file_paths"]
        addon_params = doc_info["addon_params"]

        # Initialize the RAG instance for each document type
        print("\n=====================")
        print(f"Initializing RAG for {doc_type}")
        print(f"Inserting with custom {addon_params}")
        print("=====================")
        try:
            rag = await initialize_rag(addon_params)

            contents = get_content(file_paths)
            await rag.ainsert(contents, file_paths=file_paths)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if rag:
                await rag.finalize_storages()

    rag = None
    addon_params = None
    try:
        rag = await initialize_rag(addon_params)
        # Perform naive search
        # for specific to `books` type queries
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="naive"),
                system_prompt=docs["books"]["system_prompts"][
                    "naive_rag_response"
                ],  # Use the naive mode specific system prompt for book concepts
            )
        )
        # Perform hybrid search
        # for specific to `articles` type queries
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
                system_prompt=docs["articles"]["system_prompts"][
                    "rag_response"
                ],  # Use the hybrid mode specific system prompt for article concepts
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
