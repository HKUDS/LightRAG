import os
import asyncio
import inspect
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

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
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")
    )

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
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


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
        **kwargs,
    )


async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
            max_token_size=int(
                os.getenv("EMBEDDING_TOKEN_LIMIT", os.getenv("MAX_EMBED_TOKENS", "8192"))
            ),
            func=lambda texts: openai_embed.func(
                texts,
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
                base_url=os.getenv("EMBEDDING_BINDING_HOST"),
                api_key=os.getenv("EMBEDDING_BINDING_API_KEY")
                or os.getenv("OPENAI_API_KEY"),
                embedding_dim=(
                    int(os.getenv("EMBEDDING_DIM"))
                    if os.getenv("EMBEDDING_SEND_DIM", "false").lower() == "true"
                    else None
                ),
            ),
        ),
        chunk_token_size=120,
        chunk_overlap_token_size=30
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag


async def main():
    try:
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        # with open("./book.txt", "r", encoding="utf-8") as f:
        #     await rag.ainsert(f.read())

        text = """
"Stuart Rosenberg (August 11, 1927 – March 15, 2007) was an American film and television director whose motion pictures include \"Cool Hand Luke\" (1967), \"Voyage of the Damned\" (1976), \"The Amityville Horror\" (1979), and \"The Pope of Greenwich Village\" (1984).",
"He was noted for his work with actor Paul Newman."
"Méditerranée is a 1963 French experimental film directed by Jean-Daniel Pollet with assistance from Volker Schlöndorff.",
"It was written by Philippe Sollers and produced by Barbet Schroeder, with music by Antione Duhamel.",
"The 45 minute film is cited as one of Pollet's most influential films, which according to Jonathan Rosenbaum directly influenced Jean-Luc Goddard's \"Contempt\", released later the same year.",
"Footage for the film was shot around the Mediterranean, including at a Greek temple, a Sicilian garden, the sea, and also features a fisherman, a bullfighter, and a girl on an operating table."
"Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Geneviève Waïte, and directed by Stuart Rosenberg.",
"The screenplay was written by Joel Lieber and Stanley Hart, adapted from a novel by Lieber."
"Ian Barry is an Australian director of film and TV."
"Peter Levin is an American director of film, television and theatre."
"Brian Johnson( born 1939 or 1940) is a British designer and director of film and television special effects."
"Rachel Feldman( born August 22, 1954) is an American director of film and television and screenwriter of television films."
"Hanro Smitsman, born in 1967 in Breda( Netherlands), is a writer and director of film and television."
"Jean-Daniel Pollet (1936–2004) was a French film director and screenwriter who was most active in the 1960s and 1970s.",
"He was associated with two approaches to filmmaking: comedies which blended burlesque and melancholic elements, and poetic films based on texts by writers such as the French poet Francis Ponge."
"Howard Winchel Koch( April 11, 1916 – February 16, 2001) was an American producer and director of film and television."
}"""
        await rag.ainsert(text)
        query = """Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?
"""
        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        resp = await rag.aquery(
            query,
            param=QueryParam(mode="naive", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform local search
        print("\n=====================")
        print("Query mode: local")
        print("=====================")
        resp = await rag.aquery(
            query,
            param=QueryParam(mode="local", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        resp = await rag.aquery(
            query,
            param=QueryParam(mode="global", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        #Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        resp = await rag.aquery(
            query,
            param=QueryParam(mode="hybrid", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)


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