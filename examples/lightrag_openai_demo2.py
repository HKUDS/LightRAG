# run >python examples/lightrag_openai_demo2.py 
# DB: MongoDB for document storage + LightRAG processing

import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug
import pymongo
from pymongo import MongoClient
from datetime import datetime

WORKING_DIR = "./dickens2"
# Use MongoDB configuration from .env file
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "Story"  # Your Atlas database name
COLLECTION_NAME = "book1"  # Your Atlas collection name
WORKSPACE = os.getenv("MONGODB_WORKSPACE", "demo")


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


def upload_document_to_db(file_path, document_name=None):
    """Upload a document to MongoDB"""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Read the document
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create document record
        doc_name = document_name or os.path.basename(file_path)
        document = {
            "name": doc_name,
            "content": content,
            "file_path": file_path,
            "uploaded_at": datetime.now(),
            "size": len(content)
        }
        
        # Insert or update document
        result = collection.replace_one(
            {"name": doc_name}, 
            document, 
            upsert=True
        )
        
        if result.upserted_id:
            print(f"✅ Document '{doc_name}' uploaded to MongoDB (ID: {result.upserted_id})")
        else:
            print(f"✅ Document '{doc_name}' updated in MongoDB")
            
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ Error uploading document to MongoDB: {e}")
        return False


def retrieve_document_from_db(document_name):
    """Retrieve a document from MongoDB"""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Find the document
        document = collection.find_one({"name": document_name})
        
        if document:
            print(f"✅ Retrieved document '{document_name}' from MongoDB")
            print(f"   Size: {document['size']} characters")
            print(f"   Uploaded: {document['uploaded_at']}")
            client.close()
            return document['content']
        else:
            print(f"❌ Document '{document_name}' not found in MongoDB")
            client.close()
            return None
            
    except Exception as e:
        print(f"❌ Error retrieving document from MongoDB: {e}")
        return None


def list_documents_in_db():
    """List all documents in MongoDB"""
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        documents = list(collection.find({}, {"name": 1, "size": 1, "uploaded_at": 1}))
        
        if documents:
            print(f"\n📚 Documents in MongoDB ({len(documents)} found):")
            for doc in documents:
                print(f"   - {doc['name']} ({doc['size']} chars, uploaded: {doc['uploaded_at']})")
        else:
            print("📚 No documents found in MongoDB")
            
        client.close()
        return documents
        
    except Exception as e:
        print(f"❌ Error listing documents: {e}")
        return []


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    # Check if OPENAI_API_KEY environment variable exists--> in .env is LLM_BINDING_API_KEY
    if not os.getenv("LLM_BINDING_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
        )
        print("You can set the environment variable by running:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        return  # Exit the async function

    try:
        print("\n🚀 Starting LightRAG Demo with MongoDB Document Storage")
        print("=" * 60)
        print(f"📊 MongoDB URI: {MONGO_URI}")
        print(f"🗄️  Database: {DB_NAME}")
        print(f"📁 Collection: {COLLECTION_NAME}")
        print(f"🏷️  Workspace: {WORKSPACE}")
        
        # Step 1: Upload document to MongoDB
        print("\n📤 STEP 1: Uploading document to MongoDB...")
        document_name = "tale_of_two_cities.txt"
        
        if os.path.exists("./book.txt"):
            upload_success = upload_document_to_db("./book.txt", document_name)
            if not upload_success:
                print("❌ Failed to upload document. Exiting...")
                return
        else:
            print("❌ book.txt file not found. Please ensure it exists.")
            return
        
        # Step 2: List documents in MongoDB
        print("\n📋 STEP 2: Listing documents in MongoDB...")
        list_documents_in_db()
        
        # Step 3: Retrieve document from MongoDB
        print(f"\n📥 STEP 3: Retrieving document '{document_name}' from MongoDB...")
        document_content = retrieve_document_from_db(document_name)
        
        if not document_content:
            print("❌ Failed to retrieve document from MongoDB. Exiting...")
            return
        
        print(f"✅ Successfully retrieved document ({len(document_content)} characters)")
        
        # Step 4: Clear old LightRAG data files
        print("\n🧹 STEP 4: Clearing old LightRAG data files...")
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
                print(f"🗑️  Deleted: {file_path}")

        # Step 5: Initialize LightRAG
        print("\n⚡ STEP 5: Initializing LightRAG...")
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

        # Step 6: Process document with LightRAG (using content from MongoDB)
        print("🔄 STEP 6: Processing document with LightRAG...")
        await rag.ainsert(document_content)
        print("✅ Document processed and indexed by LightRAG")

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
