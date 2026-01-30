import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# 1. Load .env environment variables
load_dotenv()

# --- Define Helper Functions ---
def create_qa_prompt_from_template(prompt, system_prompt=None, history_messages=[]):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    return messages

# 2. Check necessary packages and imports
try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_embed
    from lightrag.kg.shared_storage import initialize_pipeline_status 
    from openai import AsyncAzureOpenAI
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# --- Load Environment Variables ---
sf_api_key = os.getenv("EMBEDDING_BINDING_API_KEY")
sf_base_url = os.getenv("EMBEDDING_BINDING_HOST")
sf_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
sf_dim = int(os.getenv("EMBEDDING_DIM", 1024))

azure_api_key = os.getenv("LLM_BINDING_API_KEY")
azure_endpoint = os.getenv("LLM_BINDING_HOST")
azure_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# --- [Configuration] Directory Paths ---
WORKING_DIR = "./data/rag_storage"
INPUT_DIR = "./data/inputs"
OUTPUT_DIR = "./data/output"
PROCESSED_LOG_FILE = os.path.join(WORKING_DIR, "processed_files.log")

# --- Define Model Connection Functions ---

# Create a single, reusable client instance for Azure OpenAI
azure_client = AsyncAzureOpenAI(
    api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    api_version=azure_version,
)

# [LLM] Azure OpenAI Connection Function
async def azure_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    # [Important Fix] Remove LightRAG internal parameters not supported by OpenAI
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None) 
    
    messages = create_qa_prompt_from_template(
        prompt, system_prompt=system_prompt, history_messages=history_messages
    )
    
    response = await azure_client.chat.completions.create(
        model=azure_deployment,
        messages=messages,
        temperature=0.0,  # <-- [FIX] Set temperature to 0 for deterministic output
        **kwargs,
    )
    if response.choices:
        return response.choices[0].message.content
    return ""

# [Embedding] SiliconFlow Connection Function
async def siliconflow_embed_func(texts):
    truncated_texts = [t[:4000] for t in texts] 
    return await openai_embed(
        truncated_texts, 
        model=sf_model, 
        api_key=sf_api_key, 
        base_url=sf_base_url
    )

# [Vision] For Image Interpretation (Multimodal)
async def azure_vision_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
    # [Important Fix] Remove LightRAG internal parameters not supported by OpenAI
    kwargs.pop("hashing_kv", None)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    content = [{"type": "text", "text": prompt}]
    
    if image_data:
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        })
        
    messages.append({"role": "user", "content": content})
    
    response = await azure_client.chat.completions.create(
        model=azure_deployment, 
        messages=messages, 
        temperature=0.0,  # <-- [FIX] Set temperature to 0 for deterministic output
        **kwargs,
    )
    if response.choices:
        return response.choices[0].message.content
    return ""

# --- Core Functional Functions ---

async def initialize_rag_system():
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Proactively check for and handle any empty .graphml files to prevent startup errors
    for filename in os.listdir(WORKING_DIR):
        if filename.endswith(".graphml"):
            file_path = os.path.join(WORKING_DIR, filename)
            if os.path.exists(file_path) and os.path.getsize(file_path) == 0:
                print(f"🗑️ Found empty graph file at '{file_path}'. Removing it for a clean start.")
                os.remove(file_path)

    # --- NEW: Prepare Entity Types and Addon Params ---
    addon_params = {
        "language": os.getenv("SUMMARY_LANGUAGE", "English")
    }

    # Load ENTITY_TYPES from .env (expects a JSON list string like ["TypeA", "TypeB"])
    entity_types_str = os.getenv("ENTITY_TYPES")
    if entity_types_str:
        try:
            entity_types = json.loads(entity_types_str)
            if isinstance(entity_types, list):
                addon_params["entity_types"] = entity_types
                print(f"📋 Loaded Custom ENTITY_TYPES: {len(entity_types)} types found.")
            else:
                print("⚠️ ENTITY_TYPES is not a list. Using default types.")
        except json.JSONDecodeError:
            print("⚠️ Failed to parse ENTITY_TYPES from .env (invalid JSON). Using default types.")
    
    print(f"🚀 Initializing LightRAG...")
    lightrag_instance = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=sf_dim,
            max_token_size=8192,
            func=siliconflow_embed_func
        ),
        addon_params=addon_params  # <--- Pass the custom params here
    )
    
    await lightrag_instance.initialize_storages()
    
    print("⚙️ Initializing Pipeline Status...")
    try:
        await initialize_pipeline_status()
    except Exception:
        pass

    print("🔗 Initializing RAGAnything...")
    rag = RAGAnything(
        lightrag=lightrag_instance,
        vision_model_func=azure_vision_func,
        llm_model_func=azure_llm_func, # Optional: Uncomment if needed for discarded items
        config=RAGAnythingConfig(
            working_dir=WORKING_DIR, 
            enable_image_processing=True,
        )
    )
    return rag

async def process_new_document(rag: RAGAnything, file_name: str):
    input_file = os.path.join(INPUT_DIR, file_name)
    
    # Double check if file actually exists
    if not os.path.exists(input_file):
        return

    # Also check if the file is empty
    if os.path.getsize(input_file) == 0:
        print(f"\n⏩ File '{file_name}' is empty. Skipping.")
        return
    # Check process log
    processed_files = set()
    if os.path.exists(PROCESSED_LOG_FILE):
        with open(PROCESSED_LOG_FILE, "r") as f:
            processed_files = set(line.strip() for line in f)

    if file_name in processed_files:
        print(f"\n⏩ File '{file_name}' already processed. Skipping.")
        return 

    try:
        print(f"\n📄 Processing file: {file_name}")
        print("⏳ Parsing and building graph...")
        
        await rag.process_document_complete(
            file_path=input_file,
            output_dir=OUTPUT_DIR
        )
        
        print(f"✅ File '{file_name}' Done!")
        
        with open(PROCESSED_LOG_FILE, "a") as f:
            f.write(f"{file_name}\n")
            
    except Exception as e:
        print(f"⚠️ Error processing '{file_name}': {e}")
        # We continue to the next file even if one fails

async def main():
    rag = await initialize_rag_system()
    print("\n🎉 RAG system ready. Scanning input directory...")

    # --- NEW LOGIC: Iterate through all files in INPUT_DIR ---
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory '{INPUT_DIR}' does not exist!")
        return

    # Get list of all files
    all_files = os.listdir(INPUT_DIR)
    
    # Filter out hidden files (like .DS_Store) and ensure they are files
    valid_files = [f for f in all_files if not f.startswith('.') and os.path.isfile(os.path.join(INPUT_DIR, f))]
    
    print(f"📂 Found {len(valid_files)} files to check.")

    for file_name in valid_files:
        await process_new_document(rag, file_name)

    print("\n🏁 All files in the folder have been checked/processed.")

if __name__ == "__main__":
    asyncio.run(main())