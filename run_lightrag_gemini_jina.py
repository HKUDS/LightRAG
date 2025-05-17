# run_lightrag_gemini_jina.py
# This script demonstrates the initialization and usage of LightRAG
# with Jina embeddings and Gemini LLM.

import asyncio
import os
import re
import numpy as np
import google.generativeai as genai
import tkinter as tk
from tkinter import filedialog
import typing
try:
    import textract
except ImportError:
    textract = None

# LightRAG imports
from lightrag.lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.jina import jina_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# --- Jina Embedding Setup ---
# Note: Replace with your actual Jina API Key if JINA_API_KEY is not set as an environment variable
JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_c6322587415445b080c9b50b81ad349c5dd-MTCa0jIiRcbhHe3Nk71R4DQe")
JINA_EMBEDDING_DIM = 768
JINA_MAX_TOKEN_SIZE = 8192

async def jina_embedding_wrapper(texts: list[str]) -> np.ndarray:
    """Wrapper for Jina embedding function."""
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set. Please set it as an environment variable or directly in the script.")
    return await jina_embed(
        texts=texts,
        api_key=JINA_API_KEY,
        dimensions=JINA_EMBEDDING_DIM,
    )

jina_embedding_func = EmbeddingFunc(
    embedding_dim=JINA_EMBEDDING_DIM,
    max_token_size=JINA_MAX_TOKEN_SIZE,
    func=jina_embedding_wrapper
)
# --- End Jina Embedding Setup ---

# --- Gemini LLM Setup ---
# Note: Replace with your actual Gemini API Key if GEMINI_API_KEY is not set as an environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBi0jME5WSX9dGH-GWVYluNNnbSdJ6Pse4")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    # This branch will be taken if GEMINI_API_KEY is not found.
    # Consider raising an error or logging a warning.
    print("Warning: GEMINI_API_KEY not found. Gemini LLM functionality will be impaired.")


async def gemini_llm_complete_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list = [],
    model: str = "gemini-1.5-flash-latest", # Default model
    **kwargs
) -> str:
    """Wrapper for Gemini LLM completion function."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is not configured."
    
    # Determine the actual model to use, prioritizing kwargs over the default
    # The user's setup seems to target gemini-2.5-flash-preview-04-17
    # LightRAG might pass the model name via kwargs if configured in its own settings
    actual_model_name = kwargs.get("model", model)
    # If LightRAG passes a model name like 'gemini-2.5-flash-preview-04-17' via kwargs,
    # it will override the default "gemini-1.5-flash-latest" here.

    generative_model = genai.GenerativeModel(
        model_name=actual_model_name, # Use the determined model name
        system_instruction=system_prompt if system_prompt else None
    )
    contents_for_api = []
    if history_messages:
        for msg in history_messages:
            role = msg.get("role")
            parts_data = msg.get("parts", [])
            formatted_parts = []
            for part_item in parts_data:
                if isinstance(part_item, str):
                    formatted_parts.append({"text": part_item})
                elif isinstance(part_item, dict):
                    formatted_parts.append(part_item)
            if role and formatted_parts:
                contents_for_api.append({"role": role, "parts": formatted_parts})
    contents_for_api.append({"role": "user", "parts": [{"text": prompt}]})

    generation_config_params = {
        "temperature": kwargs.get("temperature", 0.1),
        "max_output_tokens": kwargs.get("max_output_tokens", 500),
    }
    if "top_p" in kwargs: generation_config_params["top_p"] = kwargs["top_p"]
    if "top_k" in kwargs: generation_config_params["top_k"] = kwargs["top_k"]

    output_json_requested = kwargs.get("output_json", False)
    if output_json_requested:
        generation_config_params["response_mime_type"] = "application/json"

    generation_config = genai.types.GenerationConfig(**generation_config_params)

    try:
        response = await generative_model.generate_content_async(
            contents=contents_for_api,
            generation_config=generation_config,
        )
        
        raw_output = response.text if response.text is not None else ""

        if output_json_requested and raw_output:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output, re.IGNORECASE)
            if match:
                raw_output = match.group(1).strip()
        
        return raw_output
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return f"Error communicating with Gemini API: {e}"
# --- End Gemini LLM Setup ---

def select_folder_tkinter() -> str | None:
    """Opens a dialog to select a folder and returns its path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    folder_path = filedialog.askdirectory(title="Select Folder to Index")
    root.destroy() # Destroy the root window after selection
    return folder_path if folder_path else None

# --- Main Script Logic ---
def extract_text_from_folder(folder_path: str) -> typing.Generator[tuple[str, str], None, None]:
    """
    Extracts text from all processable files in a given folder and its subfolders.
    Yields tuples of (file_path, text_content).
    """
    if textract is None:
        print("Error: The 'textract' library is not installed. Please install it (e.g., pip install textract).")
        return
    for root_dir, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root_dir, filename)
            try:
                byte_content = textract.process(file_path)
                text_content = byte_content.decode('utf-8', errors='replace')
                yield (file_path, text_content)
            except Exception as e:
                print(f"Could not process file {file_path}: {e}")

WORKING_DIR = "./lightrag_gemini_jina_storage"

async def initialize_rag(working_dir: str) -> LightRAG | None:
    """Initializes the LightRAG instance."""
    print(f"Initializing LightRAG in working directory: {working_dir}")
    if not GEMINI_API_KEY or not JINA_API_KEY:
        print("Error: API keys for Jina or Gemini are not set. Cannot initialize LightRAG.")
        return None
    try:
        rag = LightRAG(
            working_dir=working_dir,
            embedding_func=jina_embedding_func,
            llm_model_func=gemini_llm_complete_func
        )
        await rag.initialize_storages()
        await initialize_pipeline_status() # Ensure this is called after storages are initialized
        print("LightRAG initialized successfully.")
        return rag
    except Exception as e:
        print(f"Failed to initialize LightRAG: {e}")
        return None


async def main():
    """Main function to run the LightRAG demonstration."""
    setup_logger("lightrag", level="INFO") # Setup logger for LightRAG internal logs

    # Create working directory if it doesn't exist
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
        print(f"Created working directory: {WORKING_DIR}")

    rag_instance = None 
    try:
        rag_instance = await initialize_rag(WORKING_DIR)
        if not rag_instance:
            print("Exiting due to LightRAG initialization failure.")
            return

        selected_folder = select_folder_tkinter()
        if not selected_folder:
            print("No folder selected. Exiting.")
            if rag_instance: # Ensure finalize is called if rag was initialized
                await rag_instance.finalize_storages()
            return

        print(f"Selected folder for indexing: {selected_folder}")
        if textract is None:
            print("Cannot proceed with indexing as 'textract' is not available.")
            if rag_instance:
                await rag_instance.finalize_storages()
            return
        
        print("Starting to index files...")
        file_count = 0
        for file_path, text_content in extract_text_from_folder(selected_folder):
            if text_content and text_content.strip(): # Ensure content is not empty
                print(f"Indexing: {file_path}")
                try:
                    # Using file_path as ID and for citation tracking
                    await rag_instance.insert(text_content, ids=[file_path], file_paths=[file_path])
                    file_count += 1
                except Exception as e:
                    print(f"Error inserting document {file_path} into LightRAG: {e}")
            else:
                print(f"Skipping empty or unreadable content from: {file_path}")
        
        print(f"\nIndexing complete. {file_count} file(s) processed and attempted to insert.")

        # Comment out or remove the previous single document insertion and query
        # doc_text = "This is a test document for LightRAG using Gemini LLM and Jina embeddings. The setup seems to be working."
        # print(f"Inserting document: '{doc_text}'")
        # await rag_instance.insert(doc_text)
        # print("Document inserted successfully.")
        # query_text = "Describe the LLM and embedding models used in this LightRAG setup."
        # print(f"Querying with: '{query_text}'")
        # response = await rag_instance.query(query_text, param=QueryParam(mode="naive"))
        # print("Query processed successfully.")
        # print(f"\n--- Query ---")
        # print(query_text)
        # print(f"\n--- Response ---")
        # print(response)

    except Exception as e:
        print(f"An error occurred during main execution: {e}")
    finally:
        if rag_instance:
            print("Finalizing LightRAG storages...")
            await rag_instance.finalize_storages()
            print("LightRAG storages finalized.")

if __name__ == "__main__":
    asyncio.run(main())