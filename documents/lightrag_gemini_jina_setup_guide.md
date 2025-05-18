# LightRAG with Gemini LLM and Jina Embeddings: Setup Guide

## 1. Overview

This document provides a comprehensive guide to setting up and running LightRAG, utilizing Google's Gemini model as the Language Learning Model (LLM) and Jina AI for generating embeddings. This setup allows for powerful and flexible Retrieval Augmented Generation (RAG) capabilities, now including the ability to index documents from a user-selected folder and run as a server with these custom components.

## 2. Prerequisites

Before you begin, ensure you have the following:

*   **Python 3.10**: This setup was specifically configured with Python 3.10.
*   **Jina API Key**: An API key from Jina AI for accessing their embedding models.
*   **Gemini API Key**: An API key from Google AI Studio or Google Cloud for accessing Gemini models.
*   **`textract` library**: For folder indexing, the `textract` library is required. (See Section 4.5 and 7).

## 3. Installation

LightRAG core needs to be installed. If you are installing the official package, you can typically use pip:

```bash
pip install lightrag-hku
```
Alternatively, if you are working with a local development version, you might have used:
```bash
pip install -e .
```
Please use the command that corresponds to how LightRAG was installed in your environment. The interaction leading to this guide involved installing the LightRAG core.

You will also need to install `textract` if you intend to use the folder indexing feature:
```bash
pip install textract
```

## 4. Configuration

### Jina Embeddings Setup

To use Jina embeddings, you need to configure LightRAG with your Jina API key and define an embedding function.

*   **Jina API Key Used**: `jina_c6322587415445b080c9b50b81ad349c5dd-MTCa0jIiRcbhHe3Nk71R4DQe`
    *   It's recommended to set this as an environment variable `JINA_API_KEY`.

*   **Python Code Snippet for Jina `EmbeddingFunc`**:

    ```python
    # Jina Embedding Setup
    import numpy as np
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.jina import jina_embed
    import os # Added for os.getenv

    JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_c6322587415445b080c9b50b81ad349c5dd-MTCa0jIiRcbhHe3Nk71R4DQe")
    JINA_EMBEDDING_DIM = 768
    JINA_MAX_TOKEN_SIZE = 8192

    async def jina_embedding_wrapper(texts: list[str]) -> np.ndarray:
        if not JINA_API_KEY:
            raise ValueError("JINA_API_KEY is not set.")
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
    ```

*   **Key Parameters**:
    *   `JINA_EMBEDDING_DIM`: Set to `768`. This is the dimensionality of the Jina embeddings (e.g., for `jina-embeddings-v2-base-en`).
    *   `JINA_MAX_TOKEN_SIZE`: Set to `8192`. This is the maximum number of tokens Jina's embedding model can process in a single request.

### Gemini LLM Setup

To use Gemini as the LLM, configure it with your Gemini API key and define a completion function.

*   **Gemini API Key Used**: `AIzaSyBi0jME5WSX9dGH-GWVYluNNnbSdJ6Pse4`
    *   It's recommended to set this as an environment variable `GEMINI_API_KEY`.

*   **Python Code Snippet for `gemini_llm_complete_func`**:

    ```python
    # Gemini LLM Setup
    import google.generativeai as genai
    import os # Added for os.getenv

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBi0jME5WSX9dGH-GWVYluNNnbSdJ6Pse4")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        print("Warning: GEMINI_API_KEY not found.")

    async def gemini_llm_complete_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list = [],
        model: str = "gemini-2.5-flash-preview-04-17",
        **kwargs
    ) -> str:
        if not GEMINI_API_KEY:
            return "Error: GEMINI_API_KEY is not configured."
        generative_model = genai.GenerativeModel(
            model_name=model,
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
        generation_config = genai.types.GenerationConfig(**generation_config_params)
        try:
            response = await generative_model.generate_content_async(
                contents=contents_for_api,
                generation_config=generation_config,
            )
            return response.text if response.text is not None else ""
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            return f"Error communicating with Gemini API: {e}"
    ```

*   **Key Parameters**:
    *   `model`: Set to `"gemini-2.5-flash-preview-04-17"`. This specifies the Gemini model to be used. You can change this to other compatible Gemini models.
    *   `temperature`, `max_output_tokens`, `top_p`, `top_k`: These are standard LLM generation parameters that control the creativity, length, and sampling strategy of the response.

## 4.5. Folder Indexing and Content Extraction

The script has been enhanced to support indexing documents from a user-selected folder. This involves a graphical folder selection dialog and a robust mechanism for extracting text from various file types. The functions `select_folder_tkinter()` and `extract_text_from_folder()` responsible for these features are included in the main script presented in Section 5.

### Tkinter for Folder Selection

To allow users to easily specify a source folder for documents, the script utilizes the `tkinter` library via the `select_folder_tkinter()` function.
*   `tkinter` is part of Python's standard library and displays a native folder selection dialog.
*   It does not require separate installation as it is included with most Python distributions.

### File Content Extraction with `textract`

For processing files within the selected folder, the script uses the `textract` library via the `extract_text_from_folder()` function.
*   `textract` is a powerful tool that can extract text from a wide variety of file formats (e.g., .pdf, .docx, .txt, .odt, .xlsx, .pptx, and more).
*   **Important**:
    *   The `textract` library needs to be installed in your Python environment. You can install it using pip:
        ```bash
        pip install textract
        ```
    *   The script checks if `textract` is available and will print an error message and halt indexing if it's not found.

## 5. Initialization Script (`run_lightrag_gemini_jina.py`)

The following Python script, [`run_lightrag_gemini_jina.py`](run_lightrag_gemini_jina.py:1), demonstrates how to initialize and use LightRAG with the Jina embeddings, Gemini LLM configurations, and the new folder indexing capability.

*   **Full Script Content**:

    ```python
    # run_lightrag_gemini_jina.py
    import asyncio
    import os
    import numpy as np
    import google.generativeai as genai
    import tkinter as tk
    from tkinter import filedialog
    import typing
    try:
        import textract
    except ImportError:
        textract = None

    from lightrag.lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.jina import jina_embed
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.utils import setup_logger

    JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_c6322587415445b080c9b50b81ad349c5dd-MTCa0jIiRcbhHe3Nk71R4DQe")
    JINA_EMBEDDING_DIM = 768
    JINA_MAX_TOKEN_SIZE = 8192
    async def jina_embedding_wrapper(texts: list[str]) -> np.ndarray:
        if not JINA_API_KEY: raise ValueError("JINA_API_KEY is not set.")
        return await jina_embed(texts=texts, api_key=JINA_API_KEY, dimensions=JINA_EMBEDDING_DIM)
    jina_embedding_func = EmbeddingFunc(embedding_dim=JINA_EMBEDDING_DIM, max_token_size=JINA_MAX_TOKEN_SIZE, func=jina_embedding_wrapper)

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBi0jME5WSX9dGH-GWVYluNNnbSdJ6Pse4")
    if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
    else: print("Warning: GEMINI_API_KEY not found.")
    async def gemini_llm_complete_func(prompt: str, system_prompt: str | None = None, history_messages: list = [], model: str = "gemini-2.5-flash-preview-04-17", **kwargs) -> str:
        if not GEMINI_API_KEY: return "Error: GEMINI_API_KEY is not configured."
        generative_model = genai.GenerativeModel(model_name=model, system_instruction=system_prompt if system_prompt else None)
        contents_for_api = []
        if history_messages:
            for msg in history_messages:
                role, parts_data = msg.get("role"), msg.get("parts", [])
                formatted_parts = [{"text": p} if isinstance(p, str) else p for p in parts_data if isinstance(p, (str, dict))]
                if role and formatted_parts: contents_for_api.append({"role": role, "parts": formatted_parts})
        contents_for_api.append({"role": "user", "parts": [{"text": prompt}]})
        cfg_params = {"temperature": kwargs.get("temperature", 0.1), "max_output_tokens": kwargs.get("max_output_tokens", 500)}
        if "top_p" in kwargs: cfg_params["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs: cfg_params["top_k"] = kwargs["top_k"]
        gen_cfg = genai.types.GenerationConfig(**cfg_params)
        try:
            response = await generative_model.generate_content_async(contents=contents_for_api, generation_config=gen_cfg)
            return response.text if response.text is not None else ""
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            return f"Error communicating with Gemini API: {e}"

    def select_folder_tkinter() -> str | None:
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title="Select Folder to Index")
        root.destroy()
        return folder_path if folder_path else None

    def extract_text_from_folder(folder_path: str) -> typing.Generator[tuple[str, str], None, None]:
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
        print(f"Initializing LightRAG in: {working_dir}")
        if not GEMINI_API_KEY or not JINA_API_KEY: print("Error: API keys missing."); return None
        try:
            rag = LightRAG(working_dir=working_dir, embedding_func=jina_embedding_func, llm_model_func=gemini_llm_complete_func)
            await rag.initialize_storages()
            await initialize_pipeline_status()
            print("LightRAG initialized.")
            return rag
        except Exception as e: print(f"Failed to initialize LightRAG: {e}"); return None

    async def main():
        setup_logger("lightrag", level="INFO")
        if not os.path.exists(WORKING_DIR): os.makedirs(WORKING_DIR); print(f"Created: {WORKING_DIR}")
        rag_instance = None
        try:
            rag_instance = await initialize_rag(WORKING_DIR)
            if not rag_instance: print("Exiting: LightRAG init failed."); return

            selected_folder = select_folder_tkinter()
            if not selected_folder:
                print("No folder selected. Exiting.")
                if rag_instance: await rag_instance.finalize_storages()
                return
            print(f"Selected folder for indexing: {selected_folder}")
            if textract is None:
                print("Cannot proceed: 'textract' not available.")
                if rag_instance: await rag_instance.finalize_storages()
                return

            print("Starting to index files...")
            file_count = 0
            for file_path, text_content in extract_text_from_folder(selected_folder):
                if text_content and text_content.strip():
                    print(f"Indexing: {file_path}")
                    try:
                        await rag_instance.insert(text_content, ids=[file_path], file_paths=[file_path])
                        file_count += 1
                    except Exception as e: print(f"Error inserting {file_path}: {e}")
                else: print(f"Skipping empty/unreadable: {file_path}")
            print(f"\nIndexing complete. {file_count} file(s) processed.")
        except Exception as e: print(f"Error in main: {e}")
        finally:
            if rag_instance: print("Finalizing storages..."); await rag_instance.finalize_storages(); print("Storages finalized.")
    if __name__ == "__main__": asyncio.run(main())
    ```

*   **Explanation of Main Components**:
    *   `select_folder_tkinter()` and `extract_text_from_folder()`: These functions, detailed in Section 4.5, handle the folder selection GUI and file content extraction respectively.
    *   `WORKING_DIR`: Specifies the directory (`./lightrag_gemini_jina_storage`) where LightRAG will store its data. This directory will be created if it doesn't exist.
    *   `initialize_rag` function:
        *   Takes the `working_dir` as input.
        *   Checks for API keys.
        *   Instantiates `LightRAG` with the configured embedding and LLM functions.
        *   Initializes LightRAG storages.
        *   Returns the `LightRAG` instance.
    *   `main` function (Updated Flow):
        *   Sets up logging and ensures `WORKING_DIR` exists.
        *   Initializes the `LightRAG` instance.
        *   If RAG initialization is successful:
            *   Prompts the user to select a folder using `select_folder_tkinter()`.
            *   If no folder is selected, or if `textract` is not available, the script exits gracefully after finalizing storages.
            *   Otherwise, it iterates through files in the selected folder using `extract_text_from_folder()`.
            *   For each valid file, text content is extracted and inserted into LightRAG.
            *   Prints progress and a summary of indexed files.
        *   A `finally` block ensures that LightRAG storages are always finalized.

## 6. Configuring the LightRAG Server for Custom Gemini/Jina

To use the custom Jina embedding and Gemini LLM functions defined in Section 4 with the LightRAG server, you need to modify the server's initialization code located in [`lightrag/api/lightrag_server.py`](lightrag/api/lightrag_server.py:1).

### Step 1: Add Custom Functions and Imports

Near the top of the [`lightrag/api/lightrag_server.py`](lightrag/api/lightrag_server.py:1) file, add the Python code block that defines the `custom_jina_embedding_func` and `custom_gemini_llm_complete_func`, along with their necessary imports. Ensure this block is placed before the `create_app` function definition.

```python
# --- Custom Gemini/Jina Integration START ---
import os
import numpy as np
import google.generativeai as genai
from lightrag.utils import EmbeddingFunc
from lightrag.llm.jina import jina_embed

JINA_API_KEY = os.getenv("JINA_API_KEY", "jina_c6322587415445b080c9b50b81ad349c5dd-MTCa0jIiRcbhHe3Nk71R4DQe")
JINA_EMBEDDING_DIM = 768
JINA_MAX_TOKEN_SIZE = 8192
async def jina_embedding_wrapper(texts: list[str]) -> np.ndarray:
    if not JINA_API_KEY: raise ValueError("JINA_API_KEY is not set.")
    return await jina_embed(texts=texts, api_key=JINA_API_KEY, dimensions=JINA_EMBEDDING_DIM)
custom_jina_embedding_func = EmbeddingFunc(embedding_dim=JINA_EMBEDDING_DIM, max_token_size=JINA_MAX_TOKEN_SIZE, func=jina_embedding_wrapper)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBi0jME5WSX9dGH-GWVYluNNnbSdJ6Pse4")
if GEMINI_API_KEY: genai.configure(api_key=GEMINI_API_KEY)
else: print("Warning: GEMINI_API_KEY not found.")
async def custom_gemini_llm_complete_func(prompt: str, system_prompt: str | None = None, history_messages: list = [], model: str = "gemini-2.5-flash-preview-04-17", **kwargs) -> str:
    if not GEMINI_API_KEY: return "Error: GEMINI_API_KEY is not configured."
    # Note: The full implementation of this function (handling history, config, API call)
    # should be included here, as defined in Section 4.
    generative_model = genai.GenerativeModel(model_name=model, system_instruction=system_prompt if system_prompt else None)
    contents_for_api = []
    if history_messages:
        for msg in history_messages:
            role, parts_data = msg.get("role"), msg.get("parts", [])
            formatted_parts = [{"text": p} if isinstance(p, str) else p for p in parts_data if isinstance(p, (str, dict))]
            if role and formatted_parts: contents_for_api.append({"role": role, "parts": formatted_parts})
    contents_for_api.append({"role": "user", "parts": [{"text": prompt}]})
    cfg_params = {"temperature": kwargs.get("temperature", 0.1), "max_output_tokens": kwargs.get("max_output_tokens", 500)}
    if "top_p" in kwargs: cfg_params["top_p"] = kwargs["top_p"]
    if "top_k" in kwargs: cfg_params["top_k"] = kwargs["top_k"]
    gen_cfg = genai.types.GenerationConfig(**cfg_params)
    try:
        response = await generative_model.generate_content_async(contents=contents_for_api, generation_config=gen_cfg)
        return response.text if response.text is not None else ""
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return f"Error communicating with Gemini API: {e}"
# --- Custom Gemini/Jina Integration END ---
```

### Step 2: Modify `LightRAG` Initialization

Locate the `create_app` function within [`lightrag/api/lightrag_server.py`](lightrag/api/lightrag_server.py:1). Inside this function, find the line where the `LightRAG` class is instantiated. Modify this call to pass your custom functions to the `llm_model_func` and `embedding_func` parameters.

```python
# Inside the create_app function in lightrag/api/lightrag_server.py
rag_instance = LightRAG(
    llm_model_func=custom_gemini_llm_complete_func, # Use custom Gemini function
    embedding_func=custom_jina_embedding_func,     # Use custom Jina function
    working_dir=args.working_dir,
    kv_storage=args.kv_storage,
    vector_storage=args.vector_storage,
    graph_storage=args.graph_storage,
    doc_status_storage=args.doc_status_storage,
    chunk_token_size=int(args.chunk_size),
    chunk_overlap_token_size=int(args.chunk_overlap_size),
    vector_db_storage_cls_kwargs={"cosine_better_than_threshold": args.cosine_threshold},
    enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_extract,
    enable_llm_cache=args.enable_llm_cache,
    auto_manage_storages_states=False,
    max_parallel_insert=args.max_parallel_insert,
    addon_params={"language": args.summary_language},
)
```

By making these two modifications, the LightRAG server will use your specified Gemini LLM and Jina embedding functions for its operations.

## 7. Running the Setup Script and Server

### Running the Initialization Script

To run the initialization script ([`run_lightrag_gemini_jina.py`](run_lightrag_gemini_jina.py:1)) to index documents from a folder:

1.  **Ensure API Keys are Set**:
    *   The script attempts to read `JINA_API_KEY` and `GEMINI_API_KEY` from environment variables.
    *   Set these environment variables in your terminal before running the script:
        ```bash
        # Linux/macOS
        export JINA_API_KEY="your_jina_api_key"
        export GEMINI_API_KEY="your_gemini_api_key"

        # Windows Command Prompt
        set JINA_API_KEY="your_jina_api_key"
        set GEMINI_API_KEY="your_gemini_api_key"

        # Windows PowerShell
        $env:JINA_API_KEY="your_jina_api_key"
        $env:GEMINI_API_KEY="your_gemini_api_key"
        ```
    *   Alternatively, if the environment variables are not found, the script uses the hardcoded fallback API keys. **For security and best practice, using environment variables is strongly recommended.**

2.  **Install `textract` (if not already installed)**:
    *   The script relies on the `textract` library for processing files from a folder. Ensure it's installed:
        ```bash
        pip install textract
        ```
    *   The script will notify you if `textract` is missing and will not proceed with folder indexing.

3.  **Execute the Script**:
    Navigate to the directory containing [`run_lightrag_gemini_jina.py`](run_lightrag_gemini_jina.py:1) and run:
    ```bash
    python run_lightrag_gemini_jina.py
    ```

Upon execution:
*   A Tkinter dialog window will appear, prompting you to **select a folder**.
*   After selecting a folder, the script will attempt to initialize LightRAG.
*   If successful, it will then try to extract text from processable files within the selected folder (recursively) and index them into LightRAG.
*   Progress messages will be printed to the console.

This will initialize LightRAG and populate it with documents from the chosen folder, ready for querying.

### Running the Modified Server

After modifying [`lightrag/api/lightrag_server.py`](lightrag/api/lightrag_server.py:1) as described in Section 6:

1.  **Ensure API Keys are Set**:
    *   Just like the script, the server code relies on the `JINA_API_KEY` and `GEMINI_API_KEY` environment variables being set. Use the same methods as described above to set them in your terminal session before starting the server.

2.  **Start the Server**:
    *   You can start the server using the `lightrag-server` command if it's installed in your PATH and points to the modified code:
        ```bash
        lightrag-server --working-dir ./lightrag_gemini_jina_storage [other options...]
        ```
    *   Alternatively, you can run the module directly using Python:
        ```bash
        python -m lightrag.api.lightrag_server --working-dir ./lightrag_gemini_jina_storage [other options...]
        ```
    *   Replace `[other options...]` with any additional command-line arguments you need (e.g., `--port`, `--host`). Ensure the `--working-dir` points to the same directory used by the initialization script if you want the server to access the indexed data.

The server will now run using the custom Gemini LLM and Jina embedding functions.

## 8. Interaction Summary (Workflow Orchestration)

This documentation was created as part of a larger workflow orchestrated to fulfill a user request. The key steps were:

1.  **Initial Request**: User asked to read the `README.md` of the LightRAG project.
2.  **User Feedback & Goal Setting**: User requested to install and initialize LightRAG using Gemini for the LLM and Jina for embeddings, providing the necessary API keys.
3.  **Clarification**: The choice of Jina for embeddings was confirmed.
4.  **Subtask 1: Check Jina Embeddings**: Checked for existing Jina embedding integrations within LightRAG. Found [`lightrag/llm/jina.py`](lightrag/llm/jina.py:1).
5.  **Subtask 2: Install LightRAG Core**: LightRAG core components were installed.
6.  **Subtask 3: Prepare Jina Embedding Function**: A Python function (`jina_embedding_wrapper`) and `EmbeddingFunc` instance were prepared for Jina.
7.  **Subtask 4: Prepare Gemini LLM Function**: A Python function (`gemini_llm_complete_func`) was prepared for Gemini LLM completions.
8.  **Subtask 5: Create Initialization Script**: The [`run_lightrag_gemini_jina.py`](run_lightrag_gemini_jina.py:1) script was created to demonstrate the setup.
9.  **Subtask 6: Documentation**: The initial version of this setup guide ([`documents/lightrag_gemini_jina_setup_guide.md`](documents/lightrag_gemini_jina_setup_guide.md:1)) was created.
10. **Subtask 12: Tkinter Folder Selector**: Implemented `select_folder_tkinter()` function for GUI-based folder selection.
11. **Subtask 13: File Traversal and Text Extraction**: Implemented `extract_text_from_folder()` using `textract` to process various file types from a selected folder.
12. **Subtask 14: Script Integration**: Integrated folder selection and file indexing into the main execution flow of [`run_lightrag_gemini_jina.py`](run_lightrag_gemini_jina.py:1).
13. **Subtask 15: Documentation Update**: This setup guide ([`documents/lightrag_gemini_jina_setup_guide.md`](documents/lightrag_gemini_jina_setup_guide.md:1)) was updated to reflect the new folder indexing functionality.
14. **Subtask 17: Understand Server Config**: Analyzed the request to configure the LightRAG server with custom functions.
15. **Subtask 18: Identify Server Init Code**: Located the `LightRAG` instantiation within [`lightrag/api/lightrag_server.py`](lightrag/api/lightrag_server.py:1).
16. **Subtask 19: Modify Server Code**: (Conceptually planned, actual modification done by user/separate task) Determined the necessary code additions and modifications for the server file.
17. **Subtask 20: Documentation Update for Server**: This setup guide ([`documents/lightrag_gemini_jina_setup_guide.md`](documents/lightrag_gemini_jina_setup_guide.md:1)) was updated again to include instructions for configuring and running the modified server (this update).