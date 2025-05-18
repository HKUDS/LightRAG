# LightRAG API and MCP Server Setup Guide

This guide outlines the steps to set up and run the LightRAG API server and the LightRAG MCP server.

## 1. LightRAG API Server Setup

**Prerequisites:**
*   Python (version used for LightRAG, e.g., 3.10)
*   `pip` for package installation

**Steps:**

1.  **Navigate to the LightRAG project directory:**
    ```powershell
    cd c:/Users/victorpiper/Downloads/LightRAG
    ```

2.  **Create/Activate Virtual Environment:**
    Ensure you have a Python virtual environment (e.g., named `.venv`) and activate it.
    If creating a new one (e.g., with Python 3.10):
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3.  **Install LightRAG and API dependencies:**
    ```powershell
    pip install -e .
    pip install -e ".[api]"
    ```
4.  **Custom Jina Embedding Setup (for 768 dimensions):**

    a.  **Create `run_lightrag_gemini_jina.py`:**
        In the root of your LightRAG project (`c:/Users/liminalcommon/Documents/GitHub/LightRAG`), create a Python file named `run_lightrag_gemini_jina.py` with the following content:

        ```python
        import os
        import httpx # Using httpx for async requests
        import json
        from typing import List

        JINA_EMBEDDING_DIM = 768 # For LightRAG server validation

        async def jina_embedding_func(texts: List[str], **kwargs) -> List[List[float]]:
            api_key = os.getenv("JINA_API_KEY")
            if not api_key:
                raise ValueError("JINA_API_KEY environment variable not set.")

            jina_model = "jina-embeddings-v3" # User specified model
            jina_task = "text-matching"      # User specified task

            url = "https://api.jina.ai/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            data = { "model": jina_model, "task": jina_task, "input": texts }

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(url, headers=headers, data=json.dumps(data), timeout=30.0)
                    response.raise_for_status()
                    response_json = response.json()
                    if "data" not in response_json or not isinstance(response_json["data"], list):
                        raise ValueError(f"Unexpected Jina API response: {response_json}")
                    embeddings = [item["embedding"] for item in response_json["data"] if "embedding" in item]
                    if not embeddings and texts:
                         raise ValueError("Jina API returned no embeddings.")
                    return embeddings
                except httpx.HTTPStatusError as e:
                    raise ValueError(f"Jina API request failed: {e.response.status_code} - {e.response.text}") from e
                except Exception as e:
                    raise ValueError(f"Error calling Jina API: {str(e)}") from e
        ```

    c. **Install dependencies for the custom script:**
       Ensure `httpx` is installed in your LightRAG server's Python virtual environment:
       ```powershell
       # While the LightRAG .venv is active:
       pip install httpx
       ```

    d.  **Set Jina API Key in `.env` file:**
        Add your Jina API key to the `.env` file in the LightRAG project root:
        ```
        JINA_API_KEY="jina_840980afcf354b2aab94dd2383ae2d03j4Q25iCpIY9b6fBkhdXeBZdrM7QG"
        ```
        *(Ensure this key is kept secure and not committed to public repositories if your `.env` file is gitignored, which it typically should be).*

5.  **Set API Key Environment Variable (PowerShell - for the current session):**
    ```powershell
    $env:GEMINI_API_KEY = "AIzaSyBi0jME5WSX9dGH-GWVYluNNnbSdJ6Pse4"
    ```
    *Note: For persistent storage, add this to your `.env` file in the LightRAG project root as `LIGHTRAG_API_KEY=AIzaSyBi0jME5WSX9dGH-GWVYluNNnbSdJ6Pse4` or `GEMINI_API_KEY=AIzaSyBi0jME5WSX9dGH-GWVYluNNnbSdJ6Pse4` depending on what the application expects. The previous steps set `LIGHTRAG_API_KEY`.*

6.  **Run the LightRAG API Server:**
    Open a PowerShell terminal.
    ```powershell
    python -m lightrag.api.lightrag_server --use-custom-bindings --host localhost --port 9621 --working-dir "C:\Users\liminalcommon\Documents\GitHub\liminalspace\knowledge_graph" --input-dir ./input
    ```
    *Note: Ensure the `--working-dir` and `--input-dir` paths are correct for your setup.*
    *The API key is now handled via the `JINA_API_KEY` environment variable and the custom `run_lightrag_gemini_jina.py` script when using Jina embeddings. Ensure the `--use-custom-bindings` flag is present.*

## 2. LightRAG MCP Server Setup

**Prerequisites:**
*   Python 3.13+
*   `uv` (Python package manager, install with `pip install uv` if not present)
*   A running LightRAG API server (see section 1).

**Steps:**

1.  **Clone the `lightrag-mcp` repository (if not already done):**
    ```powershell
    cd c:/Users/victorpiper/Downloads/LightRAG 
    git clone https://github.com/shemhamforash23/lightrag-mcp
    ```
    *(Adjust path if LightRAG is elsewhere or if you want to clone `lightrag-mcp` to a different location alongside LightRAG).*

2.  **Navigate to the `lightrag-mcp` directory:**
    ```powershell
    cd c:/Users/victorpiper/Downloads/LightRAG/lightrag-mcp 
    ```
    *(Adjust path according to where you cloned it).*

3.  **Create and Activate Python 3.13 Virtual Environment:**
    ```powershell
    uv venv --python 3.13
    .\.venv\Scripts\Activate.ps1
    ```

4.  **Install `lightrag-mcp` dependencies:**
    ```powershell
    uv pip install -e .
    ```

5.  **Run the LightRAG MCP Server (in a new terminal):**
    Ensure the LightRAG API server is running.
    Open a new PowerShell terminal.
    Navigate to the `lightrag-mcp` directory and activate its virtual environment.
    ```powershell
    cd c:/Users/victorpiper/Downloads/LightRAG/lightrag-mcp 
    .\.venv\Scripts\Activate.ps1
    python src/lightrag_mcp/main.py --host localhost --port 9621 --api-key AIzaSyBi0jME5WSX9dGH-GWVYluNNnbSdJ6Pse4
    ```
    *Note: The `--host` and `--port` should match the running LightRAG API server. The `--api-key` should be the same key used by the LightRAG API server.*

This setup provides a fully operational LightRAG environment with your custom AI model bindings and secure API access, along with an MCP server to interface with it.