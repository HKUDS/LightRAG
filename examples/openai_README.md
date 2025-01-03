
## API Server Implementation

LightRAG also provides a FastAPI-based server implementation for RESTful API access to RAG operations. This allows you to run LightRAG as a service and interact with it through HTTP requests.

### Setting up the API Server
<details>
<summary>Click to expand setup instructions</summary>

1. First, ensure you have the required dependencies:
```bash
pip install fastapi uvicorn pydantic
```

2. Set up your environment variables:
```bash
export RAG_DIR="your_index_directory"  # Optional: Defaults to "index_default"
export OPENAI_BASE_URL="Your OpenAI API base URL"  # Optional: Defaults to "https://api.openai.com/v1"
export OPENAI_API_KEY="Your OpenAI API key"  # Required
export LLM_MODEL="Your LLM model" # Optional: Defaults to "gpt-4o-mini"
export EMBEDDING_MODEL="Your embedding model" # Optional: Defaults to "text-embedding-3-large"
```

3. Run the API server:
```bash
python examples/lightrag_api_openai_compatible_demo.py
```

The server will start on `http://0.0.0.0:8020`.
</details>

### API Endpoints

The API server provides the following endpoints:

#### 1. Query Endpoint
<details>
<summary>Click to view Query endpoint details</summary>

- **URL:** `/query`
- **Method:** POST
- **Body:**
```json
{
    "query": "Your question here",
    "mode": "hybrid",  // Can be "naive", "local", "global", or "hybrid"
    "only_need_context": true // Optional: Defaults to false, if true, only the referenced context will be returned, otherwise the llm answer will be returned
}
```
- **Example:**
```bash
curl -X POST "http://127.0.0.1:8020/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the main themes?", "mode": "hybrid"}'
```
</details>

#### 2. Insert Text Endpoint
<details>
<summary>Click to view Insert Text endpoint details</summary>

- **URL:** `/insert`
- **Method:** POST
- **Body:**
```json
{
    "text": "Your text content here"
}
```
- **Example:**
```bash
curl -X POST "http://127.0.0.1:8020/insert" \
     -H "Content-Type: application/json" \
     -d '{"text": "Content to be inserted into RAG"}'
```
</details>

#### 3. Insert File Endpoint
<details>
<summary>Click to view Insert File endpoint details</summary>

- **URL:** `/insert_file`
- **Method:** POST
- **Body:**
```json
{
    "file_path": "path/to/your/file.txt"
}
```
- **Example:**
```bash
curl -X POST "http://127.0.0.1:8020/insert_file" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "./book.txt"}'
```
</details>

#### 4. Health Check Endpoint
<details>
<summary>Click to view Health Check endpoint details</summary>

- **URL:** `/health`
- **Method:** GET
- **Example:**
```bash
curl -X GET "http://127.0.0.1:8020/health"
```
</details>

### Configuration

The API server can be configured using environment variables:
- `RAG_DIR`: Directory for storing the RAG index (default: "index_default")
- API keys and base URLs should be configured in the code for your specific LLM and embedding model providers
