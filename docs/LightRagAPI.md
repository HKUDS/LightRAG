## Install with API Support

LightRAG provides optional API support through FastAPI servers that add RAG capabilities to existing LLM services. You can install LightRAG with API support in two ways:

### 1. Installation from PyPI

```bash
pip install "lightrag-hku[api]"
```

### 2. Installation from Source (Development)

```bash
# Clone the repository
git clone https://github.com/HKUDS/lightrag.git

# Change to the repository directory
cd lightrag

# Install in editable mode with API support
pip install -e ".[api]"
```

### Prerequisites

Before running any of the servers, ensure you have the corresponding backend service running for both llm and embedding.
The new api allows you to mix different bindings for llm/embeddings.
For example, you have the possibility to use ollama for the embedding and openai for the llm.

#### For LoLLMs Server
- LoLLMs must be running and accessible
- Default connection: http://localhost:9600
- Configure using --llm-binding-host and/or --embedding-binding-host if running on a different host/port

#### For Ollama Server
- Ollama must be running and accessible
- Default connection: http://localhost:11434
- Configure using --ollama-host if running on a different host/port

#### For OpenAI Server
- Requires valid OpenAI API credentials set in environment variables
- OPENAI_API_KEY must be set

#### For Azure OpenAI Server
Azure OpenAI API can be created using the following commands in Azure CLI (you need to install Azure CLI first from [https://docs.microsoft.com/en-us/cli/azure/install-azure-cli](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)):
```bash
# Change the resource group name, location and OpenAI resource name as needed
RESOURCE_GROUP_NAME=LightRAG
LOCATION=swedencentral
RESOURCE_NAME=LightRAG-OpenAI

az login
az group create --name $RESOURCE_GROUP_NAME --location $LOCATION
az cognitiveservices account create --name $RESOURCE_NAME --resource-group $RESOURCE_GROUP_NAME  --kind OpenAI --sku S0 --location swedencentral
az cognitiveservices account deployment create --resource-group $RESOURCE_GROUP_NAME  --model-format OpenAI --name $RESOURCE_NAME --deployment-name gpt-4o --model-name gpt-4o --model-version "2024-08-06"  --sku-capacity 100 --sku-name "Standard"
az cognitiveservices account deployment create --resource-group $RESOURCE_GROUP_NAME  --model-format OpenAI --name $RESOURCE_NAME --deployment-name text-embedding-3-large --model-name text-embedding-3-large --model-version "1"  --sku-capacity 80 --sku-name "Standard"
az cognitiveservices account show --name $RESOURCE_NAME --resource-group $RESOURCE_GROUP_NAME --query "properties.endpoint"
az cognitiveservices account keys list --name $RESOURCE_NAME -g $RESOURCE_GROUP_NAME

```
The output of the last command will give you the endpoint and the key for the OpenAI API. You can use these values to set the environment variables in the `.env` file.


## Configuration

LightRAG can be configured using either command-line arguments or environment variables. When both are provided, command-line arguments take precedence over environment variables.

### Environment Variables

You can configure LightRAG using environment variables by creating a `.env` file in your project root directory. Here's a complete example of available environment variables:

```env
# Server Configuration
HOST=0.0.0.0
PORT=9621

# Directory Configuration
WORKING_DIR=/app/data/rag_storage
INPUT_DIR=/app/data/inputs

# LLM Configuration
LLM_BINDING=ollama
LLM_BINDING_HOST=http://localhost:11434
LLM_MODEL=mistral-nemo:latest

# Embedding Configuration
EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3:latest

# RAG Configuration
MAX_ASYNC=4
MAX_TOKENS=32768
EMBEDDING_DIM=1024
MAX_EMBED_TOKENS=8192

# Security
LIGHTRAG_API_KEY=

# Logging
LOG_LEVEL=INFO

# Optional SSL Configuration
#SSL=true
#SSL_CERTFILE=/path/to/cert.pem
#SSL_KEYFILE=/path/to/key.pem

# Optional Timeout
#TIMEOUT=30
```

### Configuration Priority

The configuration values are loaded in the following order (highest priority first):
1. Command-line arguments
2. Environment variables
3. Default values

For example:
```bash
# This command-line argument will override both the environment variable and default value
python lightrag.py --port 8080

# The environment variable will override the default value but not the command-line argument
PORT=7000 python lightrag.py
```

#### LightRag Server Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| --host | 0.0.0.0 | Server host |
| --port | 9621 | Server port |
| --llm-binding | ollama | LLM binding to be used. Supported: lollms, ollama, openai |
| --llm-binding-host | (dynamic) | LLM server host URL. Defaults based on binding: http://localhost:11434 (ollama), http://localhost:9600 (lollms), https://api.openai.com/v1 (openai) |
| --llm-model | mistral-nemo:latest | LLM model name |
| --embedding-binding | ollama | Embedding binding to be used. Supported: lollms, ollama, openai |
| --embedding-binding-host | (dynamic) | Embedding server host URL. Defaults based on binding: http://localhost:11434 (ollama), http://localhost:9600 (lollms), https://api.openai.com/v1 (openai) |
| --embedding-model | bge-m3:latest | Embedding model name |
| --working-dir | ./rag_storage | Working directory for RAG storage |
| --input-dir | ./inputs | Directory containing input documents |
| --max-async | 4 | Maximum async operations |
| --max-tokens | 32768 | Maximum token size |
| --embedding-dim | 1024 | Embedding dimensions |
| --max-embed-tokens | 8192 | Maximum embedding token size |
| --timeout | None | Timeout in seconds (useful when using slow AI). Use None for infinite timeout |
| --log-level | INFO | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| --key | None | API key for authentication. Protects lightrag server against unauthorized access |
| --ssl | False | Enable HTTPS |
| --ssl-certfile | None | Path to SSL certificate file (required if --ssl is enabled) |
| --ssl-keyfile | None | Path to SSL private key file (required if --ssl is enabled) |



For protecting the server using an authentication key, you can also use an environment variable named `LIGHTRAG_API_KEY`.
### Example Usage

#### Running a Lightrag server with ollama default local server as llm and embedding backends

Ollama is the default backend for both llm and embedding, so by default you can run lightrag-server with no parameters and the default ones will be used. Make sure ollama is installed and is running and default models are already installed on ollama.

```bash
# Run lightrag with ollama, mistral-nemo:latest for llm, and bge-m3:latest for embedding
lightrag-server

# Using specific models (ensure they are installed in your ollama instance)
lightrag-server --llm-model adrienbrault/nous-hermes2theta-llama3-8b:f16 --embedding-model nomic-embed-text --embedding-dim 1024

# Using an authentication key
lightrag-server --key my-key

# Using lollms for llm and ollama for embedding
lightrag-server --llm-binding lollms
```

#### Running a Lightrag server with lollms default local server as llm and embedding backends

```bash
# Run lightrag with lollms, mistral-nemo:latest for llm, and bge-m3:latest for embedding, use lollms for both llm and embedding
lightrag-server --llm-binding lollms --embedding-binding lollms

# Using specific models (ensure they are installed in your ollama instance)
lightrag-server --llm-binding lollms --llm-model adrienbrault/nous-hermes2theta-llama3-8b:f16 --embedding-binding lollms --embedding-model nomic-embed-text --embedding-dim 1024

# Using an authentication key
lightrag-server --key my-key

# Using lollms for llm and openai for embedding
lightrag-server --llm-binding lollms --embedding-binding openai --embedding-model text-embedding-3-small
```


#### Running a Lightrag server with openai server as llm and embedding backends

```bash
# Run lightrag with lollms, GPT-4o-mini  for llm, and text-embedding-3-small for embedding, use openai for both llm and embedding
lightrag-server --llm-binding openai --llm-model GPT-4o-mini --embedding-binding openai --embedding-model text-embedding-3-small

# Using an authentication key
lightrag-server --llm-binding openai --llm-model GPT-4o-mini --embedding-binding openai --embedding-model text-embedding-3-small --key my-key

# Using lollms for llm and openai for embedding
lightrag-server --llm-binding lollms --embedding-binding openai --embedding-model text-embedding-3-small
```

#### Running a Lightrag server with azure openai server as llm and embedding backends

```bash
# Run lightrag with lollms, GPT-4o-mini  for llm, and text-embedding-3-small for embedding, use openai for both llm and embedding
lightrag-server --llm-binding azure_openai --llm-model GPT-4o-mini --embedding-binding openai --embedding-model text-embedding-3-small

# Using an authentication key
lightrag-server --llm-binding azure_openai --llm-model GPT-4o-mini --embedding-binding azure_openai --embedding-model text-embedding-3-small --key my-key

# Using lollms for llm and azure_openai for embedding
lightrag-server --llm-binding lollms --embedding-binding azure_openai --embedding-model text-embedding-3-small
```

**Important Notes:**
- For LoLLMs: Make sure the specified models are installed in your LoLLMs instance
- For Ollama: Make sure the specified models are installed in your Ollama instance
- For OpenAI: Ensure you have set up your OPENAI_API_KEY environment variable
- For Azure OpenAI: Build and configure your server as stated in the Prequisites section

For help on any server, use the --help flag:
```bash
lightrag-server --help
```

Note: If you don't need the API functionality, you can install the base package without API support using:
```bash
pip install lightrag-hku
```

## API Endpoints

All servers (LoLLMs, Ollama, OpenAI and Azure OpenAI) provide the same REST API endpoints for RAG functionality.

### Query Endpoints

#### POST /query
Query the RAG system with options for different search modes.

```bash
curl -X POST "http://localhost:9621/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Your question here", "mode": "hybrid", ""}'
```

#### POST /query/stream
Stream responses from the RAG system.

```bash
curl -X POST "http://localhost:9621/query/stream" \
    -H "Content-Type: application/json" \
    -d '{"query": "Your question here", "mode": "hybrid"}'
```

### Document Management Endpoints

#### POST /documents/text
Insert text directly into the RAG system.

```bash
curl -X POST "http://localhost:9621/documents/text" \
    -H "Content-Type: application/json" \
    -d '{"text": "Your text content here", "description": "Optional description"}'
```

#### POST /documents/file
Upload a single file to the RAG system.

```bash
curl -X POST "http://localhost:9621/documents/file" \
    -F "file=@/path/to/your/document.txt" \
    -F "description=Optional description"
```

#### POST /documents/batch
Upload multiple files at once.

```bash
curl -X POST "http://localhost:9621/documents/batch" \
    -F "files=@/path/to/doc1.txt" \
    -F "files=@/path/to/doc2.txt"
```

#### DELETE /documents
Clear all documents from the RAG system.

```bash
curl -X DELETE "http://localhost:9621/documents"
```

### Utility Endpoints

#### GET /health
Check server health and configuration.

```bash
curl "http://localhost:9621/health"
```

## Development
Contribute to the project: [Guide](contributor-readme.MD)

### Running in Development Mode

For LoLLMs:
```bash
uvicorn lollms_lightrag_server:app --reload --port 9621
```

For Ollama:
```bash
uvicorn ollama_lightrag_server:app --reload --port 9621
```

For OpenAI:
```bash
uvicorn openai_lightrag_server:app --reload --port 9621
```
For Azure OpenAI:
```bash
uvicorn azure_openai_lightrag_server:app --reload --port 9621
```
### API Documentation

When any server is running, visit:
- Swagger UI: http://localhost:9621/docs
- ReDoc: http://localhost:9621/redoc

### Testing API Endpoints

You can test the API endpoints using the provided curl commands or through the Swagger UI interface. Make sure to:
1. Start the appropriate backend service (LoLLMs, Ollama, or OpenAI)
2. Start the RAG server
3. Upload some documents using the document management endpoints
4. Query the system using the query endpoints

### Important Features

#### Automatic Document Vectorization
When starting any of the servers with the `--input-dir` parameter, the system will automatically:
1. Scan the specified directory for documents
2. Check for existing vectorized content in the database
3. Only vectorize new documents that aren't already in the database
4. Make all content immediately available for RAG queries

This intelligent caching mechanism:
- Prevents unnecessary re-vectorization of existing documents
- Reduces startup time for subsequent runs
- Preserves system resources
- Maintains consistency across restarts

**Important Notes:**
- The `--input-dir` parameter enables automatic document processing at startup
- Documents already in the database are not re-vectorized
- Only new documents in the input directory will be processed
- This optimization significantly reduces startup time for subsequent runs
- The working directory (`--working-dir`) stores the vectorized documents database
