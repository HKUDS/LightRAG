## Install LightRAG as an API Server

LightRAG provides optional API support through FastAPI servers that add RAG capabilities to existing LLM services. You can install LightRAG API Server in two ways:

### Installation from PyPI

```bash
pip install "lightrag-hku[api]"
```

### Installation from Source (Development)

```bash
# Clone the repository
git clone https://github.com/HKUDS/lightrag.git

# Change to the repository directory
cd lightrag

# create a Python virtual enviroment if neccesary
# Install in editable mode with API support
pip install -e ".[api]"
```

### Starting API Server with Default Settings

LightRAG requires both LLM and Embedding Model to work together to complete document indexing and querying tasks. LightRAG supports binding to various LLM/Embedding backends:

* ollama
* lollms
* openai & openai compatible
* azure_openai

Before running any of the servers, ensure you have the corresponding backend service running for both llm and embedding.
The LightRAG API Server provides default parameters for LLM and Embedding, allowing users to easily start the service through command line. These default configurations are:

* Default endpoint of  LLM/Embeding backend(LLM_BINDING_HOST or EMBEDDING_BINDING_HOST)

```
# for lollms backend
LLM_BINDING_HOST=http://localhost:11434
EMBEDDING_BINDING_HOST=http://localhost:11434

# for lollms backend
LLM_BINDING_HOST=http://localhost:9600
EMBEDDING_BINDING_HOST=http://localhost:9600

# for openai, openai compatible or azure openai backend
LLM_BINDING_HOST=https://api.openai.com/v1
EMBEDDING_BINDING_HOST=http://localhost:9600
```

* Default model config

```
LLM_MODEL=mistral-nemo:latest

EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
MAX_EMBED_TOKENS=8192
```

* API keys for LLM/Embedding backend

When connecting to backend require API KEY, corresponding environment variables must be provided:

```
LLM_BINDING_API_KEY=your_api_key
EMBEDDING_BINDING_API_KEY=your_api_key
```

* Use command line arguments to choose LLM/Embeding backend

Use `--llm-binding` to select LLM backend type, and use `--embedding-binding` to select the embedding backend type. All the supported backend types are:

```
openai: LLM default type
ollama: Embedding defult type
lollms:
azure_openai:
openai-ollama: select openai for LLM and ollama for embedding(only valid for --llm-binding)
```

The LightRAG API Server allows you to mix different bindings for llm/embeddings. For example, you have the possibility to use ollama for the embedding and openai for the llm.With the above default parameters, you can start API Server with simple CLI arguments like these:

```
# start with openai llm and ollama embedding
LLM_BINDING_API_KEY=your_api_key Light_server
LLM_BINDING_API_KEY=your_api_key Light_server --llm-binding openai-ollama

# start with openai llm and openai embedding
LLM_BINDING_API_KEY=your_api_key Light_server --llm-binding openai --embedding-binding openai

# start with ollama llm and ollama embedding (no apikey is needed)
Light_server --llm-binding ollama --embedding-binding ollama
```

### For Azure OpenAI Backend
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

```
# Azure OpenAI Configuration in .env
LLM_BINDING=azure_openai
LLM_BINDING_HOST=your-azure-endpoint
LLM_MODEL=your-model-deployment-name
LLM_BINDING_API_KEY=your-azure-api-key
AZURE_OPENAI_API_VERSION=2024-08-01-preview  # optional, defaults to latest version
EMBEDDING_BINDING=azure_openai  # if using Azure OpenAI for embeddings
EMBEDDING_MODEL=your-embedding-deployment-name

```

### Install Lightrag as a Linux Service

Create a your service file `lightrag.sevice` from the sample file : `lightrag.sevice.example`. Modified the WorkingDirectoryand EexecStart in the service file:

```text
Description=LightRAG Ollama Service
WorkingDirectory=<lightrag installed directory>
ExecStart=<lightrag installed directory>/lightrag/api/lightrag-api
```

Modify your service startup script: `lightrag-api`. Change you python virtual environment activation command as needed:

```shell
#!/bin/bash

# your python virtual environment activation
source /home/netman/lightrag-xyj/venv/bin/activate
# start lightrag api server
lightrag-server
```

Install LightRAG service. If your system is Ubuntu, the following commands will work:

```shell
sudo cp lightrag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start lightrag.service
sudo systemctl status lightrag.service
sudo systemctl enable lightrag.service
```

### Automatic Document Indexing

When starting any of the servers with the `--auto-scan-at-startup` parameter, the system will automatically:

1. Scan for new files in the input directory
2. Indexing new documents that aren't already in the database
3. Make all content immediately available for RAG queries

> The `--input-dir` parameter specify the input directory to scan for.

## API Server Configuration

API Server can be config in three way (highest priority first):

* Command line arguments
* Enviroment variables or .env file
* Config.ini (Only for storage configuration)

Most of the configurations come with a default settings, check out details in sample file: `.env.example`. Datastorage configuration can be also set by config.ini. A sample file `config.ini.example` is provided for your convenience.

### LLM and Embedding Backend Supported

LightRAG supports binding to various LLM/Embedding backends:

* ollama
* lollms
* openai & openai compatible
* azure_openai

Use environment variables  `LLM_BINDING ` or CLI argument `--llm-binding` to select LLM backend type. Use environment variables  `EMBEDDING_BINDING ` or CLI argument `--embedding-binding` to select LLM backend type.

### Storage Types Supported

LightRAG uses 4 types of storage for difference purposes:

* KV_STORAGE：llm response cache, text chunks, document information
* VECTOR_STORAGE：entities vectors, relation vectors, chunks vectors
* GRAPH_STORAGE：entity relation graph
* DOC_STATUS_STORAGE：documents indexing status

Each storage type have servals implementations:

* KV_STORAGE supported implement-name

```
JsonKVStorage    JsonFile(default)
MongoKVStorage   MogonDB
RedisKVStorage   Redis
TiDBKVStorage    TiDB
PGKVStorage      Postgres
OracleKVStorage  Oracle
```

* GRAPH_STORAGE supported implement-name

```
NetworkXStorage      NetworkX(defualt)
Neo4JStorage         Neo4J
MongoGraphStorage    MongoDB
TiDBGraphStorage     TiDB
AGEStorage           AGE
GremlinStorage       Gremlin
PGGraphStorage       Postgres
OracleGraphStorage   Postgres
```

* VECTOR_STORAGE supported implement-name

```
NanoVectorDBStorage         NanoVector(default)
MilvusVectorDBStorge        Milvus
ChromaVectorDBStorage       Chroma
TiDBVectorDBStorage         TiDB
PGVectorStorage             Postgres
FaissVectorDBStorage        Faiss
QdrantVectorDBStorage       Qdrant
OracleVectorDBStorage       Oracle
MongoVectorDBStorage        MongoDB
```

* DOC_STATUS_STORAGE：supported implement-name

```
JsonDocStatusStorage        JsonFile(default)
PGDocStatusStorage          Postgres
MongoDocStatusStorage       MongoDB
```

### How Select Storage Implementation

You can select storage implementation by environment variables. Your can set the following environmental variables to a specific storage implement-name before the your first start of the API  Server:

```
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
```

You can not change storage implementation selection after you add documents to LightRAG. Data migration from one storage implementation to anthor is not supported yet. For further information please read the sample env file or config.ini file.

### LightRag API Server Comand Line Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| --host | 0.0.0.0 | Server host |
| --port | 9621 | Server port |
| --working-dir | ./rag_storage | Working directory for RAG storage |
| --input-dir | ./inputs | Directory containing input documents |
| --max-async | 4 | Maximum async operations |
| --max-tokens | 32768 | Maximum token size |
| --timeout | 150 | Timeout in seconds. None for infinite timeout(not recommended) |
| --log-level | INFO | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| --verbose | - | Verbose debug output (True, Flase) |
| --key | None | API key for authentication. Protects lightrag server against unauthorized access |
| --ssl | False | Enable HTTPS |
| --ssl-certfile | None | Path to SSL certificate file (required if --ssl is enabled) |
| --ssl-keyfile | None | Path to SSL private key file (required if --ssl is enabled) |
| --top-k | 50 | Number of top-k items to retrieve; corresponds to entities in "local" mode and relationships in "global" mode. |
| --cosine-threshold | 0.4 | The cossine threshold for nodes and relations retrieval, works with top-k to control the retrieval of nodes and relations. |
| --llm-binding | ollama | LLM binding type (lollms, ollama, openai, openai-ollama, azure_openai) |
| --embedding-binding | ollama | Embedding binding type (lollms, ollama, openai, azure_openai) |
| auto-scan-at-startup | - | Scan input directory for new files and start indexing |

### Example Usage

#### Running a Lightrag server with ollama default local server as llm and embedding backends

Ollama is the default backend for both llm and embedding, so by default you can run lightrag-server with no parameters and the default ones will be used. Make sure ollama is installed and is running and default models are already installed on ollama.

```bash
# Run lightrag with ollama, mistral-nemo:latest for llm, and bge-m3:latest for embedding
lightrag-server

# Using an authentication key
lightrag-server --key my-key
```

#### Running a Lightrag server with lollms default local server as llm and embedding backends

```bash
# Run lightrag with lollms, mistral-nemo:latest for llm, and bge-m3:latest for embedding
# Configure LLM_BINDING=lollms and EMBEDDING_BINDING=lollms in .env or config.ini
lightrag-server

# Using an authentication key
lightrag-server --key my-key
```

#### Running a Lightrag server with openai server as llm and embedding backends

```bash
# Run lightrag with openai, GPT-4o-mini for llm, and text-embedding-3-small for embedding
# Configure in .env or config.ini:
# LLM_BINDING=openai
# LLM_MODEL=GPT-4o-mini
# EMBEDDING_BINDING=openai
# EMBEDDING_MODEL=text-embedding-3-small
lightrag-server

# Using an authentication key
lightrag-server --key my-key
```

#### Running a Lightrag server with azure openai server as llm and embedding backends

```bash
# Run lightrag with azure_openai
# Configure in .env or config.ini:
# LLM_BINDING=azure_openai
# LLM_MODEL=your-model
# EMBEDDING_BINDING=azure_openai
# EMBEDDING_MODEL=your-embedding-model
lightrag-server

# Using an authentication key
lightrag-server --key my-key
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

All servers (LoLLMs, Ollama, OpenAI and Azure OpenAI) provide the same REST API endpoints for RAG functionality. When API Server is running, visit:

- Swagger UI: http://localhost:9621/docs
- ReDoc: http://localhost:9621/redoc

You can test the API endpoints using the provided curl commands or through the Swagger UI interface. Make sure to:

1. Start the appropriate backend service (LoLLMs, Ollama, or OpenAI)
2. Start the RAG server
3. Upload some documents using the document management endpoints
4. Query the system using the query endpoints
5. Trigger document scan if new files is put into inputs directory

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

#### POST /documents/scan

Trigger document scan for new files in the Input directory.

```bash
curl -X POST "http://localhost:9621/documents/scan" --max-time 1800
```

> Ajust max-time according to the estimated index time  for all new files.

#### DELETE /documents

Clear all documents from the RAG system.

```bash
curl -X DELETE "http://localhost:9621/documents"
```

### Ollama Emulation Endpoints

#### GET /api/version

Get Ollama version information.

```bash
curl http://localhost:9621/api/version
```

#### GET /api/tags

Get Ollama available models.

```bash
curl http://localhost:9621/api/tags
```

#### POST /api/chat

Handle chat completion requests. Routes user queries through LightRAG by selecting query mode based on query prefix. Detects and forwards OpenWebUI session-related requests (for meta data generation task) directly to underlying LLM.

```shell
curl -N -X POST http://localhost:9621/api/chat -H "Content-Type: application/json" -d \
  '{"model":"lightrag:latest","messages":[{"role":"user","content":"猪八戒是谁"}],"stream":true}'
```

> For more information about Ollama API pls. visit :  [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)

#### POST /api/generate

Handle generate completion requests. For compatibility purpose, the request is not processed by LightRAG, and will be handled by underlying LLM model.

### Utility Endpoints

#### GET /health
Check server health and configuration.

```bash
curl "http://localhost:9621/health"
```

## Ollama Emulation

We provide an Ollama-compatible interfaces for LightRAG, aiming to emulate LightRAG as an Ollama chat model. This allows AI chat frontends supporting Ollama, such as Open WebUI, to access LightRAG easily.

### Connect Open WebUI to LightRAG

After starting the lightrag-server, you can add an Ollama-type connection in the Open WebUI admin pannel. And then a model named lightrag:latest will appear in Open WebUI's model management interface. Users can then send queries to LightRAG through the chat interface. You'd better install LightRAG as service for this use case.

Open WebUI's use LLM to do the session title and session keyword generation task. So the Ollama chat chat completion API detects and forwards OpenWebUI session-related requests directly to underlying LLM.

### Choose Query mode in chat

A query prefix in the query string can determines which LightRAG query mode is used to generate the respond for the query. The supported prefixes include:

```
/local
/global
/hybrid
/naive
/mix
/bypass
```

For example, chat message "/mix 唐僧有几个徒弟" will trigger a mix mode query for LighRAG. A chat message without query prefix will trigger a hybrid mode query by default。

"/bypass" is not a LightRAG query mode, it will tell API Server to pass the query directly to the underlying LLM with chat history. So user can use LLM to answer question base on the chat history. If you are using Open WebUI as front end, you can just switch the model to a normal LLM instead of using /bypass prefix.
