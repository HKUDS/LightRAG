# LightRAG Server and WebUI

The LightRAG Server is designed to provide a Web UI and API support. The Web UI facilitates document indexing, knowledge graph exploration, and a simple RAG query interface. LightRAG Server also provides an Ollama-compatible interface, aiming to emulate LightRAG as an Ollama chat model. This allows AI chat bots, such as Open WebUI, to access LightRAG easily.

![image-20250323122538997](./LightRAG-API-Server.assets/image-20250323122538997.png)

![image-20250323122754387](./LightRAG-API-Server.assets/image-20250323122754387.png)

![image-20250323123011220](./LightRAG-API-Server.assets/image-20250323123011220.png)

## Upgrading from v1.4.16 to v1.5.0rc2

The v1.5.0rc2 release adds the new file-processing pipeline, parser routing, multimodal analysis, role-specific LLM/VLM configuration, JSON entity extraction, and several provider/storage changes. Review the [v1.5.0rc2 release notes](https://github.com/HKUDS/LightRAG/releases/tag/v1.5.0rc2) before upgrading a production instance.

- To keep the old file-processing behavior while upgrading the server, set:

```bash
LIGHTRAG_PARSER=*:legacy-F
```

- `ENTITY_TYPES` is no longer supported. Use `ENTITY_TYPE_PROMPT_FILE` instead, with a YAML profile stored under `PROMPT_DIR/entity_type` (`PROMPT_DIR` defaults to `./prompts`). A sample template is available at `prompts/samples/entity_type_prompt.sample.yml`.
- If you use OpenSearch storage and the cluster is older than OpenSearch 3.3.0, upgrade OpenSearch before enabling the v1.5 storage path and validate existing indices. For new deployments, use OpenSearch 3.3.0 or later.
- Changing the embedding model, embedding dimension, asymmetric embedding behavior, or query/document prefixes changes vector semantics. Clear the affected LightRAG workspace/vector data and re-index source files.
- Changing parser routing (`LIGHTRAG_PARSER`) or filename hints affects newly uploaded files. To switch an existing document to another parser engine, delete that document and upload it again.
- Changing chunker settings (`CHUNK_*`) affects documents enqueued after the server restarts. Reprocess older documents if you want their stored `chunk_options` snapshot to match the new settings.
- Enabling multimodal options (`i/t/e`) requires parsed sidecars plus `VLM_PROCESS_ENABLE=true`. Existing documents can be reprocessed to run VLM analysis on available sidecars; switching extraction engines still requires delete + re-upload.

## Getting Started

### Installation

* Install from PyPI

```bash
### Install LightRAG Server as tool using uv (recommended)
uv tool install "lightrag-hku[api]"

### Or using pip
# python -m venv .venv
# source .venv/bin/activate  # Windows: .venv\Scripts\activate
# pip install "lightrag-hku[api]"
```

* Installation from Source

```bash
# Clone the repository
git clone https://github.com/HKUDS/lightrag.git

# Change to the repository directory
cd lightrag

# Bootstrap the development environment (recommended)
make dev
source .venv/bin/activate  # Activate the virtual environment (Linux/macOS)
# Or on Windows: .venv\Scripts\activate

# make dev installs the test toolchain plus the full offline stack
# (API, storage backends, and provider integrations), then builds the frontend.
# Run make env-base or copy env.example to .env before starting the server.

# Equivalent manual steps with uv
# Note: uv sync automatically creates a virtual environment in .venv/
uv sync --extra test --extra offline
source .venv/bin/activate  # Activate the virtual environment (Linux/macOS)
# Or on Windows: .venv\Scripts\activate

# Or using pip with virtual environment
# python -m venv .venv
# source .venv/bin/activate  # Windows: .venv\Scripts\activate
# pip install -e ".[test,offline]"

# Build front-end artifacts
cd lightrag_webui
bun install --frozen-lockfile
bun run build
cd ..
```

### Before Starting LightRAG Server

LightRAG necessitates the integration of both an LLM (Large Language Model) and an Embedding Model to effectively execute document indexing and querying operations. Prior to the initial deployment of the LightRAG server, it is essential to configure the settings for both the LLM and the Embedding Model.

LightRAG supports these LLM backends:

* ollama
* lollms
* openai or openai compatible
* azure_openai
* bedrock
* gemini

LightRAG supports these embedding backends:

* lollms
* ollama
* openai or openai compatible
* azure_openai
* bedrock
* jina
* gemini
* voyageai

It is recommended to use environment variables to configure the LightRAG Server. There is an example environment variable file named `env.example` in the root directory of the project. Please copy this file to the startup directory and rename it to `.env`. After that, you can modify the parameters related to the LLM and Embedding models in the `.env` file. It is important to note that the LightRAG Server will load the environment variables from `.env` into the system environment variables each time it starts. **LightRAG Server will prioritize the settings in the system environment variables to .env file**.

> Since VS Code with the Python extension may automatically load the .env file in the integrated terminal, please open a new terminal session after each modification to the .env file.

If you need to configure different LLMs/VLMs for entity extraction, keyword extraction, final answers, or multimodal analysis, see the [Role-Specific LLM/VLM Configuration Guide](./RoleSpecificLLMConfiguration.md).

Here are some examples of common settings for LLM and Embedding models:

* OpenAI LLM + Ollama Embedding:

```
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key

EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
# EMBEDDING_BINDING_API_KEY=your_api_key
```

> When targeting Google Gemini, set `LLM_BINDING=gemini`, choose a model such as `LLM_MODEL=gemini-flash-latest`, and provide your Gemini key via `LLM_BINDING_API_KEY` (or `GEMINI_API_KEY`).

* Ollama LLM + Ollama Embedding:

```
LLM_BINDING=ollama
LLM_MODEL=mistral-nemo:latest
LLM_BINDING_HOST=http://localhost:11434
# LLM_BINDING_API_KEY=your_api_key
###  Ollama Server context length (Must be larger than MAX_TOTAL_TOKENS+2000)
OLLAMA_LLM_NUM_CTX=16384

EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
# EMBEDDING_BINDING_API_KEY=your_api_key
```

> **Important Note**: The embedding model and asymmetric embedding configuration must be determined before document indexing, and the same settings must be used during the query phase. For certain storage solutions (e.g., PostgreSQL), the vector dimension must be defined upon initial table creation. When changing the embedding model, embedding dimension, `EMBEDDING_ASYMMETRIC`, query/document prefixes, or provider task behavior, clear the existing LightRAG workspace/vector data and re-index the source files.

#### Asymmetric Embedding Configuration

LightRAG uses symmetric embeddings by default. Query/document asymmetric embeddings are enabled only when `EMBEDDING_ASYMMETRIC=true` is explicitly set.

- Provider task bindings such as `jina`, `gemini`, and `voyageai` use provider parameters (`task` / `task_type` / `input_type`) and should not use query/document prefixes.
- Prefix-based bindings such as `openai`, `azure_openai`, and `ollama` require both `EMBEDDING_QUERY_PREFIX` and `EMBEDDING_DOCUMENT_PREFIX`. Use `NO_PREFIX` for a side that should intentionally have no prefix.
- Any valid change to asymmetric embedding settings requires clearing existing data and re-indexing files.

For the full validation rules and examples, see [Asymmetric Embedding Configuration](./AsymmetricEmbedding.md).

### Create .env File With Setup Tool

Instead of editing `env.example` by hand, you can use the interactive setup wizard to generate a configured `.env` and, when needed, `docker-compose.final.yml`:

```bash
make env-base           # Required first step: LLM, embedding, reranker
make env-storage        # Optional: storage backends and database services
make env-server         # Optional: server port, auth, and SSL
make env-security-check # Optional: audit the current .env for security risks
```

For a full description of every target and what each flow does, see [docs/InteractiveSetup.md](./InteractiveSetup.md).
The setup wizards update configuration only; run `make env-security-check` separately to audit the
current `.env` for security risks before deployment.

### Starting LightRAG Server

The LightRAG Server supports two operational modes:
* The simple and efficient Uvicorn mode:

```
lightrag-server
```
* The multiprocess Gunicorn + Uvicorn mode (production mode, not supported on Windows environments):

```
lightrag-gunicorn --workers 4
```

When starting LightRAG, the current working directory must contain the `.env` configuration file. **It is intentionally designed that the `.env` file must be placed in the startup directory**. The purpose of this is to allow users to launch multiple LightRAG instances simultaneously and configure different `.env` files for different instances. **After modifying the `.env` file, you need to reopen the terminal for the new settings to take effect.** This is because each time LightRAG Server starts, it loads the environment variables from the `.env` file into the system environment variables, and system environment variables have higher precedence.

During startup, configurations in the `.env` file can be overridden by command-line parameters. Common command-line parameters include:

- `--host`: Server listening address (default: 0.0.0.0)
- `--port`: Server listening port (default: 9621)
- `--timeout`: LLM request timeout (default: 150 seconds)
- `--log-level`: Log level (default: INFO)
- `--working-dir`: Database persistence directory (default: ./rag_storage)
- `--input-dir`: Directory for uploaded files (default: ./inputs)
- `--workspace`: Workspace name, used to logically isolate data between multiple LightRAG instances (default: empty)
- `--api-prefix`: Reverse-proxy path prefix exposed to browsers, also configurable with `LIGHTRAG_API_PREFIX`
- `--rerank-binding`: Rerank provider (`null`, `cohere`, `jina`, or `aliyun`)

### Path Prefix and Multi-Site WebUI

Set `LIGHTRAG_API_PREFIX` or `--api-prefix` when one host serves multiple LightRAG instances behind a reverse proxy that strips a site prefix before forwarding to the backend:

```bash
LIGHTRAG_API_PREFIX=/site01
lightrag-server --port 9621
```

The backend passes this value to FastAPI as `root_path` and injects the same runtime prefix into the WebUI. The WebUI is always mounted at `/webui` inside the server, so one frontend build can serve any prefix. See [Single-Server Multi-Site Deployment](./MultiSiteDeployment.md) for full Nginx, Docker, and Kubernetes examples.

### Launching LightRAG Server with Docker

Using Docker Compose is the most convenient way to deploy and run the LightRAG Server.

- Create a project directory.
- Copy the `docker-compose.yml` file from the LightRAG repository into your project directory.
- Prepare the `.env` file: Duplicate the sample file [`env.example`](https://ai.znipower.com:5013/c/env.example)to create a customized `.env` file, and configure the LLM and embedding parameters according to your specific requirements.
- Start the LightRAG Server with the following command:

```shell
docker compose up
# If you want the program to run in the background after startup, add the -d parameter at the end of the command.
```

You can get the official docker compose file from here: [docker-compose.yml](https://raw.githubusercontent.com/HKUDS/LightRAG/refs/heads/main/docker-compose.yml). For historical versions of LightRAG docker images, visit this link: [LightRAG Docker Images](https://github.com/HKUDS/LightRAG/pkgs/container/lightrag). For more details about docker deployment, please refer to [DockerDeployment.md](./DockerDeployment.md).

### Progressive Setup Recipes

If you are new to LightRAG, start with the smallest working configuration and add capabilities only after the previous step is healthy:

1. Minimal Docker run with hosted LLM and embedding models
2. Add reranking to improve query quality
3. Add multimodal parsing with MinerU and a vision-capable model
4. Move to a GPU-backed, Docker-managed deployment with database storage

The full `env.example` file remains the complete configuration reference and is used by the `make env-*` setup wizard. The snippets below intentionally show only the values that matter for each step.

#### 1. Minimal Docker Run

Use this path when you want the WebUI and API running first, with no external database, parser service, or local model service. Create `.env` next to `docker-compose.yml` with a minimal OpenAI-compatible configuration:

```bash
###########################
### Server Configuration
###########################
PORT=9621
WEBUI_TITLE='My First LightRAG KB'
WEBUI_DESCRIPTION='Simple and Fast Graph Based RAG System'
OLLAMA_EMULATING_MODEL_TAG=latest

########################################
### Document processing configuration
########################################
SUMMARY_LANGUAGE=English
ENTITY_EXTRACTION_USE_JSON=true
LIGHTRAG_PARSER=*:native-teP,*:legacy-R
VLM_PROCESS_ENABLE=false

###########################################################################
### LLM Configuration
###########################################################################
LLM_BINDING=openai
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key
LLM_MODEL=gpt-5-mini

KEYWORD_LLM_MODEL=gpt-5-nano
QUERY_LLM_MODEL=gpt-5

#######################################################################################
### Embedding Configuration (do not change after the first file is processed)
#######################################################################################
EMBEDDING_BINDING=openai
EMBEDDING_BINDING_HOST=https://api.openai.com/v1
EMBEDDING_BINDING_API_KEY=your_api_key
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
EMBEDDING_TOKEN_LIMIT=8192
EMBEDDING_SEND_DIM=false
EMBEDDING_USE_BASE64=true

############################
### Data storage selection
############################
LIGHTRAG_KV_STORAGE=JsonKVStorage
LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage
LIGHTRAG_GRAPH_STORAGE=NetworkXStorage
LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage
```

Replace the model IDs with models available in your provider account when needed. Start the service and verify it before uploading documents:

```bash
docker compose up -d
curl http://localhost:9621/health
```

Then open the WebUI at `http://localhost:9621/webui`, upload a small text or DOCX file, wait for indexing to finish, and run a `hybrid` or `mix` query.

#### 2. Add Reranking

Reranking is a query-time improvement. Enabling, disabling, or changing the reranker usually does not require re-indexing existing documents.

For Cohere's official hosted rerank service:

```bash
RERANK_BINDING=cohere
RERANK_MODEL=rerank-v3.5
RERANK_BINDING_HOST=https://api.cohere.com/v2/rerank
RERANK_BINDING_API_KEY=your_cohere_api_key
```

For a local vLLM reranker that exposes a Cohere-compatible API:

```bash
RERANK_BINDING=cohere
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=http://localhost:8000/rerank
RERANK_BINDING_API_KEY=your_rerank_api_key_here
```

If LightRAG itself runs inside Docker and the reranker runs on the host, use a host-reachable address such as `host.docker.internal` instead of `localhost`. If the setup wizard creates the vLLM service, it injects the internal Compose service URL into `docker-compose.final.yml` for you.

#### 3. Add Multimodal Parsing With MinerU Official API

Use this after the basic document flow works. The MinerU official API avoids running a local parser service, but `MINERU_API_TOKEN` must be configured before the LightRAG server starts. The VLM role must use a provider/model that supports image input.

```bash
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R

VLM_PROCESS_ENABLE=true
VLM_LLM_MODEL=gpt-5-mini

MINERU_API_MODE=official
MINERU_API_TOKEN=your_mineru_api_token
MINERU_OFFICIAL_ENDPOINT=https://mineru.net
MINERU_MODEL_VERSION=vlm
MINERU_IS_OCR=false
```

This routing uses the built-in `native` parser for supported DOCX files, MinerU for other MinerU-supported files such as PDFs and images, and `legacy` as the fallback. The `i`, `t`, and `e` options enable VLM analysis for image, table, and equation sidecars when the parser produces them.

For official mode, Docker does not need a host-loopback MinerU endpoint. The container only needs outbound network access to `MINERU_OFFICIAL_ENDPOINT`.

#### 4. GPU All-In-One Style Deployment

For a local GPU-backed deployment, let the wizard generate `.env` and `docker-compose.final.yml` instead of hand-writing every service block:

```bash
make env-base
```

Recommended answers:

- Configure the main LLM as a hosted or OpenAI-compatible provider.
- Answer `yes` to `Run embedding model locally via Docker (vLLM)?`.
- Choose `cuda` for the embedding device.
- Enable reranking, answer `yes` to `Run rerank service locally via Docker?`, and choose `cuda` for the rerank device.

Then configure storage:

```bash
make env-storage
```

Recommended storage choices:

- `LIGHTRAG_KV_STORAGE=PGKVStorage`
- `LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage`
- `LIGHTRAG_VECTOR_STORAGE=MilvusVectorDBStorage`
- `LIGHTRAG_GRAPH_STORAGE=MemgraphStorage`
- Answer `yes` to run PostgreSQL, Milvus, and Memgraph locally via Docker.
- Choose `cuda` for Milvus if your host has NVIDIA GPU support and the NVIDIA Container Toolkit is installed.

Finally configure server-facing settings and validate the result:

```bash
make env-server
make env-validate
make env-security-check
docker compose -f docker-compose.final.yml up -d
```

Before exposing this deployment, configure authentication, API keys, and SSL in `make env-server`. The generated `.env` stays host-usable; container-only service names and Docker-specific overrides are written into `docker-compose.final.yml`.

Important rules before processing production data:

- Choose the embedding model, embedding dimension, and asymmetric embedding settings before the first upload. Changing them later requires clearing the affected workspace/vector data and re-indexing documents.
- Choose storage backends before the first upload. Direct migration between storage implementations is not supported.
- Changing `LIGHTRAG_PARSER` affects only newly uploaded files. Delete and upload an existing document again if you want it processed by a different parser route.

### Nginx Reverse Proxy Configuration

When using Nginx as a reverse proxy in front of LightRAG Server, you need to configure `client_max_body_size` for the `/documents/upload` endpoint to handle large file uploads. Without this configuration, Nginx will reject files larger than 1MB (the default limit) with a `413 Request Entity Too Large` error before the request reaches LightRAG.

**Recommended Configuration:**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Global default: 8MB for LLM queries with long context
    client_max_body_size 8M;

    # Upload endpoint: 100MB for large file uploads
    location /documents/upload {
        client_max_body_size 100M;

        proxy_pass http://localhost:9621;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeouts for large file uploads
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # Streaming endpoints: LLM response streaming
    location ~ ^/(query/stream|api/chat|api/generate) {
        gzip off;  # Disable compression for streaming responses

        proxy_pass http://localhost:9621;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Long timeout for LLM generation
        proxy_read_timeout 300s;
    }

    # Other endpoints
    location / {
        proxy_pass http://localhost:9621;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Key Points:**

1. **Global Limit (8MB)**: Sufficient for LLM queries with long conversation history and context (128K tokens ≈ 512KB + JSON overhead).
2. **Upload Endpoint (100MB)**: Must match or exceed `MAX_UPLOAD_SIZE` in your `.env` file. The default `MAX_UPLOAD_SIZE` is 100MB.
3. **Streaming Endpoints**: Disable gzip compression (`gzip off`) for streaming endpoints to ensure real-time response delivery. LightRAG automatically sets `X-Accel-Buffering: no` header to disable response buffering.
4. **Timeout Settings**: Large file uploads and LLM generation require longer timeouts; adjust `proxy_read_timeout` and `proxy_send_timeout` accordingly.
5. **Size Validation Layers**:
   - Nginx validates the `Content-Length` header first
   - LightRAG performs streaming validation during upload
   - Setting appropriate limits at both layers ensures better error messages and security

### Offline Deployment

Official LightRAG Docker images are fully compatible with offline or air-gapped environments. If you want to build up you own  offline enviroment, please refer to [Offline Deployment Guide](./OfflineDeployment.md).

### Starting Multiple LightRAG Instances

There are two ways to start multiple LightRAG instances. The first way is to configure a completely independent working environment for each instance. This requires creating a separate working directory for each instance and placing a dedicated `.env` configuration file in that directory. The server listening ports in the configuration files of different instances cannot be the same. Then, you can start the service by running `lightrag-server` in the working directory.

The second way is for all instances to share the same set of `.env` configuration files, and then use command-line arguments to specify different server listening ports and workspaces for each instance. You can start multiple LightRAG instances in the same working directory with different command-line arguments. For example:

```
# Start instance 1
lightrag-server --port 9621 --workspace space1

# Start instance 2
lightrag-server --port 9622 --workspace space2
```

The purpose of a workspace is to achieve data isolation between different instances. Therefore, the `workspace` parameter must be different for different instances; otherwise, it will lead to data confusion and corruption.

When launching multiple LightRAG instances via Docker Compose, simply specify unique `WORKSPACE` and `PORT` environment variables for each container within your `docker-compose.yml`. Even if all instances share a common `.env` file, the container-specific environment variables defined in Compose will take precedence, ensuring independent configurations for each instance.

### Data Isolation Between LightRAG Instances

Configuring an independent working directory and a dedicated `.env` configuration file for each instance can generally ensure that locally persisted files in the in-memory database are saved in their respective working directories, achieving data isolation. By default, LightRAG uses all in-memory databases, and this method of data isolation is sufficient. However, if you are using an external database, and different instances access the same database instance, you need to use workspaces to achieve data isolation; otherwise, the data of different instances will conflict and be destroyed.

The command-line `workspace` argument and the `WORKSPACE` environment variable in the `.env` file can both be used to specify the workspace name for the current instance, with the command-line argument having higher priority. Here is how workspaces are implemented for different types of storage:

- **For local file-based databases, data isolation is achieved through workspace subdirectories:** `JsonKVStorage`, `JsonDocStatusStorage`, `NetworkXStorage`, `NanoVectorDBStorage`, `FaissVectorDBStorage`.
- **For databases that store data in collections, it's done by adding a workspace prefix to the collection name:** `RedisKVStorage`, `RedisDocStatusStorage`, `MilvusVectorDBStorage`, `MongoKVStorage`, `MongoDocStatusStorage`, `MongoVectorDBStorage`, `MongoGraphStorage`, `PGGraphStorage`.
- **For Qdrant vector database, data isolation is achieved through payload-based partitioning (Qdrant's recommended multitenancy approach):** `QdrantVectorDBStorage` uses shared collections with payload filtering for unlimited workspace scalability.
- **For relational databases, data isolation is achieved by adding a `workspace` field to the tables for logical data separation:** `PGKVStorage`, `PGVectorStorage`, `PGDocStatusStorage`.
- **For graph databases, logical data isolation is achieved through labels:** `Neo4JStorage`, `MemgraphStorage`
- **For OpenSearch, data isolation is achieved through index name prefixes:** `OpenSearchKVStorage`, `OpenSearchDocStatusStorage`, `OpenSearchGraphStorage`, `OpenSearchVectorDBStorage`

To maintain compatibility with legacy data, the default workspace for PostgreSQL is `default` and for Neo4j is `base` when no workspace is configured. For all external storages, the system provides dedicated workspace environment variables to override the common `WORKSPACE` environment variable configuration. These storage-specific workspace environment variables are: `REDIS_WORKSPACE`, `MILVUS_WORKSPACE`, `QDRANT_WORKSPACE`, `MONGODB_WORKSPACE`, `POSTGRES_WORKSPACE`, `NEO4J_WORKSPACE`, `MEMGRAPH_WORKSPACE`, `OPENSEARCH_WORKSPACE`.

### Multiple workers for Gunicorn + Uvicorn

The LightRAG Server can operate in the `Gunicorn + Uvicorn` preload mode. Gunicorn's multiple worker (multiprocess) capability prevents document indexing tasks from blocking RAG queries. CPU-heavy document extraction tools should be deployed as external services so they do not block the API process.

Though LightRAG Server uses one worker to process the document indexing pipeline, with the async task support of Uvicorn, multiple files can be processed in parallel. The bottleneck of document indexing speed mainly lies with the LLM. If your LLM supports high concurrency, you can accelerate document indexing by increasing the concurrency level of the LLM. Below are several environment variables related to concurrent processing, along with their default values:

```
### Number of worker processes, not greater than (2 x number_of_cores) + 1
WORKERS=2
### Number of parallel files to process in one batch
MAX_PARALLEL_INSERT=3
### Max concurrent requests to the LLM (MAX_ASYNC is still accepted as a deprecated alias)
MAX_ASYNC_LLM=4
```

On macOS, Gunicorn multi-worker mode also requires the Objective-C fork-safety override to be present before the Python process starts. Do not rely on `.env` for this variable; `.env` is loaded after Python startup and is too late for the Objective-C runtime:

```shell
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
lightrag-gunicorn --workers 2
```

### Install LightRAG as a Linux Service

Create your service file `lightrag.service` from the sample file: `lightrag.service.example`. Modify the start options the service file:

```text
# Set Enviroment to your Python virtual enviroment
Environment="PATH=/home/netman/lightrag-xyj/venv/bin"
WorkingDirectory=/home/netman/lightrag-xyj
# ExecStart=/home/netman/lightrag-xyj/venv/bin/lightrag-server
ExecStart=/home/netman/lightrag-xyj/venv/bin/lightrag-gunicorn
```

> The ExecStart command must be either `lightrag-gunicorn` or `lightrag-server`; no wrapper scripts are allowed. This is because service termination requires the main process to be one of these two executables.

Install LightRAG service. If your system is Ubuntu, the following commands will work:

```shell
sudo cp lightrag.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start lightrag.service
sudo systemctl status lightrag.service
sudo systemctl enable lightrag.service
```

## Ollama Emulation

We provide Ollama-compatible interfaces for LightRAG, aiming to emulate LightRAG as an Ollama chat model. This allows AI chat frontends supporting Ollama, such as Open WebUI, to access LightRAG easily.

### Connect Open WebUI to LightRAG

After starting the lightrag-server, you can add an Ollama-type connection in the Open WebUI admin panel. And then a model named `lightrag:latest` will appear in Open WebUI's model management interface. Users can then send queries to LightRAG through the chat interface. You should install LightRAG as a service for this use case.

Open WebUI uses an LLM to do the session title and session keyword generation task. So the Ollama chat completion API detects and forwards OpenWebUI session-related requests directly to the underlying LLM. Screenshot from Open WebUI:

![image-20250323194750379](./LightRAG-API-Server.assets/image-20250323194750379.png)

### Choose Query mode in chat

The default query mode is `hybrid` if you send a message (query) from the Ollama interface of LightRAG. You can select query mode by sending a message with a query prefix.

A query prefix in the query string can determine which LightRAG query mode is used to generate the response for the query. The supported prefixes include:

```
/local
/global
/hybrid
/naive
/mix

/bypass
/context
/localcontext
/globalcontext
/hybridcontext
/naivecontext
/mixcontext
```

For example, the chat message `/mix What's LightRAG?` will trigger a mix mode query for LightRAG. A chat message without a query prefix will trigger a hybrid mode query by default.

`/bypass` is not a LightRAG query mode; it will tell the API Server to pass the query directly to the underlying LLM, including the chat history. So the user can use the LLM to answer questions based on the chat history. If you are using Open WebUI as a front end, you can just switch the model to a normal LLM instead of using the `/bypass` prefix.

`/context` is also not a LightRAG query mode; it will tell LightRAG to return only the context information prepared for the LLM. You can check the context if it's what you want, or process the context by yourself.

### Add user prompt in chat

When using LightRAG for content queries, avoid combining the search process with unrelated output processing, as this significantly impacts query effectiveness. User prompt is specifically designed to address this issue — it does not participate in the RAG retrieval phase, but rather guides the LLM on how to process the retrieved results after the query is completed. We can append square brackets to the query prefix to provide the LLM with the user prompt:

```
/[Use mermaid format for diagrams] Please draw a character relationship diagram for Scrooge
/mix[Use mermaid format for diagrams] Please draw a character relationship diagram for Scrooge
```

## API Key and Authentication

By default, the LightRAG Server can be accessed without any authentication. We can configure the server with an API Key or account credentials to secure it.

* API Key:

```
LIGHTRAG_API_KEY=your-secure-api-key-here
WHITELIST_PATHS=/health,/api/*
```

> Health check and Ollama emulation endpoints are excluded from API Key check by default. For security reasons, remove `/api/*` from `WHITELIST_PATHS` if the Ollama service is not required. `/health` stays whitelisted as a liveness probe but only returns its full configuration to authenticated callers — unauthenticated requests get liveness signals only.

The API key is passed using the request header `X-API-Key`. Below is an example of accessing the LightRAG Server via API:

```
curl -X 'POST' \
  'http://localhost:9621/documents/scan' \
  -H 'accept: application/json' \
  -H 'X-API-Key: your-secure-api-key-here-123' \
  -d ''
```

* Account credentials (the Web UI requires login before access can be granted):

LightRAG API Server implements JWT-based authentication using the HS256 algorithm. To enable secure access control, the following environment variables are required:

```bash
# For jwt auth
AUTH_ACCOUNTS='admin:{bcrypt}$2b$12$replace-with-generated-hash,user1:pass456'
TOKEN_SECRET='your-key'
TOKEN_EXPIRE_HOURS=4
```

Passwords without a prefix are treated as plaintext. To store a bcrypt password, prefix the generated hash with `{bcrypt}`. The easiest way to generate a value that can be pasted directly into `AUTH_ACCOUNTS` is:

```bash
lightrag-hash-password --username admin
```

The command prompts for the password and prints an `admin:{bcrypt}...` entry ready to paste into `.env`.

> Currently, only the configuration of an administrator account and password is supported. A comprehensive account system is yet to be developed and implemented.

If Account credentials are not configured, the Web UI will access the system as a Guest. Therefore, even if only an API Key is configured, all APIs can still be accessed through the Guest account, which remains insecure. Hence, to safeguard the API, it is necessary to configure both authentication methods simultaneously.

## For Azure OpenAI Backend

Azure OpenAI API can be created using the following commands in Azure CLI (you need to install Azure CLI first from [https://docs.microsoft.com/en-us/cli/azure/install-azure-cli](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)):

```bash
# Change the resource group name, location, and OpenAI resource name as needed
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
# Azure OpenAI Configuration in .env:
LLM_BINDING=azure_openai
LLM_BINDING_HOST=your-azure-endpoint
LLM_MODEL=your-model-deployment-name
LLM_BINDING_API_KEY=your-azure-api-key
### API version is optional, defaults to latest version
AZURE_OPENAI_API_VERSION=2024-08-01-preview

### If using Azure OpenAI for embeddings
EMBEDDING_BINDING=azure_openai
EMBEDDING_MODEL=your-embedding-deployment-name
```

## LightRAG Server Configuration in Detail

The API Server can be configured in two ways (highest priority first):

* Command line arguments
* Environment variables or .env file

Most of the configurations come with default settings; check out the details in the sample file: `.env.example`. Storage configuration should also be set through environment variables or the `.env` file.

### LLM and Embedding Backend Supported

LightRAG supports binding to various LLM backends:

* ollama
* openai (including openai compatible)
* azure_openai
* lollms
* bedrock
* gemini

LightRAG supports binding to various Embedding backends:

* lollms
* ollama
* openai (including openai compatible)
* azure_openai
* bedrock
* jina
* gemini
* voyageai

Use environment variables `LLM_BINDING` or CLI argument `--llm-binding` to select the LLM backend type. Use environment variables `EMBEDDING_BINDING` or CLI argument `--embedding-binding` to select the Embedding backend type.

Bedrock ignores `LLM_BINDING_API_KEY` and `EMBEDDING_BINDING_API_KEY`. Use SigV4 credentials through the AWS credential chain, or set the process-level `AWS_BEARER_TOKEN_BEDROCK` environment variable before startup for Bedrock API key / bearer-token auth:

```bash
LLM_BINDING=bedrock
LLM_BINDING_HOST=DEFAULT_BEDROCK_ENDPOINT
LLM_MODEL=us.amazon.nova-lite-v1:0
AWS_REGION=us-west-2
# Use the AWS credential chain, or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY,
# or set AWS_BEARER_TOKEN_BEDROCK before starting the server.
```

Asymmetric embedding is explicit opt-in. Set `EMBEDDING_ASYMMETRIC=true` only when the selected embedding backend supports either provider task parameters or task prefixes. See [Asymmetric Embedding Configuration](./AsymmetricEmbedding.md) before changing these settings, because existing data must be cleared and files re-indexed after any change.

For LLM and embedding configuration examples, please refer to the `env.example` file in the project's root directory. To view the complete list of configurable options for OpenAI and Ollama-compatible LLM interfaces, use the following commands:

```
lightrag-server --llm-binding openai --help
lightrag-server --llm-binding ollama --help
lightrag-server --llm-binding gemini --help
lightrag-server --embedding-binding ollama --help
lightrag-server --embedding-binding gemini --help
```

> Please use OpenAI-compatible method to access LLMs deployed by OpenRouter or vLLM/SGLang. You can pass additional parameters to OpenRouter or vLLM/SGLang through the `OPENAI_LLM_EXTRA_BODY` environment variable to disable reasoning mode or achieve other personalized controls.

Set the max_tokens to **prevent excessively long or endless output loop** during the entity relationship extraction phase for Large Language Model (LLM) responses.  The purpose of setting max_tokens parameter is to truncate LLM output before timeouts occur, thereby preventing document extraction failures. This addresses issues where certain text blocks (e.g., tables or citations) containing numerous entities and relationships can lead to overly long or even endless loop outputs from LLMs. This setting is particularly crucial for locally deployed, smaller-parameter models. Max tokens value can be calculated by this formula: `LLM_TIMEOUT * llm_output_tokens/second` (i.e. `240s * 50 tokens/s = 12000`, max_tokens should smaller than 12000)

```
# For vLLM/SGLang doployed models, or most of OpenAI compatible API provider
OPENAI_LLM_MAX_TOKENS=9000

# For Ollama Deployed Modeles
OLLAMA_LLM_NUM_PREDICT=9000

# For OpenAI o1-mini or newer modles
OPENAI_LLM_MAX_COMPLETION_TOKENS=9000
```

### Role-Specific LLM/VLM Configuration

The server can use different models for different stages without changing client APIs. Four roles are supported:

| Role | Purpose |
| --- | --- |
| `EXTRACT` | Entity/relation extraction and merge summaries |
| `KEYWORD` | Query keyword generation before retrieval |
| `QUERY` | Final answers, bypass queries, and Ollama-compatible chat responses |
| `VLM` | Multimodal analysis for images, tables, equations, and similar sidecar items |

If a role is not configured, it inherits the base `LLM_*` settings. Minimal same-provider example:

```bash
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key

EXTRACT_LLM_MODEL=gpt-5-mini
KEYWORD_LLM_MODEL=gpt-5-nano
QUERY_LLM_MODEL=gpt-5
VLM_LLM_MODEL=gpt-5-mini
```

For cross-provider rules, provider-specific options such as `QUERY_OPENAI_LLM_REASONING_EFFORT`, role-level Bedrock SigV4 credentials, and queue behavior, see [Role-Specific LLM/VLM Configuration Guide](./RoleSpecificLLMConfiguration.md).

### Multimodal Analysis Configuration

The parser can produce sidecars for drawings/images, tables, and equations. VLM analysis only runs when both conditions are true:

- The document's `process_options` contains the matching modality flag: `i` for images, `t` for tables, or `e` for equations.
- `VLM_PROCESS_ENABLE=true` and the effective VLM binding supports image input.

Current vision-capable providers are `openai`, `azure_openai`, `gemini`, `bedrock`, `ollama`, and `anthropic`; `lollms` is rejected for VLM use. Typical configuration:

```bash
VLM_PROCESS_ENABLE=true
VLM_LLM_BINDING=openai
VLM_LLM_MODEL=gpt-4o
VLM_LLM_BINDING_HOST=https://api.openai.com/v1
VLM_LLM_BINDING_API_KEY=your_vlm_api_key
VLM_MAX_IMAGE_BYTES=5242880
SURROUNDING_LEADING_MAX_TOKENS=2000
SURROUNDING_TRAILING_MAX_TOKENS=2000
```

The surrounding-context budgets control how much nearby text is included in VLM and extraction prompts for a multimodal item. Parser and per-file option examples are in [Document and Chunk Processing](#document-and-chunk-processing).

### Entity Extraction Configuration

Entity extraction is controlled by the base or `EXTRACT` role LLM. Important server-side options:

- `ENTITY_EXTRACTION_USE_JSON`: request JSON-structured extraction output. In v1.5 this is recommended for reliability, but it can increase latency.
- `ENTITY_TYPE_PROMPT_FILE`: file-name-only YAML profile for entity type guidance and examples. The file is loaded from `PROMPT_DIR/entity_type`; do not pass an absolute path here.
- `MAX_EXTRACT_INPUT_TOKENS`: maximum token budget for one extraction input context.
- `MAX_EXTRACTION_RECORDS`: per-response cap for total entity and relationship records.
- `MAX_EXTRACTION_ENTITIES`: per-response cap for entity records.

Example:

```bash
ENTITY_EXTRACTION_USE_JSON=true
ENTITY_TYPE_PROMPT_FILE=entity_type_prompt.yml
PROMPT_DIR=/opt/lightrag/prompts
MAX_EXTRACT_INPUT_TOKENS=20480
MAX_EXTRACTION_RECORDS=100
MAX_EXTRACTION_ENTITIES=40
```

If an old `.env` still contains `ENTITY_TYPES`, remove it before startup. The server fails fast because this variable has been replaced by prompt profiles.

### Storage Types Supported

LightRAG uses 4 types of storage for different purposes:

* KV_STORAGE: llm response cache, text chunks, document information
* VECTOR_STORAGE: entities vectors, relation vectors, chunks vectors
* GRAPH_STORAGE: entity relation graph
* DOC_STATUS_STORAGE: document indexing status

LightRAG Server offers various storage implementations, with the default being an in-memory database that persists data to the WORKING_DIR directory. Additionally, LightRAG supports a wide range of storage solutions including PostgreSQL, MongoDB, FAISS, Milvus, Qdrant, Neo4j, Memgraph, Redis, and OpenSearch. For detailed information on supported storage options, please refer to the storage section in the README.md file located in the root directory.

**Milvus Index Configuration:** LightRAG now supports configurable index types for Milvus vector storage (AUTOINDEX, HNSW, HNSW_SQ, IVF_FLAT, etc.) through environment variables. HNSW_SQ requires Milvus 2.6.8+ and provides significant memory savings. See the "Using Milvus for Vector Storage" section in the main README.md for complete configuration options.

You can select the storage implementation by configuring environment variables. For instance, prior to the initial launch of the API server, you can set the following environment variable to specify your desired storage implementation:

```
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
```

You cannot change storage implementation selection after adding documents to LightRAG. Data migration from one storage implementation to another is not supported yet. For further information, please read the sample `.env.example` file.

### LLM Cache Migration Between Storage Types

When switching the storage implementation in LightRAG, the LLM cache can be migrated from the existing storage to the new one. Subsequently, when re-uploading files to the new storage, the pre-existing LLM cache will significantly accelerate file processing. For detailed instructions on using the LLM cache migration tool, please refer to [README_MIGRATE_LLM_CACHE.md](../lightrag/tools/README_MIGRATE_LLM_CACHE.md)

### LightRAG API Server Command Line Options

| Parameter | Default | Description |
| --- | --- | --- |
| `--host` | `0.0.0.0` | Server host |
| `--port` | `9621` | Server port |
| `--working-dir` | `./rag_storage` | Working directory for RAG storage |
| `--input-dir` | `./inputs` | Directory containing uploaded/input documents |
| `--timeout` | `150` | Gunicorn worker timeout and fallback request timeout |
| `--max-async` | `4` | Maximum concurrent LLM operations |
| `--log-level` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |
| `--verbose` | `False` | Verbose debug output, effective with debug logging |
| `--key` | `None` | API key for authentication |
| `--ssl` | `False` | Enable HTTPS |
| `--ssl-certfile` | `None` | Path to SSL certificate file, required if `--ssl` is enabled |
| `--ssl-keyfile` | `None` | Path to SSL private key file, required if `--ssl` is enabled |
| `--workspace` | `""` | Default workspace for storage isolation |
| `--api-prefix` | `""` | Reverse-proxy path prefix, also configurable with `LIGHTRAG_API_PREFIX` |
| `--workers` | `1` | Gunicorn worker count |
| `--llm-binding` | `ollama` | LLM binding type (`lollms`, `ollama`, `openai`, `openai-ollama`, `azure_openai`, `bedrock`, `gemini`) |
| `--embedding-binding` | `ollama` | Embedding binding type (`lollms`, `ollama`, `openai`, `azure_openai`, `bedrock`, `jina`, `gemini`, `voyageai`) |
| `--rerank-binding` | `null` | Rerank binding type (`null`, `cohere`, `jina`, `aliyun`) |

### Reranking Configuration

Reranking query-recalled chunks can significantly enhance retrieval quality by re-ordering documents based on an optimized relevance scoring model. LightRAG currently supports the following rerank providers:

- **Cohere / vLLM**: Offers full API integration with Cohere AI's `v2/rerank` endpoint. As vLLM provides a Cohere-compatible reranker API, all reranker models deployed via vLLM are also supported.
- **Jina AI**: Provides complete implementation compatibility with all Jina rerank models.
- **Aliyun**: Features a custom implementation designed to support Aliyun's rerank API format.

The rerank provider is configured via the `.env` file. Below is an example configuration for a rerank model deployed locally using vLLM:

```
RERANK_BINDING=cohere
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=http://localhost:8000/rerank
RERANK_BINDING_API_KEY=your_rerank_api_key_here
```

Here is an example configuration for utilizing the Reranker service provided by Aliyun:

```
RERANK_BINDING=aliyun
RERANK_MODEL=gte-rerank-v2
RERANK_BINDING_HOST=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANK_BINDING_API_KEY=your_rerank_api_key_here
```

Reranker calls have their own concurrency and timeout controls:

```bash
MAX_ASYNC_RERANK=4
RERANK_TIMEOUT=30
```

`MAX_ASYNC_RERANK` falls back to `MAX_ASYNC_LLM` when unset (`MAX_ASYNC` is still accepted as a deprecated alias). `RERANK_TIMEOUT` has an independent default because reranker requests are usually shorter than LLM generation requests. For comprehensive reranker configuration examples, including Cohere-compatible chunking options and Jina/Aliyun endpoints, refer to the `env.example` file.

### Enable Reranking

Reranking can be enabled or disabled on a per-query basis.

The `/query` and `/query/stream` API endpoints include an `enable_rerank` parameter, which is set to `true` by default, controlling whether reranking is active for the current query. To change the default value of the `enable_rerank` parameter to `false`, set the following environment variable:

```
RERANK_BY_DEFAULT=False
```

### Include Chunk Content in References

By default, the `/query` and `/query/stream` endpoints return references with only `reference_id` and `file_path`. For evaluation, debugging, or citation purposes, you can request the actual retrieved chunk content to be included in references.

The `include_chunk_content` parameter (default: `false`) controls whether the actual text content of retrieved chunks is included in the response references. This is particularly useful for:

- **RAG Evaluation**: Testing systems like RAGAS that need access to retrieved contexts
- **Debugging**: Verifying what content was actually used to generate the answer
- **Citation Display**: Showing users the exact text passages that support the response
- **Transparency**: Providing full visibility into the RAG retrieval process

**Important**: The `content` field is an **array of strings**, where each string represents a chunk from the same file. A single file may correspond to multiple chunks, so the content is returned as a list to preserve chunk boundaries.

**Example API Request:**

```json
{
  "query": "What is LightRAG?",
  "mode": "mix",
  "include_references": true,
  "include_chunk_content": true
}
```

**Example Response (with chunk content):**

```json
{
  "response": "LightRAG is a graph-based RAG system...",
  "references": [
    {
      "reference_id": "1",
      "file_path": "/documents/intro.md",
      "content": [
        "LightRAG is a retrieval-augmented generation system that combines knowledge graphs with vector similarity search...",
        "The system uses a dual-indexing approach with both vector embeddings and graph structures for enhanced retrieval..."
      ]
    },
    {
      "reference_id": "2",
      "file_path": "/documents/features.md",
      "content": [
        "The system provides multiple query modes including local, global, hybrid, and mix modes..."
      ]
    }
  ]
}
```

**Notes**:
- This parameter only works when `include_references=true`. Setting `include_chunk_content=true` without including references has no effect.
- **Breaking Change**: Prior versions returned `content` as a single concatenated string. Now it returns an array of strings to preserve individual chunk boundaries. If you need a single string, join the array elements with your preferred separator (e.g., `"\n\n".join(content)`).

### .env Examples

The examples below are reference snippets for tuning existing deployments. For a first run, follow [Progressive Setup Recipes](#progressive-setup-recipes) instead of copying the entire `env.example` file by hand.

```bash
### Server Configuration
# HOST=0.0.0.0
PORT=9621
WORKERS=2
# LIGHTRAG_API_PREFIX=/site01

### Settings for document indexing
ENTITY_EXTRACTION_USE_JSON=true
# ENTITY_TYPE_PROMPT_FILE=entity_type_prompt.yml
# MAX_EXTRACT_INPUT_TOKENS=20480
# MAX_EXTRACTION_RECORDS=100
# MAX_EXTRACTION_ENTITIES=40
SUMMARY_LANGUAGE=Chinese
MAX_PARALLEL_INSERT=3
LIGHTRAG_PARSER=*:native-teP,*:legacy-R
# CHUNK_R_SEPARATORS=["\n\n","\n","。","！","？","；","，"," ",""]
# CHUNK_P_SIZE=2000

### LLM Configuration (Use valid host. For local services installed with docker, you can use host.docker.internal)
TIMEOUT=150
MAX_ASYNC_LLM=4

LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your-api-key
KEYWORD_LLM_MODEL=gpt-4o-mini
QUERY_LLM_MODEL=gpt-4o

### Optional VLM configuration for documents using i/t/e process options
VLM_PROCESS_ENABLE=false
# VLM_LLM_MODEL=gpt-4o
# VLM_MAX_IMAGE_BYTES=5242880
# SURROUNDING_LEADING_MAX_TOKENS=2000
# SURROUNDING_TRAILING_MAX_TOKENS=2000

### Optional reranker configuration
RERANK_BINDING=null
# MAX_ASYNC_RERANK=4
# RERANK_TIMEOUT=30

### Embedding Configuration (Use valid host. For local services installed with docker, you can use host.docker.internal)
# see also env.ollama-binding-options.example for fine tuning ollama
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
# Optional asymmetric embedding for prefix-based models:
# EMBEDDING_ASYMMETRIC=true
# EMBEDDING_QUERY_PREFIX="search_query: "
# EMBEDDING_DOCUMENT_PREFIX="search_document: "
# Use NO_PREFIX for a side that should intentionally have no prefix.

### For JWT Auth
# AUTH_ACCOUNTS='admin:{bcrypt}$2b$12$replace-with-generated-hash,user1:pass456'
# TOKEN_SECRET=your-key-for-LightRAG-API-Server-xxx
# TOKEN_EXPIRE_HOURS=48

# LIGHTRAG_API_KEY=your-secure-api-key-here-123
# WHITELIST_PATHS=/api/*
# WHITELIST_PATHS=/health,/api/*
```

## Document and Chunk Processing

v1.5 introduces a staged document pipeline. Files first go through a content extraction engine, optional multimodal analysis, text chunking, and then entity/relation extraction unless the file disables knowledge graph construction.

### Quick Recipes

Keep v1.4-compatible behavior:

```bash
LIGHTRAG_PARSER=*:legacy-F
```

Recommended starting point without external parser services:

```bash
LIGHTRAG_PARSER=*:native-teP,*:legacy-R
```

This uses the built-in `native` parser for supported files, enables table/equation sidecar analysis options for those files, uses paragraph semantic chunking where possible, and falls back to legacy extraction plus recursive chunking for other files.

Full multimodal setup with the MinerU official API and a VLM:

```bash
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R
VLM_PROCESS_ENABLE=true
VLM_LLM_MODEL=gpt-4o
MINERU_API_MODE=official
MINERU_API_TOKEN=your_mineru_api_token
MINERU_OFFICIAL_ENDPOINT=https://mineru.net
MINERU_MODEL_VERSION=vlm
MINERU_IS_OCR=false
```

Use `DOCLING_ENDPOINT=http://localhost:5001` when routing files to `docling`.

### Parser Engines and Routing

`LIGHTRAG_PARSER` defines default extraction rules by file extension. Rules are matched left to right and can be separated by commas or semicolons:

```bash
LIGHTRAG_PARSER=pdf:mineru-R,docx:native-ietP,*:legacy-R
```

Supported engines:

| Engine | Use case |
| --- | --- |
| `legacy` | Original extraction behavior. Good for compatibility and simple text-like files. |
| `native` | Built-in structured parser, currently focused on `.docx` and LightRAG Document sidecars. |
| `mineru` | External MinerU parser for PDFs, Office files, and images. Requires `MINERU_API_MODE` plus `MINERU_LOCAL_ENDPOINT` or `MINERU_API_TOKEN`. |
| `docling` | External docling-serve parser for PDFs, Office files, Markdown/HTML, and images. Requires `DOCLING_ENDPOINT`. |

Filename hints override the default rule for one uploaded file:

```text
paper.[mineru-iteP].pdf
memo.[native-R!].docx
notes.[-R].md
```

The `/documents/upload` and `/documents/scan` paths honor filename hints and `LIGHTRAG_PARSER`. The `/documents/text` and `/documents/texts` endpoints insert already-provided text and currently use fixed chunking on the server path.

### Processing Options

Processing options are appended after the engine with a hyphen, or supplied alone in a filename hint with `[-OPTIONS]`.

| Option | Meaning |
| --- | --- |
| `i` | Run VLM analysis for image/drawing sidecars when present |
| `t` | Run VLM analysis for table sidecars when present |
| `e` | Run VLM analysis for equation sidecars when present |
| `!` | Skip entity/relation extraction and graph writes; chunk vectors are still stored |
| `F` | Fixed token chunking, the legacy chunking method |
| `R` | Recursive character chunking with configurable separator cascade |
| `V` | Semantic vector chunking; oversize chunks are re-split by `R` |
| `P` | Paragraph semantic chunking for structured LightRAG Document content; falls back to `R` when structured content is unavailable |

At most one of `F`, `R`, `V`, and `P` should be selected for a file. Chunker parameters are configured with `CHUNK_SIZE`, `CHUNK_OVERLAP_SIZE`, and strategy-specific variables such as `CHUNK_R_SEPARATORS`, `CHUNK_V_BREAKPOINT_THRESHOLD_TYPE`, `CHUNK_P_SIZE`, and `CHUNK_P_OVERLAP_SIZE`. These values are read at server startup and stored as a per-document `chunk_options` snapshot when a document is enqueued.

For the full routing syntax, supported extensions, parser cache behavior, chunker configuration, concurrency rules, and Python SDK differences, see [File Processing Pipeline Specification](./FileProcessingPipeline.md). For the `P` strategy details, see [Paragraph Semantic Chunking](./ParagraphSemanticChunking.md). To debug parser output before indexing a file, see [Parser Debug CLI](./ParserDebugCLI.md).

### Pipeline Concurrency

`MAX_PARALLEL_INSERT` controls how many files are processed in parallel. `MAX_ASYNC_LLM` (deprecated alias: `MAX_ASYNC`) controls concurrent LLM calls, including extraction, merging, query keyword generation, and final answer generation. Optional staged-pipeline variables such as `MAX_PARALLEL_PARSE_NATIVE`, `MAX_PARALLEL_PARSE_MINERU`, `MAX_PARALLEL_PARSE_DOCLING`, and `MAX_PARALLEL_ANALYZE` can be used for parser-heavy deployments.

Uploads and text inserts can be accepted while the processing loop is busy; the running loop is nudged to pick up the new pending work. Destructive jobs such as document clear/delete and the classification phase of `/documents/scan` still reject concurrent enqueues to protect storage consistency. Failed files can be reprocessed from the WebUI or by triggering `/documents/scan`.

## API Endpoints

All supported backends (`lollms`, `ollama`, `openai` / OpenAI-compatible, `azure_openai`, `bedrock`, and `gemini`) expose the same LightRAG REST API surface. When the API Server is running, visit:

- Swagger UI: http://localhost:9621/docs
- ReDoc: http://localhost:9621/redoc

You can test the API endpoints using the provided curl commands or through the Swagger UI interface. Make sure to:

1. Start the appropriate backend service or confirm the hosted provider credentials
2. Start the RAG server
3. Upload some documents using the document management endpoints
4. Query the system using the query endpoints
5. Trigger document scan if new files are put into the inputs directory

The `/health` endpoint reports operational state and selected configuration, including role LLM configuration, LLM/embedding/rerank queue status, workspace/storage workspace mapping, VLM enablement, rerank enablement, and pipeline busy/scanning/destructive status. It always returns HTTP 200 so it stays usable as a liveness probe, but the configuration and operational diagnostics are returned **only to authenticated callers** (valid JWT or `X-API-Key`). Unauthenticated callers receive only liveness signals (`status`, `auth_mode`, `core_version`, `api_version`, `pipeline_busy`/`pipeline_active`, and the WebUI title/availability fields — all of which are also exposed by the unauthenticated `/auth-status` endpoint or are plain booleans). Provide credentials to retrieve the full payload, e.g. `curl -H "X-API-Key: <key>" http://localhost:9621/health`.

## Asynchronous Document Indexing with Progress Tracking

LightRAG implements asynchronous document indexing to enable frontend monitoring and querying of document processing progress. Upon uploading files or inserting text through designated endpoints, a unique Track ID is returned to facilitate real-time progress monitoring.

**API Endpoints Supporting Track ID Generation:**

* `/documents/upload`
* `/documents/text`
* `/documents/texts`

**Document Processing Status Query Endpoint:**
* `/documents/track_status/{track_id}`

This endpoint provides comprehensive status information including:
* Document processing status (pending/processing/processed/failed)
* Content summary and metadata
* Error messages if processing failed
* Timestamps for creation and updates
