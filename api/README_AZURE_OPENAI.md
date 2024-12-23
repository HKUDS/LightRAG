
# LightRAG API Server

A powerful FastAPI-based server for managing and querying documents using LightRAG (Light Retrieval-Augmented Generation). This server provides a REST API interface for document management and intelligent querying using OpenAI's language models.

## Features

- 🔍 Multiple search modes (naive, local, global, hybrid)
- 📡 Streaming and non-streaming responses
- 📝 Document management (insert, batch upload, clear)
- ⚙️ Highly configurable model parameters
- 📚 Support for text and file uploads
- 🔧 RESTful API with automatic documentation
- 🚀 Built with FastAPI for high performance

## Prerequisites

- Python 3.8+
- Azure OpenAI API key
- Azure OpenAI Deployments (gpt-4o, text-embedding-3-large)
- Required Python packages:
  - fastapi
  - uvicorn
  - lightrag
  - pydantic
  - openai
  - nest-asyncio

## Installation
If you are using Windows, you will need to download and install visual c++ build tools from [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
Make sure you install the VS 2022 C++ x64/x86 Build tools from individual components tab.

1. Clone the repository:
```bash
git clone https://github.com/ParisNeo/LightRAG.git
cd api
```

2. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
#venv\Scripts\activate for Windows
pip install -r requirements.txt
```

3. Set up environment variables:
   use the `.env` file to set the environment variables (you can copy the `.env.aoi.example` file and rename it to `.env`),
   or set them manually:
```bash
export AZURE_OPENAI_API_VERSION='2024-08-01-preview'
export AZURE_OPENAI_DEPLOYMENT='gpt-4o'
export AZURE_OPENAI_API_KEY='myapikey'
export AZURE_OPENAI_ENDPOINT='https://myendpoint.openai.azure.com'
export AZURE_EMBEDDING_DEPLOYMENT='text-embedding-3-large'
export AZURE_EMBEDDING_API_VERSION='2023-05-15'
```

## Configuration

The server can be configured using command-line arguments:

```bash
python azure_openai_lightrag_server.py --help
```

Available options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| --host | 0.0.0.0 | Server host |
| --port | 9621 | Server port |
| --model | gpt-4 | OpenAI model name |
| --embedding-model | text-embedding-3-large | OpenAI embedding model |
| --working-dir | ./rag_storage | Working directory for RAG |
| --max-tokens | 32768 | Maximum token size |
| --max-embed-tokens | 8192 | Maximum embedding token size |
| --input-dir | ./inputs | Input directory for documents |
| --enable-cache | True | Enable response cache |
| --log-level | INFO | Logging level |

## Quick Start

1. Basic usage with default settings:
```bash
python azure_openai_lightrag_server.py
```

2. Custom configuration:
```bash
python azure_openai_lightrag_server.py --model gpt-4o --port 8080 --working-dir ./custom_rag
```

## API Endpoints

### Query Endpoints

#### POST /query
Query the RAG system with options for different search modes.

```bash
curl -X POST "http://localhost:9621/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Your question here", "mode": "hybrid"}'
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

### Running in Development Mode

```bash
uvicorn azure_openai_lightrag_server:app --reload --port 9621
```

### API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:9621/docs
- ReDoc: http://localhost:9621/redoc

## Deployment
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [LightRAG](https://github.com/HKUDS/LightRAG) for document processing
- Powered by [OpenAI](https://openai.com/) for language model inference
