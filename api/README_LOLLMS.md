# LightRAG API Server

A powerful FastAPI-based server for managing and querying documents using LightRAG (Light Retrieval-Augmented Generation). This server provides a REST API interface for document management and intelligent querying using various LLM models through LoLLMS.

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
- LoLLMS server running locally or remotely
- Required Python packages:
  - fastapi
  - uvicorn
  - lightrag
  - pydantic

## Installation
If you are using windows, you will need to donwload and install visual c++ build tools from [https://visualstudio.microsoft.com/visual-cpp-build-tools/ ](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
Make sure you install the VS 2022 C++ x64/x86 Build tools like from indivisual componants tab:
![image](https://github.com/user-attachments/assets/3723e15b-0a2c-42ed-aebf-e595a9f9c946)

This is mandatory for builmding some modules.

1. Clone the repository:
```bash
git clone https://github.com/ParisNeo/LightRAG.git
cd api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure LoLLMS is running and accessible.

## Configuration

The server can be configured using command-line arguments:

```bash
python ollama_lightollama_lightrag_server.py --help
```

Available options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| --host | 0.0.0.0 | Server host |
| --port | 9621 | Server port |
| --model | mistral-nemo:latest | LLM model name |
| --embedding-model | bge-m3:latest | Embedding model name |
| --lollms-host | http://localhost:11434 | LoLLMS host URL |
| --working-dir | ./rag_storage | Working directory for RAG |
| --max-async | 4 | Maximum async operations |
| --max-tokens | 32768 | Maximum token size |
| --embedding-dim | 1024 | Embedding dimensions |
| --max-embed-tokens | 8192 | Maximum embedding token size |
| --input-file | ./book.txt | Initial input file |
| --log-level | INFO | Logging level |

## Quick Start

1. Basic usage with default settings:
```bash
python ollama_lightrag_server.py
```

2. Custom configuration:
```bash
python ollama_lightrag_server.py --model llama2:13b --port 8080 --working-dir ./custom_rag
```

Make sure the models are installed in your lollms instance
```bash
python ollama_lightrag_server.py --model mistral-nemo:latest --embedding-model bge-m3  --embedding-dim 1024
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
uvicorn ollama_lightrag_server:app --reload --port 9621
```

### API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:9621/docs
- ReDoc: http://localhost:9621/redoc


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [LightRAG](https://github.com/HKUDS/LightRAG) for document processing
- Powered by [LoLLMS](https://lollms.ai/) for LLM inference
