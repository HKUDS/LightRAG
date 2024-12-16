# LightRAG API Server

A powerful FastAPI-based server for managing and querying documents using LightRAG (Light Retrieval-Augmented Generation). This server provides a REST API interface for document management and intelligent querying using various LLM models through Ollama.

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
- Ollama server running locally or remotely
- Required Python packages:
  - fastapi
  - uvicorn
  - lightrag
  - pydantic

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lightrag-server.git
cd lightrag-server
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running and accessible.

## Configuration

The server can be configured using command-line arguments:

```bash
python rag_server.py --help
```

Available options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| --host | 0.0.0.0 | Server host |
| --port | 8000 | Server port |
| --model | gemma2:2b | LLM model name |
| --embedding-model | nomic-embed-text | Embedding model name |
| --ollama-host | http://localhost:11434 | Ollama host URL |
| --working-dir | ./dickens | Working directory for RAG |
| --max-async | 4 | Maximum async operations |
| --max-tokens | 32768 | Maximum token size |
| --embedding-dim | 768 | Embedding dimensions |
| --max-embed-tokens | 8192 | Maximum embedding token size |
| --input-file | ./book.txt | Initial input file |
| --log-level | INFO | Logging level |

## Quick Start

1. Basic usage with default settings:
```bash
python rag_server.py
```

2. Custom configuration:
```bash
python rag_server.py --model llama2:13b --port 8080 --working-dir ./custom_rag
```

3. Using the launch script:
```bash
chmod +x launch_rag_server.sh
./launch_rag_server.sh
```

## API Endpoints

### Query Endpoints

#### POST /query
Query the RAG system with options for different search modes.

```bash
curl -X POST "http://localhost:8000/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Your question here", "mode": "hybrid"}'
```

#### POST /query/stream
Stream responses from the RAG system.

```bash
curl -X POST "http://localhost:8000/query/stream" \
    -H "Content-Type: application/json" \
    -d '{"query": "Your question here", "mode": "hybrid"}'
```

### Document Management Endpoints

#### POST /documents/text
Insert text directly into the RAG system.

```bash
curl -X POST "http://localhost:8000/documents/text" \
    -H "Content-Type: application/json" \
    -d '{"text": "Your text content here", "description": "Optional description"}'
```

#### POST /documents/file
Upload a single file to the RAG system.

```bash
curl -X POST "http://localhost:8000/documents/file" \
    -F "file=@/path/to/your/document.txt" \
    -F "description=Optional description"
```

#### POST /documents/batch
Upload multiple files at once.

```bash
curl -X POST "http://localhost:8000/documents/batch" \
    -F "files=@/path/to/doc1.txt" \
    -F "files=@/path/to/doc2.txt"
```

#### DELETE /documents
Clear all documents from the RAG system.

```bash
curl -X DELETE "http://localhost:8000/documents"
```

### Utility Endpoints

#### GET /health
Check server health and configuration.

```bash
curl "http://localhost:8000/health"
```

## Development

### Running in Development Mode

```bash
uvicorn rag_server:app --reload --port 8000
```

### API Documentation

When the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Uses [LightRAG](https://github.com/HKUDS/LightRAG) for document processing
- Powered by [Ollama](https://ollama.ai/) for LLM inference

## Support
