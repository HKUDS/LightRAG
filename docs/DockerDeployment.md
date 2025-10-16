# LightRAG Docker Deployment

A lightweight Knowledge Graph Retrieval-Augmented Generation system with multiple LLM backend support.

## 🚀 Preparation

### Clone the repository:

```bash
# Linux/MacOS
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
```
```powershell
# Windows PowerShell
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
```

### Configure your environment:

```bash
# Linux/MacOS
cp .env.example .env
# Edit .env with your preferred configuration
```
```powershell
# Windows PowerShell
Copy-Item .env.example .env
# Edit .env with your preferred configuration
```

LightRAG can be configured using environment variables in the `.env` file:

**Server Configuration**

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 9621)

**LLM Configuration**

- `LLM_BINDING`: LLM backend to use (lollms/ollama/openai)
- `LLM_BINDING_HOST`: LLM server host URL
- `LLM_MODEL`: Model name to use

**Embedding Configuration**

- `EMBEDDING_BINDING`: Embedding backend (lollms/ollama/openai)
- `EMBEDDING_BINDING_HOST`: Embedding server host URL
- `EMBEDDING_MODEL`: Embedding model name

**RAG Configuration**

- `MAX_ASYNC`: Maximum async operations
- `MAX_TOKENS`: Maximum token size
- `EMBEDDING_DIM`: Embedding dimensions

## 🐳 Docker Deployment

Docker instructions work the same on all platforms with Docker Desktop installed.

### Start LightRAG  server:

```bash
docker-compose up -d
```

LightRAG Server uses the following paths for data storage:

```
data/
├── rag_storage/    # RAG data persistence
└── inputs/         # Input documents
```

### Updates

To update the Docker container:
```bash
docker-compose pull
docker-compose down
docker-compose up
```

### Offline deployment

Software packages requiring `transformers`, `torch`, or `cuda` will is not preinstalled in the dokcer images. Consequently, document extraction tools such as Docling, as well as local LLM models like Hugging Face and LMDeploy, can not be used in an off line enviroment. These high-compute-resource-demanding services should not be integrated into LightRAG. Docling will be decoupled and deployed as a standalone service.

## 📦 Build Docker Images

### For local development and testing

```bash
# Build and run with docker-compose
docker compose up --build
```

### For production release

 **multi-architecture build and push**:

```bash
# Use the provided build script
./docker-build-push.sh
```

**The build script will**:

- Check Docker registry login status
- Create/use buildx builder automatically
- Build for both AMD64 and ARM64 architectures
- Push to GitHub Container Registry (ghcr.io)
- Verify the multi-architecture manifest

**Prerequisites**:

Before building multi-architecture images, ensure you have:

- Docker 20.10+ with Buildx support
- Sufficient disk space (20GB+ recommended for offline image)
- Registry access credentials (if pushing images)

