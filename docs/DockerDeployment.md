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

### Build Optimization

The Dockerfile uses BuildKit cache mounts to significantly improve build performance:

- **Automatic cache management**: BuildKit is automatically enabled via `# syntax=docker/dockerfile:1` directive
- **Faster rebuilds**: Only downloads changed dependencies when `uv.lock` or `bun.lock` files are modified
- **Efficient package caching**: UV and Bun package downloads are cached across builds
- **No manual configuration needed**: Works out of the box in Docker Compose and GitHub Actions

### Start LightRAG  server:

```bash
docker compose up -d
```

If you used the interactive setup, start the generated stack with:

```bash
docker compose -f docker-compose.development.yml up -d
```

LightRAG Server uses the following paths for data storage:

```
data/
├── rag_storage/    # RAG data persistence
└── inputs/         # Input documents
```

### Optional: local vLLM reranker

To enable local reranking with vLLM, run a vLLM container exposing the Cohere-compatible rerank endpoint and point LightRAG to it.
You can select `vllm` in the interactive setup to add the `vllm-rerank` service automatically.
vLLM provides a `v1/rerank` endpoint that works with the `cohere` binding.

Example `docker-compose.override.yml`:

```yaml
services:
  vllm-rerank:
    image: vllm/vllm-openai:latest
    command: >
      --model BAAI/bge-reranker-v2-m3
      --port 8000
      --dtype float16
    ports:
      - "8000:8000"
    volumes:
      - ./data/hf-cache:/root/.cache/huggingface
    runtime: nvidia
```

Add the rerank config to `.env`:

```bash
RERANK_BINDING=cohere
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=http://vllm-rerank:8000/v1/rerank
RERANK_BINDING_API_KEY=local-key
VLLM_RERANK_DEVICE=cpu
VLLM_RERANK_DTYPE=float32
```

If you run vLLM on the host instead of Docker, use:

```bash
RERANK_BINDING_HOST=http://host.docker.internal:8000/v1/rerank
```

For GPU, set:

```bash
VLLM_RERANK_DEVICE=cuda
VLLM_RERANK_DTYPE=float16
```

Ensure the NVIDIA Container Toolkit is installed and the host has CUDA drivers available.
The default vLLM image is GPU-only; CPU setups require a CPU-compatible image tag.

### Updates

To update the Docker container:
```bash
docker compose pull
docker compose down
docker compose up
```

### Offline deployment

Software packages requiring `transformers`, `torch`, or `cuda` will is not preinstalled in the dokcer images. Consequently, document extraction tools such as Docling, as well as local LLM models like Hugging Face and LMDeploy, can not be used in an off line enviroment. These high-compute-resource-demanding services should not be integrated into LightRAG. Docling will be decoupled and deployed as a standalone service.

## 📦 Build Docker Images

### For local development and testing

```bash
# Build and run with Docker Compose (BuildKit automatically enabled)
docker compose up --build

# Or explicitly enable BuildKit if needed
DOCKER_BUILDKIT=1 docker compose up --build
```

**Note**: BuildKit is automatically enabled by the `# syntax=docker/dockerfile:1` directive in the Dockerfile, ensuring optimal caching performance.

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
