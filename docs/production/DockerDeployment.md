# LightRAG

A lightweight Knowledge Graph Retrieval-Augmented Generation system with multiple LLM backend support.

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Git
- Docker (optional for Docker deployment)

### Native Installation

1. Clone the repository:
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

2. Configure your environment:
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

3. Create and activate virtual environment:
```bash
# Linux/MacOS
python -m venv venv
source venv/bin/activate
```
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate
```

4. Install dependencies:
```bash
# Both platforms
pip install -r requirements.txt
```

## 🐳 Docker Deployment

Docker instructions work the same on all platforms with Docker Desktop installed.

1. Build and start the container:
```bash
docker-compose up -d
```

### Configuration Options

LightRAG can be configured using environment variables in the `.env` file:

#### Server Configuration
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 9621)

#### LLM Configuration
- `LLM_BINDING`: LLM backend to use (lollms/ollama/openai)
- `LLM_BINDING_HOST`: LLM server host URL
- `LLM_MODEL`: Model name to use

#### Embedding Configuration
- `EMBEDDING_BINDING`: Embedding backend (lollms/ollama/openai)
- `EMBEDDING_BINDING_HOST`: Embedding server host URL
- `EMBEDDING_MODEL`: Embedding model name

#### RAG Configuration
- `MAX_ASYNC`: Maximum async operations
- `MAX_TOKENS`: Maximum token size
- `EMBEDDING_DIM`: Embedding dimensions

#### Security
- `LIGHTRAG_API_KEY`: API key for authentication

### Data Storage Paths

The system uses the following paths for data storage:
```
data/
├── rag_storage/    # RAG data persistence
└── inputs/         # Input documents
```

### Example Deployments

1. Using with Ollama:
```env
LLM_BINDING=ollama
LLM_BINDING_HOST=http://host.docker.internal:11434
LLM_MODEL=mistral
EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://host.docker.internal:11434
EMBEDDING_MODEL=bge-m3
```

you can't just use localhost from docker, that's why you need to use host.docker.internal which is defined in the docker compose file and should allow you to access the localhost services.

2. Using with OpenAI:
```env
LLM_BINDING=openai
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_API_KEY=your-api-key
```

### API Usage

Once deployed, you can interact with the API at `http://localhost:9621`

Example query using PowerShell:
```powershell
$headers = @{
    "X-API-Key" = "your-api-key"
    "Content-Type" = "application/json"
}
$body = @{
    query = "your question here"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:9621/query" -Method Post -Headers $headers -Body $body
```

Example query using curl:
```bash
curl -X POST "http://localhost:9621/query" \
     -H "X-API-Key: your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"query": "your question here"}'
```

## 🔒 Security

Remember to:
1. Set a strong API key in production
2. Use SSL in production environments
3. Configure proper network security

## 📦 Updates

To update the Docker container:
```bash
docker-compose pull
docker-compose up -d --build
```

To update native installation:
```bash
# Linux/MacOS
git pull
source venv/bin/activate
pip install -r requirements.txt
```
```powershell
# Windows PowerShell
git pull
.\venv\Scripts\Activate
pip install -r requirements.txt
```
