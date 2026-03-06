#!/bin/bash

# Complete LightRAG Installation Script with Venv
# This script sets up everything in the project's .venv directory
# Usage: ./install.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/venv/${PROJECT_NAME:-LightRAG}"

echo "=================================================="
echo "🚀 LightRAG Complete Installation Script"
echo "=================================================="
echo ""

# Step 1: Create virtual environment
echo "📍 Step 1/6: Setting up virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    echo "✅ Virtual environment created at: ${VENV_DIR}"
else
    source "${VENV_DIR}/bin/activate"
    echo "✅ Virtual environment already exists at: ${VENV_DIR}"
fi

# Step 2: Upgrade pip
echo ""
echo "📍 Step 2/6: Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Step 3: Install LightRAG core dependencies
echo ""
echo "📍 Step 3/6: Installing LightRAG core dependencies..."
pip install 'google-api-core>=2.0.0,<3.0.0' 'google-genai>=1.0.0,<2.0.0' 'openai>=2.0.0,<3.0.0'

# Step 4: Install API and web framework dependencies
echo ""
echo "📍 Step 4/6: Installing API and web framework dependencies..."
pip install aiofiles fastapi httpx httpcore jiter uvicorn bcrypt Python-Jose[cryptography] python-multipart psutil gunicorn

# Step 5: Install document processing dependencies (PDF, Word, Excel, PowerPoint)
echo ""
echo "📍 Step 5/6: Installing document processing dependencies..."
pip install openpyxl pycryptodome 'python-docx>=0.8.11,<2.0.0' 'python-pptx>=0.6.21,<2.0.0' 'pypdf>=6.1.0' --no-cache-dir

# Step 6: Setup activation and start scripts
echo ""
echo "📍 Step 6/6: Creating utility scripts..."
mkdir -p "${PROJECT_ROOT}/logs"

# Create activation script
cat > "${PROJECT_ROOT}/activate_venv.sh" << 'EOF'
#!/bin/bash
# LightRAG Virtual Environment Activation Script
LIGHTRAG_VENV="$HOME/venv/${PROJECT_NAME:-LightRAG}"
if [ ! -d "$LIGHTRAG_VENV" ]; then
    echo "Error: Venv not found at $LIGHTRAG_VENV"
    exit 1
fi
source "$LIGHTRAG_VENV/bin/activate"
echo "✅ Activated LightRAG virtual environment"
EOF
chmod +x "${PROJECT_ROOT}/activate_venv.sh"

# Create embedding service start script
cat > "${PROJECT_ROOT}/start_embedding_vllm.sh" << 'EOF'
#!/bin/bash
# Start vLLM Embedding Service
LIGHTRAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LIGHTRAG_ROOT/activate_venv.sh"
PROJECT_NAME="$(basename $(dirname "${BASH_SOURCE[0]}"))"
mkdir -p "$LIGHTRAG_ROOT/logs"
echo "🚀 Starting vLLM Embedding Service (BAAI/bge-m3)..."
vllm serve "BAAI/bge-m3" \
    --host "0.0.0.0" \
    --port 8000 \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --disable-log-requests \
    --disable-log-stats &> "$LIGHTRAG_ROOT/logs/embedding-vllm.log" &
echo "✅ Embedding service started on port 8000"
echo "📊 Monitor logs: tail -f $LIGHTRAG_ROOT/logs/embedding-vllm.log"
EOF
chmod +x "${PROJECT_ROOT}/start_embedding_vllm.sh"

# Create reranking service start script
cat > "${PROJECT_ROOT}/start_rerank_vllm.sh" << 'EOF'
#!/bin/bash
# Start vLLM Reranking Service
LIGHTRAG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$LIGHTRAG_ROOT/activate_venv.sh"
PROJECT_NAME="$(basename $(dirname "${BASH_SOURCE[0]}"))"
mkdir -p "$LIGHTRAG_ROOT/logs"
echo "🚀 Starting vLLM Reranking Service (BAAI/bge-reranker-v2-m3)..."
vllm serve "BAAI/bge-reranker-v2-m3" \
    --host "0.0.0.0" \
    --port 8001 \
    --dtype float16 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --disable-log-requests \
    --disable-log-stats &> "$LIGHTRAG_ROOT/logs/rerank-vllm.log" &
echo "✅ Reranking service started on port 8001"
echo "📊 Monitor logs: tail -f $LIGHTRAG_ROOT/logs/rerank-vllm.log"
EOF
chmod +x "${PROJECT_ROOT}/start_rerank_vllm.sh"

echo ""
echo "✅ Installation complete!"
echo ""
echo "=================================================="
echo "✅ Installation Summary"
echo "=================================================="
echo ""
echo "📁 Virtual Environment: ${VENV_DIR}"
echo ""
echo "🛠️  Next Steps:"
echo ""
echo "1️⃣  Activate venv:"
echo "   source ${PROJECT_ROOT}/activate_venv.sh"
echo ""
echo "2️⃣  Start embedding service:"
echo "   ${PROJECT_ROOT}/start_embedding_vllm.sh"
echo ""
echo "3️⃣  Start reranking service:"
echo "   ${PROJECT_ROOT}/start_rerank_vllm.sh"
echo ""
echo "4️⃣  Restart LightRAG with:"
echo "   ${PROJECT_ROOT}/run.sh"
echo ""
echo "📍 Services will be available at:"
echo "   - Embedding: http://localhost:8000"
echo "   - Reranking: http://localhost:8001"
echo "   - LightRAG API: http://localhost:9621"
echo "   - LightRAG Web UI: http://localhost:9622"
echo ""
echo "📊 Monitor logs:"
echo "   tail -f ${PROJECT_ROOT}/logs/embedding-vllm.log"
echo "   tail -f ${PROJECT_ROOT}/logs/rerank-vllm.log"
echo ""
echo "🛑 Stop services:"
echo "   killall vllm"
echo ""
echo "=================================================="