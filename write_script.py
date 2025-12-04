
content = r"""#!/bin/bash

# ==============================================================================
# LightRAG E2E Test Runner
# ==============================================================================
# This script runs the End-to-End test suite for LightRAG.
# It supports multiple storage backends and configurable LLM models.
#
# Usage: ./e2e/run_isolation_test.sh [options]
#
# Options:
#   -b, --backend <type>    Storage backend to test (file, postgres, all). Default: file
#   -m, --model <name>      Ollama model to use. Default: gpt-oss:20b
#   -d, --dim <number>      Embedding dimension. Default: 1024
#   -h, --help              Show this help message
# ==============================================================================

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Defaults
BACKEND="file"
LLM_MODEL="gpt-oss:20b"
EMBEDDING_MODEL="bge-m3:latest"
EMBEDDING_DIM="1024"
SERVER_PORT=9621

# Parse Arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--backend) BACKEND="$2"; shift ;;
        -m|--model) LLM_MODEL="$2"; shift ;;
        -d|--dim) EMBEDDING_DIM="$2"; shift ;;
        -h|--help) 
            grep "^# " "$0" | cut -c 3-
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${GREEN}Starting LightRAG E2E Test Suite...${NC}"
echo "Backend: $BACKEND"
echo "Model: $LLM_MODEL"

# Function to cleanup server
cleanup_server() {
    if lsof -i :$SERVER_PORT > /dev/null; then
        echo "Stopping existing server on port $SERVER_PORT..."
        lsof -i :$SERVER_PORT | grep Python | awk '{print $2}' | xargs kill -9
        sleep 2
        echo "Server stopped."
    fi
}

# Function to configure environment
configure_env() {
    local backend_type=$1
    
    # Common Env Vars
    export LLM_BINDING="ollama"
    export LLM_MODEL="$LLM_MODEL"
    export EMBEDDING_BINDING="ollama"
    export EMBEDDING_MODEL="$EMBEDDING_MODEL"
    export EMBEDDING_DIM="$EMBEDDING_DIM"
    export LIGHTRAG_API_KEY="admin123"
    export AUTH_ACCOUNTS="admin:admin123"
    
    echo -e "\n${BLUE}Configuring for Backend: $backend_type${NC}"
    
    if [ "$backend_type" == "file" ]; then
        export LIGHTRAG_KV_STORAGE="JsonKVStorage"
        export LIGHTRAG_DOC_STATUS_STORAGE="JsonDocStatusStorage"
        export LIGHTRAG_GRAPH_STORAGE="NetworkXStorage"
        export LIGHTRAG_VECTOR_STORAGE="NanoVectorDBStorage"
        
        # Clean up file storage
        echo "Cleaning up local storage (rag_storage)..."
        rm -rf rag_storage
        
    elif [ "$backend_type" == "postgres" ]; then
        export LIGHTRAG_KV_STORAGE="PGKVStorage"
        export LIGHTRAG_DOC_STATUS_STORAGE="PGDocStatusStorage"
        export LIGHTRAG_GRAPH_STORAGE="PGGraphStorage"
        export LIGHTRAG_VECTOR_STORAGE="PGVectorStorage"
        
        # Ensure Postgres vars are set (defaults if not in env)
        export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
        export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
        export POSTGRES_USER="${POSTGRES_USER:-lightrag}"
        export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-lightrag_secure_password}"
        export POSTGRES_DATABASE="${POSTGRES_DATABASE:-lightrag_multitenant}"
        
        echo "⚠️  Ensure Postgres is running at $POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DATABASE"
        
    else
        echo -e "${RED}Unknown backend: $backend_type${NC}"
        exit 1
    fi
    
    echo "Environment Configured:"
    echo "  STORAGE: $backend_type"
    echo "  LLM: $LLM_MODEL"
}

# Function to wait for server health
wait_for_server() {
    echo "Waiting for server to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:$SERVER_PORT/health > /dev/null; then
            echo -e "${GREEN}Server is up!${NC}"
            return 0
        fi
        # Fallback check if /health doesn't exist yet, check root or docs
        if curl -s http://localhost:$SERVER_PORT/docs > /dev/null; then
             echo -e "${GREEN}Server is up!${NC}"
             return 0
        fi
        sleep 1
    done
    echo -e "${RED}Server failed to start within 30 seconds.${NC}"
    cat server.log
    return 1
}

# Function to run tests
run_test_suite() {
    local backend_name=$1
    
    cleanup_server
    configure_env "$backend_name"
    
    echo "Starting server..."
    nohup python -m lightrag.api.lightrag_server --port $SERVER_PORT > server.log 2>&1 &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    
    if ! wait_for_server; then
        kill $SERVER_PID 2>/dev/null
        return 1
    fi
    
    FAILURES=0
    
    # List of tests to run
    TESTS=(
        "e2e/test_multitenant_isolation.py"
        "e2e/test_deletion.py"
        "e2e/test_mixed_operations.py"
    )
    
    for test_script in "${TESTS[@]}"; do
        echo -e "\n${BLUE}==================================================${NC}"
        echo -e "${BLUE}Running $test_script [$backend_name]...${NC}"
        echo -e "${BLUE}==================================================${NC}"
        
        python "$test_script"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ $test_script Passed!${NC}"
        else
            echo -e "${RED}❌ $test_script Failed!${NC}"
            ((FAILURES++))
        fi
    done
    
    echo "Cleaning up server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    
    if [ $FAILURES -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed for $backend_name!${NC}"
        return 0
    else
        echo -e "${RED}💀 $FAILURES test(s) failed for $backend_name.${NC}"
        return 1
    fi
}

# Main Execution Logic
if [ "$BACKEND" == "all" ]; then
    echo "Running tests for ALL backends..."
    
    # Run File
    run_test_suite "file"
    FILE_EXIT=$?
    
    # Run Postgres
    run_test_suite "postgres"
    PG_EXIT=$?
    
    if [ $FILE_EXIT -eq 0 ] && [ $PG_EXIT -eq 0 ]; then
        echo -e "\n${GREEN}🏆 ALL BACKENDS PASSED!${NC}"
        exit 0
    else
        echo -e "\n${RED}💥 SOME BACKENDS FAILED${NC}"
        exit 1
    fi
else
    run_test_suite "$BACKEND"
    exit $?
fi
"""

with open("e2e/run_isolation_test.sh", "w") as f:
    f.write(content)
