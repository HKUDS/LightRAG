#!/bin/bash

NAMESPACE=rag

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY environment variable is not set"
  read -p "Enter your OpenAI API key: " OPENAI_API_KEY
  if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY must be provided"
    exit 1
  fi
  export OPENAI_API_KEY=$OPENAI_API_KEY
fi

if [ -z "$OPENAI_API_BASE" ]; then
  echo "OPENAI_API_BASE environment variable is not set, will use default value"
  read -p "Enter OpenAI API base URL (press Enter to skip if not needed): " OPENAI_API_BASE
  export OPENAI_API_BASE=$OPENAI_API_BASE
fi

required_env_vars=("OPENAI_API_BASE" "OPENAI_API_KEY")

for var in "${required_env_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: $var environment variable is not set"
    exit 1
  fi
done

echo "Environment variables check passed"

if ! kubectl get namespace rag &> /dev/null; then
  echo "creating namespace 'rag'..."
  kubectl create namespace rag
fi

helm upgrade --install lightrag-dev $SCRIPT_DIR \
  --namespace rag \
  --set-string env.LIGHTRAG_KV_STORAGE=JsonKVStorage \
  --set-string env.LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage \
  --set-string env.LIGHTRAG_GRAPH_STORAGE=NetworkXStorage \
  --set-string env.LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage \
  --set-string env.LLM_BINDING=openai \
  --set-string env.LLM_MODEL=gpt-4o-mini \
  --set-string env.LLM_BINDING_HOST=$OPENAI_API_BASE \
  --set-string env.LLM_BINDING_API_KEY=$OPENAI_API_KEY \
  --set-string env.EMBEDDING_BINDING=openai \
  --set-string env.EMBEDDING_MODEL=text-embedding-ada-002 \
  --set-string env.EMBEDDING_DIM=1536 \
  --set-string env.EMBEDDING_BINDING_API_KEY=$OPENAI_API_KEY

# Wait for LightRAG pod to be ready
echo "Waiting for LightRAG pod to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=lightrag-dev --timeout=60s -n rag
