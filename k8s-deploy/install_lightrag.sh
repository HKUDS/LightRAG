#!/bin/bash

NAMESPACE=rag

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

check_dependencies(){
  echo "Checking dependencies..."
  command -v kubectl >/dev/null 2>&1 || { echo "Error: kubectl command not found"; exit 1; }
  command -v helm >/dev/null 2>&1 || { echo "Error: helm command not found"; exit 1; }

  # Check if Kubernetes is available
  echo "Checking if Kubernetes is available..."
  kubectl cluster-info &>/dev/null
  if [ $? -ne 0 ]; then
      echo "Error: Kubernetes cluster is not accessible. Please ensure you have proper access to a Kubernetes cluster."
      exit 1
  fi
  echo "Kubernetes cluster is accessible."
}

check_dependencies

# Check and set environment variables
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

# Check if databases are already installed, install them if not
echo "Checking database installation status..."
if ! kubectl get clusters -n rag pg-cluster &> /dev/null || ! kubectl get clusters -n rag neo4j-cluster &> /dev/null; then
  echo "Databases not installed or incompletely installed, will install required databases first..."

  # Install KubeBlocks (if not already installed)
  echo "Preparing to install KubeBlocks and required components..."
  bash "$SCRIPT_DIR/databases/01-prepare.sh"

  # Install database clusters
  echo "Installing database clusters..."
  bash "$SCRIPT_DIR/databases/02-install-database.sh"

  # Wait for databases to be ready
  echo "Waiting for databases to be ready..."
  TIMEOUT=600  # Set timeout to 10 minutes
  START_TIME=$(date +%s)

  while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    if [ $ELAPSED -gt $TIMEOUT ]; then
      echo "Timeout waiting for databases to be ready. Please check database status manually and try again"
      exit 1
    fi

    PG_STATUS=$(kubectl get clusters -n rag pg-cluster -o jsonpath='{.status.phase}' 2>/dev/null)
    NEO4J_STATUS=$(kubectl get clusters -n rag neo4j-cluster -o jsonpath='{.status.phase}' 2>/dev/null)

    if [ "$PG_STATUS" == "Running" ] && [ "$NEO4J_STATUS" == "Running" ]; then
      echo "Databases are ready, continuing with LightRAG deployment..."
      break
    fi

    echo "Databases preparing... PostgreSQL: $PG_STATUS, Neo4J: $NEO4J_STATUS"
    sleep 10
  done
else
  echo "Databases already installed, proceeding directly to LightRAG deployment..."
fi

# Get database passwords from Kubernetes secrets
echo "Retrieving database credentials from Kubernetes secrets..."
POSTGRES_PASSWORD=$(kubectl get secrets -n rag pg-cluster-postgresql-account-postgres -o jsonpath='{.data.password}' | base64 -d)
if [ -z "$POSTGRES_PASSWORD" ]; then
  echo "Error: Could not retrieve PostgreSQL password. Make sure PostgreSQL is deployed and the secret exists."
  exit 1
fi
export POSTGRES_PASSWORD=$POSTGRES_PASSWORD

NEO4J_PASSWORD=$(kubectl get secrets -n rag neo4j-cluster-neo4j-account-neo4j -o jsonpath='{.data.password}' | base64 -d)
if [ -z "$NEO4J_PASSWORD" ]; then
  echo "Error: Could not retrieve Neo4J password. Make sure Neo4J is deployed and the secret exists."
  exit 1
fi
export NEO4J_PASSWORD=$NEO4J_PASSWORD

echo "Deploying production LightRAG (using external databases)..."

if ! kubectl get namespace rag &> /dev/null; then
  echo "creating namespace 'rag'..."
  kubectl create namespace rag
fi

helm upgrade --install lightrag . \
  --namespace $NAMESPACE \
  --set-string env.POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  --set-string env.NEO4J_PASSWORD=$NEO4J_PASSWORD \
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
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=lightrag --timeout=60s -n rag
