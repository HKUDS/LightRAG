#!/bin/bash

NAMESPACE=rag

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY environment variable is not set"
  read -s -p "Enter your OpenAI API key: " OPENAI_API_KEY
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

# Install KubeBlocks (if not already installed)
bash "$SCRIPT_DIR/databases/01-prepare.sh"

# Install database clusters
bash "$SCRIPT_DIR/databases/02-install-database.sh"

# Create vector extension in PostgreSQL if enabled
print "Waiting for PostgreSQL pods to be ready..."
if kubectl wait --for=condition=ready pods -l kubeblocks.io/role=primary,app.kubernetes.io/instance=pg-cluster -n $NAMESPACE --timeout=300s; then
    print "Creating vector extension in PostgreSQL..."
    kubectl exec -it $(kubectl get pods -l kubeblocks.io/role=primary,app.kubernetes.io/instance=pg-cluster -n $NAMESPACE -o name) -n $NAMESPACE -- psql -c "CREATE EXTENSION vector;"
    print_success "Vector extension created successfully."
else
    print "Warning: PostgreSQL pods not ready within timeout. Vector extension not created."
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

#REDIS_PASSWORD=$(kubectl get secrets -n rag redis-cluster-redis-account-default -o jsonpath='{.data.password}' | base64 -d)
#if [ -z "$REDIS_PASSWORD" ]; then
#  echo "Error: Could not retrieve Redis password. Make sure Redis is deployed and the secret exists."
#  exit 1
#fi
#export REDIS_PASSWORD=$REDIS_PASSWORD

echo "Deploying production LightRAG (using external databases)..."

if ! kubectl get namespace rag &> /dev/null; then
  echo "creating namespace 'rag'..."
  kubectl create namespace rag
fi

helm upgrade --install lightrag $SCRIPT_DIR/lightrag \
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
#  --set-string env.REDIS_URI="redis://default:${REDIS_PASSWORD}@redis-cluster-redis-redis:6379"

# Wait for LightRAG pod to be ready
echo ""
echo "Waiting for lightrag pod to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=lightrag --timeout=300s -n rag
echo "lightrag pod is ready"
echo ""
echo "Running Port-Forward:"
echo "    kubectl --namespace rag port-forward svc/lightrag 9621:9621"
echo "==========================================="
echo ""
echo "✅ You can visit LightRAG at: http://localhost:9621"
echo ""
kubectl --namespace rag port-forward svc/lightrag 9621:9621
