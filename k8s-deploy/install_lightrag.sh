#!/bin/bash

NAMESPACE=rag

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load enabled-database flags (ENABLE_POSTGRESQL, ENABLE_NEO4J,
# ENABLE_DOCUMENTDB, ...) so we only resolve credentials for engines the user
# actually deployed.
source "$SCRIPT_DIR/databases/00-config.sh"

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
if [ "$ENABLE_POSTGRESQL" = true ]; then
  print "Waiting for PostgreSQL pods to be ready..."
  if kubectl wait --for=condition=ready pods -l kubeblocks.io/role=primary,app.kubernetes.io/instance=pg-cluster -n $NAMESPACE --timeout=300s; then
      print "Creating vector extension in PostgreSQL..."
      kubectl exec -it $(kubectl get pods -l kubeblocks.io/role=primary,app.kubernetes.io/instance=pg-cluster -n $NAMESPACE -o name) -n $NAMESPACE -- psql -c "CREATE EXTENSION vector;"
      print_success "Vector extension created successfully."
  else
      print "Warning: PostgreSQL pods not ready within timeout. Vector extension not created."
  fi
fi

# Get database passwords from Kubernetes secrets
echo "Retrieving database credentials from Kubernetes secrets..."

HELM_OVERRIDES=()

if [ "$ENABLE_POSTGRESQL" = true ]; then
  POSTGRES_PASSWORD=$(kubectl get secrets -n rag pg-cluster-postgresql-account-postgres -o jsonpath='{.data.password}' | base64 -d)
  if [ -z "$POSTGRES_PASSWORD" ]; then
    echo "Error: Could not retrieve PostgreSQL password. Make sure PostgreSQL is deployed and the secret exists."
    exit 1
  fi
  export POSTGRES_PASSWORD=$POSTGRES_PASSWORD
  HELM_OVERRIDES+=(--set-string "env.POSTGRES_PASSWORD=$POSTGRES_PASSWORD")
fi

if [ "$ENABLE_NEO4J" = true ]; then
  NEO4J_PASSWORD=$(kubectl get secrets -n rag neo4j-cluster-neo4j-account-neo4j -o jsonpath='{.data.password}' | base64 -d)
  if [ -z "$NEO4J_PASSWORD" ]; then
    echo "Error: Could not retrieve Neo4J password. Make sure Neo4J is deployed and the secret exists."
    exit 1
  fi
  export NEO4J_PASSWORD=$NEO4J_PASSWORD
  HELM_OVERRIDES+=(--set-string "env.NEO4J_PASSWORD=$NEO4J_PASSWORD")
fi

if [ "$ENABLE_DOCUMENTDB" = true ]; then
  echo "Waiting for DocumentDB cluster to become healthy..."
  kubectl wait --for=jsonpath='{.status.status}'="Cluster in healthy state" \
    documentdb/documentdb-cluster -n $NAMESPACE --timeout=300s || {
      echo "Error: DocumentDB cluster did not reach healthy state."
      exit 1
  }

  # The operator publishes a connection string on the DocumentDB resource
  # status. It contains embedded $(kubectl get secret ...) substitutions, so
  # `eval` resolves the credentials. Trust this field only because we created
  # the DocumentDB resource ourselves above.
  RAW_CONN=$(kubectl get documentdb documentdb-cluster -n $NAMESPACE \
    -o jsonpath='{.status.connectionString}')
  if [ -z "$RAW_CONN" ]; then
    echo "Error: DocumentDB status.connectionString is empty."
    exit 1
  fi
  MONGO_URI=$(eval "echo \"$RAW_CONN\"")

  # Replace the gateway ClusterIP with its in-cluster DNS name so the URI
  # remains valid across pod restarts.
  SVC_IP=$(kubectl get svc "documentdb-service-documentdb-cluster" -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null) || true
  if [ -n "$SVC_IP" ]; then
    SVC_DNS="documentdb-service-documentdb-cluster.${NAMESPACE}.svc.cluster.local"
    MONGO_URI=$(echo "$MONGO_URI" | sed "s/$SVC_IP/$SVC_DNS/g")
  fi

  # The connection string sets both directConnection=true and replicaSet=rs0.
  # pymongo treats these as conflicting (the gateway advertises itself as
  # standalone, not as a member of "rs0"), so strip replicaSet for direct
  # gateway connections.
  MONGO_URI=$(echo "$MONGO_URI" | sed -E 's/[?&]replicaSet=[^&]*//g')
  export MONGO_URI

  # Switch storage backends to DocumentDB-backed Mongo* implementations.
  # NanoVectorDBStorage stays on the local PVC because DocumentDB lacks the
  # MongoDB Atlas $vectorSearch operator that MongoVectorDBStorage requires.
  HELM_OVERRIDES+=(
    --set-string "env.MONGO_URI=$MONGO_URI"
    --set-string "env.MONGO_DATABASE=lightrag"
    --set-string "env.LIGHTRAG_KV_STORAGE=MongoKVStorage"
    --set-string "env.LIGHTRAG_GRAPH_STORAGE=MongoGraphStorage"
    --set-string "env.LIGHTRAG_DOC_STATUS_STORAGE=MongoDocStatusStorage"
    --set-string "env.LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage"
  )
fi

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
  "${HELM_OVERRIDES[@]}" \
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
