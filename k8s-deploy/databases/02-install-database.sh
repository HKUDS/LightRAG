#!/bin/bash

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

print "Installing database clusters..."

# Install database clusters based on configuration
[ "$ENABLE_POSTGRESQL" = true ] && print "Installing PostgreSQL cluster..." && helm upgrade --install pg-cluster kubeblocks/postgresql-cluster -f "$DATABASE_SCRIPT_DIR/postgresql/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_REDIS" = true ] && print "Installing Redis cluster..." && helm upgrade --install redis-cluster kubeblocks/redis-cluster -f "$DATABASE_SCRIPT_DIR/redis/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_ELASTICSEARCH" = true ] && print "Installing Elasticsearch cluster..." && helm upgrade --install es-cluster kubeblocks/elasticsearch-cluster -f "$DATABASE_SCRIPT_DIR/elasticsearch/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_QDRANT" = true ] && print "Installing Qdrant cluster..." && helm upgrade --install qdrant-cluster kubeblocks/qdrant-cluster -f "$DATABASE_SCRIPT_DIR/qdrant/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_MONGODB" = true ] && print "Installing MongoDB cluster..." && helm upgrade --install mongodb-cluster kubeblocks/mongodb-cluster -f "$DATABASE_SCRIPT_DIR/mongodb/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_NEO4J" = true ] && print "Installing Neo4j cluster..." && helm upgrade --install neo4j-cluster kubeblocks/neo4j-cluster -f "$DATABASE_SCRIPT_DIR/neo4j/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION

# Wait for databases to be ready
print "Waiting for databases to be ready..."
TIMEOUT=600  # Set timeout to 10 minutes
START_TIME=$(date +%s)

while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))

  if [ $ELAPSED -gt $TIMEOUT ]; then
    print_error "Timeout waiting for databases to be ready. Please check database status manually and try again"
    exit 1
  fi

  # Build wait conditions for enabled databases
  WAIT_CONDITIONS=()
  [ "$ENABLE_POSTGRESQL" = true ] && WAIT_CONDITIONS+=("kubectl wait --for=condition=ready pods -l app.kubernetes.io/instance=pg-cluster -n $NAMESPACE --timeout=10s")
  [ "$ENABLE_REDIS" = true ] && WAIT_CONDITIONS+=("kubectl wait --for=condition=ready pods -l app.kubernetes.io/instance=redis-cluster -n $NAMESPACE --timeout=10s")
  [ "$ENABLE_ELASTICSEARCH" = true ] && WAIT_CONDITIONS+=("kubectl wait --for=condition=ready pods -l app.kubernetes.io/instance=es-cluster -n $NAMESPACE --timeout=10s")
  [ "$ENABLE_QDRANT" = true ] && WAIT_CONDITIONS+=("kubectl wait --for=condition=ready pods -l app.kubernetes.io/instance=qdrant-cluster -n $NAMESPACE --timeout=10s")
  [ "$ENABLE_MONGODB" = true ] && WAIT_CONDITIONS+=("kubectl wait --for=condition=ready pods -l app.kubernetes.io/instance=mongodb-cluster -n $NAMESPACE --timeout=10s")
  [ "$ENABLE_NEO4J" = true ] && WAIT_CONDITIONS+=("kubectl wait --for=condition=ready pods -l app.kubernetes.io/instance=neo4j-cluster -n $NAMESPACE --timeout=10s")

  # Check if all enabled databases are ready
  ALL_READY=true
  for CONDITION in "${WAIT_CONDITIONS[@]}"; do
    if ! eval "$CONDITION &> /dev/null"; then
      ALL_READY=false
      break
    fi
  done

  if [ "$ALL_READY" = true ]; then
    print "All database pods are ready, continuing with deployment..."
    break
  fi

  print "Waiting for database pods to be ready (${ELAPSED}s elapsed)..."
  sleep 10
done

print_success "Database clusters installation completed!"
print "Use the following command to check the status of installed clusters:"
print "kubectl get clusters -n $NAMESPACE"
