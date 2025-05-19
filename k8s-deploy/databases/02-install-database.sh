#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# KubeBlocks database installation script
# Install all database clusters

# Load configuration file
source "$SCRIPT_DIR/00-config.sh"

print "Installing database clusters..."

# Install database clusters based on configuration
[ "$ENABLE_POSTGRESQL" = true ] && print "Installing PostgreSQL cluster..." && helm upgrade --install pg-cluster kubeblocks/postgresql-cluster -f "$SCRIPT_DIR/postgresql/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_REDIS" = true ] && print "Installing Redis cluster..." && helm upgrade --install redis-cluster kubeblocks/redis-cluster -f "$SCRIPT_DIR/redis/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_ELASTICSEARCH" = true ] && print "Installing Elasticsearch cluster..." && helm upgrade --install es-cluster kubeblocks/elasticsearch-cluster -f "$SCRIPT_DIR/elasticsearch/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_QDRANT" = true ] && print "Installing Qdrant cluster..." && helm upgrade --install qdrant-cluster kubeblocks/qdrant-cluster -f "$SCRIPT_DIR/qdrant/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_MONGODB" = true ] && print "Installing MongoDB cluster..." && helm upgrade --install mongodb-cluster kubeblocks/mongodb-cluster -f "$SCRIPT_DIR/mongodb/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_NEO4J" = true ] && print "Installing Neo4j cluster..." && helm upgrade --install neo4j-cluster kubeblocks/neo4j-cluster -f "$SCRIPT_DIR/neo4j/values.yaml" --namespace $NAMESPACE --version $ADDON_CLUSTER_CHART_VERSION

print_success "Database clusters installation completed!"
print "Use the following command to check the status of installed clusters:"
print "kubectl get clusters -n $NAMESPACE"
