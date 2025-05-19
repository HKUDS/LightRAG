#!/bin/bash

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

print "Uninstalling database clusters..."

# Uninstall database clusters based on configuration
[ "$ENABLE_POSTGRESQL" = true ] && print "Uninstalling PostgreSQL cluster..." && helm uninstall pg-cluster --namespace $NAMESPACE 2>/dev/null || true
[ "$ENABLE_REDIS" = true ] && print "Uninstalling Redis cluster..." && helm uninstall redis-cluster --namespace $NAMESPACE 2>/dev/null || true
[ "$ENABLE_ELASTICSEARCH" = true ] && print "Uninstalling Elasticsearch cluster..." && helm uninstall es-cluster --namespace $NAMESPACE 2>/dev/null || true
[ "$ENABLE_QDRANT" = true ] && print "Uninstalling Qdrant cluster..." && helm uninstall qdrant-cluster --namespace $NAMESPACE 2>/dev/null || true
[ "$ENABLE_MONGODB" = true ] && print "Uninstalling MongoDB cluster..." && helm uninstall mongodb-cluster --namespace $NAMESPACE 2>/dev/null || true
[ "$ENABLE_NEO4J" = true ] && print "Uninstalling Neo4j cluster..." && helm uninstall neo4j-cluster --namespace $NAMESPACE 2>/dev/null || true

print_success "Database clusters uninstalled"
print "To uninstall database addons and KubeBlocks, run 04-cleanup.sh"
