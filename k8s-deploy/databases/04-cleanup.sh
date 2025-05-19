#!/bin/bash

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

print "Uninstalling KubeBlocks database addons..."

# Uninstall database addons based on configuration
[ "$ENABLE_POSTGRESQL" = true ] && print "Uninstalling PostgreSQL addon..." && helm uninstall kb-addon-postgresql --namespace kb-system 2>/dev/null || true
[ "$ENABLE_REDIS" = true ] && print "Uninstalling Redis addon..." && helm uninstall kb-addon-redis --namespace kb-system 2>/dev/null || true
[ "$ENABLE_ELASTICSEARCH" = true ] && print "Uninstalling Elasticsearch addon..." && helm uninstall kb-addon-elasticsearch --namespace kb-system 2>/dev/null || true
[ "$ENABLE_QDRANT" = true ] && print "Uninstalling Qdrant addon..." && helm uninstall kb-addon-qdrant --namespace kb-system 2>/dev/null || true
[ "$ENABLE_MONGODB" = true ] && print "Uninstalling MongoDB addon..." && helm uninstall kb-addon-mongodb --namespace kb-system 2>/dev/null || true
[ "$ENABLE_NEO4J" = true ] && print "Uninstalling Neo4j addon..." && helm uninstall kb-addon-neo4j --namespace kb-system 2>/dev/null || true

print_success "Database addons uninstallation completed!"

source "$DATABASE_SCRIPT_DIR/uninstall-kubeblocks.sh"

kubectl delete namespace $NAMESPACE
kubectl delete namespace kb-system

print_success "KubeBlocks uninstallation completed!"
