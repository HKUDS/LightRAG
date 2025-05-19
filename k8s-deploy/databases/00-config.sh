#!/bin/bash

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source "$DATABASE_SCRIPT_DIR/scripts/common.sh"

# Namespace configuration
NAMESPACE="rag"
# version
KB_VERSION="1.0.0-beta.48"
ADDON_CLUSTER_CHART_VERSION="1.0.0-alpha.0"
# Helm repository
HELM_REPO="https://apecloud.github.io/helm-charts"

# Set to true to enable the database, false to disable
ENABLE_POSTGRESQL=true
ENABLE_REDIS=false
ENABLE_QDRANT=false
ENABLE_NEO4J=true
ENABLE_ELASTICSEARCH=false
ENABLE_MONGODB=false
