#!/bin/bash

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

check_dependencies

# Check if KubeBlocks is already installed, install it if it is not.
source "$DATABASE_SCRIPT_DIR/install-kubeblocks.sh"

# Create namespaces
print "Creating namespaces..."
kubectl create namespace $NAMESPACE 2>/dev/null || true

# Install database addons
print "Installing KubeBlocks database addons..."

# Add and update Helm repository
print "Adding and updating KubeBlocks Helm repository..."
helm repo add kubeblocks $HELM_REPO
helm repo update
# Install database addons based on configuration
[ "$ENABLE_POSTGRESQL" = true ] && print "Installing PostgreSQL addon..." && helm upgrade --install kb-addon-postgresql kubeblocks/postgresql --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_REDIS" = true ] && print "Installing Redis addon..." && helm upgrade --install kb-addon-redis kubeblocks/redis --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_ELASTICSEARCH" = true ] && print "Installing Elasticsearch addon..." && helm upgrade --install kb-addon-elasticsearch kubeblocks/elasticsearch --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_QDRANT" = true ] && print "Installing Qdrant addon..." && helm upgrade --install kb-addon-qdrant kubeblocks/qdrant --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_MONGODB" = true ] && print "Installing MongoDB addon..." && helm upgrade --install kb-addon-mongodb kubeblocks/mongodb --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION
[ "$ENABLE_NEO4J" = true ] && print "Installing Neo4j addon..." && helm upgrade --install kb-addon-neo4j kubeblocks/neo4j --namespace kb-system --version $ADDON_CLUSTER_CHART_VERSION

# DocumentDB ships its own Kubernetes operator (not a KubeBlocks addon).
# Install cert-manager (a hard dependency of the DocumentDB operator) and the
# DocumentDB operator itself. cert-manager is installed only if not already
# present so we don't clobber an existing installation.
if [ "$ENABLE_DOCUMENTDB" = true ]; then
  if ! helm status cert-manager -n cert-manager &> /dev/null && ! kubectl get deployment cert-manager -n cert-manager &> /dev/null; then
    print "Installing cert-manager (DocumentDB operator dependency)..."
    helm repo add jetstack https://charts.jetstack.io 2>/dev/null || true
    helm repo update jetstack
    helm upgrade --install cert-manager jetstack/cert-manager \
      --namespace cert-manager --create-namespace \
      --set installCRDs=true --wait
  else
    print "cert-manager already present, skipping."
  fi

  print "Installing DocumentDB operator..."
  helm repo add documentdb https://documentdb.github.io/documentdb-kubernetes-operator 2>/dev/null || true
  helm repo update documentdb
  helm upgrade --install documentdb-operator documentdb/documentdb-operator \
    --namespace documentdb-operator --create-namespace --wait
fi

print_success "KubeBlocks database addons installation completed!"
print "Now you can run 02-install-database.sh to install database clusters"
