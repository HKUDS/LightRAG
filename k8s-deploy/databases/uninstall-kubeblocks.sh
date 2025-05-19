#!/bin/bash

# Get the directory where this script is located
DATABASE_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Load configuration file
source "$DATABASE_SCRIPT_DIR/00-config.sh"

# Check dependencies
print "Checking dependencies..."
command -v kubectl >/dev/null 2>&1 || { print "Error: kubectl command not found"; exit 1; }
command -v helm >/dev/null 2>&1 || { print "Error: helm command not found"; exit 1; }

print "Checking if Kubernetes is available..."
if ! kubectl cluster-info &>/dev/null; then
    print "Error: Kubernetes cluster is not accessible. Please ensure you have proper access to a Kubernetes cluster."
    exit 1
fi

print "Checking if KubeBlocks is installed in kb-system namespace..."
if ! kubectl get namespace kb-system &>/dev/null; then
    print "KubeBlocks is not installed in kb-system namespace."
    exit 0
fi

# Function for uninstalling KubeBlocks
uninstall_kubeblocks() {
    print "Uninstalling KubeBlocks..."

    # Uninstall KubeBlocks Helm chart
    helm uninstall kubeblocks -n kb-system

    # Uninstall snapshot controller
    helm uninstall snapshot-controller -n kb-system

    # Delete KubeBlocks CRDs
    kubectl delete -f https://github.com/apecloud/kubeblocks/releases/download/v${KB_VERSION}/kubeblocks_crds.yaml --ignore-not-found=true

    # Delete CSI Snapshotter CRDs
    kubectl delete -f https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/client/config/crd/snapshot.storage.k8s.io_volumesnapshotclasses.yaml --ignore-not-found=true
    kubectl delete -f https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/client/config/crd/snapshot.storage.k8s.io_volumesnapshots.yaml --ignore-not-found=true
    kubectl delete -f https://raw.githubusercontent.com/kubernetes-csi/external-snapshotter/v8.2.0/client/config/crd/snapshot.storage.k8s.io_volumesnapshotcontents.yaml --ignore-not-found=true

    # Delete the kb-system namespace
    print "Waiting for resources to be removed..."
    kubectl delete namespace kb-system --timeout=180s

    print "KubeBlocks has been successfully uninstalled!"
}

# Call the function to uninstall KubeBlocks
uninstall_kubeblocks
