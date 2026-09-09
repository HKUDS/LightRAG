#!/bin/bash

NAMESPACE=rag
RELEASE=lightrag-dev

helm uninstall $RELEASE --namespace $NAMESPACE

# The chart marks the workspace claims helm.sh/resource-policy: keep, so they
# survive this uninstall on purpose -- deleting them would destroy the graph,
# the vectors and the uploaded documents, and a dynamically provisioned volume
# with the default Delete reclaim policy goes with the claim. They are yours to
# remove once the data is backed up or confirmed unwanted; this script never
# deletes them for you.
REMAINING=$(kubectl get pvc --namespace "$NAMESPACE" \
  -o custom-columns=:.metadata.name --no-headers \
  2>/dev/null | grep -E "^(rag-storage|inputs)-${RELEASE}-[0-9]+$|^${RELEASE}-(rag-storage|inputs)$")

if [ -n "$REMAINING" ]; then
  echo
  echo "Persistent volume claims retained (they still hold your data):"
  echo "$REMAINING" | sed 's/^/  /'
  echo
  echo "To delete them and the data in them, once you are sure:"
  echo "  kubectl delete pvc --namespace $NAMESPACE $(echo "$REMAINING" | tr '\n' ' ')"
fi
