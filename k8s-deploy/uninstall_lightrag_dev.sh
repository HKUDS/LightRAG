#!/bin/bash

NAMESPACE=rag
RELEASE=lightrag-dev

# The workspace claims are named after lightrag.fullname, which equals the
# release name only when fullnameOverride is unset. The workload object carries
# that rendered name, so read it from the cluster while the release still
# exists instead of assuming, and fall back to the release name when there is
# no workload left to ask.
FULLNAME=$(kubectl get deployment,statefulset --namespace "$NAMESPACE" \
  -l app.kubernetes.io/instance="$RELEASE" \
  -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
[ -n "$FULLNAME" ] || FULLNAME=$RELEASE

helm uninstall "$RELEASE" --namespace "$NAMESPACE"
STATUS=$?

# The chart marks the workspace claims helm.sh/resource-policy: keep, so they
# survive this uninstall on purpose -- deleting them would destroy the graph,
# the vectors and the uploaded documents, and a dynamically provisioned volume
# with the default Delete reclaim policy goes with the claim. They are yours to
# remove once the data is backed up or confirmed unwanted; this script never
# deletes them for you.
if CLAIMS=$(kubectl get pvc --namespace "$NAMESPACE" \
    -o custom-columns=:.metadata.name --no-headers); then
  RETAINED=$(echo "$CLAIMS" | grep -E \
    "^(rag-storage|inputs)-${FULLNAME}-[0-9]+$|^${FULLNAME}-(rag-storage|inputs)$")
  if [ -n "$RETAINED" ]; then
    echo
    echo "Persistent volume claims retained (they still hold your data):"
    echo "$RETAINED" | sed 's/^/  /'
    echo
    echo "To delete them and the data in them, once you are sure:"
    echo "  kubectl delete pvc --namespace $NAMESPACE $(echo "$RETAINED" | tr '\n' ' ' | sed 's/ *$//')"
  fi
else
  echo "Warning: could not list persistent volume claims in namespace $NAMESPACE." >&2
  echo "Check for retained claims manually before assuming the cleanup is done." >&2
  [ "$STATUS" -eq 0 ] && STATUS=1
fi

# Report the uninstall's own result: automation must not read a failed
# `helm uninstall` as a successful cleanup just because the listing ran.
exit $STATUS
