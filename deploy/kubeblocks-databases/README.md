# KubeBlocks Databases Helm Chart

This Helm chart deploys and manages multiple database clusters (PostgreSQL, Redis, Elasticsearch, Qdrant) using [KubeBlocks](https://kubeblocks.io/).

## Prerequisites

*   Kubernetes cluster (version compatible with KubeBlocks)
*   [Helm](https://helm.sh/docs/intro/install/) (version 3+) installed.
*   [KubeBlocks](https://kubeblocks.io/docs/preview/user_docs/installation) installed in your Kubernetes cluster.
*   `kubectl` configured to interact with your cluster.

```bash
kubectl create namespace kb-system
kbcli kubeblocks install --version=1.0.0-beta.47 --namespace kb-system
```


## Installation

```bash
helm repo remove kubeblocks
helm repo add kubeblocks https://apecloud.github.io/helm-charts
helm repo update

helm upgrade --install kb-addon-elasticsearch kubeblocks/elasticsearch --namespace kb-system --version 1.0.0-alpha.0
helm upgrade --install kb-addon-qdrant kubeblocks/qdrant --namespace kb-system --version 1.0.0-alpha.0
helm upgrade --install kb-addon-postgresql kubeblocks/postgresql --namespace kb-system --version 1.0.0-alpha.0
helm upgrade --install kb-addon-redis kubeblocks/redis --namespace kb-system --version 1.0.0-alpha.0
```

```bash
kubectl create namespace demo
kubectl create secret generic postgresql-secret \
  --namespace=demo \
  --from-literal=username=postgres \
  --from-literal=password=postgres
kubectl create secret generic redis-secret \
  --namespace=demo \
  --from-literal=username=default \
  --from-literal=password=password
helm install kb-databases ./kubeblocks-databases -n demo --create-namespace \
--set redis.customSecretName=redis-secret,redis.customSecretNamespace=demo,postgresql.customSecretName=postgresql-secret,postgresql.customSecretNamespace=demo
```

generate template:
```bash
helm template kb-databases ./kubeblocks-databases -n demo --create-namespace \
--set redis.customSecretName=redis-secret,redis.customSecretNamespace=demo,postgresql.customSecretName=postgresql-secret,postgresql.customSecretNamespace=demo \
> rendered.yaml
```

## Verification

After installation, you can check the status of the deployed KubeBlocks clusters:

```bash
kubectl get clusters -n demo
kubectl get pods -n demo
```


You should see the `Cluster` resources for the enabled databases and their corresponding pods. The `NOTES.txt` output from Helm will also provide some of this information.

```bash
kubectl get clusters -n demo
NAME               CLUSTER-DEFINITION   TERMINATION-POLICY   STATUS     AGE
es-cluster                              Delete               Running    121m
pg-cluster         postgresql           Delete               Creating   121m
qdrant-cluster     qdrant               Delete               Running    121m
redis-standalone   redis                Delete               Running    121m

kubectl get pods -n demo
NAME                       READY   STATUS    RESTARTS   AGE
es-cluster-mdit-0          3/3     Running   0          110m
pg-cluster-postgresql-0    5/5     Running   0          121m
qdrant-cluster-qdrant-0    2/2     Running   0          117m
redis-standalone-redis-0   3/3     Running   0          121m
```

## Connect

port-forward:
```bash
echo "Starting Elasticsearch port-forward..."
kubectl port-forward -n demo service/es-cluster-mdit-http 9200:9200 &
ES_PID=$!
echo "Elasticsearch port-forward process ID: $ES_PID"
echo "Starting Qdrant port-forward..."
kubectl port-forward -n demo service/qdrant-cluster-qdrant-qdrant 6333:6333 &
QDRANT_PID=$!
echo "Qdrant port-forward process ID: $QDRANT_PID"
echo "Starting PostgreSQL port-forward..."
kubectl port-forward -n demo service/pg-cluster-postgresql-postgresql 5432:5432 &
PG_PID=$!
echo "PostgreSQL port-forward process ID: $PG_PID"
echo "Starting Redis port-forward..."
kubectl port-forward -n demo service/redis-standalone-redis-redis 6379:6379 &
REDIS_PID=$!
echo "Redis port-forward process ID: $REDIS_PID"
echo "All port-forwards have been started"
echo "Press Ctrl+C to stop all port-forwards"
# Capture Ctrl+C signal and clean up all processes
trap "kill $ES_PID $QDRANT_PID $PG_PID $REDIS_PID; echo 'All port-forwards stopped'; exit" INT
# Wait for any child process to finish
wait
```

## Uninstallation

To uninstall the deployed database clusters:

```bash
helm uninstall kb-databases -n demo
```
This will remove all Kubernetes resources associated with this Helm release, including the KubeBlocks `Cluster` objects. Depending on the `terminationPolicy` and KubeBlocks behavior, PVCs might also be deleted.


## Configuration

The primary way to configure the deployments is through the `values.yaml` file.

### Global Settings

These settings apply to all database clusters deployed by this chart:

```yaml
global:
  namespace: "demo"
  terminationPolicy: "Delete" # Options: DoNotTerminate, Delete, WipeOut
```

### Per-Database Settings

Each database (PostgreSQL, Redis, Elasticsearch, Qdrant) has its own configuration block. Here's an example for PostgreSQL:

```yaml
postgresql:
  enabled: true              # Set to true to deploy this database, false to skip
  name: "pg-cluster"         # Name of the KubeBlocks Cluster resource
  serviceVersion: "14.7.2"   # Database engine version
  disableExporter: false     # true to disable metrics exporter, false to enable
  replicas: 2                # Number of replicas for the main component
  resources:                 # CPU and Memory requests/limits
    limits:
      cpu: "0.5"
      memory: "0.5Gi"
    requests:
      cpu: "0.5"
      memory: "0.5Gi"
  storage: "20Gi"            # Storage size for the data volume (e.g., PVC)
```

Refer to `values.yaml` for the full set of configurable options for each database.

**Key configurable parameters for each database:**

*   `enabled`: (boolean) Deploy this database cluster.
*   `name`: (string) Name for the KubeBlocks `Cluster` resource.
*   `serviceVersion`: (string) Specific version of the database engine.
*   `disableExporter`: (boolean) Enable/disable the metrics exporter. (Note: For Elasticsearch, this might be handled differently by its `componentDef`).
*   `replicas`: (integer) Number of replicas for the primary database component.
*   `resources`: (object) Standard Kubernetes resource requests and limits.
*   `storage`: (string) Storage capacity for persistent volumes (e.g., "10Gi", "100Gi").
*   `topology`: (string, for Redis) e.g., "standalone", "replication".
*   `componentDef`: (string, for Elasticsearch) e.g., "elasticsearch-8".
