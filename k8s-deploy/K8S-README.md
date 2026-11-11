# LightRAG Helm Chart

This is the Helm chart for LightRAG, used to deploy LightRAG services on a Kubernetes cluster.

There are two recommended deployment methods for LightRAG:
1. **Lightweight Deployment**: Using built-in lightweight storage, suitable for testing and small-scale usage
2. **Production Deployment**: Using external databases (such as PostgreSQL and Neo4J), suitable for production environments and large-scale usage

> If you'd like a video walkthrough of the deployment process, feel free to check out this optional [video tutorial](https://youtu.be/JW1z7fzeKTw?si=vPzukqqwmdzq9Q4q) on YouTube. It might help clarify some steps for those who prefer visual guidance.

## Prerequisites

Make sure the following tools are installed and configured:

* **Kubernetes cluster**
  * A running Kubernetes cluster is required.
  * For local development or demos you can use [Minikube](https://minikube.sigs.k8s.io/docs/start/) (needs ≥ 2 CPUs, ≥ 4 GB RAM, and Docker/VM-driver support).
  * Any standard cloud or on-premises Kubernetes cluster (EKS, GKE, AKS, etc.) also works.

* **kubectl**
  * The Kubernetes command-line tool for managing your cluster.
  * Follow the official guide: [Install and Set Up kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl).

* **Helm** (v3.x+)
  * Kubernetes package manager used to install LightRAG.
  * Install it via the official instructions: [Installing Helm](https://helm.sh/docs/intro/install/).

## Lightweight Deployment (No External Databases Required)

This deployment option uses built-in lightweight storage components that are perfect for testing, demos, or small-scale usage scenarios. No external database configuration is required.

You can deploy LightRAG using either the provided convenience script or direct Helm commands. Both methods configure the same environment variables defined in the `lightrag/values.yaml` file.

### Using the convenience script (recommended):

```bash
export OPENAI_API_BASE=<YOUR_OPENAI_API_BASE>
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
bash ./install_lightrag_dev.sh
```

### Or using Helm directly:

```bash
# You can override any env param you want
helm upgrade --install lightrag ./lightrag \
  --namespace rag \
  --set-string env.LIGHTRAG_KV_STORAGE=JsonKVStorage \
  --set-string env.LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage \
  --set-string env.LIGHTRAG_GRAPH_STORAGE=NetworkXStorage \
  --set-string env.LIGHTRAG_DOC_STATUS_STORAGE=JsonDocStatusStorage \
  --set-string env.LLM_BINDING=openai \
  --set-string env.LLM_MODEL=gpt-4o-mini \
  --set-string env.LLM_BINDING_HOST=$OPENAI_API_BASE \
  --set-string env.LLM_BINDING_API_KEY=$OPENAI_API_KEY \
  --set-string env.EMBEDDING_BINDING=openai \
  --set-string env.EMBEDDING_MODEL=text-embedding-ada-002 \
  --set-string env.EMBEDDING_DIM=1536 \
  --set-string env.EMBEDDING_BINDING_API_KEY=$OPENAI_API_KEY
```

### Accessing the application:

```bash
# 1. Run this port-forward command in your terminal:
kubectl --namespace rag port-forward svc/lightrag-dev 9621:9621

# 2. While the command is running, open your browser and navigate to:
# http://localhost:9621
```

## Production Deployment (Using External Databases)

### 1. Install Databases
> You can skip this step if you've already prepared databases. Detailed information can be found in: [K8S-DB-README.md](databases/K8S-DB-README.md).

We recommend KubeBlocks for database deployment. KubeBlocks is a cloud-native database operator that makes it easy to run any database on Kubernetes at production scale.

First, install KubeBlocks and KubeBlocks-Addons (skip if already installed):
```bash
bash ./databases/01-prepare.sh
```

Then install the required databases. By default, this will install PostgreSQL and Neo4J, but you can modify [00-config.sh](databases/00-config.sh) to select different databases based on your needs:
```bash
bash ./databases/02-install-database.sh
```

Verify that the clusters are up and running:
```bash
kubectl get clusters -n rag
# Expected output:
# NAME            CLUSTER-DEFINITION   TERMINATION-POLICY   STATUS     AGE
# neo4j-cluster                        Delete               Running    39s
# pg-cluster      postgresql           Delete               Running    42s

kubectl get po -n rag
# Expected output:
# NAME                      READY   STATUS    RESTARTS   AGE
# neo4j-cluster-neo4j-0     1/1     Running   0          58s
# pg-cluster-postgresql-0   4/4     Running   0          59s
# pg-cluster-postgresql-1   4/4     Running   0          59s
```

### 2. Install LightRAG

LightRAG and its databases are deployed within the same Kubernetes cluster, making configuration straightforward.
The installation script automatically retrieves all database connection information from KubeBlocks, eliminating the need to manually set database credentials:

```bash
export OPENAI_API_BASE=<YOUR_OPENAI_API_BASE>
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
bash ./install_lightrag.sh
```

### Accessing the application:

```bash
# 1. Run this port-forward command in your terminal:
kubectl --namespace rag port-forward svc/lightrag 9621:9621

# 2. While the command is running, open your browser and navigate to:
# http://localhost:9621
```

## Configuration

### Replica Count

**Keep `replicaCount` at `1`.** A workspace supports a single ingesting LightRAG instance, and instances must never initialize concurrently. Additional query-only instances are possible under strict conditions - see [Running additional query-only instances](#running-additional-query-only-instances-advanced) below.

LightRAG coordinates storage initialization/migration and the document-processing pipeline through per-host shared memory (`lightrag/kg/shared_storage.py`). That coordination covers the worker processes of a single instance; it does not extend across pods. With two or more replicas started at the same time against a shared workspace:

- they race on first-boot storage creation (concurrent database/collection creation against a fresh backend);
- they can run schema/data migrations at the same time, and those migrations' crash-recovery heuristics can mistake another instance's in-flight migration for leftover state from a crash;
- each pod keeps its own pipeline status, so the same documents are scanned and processed twice.

For the same reason `updateStrategy` defaults to `Recreate`. A `RollingUpdate` briefly runs the old and the new pod at once, which is the same unsupported situation - and it is the version-upgrade path, where a migration is most likely to run.

To use more capacity on a single instance, scale vertically first: raise `resources`, and set `WORKERS` in `env` to run more server workers inside the one pod.

#### Running additional query-only instances (advanced)

Extra instances that only serve queries can share a workspace, but only if all five conditions below hold. This is an advanced setup: the chart does not produce it for you, and none of it is enforced by the application.

1. **External database backends only.** The lightweight deployment (`JsonKVStorage` / `NetworkXStorage` / `JsonDocStatusStorage`) keeps its state in files under the PVC and in process memory; two instances over one volume corrupt it. Use PostgreSQL / Neo4j / Milvus / Redis / Qdrant.
2. **Start instances in order — one at a time, each Ready before the next starts.** Initialization must never overlap. A `200` from `/health` is a sound gate: `initialize_storages()` and `check_and_migrate_data()` both run in the FastAPI lifespan before the app serves any request, so a Ready instance has provably finished creating and migrating storage, and every later instance finds it already in place. In Kubernetes this is what a **StatefulSet with `podManagementPolicy: OrderedReady`** does — pods are created in ordinal order and each must be Running and Ready before the next is created — combined with the readiness probe on `/health` that this chart already sets. Set `workload.kind: StatefulSet` to have the chart render exactly that (the default stays `Deployment`); see the values snippet below.
3. **Keep ingestion quiesced until every instance is Ready.** Ordered startup sequences pod *creation* only: pod 0 is Ready and serving while pod 1 initializes, and a rolling upgrade likewise leaves lower-ordinal pods serving while a replacement starts. A starting instance runs `check_and_migrate_data()`, and the chunk-tracking migration snapshots the graph's `source_id` values and then *replaces* each tracking row with what that snapshot said. A document ingested in that window can have its tracking row overwritten, and the migration is gated on the tracking store being empty, so it never re-runs and the loss does not self-heal. Stop ingestion before scaling out or upgrading, and resume it only once every pod is Ready. (The migration returns immediately once the tracking store is populated, so the exposure is the first rollout after upgrading from a version that predates chunk tracking - but the rule is not worth making conditional.)
4. **Send every write to one instance — allow-list the reads on the others.** LightRAG has no read-only mode, so the split has to be made in front of the pods, by an Ingress rule. Under `workload.kind: StatefulSet` the chart renders the two routing targets that rule needs: `<release>-writer`, which selects pod 0 only (through the `statefulset.kubernetes.io/pod-name` label), and the load-balanced `<release>` Service, which fronts **every** pod including the query-only replicas. Point the write rules at `<release>-writer`; sending them to `<release>` is what condition 4 forbids, because that Service round-robins them into a query replica. Give the extra replicas an allow-list of read-only endpoints (`/query`, `/query/stream`, the graph read endpoints, `/health`) rather than a deny-list of write ones, which silently goes stale as endpoints are added. Ingestion is not the only write surface: `/documents/*` and uploads, and also the graph mutations the WebUI issues — `/graph/entity/edit`, `/graph/relation/edit`, `/graph/entity/create`, `/graph/relation/create`, `/graph/entities/merge`, `/graph/entity/delete`, `/graph/relation/delete`. Those handlers do guard themselves with `check_pipeline_busy_or_raise`, but it reads that pod's own `pipeline_status`, so on a query replica it sees an idle pipeline and lets the edit through while the ingesting pod is mid-run — exactly the concurrent graph write the guard exists to refuse. Pipeline state being per-instance is also why two ingesting instances would scan and process the same documents twice.
5. **Expect only backend-level sharing.** A query-only instance still writes — the LLM response cache is written on the query path — which is harmless on a shared database but is another reason condition 1 is not optional. Anything held per-instance is not shared: the pipeline status an extra instance reports is its own idle pipeline, not the ingesting instance's progress.

**What ordered startup does not cover.** `OrderedReady` governs pod *creation* — the initial rollout, a scale-up, a rolling update. It has no say over an unplanned restart: the kubelet restarts a crashed container in place, and a pod lost with its node is recreated as soon as its lower-ordinal peers are Ready — which they are, because they are serving. Several pods can also restart at once. Each restart re-runs `initialize_storages()` and `check_and_migrate_data()` while the other instances keep writing, and nothing in LightRAG coordinates that across pods.

In steady state this is close to harmless: the storages already exist, so initialization is a series of no-ops, and each migration is gated on its target being empty, so it returns immediately. The window that matters is one where a migration is still pending — the first rollout after an upgrade that introduces one — where a restart can land the snapshot-then-replace write described in condition 3. If that happens, `lightrag-repair-chunk-tracking` rebuilds the tracking rows (offline: stop every writer against that workspace first).

Treat this as the accepted residue of running extra replicas without cross-pod coordination, not as something the chart handles. If you cannot accept it, stay at `replicaCount: 1`. Closing it properly needs a distributed initialization lock — a PostgreSQL advisory lock or a Redis lease — which LightRAG does not have today.

The chart can render the StatefulSet for you:

```yaml
workload:
  kind: StatefulSet          # default: Deployment
  podManagementPolicy: OrderedReady   # immutable after creation
  updateStrategy:
    type: RollingUpdate      # StatefulSet only; `updateStrategy` at the top level stays Deployment-only
replicaCount: 2              # 1 ingesting instance + 1 query-only instance
```

Switching `workload.kind` changes how storage is provisioned. A Deployment mounts the two shared PVCs; a StatefulSet gives every pod its own claim through `volumeClaimTemplates`, because a `ReadWriteOnce` claim cannot be mounted by pods on different nodes. Two consequences: an existing release does **not** carry its volume's data over when you switch kind, and uploaded files under `/app/data/inputs` live on whichever pod received them - which is the storage-level reason condition 4 above exists.

Switching kind on a release that already has data therefore needs care, because the two shared claims stop being rendered and Helm deletes what disappears from a manifest - which, on a dynamically provisioned volume with the default `Delete` reclaim policy, destroys the data with the claim. The chart marks both claims `helm.sh/resource-policy: keep` so they survive that upgrade (and `helm uninstall`); delete them yourself once you no longer need them. Helm decides what to delete from the manifest of the release already deployed, so that protection is in effect only once it has been rolled out: **back the volumes up, then switch in two steps** - first `helm upgrade` to this chart version with `workload.kind` unchanged, then a second upgrade that flips it - and copy any data you need into the new per-pod claims yourself. The chart also adds a headless Service (`<release>-headless`) that governs the StatefulSet, so you can address one specific instance as `<release>-0.<headless service>` from inside the cluster. Note that a StatefulSet name is capped at 52 characters here - a pod's `controller-revision-hash` label is `<name>-<hash>` and a label value caps at 63 - so a longer `fullnameOverride` (or release name) fails the render rather than installing a StatefulSet whose pods Kubernetes then refuses to create.

`volumeClaimTemplates` is immutable once a StatefulSet exists, so under `workload.kind: StatefulSet` the `persistence` settings are fixed at creation: a `helm upgrade` that changes `persistence.ragStorage.size` / `persistence.inputs.size`, or that flips `persistence.enabled` on after installing with it off, is rejected by the API server rather than applied. To grow the volumes on a running release, expand each per-pod PVC directly (`kubectl edit pvc rag-storage-<release>-0`, …), which requires a StorageClass with `allowVolumeExpansion: true`; the stale size in the template only affects claims created for new pods. To change the shape rather than the size, delete the StatefulSet with `--cascade=orphan` (the pods and PVCs survive) and let Helm recreate it.

### Modifying Resource Configuration

You can configure LightRAG's resource usage by modifying the `values.yaml` file:

```yaml
replicaCount: 1  # Keep at 1 - see "Replica Count" above

resources:
  limits:
    cpu: 1000m    # CPU limit, can be adjusted as needed
    memory: 2Gi   # Memory limit, can be adjusted as needed
  requests:
    cpu: 500m     # CPU request, can be adjusted as needed
    memory: 1Gi   # Memory request, can be adjusted as needed
```

### Modifying Persistent Storage

```yaml
persistence:
  enabled: true
  ragStorage:
    size: 10Gi    # RAG storage size, can be adjusted as needed
  inputs:
    size: 5Gi     # Input data storage size, can be adjusted as needed
```

These sizes are upgradable only under `workload.kind: Deployment`. A StatefulSet provisions its storage through `volumeClaimTemplates`, which is immutable once the object exists, so under `workload.kind: StatefulSet` a `helm upgrade` that changes a size - or that flips `persistence.enabled` on after installing with it off - is rejected by the API server rather than applied. See "Replica Count" above for the per-PVC expansion procedure.

### Configuring Environment Variables

The `env` section in the `values.yaml` file contains all environment configurations for LightRAG, similar to a `.env` file. When using helm upgrade or helm install commands, you can override these with the --set flag.

```yaml
env:
  HOST: 0.0.0.0
  PORT: 9621
  WEBUI_TITLE: Graph RAG Engine
  WEBUI_DESCRIPTION: Simple and Fast Graph Based RAG System

  # LLM Configuration
  LLM_BINDING: openai            # LLM service provider
  LLM_MODEL: gpt-4o-mini         # LLM model
  LLM_BINDING_HOST:              # API base URL (optional)
  LLM_BINDING_API_KEY:           # API key

  # Embedding Configuration
  EMBEDDING_BINDING: openai                 # Embedding service provider
  EMBEDDING_MODEL: text-embedding-ada-002   # Embedding model
  EMBEDDING_DIM: 1536                       # Embedding dimension
  EMBEDDING_BINDING_API_KEY:                # API key

  # Storage Configuration
  LIGHTRAG_KV_STORAGE: PGKVStorage              # Key-value storage type
  LIGHTRAG_VECTOR_STORAGE: PGVectorStorage      # Vector storage type
  LIGHTRAG_GRAPH_STORAGE: Neo4JStorage          # Graph storage type
  LIGHTRAG_DOC_STATUS_STORAGE: PGDocStatusStorage  # Document status storage type
```

## Notes

- Ensure all necessary environment variables (API keys and database passwords) are set before deployment
- For security reasons, it's recommended to pass sensitive information using environment variables rather than writing them directly in scripts or values files
- Keep `replicaCount` at `1` unless you have read "Replica Count" above: a workspace supports a single *writing* LightRAG instance, and extra query-only instances are possible only under the five conditions listed there
- `helm uninstall` (and `uninstall_lightrag.sh`) **keeps** the storage claims: they carry `helm.sh/resource-policy: keep`, because deleting them destroys the workspace and, on a dynamically provisioned volume with the default `Delete` reclaim policy, the data with it. The uninstall script prints the retained claims and the command that deletes them - run it yourself once the data is backed up or confirmed unwanted
- Lightweight deployment is suitable for testing and small-scale usage, but data persistence and performance may be limited
- Production deployment (PostgreSQL + Neo4J) is recommended for production environments and large-scale usage
- For more customized configurations, please refer to the official LightRAG documentation
