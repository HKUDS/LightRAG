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
> You can skip this step if you've already prepared databases. Detailed information can be found in: [README.md](databases%2FREADME.md).

We recommend KubeBlocks for database deployment. KubeBlocks is a cloud-native database operator that makes it easy to run any database on Kubernetes at production scale.

First, install KubeBlocks and KubeBlocks-Addons (skip if already installed):
```bash
bash ./databases/01-prepare.sh
```

Then install the required databases. By default, this will install PostgreSQL and Neo4J, but you can modify [00-config.sh](databases%2F00-config.sh) to select different databases based on your needs:
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

### Modifying Resource Configuration

You can configure LightRAG's resource usage by modifying the `values.yaml` file:

```yaml
replicaCount: 1  # Number of replicas, can be increased as needed

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
- Lightweight deployment is suitable for testing and small-scale usage, but data persistence and performance may be limited
- Production deployment (PostgreSQL + Neo4J) is recommended for production environments and large-scale usage
- For more customized configurations, please refer to the official LightRAG documentation
