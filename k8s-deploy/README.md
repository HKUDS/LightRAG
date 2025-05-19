# LightRAG Helm Chart

This is the Helm chart for LightRAG, used to deploy LightRAG services on a Kubernetes cluster.

There are two recommended deployment methods for LightRAG:
1. **Lightweight Deployment**: Using built-in lightweight storage, suitable for testing and small-scale usage
2. **Full Deployment**: Using external databases (such as PostgreSQL and Neo4J), suitable for production environments and large-scale usage

## Prerequisites

Make sure the following tools are installed and configured:

* **Kubernetes cluster**
  * A running Kubernetes cluster is required.
  * For local development or demos you can use [Minikube](https://minikube.sigs.k8s.io/docs/start/) (needs ≥ 2 CPUs, ≥ 4 GB RAM, and Docker/VM-driver support).
  * Any standard cloud or on-premises Kubernetes cluster (EKS, GKE, AKS, etc.) also works.

* **kubectl**
  * The Kubernetes command-line interface.
  * Follow the official guide: [Install and Set Up kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl).

* **Helm** (v3.x+)
  * Kubernetes package manager used by the scripts below.
  * Install it via the official instructions: [Installing Helm](https://helm.sh/docs/intro/install/).


## Lightweight Deployment (No External Databases Required)

Uses built-in lightweight storage components with no need to configure external databases:

```bash
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

You can refer to: [install_lightrag_dev.sh](install_lightrag_dev.sh)

You can use it directly like this:
```bash
export OPENAI_API_BASE=<YOUR_OPENAI_API_BASE>
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
bash ./install_lightrag_dev.sh
```
Then you can Access the application
```bash
  1. Run this port-forward command in your terminal:
    kubectl --namespace rag port-forward svc/lightrag-dev 9621:9621

  2. While the command is running, open your browser and navigate to:
     http://localhost:9621
```

## Full Deployment (Using External Databases)

### 1. Install Databases
> You can skip this step if you've already prepared databases. Detailed information can be found in: [README.md](databases%2FREADME.md).

We recommend KubeBlocks for database deployment. KubeBlocks is a cloud-native database operator that makes it easy to run any database on Kubernetes at production scale.
FastGPT also use KubeBlocks for their database infrastructure.

First, install KubeBlocks and KubeBlocks-Addons (skip if already installed):
```bash
bash ./databases/01-prepare.sh
```

Then install the required databases. By default, this will install PostgreSQL and Neo4J, but you can modify [00-config.sh](databases%2F00-config.sh) to select different databases based on your needs. KubeBlocks supports various databases including MongoDB, Qdrant, Redis, and more.
```bash
bash ./databases/02-install-database.sh
```

When the script completes, confirm that the clusters are up. It may take a few minutes for all the clusters to become ready,
   especially if this is the first time running the script as Kubernetes needs to pull container images from registries.
   You can monitor the progress using the following commands:
```bash
kubectl get clusters -n rag
NAME            CLUSTER-DEFINITION   TERMINATION-POLICY   STATUS     AGE
neo4j-cluster                        Delete               Running    39s
pg-cluster      postgresql           Delete               Creating   42s
```
You can see all the Database `Pods` created by KubeBlocks.
   Initially, you might see pods in `ContainerCreating` or `Pending` status - this is normal while images are being pulled and containers are starting up.
   Wait until all pods show `Running` status:
```bash
kubectl get po -n rag
NAME                      READY   STATUS    RESTARTS   AGE
neo4j-cluster-neo4j-0     1/1     Running   0          58s
pg-cluster-postgresql-0   4/4     Running   0          59s
pg-cluster-postgresql-1   4/4     Running   0          59s
```

### 2. Install LightRAG

LightRAG and its databases are deployed within the same Kubernetes cluster, making configuration straightforward.
When using KubeBlocks to provide PostgreSQL and Neo4J database services, the `install_lightrag.sh` script can automatically retrieve all database connection information (host, port, user, password), eliminating the need to manually set database credentials.

You only need to run [install_lightrag.sh](install_lightrag.sh) like this:
```bash
export OPENAI_API_BASE=<YOUR_OPENAI_API_BASE>
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
bash ./install_lightrag.sh
```

The above commands automatically extract the database passwords from Kubernetes secrets, eliminating the need to manually set these credentials.

After deployment, you can access the application:
```bash
  1. Run this port-forward command in your terminal:
    kubectl --namespace rag port-forward svc/lightrag 9621:9621

  2. While the command is running, open your browser and navigate to:
     http://localhost:9621
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
- Full deployment (PostgreSQL + Neo4J) is recommended for production environments and large-scale usage
- For more customized configurations, please refer to the official LightRAG documentation
