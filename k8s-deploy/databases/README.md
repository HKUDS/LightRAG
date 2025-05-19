# Using KubeBlocks to Deploy and Manage Databases

Learn how to quickly deploy and manage various databases in a Kubernetes (K8s) environment through KubeBlocks.

## Introduction to KubeBlocks

KubeBlocks is a production-ready, open-source toolkit that runs any database--SQL, NoSQL, vector, or document--on Kubernetes.
It scales smoothly from quick dev tests to full production clusters, making it a solid choice for RAG workloads like FastGPT that need several data stores working together.

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

## Installing

1. **Configure the databases you want**
    Edit `00-config.sh` file. Based on your requirements, set the variable to `true` for the databases you want to install.
    For example, to install PostgreSQL and Neo4j:

   ```bash
   ENABLE_POSTGRESQL=true
   ENABLE_REDIS=false
   ENABLE_ELASTICSEARCH=false
   ENABLE_QDRANT=false
   ENABLE_MONGODB=false
   ENABLE_NEO4J=true
   ```

2. **Prepare the environment and install KubeBlocks add-ons**

   ```bash
   bash ./01-prepare.sh
   ```

   *What the script does*
   `01-prepare.sh` performs basic pre-checks (Helm, kubectl, cluster reachability), adds the KubeBlocks Helm repo, and installs any core CRDs or controllers that KubeBlocks itself needs. It also installs the addons for every database you enabled in `00-config.sh`, but **does not** create the actual database clusters yet.

3. **(Optional) Modify database settings**
   Before deployment you can edit the `values.yaml` file inside each `<db>/` directory to change `version`, `replicas`, `CPU`, `memory`, `storage size`, etc.

4. **Install the database clusters**

   ```bash
   bash ./02-install-database.sh
   ```

   *What the script does*
   `02-install-database.sh` **actually deploys the chosen databases to Kubernetes**.

   When the script completes, confirm that the clusters are up. It may take a few minutes for all the clusters to become ready,
   especially if this is the first time running the script as Kubernetes needs to pull container images from registries.
   You can monitor the progress using the following commands:

   ```bash
   kubectl get clusters -n rag
   NAME              CLUSTER-DEFINITION   TERMINATION-POLICY   STATUS    AGE
   es-cluster                             Delete               Running   11m
   mongodb-cluster   mongodb              Delete               Running   11m
   pg-cluster        postgresql           Delete               Running   11m
   qdrant-cluster    qdrant               Delete               Running   11m
   redis-cluster     redis                Delete               Running   11m
   ```

   You can see all the Database `Pods` created by KubeBlocks.
   Initially, you might see pods in `ContainerCreating` or `Pending` status - this is normal while images are being pulled and containers are starting up.
   Wait until all pods show `Running` status:

   ```bash
   kubectl get po -n rag
   NAME                        READY   STATUS    RESTARTS   AGE
   es-cluster-mdit-0           2/2     Running   0          11m
   mongodb-cluster-mongodb-0   2/2     Running   0          11m
   pg-cluster-postgresql-0     4/4     Running   0          11m
   pg-cluster-postgresql-1     4/4     Running   0          11m
   qdrant-cluster-qdrant-0     2/2     Running   0          11m
   redis-cluster-redis-0       2/2     Running   0          11m
   ```

   You can also check the detailed status of a specific pod if it's taking longer than expected:

   ```bash
   kubectl describe pod <pod-name> -n rag
   ```

## Connect to Databases

To connect to your databases, follow these steps to identify available accounts, retrieve credentials, and establish connections:

### 1. List Available Database Clusters

First, view the database clusters running in your namespace:

```bash
kubectl get cluster -n rag
```

### 2. Retrieve Authentication Credentials

For PostgreSQL, retrieve the username and password from Kubernetes secrets:

```bash
# Get PostgreSQL username
kubectl get secrets -n rag pg-cluster-postgresql-account-postgres -o jsonpath='{.data.username}' | base64 -d
# Get PostgreSQL password
kubectl get secrets -n rag pg-cluster-postgresql-account-postgres -o jsonpath='{.data.password}' | base64 -d
```

If you have trouble finding the correct secret name, list all secrets:

```bash
kubectl get secrets -n rag
```

### 3. Port Forward to Local Machine

Use port forwarding to access PostgreSQL from your local machine:

```bash
# Forward PostgreSQL port (5432) to your local machine
# You can see all services with: kubectl get svc -n rag
kubectl port-forward -n rag svc/pg-cluster-postgresql-postgresql 5432:5432
```

### 4. Connect Using Database Client

Now you can connect using your preferred PostgreSQL client with the retrieved credentials:

```bash
# Example: connecting with psql
export PGUSER=$(kubectl get secrets -n rag pg-cluster-postgresql-account-postgres -o jsonpath='{.data.username}' | base64 -d)
export PGPASSWORD=$(kubectl get secrets -n rag pg-cluster-postgresql-account-postgres -o jsonpath='{.data.password}' | base64 -d)
psql -h localhost -p 5432 -U $PGUSER
```

Keep the port-forwarding terminal running while you're connecting to the database.


## Uninstalling

1. **Remove the database clusters**

   ```bash
   bash ./03-uninstall-database.sh
   ```

   The script deletes the database clusters that were enabled in `00-config.sh`.

2. **Clean up KubeBlocks add-ons**

   ```bash
   bash ./04-cleanup.sh
   ```

   This removes the addons installed by `01-prepare.sh`.

## Reference
* [Kubeblocks Documentation](https://kubeblocks.io/docs/preview/user_docs/overview/introduction)
