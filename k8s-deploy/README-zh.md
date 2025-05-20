# LightRAG Helm Chart

这是用于在Kubernetes集群上部署LightRAG服务的Helm chart。

LightRAG有两种推荐的部署方法：
1. **轻量级部署**：使用内置轻量级存储，适合测试和小规模使用
2. **生产环境部署**：使用外部数据库（如PostgreSQL和Neo4J），适合生产环境和大规模使用

> 如果您想要部署过程的视频演示，可以查看[bilibili](https://www.bilibili.com/video/BV1bUJazBEq2/)上的视频教程，对于喜欢视觉指导的用户可能会有所帮助。

## 前提条件

确保安装和配置了以下工具：

* **Kubernetes集群**
  * 需要一个运行中的Kubernetes集群。
  * 对于本地开发或演示，可以使用[Minikube](https://minikube.sigs.k8s.io/docs/start/)（需要≥2个CPU，≥4GB内存，以及Docker/VM驱动支持）。
  * 任何标准的云端或本地Kubernetes集群（EKS、GKE、AKS等）也可以使用。

* **kubectl**
  * Kubernetes命令行工具，用于管理集群。
  * 按照官方指南安装：[安装和设置kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl)。

* **Helm**（v3.x+）
  * Kubernetes包管理器，用于安装LightRAG。
  * 通过官方指南安装：[安装Helm](https://helm.sh/docs/intro/install/)。

## 轻量级部署（无需外部数据库）

这种部署选项使用内置的轻量级存储组件，非常适合测试、演示或小规模使用场景。无需外部数据库配置。

您可以使用提供的便捷脚本或直接使用Helm命令部署LightRAG。两种方法都配置了`lightrag/values.yaml`文件中定义的相同环境变量。

### 使用便捷脚本（推荐）：

```bash
export OPENAI_API_BASE=<您的OPENAI_API_BASE>
export OPENAI_API_KEY=<您的OPENAI_API_KEY>
bash ./install_lightrag_dev.sh
```

### 或直接使用Helm：

```bash
# 您可以覆盖任何想要的环境参数
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

### 访问应用程序：

```bash
# 1. 在终端中运行此端口转发命令：
kubectl --namespace rag port-forward svc/lightrag-dev 9621:9621

# 2. 当命令运行时，打开浏览器并导航到：
# http://localhost:9621
```

## 生产环境部署（使用外部数据库）

### 1. 安装数据库
> 如果您已经准备好了数据库，可以跳过此步骤。详细信息可以在：[README.md](databases%2FREADME.md)中找到。

我们推荐使用KubeBlocks进行数据库部署。KubeBlocks是一个云原生数据库操作符，可以轻松地在Kubernetes上以生产规模运行任何数据库。

首先，安装KubeBlocks和KubeBlocks-Addons（如已安装可跳过）：
```bash
bash ./databases/01-prepare.sh
```

然后安装所需的数据库。默认情况下，这将安装PostgreSQL和Neo4J，但您可以修改[00-config.sh](databases%2F00-config.sh)以根据需要选择不同的数据库：
```bash
bash ./databases/02-install-database.sh
```

验证集群是否正在运行：
```bash
kubectl get clusters -n rag
# 预期输出：
# NAME            CLUSTER-DEFINITION   TERMINATION-POLICY   STATUS     AGE
# neo4j-cluster                        Delete               Running    39s
# pg-cluster      postgresql           Delete               Running    42s

kubectl get po -n rag
# 预期输出：
# NAME                      READY   STATUS    RESTARTS   AGE
# neo4j-cluster-neo4j-0     1/1     Running   0          58s
# pg-cluster-postgresql-0   4/4     Running   0          59s
# pg-cluster-postgresql-1   4/4     Running   0          59s
```

### 2. 安装LightRAG

LightRAG及其数据库部署在同一Kubernetes集群中，使配置变得简单。
安装脚本会自动从KubeBlocks获取所有数据库连接信息，无需手动设置数据库凭证：

```bash
export OPENAI_API_BASE=<您的OPENAI_API_BASE>
export OPENAI_API_KEY=<您的OPENAI_API_KEY>
bash ./install_lightrag.sh
```

### 访问应用程序：

```bash
# 1. 在终端中运行此端口转发命令：
kubectl --namespace rag port-forward svc/lightrag 9621:9621

# 2. 当命令运行时，打开浏览器并导航到：
# http://localhost:9621
```

## 配置

### 修改资源配置

您可以通过修改`values.yaml`文件来配置LightRAG的资源使用：

```yaml
replicaCount: 1  # 副本数量，可根据需要增加

resources:
  limits:
    cpu: 1000m    # CPU限制，可根据需要调整
    memory: 2Gi   # 内存限制，可根据需要调整
  requests:
    cpu: 500m     # CPU请求，可根据需要调整
    memory: 1Gi   # 内存请求，可根据需要调整
```

### 修改持久存储

```yaml
persistence:
  enabled: true
  ragStorage:
    size: 10Gi    # RAG存储大小，可根据需要调整
  inputs:
    size: 5Gi     # 输入数据存储大小，可根据需要调整
```

### 配置环境变量

`values.yaml`文件中的`env`部分包含LightRAG的所有环境配置，类似于`.env`文件。当使用helm upgrade或helm install命令时，可以使用--set标志覆盖这些变量。

```yaml
env:
  HOST: 0.0.0.0
  PORT: 9621
  WEBUI_TITLE: Graph RAG Engine
  WEBUI_DESCRIPTION: Simple and Fast Graph Based RAG System

  # LLM配置
  LLM_BINDING: openai            # LLM服务提供商
  LLM_MODEL: gpt-4o-mini         # LLM模型
  LLM_BINDING_HOST:              # API基础URL（可选）
  LLM_BINDING_API_KEY:           # API密钥

  # 嵌入配置
  EMBEDDING_BINDING: openai                 # 嵌入服务提供商
  EMBEDDING_MODEL: text-embedding-ada-002   # 嵌入模型
  EMBEDDING_DIM: 1536                       # 嵌入维度
  EMBEDDING_BINDING_API_KEY:                # API密钥

  # 存储配置
  LIGHTRAG_KV_STORAGE: PGKVStorage              # 键值存储类型
  LIGHTRAG_VECTOR_STORAGE: PGVectorStorage      # 向量存储类型
  LIGHTRAG_GRAPH_STORAGE: Neo4JStorage          # 图存储类型
  LIGHTRAG_DOC_STATUS_STORAGE: PGDocStatusStorage  # 文档状态存储类型
```

## 注意事项

- 在部署前确保设置了所有必要的环境变量（API密钥和数据库密码）
- 出于安全原因，建议使用环境变量传递敏感信息，而不是直接写入脚本或values文件
- 轻量级部署适合测试和小规模使用，但数据持久性和性能可能有限
- 生产环境部署（PostgreSQL + Neo4J）推荐用于生产环境和大规模使用
- 有关更多自定义配置，请参考LightRAG官方文档
