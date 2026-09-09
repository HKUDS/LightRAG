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
> 如果您已经准备好了数据库，可以跳过此步骤。详细信息可以在：[K8S-DB-README.md](databases/K8S-DB-README.md)中找到。

我们推荐使用KubeBlocks进行数据库部署。KubeBlocks是一个云原生数据库操作符，可以轻松地在Kubernetes上以生产规模运行任何数据库。

首先，安装KubeBlocks和KubeBlocks-Addons（如已安装可跳过）：
```bash
bash ./databases/01-prepare.sh
```

然后安装所需的数据库。默认情况下，这将安装PostgreSQL和Neo4J，但您可以修改[00-config.sh](databases/00-config.sh)以根据需要选择不同的数据库：
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

### 副本数量

**请将 `replicaCount` 保持为 `1`。** 同一个 workspace 只支持一个执行写入的 LightRAG 实例，且多个实例绝不能并发初始化。在严格的前提条件下可以运行额外的只读查询实例，参见下方[运行额外的只读查询实例](#运行额外的只读查询实例进阶)。

LightRAG 的存储初始化/迁移与文档处理流水线是通过单机共享内存（`lightrag/kg/shared_storage.py`）协调的。该协调只覆盖同一个实例内部的多个 worker 进程，不跨 Pod。当两个及以上副本同时启动、共用同一个 workspace 时：

- 它们会在首次启动时争抢存储创建（对全新后端并发创建 database/collection）；
- 它们可能同时执行 schema/数据迁移，而迁移中的崩溃恢复逻辑会把另一个实例正在进行的迁移误判为崩溃残留；
- 每个 Pod 维护各自独立的流水线状态，同一批文档会被重复扫描和处理。

出于同样的原因，`updateStrategy` 默认为 `Recreate`。`RollingUpdate` 会让新旧 Pod 短暂同时在线，这与上述不受支持的情况相同——而且版本升级正是最可能触发迁移的场景。

如需提升处理能力，请优先纵向扩展：调高 `resources`，并在 `env` 中设置 `WORKERS` 以在同一个 Pod 内运行更多 server worker。

#### 运行额外的只读查询实例（进阶）

只提供查询服务的额外实例可以与主实例共用同一个 workspace，但必须同时满足下面五个条件。这属于进阶用法：本 chart 不会自动生成这种部署，且以下约束没有任何一条由应用本身强制执行。

1. **只能使用外部数据库后端。** 轻量级部署（`JsonKVStorage` / `NetworkXStorage` / `JsonDocStatusStorage`）把状态保存在 PVC 上的文件和进程内存中，两个实例共用一个卷必然导致数据损坏。请使用 PostgreSQL / Neo4j / Milvus / Redis / Qdrant。
2. **实例必须顺序启动——一次一个，前一个 Ready 之后再启动下一个。** 初始化阶段绝不能重叠。用 `/health` 返回 `200` 作为判据是可靠的：`initialize_storages()` 与 `check_and_migrate_data()` 都在 FastAPI lifespan 中执行，先于应用对外提供服务，因此一个 Ready 的实例必然已完成存储创建与迁移，后续实例只会看到存储已经就绪。在 Kubernetes 中，**StatefulSet 配合 `podManagementPolicy: OrderedReady`** 正是这个语义——Pod 按序号逐个创建，每个必须 Running 且 Ready 之后才创建下一个——再配合本 chart 已经配置好的 `/health` readinessProbe 即可。设置 `workload.kind: StatefulSet` 即可让 chart 直接渲染出这种部署（默认仍为 `Deployment`），参见下方的 values 示例。
3. **在所有实例都 Ready 之前，必须停止写入。** 顺序启动只保证 Pod 的**创建**有先后：pod 0 已经 Ready 并对外服务时，pod 1 才刚开始初始化；滚动升级同理，序号较低的旧 Pod 仍在服务，替换 Pod 已经启动。正在启动的实例会执行 `check_and_migrate_data()`，其中的 chunk 追踪迁移先对图中的 `source_id` 取快照，随后用该快照**整行替换**每一条追踪记录。在这个窗口内写入的文档，其追踪记录可能被覆盖；而该迁移以追踪存储为空作为闸门，因此不会再次执行，损失也不会自愈。请在扩容或升级前停止写入，待所有 Pod 都 Ready 之后再恢复。（追踪存储一旦有数据，该迁移会立即返回，因此风险窗口实际上是从早于 chunk 追踪的版本升级后的首次上线——但这条规则不值得写成有条件的。）
4. **所有写入都发往同一个实例，其余实例只放行白名单内的读接口。** LightRAG 没有只读模式，因此这个划分只能由 Pod 前面的 Ingress 规则完成。在 `workload.kind: StatefulSet` 下，本 chart 会渲染出该规则所需的两个路由目标：`<release>-writer` 只选中 pod 0（通过 `statefulset.kubernetes.io/pod-name` 标签），而负载均衡的 `<release>` Service 覆盖**所有** Pod，包括只读查询副本。请把写入规则指向 `<release>-writer`；指向 `<release>` 正是本条禁止的做法——该 Service 会把写请求轮转到查询副本上。请为额外副本配置**只读接口白名单**（`/query`、`/query/stream`、图查询接口、`/health`），而不是写接口黑名单——黑名单会随着新接口的加入而悄悄失效。写入面不只有文档摄取：除了 `/documents/*` 和上传接口，还有 WebUI 会发起的图修改接口——`/graph/entity/edit`、`/graph/relation/edit`、`/graph/entity/create`、`/graph/relation/create`、`/graph/entities/merge`、`/graph/entity/delete`、`/graph/relation/delete`。这些处理函数确实有 `check_pipeline_busy_or_raise` 自我保护，但它读的是本 Pod 自己的 `pipeline_status`，所以在查询副本上它看到的是空闲流水线，会在写入实例正在处理文档时放行这次修改——而这正是该守卫本应拒绝的并发图写入。流水线状态每实例独立，也正是两个实例同时接收写入会把同一批文档处理两遍的原因。
5. **只能指望后端层面的共享。** 查询实例并非完全不写入——查询路径会写 LLM 响应缓存；这在共享数据库后端上无害，但也正是条件 1 不能放宽的原因。所有进程内状态都不共享：额外实例上报的流水线状态是它自己的空闲流水线，而不是写入实例的处理进度。

**顺序启动覆盖不到的部分。** `OrderedReady` 管的是 Pod 的**创建**——首次上线、扩容、滚动更新。它对非计划的重启没有约束力：容器崩溃时 kubelet 会就地重启它；Pod 随节点丢失后，只要序号更低的同伴处于 Ready 就会立即被重建——而它们正在服务，当然是 Ready。多个 Pod 也可能同时重启。每一次重启都会在其他实例继续写入的同时重新执行 `initialize_storages()` 与 `check_and_migrate_data()`，而 LightRAG 内部没有任何跨 Pod 的协调机制。

在稳定状态下这基本无害：存储都已存在，初始化只是一连串空操作；每个迁移都以目标为空作为闸门，会立即返回。真正有风险的窗口是仍有迁移待执行时——即引入了新迁移的那次升级后的首次上线——此时一次重启就可能落入条件 3 描述的「快照后整行替换」写入。若真的发生，可用 `lightrag-repair-chunk-tracking` 重建追踪记录（该工具仅支持离线运行：请先停止所有对该 workspace 的写入方）。

请把这一点视为「在没有跨 Pod 协调的前提下运行额外副本」所接受的既有残留，而不是 chart 已经处理掉的问题。如果无法接受，请保持 `replicaCount: 1`。要真正消除它，需要一个分布式初始化锁——PostgreSQL advisory lock 或 Redis lease——而 LightRAG 目前还没有。

本 chart 可以直接渲染出这种 StatefulSet 部署：

```yaml
workload:
  kind: StatefulSet          # 默认为 Deployment
  podManagementPolicy: OrderedReady   # 创建后不可修改
  updateStrategy:
    type: RollingUpdate      # 仅对 StatefulSet 生效；顶层的 `updateStrategy` 仍然只用于 Deployment
replicaCount: 2              # 1 个写入实例 + 1 个只读查询实例
```

切换 `workload.kind` 会改变存储的供给方式。Deployment 挂载两个共享 PVC；StatefulSet 则通过 `volumeClaimTemplates` 为每个 Pod 分配各自的 PVC，因为 `ReadWriteOnce` 的 PVC 无法被位于不同节点的多个 Pod 同时挂载。由此带来两个后果：已有的 release 在切换 kind 后**不会**继承原卷中的数据；上传到 `/app/data/inputs` 的文件只存在于接收该请求的那个 Pod 上——这正是上面条件 4 在存储层面的原因。

因此，对已有数据的 release 切换 kind 必须谨慎：两个共享 PVC 会不再被渲染，而 Helm 会删除从 manifest 中消失的资源——对于动态供给、默认 `Delete` 回收策略的卷，这会连同 PVC 一起销毁其中的数据。本 chart 已为这两个 PVC 标注 `helm.sh/resource-policy: keep`，使其能在该次升级（以及 `helm uninstall`）中保留下来；确认不再需要后请自行删除。由于 Helm 是依据**已部署** release 的 manifest 来决定删除哪些资源，这项保护只有在它已经上线之后才生效，所以请**先备份卷，再分两步切换**——第一步用当前 `workload.kind` 升级到本版本 chart，第二步再翻转该值——并自行把需要的数据复制到新的每 Pod PVC 中。chart 同时会创建一个用于治理 StatefulSet 的 headless Service（`<release>-headless`），在集群内部可以用 `<release>-0.<headless service 名>` 精确寻址某一个实例。另请注意，这里的 StatefulSet 名上限是 52 字符——Pod 的 `controller-revision-hash` 标签形如 `<name>-<hash>`，而标签值上限为 63——因此过长的 `fullnameOverride`（或 release 名）会让渲染直接失败，而不是装出一个 Kubernetes 随后拒绝创建其 Pod 的 StatefulSet。

StatefulSet 一旦创建，`volumeClaimTemplates` 就不可变，因此在 `workload.kind: StatefulSet` 下，`persistence` 相关配置在创建时即固定：`helm upgrade` 修改 `persistence.ragStorage.size` / `persistence.inputs.size`，或在最初关闭 persistence 后再将其打开，都会被 API server 拒绝而非生效。若要为运行中的 release 扩容，请直接扩展每个 Pod 各自的 PVC（`kubectl edit pvc rag-storage-<release>-0` 等），这需要 StorageClass 设置了 `allowVolumeExpansion: true`；模板中残留的旧容量只影响之后为新 Pod 创建的 PVC。若要改变结构而非容量，请用 `--cascade=orphan` 删除 StatefulSet（Pod 与 PVC 会保留），再让 Helm 重新创建。

### 修改资源配置

您可以通过修改`values.yaml`文件来配置LightRAG的资源使用：

```yaml
replicaCount: 1  # 请保持为 1，参见上方“副本数量”

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

这两个容量只在 `workload.kind: Deployment` 下可以通过升级修改。StatefulSet 的存储由 `volumeClaimTemplates` 供给，而它在对象创建后即不可变，因此在 `workload.kind: StatefulSet` 下，`helm upgrade` 修改容量、或在最初关闭 persistence 后再打开，都会被 API server 拒绝而非生效。每个 PVC 的扩容步骤见上方「副本数量」。

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
- 除非已经读过上方「副本数量」，请把 `replicaCount` 保持为 `1`：同一个 workspace 只支持一个**执行写入**的 LightRAG 实例，额外的只读查询实例只有在该节列出的五个条件全部满足时才可行
- `helm uninstall`（以及 `uninstall_lightrag.sh`）会**保留**存储 PVC：它们带有 `helm.sh/resource-policy: keep`，因为删除它们就等于销毁整个 workspace，而对动态供给、默认 `Delete` 回收策略的卷而言，数据会随 PVC 一起消失。卸载脚本会打印被保留的 PVC 以及删除它们的命令——请在数据已备份或确认不再需要之后自行执行
- 轻量级部署适合测试和小规模使用，但数据持久性和性能可能有限
- 生产环境部署（PostgreSQL + Neo4J）推荐用于生产环境和大规模使用
- 有关更多自定义配置，请参考LightRAG官方文档
