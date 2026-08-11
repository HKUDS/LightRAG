# PRD：服务端多工作空间（第1阶段）

- 状态：设计草案（决策已由维护者逐条裁决，2026-08-11）
- 适用范围：LightRAG Core（`lightrag/`）+ Server（`lightrag/api/`）+ WebUI
- 前置依赖：[LR2-auhtorization-file-policy-phase1.md](./LR2-auhtorization-file-policy-phase1.md) 已落地（本方案的授权层建立在它的 `authorize` 咽喉与权限目录之上）
- 关键约束（维护者裁决）：工作区必须**显式创建**，不支持 auto-create；工作区身份是**去中心化短 ID**，用户可见名字可随时改；`doc_status` 与流水线**完全隔离**；LLM/VLM 并发是**Server 级共享**；本阶段**不涉及多租户**
- 来源讨论：[#3511](https://github.com/HKUDS/LightRAG/issues/3511)（RFC: Safe Multi-Workspace Architecture）、[#3563](https://github.com/HKUDS/LightRAG/issues/3563)（refining design for server-side multi-workspace pools）、[PR #3397 评审](https://github.com/HKUDS/LightRAG/pull/3397#issuecomment-5056105242)、[#2527](https://github.com/HKUDS/LightRAG/issues/2527)

## TL;DR 与阅读指南

一句话概括：新增 `WS_CATALOG` 存储类型持久化工作区目录，工作区身份是不可变的 `ws_` 短 ID（用户可见名字与之解耦、可改），Server 按 `LIGHTRAG-WORKSPACE` 头把请求路由到有界实例池里一个**永久绑定该 ID** 的 `LightRAG` 实例；四类存储 + 流水线 + doc_status + 输入目录全部按 ID 隔离，LLM/VLM/embedding/rerank 并发与摄取侧 LLM 缓存按 Server 共享；未登记的工作区一律 404，`*_WORKSPACE` 环境变量在多工作空间模式下拒绝启动。

| 读者 | 需要读的章节 |
| --- | --- |
| 运维 / 部署 | §6（`*_WORKSPACE` 处置）、§9（启动与迁移）、§16（配置项）、§17（破坏性变更）、§19（失败语义） |
| 后端实现者 | §3–§5（身份、目录、生命周期）、§6–§8（有效工作区规则、缓存分层、实例池）、§10–§13（流水线、资源、API、Ollama）、§14（授权） |
| 前端 | §12.3（`/status`）、§14.4（权限门控）、§15（WebUI） |
| 评审 | §22（与两份 RFC 的取舍）、§23（遗留风险）、附录 B（需同步修改 authz PRD 的条目） |

**唯一权威口径**（其它位置只作引用，不再复述理由）：工作区 ID 编码 = 附录 A；`legacy_layout` 物理映射表 = §6.3；LLM 缓存分层判据 = §7.1；实例可驱逐条件 = §8.3；端点实例化策略 = §12.2；有效权限公式 = §14.2；状态码语义 = §19。

---

## 1. 目标与非目标

### 1.1 目标

1. 一台 Server（一个 URL）服务多个相互隔离的知识库，请求级选择工作区，覆盖 #2527 的原始诉求。
2. 工作区只能由**已授权的管理操作**创建。数据面选择器永不创建工作区；未登记的选择器返回 404。
3. 工作区身份是**不可变的去中心化 ID**，可跨 Server 迁移而不重名；用户可见名字与 ID 解耦，改名不触碰任何存储。
4. 一个 `LightRAG` 实例在其生命周期内**永久绑定**一个工作区 ID，永不中途切换。
5. `doc_status`、流水线状态、输入目录、四类存储、查询答案缓存全部按工作区隔离；同一份文档可以插入到多个工作区，各自独立的解析/分块选项与生命周期。
6. LLM / VLM / embedding / rerank 并发是**Server 级总量**，激活 N 个工作区不会把上限变成 N×。
7. 零迁移升级：既有单工作空间部署（含 `WORKSPACE` 未设置的）升级后数据不移动、不重嵌入、不改名，行为不变。
8. 有界资源：实例数、连接数、准入队列全部有上限，不存在内存无界行为。
9. 授权支持"哪些用户可以访问哪些工作区"，且不引入第二套授权引擎。

### 1.2 非目标（本阶段明确不做）

- **不做多租户**。工作区是独立于租户的对象；租户模型、跨租户共享、配额留待后续阶段。
- 不做物理隔离 / storage profile（每工作区独立数据库或连接串）。逻辑隔离必须在不要求"一库一工作区"的前提下成立。
- 不做运行期切换实例的工作区绑定。
- 不做每工作区独立的 LLM / embedding / rerank / prompt 配置（模型面保持 Server 全局，这是 §11 共享队列的前提）。
- 不做每工作区独立认证（凭据面仍是 Server 级），也不做行级 / 单文档 ACL。
- 不做启动期自动恢复流水线。重启后由运维显式发 `/documents/scan` 或 `/documents/reprocess_failed`（与今天语义一致，见 §10.4）。
- 不做多节点（多主机）部署。同主机 Gunicorn 多 worker 在范围内。

---

## 2. 现状分析

| 现状 | 位置 | 对本方案的影响 |
| --- | --- | --- |
| `LightRAG` 已接受 `workspace` 参数，四类存储各自实现了命名空间隔离 | `lightrag/lightrag.py`、`lightrag/kg/*` | 逻辑隔离的地基已在；缺的是目录、路由、池、身份规范与一致性校验 |
| **`*_WORKSPACE` 后端变量优先级高于实例 workspace** | [postgres_impl.py:2690-2702](../../lightrag/kg/postgres_impl.py#L2690-L2702)（另有 `:3918`、`:5040`、`:6887` 三处同构） | 致命：设了 `POSTGRES_WORKSPACE` 后池里所有实例塌缩到同一 PG 命名空间，隔离承诺作废，现场只有一行 info 日志。§6.2 必须拒绝 |
| 官方 K8s chart 自带 `POSTGRES_WORKSPACE: default` | [k8s-deploy/lightrag/values.yaml:75](../../k8s-deploy/lightrag/values.yaml#L75) | 第一批撞上 §6.2 的就是官方 chart 用户；chart 必须同 PR 修改 |
| PG 把空 workspace 落成字面量 `"default"`，且 `workspace == "default"` 时按 legacy 布局处理 | [postgres_impl.py:2702](../../lightrag/kg/postgres_impl.py#L2702)、[:6844](../../lightrag/kg/postgres_impl.py#L6844) | 用户若能创建名为 `default` 的工作区，在 PG 上会静默落进原有全局数据。§3.2 的 ID 前缀结构使之不可能 |
| Redis 把空 workspace 落成 `"_"` | [redis_impl.py:832](../../lightrag/kg/redis_impl.py#L832) | 同上，`_` 进保留标识符集合 |
| workspace 字符集校验是**静默替换**而非拒绝 | [config.py:911-919](../../lightrag/api/config.py#L911-L919)：`re.sub(r"[^a-zA-Z0-9_]", "_", ...)` | 两个不同输入可被改写成同名，只留 warning。§3.3 改为拒绝 |
| 文件类存储用 `validate_workspace()` 防路径穿越，空 workspace 落在 `working_dir` 根 | [json_doc_status_impl.py:116-133](../../lightrag/kg/json_doc_status_impl.py#L116-L133) | legacy 布局的 file 侧形态，§6.3 保留 |
| 跨 worker LLM/embedding/rerank 并发闸门已存在，group 名与 workspace 无关 | [lightrag.py:1342](../../lightrag/lightrag.py#L1342)、[:1372](../../lightrag/lightrag.py#L1372)（`"rerank"` / `"embedding"` / `f"llm:{role}"`） | item 10 的地基已在，可直接复用 |
| **但全局上限只在 `workers > 1` 时注册** | [run_with_gunicorn.py:282-294](../../lightrag/api/run_with_gunicorn.py#L282-L294)：注释"Single-worker mode needs no cross-process gate — the per-process max_async already IS the total limit there" | 该前提假设"一进程一实例"。多工作空间下 uvicorn 单进程会变成 N × max_async。§11.1 修 |
| `lifespan` 启动期做 `initialize_storages()` + `check_and_migrate_data()`，**无任何启动期 scan 或流水线恢复** | [lightrag_server.py:1348-1380](../../lightrag/api/lightrag_server.py#L1348-L1380) | §10.4 的"重启后手工恢复"就是今天的语义，不是行为变更；§9 的启动 gate 沿用这里 |
| LLM 缓存是**一个** KV 命名空间，缓存类型靠扁平 key 前缀 `{mode}:{cache_type}:{hash}` 区分 | [namespace.py:10](../../lightrag/namespace.py#L10)、[utils.py:920-931](../../lightrag/utils.py#L920-L931) | §7 必须拆成两个命名空间，否则跨工作区共享会泄漏查询答案 |
| 答案缓存 key 只含 query + 检索参数，**不含 workspace，也不含被检索的内容** | [operate.py:4535-4548](../../lightrag/operate.py#L4535-L4548) | 直接共享整个缓存命名空间 = 同一问题在 A 库的答案被当作 B 库的答案返回。§7.1 的判据由此而来 |
| 抽取缓存 key 只含 chunk 内容 + prompt + 模型身份 | [utils.py:731](../../lightrag/utils.py#L731)、[pipeline.py:6800](../../lightrag/pipeline.py#L6800) | 同一份文档在不同工作区能命中缓存，维护者的 Q10 前提成立 |
| 解析/分块选项已是**每请求**可指定，默认值取自实例的 `addon_params` | [document_routes.py:2305-2357](../../lightrag/api/routers/document_routes.py#L2305-L2357) | §5.4 的每工作区默认值可挂在 `addon_params` 之前的一层，不必改请求面 |
| Ollama 兼容面只有一个模型名 `LIGHTRAG_MODEL`，且**任何 model 名都被接受并忽略** | [config.py:907-908](../../lightrag/api/config.py#L907-L908)、[ollama_api.py:363-372](../../lightrag/api/routers/ollama_api.py#L363-L372) | §13 用 tag 路由；未知 tag 改 404 属破坏性变更 |
| 存储实现注册表与 `LIGHTRAG_*_STORAGE` 环境变量约定 | [kg/\_\_init\_\_.py](../../lightrag/kg/__init__.py)、[config.py:591-596](../../lightrag/api/config.py#L591-L596) | §4 的 `WS_CATALOG` 按同一约定新增 |
| WebUI 四个 TAB（`documents` / `knowledge-graph` / `retrieval` / `api`），左上角渲染 `webuiTitle` / `webuiDescription` | [SiteHeader.tsx:41-53](../../lightrag_webui/src/features/SiteHeader.tsx#L41-L53)、[:83-96](../../lightrag_webui/src/features/SiteHeader.tsx#L83-L96) | §15 的改造面 |
| 授权咽喉 `authorize(request, security_scopes, ...)`、`PolicySnapshot.effective` 预计算、`AuthorizationContext` / `ResourceScope` 已留扩展点 | authz PRD §4.2 / §10.1 | §14 在其上加第二段判定，不新增授权引擎 |
| authz PRD v1 规则 #4 明确拒绝加载含 `scope`/`workspace`/`tenant` 字段的策略文件 | authz PRD §7.3 | §14.1 选择"目录管成员"正是为了不动这条规则；附录 B 记录需同步的措辞 |

---

## 3. 工作区身份

### 3.1 三个名字，各管一件事

| 概念 | 字段 | 谁看 | 可变 | 是否进物理命名空间 |
| --- | --- | --- | --- | --- |
| 工作区 ID | `workspace` | 运维 / API / 日志 | **不可变** | **是**（legacy 记录除外，见 §6.3） |
| 工作区名字 | `workspace_name` | 最终用户（WebUI 下拉框） | 可随时改 | **永不** |
| Ollama 模型 tag | `ollama_tag` | Ollama 客户端 | 可改 | 永不 |

三者在 Server 内各自唯一。改 `workspace_name` 或 `ollama_tag` 都不触碰任何存储、不触碰实例池、不触碰授权成员表。

### 3.2 工作区 ID 的形态

```text
ws_<16 位小写 base36>          例：ws_0mkq3f1x7d9b2v4c
总长 19 字符，字符集 [a-z0-9_]，首字符必为字母
```

编码内容与长度预算见**附录 A**（唯一权威口径）。要点：

- **80 位**信息量（48 位毫秒时间戳 || 32 位 CSPRNG），去中心化生成，跨 Server 迁移时重名概率可忽略且冲突可检测。
- **固定宽度 + 大端**，因此字典序即创建时序，目录列表天然有序。
- **只用小写 base36，不用 base62**。虽然现有字符集 `[a-zA-Z0-9_]` 允许大小写混用，但混用会产生跨后端不一致：PostgreSQL 折叠未加引号的标识符为小写、macOS/Windows 文件系统大小写不敏感（`ws_aB` 与 `ws_Ab` 会碰撞），而 Neo4j 标签大小写敏感（同一对 ID 在 PG 上是一个命名空间、在 Neo4j 上是两个）。base62 只能省 2 个字符，不值这个风险。
- `ws_` 前缀使新建 ID 在结构上**不可能**与保留标识符（`""`、`default`、`_`、以及任何存量 `WORKSPACE` 值）碰撞。

### 3.3 校验：拒绝，不替换

新增 `lightrag/workspace_id.py`：

```python
WORKSPACE_ID_RE = re.compile(r"^ws_[0-9a-z]{16}$")
RESERVED_WORKSPACE_IDS = frozenset({"", "_", "default", "DEFAULT", "Default"})

def new_workspace_id() -> str: ...          # 附录 A 的编码
def validate_workspace_id(value: str) -> str:   # 不合规 → raise，不做 re.sub
def is_generated_workspace_id(value: str) -> bool: ...
```

[config.py:911-919](../../lightrag/api/config.py#L911-L919) 的 `re.sub` 静默替换改为：**存量 `WORKSPACE` 值**保持宽松（沿用替换 + warning，因为它只用于 §9.2 的一次性 bootstrap），**目录里的 ID** 一律走 `validate_workspace_id` 硬拒绝。两条路径的差别写进函数 docstring，并各有一条回归测试钉住。

### 3.4 `workspace_name` 规则

- 允许任意 Unicode（含中文），因为它永不成为物理标识符 —— 与 authz PRD 对 `display_name` 的处理同源。
- 长度 1–128 字符（按码点计），去首尾空白，拒绝纯空白与控制字符。
- 唯一性：**Unicode NFKC 归一化 + casefold 后**在 Server 内唯一，避免"Sales"与"sales"两个下拉项肉眼无法区分。
- 不是标识符：日志与错误信息里出现 `workspace_name` 时必须转义（防日志注入），且**永不**用于构造文件路径、表名、集合名、图标签、key 前缀。

---

## 4. 工作区目录（新增 `WS_CATALOG` 存储类型）

### 4.1 为什么是一个新的存储类型

目录必须满足四条：跨 worker 可见、重启存活、支持 create-if-absent 与基于 revision 的 CAS、以及**服务端可写**（WebUI 创建工作区）。最后一条使它与 authz 的策略文件根本不同（后者服务端只读、从不回写），因此不能照搬那套"磁盘只读 + 字节进共享内存"的传播协议。

裁决：**参照 `DOC_STATUS_STORAGE` 新增第五个存储类型 `WS_CATALOG`**，用可插拔后端承载，默认 JSON 实现。理由：

- 目录的访问形态与 doc_status 高度同构（少量结构化记录、按状态查询、需要跨 worker 一致、需要原子写），可复用同一套抽象与测试骨架；
- 运维已经在为 doc_status 选后端，多一个同构选项的认知成本最低；
- 生产部署（PG/Mongo/Redis）天然获得跨 worker 与跨主机的强一致，不必自建传播协议。

### 4.2 抽象基类

`lightrag/base.py` 新增：

```python
@dataclass
class BaseWorkspaceCatalogStorage(StorageNameSpace, ABC):
    """工作区目录。这是唯一一个**不接受 workspace 参数**的存储家族 ——
    它定义工作区，因此不能被工作区参数化。namespace 固定为
    NameSpace.WS_CATALOG_STORE ("ws_catalog")；workspace 字段恒为 ""。
    """

    @abstractmethod
    async def create(self, record: WorkspaceRecord) -> WorkspaceRecord:
        """原子 create-if-absent。ID / name / ollama_tag 任一冲突 → WorkspaceConflictError。"""

    @abstractmethod
    async def get(self, workspace_id: str) -> WorkspaceRecord | None: ...

    @abstractmethod
    async def get_by_name(self, name_folded: str) -> WorkspaceRecord | None: ...

    @abstractmethod
    async def get_by_ollama_tag(self, tag: str) -> WorkspaceRecord | None: ...

    @abstractmethod
    async def list_all(self, *, include_tombstoned: bool = False) -> list[WorkspaceRecord]: ...

    @abstractmethod
    async def update(self, record: WorkspaceRecord, *, expected_revision: int) -> WorkspaceRecord:
        """revision 不匹配 → WorkspaceRevisionConflictError（CAS 失败），调用方重读后重试。"""
```

注册表（[kg/\_\_init\_\_.py](../../lightrag/kg/__init__.py)）新增：

```python
"WS_CATALOG_STORAGE": {
    "implementations": [
        "JsonWorkspaceCatalogStorage",       # 默认
        "PGWorkspaceCatalogStorage",
        "RedisWorkspaceCatalogStorage",
        "MongoWorkspaceCatalogStorage",
    ],
    "required_methods": ["create", "get", "list_all", "update"],
}
```

配置项 `LIGHTRAG_WS_CATALOG_STORAGE`，默认 `JsonWorkspaceCatalogStorage`，与 [config.py:591-596](../../lightrag/api/config.py#L591-L596) 的约定一致。第一阶段必须交付 JSON 与 PG 两个实现；Redis / Mongo 可随后补，未实现的后端在启动校验里明确报"未实现"而不是回落到 JSON。

`JsonWorkspaceCatalogStorage` 落在 `{working_dir}/ws_catalog.json`（**不在任何工作区子目录下**），跨 worker 一致性复用 `shared_storage` 的命名空间数据 + `get_namespace_lock("ws_catalog")`，写入走临时文件 + `rename` 原子落盘，模式对齐 [json_doc_status_impl.py:135-165](../../lightrag/kg/json_doc_status_impl.py#L135-L165)。

### 4.3 记录结构

```python
@dataclass(frozen=True, slots=True)
class WorkspaceRecord:
    workspace: str                  # ws_xxxxxxxxxxxxxxxx，不可变主键
    workspace_name: str             # 用户可见，可改
    name_folded: str                # NFKC + casefold，唯一性索引用，派生字段
    description: str = ""           # 说明，可改，纯展示
    ollama_tag: str | None = None    # Ollama 路由 tag，可改，Server 内唯一
    lifecycle_state: str = "CREATING"   # CREATING|MIGRATING|ACTIVE|DELETING|ERROR|TOMBSTONED
    legacy_layout: bool = False     # 不可变；True ⇒ 走 §6.3 的历史物理映射
    legacy_physical_workspace: str = ""   # 不可变；legacy_layout=True 时的历史 workspace 值
    schema_version: int = 0         # 存储迁移版本，见 §9.3
    parse_defaults: dict | None = None    # 见 §5.4；None ⇒ 继承全局
    chunk_defaults: dict | None = None    # 见 §5.4；None ⇒ 继承全局
    members: tuple[WorkspaceMember, ...] = ()   # 见 §14.1
    error_detail: str = ""          # lifecycle_state=ERROR 时的脱敏原因
    revision: int = 1               # CAS 用，单调递增
    created_at: float = 0.0
    updated_at: float = 0.0

@dataclass(frozen=True, slots=True)
class WorkspaceMember:
    principal_type: str    # "user" | "api_key"
    principal_id: str      # 与 authz PolicySnapshot 的 principal_id 同一命名空间
```

不可变字段（`workspace`、`legacy_layout`、`legacy_physical_workspace`）在 `update()` 里被显式拒绝修改，不是靠调用方自觉。

### 4.4 目录快照与请求内绑定

与 authz 的 `PolicyRuntime` 同形：

```python
class WorkspaceCatalogRuntime:
    def current(self) -> CatalogSnapshot: ...   # 无锁读，进程内唯一持有者
    async def refresh(self) -> CatalogSnapshot: ...   # 从 WS_CATALOG 存储重读并整体换入
```

- 请求路径**只读 `current()`**（一次属性读，0 次 RPC、0 次 I/O）。这是与 authz PRD §10.4 同级的**不变式**：目录读取绝不出现在数据面请求路径上。
- 刷新时机：(i) 每 worker 一个轮询任务，间隔 `WS_CATALOG_POLL_SECONDS`（默认 5.0）；(ii) 本 worker 自己执行了管理写操作后立即刷新；(iii) 数据面遇到"快照里没有该 ID"时**允许一次**按需刷新后重判（覆盖"另一 worker 刚创建的工作区"，代价是每个未知 ID 最多一次目录读，并受 §14.5 的限流保护）。
- 请求内绑定：路由依赖在第一次解析时把快照写入 `request.state.workspace_snapshot`，同一请求后续所有判定复用它，避免一个请求跨两个目录版本。

---

## 5. 工作区生命周期

### 5.1 状态机

```text
ABSENT --(管理 API 创建)--> CREATING --> MIGRATING --> ACTIVE
                                  \                      |
                                   \--> ERROR            |
                                                         v
                                                     DELETING --> TOMBSTONED
                                                         \--> ERROR
```

- 只有携带 `workspace.create` 权限的管理请求能把 ABSENT 推到 CREATING。
- 数据面请求**只接受 ACTIVE**。`CREATING`/`MIGRATING` → 503 + `Retry-After`；`ERROR` → 409；`DELETING`/`TOMBSTONED` → 404（对调用方而言它已不存在）。
- 初始化失败**绝不发布 ACTIVE**。
- 删除落 TOMBSTONED 并**永久保留 ID**，`create()` 拒绝复用已 tombstone 的 ID（防止"新工作区继承旧数据残渣"）。
- 所有状态跃迁走 `update(..., expected_revision=...)` 的 CAS，失败则重读重试，最多 3 次后返回 409。

### 5.2 创建的前置条件闸门（维护者 Q4 裁决）

`POST /workspaces` 在分配 ID **之前**执行配置合规检查，任一不满足则拒绝创建（HTTP 409 + 明确列出不合规项），**不留下任何记录**：

1. 所有**活跃**后端（即 `LIGHTRAG_{KV,VECTOR,GRAPH,DOC_STATUS,WS_CATALOG}_STORAGE` 实际选中的实现）对应的 `*_WORKSPACE` 环境变量与 `config.ini` 等价字段全部为空。检查的是"活跃后端"而非全部 8 个变量 —— 一个用 PG 的部署不该被没在用的 `NEO4J_WORKSPACE` 挡住。
2. `WS_CATALOG` 存储可写（做一次 no-op CAS 探测）。
3. 目录中 ACTIVE + CREATING + MIGRATING 记录数 < `MAX_ACTIVE_WORKSPACES`（§8.2）。
4. `workspace_name` 通过 §3.4 校验且归一化后不与现存记录冲突；`ollama_tag`（若给）通过 §13.1 校验且不冲突。

这条闸门是 §6.2 启动期检查的**前移**：让运维在"想启用多工作空间"的那一刻就看到配置问题，而不是在下次重启时才发现服务起不来。

### 5.3 创建流程

```text
1. 前置条件闸门（§5.2）           —— 失败即返回，无副作用
2. 分配 ID，写 CREATING 记录       —— create-if-absent
3. 构造 LightRAG 实例（绑定该 ID）
4. initialize_storages()
5. check_and_migrate_data()        —— 新工作区上是空操作；写 schema_version=当前版本
6. 创建输入目录 {INPUT_DIR}/{ws_id}/
7. CAS 到 ACTIVE
```

3–6 任一步失败：记录置 `ERROR` + `error_detail`，**不回滚已创建的命名空间**（部分创建的空命名空间无害，且回滚本身可能失败）；运维可对 ERROR 记录重试创建（幂等，从第 3 步重入）或删除它（走 §5.5 的删除流程清干净）。

**迁移在控制面完成，不在数据面**：这是 #3511 不变式 11 的落地点，也是 PR #3397 被指出的问题 6 的正面回答。

### 5.4 每工作区的解析 / 分块默认值（维护者 Q14 裁决）

`parse_defaults` / `chunk_defaults` 为 `None` 时继承环境变量的全局设置；非 `None` 时逐字段覆盖（**字段级**继承，不是整块替换，避免"设了一个 chunk_size 就丢掉全部其它默认值"）。

允许出现的键限定为一份显式白名单（strict schema，未知键拒绝写入）：

```text
parse_defaults:  parser_engine, parser_extra_suffixes, mineru_page_ranges, ...
chunk_defaults:  process_options, chunk_token_size, chunk_overlap_token_size,
                 chunk_strategy_params...
```

解析优先级（三层，从高到低）：**每请求参数 > 工作区默认值 > 环境变量全局值**。落地点是把工作区默认值注入实例的 `addon_params`，因此 [document_routes.py:2305-2357](../../lightrag/api/routers/document_routes.py#L2305-L2357) 的 `resolve_chunk_options(rag.addon_params, ...)` 调用面**不需要改**。

**明确不含**任何模型面配置（LLM / embedding / rerank / prompt 语言）：embedding 换模型要求清空数据目录（见 `AGENTS.md` 的 pitfall），per-workspace embedding 会让"改一个工作区的配置"变成"隐式作废它的全部向量"；而 LLM 队列按 §11 是 Server 共享的。

### 5.5 删除语义（维护者 Q10 裁决）

```text
1. 前置：该工作区无活跃流水线（pipeline_status.busy / scanning / pending_enqueues 全空）
        否则 409 + 提示先调用 /documents/cancel_pipeline
2. CAS ACTIVE -> DELETING，拒绝新租约（前台与后台）
3. 排空在途租约（有界等待；超时 -> 停在 DELETING，可续做）
4. 对四类存储调用既有 drop()，逐个校验返回 {"status": "success"}
5. 删除 {INPUT_DIR}/{ws_id}/ 与该工作区的解析产物目录
6. CAS -> TOMBSTONED（保留 ID、名字、时间戳作审计）
```

- **共享的摄取侧 LLM 缓存不被删除**（§7）。这是删除路径上唯一的例外，必须有一条测试钉住：删掉工作区 A 之后，工作区 B 的抽取缓存命中率不变。
- 任一 `drop()` 返回 error：停在 `DELETING` + `error_detail`，**绝不**在"只删了 doc_status"的状态下标记完成。运维重试删除是幂等的（`drop()` 对空命名空间返回 success）。
- 不支持 `drop()` 的后端（返回 `"unsupported"`）：整个删除拒绝并明确告知需要手工清理，而不是留一个半删状态。

### 5.6 改名与元数据更新

`PATCH /workspaces/{id}` 可改 `workspace_name`、`description`、`ollama_tag`、`parse_defaults`、`chunk_defaults`、`members`。全部走 CAS。改名与改 tag 立即对新请求生效（下一次目录刷新后全 worker 一致），**不影响任何在途请求**（它们绑定的是 ID）。

---

## 6. 有效工作区规则（四家族一致）

### 6.1 单点解析，向下传递

Server 在构造实例**之前**生成一个不可变绑定：

```python
@dataclass(frozen=True, slots=True)
class WorkspaceBinding:
    workspace: str                 # 目录里的 ID
    catalog_revision: int
    physical_workspace: str        # 传给存储层的值：legacy 记录为历史值，否则 == workspace
    legacy_layout: bool
```

四类存储的构造函数全部接收 `physical_workspace`，**没有任何后端可以自行选择另一个值**。

### 6.2 `*_WORKSPACE` 环境变量的处置（维护者 Q9 裁决：方案 b）

八个变量（`POSTGRES_` / `MONGODB_` / `REDIS_` / `NEO4J_` / `MILVUS_` / `QDRANT_` / `MEMGRAPH_` / `OPENSEARCH_WORKSPACE`，见 [env.example:1262](../../env.example#L1262) 起）与 `config.ini` 等价字段：

| 场景 | 行为 |
| --- | --- |
| 目录中只有 legacy default 一条记录（**单工作空间模式**） | 保持历史优先级，行为逐字不变；启动横幅打印弃用告警 + 移除计划 |
| 目录中存在任何非 legacy 记录（**多工作空间模式**），且任一**活跃**后端的变量非空 | **启动失败**，列出全部冲突的变量名与对应后端（不打印任何连接串或凭据） |
| 创建新工作区时（无论当前几条记录） | §5.2 闸门拒绝创建 |

三条规则合起来的效果：存量单工作空间部署升级后一切照旧；一旦运维尝试进入多工作空间，配置问题在**创建那一刻**就被拦下；而如果有人绕过 API 直接改目录后重启，启动期这道网兜住。

启用变量的部署要迁到多工作空间，路径是：把 `*_WORKSPACE` 的值搬到全局 `WORKSPACE`（若两者一致则直接删除）→ 重启一次确认 legacy 记录的物理名正确 → 再创建新工作区。文档给出 dry-run 命令与备份步骤。

### 6.3 legacy 记录的物理映射（唯一权威口径）

`legacy_layout=True` 的记录（有且仅有一条，见 §9.2）不使用自己的 ID 作物理名，而是使用 `legacy_physical_workspace`，各后端沿用其历史布局：

| 后端 | `legacy_physical_workspace == ""` 时的物理形态 | 依据 |
| --- | --- | --- |
| PostgreSQL | workspace 列值 `"default"`；图名直接用 namespace（不加前缀） | [postgres_impl.py:2702](../../lightrag/kg/postgres_impl.py#L2702)、[:6844](../../lightrag/kg/postgres_impl.py#L6844) |
| Redis | key 前缀 `_` | [redis_impl.py:832](../../lightrag/kg/redis_impl.py#L832) |
| 文件类（Json/NetworkX/Nano/Faiss） | 直接落在 `working_dir` 根目录 | [json_doc_status_impl.py:119-125](../../lightrag/kg/json_doc_status_impl.py#L119-L125) |
| 其它后端 | 各自既有的空 workspace 表现 | 现状 |

`legacy_physical_workspace` 非空（存量 `WORKSPACE=foo`）时，各后端就用 `foo`，与今天完全一致。

**这套映射是目录里的显式元数据，不是各后端各自发明的 fallback。** 这是 #3511 §9.4 的核心主张，也是"零迁移"与"核心身份无歧义"能同时成立的原因。

### 6.4 创建期一致性校验

每个存储对象实现一个无副作用的描述符：

```python
@dataclass(frozen=True, slots=True)
class NamespaceDescriptor:
    storage_family: str        # kv | vector | graph | doc_status
    storage_role: str          # full_docs | text_chunks | llm_cache | entities | ...
    implementation: str
    workspace: str             # 绑定里的 ID
    physical_workspace: str
    physical_fingerprint: str  # 脱敏诊断值：表名/集合名/前缀/目录名，绝不含凭据或连接串
```

实例构造后、`check_and_migrate_data()` 之前，逐个（**不是每家族抽一个代表**）比对：所有对象的 `workspace` 必须等于绑定，`physical_workspace` 必须等于绑定推导出的值。任一不符 → 构造失败，实例进 FAILED，**绝不"修复"成某一个后端的结果**。

`doc_status` 的不一致**永不降级为 warning**：它不是元数据，而是持久化的摄取工作队列（PENDING/PROCESSING 发现、`track_id` 查找、重启恢复都靠它），而文档 ID 是内容哈希 —— 同一份文档在两个工作区拥有相同 ID，doc_status 命名空间一旦塌缩，A 的队列记录会被 B 覆盖。§20 有专门的哨兵测试。

---

## 7. LLM 缓存分层

维护者裁决（Q10）：LLM 缓存跨工作区生效，使同一份文档插入多个工作区时能命中缓存。核实后确认前提成立 —— 抽取缓存 key 只含 chunk 内容 + prompt + 模型身份，与工作区无关（[utils.py:731](../../lightrag/utils.py#L731)、[pipeline.py:6800](../../lightrag/pipeline.py#L6800)）。

**但不能直接共享整个缓存命名空间。** 现状是一个 KV 命名空间靠扁平 key `{mode}:{cache_type}:{hash}` 区分类型（[utils.py:920-931](../../lightrag/utils.py#L920-L931)），其中查询答案缓存的 key 只含 query 与检索参数、**不含被检索的内容**（[operate.py:4535-4548](../../lightrag/operate.py#L4535-L4548)）。整体共享会导致：在 A 库问"公司去年营收多少"得到的答案，被当作 B 库同一问题的答案返回 —— 跨工作区数据泄漏，且悄无声息。

### 7.1 分层判据（唯一权威口径）

> **一条缓存记录可以进共享存储，当且仅当它的 key 完全决定它的值，且这个决定过程不依赖任何工作区的存储内容。**

按此判据分类现有 7 个 `cache_type`：

| cache_type | 输入 | 归属 | 理由 |
| --- | --- | --- | --- |
| `extract` | chunk 内容 + prompt + 模型身份 | **共享** | key 已含全部输入 |
| `analysis` | 多模态条目内容 + prompt + 模型身份 | **共享** | 同上 |
| `smartheading` | 解析期文本 + prompt | **共享** | 同上 |
| `summary` | 待归并的描述列表本身 | **共享** | key 含描述列表；输入不同则 key 天然不同，命中即等价 |
| `keywords` | 用户 query 文本 + 模型身份 | **共享** | 从 query 提关键词，不读任何库内容 |
| `query` | query + 检索参数（**不含被检索内容**） | **每工作区** | key 相同而正确答案不同 —— 判据的反例 |
| `unknown` | 不确定 | **每工作区** | fail-safe 默认 |

### 7.2 落地

`lightrag/namespace.py` 拆成两个：

```python
KV_STORE_LLM_RESPONSE_CACHE = "llm_response_cache"     # 每工作区：query / unknown
KV_STORE_LLM_SHARED_CACHE   = "llm_shared_cache"      # Server 共享：其余五类
```

- 共享缓存实例的 `physical_workspace` 恒为 legacy 布局的空值形态（与 legacy 记录同一物理位置），从而升级后**存量抽取缓存直接被复用**，不需要任何搬迁。
- `handle_cache` / `save_to_cache` 按 `cache_type` 选存储，选择逻辑集中在**一个**函数里（`_cache_kv_for(cache_type)`），新增 cache_type 时必须在此登记，否则落进 fail-safe 的每工作区分支。测试用 golden 断言钉住全部 7 类的归属。
- `/documents/clear_cache` 默认只清**本工作区**的答案缓存。清共享缓存是独立操作（`?scope=shared`），需要 `cache.clear` 权限**且**目录管理权限（`workspace.update`），因为它影响所有工作区。
- 现有工具 `lightrag/tools/{migrate,clean}_llm_cache.py` 需同步支持两个命名空间；它们目前读 `*_WORKSPACE` 变量（[tools/clean_llm_query_cache.py:56-58](../../lightrag/tools/clean_llm_query_cache.py#L56-L58)），须改为读目录。

### 7.3 共享缓存的信息面

共享摄取缓存是一条跨工作区通道，但不构成实际泄漏：命中一条 `extract` 记录要求调用方**已经持有**逐字节相同的 chunk 内容（key 就是它的哈希），返回的是对调用方自己提供的内容的抽取结果。`summary` 同理（key 含描述列表全文）。因此它是"确认预言机"而非读取通道，且预言的对象是调用方自有数据。在 release note 与安全章节写明，不做额外隔离。

---

## 8. 实例池

### 8.1 固定绑定

一个 `LightRAG` 实例由一份不可变的 `WorkspaceBinding` 构造，构造后绑定只读。切换工作区通过**获取另一个实例**实现，永不改写已有实例。这排除了在途请求、异步生成器、延迟嵌入缓冲、后台任务执行到一半时观察到绑定变化的整类竞态（#3511 §26.1 的论证，本方案照采）。

### 8.2 每 worker 的有界池

| 配置 | 默认 | 语义 |
| --- | --- | --- |
| `MAX_ACTIVE_WORKSPACES` | `10` | 每 worker 池内实例上限。**不允许为 0**，即不允许任何内存无界行为（维护者 Q8 裁决） |
| `WORKSPACE_IDLE_TTL_SECONDS` | `1800` | 无租约且空闲超过此时长的实例可被驱逐 |
| `WS_CATALOG_POLL_SECONDS` | `5.0` | 目录刷新轮询间隔 |

池的能力：

- 目录查得 ACTIVE 后才**懒构造**；
- 每 key 单飞（single-flight）：同一 worker 内并发首访只构造一次，全部调用方共享同一结果（含失败）；
- 显式状态：`INITIALIZING` / `READY` / `DRAINING` / `FINALIZING` / `FAILED`；
- **前台租约与后台租约分别计数**（流式响应、后台摄取任务各持自己的）；
- 空闲 LRU 记账；
- 构造失败进有界退避缓存，防止坏工作区造成重试风暴。

不同 worker 加载同一工作区是预期行为。连接成本约为 `workers × 每 worker 已加载工作区数 × 后端客户端成本`，必须进 §18 的可观测指标，让运维看得见。

### 8.3 安全驱逐条件（唯一权威口径）

全部满足才可驱逐：

1. 前台租约计数为 0；
2. 后台 / 流式租约计数为 0；
3. 没有正在进行的初始化、迁移、扫描、流水线、删除、终结；
4. 没有未落盘的缓冲写或可重试的延迟工作（`drop_pending_index_ops` 语义下的 `_pending_*` 缓冲为空）；
5. 空闲时长超过 `WORKSPACE_IDLE_TTL_SECONDS`；
6. 不是被启动策略钉住的 legacy default 记录（它常驻，因为探针与无头请求都落在它上面）。

驱逐先原子地把 `READY` 改成 `DRAINING`（阻止新租约）再终结。终结失败使条目进 `FAILED` 隔离，**不得**让它看起来像"已安全移除"。

池满且无安全 victim：返回 **503** `workspace_capacity_exhausted` + `Retry-After`（不是 429 —— 这不是调用方的错，与 §19 一致）。绝不取消有用工作，绝不超出配置预算。

---

## 9. 启动与迁移

### 9.1 启动序列

```text
1. initialize_share_data(workers, global_concurrency_limits=...)   # §11.1：无条件传上限
2. 初始化 WS_CATALOG 存储
3. 目录 bootstrap 事务（§9.2）—— 幂等；失败则**立即退出进程**
4. legacy default 工作区：initialize_storages() + check_and_migrate_data()
        —— 与今天的 lifespan 逐字同构（lightrag_server.py:1348-1380）；失败则退出
5. 非 legacy 的 ACTIVE 记录中 schema_version 落后的：排入有界后台迁移（§9.3）
6. 准入控制探测（每工作区，见 §11.3）
7. Server ready
```

第 3、4 步失败即退出进程，是维护者对"发现 2"的裁决：**接受一个坏的默认工作区让整个 Server 起不来**。理由是它与今天的行为一致（`check_and_migrate_data()` 抛错今天就会让 lifespan 失败），且默认工作区不可用时 Server 本身没有意义。

### 9.2 目录 bootstrap（一次性，幂等）

首次以本版本启动时：

```text
若目录为空：
    生成 ID，写入唯一一条 legacy 记录：
        workspace                  = new_workspace_id()
        workspace_name             = WORKSPACE 值 或 "Default"
        legacy_layout              = True
        legacy_physical_workspace  = WORKSPACE 的清理后值（可能为 ""）
        ollama_tag                 = 现有 f"{LIGHTRAG_NAME}:{LIGHTRAG_TAG}"
        lifecycle_state            = ACTIVE
        schema_version             = 当前版本
若目录非空：
    校验 legacy 记录的 legacy_physical_workspace 是否仍等于当前 WORKSPACE 配置
    不等 ⇒ 启动失败并给出诊断（绝不静默重映射）
```

- **幂等**：以"目录为空"为唯一触发条件，重复启动不产生第二条记录；写入用 create-if-absent，两个 worker 竞争时只有一个成功、另一个读到已存在记录后继续。
- 出错立即退出，下次启动自动重入（因为目录仍为空或仍不一致）。
- bootstrap 之后，`WORKSPACE` 环境变量**只用于上面的一致性校验**，不再参与任何路由；启动横幅打印"工作区身份已由目录接管"。

### 9.3 存储迁移的时机与所有权

| 类别 | 迁移时机 |
| --- | --- |
| 目录自身的 schema | 启动期，幂等，唯一 leader（§9.2） |
| legacy default 工作区的存量数据 | 启动期，硬闸门（§9.1 第 4 步），与今天一致 |
| 新建工作区 | 创建流程内，ACTIVE 之前（§5.3 第 5 步） |
| 软件升级后落后的既有非 legacy 工作区 | 启动期排入**有界后台**迁移，记录置 `MIGRATING`、数据面返回 503；单条失败置 `ERROR` 且**不影响 Server 与其它工作区** |
| 失败后重试 | 显式管理操作（`POST /workspaces/{id}/migrate`），永不作为查询/上传的副作用 |

最后两行是本 PRD 在维护者答复之外补齐的一环：Q7 的答复只覆盖"目录 bootstrap"，但软件升级会给存量的**非默认**工作区带来新的迁移步骤。把它放在启动期有界后台 + 每记录失败隔离，同时满足三条约束：数据面请求永不成为迁移 owner（#3511 不变式 11）、启动不被非默认工作区拖慢、单个非默认工作区的故障不牵连全局。

多 worker 下迁移所有权用 `get_namespace_lock(f"ws_migrate:{ws_id}")` + 目录里的 `schema_version` CAS：只有持锁者能推进版本，其它 worker 观察目录状态后跳过。迁移代码必须幂等且能区分未开始 / 进行中 / 成功 / 失败。

### 9.4 实例首次加载不做迁移

因为迁移已在控制面完成，数据面首访一个 ACTIVE 且 `schema_version` 为当前值的工作区时**只做 `initialize_storages()`**（建连接、载入内存结构），不调用 `check_and_migrate_data()`。首访延迟由此从"迁移量级"降到"建连接量级"。

---

## 10. 流水线与 doc_status 隔离

### 10.1 全链路显式工作区

以下每一处都必须携带具体的工作区绑定，**没有 `workspace=None`，没有回落到进程级默认值**：

入队与去重、`pipeline_status` 与全部命名空间锁、`pipeline_ingress` 三通道邮箱、`track_id` 创建与查询、输入目录选择与扫描、PENDING/PROCESSING/FAILED 查询、解析与多模态、抽取/嵌入/图更新/缓存写、清空与删除等破坏性作业、重试与恢复。

`AGENTS.md` 里已定义的流水线并发契约（`busy` / `destructive_busy` / `scanning` / `scanning_exclusive` / `pending_enqueues`、`get_namespace_lock("pipeline_status", workspace=...)`、ingress 三通道与静默点决策顺序）**按工作区各自成立一套**，语义逐字不变。这是本 PRD 对现有契约的唯一要求：把"每 workspace 一份"从事实变成不变式，并加测试钉住两个工作区的流水线互不可见。

不是天然按物理分区的持久化 key 必须含工作区 ID，例如 `(workspace, track_id)`。

### 10.2 输入目录（维护者 Q15 裁决）

```text
{INPUT_DIR}/{ws_id}/        普通工作区
{INPUT_DIR}/                legacy 记录（与文件类存储的 legacy 布局同构）
```

`POST /documents/scan` 变为工作区粒度：扫描且只扫描当前工作区的输入根。**不存在**跨工作区的全局扫描 —— 一次全局扫描要么把 A 的文件分类进 B 的 doc_status，要么需要一张跨库对照表，两者都不可接受。若将来需要"扫描全部工作区"，它只能是一个枚举目录记录、逐个发起工作区级扫描的管理协调器。

上传路径的 `sanitize_filename` / `upload_file_opener` 的目录 fd 约束（[document_routes.py:189-277](../../lightrag/api/routers/document_routes.py#L189-L277)）改为相对该工作区的输入根，防路径穿越语义不变。

### 10.3 同文档双工作区

两个工作区上传相同内容 → 相同文档 ID（内容哈希）→ **各自独立的 doc_status 记录与流水线生命周期**。一个工作区里的处理、删除、重试、重启恢复不得观察或覆写另一个的记录。解析与分块选项按 §5.4 各自解析，因此同一份文档在两个工作区里可以有完全不同的 chunk 集合与实体抽取结果。

摄取侧 LLM 缓存按 §7 共享，因此**当两个工作区的分块结果恰好相同时**才会命中；分块不同时 key 天然不同，各跑一次抽取。这正是判据（§7.1）的正确行为。

### 10.4 重启恢复：纯手工（维护者 item 9 裁决）

- 启动期**不**枚举工作区、**不**扫描 doc_status、**不**恢复任何流水线。这与今天的语义一致（[lightrag_server.py:1348-1380](../../lightrag/api/lightrag_server.py#L1348-L1380) 无任何启动期 scan）。
- 重启后有未完成文档的工作区，由运维显式发 `POST /documents/scan`（发现文件 + 重置 FAILED）或 `POST /documents/reprocess_failed`（纯存储驱动）来恢复。
- 文档与 release note 必须写明：**多工作空间部署里，非默认工作区的未完成文档在重启后不会自动恢复**。`/status`（§12.3）必须把每个工作区的 PENDING / FAILED 计数暴露出来，让运维看得见需要恢复什么 —— 手工恢复策略只有在"看得见"时才可运维。

---

## 11. 资源治理

### 11.1 LLM / VLM / embedding / rerank：Server 级共享（维护者 Q1 裁决：方案 a）

现有跨 worker 闸门的 `concurrency_group` 与工作区无关（`f"llm:{role}"` / `"embedding"` / `"rerank"`，[lightrag.py:1342](../../lightrag/lightrag.py#L1342)、[:1372](../../lightrag/lightrag.py#L1372)），直接复用。**唯一改动**在注册侧：

```python
# run_with_gunicorn.py:282-294 —— 现状
if workers_count > 1:
    initialize_share_data(workers_count, global_concurrency_limits=...)
else:
    initialize_share_data(1)                    # ← 不注册任何上限
```

改为**无条件**传入 `global_concurrency_limits`，uvicorn 单进程入口（`lightrag_server.py`）同样如此。原注释"Single-worker mode needs no cross-process gate — the per-process max_async already IS the total limit there"的前提是"一进程一实例"，多工作空间下失效：N 个实例各建一份本地队列，总并发变成 N × max_async —— 正是 PR #3397 评审问题 5 的同一个坑换到单进程模式。单进程下 slot gate 走的是本地 dict，无 IPC，成本可忽略。

必须有一条行为级测试：**加载 N 个工作区后并发观测到的 provider 调用数不超过配置总量**，在单 worker 与多 worker 两种模式下各断言一次。仅断言"注册了上限"是不够的。

按角色的队列（`EXTRACT_LLM` 等）因此天然是全 Server 共享：所有工作区的实体关系抽取排在同一条 `llm:extract` 队列上。

### 11.2 流水线并发

`max_parallel_insert` 保持每实例语义（流水线 worker 绑定在自己工作区的存储上）。真正的资源瓶颈由 §11.1 的全局队列兜住，因此**本阶段不引入"同时活跃流水线数"的全局上限**。代价写进 §23：N 个工作区同时批量摄取时，队列等待会拉长，且不存在跨工作区公平性（一个工作区的批量摄取可以长时间占满 `llm:extract`）。公平调度（WRR/DRR + 优先级老化）留待后续阶段。

### 11.3 准入控制（维护者 Q15 裁决）

`MAX_PENDING_DOCUMENTS` 保持**每工作区**语义（计数走各自的 `doc_status`）。它是背压而不是资源上限，per-workspace 语义更直观。文档必须写明：**总量是 `N × MAX_PENDING_DOCUMENTS`**。

[lightrag_server.py:1362-1379](../../lightrag/api/lightrag_server.py#L1362-L1379) 的启动期严格计数探测按工作区各做一次 —— legacy 记录在启动期做，其余在创建时与迁移完成后做，避免"某个后端不支持严格计数"变成该工作区每次上传都 503。

---

## 12. API 契约

### 12.1 选择器（维护者 Q13 裁决）

```http
LIGHTRAG-WORKSPACE: ws_0mkq3f1x7d9b2v4c
```

- **只接受工作区 ID**，不接受 `workspace_name`。名字可改，接受名字会让"改个名"变成"把别人的写入路由到别处"。
- 语义按四种状态严格区分（框架层必须保留"缺失"与"存在但为空"的差别）：

| 选择器状态 | 结果 |
| --- | --- |
| 头缺失 | 选择 legacy default 记录（零迁移的落地点） |
| 头存在但为空 / 纯空白 | **400**，不回落。客户端存在 bug，静默落到默认工作区是比 400 更糟的数据风险 |
| 语法非法（不匹配 §3.2 正则） | **400** |
| 语法合法但目录中不存在 | **404**，不创建、不实例化、不建目录 |
| 记录非 ACTIVE | 按 §5.1：503 / 409 / 404 |

- 成功的数据面响应回显解析后的工作区 ID（响应头 `LIGHTRAG-WORKSPACE`），便于审计与前端校验。

### 12.2 端点实例化策略（唯一权威口径）

每条路由必须归入且只归入一类，**新增路由未归类则启动期审计失败**（复用 authz PRD §6 的 `audit_route_coverage` 机制，golden 清单按 profile 各一份）。

| 类别 | 目录查询 | 可加载实例 | 可创建记录/命名空间 | 可迁移 | 端点 |
| --- | --- | --- | --- | --- | --- |
| 存活探针 | 不需要 | ❌ | ❌ | ❌ | `GET /health` |
| 服务状态 | 只读目录 | ❌ | ❌ | ❌ | `GET /status` |
| 工作区管理（读） | 只读目录 | ❌ | ❌ | ❌ | `GET /workspaces`、`GET /workspaces/{id}` |
| 工作区管理（创建） | 显式目标 | ✅ | ✅ | ✅（ACTIVE 之前） | `POST /workspaces` |
| 工作区管理（更新） | 只读+CAS | ❌ | ❌ | ❌ | `PATCH /workspaces/{id}` |
| 工作区管理（删除） | 显式生命周期 | ✅（维护实例） | ❌ | 仅清理 | `DELETE /workspaces/{id}` |
| 工作区管理（迁移重试） | 显式生命周期 | ✅ | ❌ | ✅ | `POST /workspaces/{id}/migrate` |
| 数据读 | 解析 ACTIVE | ✅ | ❌ | ❌ | `/query/*`、`/documents`（列表/状态/计数/track）、`/graph*`、缓存读 |
| 数据写 | 解析 ACTIVE | ✅ | ❌ | ❌ | `/documents/upload\|text\|texts\|scan\|delete_document`、`DELETE /documents`、`/graph/*` 变更、`/documents/clear_cache` |
| Ollama 推理 | 解析 ACTIVE（按 tag） | ✅ | ❌ | ❌ | `/api/generate`、`/api/chat` |
| Ollama 元数据 | 只读目录 | ❌ | ❌ | ❌ | `/api/tags`、`/api/version`、`/api/ps` |

`/health` 与 `/api/tags` 这两行是硬约束：探针每 10–30 秒一次，元数据端点会枚举全部工作区 —— 任何一个能触发实例化，就等于"看一眼状态"变成"建 N 套连接 + 跑 N 次迁移"，这正是 PR #3397 评审问题 3 的内容。

### 12.3 `/health` 与 `/status`（维护者 Q12 裁决）

| 端点 | 认证 | 权限 | 内容 |
| --- | --- | --- | --- |
| `GET /health` | 无 | — | **纯存活**：`status`、core/api 版本、WebUI 可用性。不读目录、不碰池、不看选择器 |
| `GET /status` | 需要 | `system.health.read` | Server 级配置（LLM/embedding/存储后端/队列/keyed-lock）+ **全工作区汇总**：每条记录的 ID、名字、`lifecycle_state`、`schema_version`、是否已加载、流水线忙闲、PENDING/PROCESSING/FAILED 计数 |
| `GET /documents/pipeline_status` | 需要 | `pipeline.read` | 不变，仍是**当前工作区**的流水线细节 |

`/status` 的全工作区汇总**只读目录 + `pipeline_status` 共享命名空间**，不加载任何实例（未加载的工作区报 `loaded: false`）。它是 §10.4 手工恢复策略的可运维前提。

`/health` 因此退化为纯探针，不再有"认证与否决定返回内容"的双重身份 —— 这需要同步修改 authz PRD §5.1 与 §5.2，见**附录 B**。

### 12.4 工作区管理端点

```text
GET    /workspaces                 列表（可见范围见 §14.3）
POST   /workspaces                 创建（§5.2 闸门 + §5.3 流程）
GET    /workspaces/{id}            详情
PATCH  /workspaces/{id}            改名 / 说明 / tag / 解析分块默认值 / 成员
DELETE /workspaces/{id}            删除（§5.5）
POST   /workspaces/{id}/migrate    迁移重试（仅 ERROR / schema 落后的记录）
```

- 管理端点在**路径**里指定目标，**不读** `LIGHTRAG-WORKSPACE` 头（避免"改 A 的名字时头指向 B"这类歧义）。
- `POST /workspaces` 支持幂等键（`Idempotency-Key` 头）：重复请求返回原操作结果；载荷冲突返回 409。
- 创建成功返回 201 + 完整记录（含分配的 ID）。

---

## 13. Ollama 兼容面（维护者 Q13 裁决）

### 13.1 tag 路由

- `ollama_tag` 是**目录里独立可编辑的字段**，不随 `workspace_name` 改名而变动（客户端配置的是 tag，改名不该打断它）。
- 格式 `name:tag`，`name` 与 `tag` 各自匹配 `^[A-Za-z0-9._-]{1,64}$`；Server 内唯一（大小写不敏感比较）。
- legacy 记录的 tag 由 bootstrap 写成现有 `f"{LIGHTRAG_NAME}:{LIGHTRAG_TAG}"`（来自 `--simulated-model-name` / `--simulated-model-tag`，[config.py:907-908](../../lightrag/api/config.py#L907-L908)），因此存量 Ollama 客户端配置不变即可继续工作。
- `/api/generate`、`/api/chat` 从请求体的 `model` 字段查目录 → 得到工作区 ID → 走与 REST 相同的实例获取路径。`LIGHTRAG-WORKSPACE` 头在这两个端点上被**忽略**（Ollama 客户端设不了自定义头，同时接受两个选择器只会制造冲突歧义）。

### 13.2 未知 model 的处置

现状：**任何 model 名都被接受并忽略**（[ollama_api.py:363-372](../../lightrag/api/routers/ollama_api.py#L363-L372) 只回显 `LIGHTRAG_MODEL`）。改为：未知 tag 返回 Ollama 兼容的 **404**（`{"error": "model '<name>' not found"}`），不创建、不回落到默认工作区。

这是破坏性变更（§17 #4）：今天写错 model 名的客户端能正常工作，改后会 404。选择 404 是因为静默回落到默认工作区正是 item 2 fail-closed 要排除的数据风险 —— 一个写错 tag 的批量摄取会把数据灌进错误的知识库。

### 13.3 元数据端点

- `/api/tags` 列出**调用者有权访问的**每个 ACTIVE 工作区各一个条目（`name` / `model` 用 `ollama_tag`，其余字段沿用现有取值）。只读目录，不加载实例。
- `/api/version` 不变。
- `/api/ps` 列出**已加载**的工作区（`loaded: true` 的池条目），仍不加载任何实例。

---

## 14. 授权（维护者 Q11 / Q12 裁决）

### 14.1 两层模型

| 层 | 权威 | 表达什么 | 谁维护 |
| --- | --- | --- | --- |
| 动作权限 | authz 策略文件（v1 **不变**） | 能做什么（`documents.write`、`graph.delete`…） | 运维手编 YAML + 热重载 |
| 工作区成员 | **工作区目录** `members` 字段 | 能进哪些工作区（布尔） | WebUI / 管理 API |

选择这个切分而不是把 workspace 塞进策略文件，理由有三：

1. 策略文件按 authz PRD §1.2 是**服务端只读、从不回写**的，而工作区由 WebUI 创建 —— 让手编 YAML 去引用 UI 生成的 `ws_` ID，运维体感差，且会与规则 #6 的"无悬空绑定"校验打架（策略文件引用了尚未创建或已删除的工作区 ID 时该拒绝加载还是忽略？两个答案都难看）。
2. authz v1 规则 #4 明确拒绝含 `scope`/`workspace`/`tenant` 字段的策略文件。走目录路线**完全不需要动 v1 的加载器**，两个 PRD 解耦。
3. 成员表用**布尔**而非"每工作区角色"：动作粒度已由 18 个权限码表达完整，再叠一层每工作区角色就是第二套授权引擎 —— 正是 authz PRD §10.1 明令禁止的形态。

### 14.2 有效权限（唯一权威口径）

```text
允许访问工作区 W 上的动作 A
  ⟺  principal 在 PolicySnapshot.effective 里持有 A 所需的全部权限码
      ∧  ( W 的成员表包含该 principal  ∨  principal 持有 workspace.update 或 workspace.delete )
```

即**动作权限 ∩ 工作区可达性**。第二项的析取分支让工作区管理员无需把自己加进每个工作区的成员表就能运维它们。

落地形态：`authorize` 依赖在完成现有判定后追加一段工作区判定，读的是 `request.state.workspace_snapshot`（§4.4 的请求内绑定），复杂度 O(1)（成员表在快照构建时预计算成 `frozenset[(pt, pid)]`）。**请求路径仍是 0 次跨进程 RPC、0 次文件 I/O** —— authz PRD §10.4 的不变式与它的四条禁令原样适用于目录快照。

`AuthorizationContext(scope=ResourceScope("workspace", ws_id))` 按 authz PRD §4.2 的预留形态填充并传入，使将来切到 DB provider 时接口不变。

### 14.3 可见范围

- `GET /workspaces`：持有 `workspace.update` 或 `workspace.delete`（管理者）→ 返回**全部**记录；否则只返回成员表包含自己的 ACTIVE 记录。这是对"管理员需要看全部才能授权、普通用户只该看到自己的"的解析。
- `GET /status` 的全工作区汇总同上按可见范围过滤。
- `/api/tags` 只列可访问的（§13.3）。
- 数据面拿到一个"存在但自己不是成员"的 ID：返回 **404**，不是 403。返回 403 会把"这个 ID 存在"泄漏给非成员，而 ID 虽非机密也没有理由外泄。

### 14.4 权限码

authz 权限目录新增 4 个：

```python
WORKSPACE_READ   = "workspace.read"      # GET /workspaces、/workspaces/{id}
WORKSPACE_CREATE = "workspace.create"    # POST /workspaces
WORKSPACE_UPDATE = "workspace.update"    # PATCH（含改成员表）、POST /{id}/migrate
WORKSPACE_DELETE = "workspace.delete"    # DELETE /workspaces/{id}
```

`workspace.delete` 单独成码，遵循 authz PRD"破坏性操作权限面更窄"的原则。

按 authz PRD §11.4 约束 2（权限集冻结、新权限码对旧配置 deny-by-default），这四个码**不进入** `LEGACY_USER_PERMISSIONS` 的 18 项冻结列表。后果是明确的：**未启用 policy 模式的部署无法创建或删除工作区**，只能使用 bootstrap 出来的那一条 legacy 记录，行为与今天完全一致。这与 `documents.artifacts.*` 的处置同源，同时也是迁移到 policy 模式的正向激励。

### 14.5 安全边界

- 工作区隔离**不是**授权：`LIGHTRAG-WORKSPACE` 头与 Ollama tag 都是不可信的路由输入。真正的边界是 §14.2 的成员表判定。
- legacy profile（无策略文件）下 `legacy_user` 对**所有**工作区可达（否则存量部署升级即失去数据）；但因 §14.4 它只能有一个工作区，所以这条在实践中是恒真的。
- 未知选择器探测与创建请求受限流保护（复用登录限流器的形态，按 principal + IP 预占）。
- 错误响应不得泄漏连接串、表名、目录路径；`physical_fingerprint`（§6.4）只进结构化日志，不进 HTTP 响应体。
- 工作区 ID 不透明但**不是机密**。
- 结构化日志的授权字段增加 `workspace`（ID，非名字）与 `catalog_revision`。

---

## 15. WebUI（维护者 Q16 裁决）

### 15.1 布局改动

| 位置 | 现状 | 改为 |
| --- | --- | --- |
| 左上角品牌区 | 渲染 `webuiTitle` / `webuiDescription`（[SiteHeader.tsx:83-96](../../lightrag_webui/src/features/SiteHeader.tsx#L83-L96)） | **工作区切换下拉框**；`WEBUI_TITLE` / `WEBUI_DESCRIPTION` 取代现在硬编码的 "LightRAG" 品牌文字位置 |
| TAB 栏第 4 项 | `api`（API 说明页） | **`workspaces`（工作区）**，用于创建与管理工作区 |
| 右上角工具栏 | 主题 / 语言 / 版本 / 登出 | 增加 **API 说明入口**（原 API TAB 的内容，以外链或抽屉呈现） |

### 15.2 切换器语义

- 切换器决定 **`documents` / `knowledge-graph` / `retrieval` 三个 TAB** 的工作区（知识图谱 TAB 同样跟随 —— 图数据本就按工作区隔离）。
- 选中项持久化到 `localStorage`（并入既有的 `settings-storage` 持久化 key，schema 在 `lightrag_webui/src/stores/settings.ts`）。
- 持久化的工作区已被删除或已不可访问：**静默回落**到第一个可访问的工作区，不提示。
- 前端把选中的 ID 放进每个数据面请求的 `LIGHTRAG-WORKSPACE` 头（在 `lightrag_webui/src/api/lightrag.ts` 的统一请求层注入，不逐个端点改）。
- 切换工作区时必须清空三个 TAB 的本地状态（文档列表、图视图、检索历史），避免上一个工作区的数据残留在界面上。

### 15.3 零可访问工作区

一个用户可访问 0 个工作区时，`documents` / `knowledge-graph` / `retrieval` 三个 TAB **各自空白，居中显示原因**（"当前账号未被授权访问任何工作区，请联系管理员"）。不弹错、不跳登录页、不循环重试。`workspaces` TAB 仍可打开（若持有 `workspace.read`）。

### 15.4 工作区 TAB

- 列出**有权访问的**工作区；持有管理权限（`workspace.update` / `workspace.delete`）时列出全部（与 §14.3 的后端可见范围一致）。
- 创建表单：名字、说明、Ollama tag（可留空）、解析/分块默认值（折叠的高级区，留空即继承全局）。
- 每条记录显示 ID（可复制）、`lifecycle_state`、创建/更新时间、成员管理入口。
- 创建被 §5.2 闸门拒绝时，把返回的不合规项逐条展示（例如"检测到 POSTGRES_WORKSPACE 环境变量，请先移除"），而不是只显示一个 409。
- 删除需二次确认并显式提示"将永久删除该工作区的全部数据"。

### 15.5 状态码分流

沿用 authz PRD §12 的四路分流（只有 401 清 token），并补两条工作区特有的：

- **404 + 工作区上下文** ⇒ 当前选中的工作区不再可用 → 静默回落到第一个可访问的工作区并重试一次。
- **503 + `Retry-After`** ⇒ 工作区正在初始化/迁移，或池容量耗尽 → 提示"工作区正在准备中"，按 `Retry-After` 自动重试一次，**不清 token**。

---

## 16. 配置项汇总

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `LIGHTRAG_WS_CATALOG_STORAGE` | `JsonWorkspaceCatalogStorage` | 工作区目录后端（§4.2） |
| `MAX_ACTIVE_WORKSPACES` | `10` | 每 worker 实例池上限，**不接受 0** |
| `WORKSPACE_IDLE_TTL_SECONDS` | `1800` | 空闲驱逐阈值 |
| `WS_CATALOG_POLL_SECONDS` | `5.0` | 目录刷新轮询间隔（≥1.0） |
| `WORKSPACE` | 空 | **仅**用于首次 bootstrap 决定 legacy 记录的物理名，之后只作一致性校验（§9.2） |
| `POSTGRES_WORKSPACE` 等 8 个 | 空 | **弃用**。多工作空间模式下非空即启动失败；单工作空间模式告警（§6.2） |
| `MAX_PENDING_DOCUMENTS` | 沿用 | 每工作区语义，总量为 N× （§11.3） |
| `INPUT_DIR` | 沿用 | 下挂 `{ws_id}/` 子目录（§10.2） |
| `--simulated-model-name` / `--simulated-model-tag` | 沿用 | 仅决定 legacy 记录的初始 `ollama_tag`（§13.1） |

新增权限码 4 个（§14.4）。不新增任何"是否启用多工作空间"的开关（维护者 Q4 裁决）：功能常开，目录在首次启动时自动建立，单工作空间部署就是"目录里只有 1 条 legacy 记录"。

---

## 17. 破坏性变更清单

1. **`*_WORKSPACE` 环境变量弃用**。多工作空间模式下非空即启动失败；单工作空间模式仅告警（§6.2）。官方 K8s chart 的 `POSTGRES_WORKSPACE: default`（[values.yaml:75](../../k8s-deploy/lightrag/values.yaml#L75)）必须同 PR 移除。
2. **`/health` 不再返回完整配置**，完整信息移到需要 `system.health.read` 的 `/status`（§12.3）。依赖 `/health` 抓配置的监控需改端点。
3. **workspace 字符集校验从静默替换改为拒绝**（仅对目录 ID 生效；存量 `WORKSPACE` 值仍宽松，§3.3）。
4. **Ollama 未知 model 名从"接受并忽略"改为 404**（§13.2）。
5. **LLM 缓存拆成两个命名空间**（§7.2）。存量抽取缓存因共享存储沿用 legacy 物理位置而**直接可用**；但存量查询答案缓存会因命名空间迁移而失效一次（重算即恢复，无数据损失）。
6. **`/documents/scan` 变为工作区粒度**，输入目录下挂 `{ws_id}/`（§10.2）。既有部署的 legacy 记录仍用 `INPUT_DIR` 根，行为不变。
7. **WebUI 的 API TAB 被工作区 TAB 取代**，API 说明移到右上角工具栏（§15.1）。
8. 未启用 policy 模式的部署**无法创建工作区**（§14.4）。这不是行为变更（新能力 deny-by-default），但需在 release note 写明。

除第 2、4、7 条外，其余对"从未设置 `*_WORKSPACE`、不创建新工作区"的存量部署都是无感的。

---

## 18. 可观测性

必须暴露的信号：

- 目录：记录数按 `lifecycle_state` 分布、`revision`、刷新延迟、CAS 冲突次数；
- 实例池：每 worker 的条目数与状态分布、前台/后台租约数、空闲时长、驱逐次数、`workspace_capacity_exhausted` 计数、构造失败与退避状态；
- 一致性：按存储家族与实现统计的有效工作区校验失败数（§6.4）；
- 迁移：队列长度、耗时、owner、重试与失败状态、`schema_version` 分布；
- 资源：每个 `concurrency_group` 的配置上限 vs 实际观测并发、等待时长、拒绝数、队列深度（§11.1 的验收依据）；
- 流水线：每工作区的 PENDING / PROCESSING / FAILED 计数（§10.4 手工恢复的可运维前提）；
- 缓存：共享缓存与每工作区缓存的命中率分开统计（验证 §7 的收益）。

**指标标签必须控制工作区维度的基数**：详细 ID 进结构化日志与 trace，聚合指标用有界标签（例如只按 `lifecycle_state` 与 `loaded` 分组）。`/health` 与 `/status` 的处理函数只读快照，永不初始化工作区。

---

## 19. 失败语义（唯一权威口径）

| 情形 | 状态码 | 说明 |
| --- | --- | --- |
| 头缺失 | — | 选 legacy default 记录 |
| 头存在但为空 / 语法非法 | **400** | 不回落 |
| 目录中不存在该 ID | **404** | 无任何副作用 |
| ID 存在但 principal 非成员且非管理者 | **404** | 不用 403，避免泄漏 ID 存在性（§14.3） |
| 记录 `CREATING` / `MIGRATING` | **503** + `Retry-After` | 稳定错误码 `workspace_not_ready` |
| 记录 `ERROR` | **409** | 稳定错误码 `workspace_error`，带脱敏 `error_detail` |
| 记录 `DELETING` / `TOMBSTONED` | **404** | 对调用方而言已不存在 |
| 池满且无安全 victim | **503** + `Retry-After` | `workspace_capacity_exhausted`，不是 429（非调用方之错） |
| 创建时配置不合规（含 `*_WORKSPACE`） | **409** | 逐条列出不合规项，无副作用 |
| 创建时名字 / tag 冲突 | **409** | `workspace_name_conflict` / `ollama_tag_conflict` |
| CAS 冲突（并发管理写） | **409** | 重试 3 次后返回，提示重读 |
| 删除时有活跃流水线 | **409** | 提示先 `cancel_pipeline` |
| 删除中某个 `drop()` 失败 | **500** | 记录停在 `DELETING` + `error_detail`，可幂等重试 |
| 多工作空间模式下检出 `*_WORKSPACE` | 启动失败 | 列出全部冲突变量与后端，不打印凭据 |
| 四家族有效工作区不一致 | 构造失败 | 在迁移与数据访问之前，实例进 `FAILED` |
| 目录 bootstrap 失败 / legacy 记录与配置不符 | 启动失败 | 立即退出，下次启动重入（§9.2） |
| 非默认工作区后台迁移失败 | — | 该记录 `ERROR`，Server 与其它工作区不受影响 |
| Ollama 未知 tag | **404** | Ollama 兼容错误体 |
| 缺工作区上下文的内部调用 | 类型化内部错误 | 绝不回落到默认工作区 |

---

## 20. 测试计划

放置位置遵循仓库约定：`tests/workspace/` 为主，跨层的按 `AGENTS.md` 的映射放到对应子目录。

**身份与校验**（`tests/workspace/test_workspace_id.py`）
- `new_workspace_id()` 恒匹配 `^ws_[0-9a-z]{16}$`；10 万次生成无重复；字典序等于生成序。
- `validate_workspace_id` 对保留值（`""`、`_`、`default`、大小写变体）与非法字符**抛错**而不是替换 —— fix-proof：断言"不存在任何输入被 `re.sub` 改写后通过校验"。
- 附录 A 的长度预算：对每个后端断言派生出的最长物理名不超过该后端上限。

**目录**（`tests/workspace/test_ws_catalog_*.py`，JSON 与 PG 各一套）
- `create` 的 create-if-absent 语义；ID / name / tag 三种冲突各返回专属错误。
- `update` 的 CAS：`expected_revision` 过期时拒绝；并发写不丢更新。
- 不可变字段（`workspace`、`legacy_layout`、`legacy_physical_workspace`）被拒绝修改。
- `name_folded` 唯一性覆盖 NFKC + casefold（"Sales" / "sales" / 全角变体互相冲突）。
- tombstoned ID 不可复用。
- 多 worker：两个进程并发 bootstrap 只产生一条 legacy 记录。

**有效工作区一致性**（`tests/workspace/test_effective_workspace.py`）
- 对每个后端与每种混合家族组合：断言全部存储对象报告同一 `workspace` 与 `physical_workspace`。
- 逐个注入 8 个 `*_WORKSPACE` 变量：多工作空间模式启动失败且错误信息列出变量名与后端；单工作空间模式仅告警且行为不变。
- 混合不一致（只给 doc_status 设覆盖）在数据访问**之前**失败。
- `legacy_layout` 记录在 PG / Redis / 文件类上落到 §6.3 表格里的物理位置（对着表格逐行断言）。
- 新建 ID 结构上不可能与 `default` / `_` / `""` 碰撞。

**doc_status 哨兵**（`tests/workspace/test_doc_status_sentinel.py`）
- 向工作区 A 与 B 插入**完全相同**的内容（因此同一文档 ID），独立驱动 PENDING → PROCESSING → PROCESSED、FAILED、重试、`track_id` 查询、删除、重启恢复。断言每一次状态读写与由此产生的 KV / 向量 / 图变更都留在各自工作区。
- A 的 `/documents/clear` 不影响 B 的任何记录。
- A 的删除不影响 B 的抽取缓存命中率（§5.5 的例外钉子）。

**LLM 缓存分层**（`tests/workspace/test_cache_partitioning.py`）
- golden 断言全部 7 个 `cache_type` 的归属（字面列表，新增类型必须显式登记）。
- **反例钉子**：同一 query 在 A 与 B 上返回**不同**答案（若查询缓存被误共享则此测试失败）。
- 同一份文档以相同分块插入 A 后再插入 B，B 的抽取 LLM 调用次数为 0。
- 分块选项不同时，B 的抽取正常发生（不误命中）。
- 升级路径：存量单工作空间的抽取缓存在拆分后仍被命中。

**实例池与租约**（`tests/workspace/test_workspace_pool.py`，用 barrier/event，不用 sleep）
- 同一 worker 内并发首访只构造一次（single-flight），失败也只失败一次且所有调用方拿到同一异常。
- 流式响应 / 后台任务持租约期间实例不可驱逐、不可删除。
- 池满且全部有租约 → 503 `workspace_capacity_exhausted`。
- 取消请求恰好释放一次租约与一次准入令牌（不多不少）。
- `MAX_ACTIVE_WORKSPACES=0` 被配置校验拒绝。
- 空闲 TTL 到期后驱逐；legacy 记录常驻不被驱逐。

**启动与迁移**（`tests/workspace/test_startup_migration.py`）
- bootstrap 幂等：连续启动 3 次仍只有一条 legacy 记录。
- bootstrap 失败 → 进程退出；修复后下次启动自动完成。
- legacy 记录的 `legacy_physical_workspace` 与当前 `WORKSPACE` 不符 → 启动失败（不静默重映射）。
- `/health`、`/status`、`/api/tags`、`GET /workspaces` 各调用 100 次，断言**实例构造数为 0、迁移次数为 0**。
- 任意数据面首请求不触发 `check_and_migrate_data()`。
- schema 落后的非默认记录：数据面 503，后台迁移完成后转 ACTIVE；其中一条失败置 ERROR 且不影响其它记录与 Server 存活。
- 多 worker 下同一工作区只迁移一次。

**资源治理**（`tests/workspace/test_global_concurrency.py`）
- 加载 N 个工作区并发压 LLM / embedding / rerank，**观测到的并发数不超过配置总量**；单 worker 与多 worker 各断言一次。
- fix-proof：把 `global_concurrency_limits` 的无条件注册改回 `if workers > 1`，该测试必须失败。
- 按角色队列跨工作区共享（所有工作区的抽取排在同一 `llm:extract` 上）。

**API 与授权**（`tests/api/routes/test_workspace_routes.py`、`tests/api/auth/test_workspace_authz.py`）
- 选择器四态（缺失 / 空 / 非法 / 未知）逐一断言 §19 的状态码。
- 路由分类完整性：新增未归类路由使审计失败（复用 authz 的 golden 机制，按 profile 各一份）。
- 管理端点不读 `LIGHTRAG-WORKSPACE` 头。
- 有效权限 = 动作权限 ∩ 可达性：四种组合（有权限有成员 / 有权限无成员 / 无权限有成员 / 管理者无成员）各断言一次。
- 非成员访问存在的 ID 得 404 而非 403。
- legacy profile 下 4 个新权限码不可获得（创建工作区 403），且 `legacy_user` 对唯一的 legacy 记录可达。
- 请求路径 0-RPC：断言一次数据面请求中 `get_namespace_data("ws_catalog")` 的调用次数为 0。

**Ollama**（`tests/api/test_ollama_workspace_routing.py`）
- tag → 工作区路由；未知 tag 404；`/api/tags` 只列可访问的；`/api/ps` 只列已加载的；两个元数据端点均不构造实例。
- 存量客户端（配着 `--simulated-model-name/tag` 的默认值）不改配置即可继续工作。

**WebUI**（Bun 测试）
- 切换器持久化与静默回落；切换时三个 TAB 的本地状态被清空；0 工作区时的空白态；请求层统一注入头；404/503 分流不清 token。

---

## 21. 分阶段实施（PR 拆分）

| PR | 内容 | 风险 / 验收 |
| --- | --- | --- |
| **PR1** | `lightrag/workspace_id.py`（ID 生成与硬校验）+ `WorkspaceRecord` / `WorkspaceMember` 模型 + `BaseWorkspaceCatalogStorage` + `JsonWorkspaceCatalogStorage` + `PGWorkspaceCatalogStorage` + 注册表接线。**不接任何端点、不改任何现有行为** | 纯新增。验收 = 身份与目录两节测试全绿 |
| **PR2** | `*_WORKSPACE` 处置（§6.2 三条规则）+ `NamespaceDescriptor` 与四家族一致性校验 + `WorkspaceBinding` + §6.3 legacy 映射表。仍是单工作空间（目录尚未接线） | 最容易回归的一步。验收 = 有效工作区一致性全节 + 现有 `tests/kg/` 全绿 + k8s chart 同步移除 `POSTGRES_WORKSPACE` |
| **PR3** | 目录 bootstrap（§9.2）+ `WorkspaceCatalogRuntime` 快照与轮询 + 管理端点 6 个 + 新增 4 个权限码 + §5.2 创建闸门与 §5.3/§5.5 生命周期 | 验收 = 目录、启动与迁移（bootstrap 部分）、管理端点测试全绿；此时数据面仍只走 legacy 记录 |
| **PR4** | LLM 缓存分层（§7）+ 两个工具脚本改造 | 独立可回滚，且**必须在 PR5 之前**落地 —— 否则一旦数据面能路由到第二个工作区，查询答案就会跨库串味。验收 = 缓存分层全节，特别是那条反例钉子 |
| **PR5** | 实例池与租约（§8）+ `LIGHTRAG-WORKSPACE` 路由 + 端点分类审计 + `/health` 与 `/status` 拆分 + §14 授权第二段判定。**只放开数据读路由** | 核心 PR。验收 = 实例池、API 与授权两节全绿 + 0-RPC 断言 |
| **PR6** | 流水线与 doc_status 隔离（§10）+ 每工作区输入目录 + 每工作区解析/分块默认值（§5.4）+ §11 资源治理（含单进程闸门修复）+ 准入控制按工作区。**放开数据写与摄取路由** | 验收 = doc_status 哨兵全节 + 资源治理全节（含 fix-proof）+ 现有 `tests/pipeline/` 全绿 |
| **PR7** | Ollama tag 路由（§13） | 小面积。验收 = Ollama 全节 |
| **PR8** | WebUI（§15） | 验收 = Bun 测试 + 手工走查五个场景 |
| **PR9** | 启动期后台迁移（§9.3 最后两行）+ 可观测性（§18）+ `env.example` + 文档（`docs/LightRAG-API-Server*.md` 两语言）+ 启动横幅 + release note 的 §17 全清单 + **同步修改 authz PRD**（附录 B） | 破坏性变更集中在此，PR 描述须列全 §17 |

每个 PR 独立可回滚。PR1–PR4 之后任意时刻停下来都是"能跑的单工作空间中间态"；**PR5 是第一个让第二个工作区可达的 PR，因此 PR4 必须先落地**。

---

## 22. 与两份 RFC 的取舍

| 议题 | #3511 主张 | #3563 主张 | 本 PRD | 依据 |
| --- | --- | --- | --- | --- |
| 目录位置 | 共享持久后端；否定 JSON+进程锁 | 列为开放问题（3 选项） | **新增 `WS_CATALOG` 存储类型**，JSON 与 PG 双实现 | 与 doc_status 同构，运维认知成本最低（维护者 Q3） |
| 显式创建 | 强制，无逃生门 | 显式 + `--auto-create` 开关 | **强制，无逃生门** | 维护者 item 2；typo 变成持久数据风险不可接受 |
| 身份模型 | tagged canonical key + 不透明公共 ID + display name | 直接用 workspace 字符串 | **去中心化短 ID + 可改名字**，legacy 走 `legacy_layout` 元数据 | 采 #3511 的解耦，但 ID 是 19 字符可读短串而非 `kb_` 风格不透明值（维护者 发现3/Q2） |
| `*_WORKSPACE` | 多工作空间模式启动失败 | **完全未提及** | **多工作空间失败 + 创建期闸门 + 单工作空间告警** | #3511 是对的；三层递进兼顾存量（维护者 Q9） |
| doc_status | 是工作队列，需哨兵测试 | 未触及 | **采纳，含专门测试节** | 内容哈希使同文档跨工作区同 ID，塌缩即覆写 |
| 重启恢复 | catalog 驱动 eager + fencing | pure lazy | **纯手工（`/scan` / `/reprocess_failed`）+ `/status` 暴露待恢复量** | 与今天语义一致；手工策略靠可观测性可运维（维护者 item 9 / Q12） |
| 全局并发 | 完整 admission controller + WRR/DRR 公平 | 单个全局 Semaphore | **复用现有 concurrency_group 闸门 + 修单进程注册缺口**；公平性延后 | 现有实现已是跨进程的，无需新建；缺口是真实的（维护者 Q1） |
| 池治理 | 容量 + 权重预算 + 6 条驱逐条件 | 仅硬上限 + 429 | **容量（不许为 0）+ 空闲 TTL + 租约 + 6 条驱逐条件 + 503** | 采 #3511 的驱逐条件，去掉权重预算（维护者 Q8） |
| 迁移时机 | 绝不在数据面 | 承认在请求路径上，延后 | **全部在控制面**（创建时 / 启动时 / 显式重试） | #3511 不变式 11；正面回答 PR #3397 问题 6 |
| Ollama | model alias + header 冲突 400 | 只走 default | **tag 路由 + 未知 404 + header 在该面被忽略** | 采 #3511 的能力，用"忽略 header"取代"冲突 400"（更简单，无歧义） |
| header 命名 | `LIGHTRAG-KNOWLEDGE-BASE` | `LIGHTRAG-WORKSPACE` | **`LIGHTRAG-WORKSPACE`，只收 ID** | 沿用社区已有认知（维护者 Q13） |
| 授权 | 预留 `WorkspaceAuthorizer` 钩子 | 明确 out of scope | **两层：策略文件管动作、目录管成员，取交集** | 策略文件只读、目录可写，切分沿此裂缝（维护者 Q11） |
| 无关变更 | 仅 non-goals 一句 | 点名两处并承诺拆 PR | **沿用 #3563：`trust_env=False` 与 nano_vdb 的 `Semaphore(4)` 各自独立 PR，不进本方案** | #3563 在这条上更负责 |
| LLM 缓存 | 未讨论 | 未讨论 | **按判据分两层共享** | 两份 RFC 的共同空白；维护者 Q10 提出方向，判据由本 PRD 补齐 |

两份 RFC 各自的空白，本 PRD 都补上了：#3511 缺代码级落点与无关变更处置，#3563 缺 `*_WORKSPACE`、缺 doc_status 特殊性、缺 `default` 命名碰撞；两者都缺 LLM 缓存分层与工作区授权的具体形态。

---

## 23. 遗留风险与后续阶段

| 项 | 说明 | 归属 |
| --- | --- | --- |
| 无跨工作区公平调度 | 一个工作区的批量摄取可长时间占满 `llm:extract`。全局队列保证了总量安全，但不保证公平（§11.2） | 接受，后续阶段（WRR/DRR + 优先级老化） |
| 非默认工作区重启后不自动恢复 | 刻意选择（§10.4）。缓解手段是 `/status` 暴露每工作区 PENDING/FAILED 计数 | 接受，文档写明 |
| 一个坏的默认工作区会让 Server 起不来 | 维护者明确接受；与今天行为一致（§9.1） | 接受 |
| 连接数随 `workers × 已加载工作区数` 增长 | 由 `MAX_ACTIVE_WORKSPACES` 与空闲驱逐兜住，指标暴露真实占用（§18） | 接受 |
| 共享摄取缓存是跨工作区确认预言机 | 命中要求已持有逐字节相同内容，预言对象是调用方自有数据（§7.3） | 接受，release note 写明 |
| 无多节点支持 | 目录后端选 PG/Mongo 时数据层已可跨主机，但实例池、`shared_storage` 的 Manager 锁与并发闸门仍是单主机语义 | 后续阶段 |
| 无物理隔离 / storage profile | 与 #2527 的诉求交集留待后续；`WorkspaceRecord` 未预留 `storage_profile_id` 字段，将来加字段属兼容性新增 | 后续阶段 |
| 无多租户 | 工作区已是独立于租户的对象且 ID 去中心化，为将来的租户层与跨租户共享留了空间 | 后续阶段（维护者 item 4 / item 12） |
| 工作区级配额与按工作区认证 | 本阶段成员表是布尔可达性，无配额、无每工作区凭据 | 后续阶段 |
| `credential_version` 式的成员表版本 | 移除成员后其在途请求仍持租约至结束（不中断在途），最长一个请求周期 | 接受 |
| 目录轮询延迟 | 管理变更最迟 `WS_CATALOG_POLL_SECONDS` 后全 worker 可见；数据面对未知 ID 有一次按需刷新兜底（§4.4） | 接受 |

---

## 附录 A：工作区 ID 编码与长度预算（唯一权威口径）

### A.1 编码

```text
value  = (unix_ms & 0xFFFFFFFFFFFF) << 32  |  secrets.randbits(32)      # 80 位
id     = "ws_" + base36_lower(value).rjust(16, "0")                     # 固定 16 位
```

- 36^16 ≈ 7.96e24 > 2^80 ≈ 1.21e24，16 位足够且有余量。
- 48 位毫秒时间戳可用至公元 10889 年；固定宽度 + 大端使字典序 = 创建时序。
- 32 位 CSPRNG 后缀：同一毫秒内两次生成碰撞概率 2^-32。同 Server 内碰撞由目录的 create-if-absent 直接排除；跨 Server 迁移合并时碰撞可被检测并报告（不静默覆盖）。
- 字符集只用 `[0-9a-z]`（base36 小写）。**不用 base62**：混用大小写会在 PostgreSQL（未加引号标识符折叠为小写）、macOS/Windows 文件系统（大小写不敏感）与 Neo4j（标签大小写敏感）之间产生"同一对 ID 在某些后端是一个命名空间、在另一些后端是两个"的不一致。base62 只能省 2 个字符。

### A.2 长度预算

物理名的最长形态是 `{ws_id}_{namespace}` 或等价派生（19 + 1 + namespace 长度）。最长的 namespace 是 `entity_relation_graph` 与 `vdb_relationships` 一类，约 21 字符，故最长派生名约 **41 字符**。

| 后端 | 上限 | 41 字符是否安全 |
| --- | --- | --- |
| Milvus collection | 255，须以字母/下划线开头，仅 `[A-Za-z0-9_]` | ✅（且 `ws_` 保证首字符合法） |
| PostgreSQL 标识符 / AGE 图名 | 63 字节 | ✅ |
| Neo4j / Memgraph 标签 | 实践无硬限 | ✅ |
| OpenSearch index | 255，须小写 | ✅（小写 base36 天然满足） |
| Qdrant collection | 255 | ✅ |
| Redis key 前缀 | 无实际限制 | ✅ |
| 文件系统目录名 | 255 字节 | ✅ |

实现必须有一条参数化测试对着本表逐后端断言，新增后端时同步补行。

---

## 附录 B：需要同步修改的 authz PRD 条目

本方案落地后，`LR2-auhtorization-file-policy-phase1.md` 需按下列条目更新（维护者已批准修改该 PRD）：

| 位置 | 现内容 | 改为 |
| --- | --- | --- |
| §3 权限目录 | 26 个权限码 | 增加 `workspace.read` / `workspace.create` / `workspace.update` / `workspace.delete` |
| §5.1 公共路由 | `/health` 双重身份（"始终 200，未认证只给 liveness，完整配置需 `system.health.read`"） | `/health` 退化为**纯 liveness**；完整配置移至 `GET /status`（受 `system.health.read`），并从 `PUBLIC_ROUTES` 移出 |
| §5.2 受保护路由 | 40 条 | 增加 `/status` 与 6 个 `/workspaces*` 端点，共 47 条 |
| §11.4 `LEGACY_USER_PERMISSIONS` | 18 项冻结列表 | **不变**（4 个新码按约束 2 不自动进入）；补一句说明工作区管理在 legacy 模式下不可用 |
| §1.2 非目标 | "不做 workspace / tenant 维度的授权。放到第2阶段实现" | 改为"工作区维度的**可达性**由工作区目录承载（见多工作空间 PRD §14）；策略文件 v1 的 scope 字段禁令（§7.3 #4）**保持不变**；租户维度仍是非目标" |
| §16 遗留风险 | "无 workspace/tenant 作用域 → 第三阶段" | 拆成两行：工作区可达性 → 已由多工作空间 PRD 解决；租户作用域 → 后续阶段。（顺带修正 §1.2 说"第2阶段"、§16 说"第三阶段"的既有不一致） |
| §4.2 扩展点 | `AuthorizationContext` 本版本不序列化 | 补一句：多工作空间落地后运行期会填充 `ResourceScope("workspace", ws_id)`，但**仍不进策略文件序列化** |
| §12 WebUI | 四路状态码分流 | 补两条工作区特有分流（404 静默回落、503 重试），见多工作空间 PRD §15.5 |
