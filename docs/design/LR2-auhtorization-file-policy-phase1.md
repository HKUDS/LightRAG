# PRD：文件驱动的认证与授权管理（第1阶段）

- 状态：设计草案（§15 全部 RFC 偏离已评审确认，2026-08-08）
- 适用范围：LightRAG Server（`lightrag/api/`）+ WebUI 的最小配合改造
- 关键约束（用户需求）：第一阶段用户与权限**全部用 YAML 文件管理**，**不开发管理界面**，但文件更新后可以**通过 API 不停机重载**

## TL;DR 与阅读指南

一句话概括：路由声明 permission、YAML 策略文件定义角色与绑定、`POST /auth/policy/reload` 校验后原子换入并广播到全部 worker；不设策略文件的存量部署行为逐字不变。

| 读者 | 需要读的章节 |
| --- | --- |
| 运维 / 部署 | §7（策略文件与校验）、§9.1 / §9.6（重载端点与自锁保护）、§11（profile、破坏性变更、迁移序列） |
| 后端实现者 | §3–§6（目录、模型、路由映射与审计）、§7–§10（策略文件、凭据、热重载、强制机制与中间件）、附录 A（传播协议伪代码）、§13–§14（验收与 PR 拆分） |
| 前端 | §12（401/403/500/503 分流与 `/auth/me`） |
| 评审 | §15（RFC 偏离裁决）、§16（遗留风险）、§9.7（被否决方案） |

**唯一权威口径**（其它位置只作引用，不再复述理由）：四档状态码语义 = §10.2；`documents.artifacts.*` 仅 policy 模式可授予 = §11.4；`WHITELIST_PATHS` 处置 = §11.2 #1；冷启动"磁盘是权威" = §9.5；`STALE_AFTER` 公式 = §9.4；传播协议伪代码 = 附录 A。

---

## 1. 目标与非目标

### 1.1 目标

1. 建立权限验证的持久关系：`endpoint -> permission -> role -> principal`。路由只声明 permission，永不出现 `role == "admin"` 这类判断。
2. 权限目录（catalog）由代码集中拥有；角色组合、principal 与角色的绑定放在策略文件（provider）里。
3. 认证（credential）与授权（authorization）走两个独立可替换的 Protocol，将来换数据库实现时互不牵连。
4. 默认拒绝（deny-by-default）：受保护路由必须显式声明至少一个 permission；漏声明是启动期错误，不是"任意已登录用户可访问"。
5. **热重载**：`AUTH_POLICY_FILE` 修改后，通过 API 触发校验 + 原子换入，不重启进程、不中断在途请求；多 worker（gunicorn）部署下**全部 worker 采纳同一份字节**（校验过的内容随 epoch 一起进共享内存），且收敛状态可验证。
6. 一切失败关闭（fail closed）：策略不可加载、provider 出错、worker 未采纳新版本，都不得退化成"空角色"或"匿名放行"，**也不得退化成"继续按旧版本放行"** —— 一个已看到更高 epoch 却无法采纳的 worker 必须停止服务受保护请求（HTTP 503），而不是拿旧快照继续授权（§9.4 stale 状态机）。

### 1.2 非目标（第一阶段明确不做）

- 不做用户/角色管理界面，也不做写策略文件的 API（服务端只读文件 + 重载，从不回写；策略文件与运行时真相的关系见 §9.5）。
- 不做 workspace / tenant 维度的授权。放到第2阶段实现；本版本是"单一默认数据边界"，策略文件里出现 `scope` / `workspace` / `tenant` 字段一律**拒绝加载**，代码也不允许伪造 `tenant_id="default"`。
- 不做用户自助改密、注册、找回密码、OIDC/SSO/LDAP。
- 不做行级 / 单文档 ACL（`documents.read` 是全库粒度）。
- 不做审计日志落库（只做结构化日志字段，落库留后续；但需要做必要的控制台日志输出）。
- 不做数据库 provider（接口留出，实现在第3阶段实现）。

---

## 2. 现状分析

| 现状 | 位置 | 对本方案的影响 |
| --- | --- | --- |
| `auth_handler = AuthHandler()` 在**模块 import 期**构造，账号来自 `AUTH_ACCOUNTS` 环境变量 | [auth.py:260](../../lightrag/api/auth.py#L260)、[auth.py:50-111](../../lightrag/api/auth.py#L50-L111) | 热重载的第一号障碍：账号表被 import 期冻结 |
| `whitelist_paths` / `whitelist_patterns` / `auth_configured` 均为**模块级常量** | [utils_api.py:242-257](../../lightrag/api/utils_api.py#L242-L257) | 同上；且白名单是"免认证"，无法表达权限 |
| `get_combined_auth_dependency(api_key)` 只做认证，闭包**按值捕获** `api_key` | [utils_api.py:375-519](../../lightrag/api/utils_api.py#L375-L519) | 没有任何授权概念；api_key 无法重载 |
| `credentials_accepted()` 是认证判定的唯一咽喉，被路由依赖与 ASGI 中间件共用 | [utils_api.py:336-372](../../lightrag/api/utils_api.py#L336-L372) | 保留这个"单咽喉"设计，替换其内部实现 |
| 40 处 `dependencies=[Depends(combined_auth)]` | `document_routes.py` **19** 处、`graph_routes.py` **12** 处、`query_routes.py` 3 处、`ollama_api.py` 5 处、`lightrag_server.py` 1 处（`/health`）—— 合计 40（计数已按当前代码校正） | 主体 diff 量，按 router 分期迁移 |
| JWT 载荷含 `role`（`"user"` / `"guest"`），但**从不用于授权** | [auth.py:43-47](../../lightrag/api/auth.py#L43-L47)、[auth.py:247-253](../../lightrag/api/auth.py#L247-L253) | 保留为显示字段，授权一律走 provider |
| `WHITELIST_PATHS` 默认 `"/health,/api/*"` | [config.py:804](../../lightrag/api/config.py#L804) | 默认把整个 Ollama 兼容面免认证，policy 模式下必须去掉 |
| admission 中间件在读 body 前用 `credentials_accepted()` 预认证 | [admission_middleware.py:117-135](../../lightrag/api/admission_middleware.py#L117-L135) | 必须与路由依赖共用同一 provider，否则两层判定分裂 |
| admission ticket 在 `finally` 中按"未被下游采纳"释放 | [admission_middleware.py:152-163](../../lightrag/api/admission_middleware.py#L152-L163) | **已核实**：路由依赖抛 403 时 handler 不会 adopt ticket，容量槽会被归还。因此"中间件只认证、权限交给路由"不会泄漏容量 |
| 跨进程共享状态设施（Manager dict / 命名空间锁 / pre-fork hub） | `lightrag/kg/shared_storage.py`：`get_namespace_data`、`get_namespace_lock`、`_pid_alive`、`_process_alive` | 多 worker 重载广播复用这一套，不新引入 IPC |
| 登录限流器（IP+username，预占式） | `lightrag/api/login_rate_limit.py` | 保持不变，只把"账号从哪来"换成策略文件 |
| 路由清单（现存） | 受保护 `APIRoute` 40 个 + 公共 `APIRoute`（`/login`、`/auth-status`、`/`、`/webui`、`/docs`、`/docs/oauth2-redirect`）+ **FastAPI 自动注册的 `/openapi.json` 与 `/redoc`** + 2 个 `Mount` | 见 §5 完整映射表 |
| `docs_url=None` + 自定义 `/docs`；`redoc_url="/redoc"`、`openapi_url="/openapi.json"` 走 FastAPI 自动注册 | [lightrag_server.py:1425-1434](../../lightrag/api/lightrag_server.py#L1425-L1434)、[:2328](../../lightrag/api/lightrag_server.py#L2328) | 这四个端点**今天完全无认证依赖**。审计要遍历全部 `APIRoute`，漏列它们会让 `enforce` 直接启动失败 —— §5.1 已补齐 |
| swagger 静态资产挂在 **`/static/swagger-ui`**（`name="swagger-ui-static"`） | [lightrag_server.py:2802](../../lightrag/api/lightrag_server.py#L2802) | `PUBLIC_MOUNTS` 必须写**实际挂载路径**，不是 `name=` |

> RFC 里出现的 artifact 导出路由（`/documents/{doc_id}/artifacts/...`）在当前代码中**尚不存在**（属于 LR2 下载管线设计的在途工作）。本方案把它们列为"预留行"，落地时必须按同表补齐权限声明，且启动期审计会强制这一点。

---

## 3. 权限目录（code-owned catalog）

`lightrag/api/authz/catalog.py`：

```python
class Permission(str, Enum):
    AUTH_SESSION_READ = "auth.session.read"
    AUTH_POLICY_READ = "auth.policy.read"          # 新增：读策略状态 / dry-run 校验
    AUTH_POLICY_RELOAD = "auth.policy.reload"      # 新增：触发热重载
    SYSTEM_HEALTH_READ = "system.health.read"
    SYSTEM_CONFIG_READ = "system.config.read"
    SYSTEM_CONFIG_WRITE = "system.config.write"
    QUERY_EXECUTE = "query.execute"
    OLLAMA_INFERENCE = "ollama.inference"
    OLLAMA_METADATA_READ = "ollama.metadata.read"
    DOCUMENTS_READ = "documents.read"
    DOCUMENTS_WRITE = "documents.write"
    DOCUMENTS_RETRY = "documents.retry"
    DOCUMENTS_DELETE = "documents.delete"
    DOCUMENTS_CLEAR = "documents.clear"
    DOCUMENTS_SOURCE_CONFLICTS_READ = "documents.source_conflicts.read"
    DOCUMENTS_SOURCE_CONFLICTS_REPAIR = "documents.source_conflicts.repair"
    DOCUMENTS_ARTIFACTS_READ = "documents.artifacts.read"
    DOCUMENTS_ARTIFACTS_SOURCE_DOWNLOAD = "documents.artifacts.source.download"
    DOCUMENTS_ARTIFACTS_PARSED_DOWNLOAD = "documents.artifacts.parsed.download"
    PIPELINE_READ = "pipeline.read"
    PIPELINE_CONTROL = "pipeline.control"
    CACHE_CLEAR = "cache.clear"
    GRAPH_READ = "graph.read"
    GRAPH_WRITE = "graph.write"
    GRAPH_DELETE = "graph.delete"
```

规则：

- 目录之外的权限码在加载策略时**拒绝**（未知值不是警告）。
- `"*"` 仅作为策略文件里管理员角色的书写糖，加载时**展开为当前目录全集**后落进快照；运行期不存在通配匹配逻辑。
- `auth.policy.reload` 与 `system.config.write` 分开：重载策略等于改授权本身，权限面必须比"改运行时配置"更窄（RFC 的"破坏性操作权限更窄"原则）。
- `ollama.metadata.read` 与 `auth.session.read` 分开：后者是"任何登录用户都该有"的会话自省权限，把 `/api/tags`、`/api/version` 挂上去等于任何 reader 都能枚举后端模型配置。改一个枚举值的成本远低于以后再拆。
- `documents.artifacts.*` 三个权限码**只能通过策略文件授予**（唯一权威口径与理由见 §11.4；对旧配置属新能力 deny-by-default，不是行为变更）。

---

## 4. 运行时模型

### 4.1 存储中立对象（不可变）

```python
@dataclass(frozen=True, slots=True)
class Principal:
    principal_type: Literal["user", "api_key", "anonymous"]
    principal_id: str            # 对代码不透明；DB 模式将换成 UUID
    display_name: str = ""

@dataclass(frozen=True, slots=True)
class Role:
    role_id: str
    permissions: frozenset[Permission]

@dataclass(frozen=True, slots=True)
class RoleAssignment:
    principal_type: str
    principal_id: str
    role_id: str

@dataclass(frozen=True, slots=True)
class UserRecord:
    principal_id: str
    display_name: str
    password_hash: str           # bcrypt spec
    enabled: bool
    credential_epoch: str        # 见 §8.2

@dataclass(frozen=True, slots=True)
class ApiKeyRecord:
    principal_id: str            # 服务端 key ID，客户端永远不发送
    enabled: bool

@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    revision: int                        # 单调递增（仅限一个 Manager 生命周期内）；见 §9.3 / §9.5
    source_digest: str                   # sha256(文件原始字节)
    revision_source: str                 # "disk"（冷启动发布）/ "reload"：这个 revision 的
                                         #   字节从哪来；随 epoch 传播，follower 抄 target
    adoption_source: str                 # "disk"（冷启动 leader 读盘）/ "shared"（从共享字节
                                         #   构建）/ "reload_local"（reload 发起者本地定稿）：
                                         #   本 worker 怎么拿到它；两者都进采纳报告（§9.5）
    loaded_at: float                     # time.time()
    users: Mapping[str, UserRecord]
    api_key_index: Mapping[str, str]     # HMAC(pepper, secret) -> principal_id
    api_keys: Mapping[str, ApiKeyRecord]
    roles: Mapping[str, Role]
    assignments: tuple[RoleAssignment, ...]
    effective: Mapping[tuple[str, str], frozenset[Permission]]  # 预计算
```

- `effective` 在加载期一次算完，授权判定退化为一次 dict 取值 + 一次 `in`，`has_permission` 是 O(1)，不在请求路径做集合运算。
- 所有集合是 `frozenset`，所有映射构造后不再写入 —— 快照的不可变性是"请求内一致性"的基础。

### 4.2 扩展点（本版本不序列化）

```python
@dataclass(frozen=True, slots=True)
class ResourceScope:
    scope_type: str   # provider 私有命名空间，不是 tenant/workspace 的硬编码联合
    scope_id: str

@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    scope: ResourceScope | None = None
```

当前所有请求传 `AuthorizationContext()`。文件 provider 见到任何 `scope`/`workspace`/`tenant` 字段就**报错拒绝加载**，而不是忽略。

### 4.3 快照持有者与请求内绑定

```python
class PolicyRuntime:
    """进程内唯一的快照持有者。current() 是无锁读；swap() 是整体替换。"""
    def current(self) -> PolicySnapshot: ...
    def swap(self, snapshot: PolicySnapshot) -> PolicySnapshot: ...   # 返回旧快照
```

- `swap()` 只做一次属性赋值（CPython 下原子），因此没有"半新半旧"的中间态。
- **请求内绑定**：认证依赖在第一次解析时把快照放进 `request.state.policy_snapshot`，同一请求后续所有权限判定复用它。否则一个请求的两个依赖可能落在换入前后的两个版本上，出现"能读不能写"的诡异 403。
- 纯 ASGI 层（admission / body limit）直接读 `PolicyRuntime.current()`，并把结果写入 `scope["state"]`，与上面同一把绑定。

---

## 5. 路由 → 权限映射（完整清单）

本节的映射对**所有 profile** 生效（路由声明的 permission 是代码常量，不随 profile 变化）；变化的只是"principal 从哪里拿到这些权限"：policy 模式来自策略文件，legacy / 开放模式来自内建 `legacy_user` 预设（§11.4）。§5.1 的公共路由清单只在 policy 模式取代 `WHITELIST_PATHS`，legacy 模式的免认证面仍由该环境变量决定（§11.2 #1）。

### 5.1 公共路由（显式清单，代码常量）

```python
PUBLIC_ROUTES = {
    ("POST", "/login"),
    ("GET", "/auth-status"),
    ("GET", "/health"),          # 仅最小 liveness；完整配置需 system.health.read
    ("GET", "/"),                # WebUI 重定向
    ("GET", "/webui"), ("GET", "/webui/"),   # 无资产时的重定向路由
}
PUBLIC_MOUNTS = {"/webui", "/static/swagger-ui"}   # 静态资产；后者是实际挂载路径，
                                                   # 不是 name="swagger-ui-static"
API_DOCS_ROUTES = {                # 由 AUTH_EXPOSE_API_DOCS 决定是否注册
    ("GET", "/docs"),              # 自定义端点（docs_url=None）
    ("GET", "/docs/oauth2-redirect"),
    ("GET", "/redoc"),             # FastAPI 自动注册
    ("GET", "/openapi.json"),      # FastAPI 自动注册
}
```

`/health` 的双重身份保持现状语义：始终 200，但**未认证只给 liveness**，完整配置需 `system.health.read`（现在的实现已经是"认证与否决定返回内容"，只是把"认证"换成"持有权限"）。

#### API 文档面：四个端点，全有或全无

`/redoc` 与 `/openapi.json` 是 FastAPI 根据 `app_kwargs` **自动注册**的（[lightrag_server.py:1428-1431](../../lightrag/api/lightrag_server.py#L1428-L1431)），`/docs` 与 `/docs/oauth2-redirect` 是自定义端点（[:2328](../../lightrag/api/lightrag_server.py#L2328)、[:2346](../../lightrag/api/lightrag_server.py#L2346)）。**四个今天都没有任何认证依赖**。四个都必须进公共清单、静态 mount 必须按**实际挂载路径**归类 —— 任何漏列都会让 `AUTH_ROUTE_AUDIT=enforce` 在启动时撞上未归类的 `APIRoute` / `Mount`：要么启动失败，要么迫使实现里写一份没有文档记载的排除清单。

裁决：**保持公开（现状），由 `AUTH_EXPOSE_API_DOCS`（默认 `true`）整体开关**，关闭时四个端点**不注册**（404），不做"部分需要权限"的中间态。理由是 Swagger UI 在浏览器里自行拉取 `/openapi.json`，只给它挂权限会让文档页在未登录时直接坏掉，而"坏掉的文档页"比"公开的 schema"更容易被误判成 bug；要收敛攻击面就整片关掉。权限码 `system.config.read` 保留给将来真正返回运行时配置的端点，不用来盖文档面。

审计的 golden 清单**必须从实际 `app.routes` 生成**（§6），不能从本节手抄 —— 手抄清单必然漂移。

### 5.2 受保护路由

| 方法 | 路径 | 权限 | 现位置 |
| --- | --- | --- | --- |
| POST | `/query` | `query.execute` | query_routes.py:318 |
| POST | `/query/stream` | `query.execute` | query_routes.py:657 |
| POST | `/query/data` | `query.execute` | query_routes.py:1013 |
| POST | `/api/generate` | `ollama.inference` | ollama_api.py:404 |
| POST | `/api/chat` | `ollama.inference` | ollama_api.py:599 |
| GET | `/api/version` | `ollama.metadata.read` | ollama_api.py:352 |
| GET | `/api/tags` | `ollama.metadata.read` | ollama_api.py:357 |
| GET | `/api/ps` | `ollama.metadata.read` | ollama_api.py:380 |
| GET | `/documents` | `documents.read` | document_routes.py:6008 |
| POST | `/documents/paginated` | `documents.read` | document_routes.py:6355 |
| GET | `/documents/status_counts` | `documents.read` | document_routes.py:6536 |
| GET | `/documents/track_status/{track_id}` | `documents.read` | document_routes.py:6281 |
| GET | `/documents/scan/status/{track_id}` | `documents.read` | document_routes.py:4623 |
| GET | `/documents/supported_file_types` | `documents.read` | document_routes.py:6563 |
| POST | `/documents/upload` | `documents.write` | document_routes.py:4969 |
| POST | `/documents/text` | `documents.write` | document_routes.py:5279 |
| POST | `/documents/texts` | `documents.write` | document_routes.py:5407 |
| POST | `/documents/scan` | `documents.retry` **AND** `documents.write` | document_routes.py:4254 |
| POST | `/documents/reprocess_failed` | `documents.retry` | document_routes.py:6595 |
| DELETE | `/documents/delete_document` | `documents.delete` | document_routes.py:6120 |
| DELETE | `/documents` | `documents.clear` | document_routes.py:5569 |
| GET | `/documents/source_conflicts` | `documents.source_conflicts.read` | document_routes.py:4665 |
| POST | `/documents/source_conflicts/repair` | `documents.source_conflicts.repair` | document_routes.py:4740 |
| GET | `/documents/pipeline_status` | `pipeline.read` | document_routes.py:5885 |
| POST | `/documents/cancel_pipeline` | `pipeline.control` | document_routes.py:6924 |
| POST | `/documents/recovery/force_reset` | `pipeline.control` | document_routes.py:6744 |
| POST | `/documents/clear_cache` | `cache.clear` | document_routes.py:6247 |
| GET | `/graph/label/list` | `graph.read` | graph_routes.py:165 |
| GET | `/graph/label/popular` | `graph.read` | graph_routes.py:180 |
| GET | `/graph/label/search` | `graph.read` | graph_routes.py:202 |
| GET | `/graphs` | `graph.read` | graph_routes.py:226 |
| GET | `/graph/entity/exists` | `graph.read` | graph_routes.py:262 |
| POST | `/graph/entity/edit` | `graph.write` | graph_routes.py:283 |
| POST | `/graph/relation/edit` | `graph.write` | graph_routes.py:474 |
| POST | `/graph/entity/create` | `graph.write` | graph_routes.py:510 |
| POST | `/graph/relation/create` | `graph.write` | graph_routes.py:585 |
| POST | `/graph/entities/merge` | `graph.delete` | graph_routes.py:675 |
| DELETE | `/graph/entity/delete` | `graph.delete` | graph_routes.py:761 |
| DELETE | `/graph/relation/delete` | `graph.delete` | graph_routes.py:797 |
| GET | `/auth/me` | `auth.session.read` | 新增（§12） |
| GET | `/auth/policy/status` | `auth.policy.read` | 新增（§9，仅 policy 模式挂载） |
| POST | `/auth/policy/validate` | `auth.policy.read` | 新增（§9，仅 policy 模式挂载） |
| POST | `/auth/policy/reload` | `auth.policy.reload` | 新增（§9，仅 policy 模式挂载） |

`/graph/entities/merge` 归 `graph.delete` 而非 `graph.write`：合并会销毁实体，属于破坏性操作。

`/documents/scan` 是**唯一需要两个 scope 的路由**：它名义上是"扫描"，实际会遍历 input 目录**发现并摄取新文件**（`run_scanning_process` → `pipeline_index_files`，[document_routes.py:3535](../../lightrag/api/routers/document_routes.py#L3535)），同时兼做 FAILED 重试。只挂 `documents.retry` 等于给出一条绕过 `documents.write` 的摄取通道 —— 一个"只能重试、不能上传"的 principal 把文件放进共享输入目录（挂载卷、sidecar、另一个只有写盘权限的进程）后调用 scan，就完成了写入。因此要求 `documents.retry AND documents.write`。

不拆成两个端点是刻意的：拆分要动 §9.1 的扫描栅栏与 sticky manual retry 语义（`run_scanning_process` 的 FAILED 重置与文件发现共享同一个 `scanning_exclusive` 阶段），属于管线改造而非授权改造。AND 是零风险的等价约束；真要"纯失败重试"，已有 `/documents/reprocess_failed`（纯存储驱动、不扫文件系统），它只需要 `documents.retry`。

`/auth/policy/*` 三个端点**只在 policy 模式下注册**：legacy / 开放模式没有策略文件，没有可重载的对象。它们的缺席由部署 profile 决定，而不是靠"内建预设角色不含 `auth.policy.*`"来间接实现 —— 后者会让运维在 legacy 模式下拿到 403 而不是 404，误以为"权限不够"。路由覆盖审计（§6）按 profile 分别产出 golden 清单。

### 5.3 预留行（artifact 导出落地时必须补齐）

| 方法 | 路径 | 权限 |
| --- | --- | --- |
| GET | `/documents/{doc_id}/artifacts` | `documents.artifacts.read` |
| POST | `/documents/{doc_id}/artifacts/source/exports` | `documents.artifacts.source.download` |
| GET | `/documents/artifact-exports/source/{track_id}` | `documents.artifacts.source.download` |
| GET | `/documents/artifact-exports/source/{track_id}/download` | `documents.artifacts.source.download` |
| POST | `/documents/{doc_id}/artifacts/parsed/exports` | `documents.artifacts.parsed.download` |
| GET | `/documents/artifact-exports/parsed/{track_id}` | `documents.artifacts.parsed.download` |
| GET | `/documents/artifact-exports/parsed/{track_id}/download` | `documents.artifacts.parsed.download` |

---

## 6. 启动期与 CI 路由覆盖审计

`lightrag/api/authz/inventory.py`：

```python
def audit_route_coverage(app: FastAPI) -> RouteInventory
```

递归遍历完整的 Starlette `BaseRoute` 拓扑，断言：

- 每个 `APIRoute` 要么在公共清单里（含 `API_DOCS_ROUTES`，即 FastAPI 自动注册的 `/redoc`、`/openapi.json` 与自定义 `/docs*`），要么带 `authorize` 依赖且 scopes 全部是已知权限码；
- 每个 `Mount` 要么在 `PUBLIC_MOUNTS` 里（按**实际挂载路径**匹配，不是 `name=`），要么递归审计其子路由；
- `WebSocketRoute` 与任何其它路由类型必须显式归类，否则启动失败；
- 敏感路由（`DELETE`、`clear`、`force_reset`、`cancel_pipeline`、artifact 下载）不得只挂通用 read/write 权限（白名单式断言，新增敏感路由必须同步更新，属于刻意的"改动即评审"）；
- OpenAPI 中写入 `x-required-permissions`。

产出可评审清单（CI artifact）：

```text
TYPE       METHOD  PATH                                                   PERMISSIONS/CLASS
APIRoute   POST    /query                                                 query.execute
APIRoute   POST    /documents/upload                                      documents.write
APIRoute   DELETE  /documents/delete_document                             documents.delete
APIRoute   GET     /health                                                public (+system.health.read for detail)
Mount      -       /webui                                                 public-static
```

`AUTH_ROUTE_AUDIT=report` 只打印，`enforce` 则启动失败。CI 用固化清单做 golden 对比：新增路由若未归类，测试失败。

**golden 清单必须由实际 `app.routes` 生成，不得手抄** —— 文档里的清单必然漂移（自动注册端点、mount 实际路径、路由计数都漂过）。生成脚本产出 golden，评审看 diff：这样"文档写错"最多是文档问题，不会变成 `enforce` 启不起来。

**golden 清单按 profile 各一份**：`/auth/policy/*` 三个端点只在 policy 模式挂载（§5.2），`API_DOCS_ROUTES` 受 `AUTH_EXPOSE_API_DOCS` 控制，两份清单的路由集合本就不同；用一份清单会逼实现把它们无条件挂上去，legacy 模式下就变成"存在但永远 403"。

---

## 7. 策略文件（YAML v1）

### 7.1 配置项

```dotenv
AUTH_POLICY_FILE=/etc/lightrag/auth-policy.yaml
```

### 7.2 完整示例

```yaml
version: 1

credentials:
  users:
    admin:
      display_name: "Platform Admin"
      # bcrypt 摘要，不是明文口令。必须是裸 bcrypt spec（$2b$...），不带 {bcrypt} 前缀 ——
      # 生成：lightrag-hash-password --format policy（AUTH_ACCOUNTS 模式输出的
      # username:{bcrypt}$2b$... 带两种前缀，直接粘贴会被规则 #7 拒绝）
      password_bcrypt: "$2b$12$Xm2Q....（60 chars）"
      enabled: true
      credential_version: 1        # 会话世代计数器；见 §8.2
    alice:
      display_name: "Alice"
      # 也可以从环境变量读 bcrypt 摘要（此时该用户不参与热重载改密）
      password_env: LIGHTRAG_USER_ALICE_BCRYPT
      enabled: true
      credential_version: 1

  api_keys:
    ingestion:
      secret_env: LIGHTRAG_API_KEY_INGESTION      # 明文密钥只允许来自环境变量
      enabled: true
    artifact-export:
      secret_env: LIGHTRAG_API_KEY_ARTIFACT_EXPORT
      enabled: true

authorization:
  roles:
    reader:
      permissions: [auth.session.read, query.execute, documents.read, graph.read]
    operator:
      permissions:
        - auth.session.read
        - query.execute
        - documents.read
        - documents.write
        - documents.retry
        - pipeline.read
    artifact_reader:
      permissions:
        - documents.artifacts.read
        - documents.artifacts.source.download
        - documents.artifacts.parsed.download
    admin:
      permissions: ["*"]

  assignments:
    - principal_type: user
      principal_id: admin
      roles: [admin]
    - principal_type: user
      principal_id: alice
      roles: [operator]
    - principal_type: api_key
      principal_id: artifact-export
      roles: [reader, artifact_reader]
```

`credentials` 与 `authorization` 共处一文件只是运维便利；加载时校验成**两份独立的 provider 输入**，运行期不耦合。

> 示例里的 `reader` / `operator` / `admin` 是**推荐起点，不是兼容层**。特别地，`operator` 远窄于"今天任何认证用户的能力"，不能用它承载 legacy 行为保持 —— 见 §11.4 的内建 `legacy_user` 角色与那里列出的三条约束。

### 7.3 校验规则（加载 / 重载共用同一套）

| # | 规则 | 失败行为 |
| --- | --- | --- |
| 1 | `yaml.safe_load()`（禁 `full_load`）/ 标准 `json.loads` | 拒绝 |
| 2 | 文件字节数 ≤ `AUTH_POLICY_MAX_BYTES`（默认 1 MiB） | 拒绝，不解析 |
| 3 | `version == 1`；顶层与各层级**未知字段一律拒绝**（strict schema，Pydantic `extra="forbid"`） | 拒绝 |
| 4 | 出现 `scope` / `workspace` / `tenant` / `current_workspace` 字段 | 拒绝并明确报"v1 不支持作用域" |
| 5 | 权限码必须在 catalog 内；`"*"` 仅在 `permissions` 位置合法 | 拒绝 |
| 6 | `assignments` 引用的 `role_id` / `principal_id` 必须存在（无悬空绑定） | 拒绝 |
| 7 | user 必须恰好给出 `password_bcrypt` 或 `password_env` 之一；值必须是合法 bcrypt spec，cost ∈ [10, 15] | 拒绝 |
| 8 | **不接受明文口令**（policy 模式下 `AUTH_ACCOUNTS` 的明文兼容语义不延续） | 拒绝 |
| 9 | api_key 只能用 `secret_env`；引用的环境变量必须存在且非空，长度 ≥ 32，且**不同 key 的密钥不得重复** | 拒绝 |
| 10 | principal_id 满足 `^[A-Za-z0-9._@-]{1,64}$`（既做日志注入防护，也保证能进 JWT `sub`，与 `MAX_TOKEN_SUBJECT_LENGTH` 对齐） | 拒绝 |
| 11 | 新快照必须至少有一个**可认证的**（`enabled: true`）principal 有效持有 `auth.policy.reload` | 重载拒绝（`force=true` 可绕过，见 §9.6） |
| 12 | 文件权限：`st_mode` 含 group/other **可读**且文件内含 `password_bcrypt` 时告警（可写位归 #13 统一拒绝） | 告警 |
| 13 | **信任链解析**：允许符号链接，但解析路径上每一级目录、以及最终目标，都必须 owner ∈ {`root`, 服务 euid} 且**不含 group/other 可写位**；最终目标必须是 regular file。解析后从**同一个 fd** 读取字节 | 拒绝 |
| 14 | 启动期加载失败：policy 模式下**启动失败**（fail closed），绝不降级放行 | 进程退出 |
| 15 | **`TOKEN_SECRET` 必须显式设置且不等于 `DEFAULT_TOKEN_SECRET`**（仅启动期校验，非重载期） | 进程退出 |
| 16 | `credential_version` 为 ≥1 的整数，缺省视为 `1`（不是可选的自由文本；见 §8.2） | 拒绝 |

读取顺序固定为"读原始字节 → 算 digest → 再解析"，保证 `source_digest` 描述的正是被解析的那份内容（并发编辑不会出现"digest 与内容错位"）。

#### 规则 #13 为什么是"信任链解析"而不是"拒绝符号链接"

"拒绝符号链接"（`os.open(..., O_NOFOLLOW)`）这个直观方案**既不够严也过于严**：

- **不够严**：`O_NOFOLLOW` 只检查路径的**最后一段**。攻击者若能控制任意一级父目录（把 `/etc/lightrag` 换成自己的符号链接，或该目录 group 可写），照样能在下一次冷启动时注入整份策略。且只拒 other-writable 不够，还必须拒 group-writable。
- **过于严**：**K8s 的 ConfigMap / Secret 卷就是符号链接农场** —— `<mount>/<key>` 是指向 `..data/<key>` 的符号链接，`..data` 又是指向时间戳目录的符号链接，更新时通过原子换符号链接实现。一律拒绝符号链接，等于宣布策略文件不能用 K8s 卷挂载。

而且这恰好打在热重载的要害上：本仓库的 Helm chart 目前用 `subPath` 挂 `.env`（[k8s-deploy/lightrag/templates/deployment.yaml:43-45](../../k8s-deploy/lightrag/templates/deployment.yaml#L43-L45)），而 **`subPath` 挂载在 Secret 内容更新时不会同步** —— 想让策略文件可热重载，就**必须**用非 `subPath` 的卷，也就必然是符号链接形态。拒绝符号链接会把本方案最核心的能力锁死在容器外。

因此改为按"谁能影响解析结果"来判定：**允许符号链接，但要求解析路径上每一级目录与最终目标都不可被非受信 uid 写入**。这同时封掉了父目录注入（攻击者需要某一级的写权限，而那正是被拒的条件），又放行了 K8s 卷（其目录为 root 所有、`0755`，文件 `0644` 或配 `fsGroup` 时 `0640`，均无 group/other 写位）。实现要点：

- 逐级 `openat(..., O_DIRECTORY)`（**跟随**符号链接）+ `fstat` 校验每一级；最终 `fstat` 校验 `S_ISREG` 与 mode。
- 校验通过后**从同一个 fd 读取**，不要拿路径重新 `open` —— 否则校验与读取之间存在 TOCTOU，前面的功夫全白做。这也顺带满足 §7.3 末尾"digest 描述的正是被解析的那份内容"。
- 检测到符号链接不拒绝，但把解析后的真实路径写进启动日志（可审计，K8s 场景下能看出指向了哪个 `..data` 世代）。
- Linux 上可用 `openat2(RESOLVE_NO_MAGICLINKS)` 作为快速路径，但**不要**用 `RESOLVE_NO_SYMLINKS`（同样禁掉 K8s 卷）；macOS 无 `openat2`，逐级 `openat` 是唯一可移植实现，故以它为规范形态。

规则 #15 是**必须补的安全约束，不是锦上添花**：现状由 [auth.py:53-63](../../lightrag/api/auth.py#L53-L63) 保证 —— `AUTH_ACCOUNTS` 存在时若 `TOKEN_SECRET` 未显式设置就 `ValueError`。本方案把账号来源从 `AUTH_ACCOUNTS` 换成策略文件后，那条判断的触发条件不再成立，若不补等价规则，就会出现"配了策略文件、口令是 bcrypt、JWT 却用公开的默认 guest 密钥签发"—— 任何人都能自签一个 `sub=admin` 的令牌，整套授权面直接绕过。它归在启动期而非重载期，因为 `TOKEN_SECRET` 来自 env（进程启动态），重载读不到新值。

### 7.4 与 `AUTH_ACCOUNTS` 的优先级（必须明确）

- **`AUTH_POLICY_FILE` 未设置 ⇒ legacy profile，`AUTH_ACCOUNTS` 照旧工作**，且权限行为与今天逐字一致（每个通过认证的用户拿到 §11.4 的 `legacy_user` 预设权限集）。不设策略文件不是"降级"，是一个受支持的部署形态；本方案不逼迫任何存量部署迁移。唯一的差别是 `documents.artifacts.*`（下载类新能力）在这个形态下不可获得 —— 见 §11.4。
- `AUTH_POLICY_FILE` 已设置 ⇒ **策略文件是用户与角色的唯一权威**。
- 此时若同时设置了 `AUTH_ACCOUNTS`：**启动失败**，报"两个用户来源冲突"。这是刻意的 fail fast，避免"以为改了 env 生效、其实被文件覆盖"的安全误判。
- 只有显式设置 `AUTH_LEGACY_ACCOUNTS_COMPAT=true` 时，`AUTH_ACCOUNTS` 中的账号才被映射为内置 `legacy_user` 角色（见 §11），并在启动横幅打印警告 + 移除计划。

---

## 8. 凭据与会话

### 8.1 Provider 分层

```python
class CredentialProvider(Protocol):
    async def authenticate_user(self, username: str, password: str) -> Principal | None: ...
    async def authenticate_api_key(self, secret: str) -> Principal | None: ...

class AuthorizationProvider(Protocol):
    async def has_permission(
        self, principal: Principal, permission: Permission,
        context: AuthorizationContext | None = None,
    ) -> bool: ...
```

文件实现 `FilePolicyCredentialProvider` / `FilePolicyAuthorizationProvider` 各自持有 `PolicyRuntime` 引用；两者读同一快照但走独立校验输入，第二阶段可单独替换其中之一。

**Provider 异常一律 fail closed**：`has_permission` 抛异常 ⇒ 不放行、不得被误读成"空角色"、不得回落到 legacy 判定。状态码（为什么是 500 而不是 401/403）以 §10.2 为唯一权威。

### 8.2 JWT 声明与即时失效（credential epoch）

策略文件热重载解决了"权限即时生效"，但**已签发的 JWT** 仍会让被删除/禁用/改密的用户继续通行到 `exp`。方案：

```python
TokenPayload:
    sub: str      # principal_id（不透明）
    pt: str       # principal_type（"user"）
    ce: str       # credential epoch
    exp: datetime
    ver: int = 1  # 令牌格式版本
    role: str     # 仅供显示 / 向后兼容，授权永不读它
```

`ce` 的定义（**必须是确定性的**，否则重启即全员掉线）：

```python
credential_epoch = sha256(
    f"v1|{password_hash}|{int(enabled)}|{credential_version}".encode()
).hexdigest()[:16]
```

`credential_version` 是**显式的会话世代计数器**（策略文件里的用户字段，默认 `1`，校验规则 #16）。少了它，`ce` 就是 `password_hash + enabled` 的纯函数，也就**可逆**：

| 操作序列 | 无 `credential_version` 的后果 |
| --- | --- |
| 禁用 alice → 过一会儿重新启用 | `ce` 回到原值，**禁用期间签发前就存在的旧 JWT 全部复活**（只要还没到 `exp`） |
| 删除 alice → 之后用同一 hash 重建同名用户 | 同上，旧令牌连同旧会话一起复活 |

这两条不是理论风险：禁用是"先停权、查清楚再恢复"的标准运维动作，而它恰好在恢复的一刻把被停的会话还了回去。加上计数器后，规约是"**改密不必动它**（hash 变了，`ce` 自然变），**禁用后恢复、删除后重建必须 +1**"。

诚实交代边界：这条**依赖运维纪律**，服务端无法自动强制 —— 策略文件对服务端是只读的，没有任何可写处能放一个自增计数器。CLI（`lightrag-hash-password --format policy`）在改密时会顺手打印提醒，第二阶段的 DB provider 可以把它变成自动递增。

- 与 `revision` 无关：普通策略重载（改角色）不会踢掉在线会话。
- 改密 / 禁用 / 删除该用户 / `credential_version` 递增 ⇒ `ce` 变化或用户不存在 ⇒ 下一次请求 401（`detail="Session invalidated. Please login again."`）。
- 每次请求的认证依赖都做三件事：验签 + 校验 `exp` → 在当前快照里查 `sub`（不存在或 `enabled=false` ⇒ 401）→ 比对 `ce`（不等 ⇒ 401）。这三步都是快照内 O(1)。
- 升级过渡：policy 模式下缺 `ce` / `pt` 的旧令牌一律 401（用户重登一次）。这比"兼容旧令牌"更安全，且成本仅一次登录。
- `password_env` 用户的边界：其 `password_hash` 来自环境变量，热重载读不到 env 的新值，因此**改密要等 worker 重启才会改变 `ce`**（在线会话在重启前不会失效）。这与 §8.3 的 API Key 轮换是同一条分工 —— 秘密走部署通道、策略走文件通道。要"改密即时踢人"就必须用 `password_bcrypt` 写在文件里。

授权缓存：本方案**不缓存**授权结果（快照本身就是预计算），因此 RFC 的"缓存必须有 TTL 或 revision 失效"约束自然满足。

### 8.3 API Key

- 客户端仍只发 `X-API-Key: <secret>`；服务端 key ID 只存在于策略文件、principal、审计日志。
- 启动 / 重载时构建 `HMAC-SHA256(pepper, secret) -> principal_id` 索引：
  - pepper 为**进程内随机**（`secrets.token_bytes(32)`），无需配置项；索引只在本进程内查表，跨 worker 无需一致。
  - 明文密钥不进日志、不作为字典键、不进 principal_id。
  - 请求侧先做长度上限检查（>512 字符直接 401，避免为攻击者的大 payload 做 HMAC）。
- 密钥轮换：改 `secret_env` 指向的环境变量值不会被热重载感知（env 属于进程启动态），因此**轮换需要重启 worker**；策略文件里的 key ID / 角色绑定改动可以热重载。这是刻意的分工：秘密走部署通道，策略走文件通道。文档需写清这条边界。
- 遗留 `LIGHTRAG_API_KEY` 注册为内置 principal `("api_key", "legacy")`：**policy 模式下默认零权限**（必须在策略文件里显式绑角色）；legacy profile 下持有 `legacy_user` 预设（§11.4，行为保持）。两种模式下都**永不隐式获得 `documents.artifacts.*`**。

### 8.4 guest 令牌与各 profile 的语义

| profile | 触发条件 | `/auth-status` 行为 | guest 令牌 | principal 权限来源 |
| --- | --- | --- | --- | --- |
| `policy` | `AUTH_POLICY_FILE` 已设置 | 返回 `auth_configured=true`、`auth_backend="policy"`，**不签发** guest 令牌 | 不签发；已存在的 guest 令牌认证任何东西都失败 | 策略文件（唯一来源） |
| `legacy` | 未设 `AUTH_POLICY_FILE`（默认）—— **包含"`AUTH_ACCOUNTS` / `LIGHTRAG_API_KEY` 也都没配"的裸部署**，见下 | 与今天一致 | 与今天一致（仅在完全开放模式下有效） | 内建 `legacy_user` 预设（§11.4） |
| `unauthenticated` | 显式 `AUTH_ALLOW_UNAUTHENTICATED=true` | 返回 `auth_configured=false` | 签发；映射到 `("anonymous", "dev")` principal | 有策略文件 ⇒ `AUTH_ANONYMOUS_ROLE`（默认 `reader`）指名的角色；无策略文件 ⇒ 内建 `legacy_user`（等于今天完全开放模式的能力） |

#### 裸部署（三者皆未设）归属 —— 原文档的空洞，此处补齐

先前的 profile 触发条件要求 legacy 必须设置了 `AUTH_ACCOUNTS`/`LIGHTRAG_API_KEY`，`unauthenticated` 必须显式开 flag，于是**什么都没配的部署匹配不到任何 profile**。它恰恰是今天最常见的开发形态：[utils_api.py:359-361](../../lightrag/api/utils_api.py#L359-L361) 的 `credentials_accepted` 第一分支就是"既无 `AUTH_ACCOUNTS` 又无 API key ⇒ 一切放行"，guest 令牌全权通行。

裁决：**归入 legacy profile 的开放子形态**，行为与今天逐字一致（匿名调用者持 `legacy_user`，因此 `documents.artifacts.*` 仍拿不到）。理由是它直接落在"未设策略文件 ⇒ 行为不变"这条兼容性原则之内；反之若要求裸部署显式加 flag 才能启动，就是一条**发生在 policy 模式之外**的破坏性变更（所有 quickstart / 开发环境启动即失败），必须进 §11.2 清单 —— 那与本方案"不逼迫存量部署迁移"的取向冲突。把开放形态映射成一个**有名字、被枚举钉住的**预设还有一层意义：`documents.artifacts.*` 这类新泄露面落在预设之外（§11.4），最容易在公网上裸奔的部署形态想要下载能力，就必须显式写策略文件。

**这不与"`unauthenticated` 绝不能因凭据缺失被推断出来"这条红线矛盾**：红线约束的是**policy 模式**下凭据解析失败/缺失时不得退化成匿名放行（那是安全事故），不是"历史上就开放的裸部署必须启动失败"。两者的区分点是有没有 `AUTH_POLICY_FILE`：设了就 fail closed（规则 #14），没设就保持历史行为。

`unauthenticated`（显式 flag）与裸部署都必须在启动横幅打印醒目警告并声明不适合网络暴露 —— 这是本方案对裸部署新增的**唯一**动作（只加日志，不改判定）。

> 若评审倾向另一条路（裸部署启动失败 / 全 401），需要同时：把它加进 §11.2 作为第 8 条破坏性变更、修改 §11.3 迁移序列第 1 步（"行为不变"不再成立）、并给 quickstart 文档补必配项。本节按"保持行为"落笔，改判只需替换这一小节。

### 8.5 同时携带 JWT 与 API Key 时的凭据优先级

`authorize` 同时接收 `Authorization: Bearer` 与 `X-API-Key`（§10.1），而 **WebUI 确实会同时发这两个 header**（[lightrag_webui/src/api/lightrag.ts](../../lightrag_webui/src/api/lightrag.ts)：拦截器无条件附加已配置的 API Key，登录后再叠加 Bearer）。如果 JWT 属于用户 A、API Key 属于服务账号 B，而两者权限不同，就必须有确定性答案。

**规则（policy 模式）：**

| 情形 | 判定 |
| --- | --- |
| 有 `Authorization` header | **JWT 是唯一凭据**。有效 ⇒ principal = 该用户；无效 / 过期 / `ce` 不匹配 ⇒ **401，不回退 API Key** |
| 无 `Authorization`，有 `X-API-Key` | principal = 该 key 对应的服务账号 |
| 两者都无 | 401 |

**严禁权限并集**：任何"取两者权限的 union"都会造成"用低权用户登录 + 浏览器里存着高权 API Key ⇒ 拿到高权"的越权路径，且审计日志无法回答"到底是谁做的"。授权与审计（`principal_type` / `principal_id`）、`/auth/me` 的返回身份，全部用同一个解析结果 —— 单一 principal，不做合并。

这条不是新发明，而是把**现状固化**：今天 [utils_api.py:405-460](../../lightrag/api/utils_api.py#L405-L460) 就是"先验令牌，令牌无效直接 401（`except HTTPException: if 401: raise`），不落到后面的 API Key 分支"。唯一需要保留的历史分支是 **legacy 的 guest 令牌**：API-key-only 模式下 guest 令牌认证不了任何东西（GHSA-f4vv-55c2-5789），必须"忽略它、继续走 API Key"而不是 401 —— 该分支只在 legacy profile 存在，policy 模式不签发 guest 令牌，故不适用。

**代价要说清：令牌过期即跳登录页，即便同时带着有效的 API Key。** 现状的前端就是这样 —— 拦截器无条件附加已配置的 API Key，而 401 在非 guest 分支直接 `navigateToLogin()`，不做"去掉 `Authorization` 再试一次"的重试（[lightrag.ts:414-430](../../lightrag_webui/src/api/lightrag.ts#L414-L430)、[:516-518](../../lightrag_webui/src/api/lightrag.ts#L516-L518)）。

本方案**不引入**"删掉 Authorization 后自动回落到 API Key"的重试：它需要一个防循环标记、会让"我明明登录了却以服务账号身份在操作"变得不可见，而收益仅仅是省掉一次登录点击。§12 的 401 分流（清 token、跳登录）是唯一口径，与这里逐字一致。要用纯 API Key 的客户端就不要发 `Authorization` header。

`credentials_accepted()`（ASGI 层布尔咽喉）必须用**同一套优先级**，否则中间件与路由依赖会对同一请求解析出不同 principal —— 这正是 §10.3 第 4 条"同源"的含义。

---

## 9. 热重载设计

### 9.1 端点

| 方法 | 路径 | 权限 | 语义 |
| --- | --- | --- | --- |
| POST | `/auth/policy/reload` | `auth.policy.reload` | 校验并换入新快照，广播给全部 worker |
| POST | `/auth/policy/validate` | `auth.policy.read` | 纯 dry-run：只校验当前磁盘文件，**无任何副作用** |
| GET | `/auth/policy/status` | `auth.policy.read` | 目标版本 + 每个 worker 的采纳状态 + 是否收敛 |

`/auth/policy/validate` 是自锁保护的第一道闸：运维改完文件先 validate，再 reload。

### 9.2 单进程流程（uvicorn 单 worker）

```
POST /auth/policy/reload
  → 读原始字节（受 MAX_BYTES 约束）→ digest
  → strict 校验 + 构建候选快照（§7.3 全套规则，锁外；此时还没有 revision）
      ├─ 失败：HTTP 422，返回结构化错误清单；旧快照原封不动
      └─ 成功：进 auth_policy 锁 → 取 revision → 发布共享 epoch + 原始字节
                → apply_snapshot(定稿快照)  ← 换入咽喉：swap + confirm（纯本地）
                → 出锁 → maybe_publish()  ← 共享上报调度点，自己取锁，失败不致命
                （取号与换入同在锁内，理由见 §9.3 写入侧；必须走 apply_snapshot，
                  否则发起 reload 的 worker 自己不会 confirm，刚换入就 503）
  → 200 { revision, digest, applied_at, converged, workers: [...], self_lockout }
```

在途请求持有的是旧快照引用（§4.3），因此**不存在半新半旧的判定**。

### 9.3 多 worker 传播（gunicorn N workers）

问题：`POST /auth/policy/reload` 只会打到一个 worker。

两条必须同时成立的约束：

1. **唯一采纳信号 = 共享 epoch 记录**，不是文件 mtime。若允许 worker 因为"文件变了"就自行采纳，那么编辑文件本身就成了重载触发器，绕过 validate-then-apply 语义。
2. **全部 worker 采纳的必须是"被校验过的那一份字节"**。让每个 worker 自己重读磁盘会把"校验的内容"与"采纳的内容"分离：reload 与某个 worker 的重读之间只要文件再被改一次，该 worker 就无法采纳，舰队从此 split-brain（部分 rev 7、部分 rev 6），且只能靠运维看 status 发现、再 reload 一次。因此**内容随 epoch 一起进共享内存**。

#### 共享状态布局

`shared_storage` 命名空间 `auth_policy`，写操作在 `get_namespace_lock("auth_policy")` 内。**header 与内容字节分成两个 key** —— 这是性能要件，不是风格问题：稳态 tick 只读 header，内容只在 revision 变化时取一次。放进同一个 value 会变成每 tick 每 worker 拉一次整份文件（最坏 1 MiB × N worker × 0.5 次/秒），正撞"proxy RPC 成本 = 次数 × 字节"的坑。

```python
{
  "epoch": {                      # 小 header；稳态 tick 只读这一个 key
     "revision": 7,
     "digest": "sha256:...",
     "revision_source": "reload",  # "disk"（冷启动发布）/ "reload"
     "requested_at": 1770000000.0,
     "requested_by": "user:admin",
  },
  "content": b"version: 1\n...",  # 被校验过的原始字节（sha256 == epoch.digest）
  "adoptions": {                  # pid -> 采纳报告，整体替换写入
     "51231": {"pid": 51231, "start_id": "...", "revision": 7,
               "digest": "sha256:...", "revision_source": "reload",
               "adoption_source": "shared", "adopted_at": ..., "state": "ok",
               "message": "", "reported_at": 1770000042.0},
     "51232": {"pid": 51232, ..., "revision": 6, "state": "error",
               "message": "missing env var for user alice",
               "reported_at": 1770000040.0},
  },
}
```

`reported_at` 是**新鲜度心跳**，不是装饰：采纳报告只在状态变化时写（省 RPC），于是一个**轮询任务停摆**的 worker 根本没有机会把自己写成异常 —— 共享记录还是它上一次的 `ok`，`/auth/policy/status` 甚至会报 `converged=true`，而那个 worker 的请求已经因本地时钟过期在返 503。这是"远端可观测性"与"只在状态变化时写"之间的直接冲突，只能靠低频心跳解决：轮询任务每隔至多 `AUTH_POLICY_HEARTBEAT_SECONDS`（默认 `10.0`，≥ 5 × 轮询间隔）刷一次自己的 `reported_at`，即使状态没变。每 worker 每 10 秒一次小写入，相对管线自身的 RPC 量可忽略。**心跳不是一条独立的代码路径，而是上报调度点 `maybe_publish()` 的到期分支**（附录 A.1）——tick 的每条退出路径都必须经过该调度点，否则"稳态分支 `confirm()` 后直接 return"就会让心跳只存在于文字里，首次注册记录必然在 `2 ×` 心跳间隔后被误判 `unresponsive`。

> Manager dict 陷阱：`adoptions` 是普通 dict 作为 value，**必须整体读出、修改、整体写回**（嵌套 plain dict 的原地修改不会跨进程传播）。
>
> 记录 schema 是**扁平的顶层字段**，且三种 `state`（`ok` / `pending` / `error`）与心跳路径写的都是**同一套完整字段**（`publish_report` 只收 `state` / `message`，版本字段一律取自本地已换入的快照）。任何"只写 state 和 message"的增量写法都会把版本字段抹掉，让这个 worker 从此无法参与 `(revision, digest)` 收敛比较；任何把 `target` 整个塞进记录的嵌套写法同样读不出来。注意 `revision`/`digest` 记的是**该 worker 正在服务的版本**，不是它想采纳的目标 —— 正因如此，一个 `error` 状态的 worker 会如实报出自己仍停在旧 revision，`converged` 自然为假。

**只共享字节，绝不共享构建好的 `PolicySnapshot`。** 快照里有 `api_key_index`（`HMAC(pepper, secret) -> principal_id`），共享它就得让 api_key 明文或 pepper 途经 Manager 进程；而共享的字节里只有本就落盘的 bcrypt 摘要，风险等级不变（同 uid 本来就能读那个文件）。每个 worker 从同一份字节各自构建快照、各自生成进程内随机 pepper（§8.3）。

内容常驻内存是有界的：字节数已由 `AUTH_POLICY_MAX_BYTES`（默认 1 MiB）在读取前卡住，共享副本只有一份。

#### 冷启动初始化事务（必须有唯一 leader）

**这一步不能省，否则第一次 reload 永远无法传播。** 若每个 worker 在 epoch 缺失时各自从磁盘建立本地 `revision=1` 而**不发布**，那么首次 reload 算出的 `next_rev` 也是 1（epoch `ABSENT` ⇒ `0 + 1`），其它 worker 看到 `epoch.revision == local.revision` 就直接跳过 —— 即使 digest 完全不同，新策略也永远采纳不了。

协议（每个 worker 在 lifespan startup 各跑一次）。**两条顺序要求是这段协议的全部要点**：先查 epoch 再决定要不要碰磁盘；构建永远在锁外。

协议由六个原语构成；**规范形态的完整伪代码（含三态返回与调度点的注释级推导）收在附录 A.1–A.2**，正文只列每个原语必须满足的不变式：

- `shared_get(key)` —— 三态读取：`VALUE` / `ABSENT`（确认键不存在，是合法的冷启动信号）/ `RETRY`（读不出来、真假未知）。必须用 `shared[key]` 而非 `.get()`（后者把"不存在"塌成 None，正好毁掉 ABSENT/RETRY 区分）；写侧不变式配套保证 epoch 只写完整记录、永不写 None，读到 `None` 保守当 RETRY。
- `apply_snapshot(snap)` —— **唯一换入动作**：`swap + confirm` 绑死，纯本地、零 RPC，锁内外均可安全调用；谁都不许单独做 swap（§9.4 第 4 点）。
- `publish_report(state, message)` —— 独立共享上报：自己取 `auth_policy` 锁，**只能在未持锁时调用**；`adoptions` 整体读-改-写，读到 RETRY 就放弃本次写（绝不当空字典覆盖别人的记录）；写的永远是完整顶层记录，版本字段一律取自本地已换入的快照 —— 记的是该 worker **正在服务**的版本，不是它想采纳的目标。
- `maybe_publish()` —— **唯一上报调度点**，所有主动上报（注册、reload 出锁后、tick 的每条退出路径）都走它：意图记录变了立即写；没变只在距上次成功写入 ≥ `AUTH_POLICY_HEARTBEAT_SECONDS` 时写（这就是心跳的唯一实现）。进程内 `publish_mutex` 串行化**整个调度点**（rec 与 due 都在锁内重算），否则旧心跳与 reload 交错会把旧 revision 乱序覆盖回共享表；调度状态只在写成功后推进，且**绝不参与 `is_stale()`**。`intended_record()` 里 `ok` 只留给 CONFIRMED / ADOPTED —— 确认不了自己在目标上的 worker 如实自报 `pending`，堵住"epoch 读坏、adoptions 写好"的假收敛（见下"收敛判定"）。
- `adopt_from_shared(target)` —— **唯一采纳咽喉**，启动与轮询共用，digest 守卫只有这一份实现（下面"撕裂读取"说明为什么）：`sha256(content) != target.digest` ⇒ RETRY；锁外 `build()`；构建后复检 epoch。RETRY 必须带原因（`UNREACHABLE` / `CONTENDED`），否则调用方填不了 `last_attempt_kind`。
- `startup()` + `register_or_die()` —— 冷启动初始化事务：第一步查 epoch（RETRY ≠ ABSENT，任何一侧的混淆都是事故，见下"四点实现约束"）；epoch 存在 ⇒ **绝不读磁盘**，直接 `adopt_from_shared`；ABSENT ⇒ 锁外读盘校验出不带 revision 的 draft，锁内复查并抢发布权（锁内只有 1 读 + 2 写 + 一次 O(1) `replace`）；首次上报（`register_or_die`，同样走调度点）是 lifespan `yield` 之前的**启动硬前置**，有界重试后仍失败 ⇒ worker 起不来。

#### 撕裂读取：`epoch` 与 `content` 是两次独立读取

发布侧在锁内写两个 key（先 `content`、后 `epoch`），**但读侧不持锁**，所以读到的两者可能来自不同世代：

```
启动进程                          reload 进程（持锁）
1. 读 epoch = rev7 / d7
                                  2. 写 content = raw8
3. 读 content = raw8
4. build(raw8, target=rev7)
5. 复检 shared["epoch"] == rev7 ✓（epoch 还没写）
                                  6. 写 epoch = rev8
7. 换入"标称 rev7 / 实为 rev8"的快照并上报 ok
```

后果不是"晚一点收敛"，而是**上报了一个假的收敛**：`/auth/policy/status` 会显示该 worker 已在 rev7/d7，实际执行的是 rev8 的授权。让写侧持锁不能解决 —— 问题在读侧跨了两次 RPC。轮询路径早就有 `sha256(raw) != epoch.digest ⇒ 不采纳` 这道守卫，启动路径漏了同一道；这类"同一个不变式有两处实现、只补了其中一处"正是要靠**单一咽喉**消除的，所以 digest 守卫收进 `adopt_from_shared`，启动与 tick 都只能走它。

四点实现约束：

- **共享上报绝不能被塞进换入动作里**。`apply_snapshot()` 从两种位置被调用：冷启动 leader 与 reload 发起者**持着 `auth_policy` 锁**调它，follower 与 tick **不持锁**调它。若它内部写共享 `adoptions`，两条路都堵死：
  - 上报自己取锁 ⇒ 持锁调用变成同协程重入，而 [`NamespaceLock.__aenter__` 明确抛 `RuntimeError("NamespaceLock already acquired in current coroutine context")`](../../lightrag/kg/shared_storage.py#L3764-L3768)（即便没有这道断言，底层 keyed lock 也不可重入，结果是自锁死）；
  - 上报不取锁 ⇒ tick / follower 的锁外调用会并发做 `adoptions` 的整体读-改-写，互相覆盖，**丢掉别的 worker 的报告与心跳**。

  所以换入咽喉收窄成**纯本地的 `swap + confirm`（零 RPC，锁内外都能安全调用）**，共享上报拆成独立的 `publish_report()`，它自己取锁、**只允许在未持锁时调用**。持锁路径（leader、reload 发起者）必须**出锁之后**再上报。

  另一半同样重要：**上报失败不是授权失败，`publish_report()` 自己也从不升级失败** —— 它只返回成败。已经发布并换入成功的 reload 不得因为写 `adoptions` 失败而返错；运行期 tick 的上报失败只是观测通道的问题，由调度点在下一 tick 重试、心跳自愈。**把持续失败升级成致命错误的地方只有一个：启动期的 `register_or_die()`**（见下）—— 升级逻辑住在启动路径里，不在上报原语里，所以"reload/运行期不致命"与"启动期注册不成功就起不来"这两条并不矛盾，测试也必须分开断言（§13）。

  **但"不致命"必须配一个启动硬前置，否则会伪造收敛。** `_pid_alive` 只能校验**已有记录**的 pid，共享布局里没有任何 worker 名册，所以一个"换入成功、首次上报失败"的 worker 对 `/status` 是**完全不可见**的 —— 别的 worker 只看已有记录，照样报 `converged=true`，而那个隐形 worker 正按旧 revision 服务。这是最坏的一类假收敛（CI 门禁绿灯、实际未收敛），"存活 pid 无记录也算未收敛"这条要求本身也无从实现，因为无从枚举"应该有哪些 pid"。

  **仅仅让它 503 不够**：进程照样起来了、`/health` 照样 200、照样在接流量，而它的 pid 从未进过 `adoptions` —— 远端依旧看不见它，`converged` 依旧可能为真。"不可见"这个问题不能靠"可见地拒绝服务"来解决。

  裁决：**首次注册是 lifespan 启动的硬前置**（`register_or_die()`，在 `yield` 之前执行）。有界重试后仍失败就抛出，worker 起不来 —— 与"启动期采纳失败 ⇒ 进程退出"（规则 #14）同一条口径：一个连共享状态都写不进去的进程根本无法参与传播协议，让它崩溃循环是**可见**的，而让它带着隐形身份服务是不可见的。

  于是不变式成立：**凡在服务的 worker，共享表里必有它的记录**（注册发生在开始接流量之前）。`converged = 全部已知记录都新鲜、自报 ok 且在目标上` 由此可靠，不需要外部名册。分工也就清楚了：**启动期注册失败 ⇒ 退出；运行期上报失败 ⇒ 不致命**，前者建立可见性，后者保证已建立的可见性不因抖动拖垮授权（记录还在，只是 `reported_at` 变旧，超期即 `unresponsive`）。

  补充（**仅供人眼，不进 `converged` 判据**）：arbiter 在 pre-fork 时已经知道 worker 数（[run_with_gunicorn.py:289](../../lightrag/api/run_with_gunicorn.py#L289) 把 `workers_count` 传给 `initialize_share_data`），可以把它写成 `expected_workers` 供 `/status` 对照，用来发现"某个 worker 压根没起来"。它不进收敛判据，因为 `TTIN`/`TTOU` 运行期改并发数会让这个数字过期，而收敛断言是 CI 门禁，不能建立在会过期的数字上。
- **"epoch 缺失"在锁内锁外必须是同一个判定**。锁外用 `shared_get` 得到 `ABSENT` 才进冷启动，锁内却写 `shared["epoch"] is None` —— 两种表示不通：键真的不存在时，锁内那句会抛 `KeyError`；而键存在且值为 `None` 时，锁外早就被折成 `RETRY`、根本进不来。**没有任何初始状态能走通这条路径**，冷启动一样死。锁内必须复用同一个 `shared_get`，并把它的 `RETRY` 处理成"出锁重试本轮"（锁内绝不重试，不占着跨进程锁等 RPC）。
- **`ABSENT` 与 `RETRY` 必须是两个不同的返回值**。把两者都折成"失败"会造出一个致命的死结：`startup` 正是用"epoch 不存在"判定冷启动，若那也是 `RETRY`，第一个 worker 会一直重试到 `MAX_STARTUP_ATTEMPTS` 耗尽然后退出 —— **全新部署永远选不出 leader、永远起不来**。反方向同样致命：把 `RETRY`（读不出来）当成 `ABSENT`（确认不存在），会让一个 Manager 短暂抽风的 worker 以为自己在冷启动，用磁盘内容发布一个 `revision=1` 覆盖掉舰队正在跑的 rev 7。**"不存在"与"读不出来"是两件事，任何一侧的混淆都会造成事故。**
- **重试必须有界**：`MAX_STARTUP_ATTEMPTS` 建议 5，之间让出事件循环。撕裂只在"发布进行中"这一瞬发生，重试一次几乎必然成功；连续多次失败说明有别的问题（例如共享状态被外力清空），此时启动失败退出比无限重试更可诊断。
- **冷启动 leader 自己发布的字节不需要 digest 守卫**：`raw` 就在手上，`target` 是它自己写的，不存在跨世代问题。落败者则必须走 `adopt_from_shared`。
- **leader 也不在锁内 `build()`**：`read_and_validate_file()` 在锁外就把重活做完，返回一个**不带 revision 的 draft**；锁内只有 1 读 + 2 写 + 一次 `replace(draft, revision=1)`（`dataclasses.replace` 是 O(1) 的字段替换，不是构建）。这与 §9.3 写入侧 reload 的做法逐字相同 —— 取号必须在锁内，所以定稿也在锁内，但重活一律在锁外。
- **`won` / `target` 在进锁前初始化**：`won=False, target=None`。伪代码若只在 follower 分支里给标志位赋值、出锁后又无条件读它，leader 路径会撞 `UnboundLocalError`，表现为"首个 worker 启动即失败"。这里改成 leader 分支在锁内直接定稿并 `return`，与 follower 的后续路径彻底分开，不再依赖任何跨分支标志位的初始化。

不把 `epoch`/`content` 这对读取塞进锁内的原因和 `build()` 一样 —— 锁内只放常数时间操作。digest 守卫在锁外达到同样的正确性（`digest` 是内容的自证），代价只有一次 sha256。

#### 三条必须理解的理由

1. **"先查 epoch"不是优化，是正确性**。若先无条件 `read_and_validate_file()` 再查 epoch，那么"舰队跑着 rev 7、运维正在编辑磁盘文件还没 reload、此时一个 worker 崩溃重启"这个完全正常的场景下，替换 worker 会读到半写的文件并**启动失败**，而它本该直接采纳现成的 rev 7 —— §9.5 声称"重启优先采纳共享内容"就成了空话，§16 里标为"已解决"的重启漂移也会以另一种形式回来（这次是拒绝启动而不是漂移）。**替换 worker 完全不碰磁盘。**
2. **`build()` 必须在锁外**。它要跑 bcrypt spec 校验、`effective` 预计算等全部重活；放进 `get_namespace_lock("auth_policy")` 会让一个 worker 的启动阻塞所有 worker 的 reload 发布，与"锁内只有常数时间操作、无 I/O"直接冲突。锁内只有 1 次读 + 2 次写。
3. **构建后要重新确认 epoch**。构建期间可能落了一次 reload，此时手上的快照已经过期；重新走一遍协议（此时 epoch 非空，直接采纳共享字节，不会再读磁盘）。

冷启动时 N 个 worker 会并发各读一次磁盘，只有一份字节胜出，落败者丢弃自己的候选、改用共享字节重建一次 —— 这点浪费换来的是"leader 发布的必定是已校验通过的字节"。**不能反过来"先发布未校验的字节、让大家一起构建"**：那样一份坏文件会被写进共享内存，之后每个重启的 worker 都从共享内容采纳同一份坏字节、无一能启动，而磁盘上的修复不会被读取（epoch 非空 ⇒ 不读磁盘）—— 变成需要整体冷重启才能解开的死结。

也顺带消掉一个隐性竞态：两个 worker 冷启动相差几十毫秒、中间文件被改过一次，早期设计下它们会带着不同字节各自宣称 `revision=1`。现在只有胜出者的字节算数。

> **为什么不用"锁内认领 `init_owner`、只有 leader 读盘、follower 等 leader 发布"**：那会引入一个新的失败模式 —— leader 认领后、发布前崩溃（或 `read_and_validate_file()` 因规则 #14 退出），所有 follower 就永久等待，需要给认领加租约与超时重认领，把启动路径变成一个小型选举协议。上面的形态没有任何等待：谁先进锁谁发布，落败者立刻拿到可用字节，任何一个 worker 崩溃都不影响其它 worker 启动。代价只是冷启动时 N-1 次多余的构建。

**启动期采纳失败 ⇒ 进程退出**（规则 #14），不进入下面的 stale 状态：一个从未持有过有效快照的 worker 无法服务任何请求，崩溃循环虽然难看，但它是可见的，而"起来了却全 503"会被误读成业务故障。

#### 写入侧（reload 端点）

`revision + 1`、`epoch` 写入、`content` 写入必须在**同一个临界区**内完成，否则会出现"别的 worker 看到 rev 7 却取到 rev 6 的字节"。并发 reload 打到两个 worker 时 last-writer-wins，`revision` 保持单调。临界区形态（完整伪代码见附录 A.3）：锁外读文件、算 digest、构建**不带 revision** 的 draft；锁内 `shared_get("epoch")` 三态判定（RETRY ⇒ 出锁、端点回 503）→ `next_rev = 1 if ABSENT else revision + 1` → 依次写 `content` 与 `epoch` → `apply_snapshot(replace(draft, revision=next_rev))`；出锁后 `maybe_publish()`（失败只记日志、下一 tick 重试，不影响本次 reload 的成功语义）。

**取号必须在锁内，所以 `swap()` 也在锁内**：快照的 `revision` 字段直到锁内算出 `next_rev` 才知道，锁外先做的是"不带 revision 的校验与构建"（全部重活，包括 bcrypt spec 校验与 `effective` 预计算）。锁内只有取号、两次小写入、一次属性赋值 —— 临界区仍是常数时间、无 I/O。若把 `swap()` 挪到锁外，本进程会短暂持有一个 revision 与共享 epoch 不符的快照，`/auth/policy/status` 就可能把发起 reload 的那个 worker 自己报成 `pending`。

#### 读取侧（每 worker 的轮询任务）

轮询任务宿主是 `lightrag_server.py` 的 `lifespan`：startup 起 task、shutdown 取消并 `await` 回收（gunicorn 下每个 worker 各跑自己的 lifespan，正好是"每 worker 一个轮询器"；单进程 uvicorn 同样只有一个）。间隔 `AUTH_POLICY_RELOAD_POLL_SECONDS`（默认 `2.0`）。

tick 的结构（完整伪代码见附录 A.4）= 先 `step()`（纯判定与换入，**绝不直接调 `publish_report`**），再经唯一调度点 `maybe_publish()` 退出 —— **tick 的每条退出路径都必须经过调度点**，心跳才真的有发送路径（稳态且未到期时它是零 RPC 的本地比较）。`step()` 的每条退出路径要么 `confirm()`（稳态确认 `local == epoch`，或经 `adopt_from_shared` 成功换入），要么留下 `last_attempt_kind`（UNREACHABLE / CONTENDED / ERROR；`build()` 抛 `ValidationError` 属确定性失败，立即置 `deterministic_error`）。进程内状态只有三样：`last_confirmed_at` 单时钟（初值 `None` ⇒ `is_stale()` 为真）、`deterministic_error`、`last_attempt_kind` 成因枚举 —— **时钟判"是否 stale"、枚举判"为什么"**，不需要第二个时钟；`STALE_AFTER` 的定义见 §9.4。

要点：

- **follower 完全不读磁盘**，所以"reload 之后文件又被改过"不再影响传播 —— 原先设计里的 `drift` 状态**被彻底移除**，采纳状态收敛为 `ok` / `error` / `pending` 三种 —— 三种都是 worker **自报**的（`pending` 由 `last_attempt_kind` 处于 `UNREACHABLE` / `CONTENDED` 推导，见 `intended_record()`），不是 `/status` 从版本落后倒推的。冷启动的磁盘-共享内容分歧不是一种采纳状态，而是上报里的 `revision_source` / `adoption_source` 字段（§9.5）。
- 稳态每 worker 每 2 秒一次小 RPC、零文件 I/O。这个开销相对摄取管线自身的 proxy RPC 量可忽略；**轮询无法去掉**：Manager 没有廉价的跨进程推送原语，`Event.wait` 要么每 worker 常驻烧一个线程并长期占住一条 Manager 连接（更差），要么退回轮询。共享内容是对轮询的**修正**，不是替代 —— 轮询保留为触发机制。
- 校验失败是**确定性失败** ⇒ 报 `error` 并**立即**进入 §9.4 的 stale 状态（受保护请求 503），**不享受宽限期、更不是"拿旧快照继续服务"**。**唯一还会 per-worker 分叉的原因是 env**：`password_env` / `secret_env` 按各 worker 自己的环境解析（规则 #7 / #9）。fork 模型下所有 worker 继承同一份 env，所以实际极难发生；但路径必须存在，且必须 fail closed。
- 采纳报告的写与不写**全部收在 `maybe_publish()` 一个调度点上**：意图记录（state / message / 版本字段）变化 ⇒ 立即写；不变 ⇒ 只在距上次成功写入 ≥ `AUTH_POLICY_HEARTBEAT_SECONDS` 时写。两个方向的退化都被它挡住：稳态**和持续 `error`** 都不是每 tick 一次写 RPC，而心跳也不再依赖"哪条路径记得刷 `reported_at`" —— tick 的每条退出路径（含稳态 `confirm()`）都经过调度点，`reported_at` 必然按期刷新。写失败不推进调度状态，下一 tick 自动重试。调度状态（`last_published` / `last_published_at`）是观测侧的，绝不进 `is_stale()`。一个显式接受的故障模式成本：Manager 抖动会让 `ok` ↔ `pending` 交替，每次翻转都是记录变化 ⇒ 立即写，最坏在抖动期间退化为每 tick 一写 —— 不做去抖，因为去抖推迟的正是 `pending` 的远端可见性；"不每 tick 写"的成本约束只针对稳态。
- `/auth/policy/status` 汇总时用 `shared_storage._pid_alive` / `_process_alive` 剔除已死 pid 的陈旧记录（同时顺手清理，保持 `adoptions` 有界）。

收敛判定：`converged = 全部已知记录都"报告新鲜"、state == "ok"、且 (revision, digest) == epoch 的 (revision, digest)`。**三个条件缺一不可**：只查新鲜度与版本元组的公式有一个漏洞 —— epoch 读持续失败而 `adoptions` 写正常的 worker 停在与 epoch 相同的 rev7 上，记录新鲜、版本匹配，公式给出 `converged=true`，而它本地时钟早已过期、正对受保护请求返 503。`state == "ok"` 这个条件配合"未定失败自报 `pending`"（`intended_record()`）恰好堵住它：确认不了自己在目标上的 worker 写出来的就不是 `ok`。它之所以可靠，靠的是"注册即 ready"这条不变式（见上）：凡在服务的 worker，共享表里必有它的记录，所以"已知记录"就是"在服务的 worker"全集。"报告新鲜"= `now - reported_at <= 2 × AUTH_POLICY_HEARTBEAT_SECONDS`；超期的 pid 归入 `unresponsive_workers`，**并让 `converged` 为假**。响应里给 `pending_workers`（自报 `pending` 或版本落后）/ `failed_workers` / `unresponsive_workers` 三个列表，运维和 CI 都能断言。

**`/status` 能报什么、不能报什么（必须写清，否则会被过度信任）**：

| 视角 | 能区分的 | 不能区分的 |
| --- | --- | --- |
| 远端 `/auth/policy/status`（读共享 `adoptions`） | `ok` / `pending` / `error`（这三种是该 worker 自己写进共享状态的）、以及 `unresponsive`（报告超期）。**读坏写好的部分故障**（epoch 读持续失败、`adoptions` 写正常）会以新鲜的 `pending` 记录呈现，远端可见 | **共享完全不可达（读写皆失败）时，`unreachable` 与 `stalled` 分不开** —— 两者的现象完全相同：那个 worker 没在跟共享状态说话，只剩报告超期的 `unresponsive`。这种情形下原理上就不可分，因为唯一的信息通道正是坏掉的那条 |
| 该 worker 本地（日志 + 它自己 503 响应体的 `cause` 字段） | `error` / `unreachable` / `stalled` / `contended` 可分 —— 靠 `last_attempt_kind` 这个不带时间的枚举（§9.4 第 4 点给出推导） | — |

所以 §9.4 第 4 点那三种成因的区分**是本地能力**：运维先从任一健康 worker 的 `/status` 看到"哪个 pid 不响应"，再去那个 pid 的日志里看是 `unreachable` 还是 `stalled`。文档与测试都不得声称 `/status` 能远端区分这两者。

### 9.4 stale worker 状态机（未采纳 ⇒ 停止授权）

**问题**：rev 6 允许 alice 清库，rev 7 撤销该权限。某个 worker 因故无法构建 rev 7。若它继续用 rev 6 服务，请求一旦落到它身上，alice 就还能清库 —— 已撤销的权限无限期有效。这直接违反 §1.1 目标 6，也是"保留旧快照"这一措辞最容易被实现成的样子。

**状态机**（每 worker 进程内，由轮询 tick 驱动）：

分歧的成因分两类，这是状态机的分界线：

- **确定性失败**：`build()` 抛 `ValidationError`。这个 worker **已经知道**自己采纳不了目标 revision，再等下去不会有任何变化。
- **未定失败**：还没拿到一致的内容 —— `sha256(content) != epoch.digest`（撞上并发发布）、共享读 RPC 超时/报错、本 tick 尚未跑到。结果未知，可能下一 tick 就成功。

| 状态 | 进入条件 | 受保护请求 | 公共 liveness |
| --- | --- | --- | --- |
| `ok` | 最近一次 tick 正向确认了 `local == epoch`（或成功采纳） | 正常授权 | 正常 |
| `pending` | 未确认，但距上次正向确认 < `STALE_AFTER` | **仍按旧快照服务** | 正常 |
| `stale` | `deterministic_error` 已设（`ValidationError`）—— **立即，不走宽限期**；或 `monotonic() - last_confirmed_at > STALE_AFTER`（`STALE_AFTER = max(AUTH_POLICY_STALE_GRACE_SECONDS, 3 × AUTH_POLICY_RELOAD_POLL_SECONDS)`，公式以本行为唯一权威；涵盖未定失败持续、共享读不通、轮询任务停摆三种成因，见下第 4 点） | **一律 503** + `Retry-After: 5` | 正常（`/health` 最小载荷、`/auth-status`） |

三个关键设计点：

1. **`pending` 只服务"未定失败"**。若"看到更高 epoch 就立刻 503"，每次正常 reload 都会让全舰队闪一次 503（传播天然有一个轮询周期的延迟）；宽限期默认 `10.0` 秒（≥ 2 个 tick）就是给这段传播用的。
2. **确定性失败绝不享受宽限期**。rev 7 撤销了 `documents.clear`，而这个 worker 已经明确知道自己构建不出 rev 7 —— 让它再放行 10 秒旧权限，是在一个已知的撤权失败上人为开一个 10 秒的窗口，没有任何收益。宽限期存在的唯一目的是遮掉传播延迟，而传播延迟的表现形式恰恰是"未定"，不是"确定失败"。
3. **未定失败到点也必须进 `stale`**。否则"永久 pending"（`content` 键被外力清掉、RPC 持续超时）就是一条绕过 fail-closed 的静默通道 —— 对这一类，**持续时间**才是判据。
4. **判据是"正向确认"，不是"发现分歧"** —— **授权判据**只有一个时钟 + 一个标志位，别无其它（§9.3 的上报调度状态 `last_published` / `last_published_at` 属观测侧，绝不参与 `is_stale()`，不违反本条）。

   反面教材是分开维护"分歧起始时刻"和"心跳时刻"：那样每加一条新的失败路径，都要记得去启动正确的那个时钟，漏一处就是一条永久放行的通道。实际漏过两次：**tick 的第一次 epoch 读取**若绕过 `shared_get`，Manager 持续超时会被最外层兜住并 `continue`，此时"分歧"从未被观察到（读都没读成功），而"心跳"却因为循环还活着而不断刷新 —— 两个时钟都不触发，worker 无限期沿用旧快照。

   正确的不变式是**"我在最近 `STALE_AFTER` 内正向确认过自己就在目标 revision 上"**：

   - `last_confirmed_at` **只在两个地方**更新：tick 确认 `local == epoch`，或 `apply_snapshot()` 换入成功。**任何**其它退出路径（epoch 读不出来、`content` 读不出来、digest 不匹配、构建期间 epoch 变了、tick 根本没跑、任务死了）都不更新它 —— 于是"忘记启动时钟"这个失败模式在结构上不存在：默认就是不确认，确认必须显式挣得。
   - **代价是 `swap()` 不许单独调用**：`swap + confirm` 必须绑成 `apply_snapshot()`（**纯本地，不含共享上报** —— 上报是独立的 `publish_report()`，只能在锁外调用，理由见 §9.3），四条成功换入路径（启动的三条分支、reload 发起者、tick）**全部**走 `apply_snapshot()`。漏掉任何一条的后果都是立竿见影的假 stale：启动后 `last_confirmed_at` 仍是初值 `None` ⇒ 首批请求全 503 直到第一次 tick；发起 reload 的那个 worker 自己 confirm 不了 ⇒ 刚成功换入就 503 到下一次 tick。`last_confirmed_at` 的初值**必须**是 `None` 且 `is_stale()` 视其为真 —— 这样"还没完成启动就来的请求"落在 fail closed 一侧，而不是靠 `monotonic()` 的巧合。
   - `deterministic_error`（`ValidationError` 的脱敏原因）令 `is_stale()` **立即**为真，绕过宽限期；`confirm()` 清除它。
   - 授权路径只调 `local.is_stale()`：两次进程内单调时钟比较 + 一次 `is not None`，**0 RPC**（§10.4 不变式不受影响）。
   - 心跳看门狗自然被这一个时钟涵盖：任务死了 ⇒ 没人再 `confirm()` ⇒ `STALE_AFTER` 后自动 `stale`。不再需要独立的 `last_tick_at`。

   仍然要保留的一层：**任务不许退出** —— tick 循环最外层兜住所有异常（记日志 + `continue`），只有 `CancelledError` 能终止它（shutdown 走这条）。它现在不再是 fail-closed 的前提（时钟已经兜住了），而是为了让故障可自愈：任务活着才有机会在 Manager 恢复后重新 `confirm()`。

   - **成因判定需要一个不带时间的枚举，而不是第二个时钟**。只有 `last_confirmed_at` + `deterministic_error` 的话，"共享读持续失败"与"轮询任务停摆"最终都表现为同一种状态（无确定性错误 + 时钟过期），本地日志与 503 响应体**无法区分**，§9.3 能力边界表里承诺的"本地可分"就落空了。补一个 `last_attempt_kind`（`CONFIRMED` / `ADOPTED` / `UNREACHABLE` / `CONTENDED` / `ERROR`），tick 的每条退出路径都写它。**分工是"时钟判是否 stale、枚举判为什么"**，单一 freshness 时钟的设计不受影响。

     关键推导：**上次尝试是成功的、时钟却过期了 ⇒ 之后根本没有尝试发生 ⇒ 轮询任务停摆**。因为一个还在跑的轮询器若读不到共享状态，会每 tick 把枚举刷成 `UNREACHABLE`；只有停摆的轮询器才会让枚举停在 `CONFIRMED`/`ADOPTED` 而时钟一路走到过期。这也是它不需要时间戳的原因。

   诊断上要分清三种 `stale` 成因：`error`（构建失败，去看策略文件）、`unreachable`（共享读持续失败，去看 Manager）、`stalled`（轮询任务停摆，去看任务本身）；`contended` 作为第四种（共享内容持续处在换代中）单列。四者的处置动作完全不同，混成一个 `stale` 会把运维引向错误的方向。但完整的成因区分是**本地能力**：共享**完全**不可达（读写皆失败）的 `unreachable` worker 与停摆的 `stalled` worker 都写不进共享报告，远端只能把它们标成 `unresponsive`（§9.3 末尾的能力边界表）；只有"读坏写好"的部分故障能以自报的 `pending` 呈现在远端。成因区分靠该 worker 自己的日志与它 503 响应体里的 `cause` 字段。
3. **`stale` 会自愈**：tick 继续跑，一旦成功构建（运维修好 env 并发布新 revision，或分歧的成因消失）就回到 `ok`。**重启不是解药**：重启后的 worker 会采纳同一份共享字节，遇到同样的 env 缺失 —— 所以正确的处置是修 env 后发布新 revision，或整体冷重启。

**状态码为什么是 503**：语义与反例论证见 §10.2（唯一权威）；效果是网关可以重试到别的 worker、前端**不得清 token**（§12）。

**`stale` worker 上不开授权例外**（包括 `/auth/policy/status`）：它自己就是无法信任授权判定的那个进程，给它开后门等于承认它还能授权。诊断走**任一健康 worker** 的 `/auth/policy/status` —— `adoptions` 是共享的，那里能看到这个 pid 的 `state="error"` 与脱敏后的原因。全舰队都 `stale` 时（例如所有 worker 的 env 都缺同一个变量）确实无人可查，但那时 `stale` 的 503 响应体本身会带上本进程的 `state` / `message`，且启动横幅与日志都有记录。

**可用性取舍要说清**：4 个 worker 里 1 个 `stale`，就是 25% 的请求 503。这是刻意的选择 —— 授权系统的正确性优先于可用性，且这个状态在 fork 模型下几乎不可达（同一份 env、同一份字节）。它真的发生时，是一次必须被看见的配置事故。

### 9.5 worker 重启与冷启动语义

**真相源的准确说法（唯一权威）**：磁盘上的策略文件是**运维的暂存区与冷启动权威**；运行时真相是**最后一次通过校验并被应用的 revision**（其字节存在共享内存里，§9.3）。"编辑文件"本身不产生任何效力 —— 这正是 validate-then-apply 的语义。两者的唯一分歧点在冷启动：进程组整体重启时磁盘内容成为新的第一个 revision。具体语义：

- `revision` 是共享 epoch 里的单调计数器，只由 reload 端点在锁内 `+1`。
- **进程启动优先采纳共享内容**：走 §9.3"冷启动初始化事务"那段协议 —— 锁内看到 epoch 已存在就用 `shared["content"]` 构建（与其它 worker 逐字节相同），只有 epoch 为空时才由**唯一 leader** 用磁盘内容发布 `revision=1`。
- 这一条把原先"worker 崩溃重启导致舰队漂移"的窗口**关掉了**：Manager 在 gunicorn arbiter 里预 fork 启动（[run_with_gunicorn.py:289](../../lightrag/api/run_with_gunicorn.py#L289)），跨单个 worker 的重启存活，因此重启的 worker 拿到的是"最后一次通过校验并应用的版本"，而不是可能被编辑到一半的磁盘文件。
- **冷启动**（进程组整体重启、Manager 一并重建）才回落到磁盘 —— 这正是想要的语义：那时磁盘内容成为新的第一个 revision（`revision=1`）。此时若磁盘文件校验失败，按规则 #14 启动失败。
- 因此 `revision` 的单调性**只在一个 Manager 生命周期内成立**，冷启动后重新从 1 计数。它是"变没变"的比较信号（收敛判定、`/auth/me` 的缓存失效提示），**不是跨重启的全局版本号**；任何持久化或跨部署比较都必须用 `digest`。
- 剩下唯一的漂移形态：冷启动后的磁盘内容与重启前最后一个 revision 不同（运维改了文件但没 reload 就重启了）。这不 fail closed（重启是正当行为），但 `/auth/policy/status` 必须能看出来 —— 靠的是 `revision_source`：冷启动后的舰队特征是 `revision=1, revision_source="disk"`，一眼可辨"当前生效的字节来自磁盘而非某次 reload"。`adoption_source` 回答的是另一个问题（**这个 worker** 怎么拿到快照的：leader 读盘 `"disk"`、follower 从共享字节 `"shared"`、reload 发起者本地定稿 `"reload_local"`），同一 revision 下不同 worker 的值本就不同，**不能**拿它做舰队级漂移判断。两个字段语义不重叠，都进采纳报告；运维规约仍然写明"改完文件立刻 validate + reload"。

### 9.6 自锁保护

- 校验规则 #11：新快照若无任何 principal 持有 `auth.policy.reload`，reload 返回 422（此时唯一出路是重启，属于停机）。`POST /auth/policy/reload?force=true` 可以显式接受这个后果。
- **"持有"必须叠加"可认证"**：只数 `enabled: true` 的 principal。否则一个典型的 break-glass 写法就能绕过整条自锁保护 —— 新策略把当前管理员的 `auth.policy.reload` 撤掉，只把 admin 角色绑给一个 `enabled: false` 的应急账号；规则表面通过、reload 被应用，而**没有任何人能再调用 reload**，唯一出路是重启。那正是 #11 存在的目的，也正是只有 `force=true` 才允许发生的后果。
  - `enabled` 是唯一需要额外判定的条件：`password_env` / `secret_env` 的可解析性已由规则 #7 / #9 在加载期保证（解析不到就整份拒绝），所以进了快照的凭据一定是可用的。
  - `anonymous` principal 不计入 —— 它没有凭据可言。匿名模式若真要持有 reload 权限，那是 `AUTH_ANONYMOUS_ROLE` 的显式配置，与自锁保护无关。
- 若新快照剥夺了**调用者自己**的 `auth.policy.reload`（但别人还有），仍然应用，响应带 `self_lockout: true` 提示。
- `/auth/policy/validate` 会预先报出上述两种情形。

### 9.7 被否决的替代方案

| 方案 | 否决理由 |
| --- | --- |
| 向 gunicorn arbiter 发 `SIGHUP` 滚动重启 worker | worker 内运行着摄取管线的后台任务与预约（reservation），滚动重启等于杀掉在途任务；且 arbiter 重启不是"不停机重载" |
| 每个 worker 按 mtime/digest 自动采纳 | 编辑文件即生效，绕过 validate-then-apply；半改完的文件会被采纳；违反"API 触发"语义 |
| **只共享 epoch，每个 worker 各自重读磁盘**（本方案早期形态） | 校验的字节与采纳的字节分离：reload 与某个 worker 的重读之间只要文件再被改一次，该 worker 就永远无法采纳该 revision，舰队 split-brain 且只能靠运维再 reload 一次；worker 崩溃重启会读到编辑到一半的文件。改为共享内容后这两个缺陷一起消失（§9.3 / §9.5） |
| 共享构建好的 `PolicySnapshot` 而非原始字节 | 快照含 `api_key_index`，共享它要么把 api_key 明文、要么把 pepper 送进 Manager 进程；字节里只有本就落盘的 bcrypt 摘要，风险等级严格更低 |
| 每请求读共享 epoch 判断是否过期 | 每请求一次 Manager proxy RPC，热路径不可接受 |
| 用 Manager `Event` / `Condition` 取代轮询 | 每 worker 常驻一个阻塞线程 + 一条长期占用的 Manager 连接，比每 2 秒一次小 RPC 更贵；Manager 无廉价推送原语 |
| 用 Redis/PG 做广播 | 引入对存储后端的新依赖；`shared_storage` 已有跨进程设施 |

---

## 10. FastAPI 强制机制

### 10.1 依赖形态

```python
async def authorize(
    request: Request,
    security_scopes: SecurityScopes,
    token: str | None = Security(oauth2_scheme),
    api_key: str | None = Security(api_key_header),
) -> Principal:
    """认证 + 授权的唯一咽喉。scopes 承载 LightRAG 权限码。"""
```

不需要 principal 的路由：

```python
@router.post(
    "/documents/upload",
    dependencies=[Security(authorize, scopes=[Permission.DOCUMENTS_WRITE])],
)
async def upload_document(...): ...
```

业务需要 principal 的路由：

```python
@router.get("/documents")
async def list_documents(
    principal: Annotated[Principal, Security(authorize, scopes=[Permission.DOCUMENTS_READ])],
): ...
```

规则：

- 多个 scope 语义是 **AND**（全部持有才通过），**第一阶段已经用到**：`/documents/scan` 声明 `[DOCUMENTS_RETRY, DOCUMENTS_WRITE]`（§5.2 说明了为什么）。所以 AND 不是"预留能力"，403 的 `detail` 必须能列出**所缺的那些**权限码（可能多于一个）。
- 需要 OR 时拆成两个路由或引入显式 `AnyOf` 辅助（第一阶段确实无 OR 需求）。
- 凭据解析（同时带 JWT 与 API Key 时谁说话）见 §8.5 —— 单一 principal，永不取权限并集。
- 不使用普通 Python 装饰器包裹 handler；未来可选的 `AuthorizedAPIRouter(permissions=[...])` 只允许翻译成同一个 `Security` 依赖 + OpenAPI 元数据，**不得包裹 handler 执行、不得成为第二套授权引擎**。
- OpenAPI 里为每个路由写入 `x-required-permissions`（非 OAuth 客户端也能看到要求）。

### 10.2 401 / 403 判定

**本表是全文四档状态码语义的唯一权威**：§8.1（500）、§9.4（503）、§12（前端分流）、§13（死钉矩阵）只引用本表，不再复述理由。

| 情形 | 状态码 | 说明 |
| --- | --- | --- |
| 无凭据 | 401 + `WWW-Authenticate: Bearer` | |
| 令牌无效 / 过期 / `ce` 不匹配 / principal 不存在或禁用 | 401 | 文案统一为"请重新登录"，不区分"用户不存在"与"已禁用"（防枚举） |
| API Key 无效 | 401 | **口径变更**：现状是 403（[utils_api.py:500-504](../../lightrag/api/utils_api.py#L500-L504)）；认证失败统一 401 更符合语义。验收以 §13 的精确矩阵为准：policy 模式死钉 401、legacy 模式死钉 403（放宽断言会让保留 403 的实现也通过测试） |
| 带了 `Authorization` 但令牌无效（即使 `X-API-Key` 有效） | 401 | 不回退到 API Key，见 §8.5 |
| 凭据有效但缺权限 | 403 + `detail` 含**全部**所缺权限码 | 权限码在 OpenAPI 里本就公开，回显有助排障，不构成信息泄露。多 scope 路由（`/documents/scan`）可能缺多个 |
| provider 内部错误 | 500（fail closed，不放行） | 走既有 `internal_server_error` 咽喉，不泄露内部细节。不用 401：前端见 401 会清 token 跳登录（§12），一次基础设施故障会表现成"全员被登出"，运维会去查会话而不是查故障；不用 403：会被当成权限不足 |
| 本 worker 处于 `stale` 状态（§9.4） | **503** + `Retry-After` | 不是 401（会让前端清 token 登出）、不是 500（会被当成 bug）、不是 403（会被当成权限不足）。响应体带本进程的 `state` / 脱敏原因 |

### 10.3 需要改造的 import 期单例（实现关键）

1. `auth.py:260` 的 `auth_handler = AuthHandler()` → 拆成 `TokenService`（无账号状态，只管签发/验签）+ 快照里的用户表。
2. `utils_api.py:242-257` 的模块级 `whitelist_patterns` / `auth_configured` → 移入快照 / 部署 profile 对象；policy 模式下白名单不再参与授权（只保留 §5.1 的代码内公共清单）。
3. `get_combined_auth_dependency(api_key)` 闭包按值捕获 `api_key` → 改为运行期从 `PolicyRuntime.current()` 解析。
4. `credentials_accepted()` 保留为"给 ASGI 层的布尔咽喉"，内部改成 provider 调用，**继续保持路由依赖与中间件同源**。

### 10.4 请求路径成本与 0-RPC 不变式

**授权判定在请求路径上是 0 次跨进程 RPC、0 次文件 I/O。** 这不是"优化目标"而是**不变式** —— 整套热重载设计（快照预计算 `effective`、轮询而非每请求查 epoch、§9.7 否决"每请求读共享 epoch"）都建立在它之上，一旦被破坏，退化不会表现为报错，而是全站每请求多一次 Manager proxy RPC。

| 步骤 | 成本 | RPC |
| --- | --- | --- |
| `PolicyRuntime.current()`（或 `request.state.policy_snapshot`） | 一次属性读，无锁 | 0 |
| JWT 验签 + `exp` / `ce` 校验 | HS256，约几十 µs（与今天相同） | 0 |
| 快照内查 `sub` → `UserRecord` | dict 取值 | 0 |
| `effective[(pt, pid)]` → `permission in frozenset` | O(1)，无集合运算、无分配 | 0 |
| API Key：`HMAC-SHA256(pepper, secret)` → 查 `api_key_index` | ≤512B 输入约 1–2 µs；超长 header 直接 401，不做 HMAC | 0 |

跨进程共享状态（`auth_policy` 命名空间）只有两类访问者：**每 worker 的轮询任务**（2 秒一次小 RPC）与 **`/auth/policy/*` 三个管理端点**。请求路径不在其中。

相比现状还略省一点：policy 模式下 `whitelist_patterns` 的正则列表遍历（[utils_api.py:242-257](../../lightrag/api/utils_api.py#L242-L257)）被 §5.1 的常量 set 查表取代。

唯一昂贵的是 **bcrypt（cost 12，约 100–250 ms）**，只发生在 `POST /login`，前面压着登录限流器 —— 而限流器是**纯进程内** `OrderedDict`（[login_rate_limit.py](../../lightrag/api/login_rate_limit.py)，其模块头已注明"N workers 下有效上限是 N × max_attempts"），同样 0 RPC。

#### 四条禁令（实现期红线）

1. **请求路径绝不能调 `get_namespace_data("auth_policy")`**。它是 `async def` + Manager dict，每次读都是一次 proxy RPC，首次调用还要建 namespace。最典型的破坏方式是"顺手从共享状态读一下当前 revision 填进响应" —— revision 必须取自 `request.state.policy_snapshot.revision`。
2. **`has_permission` 虽声明为 `async`，文件实现里不能有真正的 await**：不得 `asyncio.to_thread`、不得取锁、不得读共享状态。它必须是"查表后立即 return"的协程。
3. **`authorize` 必须是 `async def`**。写成同步 `def` 会被 FastAPI 扔进 threadpool，每请求多一次线程跳转。
4. **WebUI 不得轮询 `/auth/policy/status`**（§12）。那是唯一遍历 `adoptions` 的端点（每存活 worker 一条记录 + `_pid_alive` 探测）。前端权限门控走 `/auth/me`，它是 0 RPC 的快照读。

内存侧（不是性能问题，但需知晓）：`swap()` 之后旧快照会被在途请求继续引用直到它们结束，因此换代瞬间进程内最多同时存在两份快照。上限由 `AUTH_POLICY_MAX_BYTES` 卡住，可忽略。

### 10.5 admission 中间件与 body 前置检查

- 中间件继续**只做认证**（RFC 红线：中间件不得实现第二套 path→permission 表），但认证必须走同一个 `CredentialProvider` 快照，不能再按值持有 `api_key`。
- 已核实：路由依赖抛 403 时 handler 不会 adopt ticket，`admission_middleware.py:152-163` 的 `finally` 会归还容量槽。因此"有 `documents.read` 但无 `documents.write` 的调用者"最坏结果是：占用一个槽 + 读一次 body，然后 403 并归还。可接受，且回归测试要**钉住这条归还路径**。
- `/documents/upload` 等路径在 policy 模式下不再受 `WHITELIST_PATHS` 豁免。

---

## 11. 部署 profile 与迁移

### 11.1 配置项汇总

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `AUTH_POLICY_FILE` | 空 | 设置即进入 policy 模式 |
| `AUTH_PROFILE` | 自动推断 | `policy` / `legacy` / `unauthenticated`；显式设置优先，与实际配置矛盾则启动失败 |
| `AUTH_POLICY_MAX_BYTES` | `1048576` | 文件大小上限 |
| `AUTH_POLICY_RELOAD_POLL_SECONDS` | `2.0` | worker 轮询 epoch 间隔（≥0.5） |
| `AUTH_POLICY_HEARTBEAT_SECONDS` | `10.0` | 心跳写共享报告的最大间隔（≥ 5 × 轮询间隔）。调度语义、注册前置与 `unresponsive` 判据见 §9.3 |
| `AUTH_POLICY_STALE_GRACE_SECONDS` | `10.0` | stale 宽限期。实际生效值与语义见 §9.4（`STALE_AFTER` 公式的唯一权威） |
| `AUTH_EXPOSE_API_DOCS` | `true` | 是否注册 `/docs`、`/docs/oauth2-redirect`、`/redoc`、`/openapi.json`（§5.1）。`false` 时四个端点均不注册 |
| `AUTH_ALLOW_UNAUTHENTICATED` | `false` | 匿名开发模式显式开关 |
| `AUTH_ANONYMOUS_ROLE` | `reader` | 匿名模式下匿名 principal 的角色；**仅在有策略文件时生效**（角色定义在文件里）。无策略文件的匿名模式用内建 `legacy_user`（§8.4） |
| `AUTH_LEGACY_ACCOUNTS_COMPAT` | `false` | 允许 `AUTH_ACCOUNTS` 与策略文件共存并映射为 `legacy_user` |
| `AUTH_ROUTE_AUDIT` | `enforce`（policy 模式）/ `report` | 路由覆盖审计的执行强度 |
| `OLLAMA_ALLOW_UNAUTHENTICATED` | `false` | 高风险兼容开关，见下 |
| `WHITELIST_PATHS` | `/health,/api/*`（**默认值不变**） | **policy 模式**下忽略该项并在启动时告警"已被公共路由清单取代"；legacy / 开放模式沿用旧默认，行为不变 |

### 11.2 破坏性变更清单

**全部破坏性变更都只发生在 policy 模式（即显式设置了 `AUTH_POLICY_FILE`）。** 不设策略文件的部署 —— 包括配了 `AUTH_ACCOUNTS`/`LIGHTRAG_API_KEY` 的，以及什么都没配的裸部署（§8.4）—— 行为逐字不变。两处不构成行为变更的例外：`documents.artifacts.*` 这类尚不存在的新能力不会被隐式授予（§11.4）；裸部署与显式匿名模式新增一条启动横幅告警（只加日志，不改判定）。

1. ~~`WHITELIST_PATHS` 默认值收窄~~ —— **已改为不动全局默认值**。今天的默认 `/health,/api/*` 把整个 Ollama 兼容面免认证，与 policy 模型不兼容，但把默认值全局改掉会让 legacy 部署的 `/api/*` 突然要求认证，与 §14 PR2 的"legacy 行为逐字不变"验收标准直接冲突。收窄改为**只在 policy 模式生效**：该模式下 `WHITELIST_PATHS` 整项被忽略，公共面由 §5.1 的代码内清单唯一决定（`/api/*` 不在其中）。legacy 模式保留旧默认 + 启动告警。
2. policy 模式下 guest 令牌不再签发；未带凭据的 WebUI 会跳登录页。
3. policy 模式下 API Key 认证失败由 403 改 401。
4. policy 模式下缺 `ce`/`pt` 的旧 JWT 失效（需重登一次）。
5. policy 模式下明文口令不再受支持（必须 bcrypt）。
6. policy 模式下 `TOKEN_SECRET` 必须显式设置为非默认值，否则启动失败（§7.3 #15）。
7. 若保留匿名 Ollama 兼容（`OLLAMA_ALLOW_UNAUTHENTICATED=true`）：启动横幅必须打印高风险警告、文档必须记录、并写明移除计划；它不是"正常授权模式"的一部分。

### 11.3 迁移顺序（建议给运维的操作序列）

1. 升级到含 authz 包但仍跑 legacy profile 的版本（行为不变，只多打印一条"未启用 policy 模式"的提示）。
2. 用 `lightrag-hash-password --format policy`（既有命令的策略文件模式，§14 PR1）把现有 `AUTH_ACCOUNTS` 转成策略文件条目；`AUTH_ROUTE_AUDIT=report` 先跑一轮看审计输出。
3. 设置 `AUTH_POLICY_FILE`、设置非默认 `TOKEN_SECRET`、删除 `AUTH_ACCOUNTS`，重启一次进入 policy 模式（`WHITELIST_PATHS` 无需改动，policy 模式下整项被忽略）。
4. 之后所有用户/角色变更走"编辑文件 → `POST /auth/policy/validate` → `POST /auth/policy/reload` → `GET /auth/policy/status` 确认 `converged=true`"。

不迁移也是受支持的选择：停在第 1 步即可长期运行，代价是用不上 artifact 下载能力、且用户/权限仍只能靠重启改（§11.4）。

### 11.4 legacy 预设角色（行为保持的唯一定义）

legacy / 开放模式下没有策略文件，权限来自代码里的**内建 `legacy_user` 角色**。它的定义必须精确对应今天的行为：[utils_api.py:375-519](../../lightrag/api/utils_api.py#L375-L519) 的 `combined_dependency` 是**纯二元判定** —— 认证通过即通过，40 条受保护路由的能力完全相同、没有任何按路由的区分。

因此 `legacy_user` = **当前目录全集 − `documents.artifacts.*` − `auth.policy.*` − 无路由对应的权限码**，在引入本版本时**冻结为显式枚举列表**：

```python
LEGACY_USER_PERMISSIONS: frozenset[Permission] = frozenset({
    Permission.AUTH_SESSION_READ,
    Permission.SYSTEM_HEALTH_READ,
    Permission.QUERY_EXECUTE,
    Permission.OLLAMA_INFERENCE,
    Permission.OLLAMA_METADATA_READ,
    Permission.DOCUMENTS_READ,
    Permission.DOCUMENTS_WRITE,
    Permission.DOCUMENTS_RETRY,
    Permission.DOCUMENTS_DELETE,
    Permission.DOCUMENTS_CLEAR,
    Permission.DOCUMENTS_SOURCE_CONFLICTS_READ,
    Permission.DOCUMENTS_SOURCE_CONFLICTS_REPAIR,
    Permission.PIPELINE_READ,
    Permission.PIPELINE_CONTROL,
    Permission.CACHE_CLEAR,
    Permission.GRAPH_READ,
    Permission.GRAPH_WRITE,
    Permission.GRAPH_DELETE,
})   # 18 项，golden 测试钉住；新增权限码不自动进入
```

三条设计约束，缺一条就出错：

1. **不能用 §7.2 示例里的 `operator` 角色**。`operator` 只有 `auth.session.read / query.execute / documents.read / documents.write / documents.retry / pipeline.read` 六项；拿它当 legacy 预设，存量部署会**静默丢掉** `documents.delete`、`documents.clear`、`graph.write`、`graph.delete`、`cache.clear`、`pipeline.control`、`ollama.inference`、`documents.source_conflicts.*`、`system.health.read` —— WebUI 的删文档、清库、图编辑按钮当场失效。`operator` 是给策略文件用户的推荐起点，不是兼容层。
2. **不能写成"目录全集减排除清单"的动态减法**。那样将来目录里每加一个权限码，legacy 用户就自动多一项能力，恰好违背"新能力对旧配置 deny-by-default"这条本节赖以成立的逻辑。**权限集冻结、路由集不冻结**：既有权限码下新增的路由（例如再加一个 `documents.read` 类的查询端点）legacy 用户照样可用，正常演进不受影响；要给 legacy 增加**新权限码**必须显式改这份列表，属于刻意的"改动即评审"。
3. `system.config.read` / `system.config.write` 不在列表里，因为 §5.2 里今天没有任何路由使用它们 —— 列表只包含**已挂在路由上**的权限码。将来给这两个码挂上路由时，是否让 legacy 用户拿到必须显式决策（走约束 2 的评审路径）。

`AUTH_LEGACY_ACCOUNTS_COMPAT=true`（policy 模式下允许 `AUTH_ACCOUNTS` 共存）时，那些账号同样映射到本角色。

---

## 12. WebUI 影响（最小改造）

- 新增 `GET /auth/me` → `{principal_type, principal_id, display_name, permissions: [...], policy_revision}`。前端据此隐藏无权操作（上传、删除、清库、图编辑等）。**三种 profile 下都必须可用**：legacy / 开放模式返回 `legacy_user` 预设的权限集（`policy_revision: null`），这样前端只有一套门控逻辑，且 legacy 下门控结果恰好是"全部可见（除 artifact 下载）"，与今天的界面一致。
- `/auth-status` 增加 `auth_backend`（`policy`/`legacy`/`unauthenticated`）字段；policy 模式不再返回 `access_token`，前端需要处理"没有 guest 令牌"分支（当前逻辑在 `lightrag_webui/src/api/lightrag.ts` 与 `stores/state.ts`，登录流程本身不变）。
- 401 / 403 / 500 / 503 四路分流，**只有 401 清 token**：
  - `401` ⇒ 清 token、跳登录页。
  - `403` ⇒ 只提示"当前账号无此权限"，**不得**清 token（否则用户点一次无权按钮就被登出）。
  - `500` ⇒ 提示服务端错误，不清 token（provider 内部错误会走这里，§10.2）。
  - `503` ⇒ 提示"服务端授权策略未收敛，请稍后重试"，**不清 token**、可按 `Retry-After` 自动重试一次。这是 §9.4 的 stale worker；把它当会话失效清 token 会让一次配置事故变成全员登出。
- 权限列表随 `/auth/me` 拉取，热重载后前端最迟在下次刷新/轮询时看到新权限；不做 WebSocket 推送。
- **前端只允许轮询 `/auth/me`（0 RPC 的快照读），不得轮询 `/auth/policy/status`** —— 后者是运维/CI 的收敛断言工具，每次调用都要遍历 `adoptions` 并做 `_pid_alive` 探测（§10.4 禁令 4）。策略变更是低频运维动作，前端不需要秒级感知。

---

## 13. 测试计划

放置位置遵循仓库约定（`tests/api/auth/` 为主，跨层的放 `tests/api/`）。

**加载与校验**（`tests/api/auth/test_policy_loader.py`）
- 表驱动覆盖 §7.3 全部 16 条规则，每条一个"拒绝"用例 + 一个"接受"用例。
- 规则 #15：policy 模式 + `TOKEN_SECRET` 未设置 / 等于 `DEFAULT_TOKEN_SECRET` ⇒ 启动失败。**fix-proof 要求行为级**：断言"用默认密钥自签的 `sub=admin` 令牌无法通行"，而不只是断言启动抛了异常 —— 后者在实现把校验搬走后仍会因别的原因通过。
- `"*"` 展开等于 catalog 全集；catalog 新增权限时 admin 自动获得（断言展开逻辑而非硬编码列表）。
- `effective` 预计算：多角色并集正确；悬空绑定被拒。

**legacy 预设角色**（`test_legacy_preset.py`）
- golden 断言 `LEGACY_USER_PERMISSIONS` 恰好等于 §11.4 的 18 项枚举（**不是**"catalog 减排除清单"的动态计算 —— 断言必须写成字面列表，否则往 catalog 加权限码时测试跟着一起漂）。
- 往 catalog 里加一个假权限码 ⇒ `legacy_user` **不**获得它（约束 2 的 fix-proof）。
- 行为保持对照：legacy profile 下，认证用户对 §5.2 全部 40 条路由的判定与改造前逐字一致（用改造前的 `combined_dependency` 判定作为对照表）。这条要能抓住"误用 `operator` 做预设"—— 那会让 delete/clear/graph.write 等变 403。
- `documents.artifacts.*` 在 legacy profile 与无策略文件的开放模式下**都拿不到**（403），无论 env 怎么配。
- `WHITELIST_PATHS` 未设置时，legacy profile 的默认值仍是 `/health,/api/*`（§11.2 #1 的回归钉子）。
- legacy profile 下 `/auth/policy/*` 返回 404（未挂载），不是 403。

**快照与请求内一致性**（`test_policy_runtime.py`）
- `swap()` 期间正在执行的请求全程使用旧快照（用两个依赖 + 中途 swap 构造）。
- provider 抛异常 ⇒ **500**，不放行（fail closed）。死钉 500，口径以 §10.2 为唯一权威。

**请求路径 0-RPC 不变式**（`test_authz_request_cost.py`）
- 给共享命名空间的 proxy 装 spy，跑一次带完整认证 + 授权的请求，断言 `auth_policy` 命名空间的访问次数**为 0**；对 §5.2 里几条代表性路由各跑一遍（JWT 路径、API Key 路径、403 路径）。这条同时覆盖 §10.4 的禁令 1 与禁令 2 —— 它们是"退化不报错、只是变慢"的那类缺陷，只能靠断言访问次数抓住。
- 断言 `authorize` 是协程函数（`inspect.iscoroutinefunction`），钉住禁令 3：改成同步 `def` 后功能测试全绿，只有这条会红。

**会话失效**（`test_credential_epoch.py`）
- 改密 / 禁用 / 删除用户 ⇒ 旧令牌 401；仅改角色 ⇒ 旧令牌仍有效但权限即时变化。
- **确定性**：同一份文件在两个"进程"（两个 loader 实例）算出的 `ce` 相同（防止重启踢人）。
- **可逆性（§8.2）—— 两组用例，断言方向相反，不能混写**：
  - **运维边界（记录事实，不是 fix-proof）**：`credential_version` 保持 `1` 时，① 禁用 alice → 重新启用、② 删除 alice → 用同一 `password_bcrypt` 重建同名用户，两种情形下旧令牌（未过期）**都会复活**。这是算法的直接推论（`ce` 是 `password_hash + enabled + credential_version` 的纯函数），测试要把它钉成已知边界，防止实现者误以为服务端能自动兜住。
  - **fix-proof**：同样两种情形，恢复时把 `credential_version` 从 `1` 改成 `2` ⇒ 旧令牌**必须持续 401**。这才是修复证明。

**凭据优先级**（`test_credential_precedence.py`，§8.5）
- 低权用户 JWT + 高权 API Key 同时发送 ⇒ 判定与审计 principal 都是**该用户**，高权操作 403（**严禁并集**：这条是越权 fix-proof）。
- 无效 / 过期 JWT + 有效 API Key ⇒ **401**，不回退 API Key。
- 只有 API Key ⇒ principal 是服务账号；`/auth/me` 返回同一身份。
- `credentials_accepted()`（ASGI 层）与 `authorize`（路由层）对同一组 header 解析出**同一个** principal。

**API Key**（`test_api_key_index.py`）
- HMAC 查表命中/未命中；重复密钥被拒；超长 header 直接 401 且不做 HMAC（用 spy 断言）；密钥明文不出现在日志（`caplog` 需临时开 `lightrag` logger 的 `propagate`）。

**热重载**——按 PR7a / PR7b 分两个文件放，与 §14 的拆分对齐：传播机制与状态机（不经 HTTP，纯 `PolicyRuntime` + 共享 dict + 手动 tick）放 `test_policy_propagation.py`；端点、status 汇总、自锁保护放 `test_policy_reload.py`。每条只写注入与断言，设计依据见括号内章节，不再复述：

- 单进程基线：validate 无副作用；reload 成功换入；坏文件 reload ⇒ 422 且旧快照不变（断言旧用户仍能登录）。
- 多进程模拟基线：两个 `PolicyRuntime` + 共享 dict + 手动 tick，覆盖 `ok` / `error` / `pending` 与 `converged`（`drift` 已从传播路径移除，不写用例）。
- **follower 不读磁盘**（§9.3）：reload 成功后把磁盘文件改坏 ⇒ follower tick 仍采纳 epoch 字节、`converged=true`。
- **冷启动初始化事务**（§9.3）：epoch 为空、两 worker 并发启动 ⇒ 恰一个发布 `revision=1`，另一个采纳共享字节；随后 reload（`next_rev=2`）两者都采纳。fix-proof：两 worker 本地 revision=1 但 digest 不同时，发布 revision=1 的新 epoch 必须被采纳（元组比较；`epoch.revision <= local.revision` 跳过是旧缺陷）。
- **替换 worker 不读磁盘**（§9.3）：epoch=rev7 + 磁盘为半写坏内容 ⇒ 新 worker 采纳 rev7 进 `ok`，对读盘入口打 spy 断言调用 0 次。
- **构建不在锁内**（§9.3）：锁 spy 断言启动路径（**含冷启动 leader**）持锁期间无 `build()`，锁内只允许 `replace()`。
- **冷启动 leader 冒烟**：单 worker、epoch 为空 ⇒ 启动成功，上报 `revision_source="disk", adoption_source="disk"`（依赖未初始化跨分支标志位的实现在此抛 `UnboundLocalError`）。
- **构建后 epoch 变更**（§9.3）：`build()` 中途插入一次 reload ⇒ 不得带过期快照进 `ok`，必须重跑协议落在新 revision。
- **撕裂读取**（§9.3 digest 守卫）：读到 epoch=rev7 后、读 content 前把 content 换成 rev8 字节而暂不写 epoch ⇒ 必须 RETRY，不得换入；断言落在"上报的 (revision, digest) 与实际生效的权限集一致"（光看 `converged` 抓不到）。
- **单一咽喉**：对 `adopt_from_shared` 打 spy，启动与 tick 两条路径各命中一次。
- **重试有界**：digest 永不匹配 ⇒ 启动在 `MAX_STARTUP_ATTEMPTS` 后失败退出，不无限重试。
- **stale：确定性失败立即 503**（§9.4）：rev6 允许 clear、rev7 撤销且该 worker 构建 rev7 抛 `ValidationError` ⇒ ① 该 tick 后立刻 503，且直接断言 rev6 凭据调 `DELETE /documents` 拿不到 2xx（只断言 `converged=false` 不够）；② 宽限期内同样 503（确定性失败不得进 pending）；③ `/health` 最小载荷仍 200；④ 发布可构建的 revision 后自愈回 `ok`。
- **pending：未定失败享受宽限期，到点仍进 stale**（§9.4）：content 读失败 / 共享读超时 ⇒ `STALE_AFTER` 内按旧快照服务（正常 reload 的传播延迟不闪 503）；到点 503（判据是"多久没正向确认"，不是失败类型）。
- **`shared_get` 三态**（§9.3）：必须用真实空共享 dict 跑完整 `startup()`（锁内外表示不一致只在完整路径暴露）：① epoch 键不存在 ⇒ 冷启动成功发布 rev1（把 ABSENT 折成 RETRY 的实现 = 全新部署起不来）；② epoch 读抛 `TimeoutError` ⇒ RETRY，不得当冷启动发布 rev1（断言共享 epoch 未被改写）；③ 对 `.get()` 打 spy 断言零调用。
- **上报与换入分离、绝不在锁内上报**（§9.3）：① 锁 spy 断言每次 `publish_report()` 调用时本协程未持锁（塞进 `apply_snapshot` 的实现在 leader / reload 路径抛 NamespaceLock 重入 `RuntimeError`）；② `publish_report` 底层写失败 ⇒ reload 端点仍成功、本地已按新 revision 授权（**只断言 reload 路径** —— 启动期口径相反，由④⑤覆盖，不得混写）；③ 两 worker 锁外并发上报 ⇒ 两条记录都在；④ **注册是启动硬前置**：首次上报持续失败 ⇒ `register_or_die()` 有界重试后抛出、worker 起不来（`/health` 亦不可达，而非"起来了但 503"）；⑤ 首次失败但上限内成功 ⇒ 正常启动，pid 进 `adoptions`。
- **记录 schema 精确匹配**（§9.3）：三种 `state` 与心跳路径写的记录逐字段相同、全部顶层字段、无嵌套 `target`；`error` 路径的 revision/digest 是**仍在服务**的旧版本（据此 `converged=false`）；`error` 后紧跟的心跳不得丢版本字段（只传 state/message 的增量实现会让该 worker 悄悄退出收敛比较）。
- **换入单咽喉、首 tick 前可授权**（§9.4 第 4 点）：① 启动完成后立刻（不推进任何 tick）请求 ⇒ 正常授权不 503；② reload 返回后立刻请求 ⇒ 已按新权限判定；③ 对 `PolicyRuntime.swap` 打 spy，断言只被 `apply_snapshot()` 调用（四条成功路径全覆盖）；④ `last_confirmed_at=None` 时 `is_stale()` 为真。
- **tick 首读走咽喉**（§9.4 第 4 点）：把持续异常注入**第一次 epoch 读取** ⇒ ① 轮询 task 未 done；② `STALE_AFTER` 后 503。与"content 读失败"用例不得合并。
- **单时钟涵盖任务停摆**：直接停掉轮询任务 ⇒ `STALE_AFTER` 后 503；断言判定只依赖 `last_confirmed_at` / `deterministic_error`（上报调度状态属观测侧，绝不参与 `is_stale()`）。
- **RETRY 带原因**：content 读失败 ⇒ `cause=unreachable`；撕裂 / 构建期 epoch 变更 ⇒ `cause=contended`。
- **`publish_report` 三态**：adoptions 读 RETRY ⇒ 放弃本次写、其它 worker 记录不被清空（`shared_get(...) or {}` 的实现会抹掉全部报告）；ABSENT ⇒ 初始化空字典写入。
- **stale 成因本地可分**（§9.4 第 4 点）：① 读写皆失败的整体不可达（轮询器仍在跑）⇒ `cause=unreachable`；② 停掉任务 ⇒ `cause=stalled`（两条必须成对存在）；③ 构建失败 ⇒ `cause=error`。断言落在本地 `cause` 字段与日志，不断言 /status 能远端区分①②（§9.3 能力边界表）；①要读写皆失败，与"只坏读"的假收敛用例注入不要混。
- **`reported_at` 心跳与 `unresponsive`**：停掉某 worker 轮询 ⇒ ① 它自己 `STALE_AFTER` 后 503；② 从健康 worker 读 `/status`，该 pid 在 `2 × AUTH_POLICY_HEARTBEAT_SECONDS` 后落入 `unresponsive_workers` 且 `converged=false`。
- **上报调度五断言**（§9.3 `maybe_publish`）：① 稳态心跳真的在发（`reported_at` 按心跳周期刷新；稳态分支绕过调度点的实现在此失败）；② 稳态写入次数 ≈ 运行时长 / 心跳间隔，而非 / 轮询间隔；③ 持续 `error` 同样不每 tick 写（转入时立即写一次，其后按心跳频率）；④ 变化立即写，不等心跳到期；⑤ 写失败不推进调度状态（下一 tick 即重试，成功后才前进）。
- **未定失败不得自称 `ok`**（§9.3 假收敛）：epoch 读持续失败、adoptions 写正常（按调用注入，非整体断连），worker 与 epoch 同在 rev7 ⇒ ① 本地 `STALE_AFTER` 后 503；② 共享记录新鲜、版本 rev7、`state="pending"`；③ 健康 worker `/status` ⇒ `converged=false`、pid 在 `pending_workers`；④ 恢复后下一 tick `state` 立即回 `ok`（不等心跳）。
- **心跳与 reload 交错不乱序覆盖**（§9.3 `publish_mutex`）：心跳在共享锁获取处挂起（注入慢锁）、期间同 worker 的 reload 完成换入并上报 rev8 ⇒ 全部在途 `maybe_publish` 结束后，共享记录与 `last_published` 都是 rev8。
- **发布原子性**：`epoch` 与 `content` 两次写之间插入 follower tick ⇒ 不出现"新 revision 旧字节"。
- **轮询成本**：稳态 tick 只读 `epoch` 一个 key（key 级 spy，防把字节塞回 header），revision 变化的那一 tick 才读 `content` 一次。
- **三条路径的 source 字段**（§9.5）：follower 采纳 ⇒ `adoption_source="shared"`（`revision_source` 抄 epoch）；冷启动 leader ⇒ 两字段皆 `"disk"`；reload 发起者 ⇒ `("reload", "reload_local")`。三条都断言。
- **env 分叉**：某 worker 缺 `password_env` 变量 ⇒ 该 worker `error`、保留旧快照、`converged=false`，其它 worker 不受影响。
- **不共享快照**：共享命名空间只有字节 / header / adoptions 三类值，不含 `api_key_index`、pepper、api_key 明文。
- **自锁保护**（§9.6）：无人持有 `auth.policy.reload` ⇒ 422；`force=true` ⇒ 应用且响应标注；enabled fix-proof：reload 权限只绑给 `enabled: false` 的 break-glass 用户 ⇒ 非 force 必须 422（只数"持有"不数"可认证"的实现会放行）；`anonymous` 不计入。
- **死 pid 剔除**：`_pid_alive` 打桩 ⇒ 陈旧记录被清理，`adoptions` 保持有界。

**策略文件路径安全**（`test_policy_path_safety.py`，§7.3 #13）
- 父目录 group 可写 ⇒ 拒绝（`O_NOFOLLOW` 类方案漏掉的那条，§7.3 #13）；文件 group/other 可写 ⇒ 拒绝；非 regular file（FIFO / 目录）⇒ 拒绝；owner 非 root 且非服务 euid ⇒ 拒绝。
- **K8s 卷形态必须放行**：构造 `dir/..data/policy.yaml` + `dir/policy.yaml -> ..data/policy.yaml` 的符号链接农场（root 所有、`0755` 目录、`0644` 文件）⇒ **加载成功**，且启动日志里记录了解析后的真实路径。这条钉住"不能退回拒绝符号链接"，否则策略文件无法用 ConfigMap/Secret 卷挂载。
- 校验通过后从**同一 fd** 读取：在校验与读取之间替换文件内容 ⇒ 读到的仍是被校验的那份（TOCTOU）。

**路由审计**（`test_route_inventory.py`）
- golden 清单对比，且 golden **由实际 `app.routes` 生成**（§6）。
- `/redoc`、`/openapi.json`、`/docs`、`/docs/oauth2-redirect` 在 `enforce` 下不导致启动失败；`AUTH_EXPOSE_API_DOCS=false` 时四者全部 404。
- swagger 静态 mount 按实际路径 `/static/swagger-ui` 归类为 public（用 `name=` 匹配的实现会漏判，这是行为级差异）。
- `/documents/scan` 的 scopes 恰好是 `{documents.retry, documents.write}`；只持 `documents.retry` 的 principal 调它 ⇒ 403，且 `detail` 里列出缺失的 `documents.write`（§5.2 的越权 fix-proof）。
- `/api/ps` 的 scope 是 `ollama.metadata.read`；只持 `pipeline.read` 的 principal 调它 ⇒ 403。
- **fix-proof**：往测试 app 里加一个裸路由（无 `authorize`）⇒ `enforce` 下启动失败；这条必须是行为级失败，而不是靠符号缺失报 AttributeError。

**中间件协同**（`tests/api/test_admission_authz.py`）
- 有 `documents.read` 无 `documents.write` 的 principal POST `/documents/upload` ⇒ 403，且容量槽被归还（断言 `pending_enqueues` 回到 0）。
- 中间件与路由依赖对同一凭据判定一致（同源 provider）。

**CI 约束**（依据仓库 CI 现状）
- CI 只跑 `-m offline` 且**没有 `.env`**：所有 auth 测试必须自带 `monkeypatch.setenv`，不依赖 `.env`；关闭策略模式用 `setenv(..., "")`/`"false"` 而不是 `delenv`（`load_dotenv(override=False)` 会重灌）。
- **四档状态码全部死钉，按 profile 分别断言**（禁止 `status_code in (401, 403)` 式放宽 —— 它同时放过"policy 模式没改成 401"与"legacy 模式被顺手改掉"两类缺陷，§11.2 #3 的破坏性变更就失去行为级保障；尤其"缺权限被当成未认证"会让前端清 token 登出，§12）：

  | 情形 | policy 模式 | legacy 模式 |
  | --- | --- | --- |
  | 无效 API Key | 死钉 **401** | 死钉 **403**（现状，[utils_api.py:500-504](../../lightrag/api/utils_api.py#L500-L504)） |
  | 无凭据 | 死钉 401 | 现状语义 |
  | 凭据有效但缺权限 | 死钉 **403** | 死钉 403 |
  | provider 内部错误 | 死钉 **500** | — |
  | 本 worker `stale` | 死钉 **503** | —（legacy 无热重载） |

---

## 14. 分阶段实施（PR 拆分）

| PR | 内容 | 风险 / 验收 |
| --- | --- | --- |
| PR1 | `lightrag/api/authz/` 骨架：`catalog.py`（含 `ollama.metadata.read`）、`models.py`、`policy_file.py`（loader + strict schema + §7.3 全 16 条校验 + #13 的信任链路径解析）、`LEGACY_USER_PERMISSIONS` 冻结常量、`hashpw.py`（**不是新命令**：console script 沿用已注册的 `lightrag-hash-password`，entry point 从 `lightrag.tools.hash_password` 改指本模块；`AUTH_ACCOUNTS` 模式行为逐字保持——含 `{bcrypt}` 前缀输出，新增 `--format policy` 模式输出裸 bcrypt spec + `credential_version` 提醒 + `AUTH_ACCOUNTS` 批量转换）。**不接线**，纯新增 | 无行为变更；验收 = §13 加载与校验 + legacy 预设 golden 全绿 + `lightrag-hash-password` 旧模式（无 `--format`）输出格式回归（`username:{bcrypt}$2b$...` 逐字不变，钉住 entry point 迁移） |
| PR2 | `PolicyRuntime` + 两个 provider + `authorize` 依赖（含 §8.5 凭据优先级）+ `/auth/me`；拆解 §10.3 的四个 import 期单例；legacy profile 走 `legacy_user` 预设，行为**逐字保持不变**（含 `WHITELIST_PATHS` 旧默认） | 最容易回归的一步。验收 = 现有 `tests/api/` 全绿 + §13 的 40 条路由 legacy 行为对照测试 |
| PR3 | 迁移 `query_routes.py`(3) + `ollama_api.py`(5) 到 `Security(authorize, scopes=[...])`；`/api/tags`、`/api/version`、`/api/ps` 用 `ollama.metadata.read` | 小面积试点，先验证形态 |
| PR4 | 迁移 `document_routes.py`(19) | 最大 diff；逐条对照 §5.2 表，含 `/documents/scan` 的双 scope |
| PR5 | 迁移 `graph_routes.py`(12) + `lightrag_server.py` 的 `/health`、公共路由分离（public router，含 `API_DOCS_ROUTES`） | `/health` 双身份需专门测试 |
| PR6 | 路由覆盖审计 + CI golden 清单（按 profile 各一份）；`AUTH_ROUTE_AUDIT` 先 `report` 一轮再切 `enforce` | 验收 = 裸路由 fix-proof 测试 |
| **PR7a** | 传播机制骨架，**不接任何端点**：共享 epoch 与内容字节（分 key、同临界区发布）+ 附录 A 的全部原语（`shared_get` 三态 / `adopt_from_shared` / `apply_snapshot` / `publish_report` / `maybe_publish`）+ 冷启动初始化事务 + 轮询任务与 `register_or_die` 启动前置 + `stale`/`pending` 状态机与 503 闸门（§9.3–§9.4）。发布动作由内部函数触发（测试直接调），不经 HTTP | 验收 = §13“热重载”节里除 status 汇总与端点外的**全部**用例（用例清单以 §13 为唯一权威，此处不重复点名） |
| **PR7b** | 接线与可观测：三个端点（仅 policy 模式挂载）+ `/auth/policy/status` 汇总与收敛判定（含 `reported_at` 心跳与 `unresponsive_workers`）+ 死 pid 清理 + `stale` 的 503 响应体 + WebUI 的 401/403/500/503 四路分流 | 验收 = 收敛断言（`converged` / `pending_workers` / `failed_workers` / `unresponsive_workers`，含停摆 worker 不得显示收敛）+ 自锁保护三条（含 disabled 持有者 422）+ 轮询 key 级成本断言 + 前端不因 503 清 token |
| PR8 | profile 判定与 `WHITELIST_PATHS` 的 **policy 模式忽略**（不改全局默认值）、`TOKEN_SECRET` 启动期校验、`AUTH_EXPOSE_API_DOCS`、Ollama 兼容开关、启动横幅、`env.example`、文档（`docs/LightRAG-API-Server*.md` 两语言）、WebUI 权限门控 | 破坏性变更集中在此且全部限定 policy 模式，PR 描述须列全 §11.2 |

每个 PR 独立可回滚；PR2 之后任意时刻停下来都是"能跑的中间态"（legacy 行为不变）。

**PR7 拆成 7a / 7b 的理由**：§9.3–§9.4 这类传播逻辑的缺陷（撕裂读取、共享读异常分类、变量初始化、锁内构建）恰恰是伪代码没有类型检查、没有执行路径覆盖才留得住的类型，真跑一遍就立刻炸。所以 7a 刻意做成"可执行的原型 + 状态机测试"，不带任何 HTTP 面：状态机全表与 §13 的全部 fix-proof 都在进程内 + 共享 dict 上直接验证，评审在真实代码而不是伪代码上进行。7b 才把它接到端点与前端。

这也让 7a 成为整条链上唯一需要精细并发推理的 PR，7b 退化成常规的接线 + 可观测性工作。

---

## 15. 与 RFC 的偏离（已评审确认，2026-08-08）

> 评审裁决：以下 12 条偏离全部**同意**。#1–#9 为原始清单（#1 / #7 / #9 按评审意见改写：#1 点明偏离窄于 RFC 字面、#7 补 RFC Part I 的 503 背书、#9 点明放弃的两个 RFC 条件）；#10–#12 为评审补入的缺失项 —— 正文早已裁决（§8.4 / §5.1 / §11.2 #1）但未列入本表，补入以使评审签字范围与实际偏离一致。

1. **用户口令摘要写在策略文件里**（RFC 原文："Passwords, JWT secrets, and API-key secrets remain in environment variables or a secret manager"）。
   - 理由：用户需求要求"用户与权限都用 YAML 管理且可热重载"，而环境变量在进程生命周期内不可重载 —— 用户放在 env 就必然做不到热改密/热加人。
   - 偏离比 RFC 那句话的字面窄：三个宾语里只动了一个。**JWT 密钥**（`TOKEN_SECRET`）仍强制留在 env（规则 #15），**API Key 明文密钥仍然只允许 `secret_env`**（它是可直接重放的 bearer 凭据，与口令摘要风险等级不同）；进文件的只有**口令的 bcrypt 摘要**。
   - 收敛做法：文件里只允许 **bcrypt 摘要**（不是明文，也不是可重放的 bearer 凭据），并保留 `password_env` 作为逃生口；同时增加文件权限告警与信任链解析校验（§7.3 #12 / #13）。
2. **"load one immutable revision per process startup" 被放宽**为"每个 revision 一份不可变快照，原子换入，请求内绑定单一快照"。不可变性这一条不变，只是不再"每进程一次"。
3. **新增两个权限码** `auth.policy.read` / `auth.policy.reload`（RFC 目录里没有，因为 RFC 没有热重载需求）。
4. **API Key 认证失败由 403 改 401**（§10.2），与 RFC 的 401/403 语义一致但与现状不同，属破坏性变更。
5. **`ce`（credential epoch）声明**是 RFC 未提及的新增 JWT 声明。它不承载权限（不违反"JWT 不含权威权限"），只承载"凭据材料是否已变"，用来把删除/禁用/改密变成即时失效。
6. **策略文件字节进共享内存**（§9.3）。RFC 只谈单进程的不可变 revision，没有多 worker 传播模型。共享的是**原始字节**而非构建好的快照，因此 RFC 的"API-key 明文只在 env / secret manager"这条不受影响：字节里只有本就落盘的 bcrypt 摘要。运行时真相因此是"最后一次应用的 revision"，磁盘退为暂存区与冷启动权威 —— §1.2 已写明。
7. **503 语义推广到全部受保护路由**（§9.4 stale worker）。这条有 RFC 自己的背书：Part II 的授权模型虽只讨论 401/403，但 **RFC Part I 的 HTTP 语义表已明文写着 "`503`: authorization, job, cache, or shared-control-plane provider unavailable"**（作用域是 artifact 端点）。本方案不是发明新语义，而是把"authorization provider 不可用 ⇒ 503"从 artifact 面推广到舰队级的所有受保护路由。它是 fail-closed 的必然产物：既不能放行，又不是调用方的错。
8. **`credential_version` 字段**（§8.2）。RFC 未涉及会话世代；`ce` 若只由 `password_hash + enabled` 决定就是可逆的，禁用后恢复会让旧令牌复活。
9. **legacy 迁移 profile 成为缺省且无移除计划**（§11.4）。RFC 对 legacy 兼容 profile 提出四个条件：*(i)* 必须显式启用（"must be explicit"）、*(ii)* 绝不隐式授予 artifact 权限、*(iii)* 打警告、*(iv)* 有移除计划（"must have a removal plan"）。本方案满足 (ii)（§11.4 三约束 + 冻结枚举），部分满足 (iii)（§11.3 第 1 步只有一条"未启用 policy 模式"提示），**放弃了 (i) 与 (iv)**：legacy 是不设 `AUTH_POLICY_FILE` 时的缺省 profile（无需任何 flag），且 §11.3 明确"不迁移也是受支持的选择"（无移除计划）。
   - 理由：RFC 假设的是全新部署；LightRAG 是 brownfield 项目，按 (i) 字面执行等于让所有存量部署启动失败，与本方案"不逼迫存量部署迁移"的取向直接冲突。
   - 保住的底线：`documents.artifacts.*` 被定为"仅 policy 模式可授予"的能力，与 RFC"下载类权限单独成码、破坏性/泄露面权限更窄"的原则同向，且让不迁移的部署天然拿不到新泄露面；`legacy_user` 是冻结枚举而非动态减法，新权限码对旧配置 deny-by-default（§11.4 约束 2）。
10. **裸部署推断为匿名开放模式**（§8.4）。RFC 原文："Anonymous development mode, if retained, requires an explicit opt-in such as `AUTH_ALLOW_UNAUTHENTICATED=true`; **it is never inferred merely because credentials are absent** and must not be suitable for network-exposed deployments." 这个 "never" 在 RFC 里没有模式限定；而 §8.4 的裁决恰恰是"三者皆未配 ⇒ 归入 legacy 开放子形态，匿名调用者持 `legacy_user`"——即从凭据缺失推断出匿名放行。§8.4 把该红线重新限定到 policy 模式，是本方案自己做的 rescoping，RFC 文本不含这个限定。
    - 理由：与 #9 同源 —— 裸部署是今天最常见的开发形态（`credentials_accepted` 第一分支），按 RFC 字面执行 = 所有 quickstart / 开发环境启动即失败，属于发生在 policy 模式之外的破坏性变更。
    - 收敛做法：仅限未设 `AUTH_POLICY_FILE` 时成立（设了就是规则 #14 的 fail closed）；新增启动横幅警告并声明不适合网络暴露；`documents.artifacts.*` 不可获得。改判路径已在 §8.4 末尾写明。
11. **API 文档面默认公开**（§5.1）。RFC 公共面清单写的是 "API documentation **only if the deployment explicitly elects to expose it**"；本方案的 `AUTH_EXPOSE_API_DOCS` 默认 `true`，默认开启不是"显式选择"。
    - 理由：四个文档端点今天就完全无认证依赖，改默认值是 policy 模式之外的行为变更；且"部分鉴权"会做出坏掉的文档页（§5.1 已论证），收敛攻击面的正确动作是整片关闭。
    - 收敛做法：单开关全有或全无，`false` 时四端点不注册（404）；已列入 §16 接受风险。
12. **`WHITELIST_PATHS` 的 `/api/*` 豁免只在 policy 模式移除**（§11.2 #1）。RFC 原文是无条件的 "The current default exemption of `/api/*` is incompatible with this model **and must be removed**"；本方案只在 policy 模式忽略该项，legacy 模式保留旧默认 + 启动告警。#9 兼容原则的衍生物：全局改默认值会让 legacy 部署的 `/api/*` 突然要求认证，与 §14 PR2"legacy 行为逐字不变"的验收标准冲突。

> 附注（不构成违例）：`/documents/scan` 挂 `documents.retry` **AND** `documents.write`（§5.2），比 RFC 初始映射（仅 `documents.retry`）多一个 scope。这是**收窄**而非放宽（堵"只能重试者经共享输入目录绕过 `documents.write` 摄取"的旁路），且 RFC 自称 "initial semantic mapping"、把精确路由清单留给实现评审，属于 RFC 授权的裁量范围。同理，`ollama.metadata.read` 不是偏离：RFC 映射表原文即 "`auth.session.read` **or a more specific read permission agreed during implementation**"，本方案选的是被授权的后一分支。

---

## 16. 遗留风险与后续阶段

| 项 | 说明 | 归属 |
| --- | --- | --- |
| ~~worker 崩溃重启导致的舰队漂移窗口~~ | **已解决**：重启的 worker 优先从共享内容构建（§9.5），拿到的是最后一次应用的 revision，不再读可能被编辑到一半的磁盘文件 | 已解决 |
| 冷启动后磁盘内容 ≠ 重启前最后一个 revision | 进程组整体重启时磁盘是权威（这是正确语义）。运维改了文件却没 reload 就整体重启，等于提前生效；靠 `/auth/policy/status` 的 `revision_source` 字段 + 运维规约可见 | 接受 |
| `password_env` / `secret_env` 按各 worker 自己的 env 解析 | 字节一致仍可能构建失败。fork 模型下所有 worker 继承同一份 env，实际极难发生；发生时该 worker 进 `stale` 并 503（§9.4），不再"拿旧快照继续放行" | 接受，文档写明 |
| `stale` worker 造成的可用性损失 | N 个 worker 里 1 个 stale ⇒ 约 1/N 请求 503。刻意选择：授权正确性优先于可用性（§9.4 已量化） | 接受 |
| `credential_version` 依赖运维纪律 | 服务端对策略文件只读，无处持久化自增计数器；"禁用后恢复 / 删除后重建必须 +1"只能靠文档 + CLI 提醒。第二阶段 DB provider 可自动化（§8.2） | 接受，第二阶段解决 |
| API 文档面（`/docs`、`/redoc`、`/openapi.json`）默认公开 | 保持现状；要收敛攻击面用 `AUTH_EXPOSE_API_DOCS=false` 整片关闭，不做部分鉴权（§5.1） | 接受 |
| API Key 明文轮换需重启 | env 属进程启动态 | 接受，文档写明 |
| legacy 部署用不上 artifact 下载 | 刻意：下载能力仅 policy 模式可授予（§11.4），也是迁移的正向激励 | 接受 |
| 无审计日志落库 | 只做结构化日志字段 `principal_type` / `principal_id` / `permission` / `decision` / `policy_revision` | 第二阶段 |
| 无 DB provider、无管理界面 | 接口已按可替换设计 | 第二阶段 |
| 无 workspace/tenant 作用域 | `AuthorizationContext` 已留扩展点，v1 文件不含相关字段 | 第三阶段 |
| artifact 导出路由尚未落地 | §5.3 预留行 + 审计强制 | 随 LR2 下载管线 |
| ~~`/api/tags`、`/api/version` 用 `auth.session.read` 过宽~~ | **已解决**：拆出 `ollama.metadata.read`（§3） | 已解决 |

---

## 附录 A：传播协议参考伪代码（规范形态）

> 本附录是 §9.3–§9.4 传播协议的**规范形态**伪代码，注释保留了各不变式的完整推导；正文（§9.3）只保留不变式清单。实现以本附录为规范形态，验收以 §13 为准。

### A.1 三态读取、换入与上报原语

```
# 三态返回。ABSENT 与 RETRY 必须分开：
#   ABSENT = 确认"键不存在"（epoch 的 ABSENT 是合法的冷启动信号，不是故障）
#   RETRY  = 读不出来，真假未知（RPC 超时/断连）= 未定失败
# 只有 build() 的 ValidationError 是确定性失败，由调用方处理。
# 必须用 shared[key] 而不是 shared.get(key)：.get() 把"不存在"塌成 None，
# 正好毁掉 ABSENT/RETRY 这个区分。写侧的不变式配套保证：epoch 只写完整记录，
# 永不写 None，所以 v is None 属于"不该发生"，保守当 RETRY。
shared_get(key) -> VALUE(v) | ABSENT | RETRY:
  try: v = shared[key]
  except KeyError: return ABSENT                    # 确定不存在
  except (TimeoutError, ConnectionError, EOFError, BrokenPipeError): return RETRY
  return RETRY if v is None else VALUE(v)

# 唯一的换入动作，**纯本地、零 RPC、可在锁内外任意调用**：
# swap + confirm 绑在一起，谁都不许单独做 swap。共享上报是另一件事，见下。
apply_snapshot(snap):
  PolicyRuntime.swap(snap)
  local.confirm()                                   # ← 正向确认就在这里，不在别处

# 共享上报：自己取 auth_policy 锁，**只能在未持锁时调用**。
# 签名只有 state / message —— 版本字段一律取自本地已换入的快照，调用方无从传错。
# 写的永远是**完整的顶层记录**（不是嵌套、不是增量），因为收敛判定直接读这些顶层字段。
publish_report(state, message="") -> bool:          # True = 这次写成功了
  cur_snap = PolicyRuntime.current()
  rec = {"pid": pid, "start_id": ...,
         "revision": cur_snap.revision,             # ← 我**正在服务**的版本，
         "digest":   cur_snap.source_digest,        #   不是我想采纳的 target
         "revision_source": cur_snap.revision_source,
         "adoption_source": cur_snap.adoption_source,
         "state": state, "message": message,
         "adopted_at": cur_snap.loaded_at, "reported_at": now}
  try:
      在 get_namespace_lock("auth_policy") 内：      # adoptions 整体读-改-写
          cur = shared_get("adoptions")
          if cur is RETRY: return False              # ← 读不出来就放弃本次写：
                                                     #   绝不能当空字典覆盖别人的记录
          a = {} if cur is ABSENT else dict(cur)     # ABSENT 才是"初始化空字典"
          a[str(pid)] = rec                          # 整体替换本 pid 的记录
          shared["adoptions"] = a
      return True
  except Exception as e:                            # 观测失败 ≠ 授权失败（但见启动期注册）
      log.warning(...)                              # 由调度点在下一 tick 重试
      return False

# 上报调度点：**所有主动上报都走它** —— register_or_die、reload 出锁后、tick 的每条
# 退出路径。它决定"这次要不要写"：意图记录变了 ⇒ 立即写；没变 ⇒ 只在距上次成功
# 写入 ≥ AUTH_POLICY_HEARTBEAT_SECONDS 时写（这就是心跳的唯一实现）。调度状态只在
# 写成功后推进，失败自动留给下一次调用重试。
# 注意：调度状态是**观测侧**的，绝不参与 is_stale() —— §9.4 的授权判定
# 仍只读 last_confirmed_at / deterministic_error 一个时钟，单时钟不变式不受影响。
#
# 调用方有两个（tick 任务、reload 端点协程），必须用进程内 publish_mutex 串行化
# **整个调度点**，rec 与 due 都在锁内重算。不串行化的交错是实打实的乱序覆盖：
# publish_report 里 current() 的读取发生在 await 共享锁**之前**，于是旧心跳可以
# 先按 rev7 算好记录、在共享锁上等待；reload 换入并上报 rev8；旧心跳恢复后把
# rev7 覆盖回共享表、还把 last_published 一并回退 —— 刚成功的 reload 在 /status
# 上显示未收敛，要等下一 tick 才修复。串行化后，最后完成的写入必然是在最后一次
# swap 之后才计算的（每次 swap 的执行方随后都要进调度点排队），共享表的**最终**
# 状态因此总是正确；中间至多出现一次立刻被后续排队写入覆盖的过期写入，窗口以
# 锁队列为界，不再是"等到下一 tick"。
last_published    = None      # 上次成功写入的意图记录（不含时间戳字段）
last_published_at = None      # 上次成功写入的时刻（monotonic）
publish_mutex     = asyncio.Lock()   # 进程内锁，与跨进程的 NamespaceLock 无关

intended_record():            # 具名记录，**不含任何时间戳字段**（否则永远"变了"）
  snap  = PolicyRuntime.current()
  state = ("error"   if local.deterministic_error
           else "pending" if local.last_attempt_kind in (UNREACHABLE, CONTENDED)
           else "ok")
  # ↑ "ok" 只留给 CONFIRMED / ADOPTED —— "最近没确认上"绝不能自称 ok。
  #   否则 epoch 读持续失败而 adoptions 写正常的 worker 会一边本地 503（时钟
  #   过期）、一边持续上报新鲜的 ok@rev7，远端 converged 照样为真 —— 又一种
  #   假收敛。pending 是如实的自报："我还在服务旧快照，但我确认不了自己在
  #   目标上"；恢复正向确认后 confirm() 把枚举写回 CONFIRMED，state 立即回 ok
  #  （记录变化 ⇒ 调度点立即写）。
  return Record(state   = state,
                message = local.deterministic_error or "",
                revision=snap.revision, digest=snap.source_digest,
                revision_source=snap.revision_source,
                adoption_source=snap.adoption_source)

maybe_publish() -> bool:      # True = 意图记录此刻已在共享表中
  async with publish_mutex:   # 串行化整个调度点（理由见上）
      rec = intended_record()                        # ← 锁内重算，锁外算的可能已过期
      due = (last_published_at is None
             or monotonic() - last_published_at >= AUTH_POLICY_HEARTBEAT_SECONDS)
      if rec == last_published and not due: return True   # 稳态：零写入、零 RPC
      if publish_report(state=rec.state, message=rec.message):
          last_published, last_published_at = rec, monotonic()
          return True
      return False            # 共享不可达时到期重试会连续失败，那是故障模式下的
                              # 廉价重试，不违反"稳态不每 tick 写"的成本约束
```

### A.2 采纳咽喉与启动协议

```
# 采纳咽喉。RETRY 必须带原因，否则调用方没法填 last_attempt_kind：
#   UNREACHABLE = 共享状态读不出来；CONTENDED = 读到了但内容/版本在换代中
adopt_from_shared(target) -> snapshot | RETRY(kind):
  raw = shared_get("content")                       # 与 target 是两次独立读取
  if raw is not VALUE: return RETRY(UNREACHABLE)    # content 的 ABSENT 也是异常（发布保证它存在）
  if sha256(raw) != target.digest: return RETRY(CONTENDED)   # ← 撕裂读取守卫，唯一实现
  snap = build(raw, target)                         # 锁外；重活全在这里。
                                                    # revision_source 抄 target 的，
                                                    # adoption_source 固定 "shared"
  now = shared_get("epoch")                         # 构建期间又落了一次 reload？
  if now is not VALUE: return RETRY(UNREACHABLE)
  if now != target:    return RETRY(CONTENDED)
  return snap

startup(最多 MAX_STARTUP_ATTEMPTS 轮，之间让出事件循环):
  epoch = shared_get("epoch")                   # 第一步就是它，1 次小 RPC，不进锁
  if epoch is RETRY: 重试本轮                   # 读不出来 ≠ 不存在，绝不能当冷启动
  if epoch is VALUE:
      # ── 替换 worker（舰队已在运行）：绝不读磁盘 ──
      snap = adopt_from_shared(epoch)
      if snap is RETRY: 重试本轮
      apply_snapshot(snap)                      # 本地换入，未持锁
      register_or_die(); return                 # ← 首次上报必须成功，见下
  # ── epoch is ABSENT ⇒ 冷启动：磁盘是权威 ──
  raw, draft = read_and_validate_file()         # 锁外做全部重活；draft 不带 revision
                                                # 失败 ⇒ 进程退出（规则 #14）
  won = False
  在 get_namespace_lock("auth_policy") 内：     # 1 读 + 2 写 + 2 次 O(1) 本地操作
      probe = shared_get("epoch")               # ← 与锁外同一个三态判定，不是 `is None`
      if probe is RETRY: 出锁并重试本轮          # 锁内不重试，不占着锁等 RPC
      if probe is ABSENT:                       # 我抢到了发布权
          shared["content"] = raw
          shared["epoch"]   = {revision: 1, digest: sha256(raw),
                               revision_source: "disk", ...}
          target, won = shared["epoch"], True
          apply_snapshot(replace(draft, revision=1))   # 纯本地：replace + swap + confirm；
                                                       # draft 两个 source 字段皆 "disk"
      else:
          target = probe                        # VALUE：别人先发布了
  # ↑ 出锁。共享上报一律在锁外，因为 publish_report 要自己取同一把锁
  if won: register_or_die(); return
  snap = adopt_from_shared(target)              # 落败：丢弃 draft，锁外重建
  if snap is RETRY: 重试本轮
  apply_snapshot(snap); register_or_die()

# 启动期注册：**lifespan 的 yield 之前**完成，失败即抛出 ⇒ 该 worker 退出。
# 也走调度点而不是裸 publish_report：注册成功即播种调度状态，
# 首个心跳周期从注册时刻起算，不会出现"注册了却从不心跳"的缝隙。
register_or_die():
  for _ in range(MAX_STARTUP_ATTEMPTS):
      if maybe_publish(): return
      让出事件循环
  raise RuntimeError("cannot register in shared adoptions")   # → worker 起不来
```

### A.3 写入侧（reload 端点临界区）

```
锁外：raw = read(file); d = sha256(raw); draft = validate_and_build(raw, digest=d)   # 不带 revision
在 get_namespace_lock("auth_policy") 内：
  probe = shared_get("epoch")            # 同一个三态判定；RETRY ⇒ 出锁，端点回 503
  next_rev = 1 if probe is ABSENT else probe.revision + 1
  shared["content"] = raw                # 本进程刚校验通过的那一份
  shared["epoch"]   = {revision: next_rev, digest: d, revision_source: "reload", ...}
  target = shared["epoch"]
  apply_snapshot(replace(draft, revision=next_rev))
  #   ↑ 取号后才能定稿快照；纯本地（swap + confirm），必须走它，否则发起 reload 的
  #     这个 worker 自己不会 confirm()，会一直 stale/503 到它的下一次 tick。
  #     定稿快照：revision_source="reload"、adoption_source="reload_local"
出锁后：maybe_publish()                 # revision 变了 ⇒ 意图记录必然不同 ⇒ 立即写；
                                       # 失败只记日志（下一 tick 重试），
                                       # 不影响本次 reload 的成功语义
```

### A.4 读取侧（轮询 tick 与本地状态机）

```
tick:                       # 结构 = 先推进本地状态，再经唯一调度点决定要不要上报
  step()                    # 纯判定与换入；**绝不直接调 publish_report**
  maybe_publish()           # ← 所有 tick 退出路径都经过这里；心跳就住在这里。
                            #   稳态且未到期时它是零 RPC 的本地比较

step():                     # 每条退出路径都要么 confirm()，要么留下 last_attempt_kind
  epoch = shared_get("epoch")                   # ← 也走咽喉。稳态整个 tick 只有这 1 次小 RPC
  if epoch is not VALUE:                        # RETRY 或 ABSENT：本 tick 无法确认，
      local.attempt(UNREACHABLE); return        # 不更新 last_confirmed_at（见下）
  if (epoch.revision, epoch.digest) == (local.revision, local.digest):
      local.confirm(); return                   # ← 稳态的正向确认就在这里
  try: snap = adopt_from_shared(epoch)          # 同一咽喉：digest 守卫 + build + 复检
  except ValidationError as e:                  # 确定性失败：立即 stale，不走宽限期
      local.attempt(ERROR, redact(str(e)))      # → §9.4，本 worker 起 503；
      return                                    #   上报交给调度点：状态变了会立即写，
                                                #   持续失败则只按心跳频率写
  if snap is RETRY:                             # 未定失败：不确认，宽限期到点转 stale
      local.attempt(snap.kind); return          # 原因由 adopt_from_shared 带回来
  apply_snapshot(snap); local.attempt(ADOPTED)  # 换入咽喉（纯本地）；revision 变了 ⇒
                                                # 调度点会立即写，不等心跳到期

# 进程内状态：一个时钟 + 一个确定性错误 + 一个"上次尝试结果"枚举（不带时间）
last_confirmed_at   = None       # 初值 None = 从未确认 ⇒ is_stale() 为真
deterministic_error = None
last_attempt_kind   = None       # CONFIRMED | ADOPTED | UNREACHABLE | CONTENDED | ERROR

local.confirm():   last_confirmed_at = monotonic(); deterministic_error = None
                   last_attempt_kind = CONFIRMED
local.attempt(k, msg=None):      # 未能确认的那些退出路径记下"为什么"
                   last_attempt_kind = k
                   if k is ERROR: deterministic_error = msg
local.is_stale():  return (deterministic_error is not None
                           or last_confirmed_at is None
                           or monotonic() - last_confirmed_at > STALE_AFTER)
                   # STALE_AFTER = max(AUTH_POLICY_STALE_GRACE_SECONDS,
                   #                   3 × AUTH_POLICY_RELOAD_POLL_SECONDS)

# 成因推导：时钟判"是否 stale"，枚举判"为什么"。不需要第二个时钟。
local.stale_cause():
  if deterministic_error: return "error"                      # 查策略文件
  if last_attempt_kind in (CONFIRMED, ADOPTED): return "stalled"
  #  ↑ 上一次尝试是成功的，时钟却过期了 ⇒ 之后根本没有尝试发生 ⇒ 轮询任务停摆
  return last_attempt_kind.lower()   # "unreachable"（查 Manager）/ "contended"
```
