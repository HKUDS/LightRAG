# Hologres 存储指南

本文说明如何让 LightRAG 将四类存储全部接入 Hologres，比较两种图存储实现，并介绍 Hologres 后端的离线测试、真实实例集成测试与覆盖率要求。

目标读者包括：部署 LightRAG + Hologres 的运维人员，以及维护 Hologres 后端的开发人员。

## 目录

- [1. 环境要求与安装](#1-环境要求与安装)
- [2. 快速开始](#2-快速开始)
- [3. 环境变量](#3-环境变量)
- [4. 选择图存储后端](#4-选择图存储后端)
- [5. Schema 生命周期与 Workspace 隔离](#5-schema-生命周期与-workspace-隔离)
- [6. Python SDK 用法](#6-python-sdk-用法)
- [7. 运行注意事项](#7-运行注意事项)
- [8. 测试与覆盖率](#8-测试与覆盖率)
- [9. 故障排查](#9-故障排查)

## 1. 环境要求与安装

### 服务端要求

- Hologres **5.0 或更新版本**。
- 一个已经创建并可访问的数据库。LightRAG 会在数据库内创建 schema 和表，但不会创建数据库实例。
- 数据库用户需要具有：
  - 在目标数据库中创建配置 schema 和迁移 ledger 的权限；
  - 创建和修改后端表的权限；
  - 执行能力探测与 schema 校验所需 catalog 读操作的权限；
  - 选择 `HologresAGEGraphStorage` 时，创建和删除 AGE graph 对象的权限。

默认四类存储对应关系如下：

| LightRAG 存储角色 | 实现类 |
|---|---|
| KV 存储 | `HologresKVStorage` |
| 向量存储 | `HologresVectorStorage` |
| 图存储 | `HologresGraphStorage` |
| 文档状态存储 | `HologresDocStatusStorage` |

`HologresAGEGraphStorage` 是图存储角色的可选替代实现。

### 安装 Python 依赖

Hologres 后端依赖 `asyncpg`，该依赖包含在 `offline-storage` extra 中。

源码部署：

```bash
uv sync --extra offline-storage
```

包部署：

```bash
pip install lightrag-hku[offline-storage]
```

安装依赖并配置 `.env` 后，按常规方式启动 API 服务即可。

## 2. 快速开始

在 `.env` 中加入 Hologres 连接配置，并选择四个 Hologres 存储实现：

```ini
# 替换为你的实例信息。
HOLOGRES_HOST=example.hologres.aliyuncs.com
HOLOGRES_PORT=80
HOLOGRES_USER=your_username
HOLOGRES_PASSWORD=your_password
HOLOGRES_DATABASE=your_database
HOLOGRES_SCHEMA=lightrag

# 一个逻辑 LightRAG 实例对应一个 workspace。
WORKSPACE=project_a

# 选择 Hologres 一体化存储。
LIGHTRAG_KV_STORAGE=HologresKVStorage
LIGHTRAG_VECTOR_STORAGE=HologresVectorStorage
LIGHTRAG_GRAPH_STORAGE=HologresGraphStorage
LIGHTRAG_DOC_STATUS_STORAGE=HologresDocStatusStorage
```

保留现有的 LLM 与 Embedding 配置。特别地，`EMBEDDING_MODEL` 和 `EMBEDDING_DIM` 必须与实际创建向量的模型一致。

如需改用 AGE 图存储，只修改图存储选择：

```ini
LIGHTRAG_GRAPH_STORAGE=HologresAGEGraphStorage
```

选择 AGE 前，请先阅读[选择图存储后端](#4-选择图存储后端)。

## 3. 环境变量

所有 Hologres 配置都使用 `HOLOGRES_` 前缀。workspace 遵循数据库后端统一的优先级链：`HOLOGRES_WORKSPACE` > LightRAG 实例 `WORKSPACE` > `default`。

### 连接配置

| 变量 | 默认值 | 合法值 | 用途 |
|---|---:|---|---|
| `HOLOGRES_HOST` | 必填 | 非空字符串 | Hologres 接入点 |
| `HOLOGRES_PORT` | `80` | `1`–`65535` | Hologres 端口 |
| `HOLOGRES_USER` | 必填 | 非空字符串 | 数据库用户 |
| `HOLOGRES_PASSWORD` | 必填 | 非空字符串 | 数据库密码 |
| `HOLOGRES_DATABASE` | 必填 | 非空字符串 | 已存在的数据库 |
| `HOLOGRES_SCHEMA` | `public` | SQL 标识符，最长 63 字节 | 存放 LightRAG 对象的 schema |
| `HOLOGRES_WORKSPACE` | 未设置 | 合法的 LightRAG workspace | 覆盖所有 Hologres 存储角色使用的 workspace |
| `HOLOGRES_SSL_MODE` | `prefer` | `disable`、`allow`、`prefer`、`require`、`verify-ca`、`verify-full` | asyncpg SSL 模式 |

Schema 标识符必须以 ASCII 字母或 `_` 开头，后续字符只能是 ASCII 字母、数字或 `_`。

### 连接池、超时与重试配置

| 变量 | 默认值 | 合法范围 | 用途 |
|---|---:|---|---|
| `HOLOGRES_POOL_MIN_SIZE` | `1` | `1`–`1000` | 最小连接池大小 |
| `HOLOGRES_POOL_MAX_SIZE` | `10` | `1`–`1000` | 最大连接池大小，不得小于最小值 |
| `HOLOGRES_CONNECT_TIMEOUT` | `10` | `0.001`–`600` 秒 | 建连超时 |
| `HOLOGRES_COMMAND_TIMEOUT` | `60` | `0.001`–`3600` 秒 | 默认 SQL 命令超时 |
| `HOLOGRES_POOL_ACQUIRE_TIMEOUT` | `30` | `0.001`–`600` 秒 | 获取连接的等待超时 |
| `HOLOGRES_POOL_CLOSE_TIMEOUT` | `5` | `0.001`–`60` 秒 | 连接池关闭超时 |
| `HOLOGRES_CONNECTION_RETRIES` | `2` | `0`–`20` | 连接重试次数 |
| `HOLOGRES_RETRY_BACKOFF` | `0.5` | `0`–`60` 秒 | 指数退避初始值 |
| `HOLOGRES_STATEMENT_CACHE_SIZE` | `100` | `0`–`10000` | asyncpg 语句缓存大小 |

### 能力开关

| 变量 | 默认值 | 用途 |
|---|---:|---|
| `HOLOGRES_STREAM_COPY_ENABLED` | `false` | 为符合条件的批量写入开启 stream COPY。 |
| `HOLOGRES_AGE_SEARCH_PATH` | `false` | 面向调用方自供专用 AGE client 的内部兼容开关。 |
| `HOLOGRES_AGE_ALLOW_UNSUPPORTED` | `false` | 当 AGE 扩展缺失时，显式接受两张表 fallback 的开关。 |

布尔值不区分大小写，接受 `1/true/yes/on` 与 `0/false/no/off`。

普通部署应保持 `HOLOGRES_AGE_SEARCH_PATH=false`。`HologresAGEGraphStorage` 会创建自己的专用 client，并自行设置 AGE 所需 search path。

### Stream COPY 行为

Stream COPY 是显式 opt-in 的优化：

1. `HOLOGRES_STREAM_COPY_ENABLED=true` 打开配置开关。
2. 初始化时执行非阻塞能力证明。
3. 只有配置开关与能力证明都通过时，批量写入才使用 stream COPY。
4. 能力证明失败不会阻断启动；写入继续使用参数化 INSERT。

该开关只影响写入路径性能，不影响正确性语义。

## 4. 选择图存储后端

### `HologresGraphStorage`：默认实现

默认图存储使用两张共享 Hologres 表保存节点和边：

```text
lightrag_hologres_graph_nodes
lightrag_hologres_graph_edges
```

这是推荐默认值。它满足当前支持的 Hologres 基线版本，不依赖 Apache AGE。

适合以下场景：

- 希望使用兼容性最好的 Hologres 图后端；
- 希望图数据存放在按 workspace 逻辑分区的共享表中；
- 不希望引入 AGE 能力探测和 AGE 语义差异。

### `HologresAGEGraphStorage`：可选实现

AGE 后端使用 Hologres 内嵌的 Apache AGE 引擎：

```ini
LIGHTRAG_GRAPH_STORAGE=HologresAGEGraphStorage
```

初始化时会探测 AGE 合同。探测通过后，每个 workspace 对应一个物理 AGE graph：

```text
lightrag_age_<workspace>
```

探测结果按确定性分为两类：

- **AGE 扩展缺失**（detail 为 `age_extension_missing`）—— 结论明确，但 fallback 的物理图 schema 不同。初始化默认失败；只有设置 `HOLOGRES_AGE_ALLOW_UNSUPPORTED=true` 后，才会记录 WARNING 并委托给 `HologresGraphStorage`。
- **其他任何探测失败**（瞬断、权限问题、探测行为异常等）—— 结果不确定。初始化会**直接失败**而不是回退：静默回退会把写入落到另一个物理存储，并使既有 AGE 图数据不可见。

运行注意事项：

- AGE 与两张表实现的物理存储不同，切换实现不会迁移图数据。
- 如果初始化报 AGE 探测错误，先修复根因（连通性、权限）再重启；若 workspace 已有 AGE 图数据，不要切换到两张表实现。
- 不要为普通共享 client 手动设置 `HOLOGRES_AGE_SEARCH_PATH=true`；AGE 存储会自行管理专用 client。

## 5. Schema 生命周期与 Workspace 隔离

### 配置 schema 中创建的对象

初始化时，LightRAG 会在配置 schema 不存在时创建它，并应用所需对象。

固定对象名包括：

| 对象 | 用途 |
|---|---|
| `lightrag_hologres_schema_ledger` | 幂等、可恢复的迁移 ledger |
| `lightrag_hologres_kv` | 共享 KV 表 |
| `lightrag_hologres_doc_status` | 文档状态表 |
| `lightrag_hologres_vectors` | 共享向量表 |
| `lightrag_hologres_graph_nodes` | 两张表图存储的节点表 |
| `lightrag_hologres_graph_edges` | 两张表图存储的边表 |

AGE 后端还会创建名为 `lightrag_age_<workspace>` 的 AGE graph namespace。

### 迁移行为

Schema manager 会：

- 使用幂等单语句创建 schema 和 ledger；
- 在 ledger 中记录每个 descriptor 的 digest 与状态；
- 按确定顺序应用增量 schema 变更；
- 通过 ownership 和 lease 信息协调并发初始化；
- 应用 descriptor 后校验 catalog 后置条件；
- 当迁移状态或 catalog 形状无法证明一致时 fail closed。

不要手工修改这些表来绕过迁移失败。应保留 ledger 与服务端错误详情，停止其他写入者，然后重试初始化或基于 ledger 状态排查。

### Workspace 隔离

workspace 遵循 `HOLOGRES_WORKSPACE` > 实例 `WORKSPACE` > `default`：

| 后端 | 隔离方式 |
|---|---|
| `HologresKVStorage` | `workspace` 列 |
| `HologresVectorStorage` | `workspace` 与 namespace 列 |
| `HologresDocStatusStorage` | `workspace` 列 |
| `HologresGraphStorage` | `workspace` 与 namespace 列 |
| `HologresAGEGraphStorage` | 每个 workspace 一个物理 AGE graph |

Workspace 名称只能使用 ASCII 字母、数字和 `_`。LightRAG 实例初始化后，不要更改 workspace 值。

## 6. Python SDK 用法

SDK 使用相同的存储类名：

```python
from lightrag import LightRAG

rag = LightRAG(
    working_dir="./rag_storage",
    workspace="project_a",
    kv_storage="HologresKVStorage",
    vector_storage="HologresVectorStorage",
    graph_storage="HologresGraphStorage",
    doc_status_storage="HologresDocStatusStorage",
    # 继续提供与服务端部署相同的 LLM、tokenizer 和 embedding 配置。
)

# 必须调用：初始化 Hologres client 并协调 schema。
await rag.initialize_storages()

try:
    await rag.ainsert("Hologres 是实时数据仓库。")
finally:
    await rag.finalize_storages()
```

使用 AGE 时：

```python
graph_storage="HologresAGEGraphStorage"
```

除非你自己构造并注入后端 client，连接配置仍来自 `HOLOGRES_*` 环境变量。

## 7. 运行注意事项

### Embedding 模型稳定性

向量表与 embedding 维度绑定。索引和查询必须使用同一个 embedding 模型和维度。

如果模型或维度变化：

1. 停止写入者；
2. 保留图存储与 KV 数据源；
3. 使用新的模型和维度重建向量存储；
4. 重建成功后再恢复查询与写入。

### 凭据安全

Hologres 配置与后端诊断信息会对 host、user、password 和 database 脱敏。尽管如此，仍应将 `.env` 和部署密钥视为敏感信息；API 服务若绑定非 loopback 地址，必须配置认证。

### 并发写入者

一个逻辑知识库对应一个 `WORKSPACE`。多个 server worker 或存储实例可以并发初始化同一个 schema；迁移 ledger 会协调增量 descriptor 应用，并在 catalog 状态无法证明一致时 fail closed，而不是猜测状态。

### 已接受的写入残留

- **两张表图实现的端点创建。** node upsert 可能先创建端点 stub，edge upsert 也可能先补建缺失端点。Hologres 后端写入是单条 autocommit 语句，因此中断可能留下 stub 节点。后续再次 upsert 该边、执行 rebuild，或通过 purge/rebuild 会重写或清理它；这与 PostgreSQL 表格图实现接受的端点 stub 残留一致。
- **KV full-docs 的合并范围。** PostgreSQL 使用固定 full-docs 列，而 JSONB upsert 会合并本次提供的键并保留历史写入的键。如需清除非受保护键，请显式写入 `null`；该键仍会存在但值为 null，不会被移除。六个受保护字段仍使用与 PostgreSQL 兼容的恢复语义。

doc-status 有两个有意的兼容差异：upsert 校验采用整批 fail-closed（PostgreSQL 是跳过非法记录并记录日志）；在 Hologres 受限 client 下，每条记录是一条 replay-safe 单语句。配置只读取 `HOLOGRES_*` 环境变量；`config.ini` 和通用 `get_env_value` 间接层不属于这个隔离后端。测试套件中 Hologres 专用 integration gating 已在 PR 描述和 `tests/conftest.py` 中说明。

### AGE graph 清理

通过 `HologresAGEGraphStorage` 执行 drop 会清除该 workspace 的图内容。live 测试还会在清理阶段删除随机创建的 AGE graph namespace，避免测试对象累积。

## 8. 测试与覆盖率

### 离线单元测试

离线测试使用 fake client，不需要真实 Hologres：

```bash
./scripts/test.sh tests/kg/hologres_impl
```

当前预期：全部测试通过，live-only 测试默认跳过。

### Live 集成测试

Live 测试需要在环境变量中提供与后端相同的真实 Hologres 配置：

```bash
export HOLOGRES_HOST=...
export HOLOGRES_PORT=80
export HOLOGRES_USER=...
export HOLOGRES_PASSWORD=...
export HOLOGRES_DATABASE=...

./scripts/test.sh tests/kg/hologres_impl/test_hologres_live.py \
  --run-hologres-live -q
```

专用参数 `--run-hologres-live` 已足够，不需要同时传 `--run-integration`。测试 fixture 会创建隔离的随机 `lightrag_test_*` schema 并清理，不会使用应用 workspace。

Live 测试覆盖真实能力探测、schema 恢复、CRUD、相似度查询、图行为，以及 AGE 跨 chunk hydration 回归，因此可能需要数分钟。

### 覆盖率

该后端的维护阈值是：**`lightrag/kg/hologres/` 下每个模块语句覆盖率不低于 90%**。

可在不把 `coverage` 加入项目依赖的情况下测量：

```bash
uv run --extra pytest --with coverage coverage run \
  --data-file=/tmp/hologres.coverage \
  --source=lightrag/kg/hologres \
  -m pytest tests/kg/hologres_impl -q

uv run --extra pytest --with coverage coverage report \
  --data-file=/tmp/hologres.coverage -m
```

提交 `2f250901` 时测得的语句覆盖率为：

| 模块 | 覆盖率 |
|---|---:|
| `__init__.py` | 100.00% |
| `capabilities.py` | 91.83% |
| `client.py` | 91.84% |
| `config.py` | 93.16% |
| `doc_status.py` | 90.81% |
| `graph.py` | 95.25% |
| `graph_age.py` | 92.75% |
| `kv.py` | 90.62% |
| `schema.py` | 94.90% |
| `vector.py` | 90.44% |

这些数字是时间点快照；长期要求是每个模块不低于 90%。

## 9. 故障排查

### 启动提示 Hologres 配置无效

检查必填项是否缺失、数值范围是否非法、SQL schema 标识符是否非法，或 `HOLOGRES_POOL_MAX_SIZE` 是否小于 `HOLOGRES_POOL_MIN_SIZE`。错误信息会指出变量名，但不会回显凭据。

### 启动拒绝 Hologres 版本

实例必须识别为 Hologres 5.0 或更新版本。请先连接目标 endpoint 确认版本，再重试。

### 阻塞型能力探测失败

阻塞型探测表示服务端无法安全支持该后端所需的 SQL 或 driver 行为。不要绕过这些检查。应使用 Hologres 5.0+，并保留探测 detail 与服务端诊断。

### AGE 启动报探测错误

AGE 扩展缺失时默认启动失败，除非 `HOLOGRES_AGE_ALLOW_UNSUPPORTED=true` 显式接受两张表 fallback。其他探测失败始终终止初始化并给出 detail code。请修复连通性或权限后重启。若既有图数据存放在 AGE 中，不要开启 fallback；两种实现使用不同物理存储。

### Vector 启动拒绝余弦函数方向

`approx_cosine_distance` 必须返回相似度（越大越相近）。Vector 存储在初始化时用常量向量查询验证方向，服务端若返回距离语义则拒绝提供服务，否则查询会悄悄返回最不相似的行。请记录报错并核实服务端 HGraph 构建。

### Schema 初始化失败

停止并发写入者，并使用相同 schema 与 workspace 重试。如果仍失败，检查 `lightrag_hologres_schema_ledger` 状态与服务端错误。不要只删除表但保留应用数据；当迁移证据与 catalog 状态不一致时，后端会 fail closed。

### 获取连接或建连超时

检查网络可达性、Hologres 配额、最大连接数和凭据。确认实例健康后再提高 `HOLOGRES_POOL_ACQUIRE_TIMEOUT` 或连接重试次数；过长超时可能掩盖故障。

### 向量维度错误

确认 `EMBEDDING_DIM` 与所选 embedding 模型。表创建后若出现不匹配，通常需要使用统一的新模型和维度重建向量数据。
