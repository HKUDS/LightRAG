# LightRAG 主创新第一版改造方案

## 0. Ablation 是什么

`ablation` 指的是“消融实验”。

在方法论文里，它的作用是把一个完整方法拆成若干模块，分别做“去掉某个模块”或“只保留某个模块”的对比，观察性能变化，从而回答两个问题：

1. 这个模块到底有没有贡献。
2. 这个模块的贡献是不是独立且稳定的。

放到你的课题里，后面最自然的消融顺序是：

1. `LightRAG 原版单摘要`
2. `+ Query-Conditioned Profiling`
3. `+ Evidence-Grounded Composition`
4. `+ Cost-Controlled / Cached Profiling`

---

## 1. 第一版目标

这一版只实现主创新点的最小可运行闭环，不同时铺开所有能力。

### 1.1 范围

- 只做`实体侧`
- 只支持`local`查询模式
- 只做`离线 facet-schema-constrained profile 生成`
- 在线只做`profile 选择/组合`
- 保留现有图节点 `description` 作为稳定 fallback
- 不改动当前实体/关系抽取主流程

### 1.2 暂不纳入第一版

- 关系侧 profile
- `global / hybrid / mix` 模式的 profile 化
- fragment/fact 级 evidence grounding 实现
- 专门的 profile cache / cost-control 模块
- 文档删除后的 profile 重建与 profile 清理
- 已有历史索引数据的自动迁移

---

## 2. 设计原则

### 2.1 仿照现有 LightRAG 风格

- 不推翻现有 `entity -> graph node + entities_vdb` 主链
- 新增 sidecar storage，而不是重写图 schema
- 保持 `description` 字段仍然存在，避免打断下游上下文构建
- 查询路径采取“两阶段”风格：
  - 第一阶段：仍按当前方式召回 entity
  - 第二阶段：仅在召回结果内部做 profile 选择

### 2.2 Facet-Schema-First

V1 不允许 LLM 自由发明 view/facet 集合，而采用人工定义的弱约束 facet schema。

核心约束是：

- facet 集合由本地配置定义，不由 LLM 生成
- LLM 只能在既定 facet schema 内填充 profile 内容
- parser 以 `facet_id` 为准，不以模型自由命名为准
- 每条 profile 都绑定固定 facet，便于后续检索、缓存、评测和消融

### 2.3 为副创新 1 预留接口

V1 不实现完整的 `Evidence-Grounded Profile Composition`，但 profile 主 schema 必须先预留证据位。

最低限度需要预留：

- `support_chunk_ids`
- `support_fragment_ids`
- `grounding_status`

这样后续副创新 1 可以直接把 profile 从“chunk-level grounded”升级到“fragment/fact-level grounded”，而不必重新设计 profile 主 schema。

---

## 3. 主创新 V1 的核心思路

当前 LightRAG 的实体表示是：

1. 多个 chunk 中抽出同名实体片段
2. 合并 `source_id / entity_type / description`
3. 用 LLM 或直接拼接得到唯一一份 `description`
4. 把这一份 `description` 写进 graph node 和 `entities_vdb`

V1 改成：

1. 上述单摘要流程保留
2. 在实体 merge 完成后，基于同一实体的 `description_list` 和固定 facet schema 离线生成多个 facet-specific profiles
3. profile 单独存入 `entity_profiles(KV)` 和 `entity_profiles_vdb(Vector)`
4. 查询时先按现有方式召回 entity
5. 对已召回 entity，再按 query 从其 profiles 中选出最相关的 `top-1/top-2`
6. 将选中的 profile 组合成当前 query 下的“条件化实体描述”
7. 用该条件化描述替换 query 阶段送入上下文的 entity `description`

这样做有两个直接好处：

- 对现有代码扰动最小
- 后面扩展到 relation side 时，模式可以复用

### 3.1 V1 默认实体 facet schema

V1 推荐默认固定为 `4` 个 facet，不建议少于 `3` 个，也不建议一开始多于 `5` 个。

原因：

- 少于 `3` 个，query-conditioned selection 很难真正体现“多视角”
- 多于 `5` 个，facet 边界会变薄，profile 容易空洞和重叠
- `4` 个 facet 对人物、组织、概念、方法、系统都还有较好的通用性

V1 默认 facet schema 建议如下：

1. `identity_definition`
   - 定义实体“是什么”，强调类别、边界、核心身份，不讲动态行为。
2. `attributes_composition`
   - 描述实体的静态属性、组成、特征、规格，不讲职责和因果过程。
3. `role_function`
   - 描述实体承担什么角色、有什么功能、发挥什么用途，不讲内部机制。
4. `state_behavior`
   - 描述实体的动态状态、活动、运行表现、变化方式，不讲静态定义。

这 4 个 facet 的边界是刻意拉开的，目的是减少重叠。

V1 允许通过配置覆盖 facet schema，但不建议运行时动态生成 facet schema。

---

## 4. V1 新增数据结构

第一版建议新增 3 个 schema 类型，位置放在 [`lightrag/base.py`](/mnt/data_nvme/code/LightRAG/lightrag/base.py) 中，紧跟 `TextChunkSchema` 后面，保持当前风格。

### 4.1 `EntityFacetSchemaItem`

```python
class EntityFacetSchemaItem(TypedDict):
    facet_id: str
    facet_name: str
    definition: str
    include: list[str]
    exclude: list[str]
```

### 4.2 `EntityProfileSchema`

```python
class EntityProfileSchema(TypedDict):
    profile_id: str
    facet_id: str
    facet_name: str
    facet_definition: str
    profile_text: str
    support_chunk_ids: list[str]
    support_fragment_ids: list[str]
    grounding_status: str
    created_at: int
```

### 4.3 `EntityProfilesRecordSchema`

```python
class EntityProfilesRecordSchema(TypedDict):
    entity_name: str
    entity_type: str
    base_description: str
    source_ids: list[str]
    source_id: str
    file_path: str
    facet_schema_id: str
    facet_schema_version: int
    default_facet_id: str
    facet_catalog: list[EntityFacetSchemaItem]
    facet_ids: list[str]
    profile_ids: list[str]
    profiles: list[EntityProfileSchema]
    count: int
```

说明：

- `base_description`：保存当前 LightRAG 原生单摘要，作为 fallback
- `source_ids`：保存完整 chunk id 列表，仿照 `entity_chunks`
- `source_id`：保留图/VDB 兼容用的拼接字符串
- `facet_catalog`：记录生成该 profile 集合时使用的 facet schema
- `facet_ids`：记录当前 schema 中允许出现的 facet 集合
- `profiles`：真正的多视角 profile 列表
- `support_chunk_ids / support_fragment_ids / grounding_status`：为副创新 1 预留证据接口

---

## 5. V1 新增存储

第一版新增 2 个 sidecar storages，不改现有 graph storage。

### 5.1 新增 namespace

文件：[`lightrag/namespace.py`](/mnt/data_nvme/code/LightRAG/lightrag/namespace.py)

新增：

```python
KV_STORE_ENTITY_PROFILES = "entity_profiles"
VECTOR_STORE_ENTITY_PROFILE = "entity_profile_vectors"
```

注意：

- KV 和 Vector 的 namespace 值不要相同
- `postgres_impl.py` 里有 `NAMESPACE_TABLE_MAP`，如果值相同会冲突

### 5.2 `self.entity_profiles`

类型：

- `BaseKVStorage`

建议 key：

- `entity_name`

value 结构：

```json
{
  "entity_name": "Alice",
  "entity_type": "person",
  "base_description": "...",
  "source_ids": ["chunk-1", "chunk-7"],
  "source_id": "chunk-1<SEP>chunk-7",
  "file_path": "inputs/demo.txt",
  "facet_schema_id": "entity_general_v1",
  "facet_schema_version": 1,
  "default_facet_id": "identity_definition",
  "facet_catalog": [
    {
      "facet_id": "identity_definition",
      "facet_name": "Identity / Definition",
      "definition": "What the entity is, its category, boundary, and essential identity.",
      "include": ["definition", "category", "scope"],
      "exclude": ["dynamic behavior", "causal interaction"]
    }
  ],
  "facet_ids": [
    "identity_definition",
    "attributes_composition",
    "role_function",
    "state_behavior"
  ],
  "profile_ids": ["epf-xxx", "epf-yyy", "epf-zzz", "epf-www"],
  "profiles": [
    {
      "profile_id": "epf-xxx",
      "facet_id": "identity_definition",
      "facet_name": "Identity / Definition",
      "facet_definition": "What the entity is, its category, boundary, and essential identity.",
      "profile_text": "...",
      "support_chunk_ids": ["chunk-1", "chunk-7"],
      "support_fragment_ids": [],
      "grounding_status": "chunk_level",
      "created_at": 1710000000
    }
  ],
  "count": 4
}
```

### 5.3 `self.entity_profiles_vdb`

类型：

- `BaseVectorStorage`

建议向量条目 id：

- `compute_mdhash_id(f"{entity_name}:{facet_id}", prefix="epf-")`

value 结构：

```json
{
  "profile_id": "epf-xxx",
  "entity_name": "Alice",
  "entity_type": "person",
  "facet_id": "identity_definition",
  "facet_name": "Identity / Definition",
  "source_id": "chunk-1<SEP>chunk-7",
  "file_path": "inputs/demo.txt",
  "content": "Alice\nperson\nidentity_definition\nIdentity / Definition\nWhat the entity is, its category, boundary, and essential identity.\n..."
}
```

`content` 必须是可嵌入文本，建议统一格式：

```text
{entity_name}
{entity_type}
{facet_id}
{facet_name}
{facet_definition}
{profile_text}
```

这样做的作用是：

- facet 名称歧义由 `facet_definition` 进一步消解
- retrieval 空间不只包含 profile 内容，也包含 facet 语义边界

---

## 6. V1 新增配置项

### 6.1 全局配置，放在 `LightRAG` dataclass

文件：[`lightrag/lightrag.py`](/mnt/data_nvme/code/LightRAG/lightrag/lightrag.py)

建议新增字段，插在“Entity extraction”与“LLM/summary config”之间，保持语义接近：

```python
enable_entity_profiles: bool = field(
    default=get_env_value("ENABLE_ENTITY_PROFILES", False, bool)
)

entity_profile_schema_id: str = field(
    default=get_env_value(
        "ENTITY_PROFILE_SCHEMA_ID",
        DEFAULT_ENTITY_PROFILE_SCHEMA_ID,
        str,
    )
)

entity_profile_facets: list[dict[str, Any]] = field(
    default_factory=lambda: get_env_value(
        "ENTITY_PROFILE_FACETS",
        DEFAULT_ENTITY_PROFILE_FACETS,
        list,
    )
)

entity_profile_schema_version: int = field(
    default=get_env_value(
        "ENTITY_PROFILE_SCHEMA_VERSION",
        DEFAULT_ENTITY_PROFILE_SCHEMA_VERSION,
        int,
    )
)

entity_profile_default_facet_id: str = field(
    default=get_env_value(
        "ENTITY_PROFILE_DEFAULT_FACET_ID",
        "identity_definition",
        str,
    )
)
```

建议新增常量，放在 [`lightrag/constants.py`](/mnt/data_nvme/code/LightRAG/lightrag/constants.py)：

```python
DEFAULT_ENTITY_PROFILE_SCHEMA_ID = "entity_general_v1"
DEFAULT_ENTITY_PROFILE_FACETS = [
    {
        "facet_id": "identity_definition",
        "facet_name": "Identity / Definition",
        "definition": "What the entity is, its category, boundary, and essential identity.",
        "include": ["definition", "category", "scope", "canonical identity"],
        "exclude": ["dynamic behavior", "causal process", "governance issues"],
    },
    {
        "facet_id": "attributes_composition",
        "facet_name": "Attributes / Composition",
        "definition": "What stable attributes, properties, parts, or components characterize the entity.",
        "include": ["properties", "components", "features", "specifications"],
        "exclude": ["responsibilities", "mechanisms", "temporal states"],
    },
    {
        "facet_id": "role_function",
        "facet_name": "Role / Function",
        "definition": "What role the entity plays, what function it serves, or what purpose it fulfills.",
        "include": ["responsibility", "purpose", "function", "use"],
        "exclude": ["identity definition", "internal mechanism", "momentary state"],
    },
    {
        "facet_id": "state_behavior",
        "facet_name": "State / Behavior",
        "definition": "How the entity behaves, changes, operates, or appears dynamically over time or in context.",
        "include": ["behavior", "operation", "state", "change"],
        "exclude": ["static category", "stable composition", "governance policy"],
    },
]
DEFAULT_ENTITY_PROFILE_SCHEMA_VERSION = 1
DEFAULT_ENTITY_PROFILE_TOP_K = 24
DEFAULT_ENTITY_PROFILE_MAX_PER_ENTITY = 2
```

### 6.2 查询参数，放在 `QueryParam`

文件：[`lightrag/base.py`](/mnt/data_nvme/code/LightRAG/lightrag/base.py)

新增字段：

```python
enable_entity_profiles: bool = False
entity_profile_top_k: int = int(
    os.getenv("ENTITY_PROFILE_TOP_K", str(DEFAULT_ENTITY_PROFILE_TOP_K))
)
entity_profile_max_per_entity: int = int(
    os.getenv(
        "ENTITY_PROFILE_MAX_PER_ENTITY",
        str(DEFAULT_ENTITY_PROFILE_MAX_PER_ENTITY),
    )
)
```

解释：

- `enable_entity_profiles`：查询级开关，默认关闭，方便做 ablation
- `entity_profile_top_k`：在 profile VDB 中初步召回多少条 profile
- `entity_profile_max_per_entity`：对单个实体最多保留几个 profile 用于组合

---

## 7. 文件级改造清单

这一节是 V1 的主清单。

## 7.1 [`lightrag/namespace.py`](/mnt/data_nvme/code/LightRAG/lightrag/namespace.py)

### 新增

- `NameSpace.KV_STORE_ENTITY_PROFILES`
- `NameSpace.VECTOR_STORE_ENTITY_PROFILE`

### 不改

- `is_namespace()`

---

## 7.2 [`lightrag/constants.py`](/mnt/data_nvme/code/LightRAG/lightrag/constants.py)

### 新增常量

- `DEFAULT_ENTITY_PROFILE_SCHEMA_ID`
- `DEFAULT_ENTITY_PROFILE_FACETS`
- `DEFAULT_ENTITY_PROFILE_SCHEMA_VERSION`
- `DEFAULT_ENTITY_PROFILE_TOP_K`
- `DEFAULT_ENTITY_PROFILE_MAX_PER_ENTITY`

### 不建议新增

- 不要在 V1 再加一组 profile 专属 token budget
- 先复用现有 `summary_context_size / summary_max_tokens / max_entity_tokens`

---

## 7.3 [`lightrag/base.py`](/mnt/data_nvme/code/LightRAG/lightrag/base.py)

### 新增 schema

- `EntityFacetSchemaItem`
- `EntityProfileSchema`
- `EntityProfilesRecordSchema`

### 扩展 `QueryParam`

新增字段：

- `enable_entity_profiles`
- `entity_profile_top_k`
- `entity_profile_max_per_entity`

### 不新增存储抽象类

原因：

- 现有 `BaseKVStorage` 与 `BaseVectorStorage` 已足够
- V1 不需要新的 storage interface

---

## 7.4 [`lightrag/lightrag.py`](/mnt/data_nvme/code/LightRAG/lightrag/lightrag.py)

这是 storage wiring 的主入口。

### 新增属性

在现有 storage 初始化区域新增：

```python
self.entity_profiles: BaseKVStorage = self.key_string_value_json_storage_cls(
    namespace=NameSpace.KV_STORE_ENTITY_PROFILES,
    workspace=self.workspace,
    embedding_func=self.embedding_func,
)

self.entity_profiles_vdb: BaseVectorStorage = self.vector_db_storage_cls(
    namespace=NameSpace.VECTOR_STORE_ENTITY_PROFILE,
    workspace=self.workspace,
    embedding_func=self.embedding_func,
    meta_fields={
        "profile_id",
        "entity_name",
        "entity_type",
        "facet_id",
        "facet_name",
        "source_id",
        "content",
        "file_path",
    },
)
```

### 需要同步修改的位置

1. `initialize_storages()`
2. `finalize_storages()`
3. `_insert_done()`

都要把以下两项加进去：

- `self.entity_profiles`
- `self.entity_profiles_vdb`

### 需要扩展的配置校验

在现有配置校验逻辑中新增：

- 如果 `enable_entity_profiles=True` 且 `entity_profile_facets` 为空，直接报错
- 校验每个 facet 都包含 `facet_id / facet_name / definition`
- `facet_id` 必须唯一，按配置顺序保存
- `entity_profile_default_facet_id` 必须出现在 `entity_profile_facets` 中

### 暂不修改

- `ainsert_custom_kg()`
- 删除、重建、迁移逻辑

这些留到第二阶段。

---

## 7.5 [`lightrag/prompt.py`](/mnt/data_nvme/code/LightRAG/lightrag/prompt.py)

V1 只新增一个离线 profile 生成 prompt，不改查询回答 prompt。

### 新增 prompt

#### `PROMPTS["entity_profile_generation_system_prompt"]`

职责：

- 给定 `entity_name / entity_type / merged description fragments / facet_catalog`
- 输出多条 `profile` 记录

推荐输出格式：

```text
profile{tuple_delimiter}facet_id{tuple_delimiter}facet_name{tuple_delimiter}profile_text
...
{completion_delimiter}
```

#### `PROMPTS["entity_profile_generation_user_prompt"]`

输入内容：

- `entity_name`
- `entity_type`
- `facet_catalog`
- `description_list`
- `base_description`

### 推荐输出约束

- 每个 `facet_id` 只输出一条
- 不新增未声明的 facet
- `facet_name` 只允许回写本地 schema 中已有名称
- 不重复 base description 原文
- `profile_text` 应面向该 facet，尽量突出与其它 facet 的区分性

### facet schema 在 prompt 中的呈现方式

prompt 中不应只给一个 facet 名字列表，而应给结构化 facet catalog：

```text
- facet_id: identity_definition
  facet_name: Identity / Definition
  definition: explain what the entity is, its category, scope, and essential identity; do not describe dynamic behavior
  include: definition, category, scope
  exclude: behavior, mechanism, governance
```

说明：

- `definition` 必须作为 prompt 输入的一部分，用来消除 facet 歧义
- `definition` 不应作为模型输出的真值来源，本地 schema 才是真值
- parser 以 `facet_id` 为准，`facet_name` 只作为冗余校验和可读性增强

---

## 7.6 [`lightrag/operate.py`](/mnt/data_nvme/code/LightRAG/lightrag/operate.py)

这是 V1 的主要实现文件。

### 7.6.1 新增函数一：`_parse_entity_profile_generation_result`

建议签名：

```python
def _parse_entity_profile_generation_result(
    result: str,
    entity_name: str,
    facet_catalog: list[dict[str, Any]],
    created_at: int,
    tuple_delimiter: str = "<|#|>",
    completion_delimiter: str = "<|COMPLETE|>",
) -> list[dict]:
```

职责：

- 解析 profile 生成 prompt 的输出
- 校验每条记录是否是 `profile`
- 提取 `facet_id / facet_name / profile_text`
- 用本地 `facet_catalog` 校验 facet 是否存在
- 返回标准化 profile 列表

返回的单条 profile dict 建议为：

```python
{
    "profile_id": "...",
    "facet_id": "identity_definition",
    "facet_name": "Identity / Definition",
    "facet_definition": "What the entity is, its category, boundary, and essential identity.",
    "profile_text": "...",
    "support_chunk_ids": ["chunk-1", "chunk-7"],
    "support_fragment_ids": [],
    "grounding_status": "chunk_level",
    "created_at": 1710000000,
}
```

### 7.6.2 新增函数二：`_generate_entity_profiles`

建议签名：

```python
async def _generate_entity_profiles(
    entity_name: str,
    entity_type: str,
    facet_catalog: list[dict[str, Any]],
    description_list: list[str],
    base_description: str,
    source_ids: list[str],
    file_path: str,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> list[dict]:
```

职责：

- 为单个实体生成多个离线 profile
- 生成过程严格受 `facet_catalog` 约束，不允许模型发明新 facet
- 复用现有 `summary_context_size` 的 token 截断思路
- 使用 `llm_response_cache`
- 对缺失 facet 做 fallback 补全

fallback 规则建议：

1. 若 LLM 一个 facet 都没产出：
   - 为每个配置 facet 生成 fallback profile，`profile_text = base_description`
2. 若只缺部分 facet：
   - 缺的 facet 用 `base_description` 补

fallback profile 的字段建议：

- `support_chunk_ids = source_ids`
- `support_fragment_ids = []`
- `grounding_status = "fallback"`

这样可以保证 profile schema 稳定，不会让查询阶段出现空集异常。

### 7.6.3 新增函数三：`_upsert_entity_profiles`

建议签名：

```python
async def _upsert_entity_profiles(
    entity_name: str,
    entity_type: str,
    facet_catalog: list[dict[str, Any]],
    description_list: list[str],
    base_description: str,
    full_source_ids: list[str],
    source_id: str,
    file_path: str,
    entity_profiles_storage: BaseKVStorage,
    entity_profiles_vdb: BaseVectorStorage,
    global_config: dict,
    llm_response_cache: BaseKVStorage | None = None,
) -> dict:
```

职责：

- 调用 `_generate_entity_profiles()`
- 组装 `EntityProfilesRecordSchema`
- 上写 `entity_profiles`
- 上写 `entity_profiles_vdb`
- 删除 stale profile vector ids

删除 stale 向量建议逻辑：

1. 从 `entity_profiles_storage.get_by_id(entity_name)` 读取旧 `profile_ids`
2. 新 profile 生成后得到新 `profile_ids`
3. `old - new` 的部分调用 `entity_profiles_vdb.delete()`

### 7.6.4 新增函数四：`_select_profiles_for_entities`

建议签名：

```python
async def _select_profiles_for_entities(
    query: str,
    node_datas: list[dict],
    entity_profiles_storage: BaseKVStorage | None,
    entity_profiles_vdb: BaseVectorStorage | None,
    query_param: QueryParam,
    query_embedding=None,
) -> list[dict]:
```

职责：

- 在已召回的 `node_datas` 内部做第二阶段 profile selection
- 用 `entity_profiles_vdb.query()` 召回 profile
- 只保留属于 `node_datas` 中实体的结果
- 按 `entity_name` 分组
- 对每个实体最多保留 `entity_profile_max_per_entity`
- 在同一实体内部，优先保持 facet 多样性，不要重复选到语义近似的同 facet profile
- 把选中的 profile 写回 node data

返回时对每个 entity 增加字段：

```python
{
    "base_description": "...",
    "selected_profile_ids": ["epf-1", "epf-2"],
    "selected_facet_ids": ["identity_definition", "role_function"],
    "selected_profiles": [
        {
            "profile_id": "epf-1",
            "facet_id": "identity_definition",
            "facet_name": "Identity / Definition",
            "facet_definition": "What the entity is, its category, boundary, and essential identity.",
            "profile_text": "...",
            "support_chunk_ids": ["chunk-1"],
            "support_fragment_ids": [],
            "grounding_status": "chunk_level",
        }
    ],
    "description": "<组合后的条件化描述>"
}
```

### 7.6.5 新增函数五：`_compose_entity_profile_description`

建议签名：

```python
def _compose_entity_profile_description(
    selected_profiles: list[dict],
    base_description: str,
) -> str:
```

职责：

- 把 query 下选中的 1 到 2 个 profiles 组合成送给下游 context 的字符串
- 组合时按本地 facet schema 顺序输出，而不是按模型生成顺序输出

建议输出格式：

```text
[identity_definition] ...
[role_function] ...
```

如果没有 `selected_profiles`，直接返回 `base_description`。

### 7.6.6 修改点一：`_merge_nodes_then_upsert`

这是 V1 的离线构建入口。

现有逻辑：

1. merge source ids
2. deduplicate descriptions
3. 生成 `description`
4. upsert graph node
5. upsert `entities_vdb`

V1 增量改法：

在 graph node 和 `entities_vdb` upsert 完成后，新增：

```python
if (
    global_config.get("enable_entity_profiles")
    and entity_profiles_storage is not None
    and entity_profiles_vdb is not None
):
    await _upsert_entity_profiles(...)
```

同时需要给 `_merge_nodes_then_upsert()` 增加两个新参数：

```python
entity_profiles_storage: BaseKVStorage | None = None
entity_profiles_vdb: BaseVectorStorage | None = None
```

### 7.6.7 修改点二：`merge_nodes_and_edges`

需要把新增 storage 参数继续透传给 `_merge_nodes_then_upsert()`。

新增参数：

```python
entity_profiles_storage: BaseKVStorage | None = None
entity_profiles_vdb: BaseVectorStorage | None = None
```

只在 entity phase 传递，不影响 relation phase。

### 7.6.8 修改点三：`_get_node_data`

这是 V1 的在线选择入口。

新增参数：

```python
entity_profiles_storage: BaseKVStorage | None = None,
entity_profiles_vdb: BaseVectorStorage | None = None,
apply_profiles: bool = False,
```

现有逻辑：

1. `entities_vdb.query()`
2. batch get nodes
3. 找 related edges
4. 返回 `node_datas, use_relations`

V1 增量改法：

在 `node_datas` 组装完成后、调用 `_find_most_related_edges_from_entities()` 之前插入：

```python
if apply_profiles and query_param.enable_entity_profiles:
    node_datas = await _select_profiles_for_entities(
        query,
        node_datas,
        entity_profiles_storage,
        entity_profiles_vdb,
        query_param,
        query_embedding=query_embedding,
    )
```

### 7.6.9 修改点四：`_perform_kg_search`

这里只需要控制“什么时候启用 profile 选择”。

V1 规则：

- 仅当 `query_param.mode == "local"` 时启用
- `hybrid / mix` 暂不启用

因此只改 local 分支：

```python
local_entities, local_relations = await _get_node_data(
    ll_keywords,
    knowledge_graph_inst,
    entities_vdb,
    query_param,
    query_embedding=ll_embedding,
    entity_profiles_storage=entity_profiles_storage,
    entity_profiles_vdb=entity_profiles_vdb,
    apply_profiles=True,
)
```

而 `hybrid / mix` 分支仍传 `apply_profiles=False`。

### 7.6.10 修改点五：`kg_query` 和 `_build_query_context`

这两处只需要把新增 storage 参数继续向下透传。

需要新增参数：

```python
entity_profiles_storage: BaseKVStorage | None = None
entity_profiles_vdb: BaseVectorStorage | None = None
```

V1 不需要重写 `_build_context_str()`，因为 profile selection 最终仍然产出的是 `description` 字符串。

---

## 7.7 [`lightrag/utils.py`](/mnt/data_nvme/code/LightRAG/lightrag/utils.py)

V1 不改 chunk 选择逻辑，但建议增强 `convert_to_user_format()`。

### 修改 `convert_to_user_format`

对 entity 输出追加以下可选字段：

```python
"base_description": original_entity.get("base_description", ""),
"selected_profile_ids": original_entity.get("selected_profile_ids", []),
"selected_facet_ids": original_entity.get("selected_facet_ids", []),
"selected_profiles": original_entity.get("selected_profiles", []),
```

这样做的好处：

- API 返回可观察
- WebUI / 调试 / 论文分析更方便
- 不破坏现有字段

---

## 7.8 [`lightrag/kg/postgres_impl.py`](/mnt/data_nvme/code/LightRAG/lightrag/kg/postgres_impl.py)

如果使用 PostgreSQL，这个文件是必须改的。

### 必改点

#### `NAMESPACE_TABLE_MAP`

新增：

```python
NameSpace.KV_STORE_ENTITY_PROFILES: "LIGHTRAG_ENTITY_PROFILES",
NameSpace.VECTOR_STORE_ENTITY_PROFILE: "LIGHTRAG_VDB_ENTITY_PROFILE",
```

#### namespace 分支处理

在 KV 相关的 `get_by_id / get_by_ids / upsert` 分支中，加入：

- `NameSpace.KV_STORE_ENTITY_PROFILES`

在 vector 相关的 namespace 判断中，加入：

- `NameSpace.VECTOR_STORE_ENTITY_PROFILE`

### 说明

如果你第一版只跑默认 `JsonKVStorage + NanoVectorDBStorage + NetworkXStorage`，理论上可以先不实现 PostgreSQL 适配。
但设计文件里必须把它列为“受影响文件”，否则后面切换存储后端会直接坏。

---

## 7.9 测试文件

建议第一版至少新增 2 个测试：

### [`tests/test_entity_profile_generation.py`](/mnt/data_nvme/code/LightRAG/tests/test_entity_profile_generation.py)

覆盖：

- 单实体 merge 后能生成 profile record
- `entity_profiles` 中 `count/profile_ids/profiles` 正确
- `entity_profiles_vdb` 中 profile 数量与 facet 数一致
- 每条 profile 都带 `facet_id / support_chunk_ids / grounding_status`

### [`tests/test_entity_profile_local_query.py`](/mnt/data_nvme/code/LightRAG/tests/test_entity_profile_local_query.py)

覆盖：

- `QueryParam(enable_entity_profiles=False)` 时行为与原版一致
- `QueryParam(enable_entity_profiles=True)` 时返回 entity 含 `selected_profiles`
- local 模式下 `description` 被条件化改写
- profile 缺失时 fallback 到 `base_description`
- `selected_profiles` 只会落在预定义 facet schema 内

---

## 8. 第一版具体实现顺序

## Phase 1: 只把数据存起来

先完成：

1. `namespace.py`
2. `constants.py`
3. `base.py`
4. `lightrag.py` storage wiring
5. `prompt.py`
6. `operate.py` 中 `_generate_entity_profiles / _upsert_entity_profiles`

完成标准：

- 插入文档后，`entity_profiles` 与 `entity_profiles_vdb` 有数据
- 查询逻辑还没切换也没关系

## Phase 2: 接到 local 查询链

再完成：

1. `_select_profiles_for_entities`
2. `_compose_entity_profile_description`
3. `_get_node_data`
4. `_perform_kg_search`
5. `kg_query` / `_build_query_context` 参数透传
6. `utils.convert_to_user_format`

完成标准：

- local 模式下能看到 `selected_profiles`
- entity `description` 已变成按 facet schema 约束生成的条件化 profile 组合

## Phase 3: 验证与 ablation

完成：

1. 两个单测
2. 用同一批 query 跑 `enable_entity_profiles=False/True`
3. 对比答案相关性与 raw_data 中的 selected profiles

---

## 9. 第一版最小接口改动汇总

### 新增 storage 属性

- `LightRAG.entity_profiles`
- `LightRAG.entity_profiles_vdb`

### 新增 `QueryParam`

- `enable_entity_profiles`
- `entity_profile_top_k`
- `entity_profile_max_per_entity`

### 新增函数

- `_parse_entity_profile_generation_result`
- `_generate_entity_profiles`
- `_upsert_entity_profiles`
- `_select_profiles_for_entities`
- `_compose_entity_profile_description`

### 需要扩展参数的函数

- `_merge_nodes_then_upsert`
- `merge_nodes_and_edges`
- `_get_node_data`
- `_perform_kg_search`
- `kg_query`
- `_build_query_context`

---

## 10. 第一版不建议做的事情

- 不要一开始就改关系侧
- 不要一开始就改 `hybrid / mix`
- 不要把 graph node 的 `description` 删除
- 不要把 profile 直接塞进 graph storage
- 不要为了 V1 直接实现 fragment-level evidence span 抽取
- 不要为了 V1 直接做 facet schema 的在线自适应生成

原因：

- 这些都不是主创新最小闭环必需项
- 它们会显著增加联调与验证成本
- 会把主创新与副创新混在一起，影响后续 ablation

---

## 11. V1 实现完成后的判定标准

满足以下条件即可视为“主创新第一版跑通”：

1. 插入文档后，每个实体除了 graph node / entities_vdb 外，还会拥有 `entity_profiles` 记录和多条 `entity_profiles_vdb` 向量记录
2. `local` 模式查询时，系统仍先召回实体，再在实体内部选 profile
3. `selected_profiles` 能反映 query 条件化选择结果，且 facet 落在预定义 schema 内
4. 每条 profile 已预留 `support_chunk_ids / support_fragment_ids / grounding_status`，可承接副创新 1
5. 下游上下文构建仍沿用现有通路，不需要重写 `chunks` 与 `relations` 部分
6. 关闭 `enable_entity_profiles` 后，系统行为回退到原版

---

## 12. 这份方案对应的最小代码切入口

如果你要立刻开始写代码，优先顺序就是：

1. [`lightrag/namespace.py`](/mnt/data_nvme/code/LightRAG/lightrag/namespace.py)
2. [`lightrag/constants.py`](/mnt/data_nvme/code/LightRAG/lightrag/constants.py)
3. [`lightrag/base.py`](/mnt/data_nvme/code/LightRAG/lightrag/base.py)
4. [`lightrag/lightrag.py`](/mnt/data_nvme/code/LightRAG/lightrag/lightrag.py)
5. [`lightrag/prompt.py`](/mnt/data_nvme/code/LightRAG/lightrag/prompt.py)
6. [`lightrag/operate.py`](/mnt/data_nvme/code/LightRAG/lightrag/operate.py)
7. [`lightrag/utils.py`](/mnt/data_nvme/code/LightRAG/lightrag/utils.py)

第一步先做到“离线 profile 已经入库”，第二步再接查询链，不要反过来。
