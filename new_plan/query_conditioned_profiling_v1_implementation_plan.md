# Query-Conditioned Profiling V1 代码改造实施计划

## 摘要

本计划严格遵循 [`new_plan/query_conditioned_profiling_v1_design.md`](/mnt/data_nvme/code/LightRAG/new_plan/query_conditioned_profiling_v1_design.md) 的阶段划分，只分为 `Phase 1: 离线 profile 入库`、`Phase 2: 接入 local 查询链`、`Phase 3: 验证与 ablation`。
执行目标是实现实体侧、`local` 模式、固定 facet schema 约束下的 Query-Conditioned Profiling V1，且保持现有 `entity -> graph node + entities_vdb` 主链不被推翻，`description` 继续作为 fallback。

## 关键接口与约束

- 新增存储属性：
  - `LightRAG.entity_profiles`
  - `LightRAG.entity_profiles_vdb`
- 新增 `QueryParam` 字段：
  - `enable_entity_profiles`
  - `entity_profile_top_k`
  - `entity_profile_max_per_entity`
- 新增函数：
  - `_parse_entity_profile_generation_result`
  - `_generate_entity_profiles`
  - `_upsert_entity_profiles`
  - `_select_profiles_for_entities`
  - `_compose_entity_profile_description`
- 需要扩展参数透传的函数：
  - `_merge_nodes_then_upsert`
  - `merge_nodes_and_edges`
  - `_get_node_data`
  - `_perform_kg_search`
  - `kg_query`
  - `_build_query_context`
- 严格边界：
  - 只做实体侧，不做关系侧
  - 只接 `local`，不接 `global / hybrid / mix`
  - 不删除 graph node 上现有 `description`
  - 不把 profile 塞进 graph storage
  - 只预留 `support_chunk_ids / support_fragment_ids / grounding_status`，不实现 fragment-level grounding

## 分阶段实施

### Phase 1：只把数据存起来

#### 任务包 1：定义 namespace、常量和 schema

- 在 `namespace.py` 增加：
  - `KV_STORE_ENTITY_PROFILES`
  - `VECTOR_STORE_ENTITY_PROFILE`
- 在 `constants.py` 增加：
  - `DEFAULT_ENTITY_PROFILE_SCHEMA_ID`
  - `DEFAULT_ENTITY_PROFILE_FACETS`
  - `DEFAULT_ENTITY_PROFILE_SCHEMA_VERSION`
  - `DEFAULT_ENTITY_PROFILE_TOP_K`
  - `DEFAULT_ENTITY_PROFILE_MAX_PER_ENTITY`
- 在 `base.py` 紧跟 `TextChunkSchema` 后新增：
  - `EntityFacetSchemaItem`
  - `EntityProfileSchema`
  - `EntityProfilesRecordSchema`
- 扩展 `QueryParam`，但此阶段只定义字段，不接查询逻辑。

验收标准：

- 新增类型和常量可被其他模块导入。
- schema 字段名与设计文档完全一致。

#### 任务包 2：接 storage wiring 和配置校验

- 在 `lightrag.py` 初始化：
  - `self.entity_profiles: BaseKVStorage`
  - `self.entity_profiles_vdb: BaseVectorStorage`
- 在 `initialize_storages()`、`finalize_storages()`、`_insert_done()` 中纳入新存储。
- 加入配置校验：
  - `enable_entity_profiles=True` 时 `entity_profile_facets` 不能为空
  - 每个 facet 必须有 `facet_id / facet_name / definition`
  - `facet_id` 唯一
  - `entity_profile_default_facet_id` 必须存在于 facet schema 中

验收标准：

- 开启开关时系统能正常初始化新存储。
- 错误配置会在启动阶段失败，而不是运行中失败。

#### 任务包 3：实现离线 profile 生成 prompt 与解析

- 在 `prompt.py` 新增：
  - `entity_profile_generation_system_prompt`
  - `entity_profile_generation_user_prompt`
- 实现 `_parse_entity_profile_generation_result`：
  - 只接受 `profile` 类型记录
  - 以本地 `facet_catalog` 校验 `facet_id`
  - 回填标准化 profile 结构
- 解析结果必须附带：
  - `profile_id`
  - `facet_definition`
  - `support_chunk_ids`
  - `support_fragment_ids`
  - `grounding_status`
  - `created_at`

验收标准：

- 模型输出无法引入未声明 facet。
- parser 输出结构可直接写入 KV/VDB。

#### 任务包 4：实现 profile 生成与落库主链

- 在 `operate.py` 实现：
  - `_generate_entity_profiles`
  - `_upsert_entity_profiles`
- `_generate_entity_profiles` 负责：
  - 基于 `description_list + base_description + facet_catalog` 生成 facet-specific profiles
  - 使用 `llm_response_cache`
  - 对缺失 facet 做 fallback 补全
- `_upsert_entity_profiles` 负责：
  - 组装 `EntityProfilesRecordSchema`
  - 写 `entity_profiles`
  - 写 `entity_profiles_vdb`
  - 删除 stale profile vector ids
- 修改 `_merge_nodes_then_upsert`：
  - 在 graph node 和 `entities_vdb` upsert 完成后，按开关调用 `_upsert_entity_profiles`
- 修改 `merge_nodes_and_edges`：
  - 透传新 storage 参数

验收标准：

- 插入文档后，每个实体除了原有 graph node / entities_vdb 外，还会生成 `entity_profiles` 与 `entity_profiles_vdb` 数据。
- fallback 行为稳定，不会因 LLM 漏 facet 导致空 profile 集合。

#### 任务包 5：补齐 PostgreSQL 后端兼容

- 在 `lightrag/kg/postgres_impl.py` 的 `NAMESPACE_TABLE_MAP` 增加：
  - `LIGHTRAG_ENTITY_PROFILES`
  - `LIGHTRAG_VDB_ENTITY_PROFILE`
- 将新 namespace 接入 KV 和 vector 相关分支逻辑。

验收标准：

- PostgreSQL 后端不会因新增 namespace 报未知表或未知 namespace 错误。
- 默认 JSON/NanoVectorDB 路径和 PostgreSQL 路径都能持有同一组 profile 数据语义。

### Phase 2：接到 local 查询链

#### 任务包 6：实现在线 profile 选择与组合

- 在 `operate.py` 实现：
  - `_select_profiles_for_entities`
  - `_compose_entity_profile_description`
- `_select_profiles_for_entities` 负责：
  - 在已召回 `node_datas` 内部做第二阶段选择
  - 使用 `entity_profiles_vdb.query()`
  - 仅保留属于已召回实体的 profile
  - 对每个实体最多保留 `entity_profile_max_per_entity`
  - 保持同一实体内部的 facet 多样性
- `_compose_entity_profile_description` 负责：
  - 按本地 facet schema 顺序组合 `selected_profiles`
  - 无选中 profile 时回退 `base_description`

验收标准：

- 查询链不会直接替换实体召回，只会在召回结果内做二次 profile 选择。
- `description` 组合结果稳定、可读、可回退。

#### 任务包 7：把 profile 选择接入 local 查询路径

- 修改 `_get_node_data`：
  - 增加 `entity_profiles_storage`、`entity_profiles_vdb`、`apply_profiles`
  - 在组装 `node_datas` 后、关系查找前执行 profile 选择
- 修改 `_perform_kg_search`：
  - 仅在 `local` 分支传 `apply_profiles=True`
  - `hybrid / mix` 保持 `apply_profiles=False`
- 修改 `kg_query` 和 `_build_query_context`：
  - 继续透传新 storage 参数
- 不改 `_build_context_str()` 的主体结构，只让上游 `description` 被条件化替换。

验收标准：

- `local` 模式下 entity 的 `description` 变为 facet-schema-constrained 的条件化描述。
- 非 `local` 模式行为与原版保持一致。

#### 任务包 8：暴露调试与 API 可观察性

- 在 `utils.py` 的 `convert_to_user_format()` 增加可选返回字段：
  - `base_description`
  - `selected_profile_ids`
  - `selected_facet_ids`
  - `selected_profiles`

验收标准：

- API / raw_data 可直接观察 profile 选择结果。
- 不破坏现有字段结构和老调用方兼容性。

### Phase 3：验证与 ablation

#### 任务包 9：补齐单测

- 新增 `tests/test_entity_profile_generation.py`
  - 覆盖 profile record 生成
  - 覆盖 KV/VDB 数量一致性
  - 覆盖 `facet_id / support_chunk_ids / grounding_status`
- 新增 `tests/test_entity_profile_local_query.py`
  - 覆盖 `enable_entity_profiles=False` 回退原行为
  - 覆盖 `enable_entity_profiles=True` 时存在 `selected_profiles`
  - 覆盖 `local` 模式 description 被条件化改写
  - 覆盖 profile 缺失 fallback 到 `base_description`
  - 覆盖 facet 只落在预定义 schema 内

验收标准：

- 新测试能覆盖离线入库链和 local 查询链两个闭环。
- 开关关闭时行为与原版一致。

#### 任务包 10：执行 ablation 验证

- 使用同一批 query 运行两组实验：
  - `enable_entity_profiles=False`
  - `enable_entity_profiles=True`
- 对比：
  - 答案相关性
  - raw_data 中 `selected_profiles`
  - `selected_facet_ids` 是否符合 query 条件化预期
- 记录至少以下结论：
  - profile 是否真实参与了 local context 构建
  - fallback 是否只在缺 profile 或空选择时触发
  - 关闭开关后是否完整回退原版

验收标准：

- 满足设计文档第 11 节的 6 条“V1 跑通”判定标准。
- 可以支撑后续 `+ Evidence-Grounded Composition` 的消融基线。

## 测试与检查清单

- 单元测试：
  - profile parser
  - profile 生成 fallback
  - KV/VDB 写入一致性
  - local query profile 选择
- 集成检查：
  - 文档插入后 `entity_profiles` 与 `entity_profiles_vdb` 均有数据
  - `local` 下能看到 `selected_profiles`
  - `global / hybrid / mix` 不受影响
- 回归检查：
  - 关闭 `enable_entity_profiles` 时行为退回原版
  - `description` 仍保留为稳定 fallback
  - PostgreSQL namespace 兼容不报错

## 默认假设

- 计划文档输出粒度采用“任务包级”，每个阶段拆成可直接分配和验收的任务。
- 设计文档中的默认 4-facet schema、字段名、fallback 规则、函数命名和阶段顺序全部视为冻结要求，不在实施计划中重新设计。
- 计划产物建议保存为：
  - `/mnt/data_nvme/code/LightRAG/new_plan/query_conditioned_profiling_v1_implementation_plan.md`
- 当前计划不包含实际写文件动作；若切回执行模式，按上述路径落盘即可。
