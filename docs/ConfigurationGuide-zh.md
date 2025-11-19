# LightRAG 配置指南

## 概述

LightRAG 采用 **Schema-Driven Configuration Pattern**（架构驱动配置模式），通过单一数据源管理所有配置，自动生成本地配置文件和环境变量。

### 核心设计原则

**单一数据源 (Single Source of Truth)**:
- `config/config.schema.yaml` - 配置元数据（Git 追踪）
- `config/local.yaml` - 本地配置（自动生成，Git 忽略）
- `.env` - 环境变量（自动生成，Git 忽略）

**自动化工作流**:
```bash
config.schema.yaml → config/local.yaml → .env
```

**关键特性**:
- ✅ 深度合并 - 保留现有值
- ✅ 自动生成密钥 - 无需手动管理
- ✅ 类型推断 - 自动转换数据类型
- ✅ 安全性 - 配置文件不会提交到 Git

---

## 快速开始

### 1. 初始化配置

```bash
cd /path/to/LightRAG
./scripts/setup.sh
```

这会自动：
1. 读取 `config/config.schema.yaml`
2. 生成 `config/local.yaml`（包含自动生成的密钥）
3. 生成 `.env`（环境变量格式）

### 2. 检查生成的配置

```bash
# 查看本地配置
cat config/local.yaml

# 查看环境变量
cat .env
```

### 3. 修改配置（可选）

```bash
# 编辑本地配置
nano config/local.yaml

# 修改后重新生成 .env
./scripts/setup.sh
```

---

## 配置文件说明

### config.schema.yaml（配置元数据）

**位置**: `config/config.schema.yaml`

**用途**: 定义所有配置字段的元数据

**格式**:
```yaml
- section: trilingual.enabled
  default: true
  description: "Enable trilingual entity extractor (Chinese/English/Swedish)"

- section: lightrag.api.secret_key
  type: secret
  auto_generate: true
  description: "API secret key (auto-generated, 32 characters)"

- section: lightrag.llm.api_key
  type: secret
  auto_generate: false
  description: "LLM API key (user-provided)"
```

**字段说明**:
- `section`: 配置路径（点分隔，如 `trilingual.chinese.enabled`）
- `default`: 默认值（如果有）
- `type`: 类型标记（`secret` = 密钥，留空 = 自动推断）
- `auto_generate`: 是否自动生成密钥（仅适用于 `type: secret`）
- `description`: 字段描述

**重要**:
- ✅ 此文件会提交到 Git
- ✅ 修改此文件后运行 `./scripts/setup.sh` 更新配置
- ✅ 新增字段会使用默认值，现有值会保留

### config/local.yaml（本地配置）

**位置**: `config/local.yaml`

**用途**: 实际的配置文件（YAML 格式）

**格式**:
```yaml
trilingual:
  enabled: true
  default_language: en
  lazy_loading: true
  chinese:
    enabled: true
    model: CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH
  english:
    enabled: true
    model: en_core_web_trf
    batch_size: 32

lightrag:
  api:
    secret_key: abc123...  # 自动生成
    host: 0.0.0.0
    port: 9621
  llm:
    provider: openai
    model: gpt-4o-mini
    api_key: sk-...  # 手动填写
```

**重要**:
- ❌ 此文件不会提交到 Git（已添加到 `.gitignore`）
- ✅ 可以直接编辑此文件修改配置
- ✅ 修改后运行 `./scripts/setup.sh` 更新 `.env`
- ✅ 包含自动生成的密钥，请妥善保管

### .env（环境变量）

**位置**: `.env`（项目根目录）

**用途**: 环境变量格式的配置文件

**格式**:
```bash
# TRILINGUAL
TRILINGUAL_ENABLED=true
TRILINGUAL_DEFAULT_LANGUAGE=en
TRILINGUAL_LAZY_LOADING=true
TRILINGUAL_CHINESE_ENABLED=true
TRILINGUAL_CHINESE_MODEL=CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH

# LIGHTRAG
LIGHTRAG_API_SECRET_KEY=abc123...
LIGHTRAG_API_HOST=0.0.0.0
LIGHTRAG_API_PORT=9621
LIGHTRAG_LLM_PROVIDER=openai
LIGHTRAG_LLM_MODEL=gpt-4o-mini
```

**命名规则**:
- 嵌套路径 → 大写 + 下划线
- 例如: `trilingual.chinese.enabled` → `TRILINGUAL_CHINESE_ENABLED`

**重要**:
- ❌ 此文件不会提交到 Git（已添加到 `.gitignore`）
- ⚠️ 此文件由脚本自动生成，**不要手动编辑**
- ✅ 修改 `config/local.yaml` 后重新运行 `./scripts/setup.sh`

---

## 配置项说明

### 三语言实体提取器配置

#### 通用配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `trilingual.enabled` | `true` | 启用三语言实体提取器 |
| `trilingual.default_language` | `en` | 默认语言（zh/en/sv） |
| `trilingual.lazy_loading` | `true` | 启用延迟加载（节省内存） |

#### 中文配置（HanLP）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `trilingual.chinese.enabled` | `true` | 启用中文提取 |
| `trilingual.chinese.model` | `CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH` | HanLP 模型名 |
| `trilingual.chinese.cache_dir` | `""` | 模型缓存目录（空 = 默认 `~/.hanlp`） |

#### 英文配置（spaCy）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `trilingual.english.enabled` | `true` | 启用英文提取 |
| `trilingual.english.model` | `en_core_web_trf` | spaCy 模型名 |
| `trilingual.english.batch_size` | `32` | 批处理大小 |

**可选模型**:
- `en_core_web_trf`: Transformer 模型（最高质量，~440 MB）
- `en_core_web_lg`: 大模型（高质量，~440 MB）
- `en_core_web_sm`: 小模型（较低质量，~12 MB）

#### 瑞典语配置（spaCy）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `trilingual.swedish.enabled` | `true` | 启用瑞典语提取 |
| `trilingual.swedish.model` | `sv_core_news_lg` | spaCy 模型名 |
| `trilingual.swedish.batch_size` | `32` | 批处理大小 |

**可选模型**:
- `sv_core_news_lg`: 大模型（最高质量，~545 MB）
- `sv_core_news_md`: 中等模型（~40 MB）
- `sv_core_news_sm`: 小模型（~12 MB）

#### 性能配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `trilingual.performance.max_text_length` | `1000000` | 最大文本长度（字符） |
| `trilingual.performance.enable_gpu` | `false` | 启用 GPU 加速 |
| `trilingual.performance.num_threads` | `4` | 并行处理线程数 |

#### 缓存配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `trilingual.cache.enabled` | `true` | 启用结果缓存 |
| `trilingual.cache.ttl` | `3600` | 缓存 TTL（秒，0 = 永不过期） |
| `trilingual.cache.max_size` | `1000` | 最大缓存数量 |

#### 日志配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `trilingual.logging.level` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `trilingual.logging.format` | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | 日志格式 |

### LightRAG 通用配置

#### API 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `lightrag.api.secret_key` | *自动生成* | API 密钥（32 字符） |
| `lightrag.api.host` | `0.0.0.0` | API 服务器地址 |
| `lightrag.api.port` | `9621` | API 服务器端口 |
| `lightrag.api.debug` | `false` | 调试模式 |

#### 数据库配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `lightrag.database.type` | `sqlite` | 数据库类型（sqlite/postgres/mysql） |
| `lightrag.database.path` | `./data/lightrag.db` | 数据库路径（SQLite） |

#### 向量数据库配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `lightrag.vector_db.type` | `nano` | 向量数据库类型（nano/milvus/qdrant/chroma） |
| `lightrag.vector_db.dimension` | `1536` | 向量维度 |

#### LLM 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `lightrag.llm.provider` | `openai` | LLM 提供商（openai/anthropic/ollama/custom） |
| `lightrag.llm.model` | `gpt-4o-mini` | LLM 模型名 |
| `lightrag.llm.api_key` | *需手动填写* | LLM API 密钥 |
| `lightrag.llm.base_url` | `""` | 自定义 LLM 基础 URL（可选） |
| `lightrag.llm.max_tokens` | `4096` | 最大 tokens 数 |
| `lightrag.llm.temperature` | `0.0` | 温度参数（0.0-1.0） |

#### 实体提取配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `lightrag.entity_extraction.max_gleaning` | `1` | Gleaning 轮数（0=禁用，1=启用） |
| `lightrag.entity_extraction.use_trilingual` | `false` | 使用三语言提取器（而非 LLM） |

#### 关系提取配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `lightrag.relation_extraction.enabled` | `true` | 启用关系提取 |
| `lightrag.relation_extraction.method` | `llm` | 提取方法（llm/pattern/hybrid） |

#### 安全配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `lightrag.security.enable_api_key` | `false` | 启用 API 密钥验证 |
| `lightrag.security.allowed_origins` | `*` | CORS 允许的源（逗号分隔） |
| `lightrag.security.rate_limit` | `100` | API 速率限制（请求/分钟/IP） |

---

## 高级用法

### 1. 添加新配置字段

**步骤**:
1. 编辑 `config/config.schema.yaml`，添加新字段：
```yaml
- section: my_module.new_feature.enabled
  default: true
  description: "Enable my new feature"
```

2. 重新运行设置脚本：
```bash
./scripts/setup.sh
```

3. 检查 `config/local.yaml` 和 `.env`，新字段已自动添加。

### 2. 自动生成密钥

**用途**: API 密钥、JWT 密钥、加密密钥等

**步骤**:
1. 在 `config.schema.yaml` 中标记为 `type: secret` 和 `auto_generate: true`：
```yaml
- section: my_module.secret_token
  type: secret
  auto_generate: true
  description: "Secret token for authentication"
```

2. 运行 `./scripts/setup.sh`，密钥会自动生成（32 字符）

**注意**:
- 密钥只生成一次，后续运行会保留现有值
- 如需重新生成，删除 `config/local.yaml` 中对应字段后重新运行

### 3. 用户提供的密钥

**用途**: LLM API 密钥等需要用户提供的敏感信息

**步骤**:
1. 在 `config.schema.yaml` 中标记为 `type: secret` 和 `auto_generate: false`：
```yaml
- section: lightrag.llm.api_key
  type: secret
  auto_generate: false
  description: "LLM API key (user-provided)"
```

2. 运行 `./scripts/setup.sh` 生成配置框架

3. 手动编辑 `config/local.yaml`，填写 API 密钥：
```yaml
lightrag:
  llm:
    api_key: sk-your-actual-key-here
```

4. 重新运行 `./scripts/setup.sh` 更新 `.env`

### 4. 深度合并逻辑

**特性**: 修改配置时，现有值会被保留，新字段会使用默认值

**示例**:

**现有配置** (`config/local.yaml`):
```yaml
trilingual:
  chinese:
    enabled: false  # 用户修改过
```

**Schema 更新** (`config.schema.yaml`):
```yaml
- section: trilingual.chinese.enabled
  default: true

- section: trilingual.chinese.cache_dir  # 新增字段
  default: "/custom/path"
```

**运行** `./scripts/setup.sh` **后** (`config/local.yaml`):
```yaml
trilingual:
  chinese:
    enabled: false      # 保留现有值 ✅
    cache_dir: /custom/path  # 使用默认值 ✅
```

### 5. 类型推断

**支持的类型**:
- 布尔值: `true`, `false`
- 整数: `123`, `-456`
- 浮点数: `0.5`, `3.14`
- 字符串: `hello`, `api_key`

**自动转换**:
```yaml
# Schema 中定义
- section: my_module.timeout
  default: 30  # 整数

- section: my_module.enabled
  default: true  # 布尔值

# 生成的 .env
MY_MODULE_TIMEOUT=30  # 整数字符串
MY_MODULE_ENABLED=true  # 布尔值字符串
```

---

## 常见问题

### Q1: 如何修改配置？

**A**: 有两种方法：

**方法 1: 编辑 local.yaml（推荐）**
```bash
# 1. 编辑配置
nano config/local.yaml

# 2. 更新 .env
./scripts/setup.sh
```

**方法 2: 编辑 schema.yaml（添加新字段）**
```bash
# 1. 添加新字段到 schema
nano config/config.schema.yaml

# 2. 重新生成配置
./scripts/setup.sh
```

### Q2: 配置文件丢失了怎么办？

**A**: 重新运行设置脚本即可：
```bash
./scripts/setup.sh
```

**注意**:
- ✅ 如果只是 `.env` 丢失，重新运行会从 `config/local.yaml` 重新生成
- ⚠️ 如果 `config/local.yaml` 也丢失，会使用默认值重新生成（自动生成的密钥会改变）

### Q3: 为什么修改 .env 后重新运行脚本配置被覆盖？

**A**: `.env` 是自动生成的文件，**不应该手动编辑**。

**正确做法**:
1. 编辑 `config/local.yaml`
2. 运行 `./scripts/setup.sh`
3. `.env` 会自动更新

### Q4: 如何在不同环境使用不同配置？

**A**: 使用环境变量覆盖：

**开发环境** (`.env`):
```bash
LIGHTRAG_API_DEBUG=true
LIGHTRAG_LLM_MODEL=gpt-4o-mini
```

**生产环境** (在服务器上设置环境变量):
```bash
export LIGHTRAG_API_DEBUG=false
export LIGHTRAG_LLM_MODEL=gpt-4o
```

环境变量的优先级 > `.env` 文件

### Q5: 密钥泄露了怎么办？

**A**: 重新生成密钥：

1. 删除 `config/local.yaml` 中的密钥字段
2. 运行 `./scripts/setup.sh`
3. 新密钥会自动生成

**示例**:
```bash
# 1. 编辑 config/local.yaml，删除这一行：
# lightrag.api.secret_key: abc123...

# 2. 重新生成
./scripts/setup.sh

# 3. 新密钥已生成并保存
```

### Q6: 如何检查配置是否正确？

**A**: 查看生成的文件：

```bash
# 查看本地配置
cat config/local.yaml

# 查看环境变量
cat .env

# 查看特定配置项
grep "TRILINGUAL_ENABLED" .env
```

### Q7: 脚本运行失败怎么办？

**A**: 检查以下几点：

1. **Python 依赖**:
```bash
pip install pyyaml
```

2. **Schema 文件存在**:
```bash
ls -la config/config.schema.yaml
```

3. **脚本权限**:
```bash
chmod +x scripts/setup.sh
chmod +x scripts/lib/generate_from_schema.py
chmod +x scripts/lib/generate_env.py
```

4. **查看详细错误**:
```bash
./scripts/setup.sh 2>&1 | tee setup.log
```

---

## 文件结构

```
LightRAG/
├── config/
│   ├── config.schema.yaml   # 配置元数据（Git 追踪）
│   └── local.yaml           # 本地配置（Git 忽略，自动生成）
├── scripts/
│   ├── setup.sh             # 一键设置脚本
│   └── lib/
│       ├── generate_from_schema.py  # Schema → local.yaml
│       └── generate_env.py          # local.yaml → .env
├── .env                     # 环境变量（Git 忽略，自动生成）
└── .gitignore               # 忽略 local.yaml 和 .env
```

---

## 工作流示意图

```
┌──────────────────────┐
│ config.schema.yaml   │ ← 配置元数据（Git 追踪）
│ (单一数据源)        │
└──────────┬───────────┘
           │
           │ 运行 ./scripts/setup.sh
           │
           ▼
┌──────────────────────┐
│ generate_from_schema │ ← 读取 schema，生成配置
│                      │   - 深度合并
│                      │   - 自动生成密钥
│                      │   - 保留现有值
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ config/local.yaml    │ ← 本地配置（Git 忽略）
│                      │   - 可手动编辑
│                      │   - 包含密钥
└──────────┬───────────┘
           │
           │ 运行 ./scripts/setup.sh
           │
           ▼
┌──────────────────────┐
│ generate_env.py      │ ← 转换为环境变量格式
│                      │   - 扁平化嵌套结构
│                      │   - 大写 + 下划线
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ .env                 │ ← 环境变量（Git 忽略）
│                      │   - 不要手动编辑
└──────────────────────┘
```

---

## 最佳实践

### ✅ 推荐做法

1. **修改配置**: 编辑 `config/local.yaml`，然后运行 `./scripts/setup.sh`
2. **添加字段**: 编辑 `config.schema.yaml`，然后运行 `./scripts/setup.sh`
3. **版本控制**: 只提交 `config.schema.yaml`，不要提交 `config/local.yaml` 和 `.env`
4. **密钥管理**: 使用 `auto_generate: true` 自动生成密钥
5. **环境隔离**: 使用环境变量覆盖 `.env` 中的配置

### ❌ 避免做法

1. **手动编辑 .env**: 修改会被覆盖
2. **提交密钥**: `config/local.yaml` 和 `.env` 包含敏感信息
3. **硬编码配置**: 在代码中硬编码配置值
4. **跳过脚本**: 手动创建配置文件（会丢失深度合并等特性）

---

## 总结

### 核心优势

✅ **单一数据源**: 所有配置元数据集中管理

✅ **自动化**: 一键生成配置，无需手动管理

✅ **安全性**: 配置文件不会提交到 Git

✅ **灵活性**: 支持深度合并、自动生成密钥、类型推断

✅ **可维护性**: 配置修改清晰可追溯

### 适用场景

- ✓ 需要管理大量配置项
- ✓ 需要自动生成密钥
- ✓ 需要在多个环境部署
- ✓ 需要配置版本控制
- ✓ 团队协作开发

---

## 参考资源

- **Schema 文件**: `config/config.schema.yaml`
- **生成脚本**: `scripts/lib/generate_from_schema.py`
- **环境变量脚本**: `scripts/lib/generate_env.py`
- **设置脚本**: `scripts/setup.sh`

---

## 支持和反馈

如有问题或建议，请：
1. 查看本文档的常见问题部分
2. 查看生成脚本的源代码和注释
3. 提交 Issue 到 LightRAG 仓库
