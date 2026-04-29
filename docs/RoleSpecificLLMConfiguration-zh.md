# 基于角色的 LLM/VLM 配置指南

LightRAG 支持为不同处理阶段配置不同的 LLM 或 VLM。这个机制适合把低成本模型用于抽取，把更强模型用于最终回答，或为多模态分析单独指定视觉语言模型。

## 角色说明

当前支持四个角色：

| 角色 | 用途 |
| --- | --- |
| `EXTRACT` | 实体/关系抽取，以及实体/关系描述摘要。 |
| `KEYWORD` | 查询关键词抽取，用于检索前的 high-level / low-level keyword 生成。 |
| `QUERY` | 最终问答、普通查询、bypass 查询，以及 Ollama-compatible API 的查询路径。 |
| `VLM` | 多模态分析阶段，用于图片、表格、公式等内容的 VLM 分析。 |

如果某个角色没有专门配置，LightRAG 会使用基础 `LLM_*` 配置。

## 基础 LLM 配置

基础配置定义默认 LLM provider、模型、服务地址、认证信息和并发控制：

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key

# 所有 LLM 请求的默认超时时间
LLM_TIMEOUT=180

# 所有 LLM 调用的默认最大并发数
MAX_ASYNC=4
```

常用字段：

| 变量 | 说明 |
| --- | --- |
| `LLM_BINDING` | 基础 LLM provider。支持 `openai`、`ollama`、`lollms`、`azure_openai`、`bedrock`、`gemini`。 |
| `LLM_MODEL` | 基础模型名。对 Azure OpenAI 通常使用 deployment 名称。 |
| `LLM_BINDING_HOST` | 基础 provider endpoint。对于 SDK 默认 endpoint，可使用对应 sentinel，例如 `DEFAULT_GEMINI_ENDPOINT` 或 `DEFAULT_BEDROCK_ENDPOINT`。 |
| `LLM_BINDING_API_KEY` | 基础 API key。Bedrock 不使用这个字段。 |
| `LLM_TIMEOUT` | 基础 LLM timeout。角色未设置 timeout 时继承它。 |
| `MAX_ASYNC` | 基础 LLM 最大并发。角色未设置 `{ROLE}_MAX_ASYNC_LLM` 时继承它。 |

## 角色覆盖变量

每个角色都可以覆盖 binding、模型、endpoint、API key、并发和 timeout：

```env
QUERY_LLM_BINDING=openai
QUERY_LLM_MODEL=gpt-5
QUERY_LLM_BINDING_HOST=https://api.openai.com/v1
QUERY_LLM_BINDING_API_KEY=your_query_api_key
QUERY_MAX_ASYNC_LLM=2
LLM_TIMEOUT_QUERY_LLM=240
```

变量格式：

| 变量 | 说明 |
| --- | --- |
| `{ROLE}_LLM_BINDING` | 覆盖角色 provider。`ROLE` 可为 `EXTRACT`、`KEYWORD`、`QUERY`、`VLM`。 |
| `{ROLE}_LLM_MODEL` | 覆盖角色模型名。 |
| `{ROLE}_LLM_BINDING_HOST` | 覆盖角色 endpoint。 |
| `{ROLE}_LLM_BINDING_API_KEY` | 覆盖角色 API key。Bedrock 不支持。 |
| `{ROLE}_MAX_ASYNC_LLM` | 覆盖角色最大并发。未设置时继承 `MAX_ASYNC`。 |
| `LLM_TIMEOUT_{ROLE}_LLM` | 覆盖角色 timeout。未设置时继承 `LLM_TIMEOUT`。 |

## Provider 参数覆盖

provider 细项使用下面的格式：

```env
{ROLE}_{PROVIDER_PREFIX}_{FIELD}
```

例如：

```env
# 只覆盖 QUERY 角色的 OpenAI reasoning effort
QUERY_OPENAI_LLM_REASONING_EFFORT=medium

# 只覆盖 EXTRACT 角色的 Bedrock 生成参数
EXTRACT_BEDROCK_LLM_TEMPERATURE=0.0
EXTRACT_BEDROCK_LLM_MAX_TOKENS=2048

# 只覆盖 VLM 角色的 Gemini 生成参数
VLM_GEMINI_LLM_MAX_OUTPUT_TOKENS=4096
VLM_GEMINI_LLM_TEMPERATURE=0.2
```

常见 provider 前缀：

| Provider | 基础参数前缀 | 角色参数示例 |
| --- | --- | --- |
| `openai` / `azure_openai` | `OPENAI_LLM_*` | `QUERY_OPENAI_LLM_REASONING_EFFORT` |
| `ollama` | `OLLAMA_LLM_*` | `EXTRACT_OLLAMA_LLM_NUM_PREDICT` |
| `lollms` | 使用 Ollama 兼容参数集合 | `QUERY_OLLAMA_LLM_TEMPERATURE` |
| `bedrock` | `BEDROCK_LLM_*` | `EXTRACT_BEDROCK_LLM_MAX_TOKENS` |
| `gemini` | `GEMINI_LLM_*` | `VLM_GEMINI_LLM_THINKING_CONFIG` |

## 继承规则

### 同一个 provider 内覆盖

如果角色没有设置 `{ROLE}_LLM_BINDING`，或设置成与基础 `LLM_BINDING` 相同，角色会继承基础配置：

- 未设置 `{ROLE}_LLM_MODEL` 时继承 `LLM_MODEL`。
- 未设置 `{ROLE}_LLM_BINDING_HOST` 时继承 `LLM_BINDING_HOST`。
- 未设置 `{ROLE}_LLM_BINDING_API_KEY` 时继承 `LLM_BINDING_API_KEY`。
- 未设置 `LLM_TIMEOUT_{ROLE}_LLM` 时继承 `LLM_TIMEOUT`。
- 未设置 `{ROLE}_MAX_ASYNC_LLM` 时继承 `MAX_ASYNC`。
- provider 参数先继承基础 provider options，再叠加角色专属 provider options。

因此，同一个 provider 下只想换模型时，只需要写模型名：

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key
OPENAI_LLM_REASONING_EFFORT=minimal

# QUERY 继承 host、API key、timeout、并发和 OPENAI_LLM_REASONING_EFFORT
QUERY_LLM_MODEL=gpt-5
```

### 跨 provider 覆盖

如果角色的 `{ROLE}_LLM_BINDING` 与基础 `LLM_BINDING` 不同，就是跨 provider 配置。当前规则是：

- 必须设置 `{ROLE}_LLM_MODEL`。
- 非 Bedrock provider 必须设置 `{ROLE}_LLM_BINDING_API_KEY`。
- 如果没有设置 `{ROLE}_LLM_BINDING_HOST`，LightRAG 会尝试使用该 provider 的默认 host。
- provider 参数不继承基础 provider options，而是从空配置开始，只叠加角色专属 provider options。

示例：基础使用 Ollama，本地抽取；最终回答改用 OpenAI：

```env
LLM_BINDING=ollama
LLM_MODEL=qwen3.5:9b
LLM_BINDING_HOST=http://localhost:11434
OLLAMA_LLM_NUM_CTX=32768

QUERY_LLM_BINDING=openai
QUERY_LLM_MODEL=gpt-5-mini
QUERY_LLM_BINDING_HOST=https://api.openai.com/v1
QUERY_LLM_BINDING_API_KEY=your_openai_api_key
QUERY_OPENAI_LLM_REASONING_EFFORT=minimal
```

跨 provider 时建议显式设置 `{ROLE}_LLM_BINDING_HOST`，避免默认 host 与基础 provider 的 endpoint 混淆。

### Bedrock 认证规则

Bedrock 不使用 `LLM_BINDING_API_KEY`，也不支持 `{ROLE}_LLM_BINDING_API_KEY`。可用认证方式：

- 全局 SigV4：`AWS_ACCESS_KEY_ID`、`AWS_SECRET_ACCESS_KEY`、`AWS_SESSION_TOKEN`、`AWS_REGION`。
- 角色级 SigV4：`{ROLE}_AWS_ACCESS_KEY_ID`、`{ROLE}_AWS_SECRET_ACCESS_KEY`、`{ROLE}_AWS_SESSION_TOKEN`、`{ROLE}_AWS_REGION`。
- 进程级 bearer token：`AWS_BEARER_TOKEN_BEDROCK`。这是 AWS SDK 进程级设置，不能按角色覆盖。

角色级 Bedrock 示例：

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_openai_api_key

EXTRACT_LLM_BINDING=bedrock
EXTRACT_LLM_MODEL=us.amazon.nova-lite-v1:0
EXTRACT_LLM_BINDING_HOST=DEFAULT_BEDROCK_ENDPOINT
EXTRACT_AWS_REGION=us-west-2
EXTRACT_AWS_ACCESS_KEY_ID=your_extract_access_key
EXTRACT_AWS_SECRET_ACCESS_KEY=your_extract_secret_key
EXTRACT_AWS_SESSION_TOKEN=your_optional_session_token
EXTRACT_BEDROCK_LLM_TEMPERATURE=0.0
EXTRACT_BEDROCK_LLM_MAX_TOKENS=2048
```

## Provider 行为对照

| Provider | 角色级 host/base_url | 角色级 API key | 认证限制 |
| --- | --- | --- | --- |
| `openai` | 支持，通过 `{ROLE}_LLM_BINDING_HOST` 传给 OpenAI-compatible client。 | 支持 `{ROLE}_LLM_BINDING_API_KEY`，未设置时同 provider 继承基础 `LLM_BINDING_API_KEY`。 | 当前主要是 API key / Bearer 模式。 |
| `ollama` | 支持，通过 `{ROLE}_LLM_BINDING_HOST` 传给 Ollama client。 | 支持 `{ROLE}_LLM_BINDING_API_KEY`，未设置时同 provider 继承基础 key；底层未收到 key 时会再回退 `OLLAMA_API_KEY`。 | Bearer header。 |
| `lollms` | 支持，通过 `{ROLE}_LLM_BINDING_HOST` 作为 `base_url`。 | 支持 `{ROLE}_LLM_BINDING_API_KEY`，未设置时同 provider 继承基础 key。 | Bearer header。 |
| `azure_openai` | 支持，通过 `{ROLE}_LLM_BINDING_HOST` 作为 Azure endpoint。 | 支持 `{ROLE}_LLM_BINDING_API_KEY`，未设置时同 provider 继承基础 key，也可能回退 `AZURE_OPENAI_API_KEY`。 | `AZURE_OPENAI_API_VERSION` 是全局环境变量，不支持角色级覆盖。 |
| `bedrock` | 支持，通过 `{ROLE}_LLM_BINDING_HOST` 作为 `endpoint_url`；`DEFAULT_BEDROCK_ENDPOINT` 表示交给 AWS SDK 选择。 | 不支持 generic API key。 | 使用全局或角色级 SigV4。`AWS_BEARER_TOKEN_BEDROCK` 是进程级，不能按角色覆盖。 |
| `gemini` | 支持，通过 `{ROLE}_LLM_BINDING_HOST` 传给 Google GenAI client；`DEFAULT_GEMINI_ENDPOINT` 表示使用 SDK 默认 endpoint。 | AI Studio 模式支持 `{ROLE}_LLM_BINDING_API_KEY`。 | Vertex AI 由 `GOOGLE_GENAI_USE_VERTEXAI`、`GOOGLE_CLOUD_PROJECT`、`GOOGLE_CLOUD_LOCATION`、`GOOGLE_APPLICATION_CREDENTIALS` 控制，都是进程级设置。 |

## 推荐配置模式

### 1. 同 provider 只更换模型

适合用同一个 OpenAI key 和 endpoint，但让最终回答使用更强模型：

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key
OPENAI_LLM_REASONING_EFFORT=minimal

QUERY_LLM_MODEL=gpt-5
QUERY_MAX_ASYNC_LLM=2
```

`QUERY` 会继承基础 host、API key 和 `OPENAI_LLM_REASONING_EFFORT`。

### 2. 同 provider 更换模型并调整参数

适合基础模型用于抽取，最终回答使用更高 reasoning effort：

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key
OPENAI_LLM_REASONING_EFFORT=minimal
OPENAI_LLM_MAX_COMPLETION_TOKENS=4096

QUERY_LLM_MODEL=gpt-5
QUERY_OPENAI_LLM_REASONING_EFFORT=medium
QUERY_OPENAI_LLM_MAX_COMPLETION_TOKENS=9000
LLM_TIMEOUT_QUERY_LLM=240
```

### 3. 同 provider 使用不同 endpoint 和 API key

适合所有角色都走 `openai` binding，但其中一些角色访问 OpenAI 官方接口，另一些角色访问本地 vLLM、SGLang 或 OpenRouter 等 OpenAI-compatible endpoint。下面的例子中：

- `EXTRACT` 使用 OpenAI 官方 `gpt-5-mini`。
- `QUERY` 使用 OpenAI 官方 `gpt-5.4`，并使用单独的 OpenAI key。
- `KEYWORD` 使用本地 vLLM 部署的 `Qwen3.5-35B-A3B`。

```env
###########################################################################
# Base LLM fallback. Keep it aligned with EXTRACT so unspecified roles still
# have a valid OpenAI configuration.
###########################################################################
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_extract_openai_api_key
LLM_TIMEOUT=180
MAX_ASYNC=4

###########################################################################
# IMPORTANT:
# Do not set global OPENAI_LLM_REASONING_EFFORT here if any same-provider role
# points to a local OpenAI-compatible server that does not support it.
# Use role-specific OPENAI options instead.
###########################################################################
# OPENAI_LLM_REASONING_EFFORT=none

###########################################################################
# EXTRACT: OpenAI official API, gpt-5-mini
###########################################################################
EXTRACT_LLM_BINDING=openai
EXTRACT_LLM_MODEL=gpt-5-mini
EXTRACT_LLM_BINDING_HOST=https://api.openai.com/v1
EXTRACT_LLM_BINDING_API_KEY=your_extract_openai_api_key
EXTRACT_OPENAI_LLM_REASONING_EFFORT=low
EXTRACT_OPENAI_LLM_MAX_COMPLETION_TOKENS=4096
EXTRACT_MAX_ASYNC_LLM=4
LLM_TIMEOUT_EXTRACT_LLM=180

###########################################################################
# QUERY: OpenAI official API, gpt-5.4, separate API key
###########################################################################
QUERY_LLM_BINDING=openai
QUERY_LLM_MODEL=gpt-5.4
QUERY_LLM_BINDING_HOST=https://api.openai.com/v1
QUERY_LLM_BINDING_API_KEY=your_query_openai_api_key
QUERY_OPENAI_LLM_REASONING_EFFORT=medium
QUERY_OPENAI_LLM_MAX_COMPLETION_TOKENS=9000
QUERY_MAX_ASYNC_LLM=2
LLM_TIMEOUT_QUERY_LLM=240

###########################################################################
# KEYWORD: local vLLM OpenAI-compatible endpoint, Qwen3.5-35B-A3B
###########################################################################
KEYWORD_LLM_BINDING=openai
KEYWORD_LLM_MODEL=Qwen3.5-35B-A3B
KEYWORD_LLM_BINDING_HOST=http://localhost:8000/v1
# If vLLM was started with --api-key, use the same value here.
# If vLLM has no auth, still set a non-empty dummy value to avoid falling
# back to the official OpenAI key.
KEYWORD_LLM_BINDING_API_KEY=local-vllm-api-key
KEYWORD_OPENAI_LLM_MAX_TOKENS=2048
# Optional for Qwen-style models served by vLLM when you want to disable thinking.
KEYWORD_OPENAI_LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
KEYWORD_MAX_ASYNC_LLM=4
LLM_TIMEOUT_KEYWORD_LLM=180
```

这个模式不是跨 provider，因为三个角色的 binding 都是 `openai`。LightRAG 会分别把每个角色的 `*_LLM_BINDING_HOST` 和 `*_LLM_BINDING_API_KEY` 传给 OpenAI-compatible client。

注意：同 provider 的 provider options 会继承基础 `OPENAI_LLM_*`。如果本地 vLLM 不支持 OpenAI 官方参数，例如 `reasoning_effort`，不要设置全局 `OPENAI_LLM_REASONING_EFFORT`；改用 `EXTRACT_OPENAI_LLM_REASONING_EFFORT`、`QUERY_OPENAI_LLM_REASONING_EFFORT` 这类角色级变量。

### 4. 某个角色跨 provider

适合基础使用 OpenAI 官方模型，只有关键词抽取使用本地 Ollama：

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_openai_api_key
OPENAI_LLM_REASONING_EFFORT=medium

KEYWORD_LLM_BINDING=ollama
KEYWORD_LLM_MODEL=qwen3.5:9b
KEYWORD_LLM_BINDING_HOST=http://localhost:11434
KEYWORD_LLM_BINDING_API_KEY=ollama-local-key
KEYWORD_OLLAMA_LLM_NUM_CTX=32768
```

跨 provider 时，Ollama 参数不会继承 OpenAI 参数。`KEYWORD_LLM_BINDING_API_KEY` 对本地 Ollama 通常可以使用占位值；当前跨 provider 校验会要求非 Bedrock 角色显式提供角色级 API key。

### 5. 为 VLM 单独指定多模态模型

适合文本任务使用便宜模型，多模态分析使用视觉语言模型：

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key

VLM_LLM_BINDING=openai
VLM_LLM_MODEL=gpt-4o
VLM_OPENAI_LLM_MAX_TOKENS=4096
VLM_MAX_ASYNC_LLM=2
LLM_TIMEOUT_VLM_LLM=240
```

如果 VLM 使用同一个 provider 和 key，可以省略 `VLM_LLM_BINDING_HOST` 与 `VLM_LLM_BINDING_API_KEY`。

### 6. Bedrock 角色级 SigV4 凭证

适合只有某个角色访问 Bedrock，并使用独立 IAM/STS 凭证：

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_openai_api_key

QUERY_LLM_BINDING=bedrock
QUERY_LLM_MODEL=us.amazon.nova-lite-v1:0
QUERY_LLM_BINDING_HOST=DEFAULT_BEDROCK_ENDPOINT
QUERY_AWS_REGION=us-east-1
QUERY_AWS_ACCESS_KEY_ID=your_query_access_key
QUERY_AWS_SECRET_ACCESS_KEY=your_query_secret_key
QUERY_AWS_SESSION_TOKEN=your_optional_session_token
QUERY_BEDROCK_LLM_MAX_TOKENS=4096
QUERY_BEDROCK_LLM_TEMPERATURE=0.2
```

不要设置 `QUERY_LLM_BINDING_API_KEY`，Bedrock 会拒绝该配置。

## 注意事项

- 同 provider 下，`OPENAI_LLM_REASONING_EFFORT`、`OPENAI_LLM_MAX_TOKENS`、`OLLAMA_LLM_NUM_CTX`、`GEMINI_LLM_THINKING_CONFIG` 等 provider 参数会自动继承。
- 当前没有干净的角色级“取消继承某个 provider 参数”的语义。如果某个同 provider 角色模型不支持基础参数，需要为该角色显式覆盖为可用值，或将它配置成跨 provider，并且只设置该角色支持的 provider 参数。
- `azure_openai` 的 `AZURE_OPENAI_DEPLOYMENT` 和 `AZURE_OPENAI_API_VERSION` 是全局环境变量。若设置了 `AZURE_OPENAI_DEPLOYMENT`，它可能优先于角色模型名。
- Gemini Vertex AI 模式由进程级 Google 环境变量控制，不能在同一个 LightRAG 进程里让某些角色使用 Vertex AI、另一些角色使用 AI Studio API key。
- `LLM_BINDING_HOST` 在 Docker/Compose 中通常需要使用容器可访问地址，例如 `host.docker.internal`，角色级 host 也遵循相同原则。
- 修改 `.env` 后请重启 LightRAG Server。部分 IDE 终端会预加载 `.env`，建议打开新的终端会话确认环境变量生效。
