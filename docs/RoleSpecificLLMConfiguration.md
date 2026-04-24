# Role-Specific LLM/VLM Configuration Guide

LightRAG supports configuring different LLMs or VLMs for different processing stages. This mechanism is useful when using a lower-cost model for extraction, a stronger model for final answers, or a dedicated vision-language model for multimodal analysis.

## Role Overview

Four roles are currently supported:

| Role | Purpose |
| --- | --- |
| `EXTRACT` | Entity/relation extraction and entity/relation description summarization. |
| `KEYWORD` | Query keyword extraction for high-level / low-level keyword generation before retrieval. |
| `QUERY` | Final QA, regular queries, bypass queries, and the query path of the Ollama-compatible API. |
| `VLM` | Multimodal analysis stage for VLM analysis of images, tables, formulas, and similar content. |

If a role has no dedicated configuration, LightRAG uses the base `LLM_*` configuration.

## Base LLM Configuration

The base configuration defines the default LLM provider, model, service endpoint, authentication information, and concurrency control:

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key

# Default timeout for all LLM requests
LLM_TIMEOUT=180

# Default maximum concurrency for all LLM calls
MAX_ASYNC=4
```

Common fields:

| Variable | Description |
| --- | --- |
| `LLM_BINDING` | Base LLM provider. Supported values are `openai`, `ollama`, `lollms`, `azure_openai`, `bedrock`, and `gemini`. |
| `LLM_MODEL` | Base model name. For Azure OpenAI, this is usually the deployment name. |
| `LLM_BINDING_HOST` | Base provider endpoint. For SDK default endpoints, use the corresponding sentinel, such as `DEFAULT_GEMINI_ENDPOINT` or `DEFAULT_BEDROCK_ENDPOINT`. |
| `LLM_BINDING_API_KEY` | Base API key. Bedrock does not use this field. |
| `LLM_TIMEOUT` | Base LLM timeout. A role inherits it when no role timeout is set. |
| `MAX_ASYNC` | Base maximum LLM concurrency. A role inherits it when `MAX_ASYNC_{ROLE}_LLM` is not set. |

## Role Override Variables

Each role can override the binding, model, endpoint, API key, concurrency, and timeout:

```env
QUERY_LLM_BINDING=openai
QUERY_LLM_MODEL=gpt-5
QUERY_LLM_BINDING_HOST=https://api.openai.com/v1
QUERY_LLM_BINDING_API_KEY=your_query_api_key
MAX_ASYNC_QUERY_LLM=2
LLM_TIMEOUT_QUERY_LLM=240
```

Variable format:

| Variable | Description |
| --- | --- |
| `{ROLE}_LLM_BINDING` | Overrides the role provider. `ROLE` can be `EXTRACT`, `KEYWORD`, `QUERY`, or `VLM`. |
| `{ROLE}_LLM_MODEL` | Overrides the role model name. |
| `{ROLE}_LLM_BINDING_HOST` | Overrides the role endpoint. |
| `{ROLE}_LLM_BINDING_API_KEY` | Overrides the role API key. Bedrock does not support it. |
| `MAX_ASYNC_{ROLE}_LLM` | Overrides the role maximum concurrency. Inherits `MAX_ASYNC` when unset. |
| `LLM_TIMEOUT_{ROLE}_LLM` | Overrides the role timeout. Inherits `LLM_TIMEOUT` when unset. |

## Provider Option Overrides

Provider-specific options use the following format:

```env
{ROLE}_{PROVIDER_PREFIX}_{FIELD}
```

Examples:

```env
# Override only the OpenAI reasoning effort for the QUERY role
QUERY_OPENAI_LLM_REASONING_EFFORT=medium

# Override only Bedrock generation parameters for the EXTRACT role
EXTRACT_BEDROCK_LLM_TEMPERATURE=0.0
EXTRACT_BEDROCK_LLM_MAX_TOKENS=2048

# Override only Gemini generation parameters for the VLM role
VLM_GEMINI_LLM_MAX_OUTPUT_TOKENS=4096
VLM_GEMINI_LLM_TEMPERATURE=0.2
```

Common provider prefixes:

| Provider | Base option prefix | Role option example |
| --- | --- | --- |
| `openai` / `azure_openai` | `OPENAI_LLM_*` | `QUERY_OPENAI_LLM_REASONING_EFFORT` |
| `ollama` | `OLLAMA_LLM_*` | `EXTRACT_OLLAMA_LLM_NUM_PREDICT` |
| `lollms` | Uses the Ollama-compatible option set | `QUERY_OLLAMA_LLM_TEMPERATURE` |
| `bedrock` | `BEDROCK_LLM_*` | `EXTRACT_BEDROCK_LLM_MAX_TOKENS` |
| `gemini` | `GEMINI_LLM_*` | `VLM_GEMINI_LLM_THINKING_CONFIG` |

## Inheritance Rules

### Overrides Within the Same Provider

If a role does not set `{ROLE}_LLM_BINDING`, or sets it to the same value as the base `LLM_BINDING`, the role inherits the base configuration:

- Inherits `LLM_MODEL` when `{ROLE}_LLM_MODEL` is not set.
- Inherits `LLM_BINDING_HOST` when `{ROLE}_LLM_BINDING_HOST` is not set.
- Inherits `LLM_BINDING_API_KEY` when `{ROLE}_LLM_BINDING_API_KEY` is not set.
- Inherits `LLM_TIMEOUT` when `LLM_TIMEOUT_{ROLE}_LLM` is not set.
- Inherits `MAX_ASYNC` when `MAX_ASYNC_{ROLE}_LLM` is not set.
- Provider options first inherit the base provider options, then apply role-specific provider options.

Therefore, when you only want to change the model within the same provider, you only need to set the model name:

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key
OPENAI_LLM_REASONING_EFFORT=minimal

# QUERY inherits host, API key, timeout, concurrency, and OPENAI_LLM_REASONING_EFFORT
QUERY_LLM_MODEL=gpt-5
```

### Cross-Provider Overrides

If a role's `{ROLE}_LLM_BINDING` differs from the base `LLM_BINDING`, it is a cross-provider configuration. The current rules are:

- `{ROLE}_LLM_MODEL` must be set.
- Non-Bedrock providers must set `{ROLE}_LLM_BINDING_API_KEY`.
- If `{ROLE}_LLM_BINDING_HOST` is not set, LightRAG tries to use that provider's default host.
- Provider options do not inherit base provider options. They start from the target provider defaults, then apply role-specific provider options.

Example: use Ollama as the base for local extraction, then use OpenAI for final answers:

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

For cross-provider configurations, explicitly setting `{ROLE}_LLM_BINDING_HOST` is recommended to avoid confusion between the default host and the base provider endpoint.

### Bedrock Authentication Rules

Bedrock does not use `LLM_BINDING_API_KEY` and does not support `{ROLE}_LLM_BINDING_API_KEY`. Available authentication methods are:

- Global SigV4: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, and `AWS_REGION`.
- Role-level SigV4: `{ROLE}_AWS_ACCESS_KEY_ID`, `{ROLE}_AWS_SECRET_ACCESS_KEY`, `{ROLE}_AWS_SESSION_TOKEN`, and `{ROLE}_AWS_REGION`.
- Process-level bearer token: `AWS_BEARER_TOKEN_BEDROCK`. This is an AWS SDK process-level setting and cannot be overridden per role.

Role-level Bedrock example:

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

## Provider Behavior Matrix

| Provider | Role-level host/base_url | Role-level API key | Authentication limitations |
| --- | --- | --- | --- |
| `openai` | Supported, passed to the OpenAI-compatible client through `{ROLE}_LLM_BINDING_HOST`. | Supports `{ROLE}_LLM_BINDING_API_KEY`; when unset within the same provider, it inherits the base `LLM_BINDING_API_KEY`. | Currently mainly API key / Bearer mode. |
| `ollama` | Supported, passed to the Ollama client through `{ROLE}_LLM_BINDING_HOST`. | Supports `{ROLE}_LLM_BINDING_API_KEY`; when unset within the same provider, it inherits the base key. If no key reaches the lower layer, it falls back to `OLLAMA_API_KEY`. | Bearer header. |
| `lollms` | Supported, using `{ROLE}_LLM_BINDING_HOST` as `base_url`. | Supports `{ROLE}_LLM_BINDING_API_KEY`; when unset within the same provider, it inherits the base key. | Bearer header. |
| `azure_openai` | Supported, using `{ROLE}_LLM_BINDING_HOST` as the Azure endpoint. | Supports `{ROLE}_LLM_BINDING_API_KEY`; when unset within the same provider, it inherits the base key and may also fall back to `AZURE_OPENAI_API_KEY`. | `AZURE_OPENAI_API_VERSION` is a global environment variable and does not support role-level overrides. |
| `bedrock` | Supported, using `{ROLE}_LLM_BINDING_HOST` as `endpoint_url`; `DEFAULT_BEDROCK_ENDPOINT` means letting the AWS SDK choose. | Generic API keys are not supported. | Uses global or role-level SigV4. `AWS_BEARER_TOKEN_BEDROCK` is process-level and cannot be overridden per role. |
| `gemini` | Supported, passed to the Google GenAI client through `{ROLE}_LLM_BINDING_HOST`; `DEFAULT_GEMINI_ENDPOINT` means using the SDK default endpoint. | AI Studio mode supports `{ROLE}_LLM_BINDING_API_KEY`. | Vertex AI is controlled by `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GOOGLE_APPLICATION_CREDENTIALS`; all are process-level settings. |

## Recommended Configuration Patterns

### 1. Same Provider, Only Change the Model

Suitable when using the same OpenAI key and endpoint, but using a stronger model for final answers:

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key
OPENAI_LLM_REASONING_EFFORT=minimal

QUERY_LLM_MODEL=gpt-5
MAX_ASYNC_QUERY_LLM=2
```

`QUERY` inherits the base host, API key, and `OPENAI_LLM_REASONING_EFFORT`.

### 2. Same Provider, Change the Model and Tune Options

Suitable when the base model is used for extraction and final answers use a higher reasoning effort:

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

### 3. Same Provider with Different Endpoints and API Keys

Suitable when all roles use the `openai` binding, but some roles access the official OpenAI API while others access a local vLLM, SGLang, OpenRouter, or another OpenAI-compatible endpoint. In the example below:

- `EXTRACT` uses the official OpenAI `gpt-5-mini`.
- `QUERY` uses the official OpenAI `gpt-5.4` with a separate OpenAI key.
- `KEYWORD` uses `Qwen3.5-35B-A3B` deployed by local vLLM.

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
MAX_ASYNC_EXTRACT_LLM=4
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
MAX_ASYNC_QUERY_LLM=2
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
MAX_ASYNC_KEYWORD_LLM=4
LLM_TIMEOUT_KEYWORD_LLM=180
```

This pattern is not cross-provider because all three roles use the `openai` binding. LightRAG passes each role's `*_LLM_BINDING_HOST` and `*_LLM_BINDING_API_KEY` to the OpenAI-compatible client separately.

Note: provider options within the same provider inherit the base `OPENAI_LLM_*`. If the local vLLM server does not support official OpenAI parameters such as `reasoning_effort`, do not set the global `OPENAI_LLM_REASONING_EFFORT`; use role-level variables such as `EXTRACT_OPENAI_LLM_REASONING_EFFORT` and `QUERY_OPENAI_LLM_REASONING_EFFORT` instead.

### 4. One Role Crosses Provider

Suitable when the base uses an official OpenAI model and only keyword extraction uses local Ollama:

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

For cross-provider configurations, Ollama options do not inherit OpenAI options. For local Ollama, `KEYWORD_LLM_BINDING_API_KEY` can usually use a placeholder value; the current cross-provider validation requires non-Bedrock roles to explicitly provide a role-level API key.

### 5. Specify a Dedicated Multimodal Model for VLM

Suitable when text tasks use a cheaper model and multimodal analysis uses a vision-language model:

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key

VLM_LLM_BINDING=openai
VLM_LLM_MODEL=gpt-4o
VLM_OPENAI_LLM_MAX_TOKENS=4096
MAX_ASYNC_VLM_LLM=2
LLM_TIMEOUT_VLM_LLM=240
```

If VLM uses the same provider and key, `VLM_LLM_BINDING_HOST` and `VLM_LLM_BINDING_API_KEY` can be omitted.

### 6. Bedrock Role-Level SigV4 Credentials

Suitable when only one role accesses Bedrock and uses independent IAM/STS credentials:

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

Do not set `QUERY_LLM_BINDING_API_KEY`; Bedrock rejects that configuration.

## Caveats

- Within the same provider, provider options such as `OPENAI_LLM_REASONING_EFFORT`, `OPENAI_LLM_MAX_TOKENS`, `OLLAMA_LLM_NUM_CTX`, and `GEMINI_LLM_THINKING_CONFIG` are inherited automatically.
- There is currently no clean role-level semantic for "unsetting an inherited provider option". If a model in a same-provider role does not support a base option, explicitly override that option for the role with a supported value, or configure the role as cross-provider and use the target provider defaults.
- `AZURE_OPENAI_DEPLOYMENT` and `AZURE_OPENAI_API_VERSION` for `azure_openai` are global environment variables. If `AZURE_OPENAI_DEPLOYMENT` is set, it may take precedence over the role model name.
- Gemini Vertex AI mode is controlled by process-level Google environment variables. In the same LightRAG process, some roles cannot use Vertex AI while others use AI Studio API keys.
- In Docker/Compose, `LLM_BINDING_HOST` usually needs to use a container-reachable address such as `host.docker.internal`; role-level hosts follow the same principle.
- Restart LightRAG Server after modifying `.env`. Some IDE terminals preload `.env`, so opening a new terminal session is recommended to confirm that environment variables take effect.
