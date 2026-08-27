# LLM and Embedding Provider Options Reference

This document is the complete reference for **provider-specific generation options** —
the `OPENAI_LLM_*`, `OLLAMA_LLM_*`, `GEMINI_LLM_*`, `BEDROCK_LLM_*`,
`OLLAMA_EMBEDDING_*`, and `GEMINI_EMBEDDING_*` families. These control sampling,
output length, thinking/reasoning behavior, and provider pass-through payloads.

It does **not** cover binding selection and credentials (`LLM_BINDING`, `LLM_MODEL`,
`LLM_BINDING_HOST`, `LLM_BINDING_API_KEY`, `LLM_TIMEOUT`, `MAX_ASYNC_LLM`) — those are
described in [LightRAG Server and WebUI](./LightRAG-API-Server.md) — nor role-level
inheritance rules, which are described in
[Role-Specific LLM/VLM Configuration Guide](./RoleSpecificLLMConfiguration.md).

The single source of truth for the field lists below is
`lightrag/llm/binding_options.py`. This document is intentionally English-only: it is a
technical reference, not a tutorial.

## Contents

- [1. Naming and resolution](#1-naming-and-resolution)
- [2. Coverage matrix](#2-coverage-matrix)
- [3. OpenAI / Azure OpenAI LLM options](#3-openai--azure-openai-llm-options)
- [4. Ollama LLM options](#4-ollama-llm-options)
- [5. Gemini LLM options](#5-gemini-llm-options)
- [6. Bedrock LLM options](#6-bedrock-llm-options)
- [7. Ollama embedding options](#7-ollama-embedding-options)
- [8. Gemini embedding options](#8-gemini-embedding-options)
- [9. Where the values end up](#9-where-the-values-end-up)
- [10. Provider options and the LLM cache](#10-provider-options-and-the-llm-cache)
- [11. Common tasks](#11-common-tasks)
- [12. Inspecting the options yourself](#12-inspecting-the-options-yourself)

## 1. Naming and resolution

### 1.1 Name forms

Every option exists under one prefix per provider *and* per direction (LLM vs
embedding). The three usable forms are:

| Scope | Form | Example |
| --- | --- | --- |
| Base, environment / `.env` | `{PREFIX}_{FIELD}` | `OPENAI_LLM_MAX_COMPLETION_TOKENS=9000` |
| Base, command line | `--{prefix-in-lowercase-dashes}-{field}` | `--openai-llm-max_completion_tokens 9000` |
| Role, environment / `.env` only | `{ROLE}_{PREFIX}_{FIELD}` | `QUERY_OPENAI_LLM_MAX_COMPLETION_TOKENS=9000` |

Note the CLI spelling: the *prefix* uses dashes, the *field* keeps its underscores
(`--ollama-llm-num_ctx`, not `--ollama-llm-num-ctx`).

`ROLE` is one of `EXTRACT`, `KEYWORD`, `QUERY`, `VLM`. **Role-level provider options
have no command-line equivalent** — they are read from the environment only.

Precedence, highest first: base command-line argument, base environment variable,
option unset. A role-level environment variable overlays whatever the base resolved
to (for a same-provider role) or an empty set (for a cross-provider role); see
[RoleSpecificLLMConfiguration.md](./RoleSpecificLLMConfiguration.md#inheritance-rules).

### 1.2 Unset means "not sent"

Every option is registered with `argparse.SUPPRESS` as its default. An option you do
not configure is **absent from the outgoing request**, so the provider applies its own
default. Consequences worth internalizing:

- The default values visible in `lightrag/llm/binding_options.py` (and in the generated
  `.env` sample of §12) are **documentation of the provider's typical default, not
  values LightRAG ships**. `OLLAMA_LLM_THINK` shows `True` and
  `OPENAI_LLM_REASONING_EFFORT` shows `medium`, yet neither is sent unless you set it.
- `lightrag-server --llm-binding openai --help` therefore prints no default values.
- Removing an option means **commenting the line out**, not setting it to an empty
  value (see §1.4).

### 1.3 Only the selected binding's options are read at base level

Base-level options are registered for the binding actually selected by `LLM_BINDING` /
`EMBEDDING_BINDING` (`lightrag/api/config.py`). With `LLM_BINDING=ollama`, every
`OPENAI_LLM_*` variable in your `.env` is ignored — silently, because it is never read.
This is a frequent source of "my setting has no effect".

Role-level variables are read straight from the environment instead, so a
cross-provider role does honor `{ROLE}_{PREFIX}_*` for its own provider — but it does
**not** inherit the base provider's options (a cross-provider role starts from an empty
option set by design).

### 1.4 Value syntax

| Field type | Accepted spelling | Notes |
| --- | --- | --- |
| bool | `true`/`1`/`yes`/`t`/`on` → true; anything else → false | Case-insensitive. |
| bool-or-level (`OLLAMA_LLM_THINK` only) | `true`/`false`/… or `low`/`medium`/`high` | An unrecognized word is a **startup error**, not a silent `false`. An empty value means `false`, not "unset". |
| int / float | plain number | An empty value is a startup error: `argument --openai-llm-max_tokens: invalid int value: ''`. |
| list | JSON array, e.g. `'["</s>", "\n\n"]'` | Quote it in `.env` / the shell. |
| dict | JSON object, e.g. `'{"reasoning": {"enabled": false}}'` | Quote it in `.env` / the shell. |
| str | verbatim | An empty value **is** a value: `OPENAI_LLM_SERVICE_TIER=` sends `service_tier: ""` to the provider. Comment the line out instead. |

Malformed JSON behaves differently by scope, which matters when debugging:

- **Base level** (`OPENAI_LLM_EXTRA_BODY=not json`): the option is dropped and the
  server starts as if it were unset.
- **Role level** (`QUERY_OPENAI_LLM_EXTRA_BODY=not json`): the raw string is kept and
  forwarded, so the failure surfaces later as a provider-side request error.

## 2. Coverage matrix

Not every binding has a provider option set. Bindings absent from the LLM column below
accept no generation options at all.

| Binding | LLM options | Embedding options | Notes |
| --- | --- | --- | --- |
| `openai` | `OPENAI_LLM_*` | none | Also the binding for OpenAI-compatible servers: vLLM, SGLang, OpenRouter, and similar. |
| `azure_openai` | `OPENAI_LLM_*` (same option set) | none | `AZURE_OPENAI_API_VERSION` / `AZURE_OPENAI_DEPLOYMENT` are separate process-level variables, not provider options. |
| `ollama` | `OLLAMA_LLM_*` | `OLLAMA_EMBEDDING_*` | |
| `gemini` | `GEMINI_LLM_*` | `GEMINI_EMBEDDING_*` | |
| `bedrock` | `BEDROCK_LLM_*` | none | The legacy alias `aws_bedrock` is normalized to `bedrock`. |
| `lollms` | see note | none | **Base level: no options at all.** `lightrag-server --llm-binding lollms --help` prints no option group, and the base binding is wired with an empty options payload. Role level maps lollms to the Ollama option set, so `{ROLE}_OLLAMA_LLM_*` is read — but the lollms driver only consumes `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, and `seed`; every other field is ignored. |
| `jina` | — (embedding-only binding) | none | |
| `voyageai` | — (embedding-only binding) | none | |

## 3. OpenAI / Azure OpenAI LLM options

Prefix `OPENAI_LLM_` — shared by the `openai` and `azure_openai` bindings.

| Environment variable | Type | Description |
| --- | --- | --- |
| `OPENAI_LLM_TEMPERATURE` | float | Controls randomness (0.0-2.0, higher = more creative). |
| `OPENAI_LLM_TOP_P` | float | Nucleus sampling parameter (0.0-1.0, lower = more focused). |
| `OPENAI_LLM_MAX_COMPLETION_TOKENS` | int | Maximum number of tokens to generate. The current parameter for OpenAI reasoning-capable models. |
| `OPENAI_LLM_MAX_TOKENS` | int | Maximum number of tokens to generate. **Deprecated by OpenAI** in favor of `max_completion_tokens`, but still the correct field for most OpenAI-compatible servers (vLLM, SGLang, many gateways). |
| `OPENAI_LLM_FREQUENCY_PENALTY` | float | Penalty for token frequency (-2.0 to 2.0, positive values discourage repetition). |
| `OPENAI_LLM_PRESENCE_PENALTY` | float | Penalty for token presence (-2.0 to 2.0, positive values encourage new topics). |
| `OPENAI_LLM_REASONING_EFFORT` | str | Reasoning effort for reasoning-capable models. Forwarded verbatim, so the accepted words are the provider's, not LightRAG's — OpenAI currently accepts `minimal`/`low`/`medium`/`high`, other gateways add their own (e.g. `none`). |
| `OPENAI_LLM_STOP` | JSON list | Stop sequences, e.g. `'["</s>", "\n\n"]'`. |
| `OPENAI_LLM_SERVICE_TIER` | str | Service tier for API usage. |
| `OPENAI_LLM_SAFETY_IDENTIFIER` | str | Safety identifier for content filtering. |
| `OPENAI_LLM_EXTRA_BODY` | JSON dict | Extra top-level request-body fields, merged into the request. The escape hatch for anything not in this table — vLLM/SGLang chat-template switches, OpenRouter routing and reasoning controls. |

Notes:

- These options are merged into the Chat Completions request as keyword arguments and
  forwarded as-is. LightRAG does not validate them against the target model, so a
  parameter your endpoint does not implement surfaces as a provider-side error.
- A local OpenAI-compatible server that rejects `reasoning_effort` is the usual reason
  to keep that option role-level (`EXTRACT_OPENAI_LLM_REASONING_EFFORT=…`) rather than
  global.

## 4. Ollama LLM options

Prefix `OLLAMA_LLM_`. All fields except `think` travel to the Ollama server inside the
request's `options` payload.

### 4.1 Context and output length

| Environment variable | Type | Description |
| --- | --- | --- |
| `OLLAMA_LLM_NUM_CTX` | int | Context window size in tokens. Must be larger than `MAX_TOTAL_TOKENS + 2000`; `env.example` ships it set to `32768` because the Ollama default is usually too small for LightRAG's prompts. |
| `OLLAMA_LLM_NUM_PREDICT` | int | Maximum number of tokens to predict — the output budget. Use it to stop runaway extraction output. |
| `OLLAMA_LLM_NUM_KEEP` | int | Number of tokens to keep from the initial prompt. |
| `OLLAMA_LLM_STOP` | JSON list | Stop sequences, e.g. `'["</s>", "\n\n"]'`. |
| `OLLAMA_LLM_PENALIZE_NEWLINE` | bool | Penalize newline tokens. |

### 4.2 Sampling

| Environment variable | Type | Description |
| --- | --- | --- |
| `OLLAMA_LLM_TEMPERATURE` | float | Controls randomness (0.0-2.0, higher = more creative). |
| `OLLAMA_LLM_TOP_K` | int | Top-k sampling parameter (0 = disabled). |
| `OLLAMA_LLM_TOP_P` | float | Top-p (nucleus) sampling parameter (0.0-1.0). |
| `OLLAMA_LLM_MIN_P` | float | Minimum probability threshold (0.0 = disabled). |
| `OLLAMA_LLM_TYPICAL_P` | float | Typical probability mass (1.0 = disabled). |
| `OLLAMA_LLM_TFS_Z` | float | Tail free sampling parameter (1.0 = disabled). |
| `OLLAMA_LLM_SEED` | int | Random seed for generation (-1 for random). |

### 4.3 Repetition control and Mirostat

| Environment variable | Type | Description |
| --- | --- | --- |
| `OLLAMA_LLM_REPEAT_PENALTY` | float | Penalty for repetition (1.0 = no penalty). |
| `OLLAMA_LLM_REPEAT_LAST_N` | int | Number of tokens to consider for the repetition penalty. |
| `OLLAMA_LLM_PRESENCE_PENALTY` | float | Penalty for token presence (-2.0 to 2.0). |
| `OLLAMA_LLM_FREQUENCY_PENALTY` | float | Penalty for token frequency (-2.0 to 2.0). |
| `OLLAMA_LLM_MIROSTAT` | int | Mirostat sampling algorithm (0 = disabled, 1 = Mirostat 1.0, 2 = Mirostat 2.0). |
| `OLLAMA_LLM_MIROSTAT_TAU` | float | Mirostat target entropy. |
| `OLLAMA_LLM_MIROSTAT_ETA` | float | Mirostat learning rate. |

### 4.4 Thinking / reasoning

| Environment variable | Type | Description |
| --- | --- | --- |
| `OLLAMA_LLM_THINK` | bool or `low`/`medium`/`high` | Enables the model's extended-thinking trace. LLM only — the embedding option set has no such field. |

`think` is the one Ollama option that does not travel inside `options`; it is a
top-level request field. Details that matter:

- Leaving it unset follows the model's own default. Setting `true` for a model without
  thinking support makes Ollama reject the request.
- Reasoning levels (`low`/`medium`/`high`) additionally require an Ollama server that
  supports levels; `think` in any form requires `ollama>=0.5.4` on the LightRAG side,
  and a misconfiguration fails at server startup rather than mid-pipeline.
- An empty value (`OLLAMA_LLM_THINK=`) means `false`, not "unset".
- A thinking-capable model can spend its whole generation budget on hidden reasoning
  and return empty entities/relations during extraction (issue #3597). The fix is
  `EXTRACT_OLLAMA_LLM_THINK=false` (and `KEYWORD_OLLAMA_LLM_THINK=false` if keyword
  extraction is affected too) rather than a global switch, because disabling thinking
  measurably hurts extraction quality on some models.

### 4.5 Hardware and memory

These are Ollama server-side runtime knobs. Set them only if you know why.

| Environment variable | Type | Description |
| --- | --- | --- |
| `OLLAMA_LLM_NUMA` | bool | Enable NUMA optimization. |
| `OLLAMA_LLM_NUM_BATCH` | int | Batch size for processing. |
| `OLLAMA_LLM_NUM_GPU` | int | Number of GPUs to use (-1 for auto). |
| `OLLAMA_LLM_MAIN_GPU` | int | Main GPU index. |
| `OLLAMA_LLM_LOW_VRAM` | bool | Optimize for low VRAM. |
| `OLLAMA_LLM_NUM_THREAD` | int | Number of CPU threads (0 for auto). |
| `OLLAMA_LLM_F16_KV` | bool | Use half-precision for the key/value cache. |
| `OLLAMA_LLM_USE_MMAP` | bool | Use memory mapping for model files. |
| `OLLAMA_LLM_USE_MLOCK` | bool | Lock the model in memory. |
| `OLLAMA_LLM_VOCAB_ONLY` | bool | Only load the vocabulary. |
| `OLLAMA_LLM_LOGITS_ALL` | bool | Return logits for all tokens. |
| `OLLAMA_LLM_EMBEDDING_ONLY` | bool | Only use for embeddings. |

## 5. Gemini LLM options

Prefix `GEMINI_LLM_`. Each field maps onto the Google GenAI
`GenerateContentConfig` field of the same name.

| Environment variable | Type | Description |
| --- | --- | --- |
| `GEMINI_LLM_TEMPERATURE` | float | Controls randomness (0.0-2.0, higher = more creative). |
| `GEMINI_LLM_TOP_P` | float | Nucleus sampling parameter (0.0-1.0). |
| `GEMINI_LLM_TOP_K` | int | Limits sampling to the top K tokens (1 disables the limit). |
| `GEMINI_LLM_MAX_OUTPUT_TOKENS` | int | Maximum tokens generated in the response. |
| `GEMINI_LLM_CANDIDATE_COUNT` | int | Number of candidates returned per request. |
| `GEMINI_LLM_PRESENCE_PENALTY` | float | Penalty for token presence (-2.0 to 2.0). |
| `GEMINI_LLM_FREQUENCY_PENALTY` | float | Penalty for token frequency (-2.0 to 2.0). |
| `GEMINI_LLM_STOP_SEQUENCES` | JSON list | Stop sequences, e.g. `'["END"]'`. |
| `GEMINI_LLM_SEED` | int | Random seed for reproducible generation. |
| `GEMINI_LLM_THINKING_CONFIG` | JSON dict | Thinking configuration, e.g. `'{"thinking_budget": 1024}'`, `'{"include_thoughts": true}'`, or `'{"thinking_budget": 0, "include_thoughts": false}'` to turn thinking off. `thinking_budget: -1` selects Gemini's dynamic budget (the model chooses). |
| `GEMINI_LLM_SAFETY_SETTINGS` | JSON dict | Gemini safety-settings overrides. |

Note: entries whose value is `None` or an empty string are dropped before the config
object is built, so an empty setting cannot produce a type error at request time.

## 6. Bedrock LLM options

Prefix `BEDROCK_LLM_`. These map onto the Converse API.

| Environment variable | Type | Description |
| --- | --- | --- |
| `BEDROCK_LLM_TEMPERATURE` | float | Controls randomness (0.0-1.0 for most Bedrock models). |
| `BEDROCK_LLM_MAX_TOKENS` | int | Maximum tokens generated in the response → `inferenceConfig.maxTokens`. |
| `BEDROCK_LLM_TOP_P` | float | Nucleus sampling parameter (0.0-1.0) → `inferenceConfig.topP`. |
| `BEDROCK_LLM_STOP_SEQUENCES` | JSON list | Stop sequences → `inferenceConfig.stopSequences`. |
| `BEDROCK_LLM_EXTRA_FIELDS` | JSON dict | Model-specific request fields forwarded as `additionalModelRequestFields`, e.g. `'{"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}}'`. The Bedrock counterpart of `OPENAI_LLM_EXTRA_BODY`. |

Note: the driver builds `inferenceConfig` from exactly the four fields above. Anything
model-specific must go through `BEDROCK_LLM_EXTRA_FIELDS`.

Bedrock authentication uses SigV4 or `AWS_BEARER_TOKEN_BEDROCK`, never
`LLM_BINDING_API_KEY`; see
[RoleSpecificLLMConfiguration.md](./RoleSpecificLLMConfiguration.md#bedrock-authentication-rules).

## 7. Ollama embedding options

Prefix `OLLAMA_EMBEDDING_`. The field set is identical to §4 **minus `think`** —
`num_ctx`, `num_predict`, `num_keep`, `seed`, `temperature`, `top_k`, `top_p`, `tfs_z`,
`typical_p`, `min_p`, `repeat_last_n`, `repeat_penalty`, `presence_penalty`,
`frequency_penalty`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `numa`, `num_batch`,
`num_gpu`, `main_gpu`, `low_vram`, `num_thread`, `f16_kv`, `logits_all`, `vocab_only`,
`use_mmap`, `use_mlock`, `embedding_only`, `penalize_newline`, and `stop`.

In practice only the runtime knobs are meaningful for an embedding request:

| Environment variable | Type | Description |
| --- | --- | --- |
| `OLLAMA_EMBEDDING_NUM_CTX` | int | Context window of the embedding model. Ollama needs this set in addition to `EMBEDDING_TOKEN_LIMIT`; `env.example` ships `8192`. |
| `OLLAMA_EMBEDDING_NUM_GPU` / `_MAIN_GPU` / `_NUM_THREAD` / `_NUM_BATCH` / `_LOW_VRAM` / `_USE_MMAP` / `_USE_MLOCK` / `_NUMA` / `_F16_KV` | int / bool | Same meaning as their `OLLAMA_LLM_*` counterparts in §4.5. |

The sampling and repetition fields are accepted for symmetry with the LLM option set
but have no effect on an embedding response.

> Changing the embedding model or its effective dimension invalidates all stored
> vectors. See the embedding-model warning in
> [LightRAG Server and WebUI](./LightRAG-API-Server.md).

## 8. Gemini embedding options

Prefix `GEMINI_EMBEDDING_`.

| Environment variable | Type | Description |
| --- | --- | --- |
| `GEMINI_EMBEDDING_TASK_TYPE` | str | Task type for embedding optimization. If unset, it is derived from context (`RETRIEVAL_QUERY` for queries, `RETRIEVAL_DOCUMENT` for documents). Supported values: `RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`, `CODE_RETRIEVAL_QUERY`, `QUESTION_ANSWERING`, `FACT_VERIFICATION`. |

Pinning a single `task_type` disables the query/document distinction that asymmetric
embedding relies on; see [Asymmetric Embedding Configuration](./AsymmetricEmbedding.md).

## 9. Where the values end up

| Binding | Destination | Silently dropped |
| --- | --- | --- |
| `openai`, `azure_openai` | Merged into the Chat Completions request as keyword arguments. | Nothing — unsupported fields reach the provider and become provider errors. |
| `ollama` | The request's `options` payload, plus top-level `think`. | Nothing. |
| `gemini` | `GenerateContentConfig(**options)`. | Entries whose value is `None` or `""`. |
| `bedrock` | `inferenceConfig` (`temperature`, `maxTokens`, `topP`, `stopSequences`) plus `additionalModelRequestFields` from `extra_fields`. | Any field outside that set. |
| `lollms` | Top-level request fields. | Every field except `temperature`, `top_k`, `top_p`, `repeat_penalty`, `repeat_last_n`, `seed`. |

## 10. Provider options and the LLM cache

The LLM cache key is partitioned by a non-secret **identity** consisting of the role,
binding, model, and host. `api_key` and provider options are deliberately excluded, so
cache keys stay safe to persist. The practical consequence:

- Changing `LLM_MODEL`, `LLM_BINDING`, or `LLM_BINDING_HOST` produces new cache keys, so
  new calls are made.
- Changing a provider option — `temperature`, `think`, `reasoning_effort`,
  `max_tokens`, … — does **not**. A cached extraction or query result is still served
  with the old value.

When you need a tuning change to take effect on content that has already been
processed, clear the relevant cache explicitly (`/documents/clear_cache`, or
`ENABLE_LLM_CACHE=false` while experimenting on queries). Note that clearing the LLM
cache drops the extraction cache too, which is what entity/relation rebuild after a
document delete relies on.

## 11. Common tasks

### Cap output length (prevents endless extraction output)

Set an output cap so a runaway response is truncated before the request times out. A
usable ceiling is `LLM_TIMEOUT * output_tokens_per_second` (e.g. `240s * 50 tok/s`, so
stay under 12000).

```env
# OpenAI-compatible servers (vLLM/SGLang/most gateways)
OPENAI_LLM_MAX_TOKENS=9000
# OpenAI reasoning-capable models
OPENAI_LLM_MAX_COMPLETION_TOKENS=9000
# Ollama
OLLAMA_LLM_NUM_PREDICT=9000
# Gemini
GEMINI_LLM_MAX_OUTPUT_TOKENS=9000
# Bedrock
BEDROCK_LLM_MAX_TOKENS=9000
```

### Turn thinking off for extraction and keyword generation

Pick the line that matches your provider — these are alternatives, not a block to
paste as a whole (a repeated variable would just keep its last value).

```env
# Ollama
EXTRACT_OLLAMA_LLM_THINK=false
KEYWORD_OLLAMA_LLM_THINK=false

# Gemini
EXTRACT_GEMINI_LLM_THINKING_CONFIG='{"thinking_budget": 0, "include_thoughts": false}'

# OpenAI reasoning-capable models
EXTRACT_OPENAI_LLM_REASONING_EFFORT=minimal

# OpenRouter (one EXTRA_BODY per role — choose one of the two forms below)
EXTRACT_OPENAI_LLM_EXTRA_BODY='{"reasoning": {"enabled": false}}'
# Qwen-style models served by vLLM
# EXTRACT_OPENAI_LLM_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'

# Bedrock — the field name and accepted values belong to the target model, not to
# LightRAG; check that model's Converse API documentation. Example shape:
# EXTRACT_BEDROCK_LLM_EXTRA_FIELDS='{"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}}'
```

### Keep a provider option out of one role

There is no "unset an inherited option" syntax. Either override the option for that
role with a value the role's model accepts, or keep the option role-level from the
start instead of setting it globally. A local OpenAI-compatible endpoint sharing the
`openai` binding with the official API is the usual case:

```env
LLM_BINDING=openai
LLM_MODEL=gpt-5-mini
# Do NOT set OPENAI_LLM_REASONING_EFFORT globally here — the local server rejects it.
QUERY_OPENAI_LLM_REASONING_EFFORT=medium
KEYWORD_OPENAI_LLM_MAX_TOKENS=2048
```

## 12. Inspecting the options yourself

The `--help` output lists the option group of the **currently selected** binding only,
so pass the binding you want to inspect:

```bash
lightrag-server --llm-binding openai --help
lightrag-server --llm-binding ollama --help
lightrag-server --llm-binding gemini --help
lightrag-server --llm-binding bedrock --help
lightrag-server --embedding-binding ollama --help
lightrag-server --embedding-binding gemini --help
```

To dump every binding's options at once as a commented `.env` block:

```bash
python -m lightrag.llm.binding_options
```

Both outputs are generated from `lightrag/llm/binding_options.py`, so they are always
current. Remember that the values shown in the generated `.env` sample are the
providers' typical defaults, not values LightRAG sends (§1.2) — the lines are commented
out for exactly that reason.
