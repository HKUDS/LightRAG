# Asymmetric Embedding Configuration

LightRAG keeps embedding behavior symmetric by default. Query/document asymmetric
embedding is enabled only when `EMBEDDING_ASYMMETRIC=true` is explicitly set.

This avoids accidental retrieval changes when prefix variables are present in an
environment but the user did not intentionally enable asymmetric embeddings.

Before enabling asymmetric embeddings for any model, check the model's current
model card or provider documentation. Do not infer the right behavior from the
API binding alone: an `openai`-compatible endpoint can serve instruction-free
models, prefix-based models, or provider-specific models behind the same API
shape.

## Reindexing Requirement

Changing asymmetric embedding settings changes the vectors produced for stored
documents and for future queries. After enabling, disabling, or changing any of
these settings, clear the existing LightRAG data for the workspace and re-index
the source files:

- `EMBEDDING_ASYMMETRIC`
- `EMBEDDING_QUERY_PREFIX`
- `EMBEDDING_DOCUMENT_PREFIX`
- Provider task behavior such as Jina `task`, Gemini `task_type`, or VoyageAI
  `input_type`

Do not reuse an existing vector store across asymmetric embedding configuration
changes. Mixing vectors generated with different query/document behavior can
make retrieval quality unpredictable.

## Binding Types

LightRAG distinguishes two asymmetric embedding styles:

| Style | Bindings | How asymmetric behavior is applied |
| --- | --- | --- |
| Provider task parameters | `jina`, `gemini`, `voyageai` | LightRAG passes query/document context to the provider-specific `task`, `task_type`, or `input_type` parameter. |
| Text task prefixes | `openai`, `azure_openai`, `ollama` | LightRAG prepends configured text prefixes before calling the embedding API. Use this only when the model card explicitly requires separate query/document prefixes. |

Other server embedding bindings do not currently support
`EMBEDDING_ASYMMETRIC=true`.

## Default: Symmetric Embeddings

When `EMBEDDING_ASYMMETRIC` is unset, LightRAG does not enable asymmetric
embedding behavior, even if prefix variables exist:

```env
# EMBEDDING_ASYMMETRIC is unset
# EMBEDDING_QUERY_PREFIX="search_query: "
# EMBEDDING_DOCUMENT_PREFIX="search_document: "
```

The prefixes are ignored and a warning is logged.

The same is true when the flag is explicitly false:

```env
EMBEDDING_ASYMMETRIC=false
```

## Instruction-Free Models: Keep Symmetric

Some embedding models are instruction-free, sometimes described as using
implicit intent. They are trained to handle query/document matching from the raw
text itself and do not require query/document prefixes or provider task
parameters. For these models, do not set `EMBEDDING_ASYMMETRIC=true`; leave it
unset or set it to `false`, and do not configure `EMBEDDING_QUERY_PREFIX` or
`EMBEDDING_DOCUMENT_PREFIX`.

Common examples that should normally stay in symmetric mode:

| Model family | Example model IDs | Notes |
| --- | --- | --- |
| BGE-M3 | `BAAI/bge-m3` | Use plain text input. Do not add `search_query:` / `search_document:` unless the specific serving wrapper's model card says otherwise. |
| OpenAI Text Embedding 3 | `text-embedding-3-small`, `text-embedding-3-large` | The OpenAI embeddings API uses text input plus the model name; it does not expose a query/document task parameter. |
| Mistral Embed | `mistral-embed` | Use the provider's plain embedding input. Do not invent task prefixes. |
| Alibaba GTE base models | `gte-large`, `gte-large-zh` | Base GTE models use plain text for normal retrieval. This does not apply to newer `instruct` variants such as `gte-Qwen2-1.5B-instruct`; check that model card. |
| Jina Embeddings v2 | `jina-embeddings-v2-base-en`, `jina-embeddings-v2-base-zh` | Jina v2 is plain-text input. Jina v3/v4 are different and use the `task` parameter for retrieval tasks. |

If a model is instruction-free, enabling LightRAG's asymmetric mode can make the
input different from what the model was trained or documented to expect. That can
reduce retrieval quality even though the server starts successfully.

## Provider Task Parameter Bindings

Use this mode for providers that expose separate query/document embedding tasks.
Do not configure prefix variables for these bindings.

Jina example:

```env
EMBEDDING_BINDING=jina
EMBEDDING_ASYMMETRIC=true
EMBEDDING_MODEL=jina-embeddings-v4
```

Gemini example:

```env
EMBEDDING_BINDING=gemini
EMBEDDING_ASYMMETRIC=true
EMBEDDING_MODEL=gemini-embedding-001
```

VoyageAI example:

```env
EMBEDDING_BINDING=voyageai
EMBEDDING_ASYMMETRIC=true
EMBEDDING_MODEL=voyage-3
```

If `EMBEDDING_QUERY_PREFIX` or `EMBEDDING_DOCUMENT_PREFIX` is also configured
for these bindings, LightRAG logs a warning and ignores the prefixes.

## Text Task Prefix Bindings

Use this mode for embedding models that expect task instructions in the input
text, such as models whose card documents prefixes like `search_query:`,
`search_document:`, `query:`, or `passage:`. Do not enable this mode just
because the model is served through `openai`, `azure_openai`, or `ollama`.

Both prefix variables must be explicitly configured:

```env
EMBEDDING_ASYMMETRIC=true
EMBEDDING_QUERY_PREFIX="search_query: "
EMBEDDING_DOCUMENT_PREFIX="search_document: "
```

If one side should intentionally have no prefix, use the sentinel `NO_PREFIX`:

```env
EMBEDDING_ASYMMETRIC=true
EMBEDDING_QUERY_PREFIX="search_query: "
EMBEDDING_DOCUMENT_PREFIX=NO_PREFIX
```

`NO_PREFIX` is converted to an empty string internally. It is different from an
unset variable: it means the side was reviewed and intentionally left without a
prefix.

At least one side must have a non-empty prefix. This is invalid:

```env
EMBEDDING_ASYMMETRIC=true
EMBEDDING_QUERY_PREFIX=NO_PREFIX
EMBEDDING_DOCUMENT_PREFIX=NO_PREFIX
```

## Invalid Empty Prefixes

Do not use an empty environment value for an intentional empty prefix:

```env
EMBEDDING_DOCUMENT_PREFIX=
```

Use `NO_PREFIX` instead. Empty values are rejected because shell, `.env`, and
Docker Compose handling can make empty strings indistinguishable from accidental
missing configuration.

## Validation Summary

| Configuration | Result |
| --- | --- |
| `EMBEDDING_ASYMMETRIC` unset | Symmetric mode; prefixes ignored with a warning. |
| `EMBEDDING_ASYMMETRIC=false` | Symmetric mode; prefixes ignored with a warning. |
| Instruction-free model such as `BAAI/bge-m3`, `text-embedding-3-small`, `mistral-embed`, base GTE, or Jina v2 | Keep symmetric mode; do not configure prefixes or provider tasks unless the model card says to. |
| `EMBEDDING_ASYMMETRIC=true` with `jina`/`gemini`/`voyageai` | Provider task mode; prefixes ignored with a warning. |
| `EMBEDDING_ASYMMETRIC=true` with `openai`/`azure_openai`/`ollama` and both prefix variables configured | Prefix mode. |
| Prefix mode with a missing prefix variable | Startup error; use a real prefix or `NO_PREFIX`. |
| Prefix mode with both sides `NO_PREFIX` | Startup error; no asymmetric behavior would occur. |
| Prefix variable set to an empty value | Startup error; use `NO_PREFIX`. |

Any valid change from one asymmetric embedding configuration to another still
requires clearing the workspace data and re-indexing the source files.
