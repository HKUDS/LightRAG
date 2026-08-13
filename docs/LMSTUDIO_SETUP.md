# LM Studio Setup Guide

Use this guide when running LightRAG against a local [LM Studio](https://lmstudio.ai/) server (OpenAI-compatible API on port 1234 by default).

## Fastest path: interactive setup wizard

The setup wizard applies LM Studio–specific fixes that are easy to get wrong when copying from `env.example`:

```bash
make env-base
```

Choose **lmstudio** as the LLM provider (embeddings default to LM Studio as well). When the wizard finishes:

1. **Does not set `EMBEDDING_DIM`** — LightRAG probes the vector size from your loaded embedding model at server startup.
2. **Prompts for single-threaded concurrency** — LM Studio often crashes under parallel load; the wizard explains this and asks whether to apply safe limits (`MAX_ASYNC_LLM=1`, `EMBEDDING_BATCH_NUM=16`, etc.). Default answer is **yes**.
3. **Defaults models** to `any-available` (resolved to the model loaded in LM Studio).

Re-run `make env-base` (not `env-storage` or `env-server`) to change LM Studio concurrency after you switch providers or want to re-answer the prompt.

## Before you start LightRAG

1. Start LM Studio and enable the **local server** (default `http://localhost:1234`).
2. Load a **chat** model and an **embedding** model (or one multimodal model if you only use chat).
3. Keep LM Studio running while `lightrag-server` is up.

## Recommended `.env` bindings

```bash
LLM_BINDING=lmstudio
LLM_BINDING_HOST=http://localhost:1234/v1
LLM_BINDING_API_KEY=lm-studio
LLM_MODEL=any-available

EMBEDDING_BINDING=lmstudio
EMBEDDING_BINDING_HOST=http://localhost:1234/v1
EMBEDDING_BINDING_API_KEY=lm-studio
EMBEDDING_MODEL=any-available
# EMBEDDING_DIM unset — auto-probed at startup (do not copy a value from env.example)

KEYWORD_LLM_MODEL=any-available
QUERY_LLM_MODEL=any-available
```

Leave `EMBEDDING_DIM` unset or commented out unless you intentionally override auto-probe and know the exact dimension of your embedding model.

## Why concurrency is limited

LM Studio loads one (or few) models in process. Parallel LLM or embedding requests often trigger:

- `The model has crashed without additional information`
- Timeouts during entity extraction
- Embedding batch size mismatches

The wizard’s defaults trade speed for stability. After you have a stable setup on a larger GPU, you can raise limits gradually (see tuning below).

## Changing embedding models

If you load a different embedding model in LM Studio:

1. Stop LightRAG.
2. Clear vector storage for your workspace (wrong dimensions or mixed embeddings break retrieval):
   ```bash
   rm -rf rag_storage/*
   ```
3. Restart LM Studio (reload the new embedding model).
4. Restart `lightrag-server` — dimension and resolved model id are picked up at startup.

## Manual tuning (after stable runs)

Increase one setting at a time and watch LM Studio logs:

| Setting | Wizard default | Notes |
|--------|----------------|-------|
| `MAX_ASYNC_LLM` | `1` | Global LLM cap; EXTRACT role uses this when not overridden |
| `KEYWORD_MAX_ASYNC_LLM` | `1` | Keyword-extraction stage |
| `QUERY_MAX_ASYNC_LLM` | `1` | Query / answer generation |
| `MAX_PARALLEL_INSERT` | `1` | Parallel documents in the ingestion pipeline |
| `EMBEDDING_FUNC_MAX_ASYNC` | `1` | Parallel embedding API calls |
| `EMBEDDING_BATCH_NUM` | `16` | Chunks per embedding request; lower if you see vector count errors |

## Start the server

```bash
uv sync --extra api
uv run lightrag-server
```

Or, with an existing venv: `.venv/bin/lightrag-server`

## Troubleshooting

| Symptom | Likely cause | Fix |
|--------|----------------|-----|
| Embedding dimension mismatch | Stale `EMBEDDING_DIM` in `.env` or old vectors | Run `make env-base` with lmstudio, clear `rag_storage/*`, restart |
| Model crash under ingest | Too much parallelism | Re-run wizard or set concurrency defaults from this doc |
| Fewer vectors than chunks | Large `EMBEDDING_BATCH_NUM` | Set `EMBEDDING_BATCH_NUM=16` or lower |
| `any-available` not resolving | No model loaded in LM Studio | Load chat/embedding model in LM Studio UI |

## Related docs

- [LightRAG API Server](LightRAG-API-Server.md) — full server configuration
- [Interactive Setup](InteractiveSetup.md) — wizard overview (`make env-base`)
