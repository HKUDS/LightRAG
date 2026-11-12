# Exploring LightRAG: From the Default Setup to a Production-Ready Configuration

This guide picks up where the README's installation section ends: the server runs with the default `.env`, a few documents are indexed, and queries work. It then switches on, one at a time, the capabilities that the default configuration leaves off:

| Step | Capability | What it gives you | What it needs |
|---|---|---|---|
| 1 | [PostgreSQL storage](#step-1--postgresql-storage) | Storage you can run in production | A PostgreSQL with pgvector + Apache AGE (a container is fine) |
| 2 | [Authentication](#step-2--authentication) | Login for the WebUI, API keys for clients | Nothing |
| 3 | [Role-specific LLMs](#step-3--role-specific-llms) | A stronger model for answers, a cheaper one for extraction | Nothing beyond your LLM provider |
| 4 | [Reranking](#step-4--reranking) | Better retrieval precision | A rerank endpoint (Cohere-, Jina- or Aliyun-compatible) |
| 5 | [Image analysis (VLM)](#step-5--image-analysis-with-a-vlm) | Images in Markdown / Word documents become searchable knowledge | A vision-capable model |
| 6 | [Word smart headings](#step-6--smart-heading-detection-for-word-documents) | Correct section structure for `.docx` files without heading styles | The pinned spaCy models |
| 7 | [MinerU](#step-7--mineru-for-pdf-and-office-files) | High-quality parsing of PDF, PPTX, XLSX and scanned documents | A MinerU API token, or a self-hosted MinerU with a GPU |
| 8 | [Concurrency tuning](#step-8--concurrency-tuning) | Several times faster ingestion | Nothing |

Every step below was carried out, in this order, on a real deployment: a 2-vCPU / 4 GB cloud VM without a GPU, OpenAI models (`gpt-6-luna` for extraction, `gpt-6-sol` for answers, `text-embedding-3-large`), and a test corpus of seven documents (Markdown, a `.textpack`, a PDF and a Chinese `.docx`). The timings quoted come from that machine.

## Before You Start

**Why this order.** Storage comes first. Image analysis, smart headings and MinerU all act at upload time and only on documents uploaded after you enable them, so documents you upload to test them should already land in the storage you will keep. Authentication is next because it is cheap and you will want it before anyone else can reach the server. Concurrency comes last, once you can measure a realistic ingest.

**The loop for every step:**

1. Edit `.env`.
2. Restart `lightrag-server`. `.env` is read only at startup.
3. Read the startup log. It prints the storage backends, every LLM role with its model and options, and any configuration it rejects.
4. Verify the capability with a document or a query, as each step shows.

**Settings frozen at upload time.** A document's parser engine, processing options (`i`, `t`, `e`, chunking) and chunk parameters are recorded when it is enqueued. Editing `.env` later does not change already-indexed documents. To apply new settings to a document, delete it and upload it again. Re-processing an unchanged document is cheap: the extraction cache (`ENABLE_LLM_CACHE_FOR_EXTRACT`, on by default) answers the LLM calls it already made.

**Keep a small test corpus** that covers the formats you care about: a Markdown file with images, a Word file, a PDF. Re-uploading it after each step shows the effect directly.

## Step 1 — PostgreSQL Storage

The default storages (`JsonKVStorage`, `NanoVectorDBStorage`, `NetworkXStorage`, `JsonDocStatusStorage`) keep everything in memory and rewrite whole files. They are meant for small-scale testing only. PostgreSQL can hold all four storage types in one database: key-value and document status in tables, vectors with pgvector, and the graph with Apache AGE.

### 1.1 Generate the PostgreSQL Service with the Setup Wizard

```bash
make env-storage
```

Choose `PGKVStorage`, `PGVectorStorage`, `PGGraphStorage` and `PGDocStatusStorage` for the four storage types, let the wizard run PostgreSQL locally via Docker, and enter the user, password and database name. The wizard then:

- writes the storage classes and the `POSTGRES_*` connection settings into `.env`, after backing up the previous file;
- adds a `postgres` service to `docker-compose.final.yml`. Its image, `gzdaniel/postgres-for-rag:pg18-age-pgvector`, ships PostgreSQL 18 with pgvector and Apache AGE.

`.env` stays usable from the host (`POSTGRES_HOST=localhost`); the generated compose file points its containerized `lightrag` service at `postgres` instead. See [InteractiveSetup.md](./InteractiveSetup.md) for every question the wizard asks.

Start the database:

```bash
# LightRAG Server runs on the host (source or PyPI install): start only the database
docker compose -f docker-compose.final.yml up -d postgres

# LightRAG Server runs in Docker as well: start the whole stack
docker compose -f docker-compose.final.yml up -d
```

Check that both extensions are available:

```bash
docker compose -f docker-compose.final.yml exec postgres psql -U <user> -d <database> -tAc \
  "select name, default_version from pg_available_extensions where name in ('vector','age')"
```

The service template publishes port 5432 on all interfaces. On a host reachable from the internet, use a strong password and restrict the port with your firewall.

### 1.2 Check the Vector Index Type

The wizard does not ask for `POSTGRES_VECTOR_INDEX_TYPE`, so check it in `.env`:

```bash
POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC
```

**Pick the vector index type from your embedding dimension.** pgvector cannot build an `HNSW` or `IVFFlat` index above 2000 dimensions. `HNSW_HALFVEC` goes up to 4000 and needs pgvector 0.7 or later, which the wizard's image has. The default `text-embedding-3-large` has 3072 dimensions, so it needs `HNSW_HALFVEC`. With `HNSW`, index creation fails. Releases without the dimension check log an `ERROR` and carry on with no vector index (every query then scans the table); releases with it refuse to start and name the fix. Choose it before the first upload: switching a table that already holds vectors to `HNSW_HALFVEC` converts them in place to 16-bit floats, which cannot be undone.

After restarting the server, the startup log lists the four `PG*` storages. The tables appear on first start, and the vector tables carry the embedding model and dimension in their name, for example `lightrag_vdb_chunks_text_embedding_3_large_3072d`.

### 1.3 Bring Over Existing Documents

Switching storage does not move data. The cheapest way to carry over what you already indexed is to migrate the extraction cache and then upload the documents again:

1. With PostgreSQL running (1.1) and before restarting the server, run `python -m lightrag.tools.migrate_llm_cache`. Choose `JsonKVStorage` as source and `PGKVStorage` as target, and confirm. It moves the extraction, summary and multimodal-analysis caches (see [README_MIGRATE_LLM_CACHE.md](../lightrag/tools/README_MIGRATE_LLM_CACHE.md)).
2. Restart the server on PostgreSQL.
3. Upload the original files again. Extraction hits the migrated cache.

On the test deployment, 139 cache records moved. Re-uploading the six documents took 60 seconds and made 17 new LLM calls; all other extraction came from the cache. The new store held exactly the same 56 chunks as the old one.

## Step 2 — Authentication

With neither `AUTH_ACCOUNTS` nor `LIGHTRAG_API_KEY` set, every endpoint is public. Configure both: accounts for people using the WebUI, a key for scripts and API clients.

```bash
AUTH_ACCOUNTS='admin:{bcrypt}$2b$12$...'   # single quotes: the hash contains $
TOKEN_SECRET=<openssl rand -hex 32>         # required with AUTH_ACCOUNTS
LIGHTRAG_API_KEY=<a long random string>
WHITELIST_PATHS=/health                     # also protect the Ollama-compatible /api/*
```

Store a bcrypt hash rather than a plain password. Generate one with the `bcrypt` package already in the environment:

```bash
python -c "import bcrypt, getpass; print(bcrypt.hashpw(getpass.getpass().encode(), bcrypt.gensalt(12)).decode())"
```

The server refuses to start with `AUTH_ACCOUNTS` but no non-default `TOKEN_SECRET`. The default `WHITELIST_PATHS` keeps `/api/*` open so Ollama clients work unchanged. Narrow it to `/health` once the server is reachable from a network.

**Verify:**

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://HOST:9621/documents/status_counts               # 401
curl -s -o /dev/null -w '%{http_code}\n' http://HOST:9621/api/tags                              # 401
curl -s -o /dev/null -w '%{http_code}\n' http://HOST:9621/health                                # 200
curl -s -o /dev/null -w '%{http_code}\n' -H "X-API-Key: $KEY" http://HOST:9621/documents/status_counts  # 200
curl -s -X POST http://HOST:9621/login -d 'username=admin&password=...'                         # access_token
```

## Step 3 — Role-Specific LLMs

LightRAG calls the LLM in four roles: `EXTRACT` (entities and relations from every chunk), `KEYWORD` (query keywords), `QUERY` (the final answer) and `VLM` (images). They all default to `LLM_MODEL`. Extraction runs once per chunk, so it should be fast and cheap. The answer is written from long, noisy context, so it benefits from a stronger model.

```bash
# Base model: used by EXTRACT, KEYWORD and VLM
LLM_MODEL=gpt-6-luna
OPENAI_LLM_REASONING_EFFORT=low

# Answers: one tier up, more reasoning
QUERY_LLM_MODEL=gpt-6-sol
QUERY_OPENAI_LLM_REASONING_EFFORT=medium

# Keywords: latency-sensitive, no reasoning
KEYWORD_OPENAI_LLM_REASONING_EFFORT=none
```

A role inherits the base binding, host, API key and options unless you override them with its prefix (`QUERY_LLM_BINDING_HOST`, `QUERY_LLM_BINDING_API_KEY`, ...), so a role can also point at another provider. Provider options take the role prefix too: `<ROLE>_OPENAI_LLM_REASONING_EFFORT`. See [RoleSpecificLLMConfiguration.md](./RoleSpecificLLMConfiguration.md).

**Verify** in the startup log, which prints every role:

```
Role LLM Configuration (initialized):
 - extract: openai/gpt-6-luna, host=https://api.openai.com/v1, max_async=4, timeout=240
 - keyword: openai/gpt-6-luna, ...
 - query: openai/gpt-6-sol, ...
 - vlm: openai/gpt-6-luna, ...
Role LLM Option:
 - extract: OpenAI {'reasoning_effort': 'low'}
 - keyword: OpenAI {'reasoning_effort': 'none'}
 - query: OpenAI {'reasoning_effort': 'medium'}
```

**Also consider `SUMMARY_LANGUAGE`.** It sets the language of extracted entity and relation descriptions (default `English`). Set it to your documents' main language, for example `Chinese`, before indexing. Changing it later changes the extraction prompt, so documents re-processed afterwards miss the cache.

## Step 4 — Reranking

Vector search returns chunks that are *similar* to the query, which is not the same as chunks that *answer* it. A reranker scores every candidate chunk against the query and keeps the best ones. It matters most in `mix` and `naive` modes, where text chunks carry the answer.

```bash
RERANK_BINDING=cohere                          # cohere, jina or aliyun
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=https://your-host/rerank   # full endpoint URL
RERANK_BINDING_API_KEY=...
# MIN_RERANK_SCORE=0.0       # drop chunks scoring below this
# RERANK_BY_DEFAULT=True     # queries rerank unless they set enable_rerank=false
```

Choose the binding by the API format, not the vendor: a model served with vLLM, or any service with a Cohere-style `/rerank` endpoint, uses `cohere`. OpenAI offers no rerank API. Check the endpoint first:

```bash
curl -s https://your-host/rerank -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-reranker-v2-m3","query":"什么是知识图谱","documents":["知识图谱是用图结构表示实体及其关系的知识库。","今天天气很好。"]}'
```

The startup log confirms `Reranking is enabled: BAAI/bge-reranker-v2-m3 using cohere provider`. `MAX_ASYNC_RERANK` falls back to `MAX_ASYNC_LLM` and `RERANK_TIMEOUT` defaults to 30 s.

**Verify** by comparing the same query with `enable_rerank` off and on through `POST /query/data`, which returns the retrieved context without generating an answer. On the test corpus, for "LazyGraphRAG 的索引成本和 GraphRAG 相比如何？" (`mix`, `chunk_top_k=5`):

| | Top-5 chunks from the LazyGraphRAG document | Latency |
|---|---|---|
| Rerank off | 0 of 5 (all about GraphRAG and LightRAG in general) | 4.3 s |
| Rerank on | 4 of 5 | 4.7 s |

The log line `Successfully reranked: 5 chunks from 34 original chunks` shows the reranker choosing among the candidates.

## Step 5 — Image Analysis with a VLM

The built-in `native` parser extracts tables, equations and images from Markdown, `.textpack` and Word files. Tables (`t`) and equations (`e`) are analysed by the extraction LLM and are on in the default rules. Images (`i`) need a vision model and are off.

```bash
LIGHTRAG_PARSER=*:native-iteP,*:legacy-R   # add i to the native rule
VLM_PROCESS_ENABLE=true
# VLM_LLM_MODEL=...                        # optional: defaults to LLM_MODEL
```

The `VLM` role defaults to `LLM_MODEL`. `gpt-6-luna` accepts image input, so nothing else is needed here. For a text-only base model, set `VLM_LLM_MODEL` (and, if needed, `VLM_LLM_BINDING_HOST` / `VLM_LLM_BINDING_API_KEY`) to a vision model.

> [!IMPORTANT]
> Turn on `VLM_PROCESS_ENABLE` together with the `i` option. A document processed with `i` while the switch is off **fails** at its first qualifying image ("VLM analysis required but VLM role is not available") instead of skipping the image.

**Where images come from:**

| Source | Handled |
|---|---|
| Word `.docx` embedded images | Yes |
| Markdown base64 `data:` images | Yes |
| Markdown remote `http(s)` images | Downloaded (guarded against private addresses); a failed download keeps the link and logs a warning |
| Markdown **local** images such as `./doc.assets/a.png` | **Only inside a `.textpack`**, a zip of the `.md` plus its images. A plain `.md` upload does not carry the image files, so the parser logs `skipping image (file reference outside .textpack)` |

Editors such as Typora write a local image path that contains spaces as `![alt](./my doc.assets/a.png)`, without angle brackets. That is not standard Markdown, but LightRAG still reads it as an image when the path ends in an image extension (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.svg`). In any other case use the standard form `![alt](<./my doc.assets/a.png>)` or write the spaces as `%20`; otherwise it is not an image, so it stays as plain text and no warning is logged.

Images smaller than `VLM_MIN_IMAGE_PIXEL` (64 px on either side) are treated as decoration and skipped.

**Verify** by re-uploading a document with images. The log line for its sidecar counts what the parser found, and the analysis stage follows:

```
[sidecar] wrote 29 blocks for doc_id=... (6 tables, 6 drawings, 2 equations, assets=True, engine=native)
[analyze_multimodal] drawings=6, tables=6, equations=2 surrounding backfilled: doc-...
```

The image descriptions become part of the chunk text next to a `<drawing .../>` tag, so they take part in entity extraction and retrieval.

## Step 6 — Smart Heading Detection for Word Documents

Many Word documents, government notices in particular, mark headings with bold, larger text and numbering such as `一、` / `（一）` instead of Word's heading styles. The native parser reads headings from the Word outline, so such a document arrives as one undivided block, and paragraph chunking (`P`) has no structure to work with. `smart_heading` detects headings from layout, numbering and language cues, and asks the extraction LLM to judge the title block.

```bash
DOCX_SMART_HEADING=true
```

It needs the pinned spaCy models: `lightrag-download-cache --spacy-install` (see the README). With the switch on, the server checks for them at startup and refuses to start without them. To enable it for one file only, upload it with a filename hint: `report.[native(smart_heading)].docx`.

> [!NOTE]
> **Short documents are skipped.** Below `DOCX_SMART_MIN_TOKENS` (default 1800) a document is left to the Word outline, since a document that fits in about one chunk needs no heading split. A short test document therefore looks exactly as if the switch did nothing. Test with a realistic document, or lower `DOCX_SMART_MIN_TOKENS` for short notices.

**Verify.** The decisions are recorded in `<name>.smart_audit.json` inside the document's `.parsed/` directory under `INPUT_DIR/__parsed__/`. A skip is recorded there as `smart_skipped_short_document`. The doc-status record shows `parse_engine: native(smart_heading=true)`.

On the test document (53 paragraphs, about 2400 tokens, headings as bold `Normal` paragraphs), the plain parse produced a single block titled `Preface/Uncategorized`. With `smart_heading` it produced the title (level 0), four `一、` headings (level 1) and fifteen `（一）` headings (level 2), using one LLM call, and the document split into four section-aligned chunks.

## Step 7 — MinerU for PDF and Office Files

The `native` parser covers Markdown, `.textpack` and Word. Everything else (PDF, PPTX, XLSX, images) falls through to `legacy`, which extracts plain text only: no tables, no images, no layout. [MinerU](https://github.com/opendatalab/MinerU) does layout analysis, OCR, table and formula recognition, and returns the same structured sidecar the native parser produces, so the `i` / `t` / `e` analysis and `P` chunking work on its output too.

MinerU can run in two ways:

- **`official`**: the hosted precision API at mineru.net. Create an API token on the site's API management page. Files are uploaded to MinerU's cloud, and the service has page and size limits (200 MB and 200 pages per file at the time of writing) and a daily quota.
- **`local`**: a self-hosted `mineru-api`. It needs a GPU; see [ParserServiceDeployment.md](./ParserServiceDeployment.md). The test machine has no GPU, so this guide uses `official`.

```bash
MINERU_API_MODE=official
MINERU_API_TOKEN=sk-...
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R
# MAX_PARALLEL_PARSE_MINERU=1     # files parsed by MinerU at the same time
```

Rules are tried left to right per file: a format `native` supports goes to `native`, the formats MinerU supports go to `mineru`, and anything left goes to `legacy`. When `LIGHTRAG_PARSER` names `mineru`, configure its mode and token (or local endpoint) before starting the server.

**Check the token before you blame LightRAG.** If parsing fails with `MinerU official upload URL request failed: HTTP 401 ... "msgCode": "A0202"`, MinerU rejected the token ("Token 错误" in its error table; `A0211` means expired). Call the API directly with the value from `.env`:

```bash
curl -s -X POST https://mineru.net/api/v4/file-urls/batch \
  -H "Authorization: Bearer $(grep ^MINERU_API_TOKEN= .env | cut -d= -f2-)" \
  -H 'Content-Type: application/json' -d '{"files":[{"name":"probe.pdf"}],"model_version":"vlm"}'
```

`"code":0` means the token works. On the test deployment the first attempt failed with A0202 because `.env` held a different, shorter string than the `sk-...` key the site shows.

**Retry the failed document.** A document that failed during parsing stays `FAILED` and is never retried automatically. After fixing the cause, request one retry per failed document with `POST /documents/reprocess_failed`. A retry keeps the document's original engine and options.

**Verify.** The log shows the engine and what MinerU found:

```
[File Extraction]Deferred HybridRAG ... .pdf to mineru parser
[sidecar] wrote 19 blocks for doc_id=... (5 tables, 2 drawings, 0 equations, assets=True, engine=mineru)
```

On the test PDF, a research paper, MinerU produced 19 structured blocks with 5 tables and 2 figures, all analysed by the extraction LLM and the VLM; the whole document finished in about 100 seconds. With `legacy` the same file had been plain text.

## Step 8 — Concurrency Tuning

The defaults are conservative. Most of an ingest is spent waiting on LLM and embedding calls, so raising the concurrency costs little on your own machine; the limit is usually your provider's rate limit.

| Setting | Controls | Default |
|---|---|---|
| `MAX_ASYNC_LLM` | Concurrent LLM requests per role, and chunk-extraction tasks per document | 4 |
| `MAX_PARALLEL_INSERT` | Documents processed at the same time; about `MAX_ASYNC_LLM / 3` | 3 |
| `EMBEDDING_FUNC_MAX_ASYNC` | Concurrent embedding requests | 8 |
| `VLM_MAX_ASYNC_LLM` | Concurrent image-analysis requests | 4 |
| `POSTGRES_MAX_CONNECTIONS` | PostgreSQL pool size; raise it with the others | 25 |

Measured on the test deployment by deleting every document and uploading the seven-document corpus again, with `ENABLE_LLM_CACHE_FOR_EXTRACT=false` so that every LLM call is real:

| Run | LLM / insert / embedding / VLM / PG pool | Wall time | Peak server memory |
|---|---|---|---|
| Defaults | 4 / 3 / 8 / 4 / 25 | 449 s | 839 MB |
| Tuned | 12 / 4 / 16 / 8 / 40 | 170 s | 866 MB |
| Final | 24 / 7 / 24 / 12 / 40 | 120 s | 876 MB |

There were no rate-limit errors and memory barely moved. In the final run all seven documents were processed at once, so the remaining time is the largest document; a bigger corpus keeps benefiting from `MAX_PARALLEL_INSERT`. The configuration kept:

```bash
MAX_ASYNC_LLM=24
MAX_PARALLEL_INSERT=8
EMBEDDING_FUNC_MAX_ASYNC=24
VLM_MAX_ASYNC_LLM=12
POSTGRES_MAX_CONNECTIONS=40
```

Increase step by step and watch the log for `429` / `RateLimitError`. Turn the extraction cache back on after benchmarking.

## The Result

With all eight steps applied, a full re-ingest of the seven-document corpus finished in 72 seconds with no errors. The extraction cache and MinerU's cached parse result were both warm. Every document landed on the intended engine:

| Document | Engine | Options | Chunks |
|---|---|---|---|
| HybridRAG paper (`.pdf`) | `mineru` | `iteP` | 16 |
| LightRAG paper (`.textpack`) | `native` | `iteP` | 30 |
| Four Markdown articles | `native` | `iteP` | 1–11 |
| Chinese implementation plan (`.docx`) | `native(smart_heading=true)` | `iteP` | 4 |

A question answered from the Word document, "企业知识图谱建设方案分哪几个实施阶段？每个阶段做什么？", came back in 9.5 seconds. The answer combined the schedule table (months per phase) with the section text (what each phase does) and cited the `.docx`.

The settings changed from `env.example`, all in one place:

```bash
# Step 1 — PostgreSQL
LIGHTRAG_KV_STORAGE=PGKVStorage
LIGHTRAG_DOC_STATUS_STORAGE=PGDocStatusStorage
LIGHTRAG_GRAPH_STORAGE=PGGraphStorage
LIGHTRAG_VECTOR_STORAGE=PGVectorStorage
POSTGRES_HOST=localhost
POSTGRES_USER=lightrag
POSTGRES_PASSWORD='...'
POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC
POSTGRES_MAX_CONNECTIONS=40

# Step 2 — Authentication
AUTH_ACCOUNTS='admin:{bcrypt}$2b$12$...'
TOKEN_SECRET=...
LIGHTRAG_API_KEY=...
WHITELIST_PATHS=/health

# Step 3 — Role-specific LLMs
LLM_MODEL=gpt-6-luna
OPENAI_LLM_REASONING_EFFORT=low
QUERY_LLM_MODEL=gpt-6-sol
QUERY_OPENAI_LLM_REASONING_EFFORT=medium
KEYWORD_OPENAI_LLM_REASONING_EFFORT=none

# Step 4 — Reranking
RERANK_BINDING=cohere
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=https://your-host/rerank
RERANK_BINDING_API_KEY=...

# Steps 5 and 7 — parser routing, VLM, MinerU
LIGHTRAG_PARSER=*:native-iteP,*:mineru-iteP,*:legacy-R
VLM_PROCESS_ENABLE=true
MINERU_API_MODE=official
MINERU_API_TOKEN=sk-...

# Step 6 — Word smart headings
DOCX_SMART_HEADING=true

# Step 8 — Concurrency
MAX_ASYNC_LLM=24
MAX_PARALLEL_INSERT=8
EMBEDDING_FUNC_MAX_ASYNC=24
VLM_MAX_ASYNC_LLM=12
```

## Pitfalls Seen Along the Way

| Symptom | Cause | Fix |
|---|---|---|
| Vector index creation fails, or the server refuses to start after switching to PostgreSQL | `HNSW` / `IVFFlat` with an embedding above 2000 dimensions | `POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC` |
| Images in a Markdown file are not analysed | Local image files are not uploaded with a plain `.md`; or the path contains spaces, does not end in an image extension and has no `<...>` | Upload a `.textpack`; write `![](<path with spaces>)` |
| A document with images ends up `FAILED` | `i` in its options while `VLM_PROCESS_ENABLE=false` | Enable both together |
| `DOCX_SMART_HEADING=true` changes nothing | The document is below `DOCX_SMART_MIN_TOKENS` | Check `smart_audit.json`; test with a longer document |
| MinerU parsing fails with HTTP 401 `A0202` | MinerU rejected the token | Test it with the `curl` in Step 7; then `POST /documents/reprocess_failed` |
| A failed document never comes back by itself | `FAILED` documents are only retried on request | `POST /documents/reprocess_failed` |
| A new setting has no effect on old documents | Parser and options are frozen at upload | Delete and upload again; the extraction cache keeps it cheap |
