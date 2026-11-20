# Hybrid Retrieval Benchmark Harness

Reproducible benchmark for the PostgreSQL hybrid retrieval path
(`enable_hybrid`, RRF k=60, `content_tsv`/GIN). Extends this evaluation
directory; run it as a module, never by importing pieces ad hoc:

```bash
python -m lightrag.evaluation.hybrid_bench --dataset-dir DIR --output-dir DIR [knobs]
```

The harness ingests a normalized dataset into LightRAG, runs the retrieval
arms, scores metrics with fixed-seed bootstrap confidence intervals, measures
latency / write amplification / backfill / reranker survival, and PRINTS gate
verdicts against the proposal thresholds. **It never adjusts a threshold.**
Calibration is a separate, user-confirmed decision.

## Normalized dataset contract (fixed)

One directory per dataset; a `--dataset-dir` holding several such
subdirectories runs them all and pools gates per domain:

```
<name>/corpus.jsonl    {"id": str, "text": str} per line
<name>/queries.jsonl   {"id": str, "text": str} per line
<name>/qrels.tsv       query_id<TAB>corpus_id<TAB>score   (int; >=1 relevant, 0 loads but is not positive)
<name>/meta.json       {"domain": "terminology"|"general", "language": "zh"|"en",
                        "source": URL, "license": str, "subset_protocol": str, "counts": {...}}
```

Malformed rows, duplicate ids, unknown qrels references and negative scores
raise `DatasetError` naming the file and line; nothing is silently repaired.
Passage ids become LightRAG document ids (`ids=corpus_id`), so chunk
`full_doc_id` attribution maps retrieval back to qrels corpus ids.

## CLI surface

| flag | default | meaning |
| --- | --- | --- |
| `--dataset-dir` / `--output-dir` | required | dataset root; report + scratch working dir |
| `--storage` | `pg` | `pg` = PGKV/PGVector/PGTableGraph/PGDocStatus; `file` = defaults (lexical arm, write-amp, backfill then report skipped) |
| `--embedding-backend` | `ollama` | `openai` = any OpenAI-compatible embedding API (see below); `fake-hash` = deterministic smoke projection, labels the report SYNTHETIC |
| `--embedding-host` / `--embedding-model` / `--embedding-dim` | unset / `bge-m3:latest` / `1024` | embedding endpoint; unset = local ollama URL for `ollama`, `EMBEDDING_BINDING_HOST` env for `openai`; dim is requested explicitly on the openai path |
| `--llm-backend` | `ollama` | `openai` = any OpenAI-compatible chat API (see below); `fake` = corpus-aware FakeLLM (smoke only) |
| `--llm-host` / `--llm-model` | unset / `qwen2.5:3b` | answer + extraction LLM, identical across arms; unset host = local ollama URL for `ollama`, `LLM_BINDING_HOST` env for `openai` |
| `--llm-enable-thinking` / `--no-llm-enable-thinking` | unset | openai backend only: forwards `enable_thinking` via the SDK `extra_body` passthrough; disable it for thinking models used in extraction; unset sends nothing |
| `--max-parallel-insert` | `8` | documents processed in parallel (LightRAG `max_parallel_insert`) |
| `--max-async-llm` | `16` | concurrent LLM requests (LightRAG `llm_model_max_async`) |
| `--embedding-func-max-async` | `16` | concurrent embedding requests (LightRAG `embedding_func_max_async`) |
| `--embedding-batch-num` | `32` | texts per embedding request (LightRAG `embedding_batch_num`) |
| `--extraction-cap` | `0` | KG extraction subset size; 0 = chunks-only ingest, no extraction LLM |
| `--max-gleaning` | product default | override LightRAG `entity_extract_max_gleaning` (0 = single extraction pass, no gleaning); unset keeps the product default incl. `MAX_GLEANING` env |
| `--llm-num-predict` | uncapped | cap output tokens per LLM call (ollama `options.num_predict`; openai `max_tokens`); bounds rambling generations that would otherwise hit the request timeout and fail documents |
| `--ingest-timeout` | `86400` | bounded wait for the WHOLE dataset ingest in seconds, including recovery rounds; >=1 enforced |
| `--ingest-recovery-rounds` | `2` | sanctioned manual-retry rounds for FAILED extraction documents (one attempt per document per round, same parameters); 0 disables recovery |
| `--top-k` | `10` | same across all arms; metric labels follow it |
| `--seed` / `--bootstrap-resamples` | `42` / `1000` | fixed-seed per-query bootstrap, >=1000 enforced |
| `--latency-queries` / `--latency-rounds` | `100` / `3` | >=100 slots x >=3 rounds enforced; queries cycle deterministically when fewer exist |
| `--rerank-model` / `--rerank-min-score` | `BAAI/bge-reranker-v2-m3` / `0.5` | CrossEncoder used by the survival measurement, lazy import |
| `--phases` | all | subset of `ablation,latency,write-amp,backfill,rerank,report`; ingest + arms + scoring are the always-on core |
| `--write-amp-baseline` / `--baseline-ref` | off / `origin/main` | run the baseline half of the write-amplification comparison in a temporary detached worktree |
| `--cosine-threshold` / `--chunk-token-size` | `0.2` / `8192` | retrieval threshold; high chunk size keeps a passage one chunk |
| `--workspace-prefix` / `--working-dir` / `--keep-workspace` | `hbench` / `<output>/working` / off | scratch isolation; PG rows are cleaned unless kept |
| `--workspace` | unset | stable workspace base for resumable runs: per-dataset workspace = `<workspace>_<dataset>` (no random suffix), implies `--keep-workspace`; see the resume contract below |
| `--request-timeout` | `300` | bounded wait per retrieval/HTTP call; also the openai binding request timeout |

PG connection uses the standard environment recipe, exported before the run
(the harness reads no repo `.env`):

```bash
export POSTGRES_HOST=localhost POSTGRES_PORT=5432 POSTGRES_USER=lightrag \
       POSTGRES_PASSWORD=lightrag_smoke POSTGRES_DATABASE=lightrag_test
# leave POSTGRES_WORKSPACE unset
```

## OpenAI-compatible API backend

`--llm-backend openai --embedding-backend openai` runs the bench against any
OpenAI-compatible chat + embedding API (a cloud MaaS gateway, a local proxy,
or OpenAI itself) through the product's `lightrag.llm.openai` binding — the
same retry/rate-limit handling as the server, with the harness never
swallowing a provider error. Endpoints and credentials are runtime values
only; nothing is committed:

```bash
export LLM_BINDING_HOST=https://your-gateway.example/v1
export LLM_BINDING_API_KEY=sk-...          # or OPENAI_API_KEY
export EMBEDDING_BINDING_HOST=https://your-gateway.example/v1
export EMBEDDING_BINDING_API_KEY=sk-...    # or OPENAI_API_KEY
python -m lightrag.evaluation.hybrid_bench \
  --dataset-dir DIR --output-dir DIR \
  --llm-backend openai --llm-model YOUR_CHAT_MODEL \
  --no-llm-enable-thinking --llm-num-predict 4096 \
  --embedding-backend openai --embedding-model YOUR_EMBEDDING_MODEL \
  --embedding-dim 1024 --workspace cloud_run1
```

- Host precedence: `--llm-host` / `--embedding-host` flag, then
  `LLM_BINDING_HOST` / `EMBEDDING_BINDING_HOST`, then the binding's own
  default (`OPENAI_API_BASE` env or the official OpenAI endpoint).
- Key precedence: `LLM_BINDING_API_KEY` / `EMBEDDING_BINDING_API_KEY`, then
  `OPENAI_API_KEY`.
- `--no-llm-enable-thinking` sends `extra_body={"enable_thinking": false}`,
  the SDK's generic passthrough for provider-specific body fields: thinking
  models burn output budget and wall time on reasoning traces during
  extraction, so disable them there. `--llm-enable-thinking` sends true;
  unset sends nothing.
- `--llm-num-predict` maps to `max_tokens` on this path.
- `--embedding-dim` is requested explicitly (`dimensions` parameter) and
  must match the model's configured output dimension; the vector tables are
  named after model + dim, so changing either means a fresh workspace.

**Concurrency defaults.** The harness always passes `max_parallel_insert=8`,
`llm_model_max_async=16`, `embedding_func_max_async=16`,
`embedding_batch_num=32` to the LightRAG constructor (flags above override).
These are aggressive-but-bounded for a network-bound API stack on a small
CPU-only host: high enough to keep a remote API busy, bounded so in-flight
document state cannot exhaust RAM. An OOM kill is recoverable, not fatal —
the resume contract below is the safety net that makes the aggressive
defaults safe. For a single local ollama, lower them (e.g.
`--max-async-llm 4 --max-parallel-insert 3`) to match its serial queue.

## Protocols

**Ingest and extraction cap.** The extraction subset ALWAYS contains every
qrels-positive passage, filled from the sorted remaining ids up to
`--extraction-cap`; a cap below the positive count fails loud. The subset runs
the real pipeline with the answer-LLM binding; every other passage ingests
chunks-only through `ainsert_custom_kg` with `full_doc_id=corpus_id` and no
LLM call. Set the cap from measured extraction throughput.

**Ingest timeout and FAILED-document recovery.** The whole ingest (extraction
subset + chunks-only remainder + recovery rounds) runs under
`asyncio.wait_for(--ingest-timeout)`; the 24 h default covers the longest
dataset chain plus retry overhead. After the main ingest, FAILED extraction
documents re-enter through the product's sanctioned sticky manual retry path
— the exact `commit_manual_retry_request` +
`apipeline_process_enqueue_documents` sequence behind POST
`/documents/reprocess_failed` — granting ONE attempt per document per round
with the SAME model and parameters: execution robustness for intermittent
extraction failures (e.g. ollama's token-repeat guard), not a protocol
deviation, and never direct doc_status state surgery. Rounds are bounded by
`--ingest-recovery-rounds` (0 disables); each round logs the document ids and
the recovered/still-failed split loudly, and the report's ingest section
annotates `recovery_rounds_used`, `recovered_doc_ids` and
`persistently_failed_doc_ids`. Any document still FAILED after the rounds are
exhausted raises `RuntimeError` — the fail-loud contract is unchanged, and
recovery never swallows a failure.

**Resumable runs (OOM / interruption contract).** Without `--workspace`
every run mints a fresh `hbench_<dataset>_<uuid8>` workspace and starts from
scratch. With `--workspace BASE` the per-dataset workspace is the stable
`BASE_<dataset>` (no random suffix) and `--keep-workspace` is implied, so
re-invoking the SAME command after an OOM kill, timeout or crash resumes
instead of repeating work:

- PROCESSED extraction documents are skipped; the product pipeline itself
  auto-resumes interrupted PENDING/PARSING/ANALYZING/PROCESSING rows. The
  harness never edits doc_status state directly.
- FAILED documents re-enter only through the sanctioned recovery rounds
  above (one attempt per document per round, same parameters).
- A chunks-only document is re-ingested unless its chunk row AND its vector
  both exist — the two stores `ainsert_custom_kg` writes for a chunks-only
  payload. A kill between the two upserts therefore heals on resume;
  re-ingesting an already-complete document is an idempotent upsert.
- When every extraction document is PROCESSED and every chunks-only
  document is complete, ingest returns immediately (`resumed: true` in the
  report's ingest section) and the run continues straight to arms/metrics/
  report. All phases are re-runnable.
- Progress checkpoints: a loud `[hybrid-bench] ingest checkpoint` log line
  plus `<working-dir>/<workspace>/ingest_progress.json` (dataset, workspace,
  phase, doc-status counts, UTC timestamp), refreshed every 30 s during
  ingest and at every phase boundary. After a kill, read it to see how far
  the run got, then re-invoke with the same `--workspace`.
- Resume identity is the operator's responsibility: the same dataset,
  embedding model + dim, and extraction parameters must be reused. Changing
  the embedding model or dim requires a fresh workspace (vector tables are
  model/dim-scoped); the harness does not detect parameter drift inside a
  reused workspace.

**Arms.** Dense and hybrid chunk retrieval go through the real
`operate._get_vector_context` dispatch (`QueryParam.enable_hybrid`); entity
seeding calls `entities_vdb.query` / `hybrid_query`. All arms share one
precomputed query embedding and the same `top_k`. The lexical-only arm
(diagnostic) executes the SAME `SQL_TEMPLATES["lexical_chunks"]` template
`hybrid_query` uses, with the same tokenizer, against the same workspace — no
product change. Note the production tokenizer ANDs every query token/bigram:
a natural-language query whose bigrams cross word boundaries can match
nothing lexically; that is the measured product behavior, not a harness bug.

**Derived entity truth.** An entity is relevant to a query iff the
`full_doc_id` set of its source chunks intersects the query's qrels-positive
corpus ids (graph `source_id` split on `GRAPH_FIELD_SEP`, resolved through
`text_chunks`). Queries with empty derived truth are reported separately and
never averaged into hit@k.

**Metrics.** recall@k = distinct retrieved documents in the first k ranks over
positive count. nDCG@k = linear qrels-score gain, discount `1/log2(rank+1)`,
chunk ranks deduplicated to documents at first occurrence (bounded [0,1]);
IDCG ranks all positive scores descending. hit@k = any truth entity in the
first k. Empty retrieval scores 0.0 and is listed per arm. Bootstrap: >=1000
per-query resamples from `random.Random(seed)`; paired arm differences share
one index list per resample; nearest-rank 2.5/97.5 percentiles. Same seed +
inputs = byte-identical intervals (test-pinned).

**Ablation.** Storage-level dispatch, no product change: entity-only =
entities hybrid + chunks dense; chunk-only = chunks hybrid + entities dense;
both = both hybrid. Gate: both >= each single arm per metric.

**Latency.** The production query router (`create_query_routes`) is
mounted on a minimal FastAPI app and driven via `httpx.ASGITransport`:
requests pass through `QueryRequest` validation and `to_query_params` exactly
as on a live server, minus the network hop — the documented direct-call
equivalent of POST `/query/data` (`only_need_context=True`, retrieval
latency) and `/query` (`only_need_context=False`, answer latency).
`hl_keywords`/`ll_keywords` are injected
equal to the query text so no keyword LLM runs in the measured path; the
hybrid-minus-dense delta therefore includes the per-query index-validity
probe (3 round-trips vs 1). The bench LightRAG disables the LLM cache so
every round measures real work. p50/p95 are nearest-rank over all samples;
the overhead CI resamples per-query means paired across arms.

**Write amplification.** `write_amp_driver.py` is self-contained
(imports only lightrag + numpy, baseline-safe) and runs UNCHANGED under the
branch interpreter and a pristine `--baseline-ref` checkout's own interpreter,
on the same corpus and PG load, measuring per-batch upsert+flush p95. With
`--write-amp-baseline` the harness creates a temporary detached worktree under
the system temp dir, `uv sync`s it, runs both sides, and removes the worktree
in `finally`. The driver uses synthetic hash embeddings on BOTH sides: this is
a write-path measurement, not a quality claim. GIN-vs-vector index sizes come
from per-index `pg_relation_size` on the ingested chunks table (CJK corpus for
the gate).

**Backfill.** Seeds NULL-tsv rows for the run's workspace, triggers the
production backfill scheduling path, polls the pending count to zero under a
bounded deadline, reports chunks/min. Report-only.

**Reranker survival.** qrels-positive lexical-only chunk hits are scored
as (query, content) pairs by a CrossEncoder; survival = fraction reaching
`--rerank-min-score`. sentence-transformers is imported LAZILY inside the
reranker function only — importing the harness never requires it. Below the floor is a
documented RESIDUE verdict, not a run failure. Runtime overlay (no repo
dependency changes):

```bash
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
  uv run --no-sync --with sentence-transformers python -m lightrag.evaluation.hybrid_bench ...
```

## Gates (PROPOSALS — printed verdicts, never tuned)

Each gate applies to its own domain (terminology or general). Statuses:
`pass`, `fail`, `residue` (reranker survival), `report` (backfill), `not-run`
(measurement absent).

| gate | proposal |
| --- | --- |
| terminology entity hit | entity hit@10: hybrid >= dense +15% relative AND paired CI lower > 0 |
| general entity hit | entity hit@10: hybrid >= dense -1% relative AND CI upper of dense-minus-hybrid < 1% of dense |
| terminology chunk quality | recall@10 and nDCG@10: hybrid >= dense +10% relative AND paired CI lower > 0 |
| general chunk quality | nDCG@10: abs delta <= 0.005 AND CI upper of the absolute delta < 0.005 |
| retrieval latency | retrieval p95 overhead <= 25% |
| answer latency | answer p95 overhead <= 15% |
| write amplification | upsert p95 increase <= 10% AND GIN <= 1.5x vector index size (CJK) |
| backfill | >= 5000 chunks/min, report-only |
| reranker survival | survival >= 50%; below = residue flag |
| ablation | both-hybrid >= each single-hybrid arm per metric |

Gate pooling: write amplification and backfill use the first dataset in sorted
order that measured them (all datasets' numbers are in the report); reranker
survival pools scored hits across datasets; latency gates pool per-query
latencies across datasets.

## Outputs

`<output-dir>/report.md` + `report.json`: domain x arm x metric tables with
95% CIs, per-query values, zero-positive/zero-truth/empty-retrieval query
lists, latency/write-amp/backfill/rerank sections, gate verdicts, and a
hardware/model/parameter appendix. A run with `fake-hash` embeddings or the
fake LLM is labeled **SYNTHETIC** in both outputs: mechanics proof, not
quality evidence.

## Smokes

Offline permanent smoke (file backends, no services):

```bash
./scripts/test.sh tests/evaluation
```

PG mechanics smoke (synthetic, tiny fixture):

```bash
python -m lightrag.evaluation.hybrid_bench \
  --dataset-dir <tiny-fixture> --output-dir <out> --storage pg \
  --embedding-backend fake-hash --embedding-dim 32 --llm-backend fake \
  --extraction-cap <corpus-size> --cosine-threshold 0.1
```

Real mini-smoke (<=20 passages, local serving stack + reranker overlay): drop
`--embedding-backend/--llm-backend` fakes, keep `--phases
ablation,write-amp,backfill,rerank,report` (the 600 answer calls belong to a
full run), and run under the overlay command above.

## Full-run checklist

1. Serving stack up (PG + pgvector, ollama with the embedding and answer models).
2. Datasets normalized to the contract above.
3. Measure extraction throughput on a pilot; set `--extraction-cap` so the
   subset (all qrels-positive passages + fill) finishes in budget.
4. Full run: all phases, `--write-amp-baseline`, reranker overlay, real
   embeddings/LLM, `--latency-queries >= 100 --latency-rounds 3`.
5. Read the printed gate verdicts. A terminology-gain failure is an honest
   failure: stop, report, back to design — no threshold tuning.

## Known residues

- The lexical arm inherits the production tokenizer's AND-of-all-bigrams
  semantics; natural-language queries can lexically match nothing. The arm
  reports empty retrieval explicitly instead of hiding it.
- Write amplification on a tiny corpus is dominated by per-batch fixed
  overhead; a full run must use the real corpus load for a
  meaningful upsert p95 comparison.
- File-backends runs skip the lexical arm, write-amp and backfill (PG-only
  surfaces) and label the skips; the hybrid arm falls back to dense because
  the file vector backend does not declare `supports_hybrid_query`.
