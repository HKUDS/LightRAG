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

A workspace override (`POSTGRES_WORKSPACE` or a `config.ini` `[postgres]`
`workspace` value) fails the run BEFORE any storage is initialized, in both
the harness and the write-amp driver: PG storages honor the override over the
run's minted workspace, so the benchmark would write into the configured
workspace while cleanup deletes the minted name.

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
recovery never swallows a failure. ONE permanent failure class is excepted,
with proof: provider output content moderation (Aliyun MaaS
`BadRequestError 400` with code `data_inspection_failed`, "Output data may
contain inappropriate content"). For flagged passages every faithful
extraction output is rejected deterministically, so the rounds cannot
recover it; such a document ingests CHUNKS-ONLY through
`ainsert_custom_kg` (`full_doc_id=corpus_id`, no LLM call — the embedding
API accepts the text) and the report annotates `moderation_fallback_doc_ids`
with the honest entity-arm gap: the chunk arm is intact, NO entities are
extracted, and queries positive only on that document can derive empty
entity truth (excluded from hit@k, listed as zero-truth). Its doc_status row
stays FAILED (truthful — extraction failed; never direct state surgery), so
every resume re-runs the bounded rounds and re-applies the idempotent
fallback. Every OTHER permanent failure class still raises.

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
  above (one attempt per document per round, same parameters). A
  moderation-fallback document stays FAILED, so each resume re-runs its
  bounded rounds and re-applies the idempotent chunks-only fallback.
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
`operate._get_vector_context` dispatch (`QueryParam.enable_hybrid`) with the
whole query and one precomputed whole-query embedding; the chunk lexical leg
stays whole-query AND (the entity-side OR fix does not apply to chunks). All
arms share the same `top_k`. The lexical-only arm (diagnostic) executes the
SAME `SQL_TEMPLATES["lexical_chunks"]` template `hybrid_query` uses, with the
same tokenizer, against the same workspace — no product change;
`postgres_impl` is imported lazily inside that arm, so a file-backend run
never loads the PG driver stack. Note the production tokenizer ANDs every
query token/bigram: a natural-language query whose bigrams cross word
boundaries can match nothing lexically; that is the measured product
behavior, not a harness bug.

**Entity seeding (production-faithful multi-keyword).** The entity arms
mirror `operate._get_node_data`. One keyword-extraction pass per query runs
through the product's `get_keywords_from_query` — the same keyword-role LLM
mechanism `kg_query` uses when keywords are not injected — cached within the
run: the bench disables the LLM cache, so a harness dict keyed on the query
text guarantees one extraction shared by both entity arms. Both arms query
`entities_vdb` with the joined keyword string `", ".join(keywords)` and ITS
embedding (production embeds the keyword string, not the raw query); the
hybrid arm additionally passes `lexical_keywords=keywords_str.split(", ")`,
so each keyword seeds entities on its own: AND inside a keyword, OR across
keywords — the entity-side hybrid fix. Omitting `lexical_keywords` would
measure the legacy whole-query AND path (the pre-fix protocol null).
Fallbacks are logged and annotated per query in `entity_seeding_fallbacks`:
an empty low-level list with non-empty high-level keywords uses the hl list
(the product's `entities_by_high_level_keywords` fallback); both lists empty
falls back to the query as ONE keyword, byte-identical to whole-query AND.
Production refuses zero-keyword queries over 50 chars; the benchmark must
score every query, so it applies that single-keyword fallback at any length
— a documented divergence. Extraction FAILURE (a provider error) always
propagates and fails the run; it never degrades to whole-query seeding. The
report carries `entity_seeding_protocol="production-faithful-multi-keyword"`.
Ablation and latency keep their injected-keyword protocols (dispatch
attribution and keyword-LLM exclusion respectively); the faithful entity
quality measurement is the dense/hybrid entity arms.

**Chunk attribution and duplicate content.** Chunk ids are content hashes on
the chunks-only path (`ainsert_custom_kg`), so documents with identical text
share one chunk; the attribution map keeps EVERY corpus alias per chunk id
and scoring expands all aliases of a retrieved chunk in sorted order at that
chunk's rank position. A query positive on any alias counts as retrieved.
Extraction-path chunk ids are doc-scoped (`hash("doc_id:content")`), so
extracted documents never share chunk ids with each other or with the
remainder.

**Content-duplicate twins in the extraction subset.** The product pipeline
dedups by content hash unconditionally (within the batch AND against
existing rows): when two subset passages have identical text, the second
twin is demoted to a `dup-<hash>` FAILED pointer row naming the original
(`Status: batch_duplicate` on first ingest, the original's live status on a
resume re-enqueue) and the twin id itself gets NO doc_status row. The
harness accepts that demotion as EXPECTED, but only with the full proof
chain: a content-hash pointer row whose `file_path` is the twin's corpus id
names an original that is PROCESSED, AND the product's content-hash lookup
(`get_doc_by_content_hash`) confirms that same original holds the twin's
sanitized text hash. The twin's corpus id then joins the original's chunk
attribution as an alias (the same list-valued map as chunks-only duplicates),
so both qrels-positive twins score on the shared chunk, and the report
annotates `duplicate_aliased_doc_ids`. Any other missing status — no pointer,
a stale pointer whose claimed content no longer matches, or an original that
is not PROCESSED — is a silent skip and still raises `RuntimeError`. Each
resume re-enqueues the twins (dedup-only, no LLM or chunk writes) and appends
fresh pointer rows; the alias map rebuilds identically.

**Derived entity truth.** An entity is relevant to a query iff the
`full_doc_id` set of its source chunks intersects the query's qrels-positive
corpus ids. Source chunks come from the `entity_chunks` tracking row — the
FULL attribution list; the graph `source_id` split on `GRAPH_FIELD_SEP` is
the fallback ONLY when no tracking row exists, because the graph field is
capped and would drop truth for entities appearing in more passages than the
cap. Chunk ids resolve to documents through `text_chunks`. A present tracking
row is authoritative even when empty. Queries with empty derived truth are
reported separately and never averaged into hit@k.

**Metrics.** recall@k = distinct retrieved documents in the first k ranks over
positive count. nDCG@k = linear qrels-score gain, discount `1/log2(rank+1)`,
chunk ranks deduplicated to documents at first occurrence (bounded [0,1]);
IDCG ranks all positive scores descending. hit@k = any truth entity in the
first k. Empty retrieval scores 0.0 and is listed per arm. Bootstrap: >=1000
per-query resamples from `random.Random(seed)`; paired arm differences share
one index list per resample; nearest-rank 2.5/97.5 percentiles. Resample index
lists are streamed one at a time (peak memory independent of the resample
count) with the seeded sequence preserved. Same seed + inputs = byte-identical
intervals (test-pinned).

**Ablation.** Each combination runs the COMBINED mix-mode retrieval path —
`operate.kg_query`, the same call `aquery_data` makes, with keywords injected
and context-only — with the two storage dispatches controlled independently
through instance-level capability overrides (a disabled dispatch shadows the
class `supports_hybrid_query` flag for the call and is removed afterwards; a
backend without the capability is never enabled): entity-only = entities
hybrid + chunks dense; chunk-only = chunks hybrid + entities dense; both =
both hybrid. Because the path is combined, the entity leg can change chunk
recall (entity-derived chunks merge with the vector leg), and metrics score
the FINAL context chunks and entities after the product's merge/truncation —
not two independent searches. No product change. Gate: both >= each single
arm per metric. On file backends no storage declares hybrid capability, so
all three combinations reduce to the dense path (labeled skip). The injected
whole-query keyword is deliberate and unchanged: ablation attributes quality
to the storage DISPATCHES, while the production-faithful multi-keyword
entity measurement lives in the entity arms (see above).

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
every round measures real work. p50/p95 are nearest-rank over ALL samples
(flat); the overhead point estimate AND its paired bootstrap CI use that
same flat all-samples definition (both arms run the same slots in the same
round order, so sample index i is the same (slot, round) on both sides).
Per-query round means are kept in the JSON for diagnostics only and never
feed a p95 gate.

**Write amplification.** `lightrag/evaluation/hybrid_bench/write_amp_driver.py` is self-contained
(imports only lightrag + numpy, baseline-safe) and runs UNCHANGED under the
branch interpreter and a pristine `--baseline-ref` checkout's own interpreter,
on the same corpus and PG load, measuring per-batch upsert+flush p95. The
corpus path is resolved to absolute before the subprocess (the driver's cwd
is a temp dir, so a relative `--dataset-dir` still works). With
`--write-amp-baseline` the harness creates a temporary detached worktree under
the system temp dir, `uv sync`s it, runs both sides, and removes the worktree
in `finally`. The driver uses synthetic hash embeddings on BOTH sides: this is
a write-path measurement, not a quality claim. GIN-vs-vector index sizes come
from per-index `pg_relation_size` on the ingested chunks table (CJK corpus for
the gate). Both the driver and the harness cleanup delete ONLY their
workspace's rows and RETAIN the shared model/dim vector tables — an
emptiness-check-then-DROP races a concurrent run's commits — and their
cleanup connections reuse the storage's `POSTGRES_SSL_*` context/cert
settings, so SSL-required servers are reached exactly like the measured path.

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
survival pools scored hits across datasets; latency gates pool the flat
all-samples latencies across datasets — the same p95 definition each dataset's
report table displays.

## Outputs

`<output-dir>/report.md` + `report.json`: domain x arm x metric tables with
95% CIs, per-query values, zero-positive/zero-truth/empty-retrieval query
lists, the entity-seeding protocol annotation (`entity_seeding_protocol`,
`entity_seeding_fallbacks`), latency/write-amp/backfill/rerank sections, gate
verdicts, and a hardware/model/parameter appendix. A run with `fake-hash` embeddings or the
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
- Duplicate-content aliases occupy consecutive document ranks after
  expansion; when a shared chunk sits exactly at the k-th rank, its
  later-sorting alias can fall outside the metric's document-level `[:k]`
  cut. Direction-harmless (it can only undercount, and only for datasets
  with duplicate text at the cutoff); the pre-fix map lost an alias
  systematically instead.
- Content-duplicate twins in the extraction subset are accepted with proof
  and aliased to their PROCESSED original (see the twin contract above);
  unproven missing statuses still fail loud. Residues: each resume appends
  fresh `dup-<hash>` pointer rows for the same twins (product-managed,
  cheap, never subset ids, so they never enter the recovery verdict); a
  twin whose original is FAILED (e.g. moderation) is NOT aliased — the
  proof chain requires a PROCESSED holder — and fails loud.
- A moderation-fallback document keeps two attributed copies of its chunk:
  the failed attempt's doc-scoped chunk (stage 1 writes chunks before
  extraction, and they survive the failure) and the fallback's content-hash
  chunk. Both alias to the corpus id and occupy consecutive ranks like any
  duplicate-content alias (same direction-harmless cutoff residue). Its
  doc_status row stays FAILED, so every resume pays the bounded recovery
  rounds again before re-applying the idempotent fallback; the entity-arm
  gap for that document is permanent for the workspace and annotated.
- Cleanup retains shared model/dim vector tables even when empty: leftover
  empty tables are harmless, and dropping them on an emptiness check would
  race a concurrent run's commits.
