# SteinerPy context selection: a small LightRAG prototype

This opt-in experiment implements the suggestion in [LightRAG #3866](https://github.com/HKUDS/LightRAG/issues/3866#issuecomment-5574596724): select useful context within the existing token limits while treating connectivity as a bonus. It calls SteinerPy's **actual directed prize-collecting heuristic**, preserves isolated records as candidates, and plugs into `_apply_token_truncation`.

The current evidence supports a working prototype, not a demonstrated answer-quality improvement. The bundled diagnostics exercise real LightRAG context selection and chunk assembly with synthetic candidate lists. They do not call an LLM. The live evaluation route generates answers from an existing indexed corpus. The [review follow-up](SteinerContextReview.md) lists the six-arm design, verification, and remaining acceptance evidence.

## What is implemented

* `lightrag/steiner_context.py`: optional selector and transformation.
* `lightrag/operate.py`: stage-2 integration, diagnostic metadata, and answer-cache separation.
* `lightrag/evaluation/steiner_ablation.py`: paired end-to-context comparison and optional real answer generation.
* `lightrag/evaluation/steiner_scaling.py`: candidate-count/selector-latency experiment.
* `tests/evaluation/test_steiner_context.py`: mathematical oracle, budget, isolation, mapping, and pipeline tests.

The default rank-and-truncate route remains the default. SteinerPy is an optional dependency. No database migration, ingestion change, extra retrieval call, or new embedding model is required. This is an SDK experiment; it does not add UI controls.

## The idea in plain English

Give every retrieved entity description and relation description a relevance prize. Charge tokens for every selected description. Nearby records receive a modest bonus when selected together. A relevant isolated entity can still be selected independently and earns its full relevance prize.

An artificial root connects every candidate in the optimization graph. Removing that root can leave several independent groups and isolated entities. The root is never rendered to the model, and its links never become factual relationships.

This follows [mhoangvslev's idea in SteinerPy #28](https://github.com/berendmarkhorst/SteinerPy/issues/28): prize-collecting retrieval of a candidate subgraph for graph question answering, inspired by G-Retriever. **The soft-connectivity construction here is an adaptation for LightRAG**, not something claimed to have been proposed in that issue. SteinerPy now provides the directed class and `exact=False` path needed for it.

LightRAG's current retrieval treats edge topology as undirected and may deduplicate opposite orientations before stage 2. Accordingly, the selector uses undirected incidence for its bonus. It preserves the descriptions and endpoint fields it receives; it does not claim to recover directed information already lost upstream. The directed optimization graph is used for node-cost accounting.

## Formulation and its limits

Let each candidate record be a vertex in an incidence graph H. An entity vertex is adjacent to a relation vertex only when that entity is an endpoint of the relation. Absent entity descriptions are not inserted as free vertices. Every relation row is a separate selectable record, including a relation with missing endpoints.

For a set S of records, maximize heuristically

\[
F(S)=\sum_{i\in S} r_i + \beta\bigl(|S|-k(H[S])\bigr),
\]

subject to separate entity and relation token limits. Here k is the number of connected components, with k(empty)=0. The bonus is the number of links in a spanning forest, multiplied by beta. It avoids rewarding arbitrary extra cycles. A single isolated record receives zero bonus and zero connectivity penalty. The objective can still trade an isolated record against other evidence; eligibility is guaranteed, retention of every high-ranked record is not.

This is **record coherence**, not a guarantee that all selected entities are connected, that both endpoints of every relation are selected, or that an answer's reasoning path is complete. The formulation imposes neither of those requirements.

There are two distinct prize sources. `ordering` uses `1/sqrt(position)` separately for entities and relations. It is an **ordering control**, not a calibrated relevance measure: the hybrid list interleaves sources and local relation ordering uses degree/support. `rerank_score` scores every candidate's name/endpoints and description using the configured `rerank_model_func`, orders each record type by descending score, and uses the numerical scores themselves as prizes in both packing arms. It does not convert them back to rank. All B arms use the same model as downstream chunk reranking.

Every returned index must be unique and valid, with a finite non-negative `relevance_score`. Providers may cap the returned list: records without a returned score are excluded from the KG candidate set, including an entirely unscored entity/relation type. An empty result selects no KG records. No ordering-based scores are invented for the tail. Duplicate, invalid, negative and non-finite scores still fail explicitly; logits require an explicit provider transform. Diagnostics report input, scored and unscored candidate counts, and the evaluator summarizes the excluded count. The candidate cap applies after this filtering. Scores are held outside serialized context and original source records. No score threshold removes KG candidates: the existing `min_rerank_score` applies to downstream text chunks.

`connectivity_bonus` is now a dimensionless coefficient: beta equals that coefficient times the largest candidate prize, for either prize source. Thus 0.15 means 15% of the highest prize per forest link, and multiplying all scores by a positive constant leaves the objective's relative scale unchanged. This is a stated scale convention, not evidence that 0.15 is calibrated or useful. Choose it on separate development questions, freeze it before the held-out run, and include zero as a sensitivity control.

The SteinerPy transformation is:

1. Split each record i into `i_in -> i_out`, charging its token-price penalty on that arc.
2. Give `i_out` prize `r_i + beta`.
3. Add an artificial-root arc to each `i_in` with cost beta.
4. Translate incidence links into zero-cost arcs from one record's output to the other's input, in both directions.

A chosen forest pays beta per component and collects beta per record, yielding the stated bonus. Root-only is feasible. Token penalties are normalized by the applicable entity/relation cap. Three fixed multiplier values, scaled by candidate relevance/token density, produce proposals through:

```python
DirectedPrizeCollectingProblem(
    directed_graph, node_prizes=prizes, root=artificial_root
).get_solution(exact=False)
```

These are Lagrangian proposals, **not solutions to the exact token-budget problem**. An exact, exhaustive subset oracle validates the transformation on tiny graphs. It does not establish optimality of the production heuristic or its postprocessing.

The wrapper considers the rank baseline, repaired proposals, and greedy selection using singleton records and proposed components. It returns the feasible candidate with highest proxy objective. Additive token costs guide search; admission and final feasibility use the actual newline-joined JSON serialization and the configured tokenizer. A singleton refill recovers some records missed by approximate screening. Both entity and relation caps remain hard. Source IDs and the original filtered rows continue into LightRAG's normal chunk selection.

Keeping both caps is deliberate: a joint graph objective can have two resource constraints. Pooling the budgets is not necessary to test connectivity. A future pooled-budget experiment should give every comparison arm the same pooled budget.

## Run the prototype

The prototype was developed and measured against:

* LightRAG `d964d92b1018c27983d1dcf6ca19ebbaebeb262e`
* SteinerPy `9d8cde91bb99029bc956b1a94d54bba865ffc58e` (package metadata: 1.0.19)

From a checkout containing this prototype:

```bash
uv venv
uv pip install -e '.[pytest]'
uv pip install 'steinerpy @ git+https://github.com/berendmarkhorst/SteinerPy.git@9d8cde91bb99029bc956b1a94d54bba865ffc58e'
.venv/bin/python -m lightrag.evaluation.steiner_ablation
.venv/bin/python -m lightrag.evaluation.steiner_scaling
PYTHON=.venv/bin/python ./scripts/test.sh tests/evaluation/test_steiner_context.py tests/llm/test_query_cache_user_prompt_prefix.py -q
```

Use the pinned SteinerPy commit for reproduction; `.[steiner]` is the convenience extra for installations with a release containing that API. The first tokenizer use may download its encoding. No Gurobi license is needed. The heuristic does not invoke a MIP solver, although SteinerPy's package dependencies include HiGHS and SciPy.

Enable it when constructing a normal LightRAG instance:

```python
rag = LightRAG(
    # Keep your existing model, storage, and embedding configuration here.
    addon_params={
        "context_selection": {
            "strategy": "steiner_soft",
            "connectivity_bonus": 0.15,
            "max_candidates": 120,
        }
    },
    # ...your existing constructor arguments...
)
await rag.initialize_storages()
```

Configure this at construction so storage snapshots receive it. Use `rank` with `prize_source="ordering"` or omit the setting for the baseline. Set `prize_source="rerank_score"` with a configured `rerank_model_func` for B0 (`rank`), B1 (`relevance`), or B2 (`steiner_soft`). Per-query diagnostics are in `raw_data["metadata"]["context_selection"]`. Unknown options and missing optional dependencies raise explicit errors. Configuring `prize_source="rerank_score"` without a callable `rerank_model_func` is rejected during construction/config refresh, before storage creation. The legacy chunk-only configuration retains its existing warning/pass-through contract; opting into KG score prizes requires a model because those prizes define the optimization objective.

A query with `enable_rerank=False` skips both KG and chunk reranker calls. The configured selector runs with ordering prizes for that query, explicitly reporting `requested_prize_source="rerank_score"`, `score_source="ordering"` and `rerank_disabled=true`. This query-level opt-out is not a B-arm measurement; the six-arm evaluator requires reranking on.

## Evaluation design

| Arm | Prize source | Selector |
|---|---|---|
| A0 | Current ordering | Prefix truncation |
| A1 | Current ordering | Relevance packing |
| A2 | Current ordering | Steiner soft |
| B0 | KG reranker score | Score-sorted prefix truncation |
| B1 | KG reranker score | Relevance packing using numerical scores |
| B2 | KG reranker score | Steiner soft using numerical scores |

Chunk reranking is enabled throughout, using the same configured model, `chunk_top_k`, entity/relation/total limits, and project-default `min_rerank_score`. At this base revision the constant `DEFAULT_MIN_RERANK_SCORE` is **0.0**. The runner imports it instead of pinning a literal or changing the threshold to 0.5. A separate positive-threshold regression exercises downstream filtering.

The optional legacy `soft_greedy` and `steiner_hard` controls remain callable, but the acceptance runner uses the six factorial arms. `steiner_hard` restricts the postprocessed selection to one component and is only a diagnostic. Above `max_candidates`, packing/Steiner arms explicitly report a fallback to prefix truncation **under the same prize source**: A2 to A0, B2 to B0. KG reranking is still paid for in a B fallback. `rank` itself is not capped. Raw and summary results expose fallback frequency and candidate counts; top_k is not a candidate-record count.

### Fixed real-index study

Create an importable `my_setup.py` whose `make_rag()` returns an **initialized** LightRAG instance over a fixed, existing index. Configure its answer model, matching embeddings, and one reranker shared by KG and chunks. Disable response caching (`enable_llm_cache=False`, including provider caching), use no user prompt prefix, and set `min_rerank_score=DEFAULT_MIN_RERANK_SCORE`. The runner finalizes storages and does not ingest documents.

Prepare 30–60 held-out questions, with unique IDs and categories including `isolated evidence`, `multi-hop evidence`, and `disconnected evidence`. Use separate development questions to choose beta. Fix keywords once using the same extractor. A dataset row has this schema; the example is synthetic and must be replaced:

```json
{
  "id": "q001",
  "category": "isolated evidence",
  "question": "What is the remote access code?",
  "ll_keywords": "remote access code",
  "hl_keywords": "access policy",
  "answer": "amber",
  "answer_aliases": ["amber"],
  "evidence": ["remote access code is amber"]
}
```

Save the rows as a JSON array, then run:

```bash
.venv/bin/python -m lightrag.evaluation.steiner_ablation \
  --factory my_setup:make_rag --dataset qa.json \
  --index-id 'corpus-v1/index-v1/embedding-version' \
  --answer-model-id 'provider/model-version' \
  --rerank-model-id 'provider/model-version/hardware-and-worker-config' \
  --budgets 30000 --entity-budget 6000 --relation-budget 8000 \
  --chunk-top-k 20 --top-k 20 40 60 --mode hybrid \
  --concurrency 1 8 16 32 --repeats 3 --load-repeats 3 \
  --output temp/steiner-real-qa
```

Those are this revision's production budget defaults. Supply additional total limits as a predeclared sensitivity study. The same caps apply to all six arms. The runner checks the rendered system prompt plus question against the total cap and checks the exact serialized entity/relation counts independently. Provider message framing and output tokens are outside that input-budget convention. Realized input and answer tokens are reported separately. Equal caps do not imply equal realized usage.

The factory must configure the same embeddings that built the index. Model and index identifiers, dataset SHA-256, source-code hashes and runtime versions are saved. `--smoke-test` permits smaller live datasets but labels their output as smoke tests. A missing/failed reranker, invalid returned scores, wrong default threshold, or enabled response cache prevents a valid live comparison. A provider-capped KG response is allowed, with the unscored tail excluded and reported; use identical provider settings for all B arms. No factory means **synthetic inputs with a lexical test double and no generated answers**; it never substitutes for a real model study.

### Timing and net rerank load

The quality phase shares a raw candidate snapshot across arms, warms each arm, randomizes execution order, and generates factual answers with the same prompt and answer model. Raw snapshots, contexts, answers, selection diagnostics and every rerank call are exported. `selector_ms` includes KG scoring, token accounting and CPU worker dispatch. Its metadata separates `kg_rerank_ms`, `solver_ms` and `worker_ms`. Serial `retrieval_ms` adds shared stage-1 time to measured stages 2–4; keyword extraction and generation are excluded.

The separate load phase runs each arm alone at 1/8/16/32 concurrent requests. Every request repeats actual stage-1 retrieval and stages 2–4, verifies that the fixed index produced the same candidate snapshot, and measures its own elapsed time. It does not add a shared serial search time to concurrent selection time. It omits answer generation to isolate the retrieval hot path. Requests use bounded, closed-loop concurrency; time before admission is excluded. Provider queues and CPU executor queues after admission are included. Each trial has at least two requests per worker and reports throughput, p50/p95 and sample count. Increase load repetitions for a stable tail estimate. This is not a full HTTP-server saturation or GPU-utilization benchmark.

Request-local counters wrap the same configured reranker for KG and chunks. They count **all submitted chunk documents**, even when `top_n=chunk_top_k` returns fewer. Outputs include logical calls, submitted descriptions/chunks, estimated document tokens, estimated query–document pair tokens, and adapter wall time. Totals sum KG and actual downstream chunk work. Summaries subtract A0's total for the same setting to show net additional or saved work. Adapter wall time includes waiting and is not GPU compute time. Internal provider retries, document splitting, HTTP calls, billing and device utilization require provider instrumentation; these counters do not claim to measure them. Failed chunk reranking is detected even though LightRAG normally catches the error and continues.

### Answer quality and decision rule

For factual answers the runner reports normalized exact match and token F1. Repetitions are averaged within question, then paired question differences are bootstrapped for A2−A0, A2−A1, B2−B0 and B2−B1, overall and by category. Load repetitions never enter those intervals. These are descriptive percentile intervals, not a multiplicity-adjusted hypothesis test across every budget/top_k/metric. Predeclare the primary setting and metric; inspect blinded correctness/groundedness and citations for answers that lexical scoring cannot assess.

Adoption requires an answer-quality improvement over **both prefix truncation and packing on the same prize source**, with paired uncertainty supporting the difference and a useful margin relative to total retrieval cost. A B0/B2 tie supports the simpler reranking change. A B1/B2 tie gives no reason to add the solver. There is no automatic adoption decision and no real-quality result yet.

## What the mathematics does and does not guarantee

The exact small-instance oracle validates the transformation, not the production algorithm. SteinerPy's directed `exact=False` path supplies a dual-ascent primal and an instance-specific gap for the penalized PCST problem. Each proposal now exports that gap and the corresponding upper bound on the Lagrangian maximization objective; an exhaustive regression checks the bound. A zero gap certifies only that proposal problem. No uniform approximation ratio is claimed for this directed heuristic, and the certificate does **not** survive budget repair/growth/refill as a certificate for the original constrained problem.

The final soft selector guarantees feasibility via explicit runtime checks and proxy-objective value at least that of the same-prize prefix baseline, because the feasible baseline remains in its candidate set. This says nothing about answer quality. The forest term is not generally submodular as a function of selected vertices: for two adjacent records, the gain from adding the second is beta when the first is present, but zero when absent. Therefore a standard submodular-greedy approximation argument does not apply to this pipeline.

For any nonempty selected set of m records, the connectivity contribution lies between zero and beta(m−1). At a fixed size, a prize-sum advantage greater than beta(m−1) cannot be reversed by the bonus. This explains why prize quality can dominate when only roughly ten entities and a dozen relations fit. Conversely, a few near-tied evidence choices might differ in whether they jointly support a multi-hop answer; topology could help there, but descriptions can already contain the complete answer and a bridge can be irrelevant. That is a falsifiable motivation, not a proof that F predicts answer quality. The held-out same-prize comparisons decide whether the term earns its cost.

## Current limitations

This implementation is intended for small candidate sets. The default cap is 120 total records. Its worker runs off the event loop, but SteinerPy's `exact=False` call does not honor the exact solver's time limit; the cap is not a wall-clock SLA or cancellation guarantee. The timings show a material cost near the cap, and production concurrency has not been benchmarked; only the diagnostic load path has been exercised.

The objective rewards structural coherence, which need not correlate with answer quality. Ordering prizes can reflect node degree; reranker prizes also remain a relevance proxy, not answer utility. Missing bridge candidates cannot be recovered at this integration point. The selector guarantees the stage-2 caps; the evaluation runner additionally rejects an over-budget final prompt rather than quietly comparing it against a smaller baseline.

## Source pointers

* [Collaborator feedback, LightRAG #3866](https://github.com/HKUDS/LightRAG/issues/3866#issuecomment-5574596724)
* [Contributor proposal, SteinerPy #28](https://github.com/berendmarkhorst/SteinerPy/issues/28)
* [Inspected LightRAG retrieval code](https://github.com/HKUDS/LightRAG/blob/d964d92b1018c27983d1dcf6ca19ebbaebeb262e/lightrag/operate.py)
* [Inspected SteinerPy directed PCST implementation](https://github.com/berendmarkhorst/SteinerPy/blob/9d8cde91bb99029bc956b1a94d54bba865ffc58e/steinerpy/objects.py)

## Evaluation prerequisites and upstream ordering fix

The maintainers are handling [#3917](https://github.com/HKUDS/LightRAG/issues/3917): stage-2 ordering must reach the filtered original records used for chunk quotas. This PR deliberately does not work around that upstream bug. Rebase on their fix and then run the real-index comparison; the existing smoke artifact predates it and is only instrumentation evidence. The maintainers have offered to run the held-out study on their configured index after the fix, subject to their internal confirmation. A real hosted-provider compatibility check remains part of that run; unit tests of capped responses are not a live-service validation.

The tokenizer-based tests marked `offline` do not call model services, but `cl100k_base` needs its encoding file cached first. In a network-enabled setup step, use the **same** cache directory that the test environment will receive:

```bash
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
mkdir -p "$TIKTOKEN_CACHE_DIR"
.venv/bin/python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base")'
```

Preserve/copy that directory into an offline runner and set `TIKTOKEN_CACHE_DIR` there before invoking pytest. With an empty cache and blocked downloads, these tests cannot run offline. This dependency should be prepared before measuring latency as well.
