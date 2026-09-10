# SteinerPy context selection: a small LightRAG prototype

This opt-in experiment implements the suggestion in [LightRAG #3866](https://github.com/HKUDS/LightRAG/issues/3866#issuecomment-5574596724): select useful context within the existing token limits while treating connectivity as a bonus. It calls SteinerPy's **actual directed prize-collecting heuristic**, preserves isolated records as candidates, and plugs into `_apply_token_truncation`.

The current evidence supports a working prototype, not a demonstrated answer-quality improvement. The bundled diagnostics exercise real LightRAG context selection and chunk assembly with synthetic candidate lists. They do not call an LLM. The live evaluation route generates answers from an existing indexed corpus.

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

Numerical vector similarities are not available on the current stage-2 rows. In particular, `rank` in `_get_node_data` is node degree. For a minimal integration, every arm uses the existing retrieval order: `r_i = 1/sqrt(position_i)` separately within entities and relations. These are explicit proxy scores, not calibrated probabilities or similarities. Relation support `weight` is not used as query relevance.

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

Configure this at construction so storage snapshots receive it. Use `rank` or omit the setting for the baseline. Per-query diagnostics are in `raw_data["metadata"]["context_selection"]`. Unknown options and missing optional dependencies raise explicit errors.

## Evaluation design

| Arm | What it isolates |
|---|---|
| `rank` | Existing LightRAG prefix truncation |
| `relevance` | Better token packing with no connectivity reward |
| `soft_greedy` | Connectivity reward with no SteinerPy calls |
| `steiner_soft` | Additional candidate bundles from SteinerPy |
| `steiner_hard` | Diagnostic restriction to one connected component |

`steiner_hard` is a postprocessed heuristic control, not an exact hard-connectivity solver. Above the candidate limit, all experimental arms explicitly report fallback to rank truncation; exclude those runs from claims about the heuristic while reporting their frequency. `top_k` is not the same as the number of returned entity-plus-relation rows.

For real QA, create an importable `my_setup.py` whose `make_rag()` returns an **initialized** LightRAG instance over the intended existing corpus. Keep the corpus, index, embeddings, chunk picker, reranker, answer model, and provider options fixed. Disable model response caching in that factory (`enable_llm_cache=False` and any provider cache), and leave `user_prompt_prefix` empty. The runner finalizes the storages after completion. It does not ingest or modify the corpus.

The input is a JSON array with this schema:

```json
[
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
]
```

That example must be replaced with questions and answers grounded in the chosen corpus. Generate/fix keywords once, using the same keyword extractor for all arms; keyword generation time is outside the reported retrieval timing. `evidence` is optional literal text for a simple coverage diagnostic, not a semantic evaluator.

```bash
.venv/bin/python -m lightrag.evaluation.steiner_ablation \
  --factory my_setup:make_rag --dataset qa.json \
  --budgets 2500 4000 --entity-budget 600 --relation-budget 800 \
  --top-k 20 40 60 --repeats 3 --mode hybrid \
  --output temp/steiner-real-qa
```

Use `--rerank` if that is the fixed baseline setting. This command generates real answers and incurs the configured model's normal usage. The factory must configure the same embedding model used to build its existing index.

The runner shares one raw search snapshot among arms per question/repeat, reruns stages 2–4, and allows selected records to change downstream source chunks. It warms each arm, randomizes arm order, saves every context, answer, selection, token count, and timing, and holds `max_entity_tokens`, `max_relation_tokens`, `max_total_tokens`, and `chunk_top_k` fixed. The total budget gate counts the rendered system prompt plus the question; provider message framing and response tokens are outside that convention. Equal limits do not mean identical realized token usage, so usage is reported too.

`selector_ms` covers stage 2, including token accounting and thread dispatch. `retrieval_ms` adds shared stage-1 time to the measured stages 2–4. It excludes keyword extraction and answer generation; this is a controlled paired experiment, not a concurrent server-load benchmark. The scaling script measures stage 2 only. The first warm-up is retained separately in scaling results and excluded from median/p95 summaries.

For short factual QA, the live runner reports normalized exact match and answer token F1. For long answers, add blinded correctness/groundedness review or use LightRAG's existing evaluation tooling. Review source attribution as well as correctness. Do not interpret lexical evidence recall as answer quality.

Start with 30–60 held-out questions, including isolated-description questions, multi-hop questions, and questions requiring disconnected facts. Fix beta and multiplier choices on separate development questions. Aggregate repeats per question, report paired quality differences with question-level uncertainty, and show median/p95 latency and fallback rates. Repeats are not independent new questions. A meaningful result is a quality gain over **both relevance packing and soft greedy** that justifies the added latency. The bundled toy results have not established that gain.

See [measured results](SteinerContextResults.md) for the synthetic diagnostics and latency measurements. Real-model answer-quality evaluation remains pending.

## Current limitations

This implementation is intended for small candidate sets. The default cap is 120 total records. Its worker runs off the event loop, but SteinerPy's `exact=False` call does not honor the exact solver's time limit; the cap is not a wall-clock SLA or cancellation guarantee. The timings show a material cost near the cap, and production concurrency has not been benchmarked.

The objective rewards structural coherence, which need not correlate with answer quality. Retrieval-order prizes are a pragmatic starting point. Missing bridge candidates cannot be recovered at this integration point. The selector guarantees the stage-2 caps; the evaluation runner additionally rejects an over-budget final prompt rather than quietly comparing it against a smaller baseline.

## Source pointers

* [Collaborator feedback, LightRAG #3866](https://github.com/HKUDS/LightRAG/issues/3866#issuecomment-5574596724)
* [Contributor proposal, SteinerPy #28](https://github.com/berendmarkhorst/SteinerPy/issues/28)
* [Inspected LightRAG retrieval code](https://github.com/HKUDS/LightRAG/blob/d964d92b1018c27983d1dcf6ca19ebbaebeb262e/lightrag/operate.py)
* [Inspected SteinerPy directed PCST implementation](https://github.com/berendmarkhorst/SteinerPy/blob/9d8cde91bb99029bc956b1a94d54bba865ffc58e/steinerpy/objects.py)
