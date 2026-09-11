# Follow-up to the adoption review

This update responds to [danielaskdd's review of PR #3914](https://github.com/HKUDS/LightRAG/pull/3914#issuecomment-5628720723) and [issue #3866](https://github.com/HKUDS/LightRAG/issues/3866#issuecomment-5574596724). The implementation and evaluation tools have been revised. **The real-index answer-quality study remains unperformed; the method has not met the acceptance criterion.**

## Changes made

| Review request | Implementation / evidence |
|---|---|
| Six matched arms | A0/A1/A2 use the existing ordering; B0/B1/B2 use KG rerank scores. Both rows compare prefix, relevance packing, and Steiner soft. |
| Numerical prizes in B2 | The configured reranker scores all entity/relation descriptions. B1 and B2 consume the actual non-negative scores, with strict validation of returned indices/scores; provider-capped tails are excluded and counted. |
| Chunk reranking throughout | All six arms enable it and share the same model and limits. Invalid/failed scoring prevents a valid benchmark result. |
| Default score threshold | The evaluator imports `DEFAULT_MIN_RERANK_SCORE`, which is 0.0 at the tested base. A separate positive-threshold regression tests downstream removal of low-score chunks. |
| Net rerank load | Per-query KG and chunk adapter calls, all submitted documents, token estimates and adapter wall time; total and net difference versus A0. |
| Serial and concurrent latency | Paired serial QA plus separate retrieval-only load trials at 1/8/16/32 concurrent queries, with real stage-1 calls in live load mode. |
| Realistic top_k fallback frequency | Live CLI defaults to 20/40/60. Results report counts and fallback rates per setting. B2 falls back to B0, retaining rerank scoring. |
| Paired uncertainty | Question-level bootstrap of repeated-answer means for the four same-prize Steiner/control contrasts; load rows excluded. |
| Theory and scale | Documented instance-specific proposal certificates, limits of repair/growth, baseline proxy dominance, and the bounded role of connectivity at small selection sizes. Beta scales with the largest prize. |
| Budget invariant under `python -O` | Explicit exception on every selector return path; regression includes an optimized Python subprocess. |
| Optional CI dependency | SteinerPy/HiGHS/SciPy are installed in a dedicated optional-selector job. The shared offline job no longer requests the extra. |

The original rank path remains the default. B0 can be enabled independently of the SteinerPy backend with `strategy="rank", prize_source="rerank_score"`; the complete acceptance comparison additionally needs the optional backend for A2/B2.

## Verification

124 tests passed in:

```bash
PYTHON=.venv/bin/python ./scripts/test.sh \
  tests/evaluation \
  tests/llm/test_query_cache_user_prompt_prefix.py \
  tests/llm/test_query_cache_conversation_history.py \
  tests/llm/test_apply_rerank_result_validation.py \
  tests/api/routes/test_concurrent_query_tokenizer.py -q
```

Coverage includes score-to-record identity, isolated descriptions, exact serialized budgets, preservation of source rows, chunk filtering, separate caches for B0, fallback scoring, rejection of invalid reranker output, request-local concurrent accounting, question-level uncertainty and the live-driver contract. Scripted answers in contract tests are not evaluation evidence. Focused Ruff checks and `git diff --check` pass.

The six-arm diagnostic exercises the real selection/chunk assembly functions with four synthetic cases and a deterministic lexical reranker test double:

```bash
.venv/bin/python -m lightrag.evaluation.steiner_ablation \
  --repeats 3 --top-k 20 --concurrency 1 8 16 32 --load-repeats 2 \
  --output temp/steiner-review-diagnostics
```

It produces 72 paired context rows and 720 load rows, all within the specified token caps. There are only four unique synthetic questions. No answer model is called and the paired answer-quality report is empty. The archived initial-follow-up (79cb5e9) [machine-readable summary](evaluation/steiner-review-smoke.json) records settings, source hashes, runtime versions, all six arms, logical rerank load and latency at each concurrency level. These timings verify instrumentation and must not be interpreted as GPU throughput or production performance. The `top_k=20` label does not affect these fixed synthetic candidates, so this run supplies no real-index fallback estimate.

The old 50%→100% evidence-coverage comparison is retained only as [historical, reranking-disabled data](SteinerContextResults.md). It is not evidence of an answer-quality improvement against the production baseline.

## Still required for adoption

The workspace used for this revision had no fixed real LightRAG index and no configured answer, embedding or rerank models. Consequently, the following results remain pending:

* 30–60 held-out real-index questions spanning isolated, multi-hop and disconnected evidence, answered by a fixed real model.
* Same-prize quality differences against both controls, with question-level uncertainty and groundedness/citation review.
* Real reranker net load and latency at 1/8/16/32 concurrent queries, plus fallback frequency at top_k 20/40/60.

The [prototype guide](SteinerContextPrototype.md#fixed-real-index-study) gives the runnable command and input/factory contract. It records immutable index/model identifiers and rejects missing model configuration. Held-out status and corpus provenance still require experimenter verification. No synthetic substitute is presented as satisfying the collaborator's adoption gate.

If B0 matches B2, the evidence favors the simpler reranking change. If B1 matches B2, it does not justify adding the solver. Proposal dual bounds certify an optimization surrogate only; they cannot settle either answer-quality comparison.

## Remaining-comments follow-up

Responding to [the follow-up review](https://github.com/HKUDS/LightRAG/pull/3914#issuecomment-5630770491):

* `enable_rerank=False` now suppresses KG scoring as well as chunk scoring. The selector uses the explicitly reported ordering control for that query; the six-arm evaluator continues to require reranking enabled.
* A KG rerank-score configuration without a callable model is rejected during LightRAG construction/config refresh, before creating storage resources. The legacy chunk-only warning/pass-through contract is unchanged.
* A provider may return a capped list. Every returned score is validated; unscored records are excluded instead of receiving invented prizes. Empty responses select no KG records. Input/unscored counts are exposed in selection metadata, and the evaluator accepts and reports capped KG responses.
* The guide now documents pre-seeding `TIKTOKEN_CACHE_DIR` before offline execution.

No workaround for #3917 is included. The maintainers proposed running the real-index study after their ordering fix lands, pending internal confirmation; that is the planned handoff. No real hosted-provider call or new answer-quality claim is made by this update.

Validation for this follow-up: 138 focused tests passed (the command above plus `tests/test_addon_params.py`). Ruff 0.15.17 checks and `git diff --check` pass. The prior CI failures from formatter wrapping, the JSON final newline and the bare source-docstring issue reference are corrected. The archived smoke measurements were not rerun or relabelled as new results.
