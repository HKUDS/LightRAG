# Measured results and readiness

These are **historical measurements of the initial PR revision**, commit `32913da1ad05ecbc8da38e666b860ccd45b3fac0`. They used chunk reranking disabled and the old five-arm diagnostic. They do not represent a reranking-enabled LightRAG deployment. See the [review follow-up](SteinerContextReview.md) for the revised six-arm evaluator and its current verification status.

The prototype works, but these results do not establish that SteinerPy improves real answer quality. No real model was configured or called. All answer-quality fields in the diagnostic runs are null.

## Four hand-authored diagnostic questions

Real LightRAG stages 2–4 ran over synthetic candidate lists and in-memory source chunks. Questions cover an isolated description, a bridge, disconnected facts, and an oversized first hit. There are four unique questions, two total-input limits (1,200 and 1,600), five repeats, and five arms: 200 runs. Entity and relation limits are 100 and 90 throughout. Every run passed all three budget gates. Repeats are timing repetitions, not 200 independent questions.

| Selector | Mean required-evidence coverage | Median stage-2 latency (ms) |
|---|---:|---:|
| rank | 50% | 0.56 |
| relevance | 100% | 1.15 |
| soft_greedy | 100% | 1.12 |
| steiner_soft | 100% | 2.79 |
| steiner_hard | 75% | 2.88 |

Evidence coverage is literal presence of required text in the assembled context, not generated-answer correctness. The relevant isolated description survives soft selection and propagates to the final prompt. The hard control loses that evidence. Relevance-only packing and soft greedy match SteinerPy on all four diagnostics: there is no incremental SteinerPy quality result here. The rank-prefix cases intentionally include packing failures, so these percentages must not be generalized to real corpora.

## Candidate-count scaling

Synthetic entity/relation lists, fixed seed 3866, five measured repeats after one warm-up per arm/size, sequential requests. Every entry below is median / empirical p95 in milliseconds for stage 2 only. The small sample makes the tail estimate descriptive, not a production SLA. These are candidate-record counts, not top_k values.

| Records | Rank | Relevance | Soft greedy | Steiner soft |
|---:|---:|---:|---:|---:|
| 30 | 2.6 / 3.2 | 7.9 / 10.6 | 8.4 / 11.2 | 23.4 / 26.0 |
| 60 | 4.4 / 4.7 | 22.4 / 23.2 | 24.6 / 25.8 | 79.1 / 81.5 |
| 120 | 9.7 / 9.8 | 85.2 / 93.3 | 98.4 / 103.3 | 433.1 / 454.2 |
| 240 | 18.7 / 19.9 | 26.5 / 27.6 (fallback) | 24.8 / 25.1 (fallback) | 24.4 / 28.8 (fallback) |

At 240 records the experimental arms deliberately use the rank fallback because the cap is 120. Their low time is not evidence that the heuristic scales better. At 120 records, selection still adds roughly 0.4 seconds; even without a MIP solver, this must be justified by a quality gain. Solver time and wrapper time are separated in the raw scaling JSON.

## Validation

* 75 tests passed in the focused selector/evaluation and query-cache files.
* Eight five-vertex exhaustive-subset oracles compare the exact transformed SteinerPy solve with the mathematical objective; heuristic outputs are also checked.
* Tests cover zero/very small budgets, Unicode and literal special-token text, immutable inputs, isolated evidence and downstream chunks, missing endpoints, relation-row identity, candidate-limit fallback, default behavior, and answer-cache separation.
* The live evaluation route is exercised with a scripted answerer solely as a contract test. Those answers are excluded from the results above.
* Ruff passes for all new Python files; git diff has no whitespace errors. The existing operate.py has pre-existing lint findings and was not broadly reformatted.

## What remains before making a benefit claim

Run the included real-QA driver on a fixed existing LightRAG index and held-out questions with a configured answer model. It measures exact match/token F1 for factual answers, exports contexts for blinded groundedness review, records token use, and reports selector and retrieval latency. Keep the same entity/relation/total caps across arms. Compare SteinerPy against both relevance-only packing and soft greedy. Add quality evaluation for multi-hop and disconnected-evidence questions, report paired uncertainty at the question level, and measure concurrency before production adoption.

The prototype is ready for feedback on the integration and experiment design. These measurements do not establish a production benefit.

Historical reproduction: use the initial PR commit above and its dependency instructions, then run the two commands below. The current branch's ablation command instead runs the revised six-arm experiment with chunk reranking enabled.

```bash
.venv/bin/python -m lightrag.evaluation.steiner_ablation --repeats 5 --budgets 1200 1600
.venv/bin/python -m lightrag.evaluation.steiner_scaling --repeats 5
```

Measured with Python 3.12.14 on Linux x86_64; SteinerPy 1.0.19 at `9d8cde91bb99029bc956b1a94d54bba865ffc58e`, LightRAG base `d964d92b1018c27983d1dcf6ca19ebbaebeb262e`, NetworkX 3.6.1, tiktoken 0.14.0, HiGHS Python package 1.15.1, and SciPy 1.18.1.

The test file skips only cases that actually need SteinerPy when the optional package is absent. Install the optional dependency to run the full selector test subset.
