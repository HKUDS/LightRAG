# LightRAG Entity Extraction Full-Service Comparison

## 1. Purpose

This document is the latest full-service comparison under the new online configuration requested for upstream-style validation.

It compares two real end-to-end service runs on the same long PDF input:

- `Original PR #2864` as-is
- `Best combo on test branch`

The goal is to answer one practical question:

- under the new OpenAI-compatible runtime setup, does the landed best combo still outperform the original PR in a real LightRAG service pipeline?

## 2. Test Conditions

- Input document: `Attention Is All You Need.pdf`
- Service mode: full LightRAG server pipeline
- Chunk settings: `CHUNK_SIZE=450`, `CHUNK_OVERLAP_SIZE=50`
- Expected chunk count: `21`
- LLM endpoint: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- LLM model: `qwen3.5-flash`
- Embedding endpoint: `https://openrouter.ai/api/v1`
- Embedding model: `openai/text-embedding-3-large`
- Embedding dimension: `3072`
- Same host, same document, same chunking, same service flow

Embedding model choice rationale:

- OpenRouter's embeddings API explicitly supports `openai/text-embedding-3-large`
- the model page documents a `3072`-dimensional output
- compared with `text-embedding-3-small`, it is the higher-quality choice for retrieval and graph construction workloads

References:

- [OpenRouter Embeddings API](https://openrouter.ai/docs/api-reference/embeddings)
- [OpenRouter: openai/text-embedding-3-large](https://openrouter.ai/openai/text-embedding-3-large)
- [OpenRouter: text-embedding-3-large vs text-embedding-3-small](https://openrouter.ai/compare/openai/text-embedding-3-large/openai/text-embedding-3-small)

## 3. Compared Variants

### Original PR #2864

- `entity_extraction` uses strict `Pydantic schema`
- `entity_extraction_json_user_prompt` does **not** re-insert `Entity Types` into the user prompt

### Best combo on test branch

- `entity_extraction` uses `json_object`
- `entity_extraction_json_user_prompt` re-inserts `Entity Types`

## 4. Main Results

| Variant | Service status | Doc chunks | Contiguous | Graph entities | Graph relations | Graph total | Raw total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original PR #2864 | `success` | 21 | Yes | 433 | 408 | 841 | 976 |
| Best combo on test branch | `success` | 21 | Yes | 443 | 463 | 906 | 1019 |

## 5. Key Findings

### A. Under the new configuration, both variants complete successfully

This is different from the earlier `kimi + doubao embedding` experiment, where the original PR could fail early because of schema-related issues.

Under the current setup:

- `qwen3.5-flash` produces outputs stable enough for the original PR to finish
- `openai/text-embedding-3-large` also runs cleanly with the correct `3072`-dimensional configuration

Practical implication:

- this comparison is no longer just “failed vs success”
- it is now a true quality comparison under a stable online environment

### B. The best combo still wins in real service

Compared with the original PR, the best combo on the test branch improves:

- entities: `443 - 433 = +10`
- relations: `463 - 408 = +55`
- graph total volume: `906 - 841 = +65`
- raw extraction total: `1019 - 976 = +43`

The main gain comes from relations, not entities.

Interpretation:

- `json_object + type guidance` still delivers a better final graph
- the improvement is especially strong on relationship retention and consolidation

### C. The new result changes the failure story, but not the recommendation

The previous environment showed:

- original PR can fail hard because strict schema validation is fragile

The current environment shows:

- with a different LLM, the original PR can run through successfully

So the updated interpretation is:

- the original PR's failure risk is model-dependent
- the best combo is more robust across environments
- even when the original PR no longer hard-fails, the best combo still yields a larger graph

## 6. Why the Best Combo Still Wins

The best combo combines two advantages:

1. `json_object` ingress is more tolerant than strict schema parsing
2. `Entity Types` guidance in the user prompt improves extraction coverage and categorization stability

In this new setup, the first advantage mainly improves robustness margin rather than deciding pure pass/fail.
The second advantage still improves extraction completeness, especially for relations.

## 7. Comparison With Earlier Full-Service Results

This repository also contains earlier full-service runs under different model settings.

Examples:

- earlier isolated best combination result: `673`
- earlier test-branch best-combo result under a different provider mix: `615`
- current latest result under `qwen3.5-flash + text-embedding-3-large`: `906`

These numbers should not be compared as if they came from the same environment.
They reflect different provider combinations and therefore different LLM output behavior.

The stable cross-run conclusion is not the absolute total count.
The stable conclusion is:

- `json_object + Entity Types guidance` remains the strongest combination among the tested options

## 8. Supplemental Note on the Diagnostic Continue Variant

There is also a diagnostic variant that changes only one behavior:

- `failed chunk -> warning -> continue`

That result is still useful as historical evidence that document-level abort policy can materially affect observed outcomes in stricter environments.

Diagnostic result:

- `310` entities
- `282` relations
- `592` total graph items

Source:

- `fullsvc_diag_continue_run/diagnostic_result.json`

This is now supplementary context, not the main conclusion for the latest configuration.

## 9. Bottom-Line Conclusion

Under the latest requested online configuration:

- `Original PR #2864` succeeds end-to-end
- `Best combo on test branch` also succeeds end-to-end
- `Best combo on test branch` produces the better graph:
  - `906 > 841`
  - with the largest gain coming from relations: `463 > 408`

Therefore, the latest recommendation remains:

- use `json_object` for entity extraction ingress
- keep `Entity Types` guidance in the user prompt

because even in an environment where the original PR no longer crashes, the best combo still delivers better real-service extraction quality.

## 10. Result File References

- `fullsvc_original_vs_best_combo/original_vs_best_combo_results.json`
- `fullsvc_diag_continue_run/diagnostic_result.json`
- `fullsvc_attention_runs/full_service_attention_results.json`
