# LightRAG Entity Extraction Prompt/JSON Controlled Benchmark

> Test date: 2026-04-07  
> Target branch: [HKUDS/LightRAG `feat/Enhance_entity_extraction_stability`](https://github.com/HKUDS/LightRAG/tree/feat/Enhance_entity_extraction_stability)  
> Benchmark script: `reproduce/attention_entity_extraction_controlled_benchmark.py`

---

## 1. Goal

This document reports a single controlled experiment designed to answer two isolated questions on the original PR #2864 branch:

1. Does adding explicit entity type guidance to the `user prompt` improve extraction volume?
2. Between `json_object` and `Pydantic schema`, which JSON mode is more tolerant to malformed or partially truncated LLM outputs?

The benchmark also adds explicit JSON diagnostics so we can tell whether valid extractions were dropped because of JSON formatting or schema validation failures.

---

## 2. Experiment Design

### 2.1 Input Document

- Document: `Attention Is All You Need.pdf`
- Parser: `pypdf`
- Extracted text length: about `32,633` characters

To satisfy the requirement of testing on a continuous article with more than 20 blocks, this benchmark fixes:

- `chunk_token_size = 450`
- `chunk_overlap = 50`

The document is therefore split into:

- **21 actual chunks**
- **continuous chunk order**
- `chunk_order_index = 0..20`

This is not an artificial collection of short passages. It is one continuous paper, chunked into 21 sequential units.

### 2.2 Controlled Variables

The following settings are held constant across all variants:

- Same repo baseline: original PR #2864 branch state
- Same PDF input
- Same 21 chunk boundaries
- Same system prompt
- Same few-shot examples
- Same model
- Same temperature (`0.0`)
- Same `max_tokens` (`1800`)

### 2.3 Independent Variables

Only two factors change:

1. Whether `entity_types_guidance` is injected into the `user prompt`
2. Whether structured output uses `json_object` or `Pydantic schema`

This produces four variants:

| Variant | Type Guidance | JSON Mode |
|---------|---------------|-----------|
| `schema_no_type_guidance` | No | Pydantic schema |
| `schema_with_type_guidance` | Yes | Pydantic schema |
| `json_object_no_type_guidance` | No | `json_object` |
| `json_object_with_type_guidance` | Yes | `json_object` |

---

## 3. JSON Diagnostics

The benchmark does not only compare final extraction counts. It also inspects the raw LLM response for every chunk.

### 3.1 Diagnostic Flow

For each chunk:

1. Read the raw LLM response text
2. In `json_object` mode:
   - try strict `json.loads()`
   - if strict parsing fails, try `json_repair`
3. In `schema` mode:
   - run `Pydantic schema` parsing
   - if schema parsing fails, record `schema_parse_failed`
   - then run a diagnostic `json_object` probe with the same prompt
   - if the probe recovers entities or relationships, record `schema_parse_failed_salvageable`

### 3.2 Warning Types

| Warning | Meaning |
|---------|---------|
| `malformed_json_repaired` | Raw output was not valid strict JSON but could be repaired |
| `schema_parse_failed` | Pydantic schema validation/parsing failed |
| `schema_parse_failed_salvageable` | Schema failed, but a diagnostic `json_object` probe recovered parseable extractions |

This allows the benchmark to distinguish:

- model did not extract anything
- model extracted something but the strict structured-output path dropped it

---

## 4. Results

### 4.1 Summary Table

| Variant | Raw Extraction Volume | Deduplicated Volume | Warnings |
|---------|:---------------------:|:-------------------:|----------|
| `schema_no_type_guidance` | 280 | 242 | `schema_parse_failed=6`, `schema_parse_failed_salvageable=6`, `malformed_json_repaired=4` |
| `schema_with_type_guidance` | 316 | 270 | `schema_parse_failed=4`, `schema_parse_failed_salvageable=4`, `malformed_json_repaired=4` |
| `json_object_no_type_guidance` | 489 | 406 | `malformed_json_repaired=4` |
| `json_object_with_type_guidance` | 452 | 381 | `malformed_json_repaired=3` |

### 4.2 Single-Variable Deltas

Using `schema_no_type_guidance` as the reference:

| Single Change | Raw Volume Delta | Deduplicated Delta |
|---------------|:----------------:|:------------------:|
| Add type guidance only | +36 | +28 |
| Switch `schema` to `json_object` only | +209 | +164 |

This shows that:

- JSON mode has a much larger effect than type guidance
- type guidance helps, but it is not the dominant factor

### 4.3 Direct Evidence of Dropped Extractions

In `schema_no_type_guidance`, the benchmark recorded:

- `schema_parse_failed = 6`
- `schema_parse_failed_salvageable = 6`

In other words, every schema failure in that variant was confirmed recoverable by the diagnostic `json_object` probe.

Representative failure patterns included:

1. missing `relationship_description`
2. response cut off near the token limit
3. raw JSON malformed but still recoverable with `json_repair`

This means some chunks were not empty because the model failed semantically. They were empty because the strict schema path discarded otherwise usable output.

---

## 5. Interpretation

### 5.1 JSON Mode Matters More Than Prompt Type Guidance

The clearest outcome of this experiment is:

> `json_object` is substantially more robust than `Pydantic schema`, and the difference mainly comes from avoiding chunk-level total drops.

Under the same “no type guidance” setting:

- `schema_no_type_guidance`: raw volume `280`
- `json_object_no_type_guidance`: raw volume `489`

That is roughly:

- `+74.6%` raw volume
- `+67.8%` deduplicated volume

This is not a small fluctuation. It is a structural difference.

### 5.2 Type Guidance Helps, But It Is Secondary

Type guidance still has a measurable effect:

- under `schema`: `280 -> 316`
- `schema_parse_failed`: `6 -> 4`

So restoring `entity_types_guidance` in the `user prompt` does improve stability and extraction volume.

However, its effect is clearly smaller than switching away from strict schema parsing.

### 5.3 `json_object_with_type_guidance` Is Not the Highest-Volume Variant

On this paper:

- `json_object_no_type_guidance = 489`
- `json_object_with_type_guidance = 452`

So type guidance is not a guaranteed “more volume” switch. It changes extraction behavior and may make outputs more conservative or more regularized.

The supported conclusion is therefore:

> Type guidance changes extraction behavior, but JSON mode is the primary determinant of robustness.

---

## 6. Conclusion

### 6.1 Findings

| Finding | Verdict |
|---------|---------|
| `Attention Is All You Need.pdf` satisfies the 20+ continuous chunk requirement | Confirmed |
| The benchmark checks raw JSON failures and outputs explicit warnings | Confirmed |
| Strict schema causes recoverable extractions to be discarded | Confirmed |
| `json_object` is the primary source of robustness gain over `schema` | Confirmed |
| Adding entity type guidance helps, but less than changing JSON mode | Confirmed |

### 6.2 Recommendation

If the goal is to improve extraction stability and preserve extraction volume on the PR #2864 branch, the recommended order is:

1. **Prefer `json_object` over strict `Pydantic schema` in the extraction path**
2. **Keep JSON diagnostics and warning output**
3. **Preserve entity type guidance in the `user prompt`, but treat it as a secondary factor**

In short:

> Fix chunk loss caused by strict JSON/schema handling first, then iterate on prompt guidance.

---

## 7. Engineering Takeaway: Which JSON Style Is Better for Fault Tolerance?

The engineering conclusion is straightforward:

> If the priority is fault tolerance and minimizing chunk-level total loss, `json_object` is better than `Pydantic schema`.

### 7.1 Why `json_object` Is More Tolerant

- It constrains the model to return a JSON object
- But it is more forgiving when fields are missing, content is partially truncated, or syntax is slightly malformed
- Even when strict parsing fails, `json_repair` can often recover usable entities and relationships
- Recovered output can still contribute to extraction volume instead of being discarded wholesale

This makes `json_object` better suited for the **primary extraction path**.

### 7.2 Why `Pydantic schema` Is More Fragile

- It requires not only JSON, but a fully valid schema-conforming structure
- A missing required field such as `relationship_description`
- Or a response truncated near the token limit
- Can cause the entire chunk to fail schema parsing

This happened multiple times in this benchmark, and several failed chunks were later shown to be recoverable.

This makes strict schema parsing more suitable for a **post-processing validation layer**, not for the first gate of extraction.

### 7.3 Recommended Strategy

A more robust production strategy is:

1. Use `json_object` for extraction
2. Attempt strict JSON parsing
3. If strict parsing fails, apply `json_repair`
4. Emit warnings for malformed or incomplete output
5. Apply targeted fallbacks for missing fields instead of dropping the entire chunk

That is:

> **lenient ingress, strict egress**

rather than:

> **strict ingress, strict egress**

The former matches real LLM output behavior much better and avoids losing semantically valid but slightly dirty responses.

---

## 8. Reproduction

Key file:

- `reproduce/attention_entity_extraction_controlled_benchmark.py`

Environment:

- Set `BENCH_LLM_API_KEY` (or `OPENAI_API_KEY`)
- Provide a local copy of `Attention Is All You Need.pdf`

Run all variants:

```bash
cd LightRAG
BENCH_LLM_API_KEY=... \
python reproduce/attention_entity_extraction_controlled_benchmark.py \
  --pdf "/path/to/Attention Is All You Need.pdf"
```

Run schema-only variants:

```bash
cd LightRAG
BENCH_LLM_API_KEY=... \
python reproduce/attention_entity_extraction_controlled_benchmark.py \
  --pdf "/path/to/Attention Is All You Need.pdf" \
  --run-variants "schema_no_type_guidance,schema_with_type_guidance"
```
