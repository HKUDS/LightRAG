# Evidence Policy Contract

**Release scope (2026-10-29): a limited `authority-v1` feature demonstration.**
The two-document `naive` demo completed five real questions and 45 related
regression tests; its [run instructions](../../examples/evidence_policy/README.md)
and [validation record](../../examples/evidence_policy/VALIDATION.md) describe
what was exercised. This does not establish general production correctness.
An earlier synthetic review found decision, evidence-label, budget and
answer-publication gaps outside that demonstration. They are retained below
under **Known limitations**, rather than treated as fixed by the demo results.

Read this before changing `lightrag/evidence_policy/`, the policy branch of
`lightrag/operate.py`, or the three opt-in `QueryParam` fields.

## Scope

An answer about a clause depends on **which version of that clause is in force
at the date asked about**, not on which chunk a retriever ranked first. When a
document is amended, the old text stays in the corpus, keeps producing graph
edges, and often ranks *higher* than its own amendment because it is longer and
better connected. Similarity search cannot resolve that: both passages are
about the same subject, and one of them is simply no longer the rule.

This policy answers that question before retrieval scores are consulted. It is
opt-in, covers `naive` and `mix` only, and is off unless a request names it.

Enabling it:

```python
from lightrag.evidence_policy import load_catalog

rag = LightRAG(..., evidence_catalog=load_catalog("catalog.json"))
rag.query(
    "标准成员办理申请的期限是多少天？",
    param=QueryParam(
        mode="mix",
        evidence_policy="authority-v1",
        as_of="2026-08-01",
        scope_keys=[{...}],   # optional: answer even if retrieval found nothing
    ),
)
```

A request cannot name an authority class, a catalog, or a policy version this
build does not implement. Those are operator decisions, injected on the
instance.

## Data model

The unit is a **clause version**, never a whole file: article 8 changing does
not touch article 9, and one chunk may carry several clauses.

| Field | Meaning |
| --- | --- |
| `scope_key` | `{jurisdiction, instrument_id, article_id, subject_scope}`. Compared as four fields, never as a joined string. |
| `version_id` | Stable identity of one clause version inside a catalog. |
| `kind` | `rule`, `amendment`, `repeal`, `commentary`. Commentary never determines a rule. |
| `authority_class` | Ranked by the catalog's `policy.authority_ranking`. A class not in the ranking is untrusted and cannot establish a scope. |
| `valid_from` / `valid_to` | Half-open `[from, to)`, full `YYYY-MM-DD` dates only. `null` from means unknown; `null` to means open-ended. |
| `supersedes` | Approved supersession edges, same scope, acyclic, pointing at real versions. |
| `chunk_refs` | Real `text_chunks` IDs plus a character span and the SHA-256 of `content[span]`. |
| `requires_evidence` | Versions needed to interpret this one. They add context; they never restore an old rule's effect. |

Dates are never inferred. `"2026"` and `"2026-10"` are rejected rather than
padded to January 1st, and a document's publication or ingestion timestamp is
not an effective date.

## Resolution order

1. **Freeze the date.** `as_of` if given, otherwise today in
   `LightRAG.evidence_policy_timezone`, resolved once per request. Every stage
   of that request sees the same date.
2. **Identify scopes.** From `QueryParam.scope_keys`, or from the catalog
   versions that own the chunks the retriever actually returned. An explicitly
   named scope is resolved with zero retrieved chunks.
3. **Time before authority.** A version is a candidate only inside its own
   window. A revision that takes effect later never displaces the rule in force
   now.
4. **Supersession is an approved event.** A predecessor stops applying at the
   earlier of its approved successor's effective date and its own `valid_to`.
   If the successor later expires or is repealed, the predecessor does not come
   back; the scope can end with no applicable rule.
5. **Authority decides only contradictions.** Two effective versions of one
   scope, from different sources, with no approved relation between them and
   different answers, are `unresolved_conflict` when they share a rank. The
   resolver never breaks that tie on publication date, retrieval score, or
   document length.
6. **Evidence groups.** The winner plus everything needed to interpret it, as
   spans with a citation key, rendered for the model. The replaced wording is
   labelled `superseded_for_current_answer` and is never presented as current.
7. **Group budget.** Groups are kept whole. If their measured text does not
   fit, the answer is `insufficient_context_budget` rather than a partial chain.
   Final rendered-context overhead is not fully covered; see the limitations.

### An edition is the head of its chain

An amendment that supersedes a version and carries no value of its own is an
*edition*: "article 8 changes 180 to 270" does not restate the rule, it edits
it. It never competes with the rule it edits, but it **is** what is in force,
and the rule it replaced travels with it as required evidence. Treating
editions as non-candidates made every amended clause resolve to
`no_applicable_rule`, which is the case this policy exists to answer.

### An undated version blocks only a request that surfaced it

A version whose `valid_from` is unknown may or may not be in force, so it
blocks a definitive answer **when the retriever returned it** (or when nothing
dated applies). A catalog-wide undated notice that no query ever returns would
otherwise make every clause in the workspace unanswerable. The verdict is still
recorded as `unknown_time`, so the reason is visible rather than dropped.

### A request is resolved or it is not

If any requested scope fails, the request has no selected versions and no
required evidence. A partially verified request must not answer from the scopes
that happened to work, because the caller would mix verified and unverified
clauses in one answer.

## Failure statuses

Each status is returned as-is. None of them is a silent fallback to ordinary
semantic retrieval, and none is reported as "no results".

| Status | Cause |
| --- | --- |
| `resolved` | A version applies and its whole evidence group is bound and within budget. |
| `insufficient_evidence` | No scope identified, or the catalog holds nothing for the scope. |
| `unknown_time` | A surfaced version's effective date is unknown. |
| `no_applicable_rule` | Every version expired, was superseded, or was repealed. |
| `unresolved_conflict` | Same-rank contradiction with no approved supersession. |
| `missing_evidence` | A referenced chunk is gone, or its span no longer matches its hash. |
| `mapping_ambiguous` | Two versions claim the same span of one chunk. |
| `expansion_limit` | The closure exceeds `max_scopes` / `max_versions_per_scope` / `max_extra_chunks`. |
| `insufficient_context_budget` | The group does not fit whole. |
| `backend_error` | The chunk store raised. Distinct from "the chunk is absent". |
| `unsupported` | Mode outside `naive`/`mix`, or streaming requested. |
| `invalid_request` | Bad date, unknown policy version, missing catalog, `as_of` without the policy. |

### Diagnostic reason codes

Per-version verdicts retain the selected and excluded sets, `as_of`, and a
reason. A final status is distinct from an individual version's reason.
The vocabulary in `lightrag/evidence_policy/contracts.py` is:

- Eligibility and time: `active`, `superseded`, `not_yet_effective`, `expired`,
  `repealed`, `different_scope`, `untrusted_source`, `unknown_time`,
  `missing_metadata`.
- Mapping and resolution: `mapping_ambiguous`, `unresolved_conflict`,
  `catalog_cycle`, `missing_evidence`, `expansion_limit`.
- Processing diagnostics: `group_budget_exceeded`, `backend_error`,
  `rerank_degraded`, `cache_mismatch`, `unknown`.

The chunk store raising is different from returning an absent row. Retain
seed IDs, requested and expanded IDs, selection reasons, actual model context
and the answer when diagnosing a result. A reason-code definition does not
imply every stage emits that code. Missing observations stay unobserved;
neither a final elapsed time nor a passing count supplies a missing trace.

## Knowledge-graph isolation

Under this policy the answer context is the evidence groups. Merged entity and
relationship descriptions are **not** sent: a description merges what every
version said and cannot prove which version a sentence came from, so it would
reintroduce the stale rule that was just removed. Entities and relationships
still participate in *finding* candidates; they stop participating in
*answering*.

`raw_data.data.entities` and `.relationships` are therefore empty, and the
chunks array holds spans (`content` is the span text, not the whole chunk), so
a chunk carrying an untouched article 9 keeps that article available for its
own scope.

## Caching

The answer cache key gains the policy version, `resolved_as_of`, the catalog
revision and fingerprint, the decision status and a digest of the rendered
context. With the policy off, the extra components are an empty tuple, so
existing keys are byte-identical and no entry is invalidated.

Operators must treat the catalog's nested mappings as read-only: the frozen
dataclass does not enforce deep immutability or update the fingerprint after
an in-place mutation. The recorded live demo disables the answer cache.

## Accepted residue

These are known, deliberate, and heal or fail loud:

- **A stale catalog fails loud.** Re-ingesting or deleting a document changes
  chunk IDs and content. Binding then returns `missing_evidence`; it never
  answers from a stale span. Rebuilding the catalog is the operator's step.
- **A catalog is loaded once per instance.** Editing the file does not affect a
  running process until the catalog is reloaded and re-injected. The
  fingerprint in diagnostics names the snapshot that produced an answer.
- **One scope may hold a shared chunk.** Two versions of the same clause can
  share a chunk (an in-place amendment). Each keeps its own span; only a span
  claimed by two versions is `mapping_ambiguous`.
- **No streaming.** The policy helper refuses streaming requests. This is a
  supported-mode restriction, not proof of answer validation: the generation
  path does not call the exported citation verifier before publishing.
- **Limits are configuration, not truth.** `8` scopes, `16` versions per scope
  and `64` extra chunks are starting values. Hitting one returns
  `expansion_limit`; it never truncates a version chain and then asserts a
  current rule.

## Known limitations

These findings from the earlier v1 synthetic review remain outside this
release's two-document acceptance scope. This upload does not rerun those
probes or migrate the separate replacement prototype.

| Area | Limitation |
| --- | --- |
| Supersession eligibility | An untrusted successor may still affect chain traversal after being excluded as an answer candidate. |
| Authority conflicts | Two dominated lower-authority roots can still create a conflict even when a higher-authority root exists. |
| Unknown dates | The result can change from `resolved` to `unknown_time` when an undated version enters the seed set. This seed dependence is implemented behavior, not a general temporal-correctness guarantee. |
| Dependency labels | Every non-selected dependency is labelled as replaced wording; an explanatory note that was never superseded can therefore receive the wrong label. |
| Budget | Group measurement omits later prompt headers, instructions, separators and request-date text. The final context can exceed the nominal budget; a negative helper budget can also raise instead of producing a typed refusal. |
| Catalog mutation | Nested dictionaries remain mutable; an in-place edit can leave the recorded fingerprint unchanged. Load a fresh snapshot instead. |
| Answer publication | `resolved` describes evidence processing. The query generation path does not automatically check answer values or invoke `verify_citations`; incorrect or uncited model text can still be returned. |
| Entry-point coverage | Validation lives inside retrieval branches. `bypass` can avoid it, and `mix` can return early on empty keywords before producing policy diagnostics. Use the demonstrated `naive` path for this release. |
| Scope discovery | Inferring every retrieved scope can make an unrelated unresolved clause block the intended question. Explicit scope keys constrain the request. |
| Negative decisions | A non-resolved decision returns before chunk binding. A `no_applicable_rule` status must not be advertised as a source-verified repeal conclusion. |

The source-span binding and prompt labels help inspect evidence; they do not
establish automatic legal reasoning or guarantee that a model follows it.

## Not in this version

Automatic legal hierarchy inference, automatic supersession discovery, entity
identity resolution across instruments, graph identity rewriting, catalog
publication or migration for production workspaces, cross-backend metadata
persistence, `local`/`global`/`hybrid` modes, multi-jurisdiction conflict
resolution, retroactive effect, and `known_at` (what the system knew at the
time). Tests that feed these inputs must mark them `unsupported` rather than
infer a behaviour.

REST exposure of the policy parameters is also out of scope here: the SDK path
is implemented and tested, the HTTP surface is not.
