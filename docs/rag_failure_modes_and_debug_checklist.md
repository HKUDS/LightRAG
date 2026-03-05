<a id="top"></a>
<!-- Source note: Derived from an open source RAG 16 problem map. -->

# RAG Failure Modes and Debug Checklist

This page is a docs only troubleshooting reference for LightRAG users.
It is designed to help you quickly identify which layer is failing, collect the right evidence, and write high signal bug reports that are easy to review.

Quick navigation
[Quick triage](#quick-triage) | [Bug report template](#bug-report-template) | [Failure map table](#failure-map-at-a-glance) | [Mode specific checks](#mode-specific-checks) | [Failure modes 1 to 16](#failure-modes-1-to-16) | [Appendix](#appendix)

---

<a id="scope"></a>

## Scope

What this is
A fast diagnostic checklist and failure map for RAG systems, adapted for LightRAG usage patterns.

What this is not
This is not a benchmark, not a tuning guide, and not a replacement for maintainers triaging issues.
If you only need one rule, it is this: verify retrieval context first, before blaming the model.

Back to top: [top](#top)

---

<a id="quick-triage"></a>

## Quick triage

If you have 60 seconds, do these four checks in order.

1. Confirm which storage you are querying
Make sure the `working_dir` is the one you indexed, not an old directory or a different environment.

2. Inspect the retrieved context directly
Use the context only path when possible.
In LightRAG, this is commonly done via a query parameter such as `only_need_context=True`.
If the returned context is wrong, stop and debug ingestion or retrieval first.

3. Compare query modes on the same question
Run the same prompt using more than one `mode`, such as `naive`, `local`, `global`, `hybrid`, and if applicable `mix`.
If one mode works and others fail, you have narrowed the failure family immediately.

4. Freeze a minimal reproduction
Pick one short document snippet and one question.
If you cannot reproduce on a minimal case, the issue is usually data scale, stale storage, or pipeline ordering.

Back to top: [top](#top)

---

<a id="bug-report-template"></a>

## Bug report template

Copy paste the template below into your GitHub issue.
If you include the retrieved context for the failing query, triage time usually drops sharply.

```text
### Environment
- LightRAG version:
- OS:
- Python:
- LLM provider and model:
- Embedding model:
- Storage backend and working_dir:

### How you run it
- CLI or script or server:
- Key configs:

### Query settings
- QueryParam.mode:
- QueryParam.only_need_context:
- QueryParam.only_need_prompt:
- QueryParam.response_type:
- QueryParam.stream:
- QueryParam.top_k:
- QueryParam.chunk_top_k:
- QueryParam.max_entity_tokens:
- QueryParam.max_relation_tokens:
- QueryParam.max_total_tokens:
- QueryParam.enable_rerank:
- QueryParam.user_prompt (if set):

### Minimal reproduction
- Document snippet (small):
- Question:
- Expected:
- Actual:

### Retrieved context
- Paste the returned context for the failing query if possible

### Logs
- Ingestion logs:
- Query logs:
````

Back to top: [top](#top)

---

<a id="failure-map-at-a-glance"></a>

## Failure map at a glance

Legend

* `[IN]` Input and Retrieval
* `[RE]` Reasoning and Planning
* `[ST]` State and Context
* `[OP]` Infra and Deployment

Tags

* `{OBS}` Observability and evaluation
* `{SEC}` Security
* `{LOC}` Language or OCR

Use the table as your entry point.
Click the mode number to jump to details.

| No | Lane       | Typical symptom                      | First checks                                                        | Jump         |
| -- | ---------- | ------------------------------------ | ------------------------------------------------------------------- | ------------ |
| 01 | [IN] {OBS} | wrong chunks, irrelevant retrieval   | verify working_dir, inspect context only, compare modes             | [go](#no-01) |
| 02 | [RE]       | context is right but answer is wrong | ask for quotes, reduce task, ensure evidence exists                 | [go](#no-02) |
| 03 | [RE] {OBS} | multi step drift                     | split steps, check each hop evidence, compare hybrid vs local       | [go](#no-03) |
| 04 | [RE]       | overconfident guessing               | require evidence, enforce citations style, check prompt constraints | [go](#no-04) |
| 05 | [IN] {OBS} | embedding matches but meaning is off | verify embed model, dimension, casing language consistency          | [go](#no-05) |
| 06 | [RE] {OBS} | collapse then stuck, cannot recover  | reduce complexity, confirm mode fits task, isolate one hop          | [go](#no-06) |
| 07 | [ST]       | memory or continuity missing         | confirm your app passes history, do not assume persistence          | [go](#no-07) |
| 08 | [IN] {OBS} | black box debugging                  | capture context, mode, params, storage identity, logs               | [go](#no-08) |
| 09 | [ST]       | unstable, incoherent output          | reduce randomness, check context overload, check length limits      | [go](#no-09) |
| 10 | [RE]       | flat, literal responses              | check system prompt, verify you are not in context only mode        | [go](#no-10) |
| 11 | [RE]       | symbolic tasks fail                  | provide anchors, definitions, test smaller symbolic query           | [go](#no-11) |
| 12 | [RE]       | self reference loops                 | set bounds, request assumptions, add stop rules                     | [go](#no-12) |
| 13 | [ST] {OBS} | multi agent chaos                    | lock roles, ensure one final writer, isolate tool boundaries        | [go](#no-13) |
| 14 | [OP]       | startup order bugs                   | ensure index and storage ready before first query                   | [go](#no-14) |
| 15 | [OP]       | deadlock or circular waits           | lower concurrency, verify external services, check locks            | [go](#no-15) |
| 16 | [OP] {OBS} | deploy works but first call fails    | verify secrets, config parity, version pinning                      | [go](#no-16) |

Back to top: [top](#top)

---

<a id="mode-specific-checks"></a>

## Mode specific checks

This section helps when one mode works but others do not.

Case A. `naive` works but `local` or `global` or `hybrid` fails
Likely causes include incomplete graph build, extraction tasks not finished, or mixed stale storage.

Minimum checks

* Rebuild on a tiny dataset and confirm graph construction completes.
* Confirm you did not reuse a partially built working_dir.
* Compare the returned context across modes to see where divergence begins.

Case B. `only_need_context=True` returns an answer instead of context
This is often a version or integration mismatch.
Record LightRAG version and paste the exact QueryParam used.

Case C. `hybrid` is slow or returns too much context
Reduce top_k or chunk_top_k, and confirm max_total_tokens is not exceeding your model context window.
Confirm your LLM context window is not being exceeded.

Back to top: [top](#top)

---

<a id="failure-modes-1-to-16"></a>

## Failure modes 1 to 16

Navigation
[Table](#failure-map-at-a-glance) | [No 01](#no-01) | [No 02](#no-02) | [No 03](#no-03) | [No 04](#no-04) | [No 05](#no-05) | [No 06](#no-06) | [No 07](#no-07) | [No 08](#no-08) | [No 09](#no-09) | [No 10](#no-10) | [No 11](#no-11) | [No 12](#no-12) | [No 13](#no-13) | [No 14](#no-14) | [No 15](#no-15) | [No 16](#no-16)

---

<a id="no-01"></a>

### No 01. [IN] Wrong retrieval or chunk drift {OBS}

Typical symptom
The model answers confidently, but the retrieved context is irrelevant, outdated, or mismatched in granularity.

Likely root causes

* Query hits the wrong storage or stale working_dir.
* Ingestion did not complete, or you indexed a different dataset than you think.
* Chunking strategy does not match the questions you ask.

Minimum checks

* Run the same question with `only_need_context=True` and inspect the retrieved context.
* Confirm `working_dir` and storage identity.
* Compare `naive` vs `local` vs `global` vs `hybrid` on the same question.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-02"></a>

### No 02. [RE] Interpretation collapse

Typical symptom
The context contains the answer, but the generated response contradicts it.

Likely root causes

* The model is not grounding in the retrieved context.
* The prompt asks for synthesis without forcing evidence.

Minimum checks

* Ask for exact quotes or extracted facts from the retrieved context.
* Reduce the question to a single fact lookup.
* If quoting works but synthesis fails, the issue is reasoning and prompt constraints rather than retrieval.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-03"></a>

### No 03. [RE] Long chain drift {OBS}

Typical symptom
The answer starts correct, then drifts across multi step reasoning.

Likely root causes

* Missing intermediate evidence.
* Context coverage changes across steps.

Minimum checks

* Split the task into smaller questions.
* Verify evidence for each hop, not only the final hop.
* Compare `hybrid` vs `local` to see whether local coverage is sufficient.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-04"></a>

### No 04. [RE] Overconfidence and bluffing

Typical symptom
Confident answers without support.

Likely root causes

* Prompt allows guessing.
* Missing citation or evidence requirement.

Minimum checks

* Require evidence, quotes, or references from retrieved context.
* Add a constraint: if not found in context, say not found.
* Verify you are not accidentally querying without retrieval context.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-05"></a>

### No 05. [IN] Semantic mismatch between meaning and embedding {OBS}

Typical symptom
Nearest neighbors are technically similar in vector space but wrong in meaning.

Likely root causes

* Embedding model changed after indexing.
* Dimension mismatch or mixed tokenization and casing across datasets.
* Multilingual mixing without consistent embedding choice.

Minimum checks

* Record embedding model name and dimension and verify it matches the indexed store.
* If you changed embedding settings, rebuild the store and retest.
* Test with a tiny dataset in one language first.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-06"></a>

### No 06. [RE] Collapse with no recovery {OBS}

Typical symptom
The system gets stuck, repeats, or fails to converge.

Likely root causes

* The task is too open ended or unbounded.
* The selected mode does not fit the question style.

Minimum checks

* Reduce scope to one hop.
* Switch modes and compare returned context shape.
* Add bounded constraints in the prompt, such as maximum steps or required assumptions.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-07"></a>

### No 07. [ST] Memory or continuity breaks

Typical symptom
Across turns or sessions, the system forgets what you expected it to remember.

Likely root causes

* The application does not pass history into the query.
* The integration assumes server side memory that does not exist.

Minimum checks

* Confirm what your integration actually passes per turn.
* Test with explicit context included in the prompt to isolate whether memory is the issue.
* If you need persistent memory, implement it explicitly at the app layer.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-08"></a>

### No 08. [IN] Debugging is a black box {OBS}

Typical symptom
You cannot tell whether ingestion, retrieval, or generation failed.

Likely root causes

* Missing logs and missing captured context.
* Reproduction steps not pinned.

Minimum checks

* Always record: working_dir, mode, QueryParam values, and retrieved context.
* Include ingestion logs and query logs when reporting issues.
* Use a minimal reproduction snippet.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-09"></a>

### No 09. [ST] Instability and incoherence

Typical symptom
The answer is inconsistent or incoherent across runs.

Likely root causes

* High randomness on the LLM side.
* Overloaded context or contradictory chunks.

Minimum checks

* Lower randomness in your LLM settings if you control them.
* Reduce top_k and context limits to avoid overload.
* Verify the retrieved context is stable across restarts.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-10"></a>

### No 10. [RE] Creative freeze

Typical symptom
Flat responses even when you expect synthesis.

Likely root causes

* Prompt is too restrictive.
* You are using context only paths when you want an answer.

Minimum checks

* Confirm `only_need_context` is false when you want generated output.
* Review the system prompt and response style instructions.
* If retrieval is correct, this is mostly a prompt design issue.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-11"></a>

### No 11. [RE] Symbolic and abstract failure

Typical symptom
The system fails on logic, math, or symbolic reasoning prompts.

Likely root causes

* Missing definitions or anchors in retrieved context.
* The question expects reasoning without evidence.

Minimum checks

* Provide explicit anchors, definitions, or constraints.
* Test a smaller symbolic query.
* Confirm retrieval includes the relevant definitions.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-12"></a>

### No 12. [RE] Self reference and recursion traps

Typical symptom
Circular answers, paradox loops, or unbounded recursion.

Likely root causes

* The prompt has no stopping rule.
* The question is inherently self referential.

Minimum checks

* Add a strict bound: assumptions, maximum steps, and explicit stopping conditions.
* Ask for concrete examples and avoid circular definitions.
* If needed, request a short answer with stated limits.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-13"></a>

### No 13. [ST] Multi agent chaos {OBS}

Typical symptom
Multiple agents or tools overwrite each other and the final output becomes inconsistent.

Likely root causes

* Roles are not locked.
* Message routing and state passing are ambiguous.

Minimum checks

* Ensure one final writer agent produces the final answer.
* Lock roles and delimit tool outputs.
* Isolate LightRAG calls and compare their retrieved context across agents.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-14"></a>

### No 14. [OP] Bootstrap ordering

Typical symptom
Works locally sometimes, fails on first run or after restart.

Likely root causes

* Query starts before ingestion and indexing finished.
* Storage paths are not ready when the server accepts requests.

Minimum checks

* Ensure ingestion and storage initialization completes before query.
* Verify permissions and file paths for working_dir.
* Try a minimal dataset and confirm deterministic startup.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-15"></a>

### No 15. [OP] Deployment deadlock

Typical symptom
The system hangs, times out, or becomes stuck under concurrency.

Likely root causes

* Circular waits, locks, or resource contention.
* External services are slow or unavailable.

Minimum checks

* Reduce concurrency to isolate deadlock.
* Verify external services availability and timeouts.
* Capture logs around the hang point.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="no-16"></a>

### No 16. [OP] Pre deploy collapse {OBS}

Typical symptom
Everything looks correct but deployed runs fail on first call.

Likely root causes

* Secret missing, config mismatch, version skew.
* Different storage path or different embed model than local.

Minimum checks

* Compare config parity between local and deploy.
* Verify secrets and API keys are present before first request.
* Confirm embed model and storage are consistent with the indexed artifacts.

Back to: [table](#failure-map-at-a-glance) | [top](#top)

---

<a id="appendix"></a>

## Appendix

### A. What to record when you say “it does not work”

At minimum, record these together as a single block:

* working_dir
* mode
* only_need_context
* top_k
* retrieved context
* LightRAG version

This is the fastest path to a useful maintainer response.

### B. Footer navigation

[Quick triage](#quick-triage) | [Bug report template](#bug-report-template) | [Failure map table](#failure-map-at-a-glance) | [Mode specific checks](#mode-specific-checks) | [Failure modes 1 to 16](#failure-modes-1-to-16) | [Top](#top)
