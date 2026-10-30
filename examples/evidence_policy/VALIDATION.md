# Validation record and scope

The live-model results below were recorded on **2026-10-29 (Asia/Shanghai)**.
The upstream preparation on 2026-10-30 does not repeat paid model calls.

## Recorded live questions: 5/5

Environment: Python 3.11.9, OpenAI client 2.54.0, fastembed 0.9.1, and
onnxruntime 1.31.1. Generation used DeepSeek `deepseek-flash`; embeddings used
local `BAAI/bge-small-zh-v1.5` vectors (512 dimensions).

Both documents went through `ainsert`. The catalog was built from their real
chunk rows, and questions used `aquery_llm(mode="naive")`.

| Scenario | Requested date | Initial versions | Selected version | Recorded answer |
| --- | --- | --- | --- | --- |
| F01 | 2026-08-01 | V-old, V-new | V-new | 270 days; 180 days labelled historical |
| F02 | 2026-08-01 | V-old | Fetch V-new, select V-new | 270 days |
| F03 | 2026-03-01 | V-old, V-new | V-old | 180 days |
| F04-pre | 2026-06-15 | V-old, V-new | V-old | 180 days |
| F04-post | 2026-08-01 | V-old, V-new | V-new | 270 days |

The [complete record](recorded/2026-10-29.json) includes initial and fetched
chunks, actual model context, answers, checks and diagnostics. Its published
copy replaces workstation-specific directories with `<local-workdir>` and
updates the demo-document pointer. Case contents and summary are unchanged;
`publication_note` retains the source record's SHA-256.

The final recorded run reused existing ingestion, completed five chat calls,
and exited with code 0. Automated checks cover version selection, the expected
number, model calls, and F02 expansion. They do not establish general answer
semantics, citation correctness, or independent legal review.

The first run exposed catalog injection through `asdict` and a missing requested
date in model context. The implementation now excludes the catalog from the
dataclass traversal on a shallow copy, preserves catalog identity, and renders
`requested_as_of`. The table reports the subsequent successful run.

## Recorded related regressions: 45 passed

The original test selection covered catalog injection, query integration,
context dates, cache identity, and policy runtime checks:

```bash
PYTHONDONTWRITEBYTECODE=1 ./scripts/test.sh tests/evidence_policy \
  -q -p no:cacheprovider --capture=sys \
  --basetemp=/private/tmp/3037-live-date-regression-20261029
```

It completed with **45 passed**. In a fresh environment, install the project's
`pytest` extra before running tests:

```bash
python -m pip install -e ".[pytest]"
```

## Upstream preparation checks: 2026-10-30

These checks use the existing Python 3.11 environment. No live model request,
new ingestion against a paid service, or deployment was performed.

The feature, positional-compatibility, and docstring-contract selection passed:

```bash
PYTHON="$PWD/.venv/bin/python" PYTHONDONTWRITEBYTECODE=1 \
  PYTHON_DOTENV_DISABLED=1 ./scripts/test.sh \
  tests/evidence_policy tests/test_dataclass_positional_compatibility.py \
  tests/test_docstring_budget.py -q -p no:cacheprovider --capture=sys
```

Result: **68 passed**, including the same **45 evidence-policy tests**.

The separate offline wiring check also exited with code 0 and `wiring ok`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHON_DOTENV_DISABLED=1 \
  .venv/bin/python examples/evidence_policy/smoke_plumbing.py
```

All five scenarios reached the expected selected version, including F02's
catalog expansion. Its embeddings and answer model are deterministic stubs;
these are structural checks rather than new live-answer results.

Because the proposal touches `base.py`, the offline suite was also run on both
the untouched upstream `main` snapshot (`aaebfe1bb`) and the feature snapshot,
using the same interpreter, disabled dotenv loading, and process permissions:

```bash
PYTHON="$PWD/.venv/bin/python" PYTHONDONTWRITEBYTECODE=1 \
  PYTHON_DOTENV_DISABLED=1 ./scripts/test.sh tests -m offline \
  -n auto --dist worksteal --ignore=tests/evaluation/authority_aware \
  -q -p no:cacheprovider
```

The untracked local `authority_aware` prototype is outside the proposal and was
excluded from collection. The baseline used the same pytest selection and the
feature checkout's interpreter against an exported upstream source snapshot.

| Snapshot | Passed | Failed | Errors | Skipped |
| --- | ---: | ---: | ---: | ---: |
| Upstream baseline | 5806 | 55 | 101 | 93 |
| Feature snapshot | 5851 | 55 | 101 | 93 |

The failure/error test identities and outcome types match exactly; the feature
adds 45 passing tests. Both suite runs exited nonzero. Missing optional modules
include `asyncpg`, `ollama`, `neo4j`, `redis`, and other provider/storage SDKs.
These results establish the comparison in this environment, not a fully passing
offline suite. An earlier sandboxed run also had multiprocessing failures;
the table uses the subsequent runs with local process communication permitted.

All five configured commit-stage hooks passed over all tracked files:
trailing whitespace, final newlines, requirements sorting, Ruff formatting, and
Ruff lint (`v0.15.17`). The first invocation could not fetch the separately
configured manual-only `check-manifest` repository because GitHub was unreachable.
The successful invocation used a temporary copy of the configuration containing
the five unchanged commit-stage hooks. The manual `check-manifest` hook was not run.

## What these results support

For two fictional documents and one explicit version chain on the `naive`
path, the demo selects an original clause or its amendment by effective date,
and fetches the amendment when only the original is initially retrieved.

F02 constructs the retrieval miss by removing a vector entry while retaining
the real text in KV storage. These results do not measure natural retrieval
miss rates or establish large-scale regulatory quality, other models or
storage backends, live `mix` behavior, or upstream acceptance.

The `authority-v1` [core contract](../../docs/design/EvidencePolicyContract.md#known-limitations)
lists unresolved supersession/authority cases, seed-dependent unknown-date
handling, dependency labels, final prompt budgeting, mutable catalog mappings,
entry-point coverage, and answer-publication checks. Those limitations remain
part of this experimental proposal.
