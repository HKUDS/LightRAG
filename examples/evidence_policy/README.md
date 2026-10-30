# Select clause versions by effective date: experimental demo

English | [简体中文](README-zh.md)

When an original clause and its amendment coexist, this demo selects the version
in force at the requested date, fetches the source spans needed to interpret it,
and passes those spans to the answer model. The SDK switch is
`QueryParam(evidence_policy="authority-v1", as_of="YYYY-MM-DD")`.

The recorded run on 2026-10-29 completed **5/5 real questions** on two fictional
documents. See the [validation record](VALIDATION.md),
[complete recorded contexts and answers](recorded/2026-10-29.json), and
[core contract, including known limitations](../../docs/design/EvidencePolicyContract.md).
This is an experimental SDK feature with a deliberately limited demonstration.

## Install

Run from a source checkout containing this change. Python 3.11 matches the
recorded environment (3.11.9):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r examples/evidence_policy/requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. The root `pyproject.toml`
provides the project dependencies; this example's `requirements.txt` pins the
OpenAI-compatible client and local embedding dependencies used in the recording.

## Run the five scenarios

Set a DeepSeek API key in the current terminal:

```bash
export LIGHTRAG_LLM_API_KEY="<your-api-key>"
python examples/evidence_policy/run_live.py \
  --workdir temp/evidence-policy-demo \
  --out temp/evidence-policy-demo/record.json
```

The defaults use `deepseek-flash` at `https://api.deepseek.com` and local
`BAAI/bge-small-zh-v1.5` embeddings (512 dimensions through fastembed/ONNX).
The first embedding call downloads the model. Preflight, extraction, and answer
generation call the model API.

Use a fresh, disposable `--workdir` for a full run: the script recreates its
`main` subdirectory and creates an `unrecalled` copy. Generated stores and
`record.json` remain under the ignored `temp/` directory. Supply keys through
environment variables.

The script ingests two fictional Chinese documents, reads their real
`text_chunks` rows to build a catalog with chunk IDs, document IDs, character
spans and SHA-256 hashes, then injects it as `LightRAG.evidence_catalog`.
All five questions use `aquery_llm(mode="naive")`; answer caching and thinking
are disabled. The output records model context, answers, seed IDs and expansion IDs.

| Scenario | Requested date | Expected result |
| --- | --- | --- |
| F01: both versions retrieved | 2026-08-01 | 270 days; 180 days is historical |
| F02: only the old version initially retrieved | 2026-08-01 | Fetch the amendment from the catalog; 270 days |
| F03: historical question | 2026-03-01 | 180 days |
| F04-pre: amendment published but not effective | 2026-06-15 | 180 days |
| F04-post: amendment effective | 2026-08-01 | 270 days |

The amendment was published on 2026-05-20 and became effective on 2026-07-01.
F02 deliberately removes its vector entry in the copy while retaining its text
and catalog entry. This exercises recovery from a controlled retrieval miss;
it does not measure natural retrieval miss rates.

## Reuse ingestion and configure providers

After a complete ingestion, repeat only the five questions with a new output path:

```bash
python examples/evidence_policy/run_live.py \
  --reuse-ingestion \
  --workdir temp/evidence-policy-demo \
  --out temp/evidence-policy-demo/record-next.json
```

`--preflight-only` checks the model endpoint and embedding dimension without
ingestion; by default it still sends one chat request.

| Environment variable | Default or purpose |
| --- | --- |
| `LIGHTRAG_LLM_API_KEY` | Required generation key |
| `LIGHTRAG_LLM_BASE_URL` | `https://api.deepseek.com` |
| `LIGHTRAG_LLM_MODEL` | `deepseek-flash` |
| `LIGHTRAG_EMBED_BACKEND` | `fastembed`; also accepts `openai` |
| `LIGHTRAG_EMBED_MODEL` | Local default `BAAI/bge-small-zh-v1.5`; required for hosted embeddings |
| `LIGHTRAG_EMBED_BASE_URL` | Hosted embedding endpoint; defaults to the generation endpoint |
| `LIGHTRAG_EMBED_API_KEY` | Hosted embedding key; defaults to the generation key |
| `LIGHTRAG_EMBED_DIM` | Optional dimension checked against preflight output |

Chat requests include DeepSeek's `thinking: {"type": "disabled"}` parameter.
Check support before changing endpoints; the recorded live run covers only
the DeepSeek configuration above.

## Files and supported scope

| File | Purpose |
| --- | --- |
| `run_live.py` | Ingest, build the catalog, run five real questions, save the record |
| `corpus_3037.py` | Two fictional documents and explicit version relationships |
| `smoke_plumbing.py` | Offline wiring check with deterministic model and embedding stubs |
| `requirements.txt` | Additional demo dependencies |
| `VALIDATION.md` | Observed results and their limits |
| `recorded/2026-10-29.json` | Published copy of the completed live run |

The implementation has `naive` and `mix` branches; only `naive` has been exercised
in the recorded live demo. Operators supply version relationships explicitly.
Automatic amendment discovery, REST parameters, production catalog management,
and all-mode integration are outside this demonstration. `resolved` describes
evidence processing; it does not certify the generated answer's meaning or citations.
