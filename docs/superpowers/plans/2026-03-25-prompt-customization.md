# Prompt Customization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement structured prompt customization across LightRAG core, API, and WebUI, including validation, cache-aware prompt fingerprints, API safety gating, and backward-compatible `user_prompt` / `system_prompt` behavior.

**Architecture:** Add a prompt-configuration layer in `lightrag/prompt.py` that materializes built-in defaults, validates and merges overrides, and fingerprints relevant prompt families. Thread that layer through `LightRAG`, `QueryParam`, runtime consumers in `operate.py`, FastAPI query parsing and `/health` capability exposure, then add a query-time-only WebUI editor that is gated by backend capability and persists local drafts.

**Tech Stack:** Python dataclasses, FastAPI, Pydantic, pytest via `./scripts/test.sh`, Bun, React 19, Zustand, Vite

---

## File Structure

- Modify: `lightrag/prompt.py`
  - Build the structured prompt schema, merge/validation helpers, and prompt fingerprint utilities while keeping `PROMPTS` as the default source of truth.
- Modify: `lightrag/base.py`
  - Add `QueryParam.prompt_overrides`.
- Modify: `lightrag/lightrag.py`
  - Add instance-level `prompt_config` and carry it into `global_config`.
- Modify: `lightrag/operate.py`
  - Replace direct `PROMPTS[...]` reads in query, keyword, summary, and entity-extraction flows with resolved prompt config and prompt-aware cache hashing.
- Modify: `lightrag/api/config.py`
  - Parse `ALLOW_PROMPT_OVERRIDES_VIA_API`.
- Modify: `lightrag/api/lightrag_server.py`
  - Expose the API capability in `/health`.
- Modify: `lightrag/api/routers/query_routes.py`
  - Accept/gate `prompt_overrides`, reject unsupported request overrides when disabled, and pass them into `QueryParam`.
- Modify: `lightrag_webui/src/api/lightrag.ts`
  - Add query prompt override types and health capability typing.
- Modify: `lightrag_webui/src/stores/state.ts`
  - Surface backend capability state from `/health`.
- Modify: `lightrag_webui/src/stores/settings.ts`
  - Persist `querySettings.prompt_overrides` and migrate old local state.
- Modify: `lightrag_webui/src/features/RetrievalTesting.tsx`
  - Send `prompt_overrides` only when backend capability is enabled.
- Modify: `lightrag_webui/src/components/retrieval/QuerySettings.tsx`
  - Keep the existing `user_prompt` UI and mount a dedicated prompt override editor.
- Create: `lightrag_webui/src/components/retrieval/PromptOverridesEditor.tsx`
  - Focused UI for query-time prompt override editing.
- Create: `lightrag_webui/src/utils/promptOverrides.ts`
  - Pure helpers for empty-state pruning, defaults, and list/string handling.
- Create: `lightrag_webui/src/utils/promptOverrides.test.ts`
  - Bun tests for prompt override utilities.
- Modify: `lightrag_webui/src/locales/en.json`
- Modify: `lightrag_webui/src/locales/zh.json`
  - Add prompt customization copy, warnings, and capability-disabled text.
- Create: `tests/test_prompt_config.py`
  - Offline tests for schema, validation levels, merge precedence, and prompt fingerprints.
- Create: `tests/test_query_prompt_customization.py`
  - Offline tests for query-time prompt resolution, `only_need_prompt`, `user_prompt` fallback, and cache-key changes.
- Create: `tests/test_query_prompt_overrides_api.py`
  - API model / route tests for `prompt_overrides`, capability gating, and `/health` capability exposure.
- Modify: `tests/test_extract_entities.py`
  - Add prompt-config-aware extraction coverage without breaking existing gleaning tests.
- Modify: `README.md`
- Modify: `README-zh.md`
- Modify: `lightrag/api/README.md`
- Modify: `lightrag/api/README-zh.md`
  - Document SDK config, API request schema, capability gate, and WebUI scope boundary.

### Task 1: Build Structured Prompt Config Core

**Files:**
- Modify: `lightrag/prompt.py`
- Test: `tests/test_prompt_config.py`

- [ ] **Step 1: Write the failing prompt-config tests**

```python
def test_get_default_prompt_config_contains_query_keyword_and_indexing_families():
    config = get_default_prompt_config()
    assert config["query"]["rag_response"]
    assert config["keywords"]["keywords_extraction_examples"]
    assert config["entity_extraction"]["system_prompt"]


def test_validate_prompt_config_rejects_missing_strict_placeholders():
    validate_prompt_config(
        {"query": {"rag_response": "Answer plainly."}},
        allowed_families={"query"},
    )


def test_validate_prompt_config_warns_for_missing_recommended_placeholders():
    result = validate_prompt_config(
        {"query": {"rag_response": "{context_data}"}},
        allowed_families={"query"},
    )
    assert "user_prompt" in result.warnings[0]


def test_prompt_fingerprint_changes_when_effective_query_prompt_changes():
    left = get_prompt_fingerprint({"query": {"rag_response": "A {context_data}"}})
    right = get_prompt_fingerprint({"query": {"rag_response": "B {context_data}"}})
    assert left != right
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./scripts/test.sh tests/test_prompt_config.py -q`

Expected: FAIL with `ImportError` / `AttributeError` because the structured prompt helpers do not exist yet.

- [ ] **Step 3: Implement prompt schema, merge, validation, and fingerprint helpers**

```python
PROMPT_SCHEMA = {
    "query": {
        "rag_response": PromptRule(required={"context_data"}, recommended={"response_type", "user_prompt"}),
        ...
    },
    ...
}


def get_default_prompt_config() -> dict[str, Any]:
    return {
        "shared": {...},
        "query": {...},
        "keywords": {...},
        "entity_extraction": {...},
        "summary": {...},
    }


def merge_prompt_config(
    base: dict[str, Any],
    override: dict[str, Any] | None,
    *,
    allowed_families: set[str] | None = None,
) -> dict[str, Any]:
    ...


def validate_prompt_config(...) -> PromptValidationResult:
    ...


def get_prompt_fingerprint(config: dict[str, Any]) -> str:
    ...
```

- [ ] **Step 4: Re-run the tests and keep them green**

Run: `./scripts/test.sh tests/test_prompt_config.py -q`

Expected: PASS

- [ ] **Step 5: Commit the prompt-core slice**

```bash
git add lightrag/prompt.py tests/test_prompt_config.py
git commit -m "feat: add structured prompt config core"
```

### Task 2: Extend Core Runtime Models For Prompt Resolution

**Files:**
- Modify: `lightrag/base.py`
- Modify: `lightrag/lightrag.py`
- Test: `tests/test_prompt_config.py`
- Test: `tests/test_query_prompt_customization.py`

- [ ] **Step 1: Write failing model-layer tests for instance config and request overrides**

```python
def test_query_param_accepts_prompt_overrides():
    param = QueryParam(prompt_overrides={"query": {"rag_response": "{context_data}"}})
    assert "query" in param.prompt_overrides


def test_lightrag_global_config_carries_prompt_config(tmp_path):
    rag = LightRAG(working_dir=str(tmp_path), prompt_config={"query": {"rag_response": "{context_data}"}})
    global_config = asdict(rag)
    assert "prompt_config" in global_config
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `./scripts/test.sh tests/test_prompt_config.py tests/test_query_prompt_customization.py -q`

Expected: FAIL because `QueryParam.prompt_overrides` and `LightRAG.prompt_config` are not defined yet.

- [ ] **Step 3: Add runtime fields and propagate them into `global_config`**

```python
@dataclass
class QueryParam:
    ...
    prompt_overrides: dict[str, Any] | None = None


@dataclass
class LightRAG:
    ...
    prompt_config: dict[str, Any] = field(default_factory=dict)
```

Implementation notes:

- Ensure `aquery_data()` copies `prompt_overrides` into its derived `QueryParam`
- Keep `prompt_config` compatible with `asdict(self)`
- Do not remove or rename `user_prompt` / `system_prompt`

- [ ] **Step 4: Re-run the focused tests**

Run: `./scripts/test.sh tests/test_prompt_config.py tests/test_query_prompt_customization.py -q`

Expected: PASS for model-layer assertions, with later runtime assertions still failing until Task 3.

- [ ] **Step 5: Commit the model-layer slice**

```bash
git add lightrag/base.py lightrag/lightrag.py tests/test_prompt_config.py tests/test_query_prompt_customization.py
git commit -m "feat: add prompt config runtime models"
```

### Task 3: Wire Query-Time Prompt Resolution, `only_need_prompt`, and Cache Fingerprints

**Files:**
- Modify: `lightrag/operate.py`
- Test: `tests/test_query_prompt_customization.py`

- [ ] **Step 1: Write failing runtime tests for query-time prompt behavior**

```python
@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_uses_query_prompt_override_in_only_need_prompt_mode(...):
    param = QueryParam(
        mode="mix",
        only_need_prompt=True,
        prompt_overrides={"query": {"rag_response": "CTX={context_data}"}},
    )
    result = await kg_query(...)
    assert "CTX=" in result.content


@pytest.mark.offline
@pytest.mark.asyncio
async def test_user_prompt_is_appended_when_custom_template_omits_placeholder(...):
    param = QueryParam(
        mode="naive",
        user_prompt="Use bullet points",
        only_need_prompt=True,
        prompt_overrides={"query": {"naive_rag_response": "{content_data}"}},
    )
    result = await naive_query(...)
    assert "Use bullet points" in result.content


def test_query_cache_hash_changes_with_prompt_override(...):
    ...
```

- [ ] **Step 2: Run the runtime tests to verify they fail**

Run: `./scripts/test.sh tests/test_query_prompt_customization.py -q`

Expected: FAIL because query flows still read `PROMPTS[...]` directly and cache hashes do not yet include prompt fingerprints.

- [ ] **Step 3: Implement resolved query prompt access and prompt-aware hashes**

```python
effective_prompts = resolve_effective_prompt_config(
    global_config.get("prompt_config"),
    query_param.prompt_overrides,
    system_prompt=system_prompt,
    allowed_request_families={"query", "keywords"},
)

answer_template = effective_prompts["query"]["rag_response"]
context_template = effective_prompts["query"]["kg_query_context"]
keywords_template = effective_prompts["keywords"]["keywords_extraction"]

query_prompt_fingerprint = get_prompt_fingerprint({"query": effective_prompts["query"]})
```

Implementation notes:

- Replace direct `PROMPTS["rag_response"]` / `PROMPTS["naive_rag_response"]` usage
- Replace direct `PROMPTS["kg_query_context"]` / `PROMPTS["naive_query_context"]` usage
- Replace direct `PROMPTS["keywords_extraction"]` / `PROMPTS["keywords_extraction_examples"]` usage
- Include prompt fingerprints in query and keyword cache args
- Preserve `system_prompt` as a legacy shortcut override
- Make `only_need_prompt=True` return the final merged prompt, not the default template

- [ ] **Step 4: Re-run the runtime tests**

Run: `./scripts/test.sh tests/test_query_prompt_customization.py -q`

Expected: PASS

- [ ] **Step 5: Commit the query-runtime slice**

```bash
git add lightrag/operate.py tests/test_query_prompt_customization.py
git commit -m "feat: wire query prompt overrides"
```

### Task 4: Wire Indexing-Time Prompt Families and Prompt-Aware Indexing Caches

**Files:**
- Modify: `lightrag/operate.py`
- Modify: `tests/test_extract_entities.py`
- Test: `tests/test_prompt_config.py`

- [ ] **Step 1: Add failing extraction/summary tests for instance-level prompt config**

```python
@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_uses_instance_prompt_config_templates(...):
    global_config = {
        ...,
        "prompt_config": {
            "entity_extraction": {
                "system_prompt": "SYS {tuple_delimiter} {examples}",
                "user_prompt": "USR {input_text}",
                "continue_prompt": "CONT {input_text}",
            }
        },
    }
    await extract_entities(...)
    llm_func.assert_awaited()
    assert "USR" in llm_func.await_args_list[0].args[0]


def test_extract_prompt_fingerprint_changes_with_entity_prompt_override():
    ...
```

- [ ] **Step 2: Run the extraction tests to verify they fail**

Run: `./scripts/test.sh tests/test_extract_entities.py tests/test_prompt_config.py -q`

Expected: FAIL because extraction and summary paths still format from the global `PROMPTS` dictionary only.

- [ ] **Step 3: Implement indexing prompt resolution and prompt-aware cache hashing**

```python
effective_prompts = resolve_effective_prompt_config(global_config.get("prompt_config"))
shared = effective_prompts["shared"]
extract_cfg = effective_prompts["entity_extraction"]
summary_cfg = effective_prompts["summary"]

examples = "\n".join(extract_cfg["examples"]).format(...)
system_prompt = extract_cfg["system_prompt"].format(...)
summary_prompt = summary_cfg["summarize_entity_descriptions"].format(...)
```

Implementation notes:

- Drive `extract_entities()` from resolved `entity_extraction` + `shared`
- Drive summary generation from resolved `summary`
- Include indexing prompt fingerprints in extract / summary cache identity
- Log when effective indexing prompt config differs from defaults so the operator understands reprocessing cost

- [ ] **Step 4: Re-run the extraction tests**

Run: `./scripts/test.sh tests/test_extract_entities.py tests/test_prompt_config.py -q`

Expected: PASS

- [ ] **Step 5: Commit the indexing-runtime slice**

```bash
git add lightrag/operate.py tests/test_extract_entities.py tests/test_prompt_config.py
git commit -m "feat: apply prompt config to indexing flows"
```

### Task 5: Add API Request Schema, Safety Gate, and Health Capability Exposure

**Files:**
- Modify: `lightrag/api/config.py`
- Modify: `lightrag/api/lightrag_server.py`
- Modify: `lightrag/api/routers/query_routes.py`
- Test: `tests/test_query_prompt_overrides_api.py`

- [ ] **Step 1: Write failing API tests for request schema and capability gating**

```python
def test_query_request_converts_prompt_overrides_to_query_param():
    request = QueryRequest(
        query="hello world",
        mode="mix",
        prompt_overrides={"query": {"rag_response": "{context_data}"}},
    )
    param = request.to_query_params(False)
    assert param.prompt_overrides["query"]["rag_response"] == "{context_data}"


def test_query_endpoint_rejects_prompt_overrides_when_capability_disabled(test_client):
    response = test_client.post(
        "/query",
        json={
            "query": "hello world",
            "mode": "mix",
            "prompt_overrides": {"query": {"rag_response": "{context_data}"}},
        },
    )
    assert response.status_code == 403


def test_health_exposes_prompt_override_capability(test_client):
    data = test_client.get("/health").json()
    assert "allow_prompt_overrides_via_api" in data["configuration"]
```

- [ ] **Step 2: Run the API tests to verify they fail**

Run: `./scripts/test.sh tests/test_query_prompt_overrides_api.py -q`

Expected: FAIL because the request model, server config flag, and `/health` capability field do not exist yet.

- [ ] **Step 3: Implement the API-side prompt override contract**

```python
class QueryRequest(BaseModel):
    ...
    prompt_overrides: Optional[Dict[str, Any]] = None


if request.prompt_overrides and not args.allow_prompt_overrides_via_api:
    raise HTTPException(status_code=403, detail="Prompt overrides are disabled on this server")
```

Implementation notes:

- Parse `ALLOW_PROMPT_OVERRIDES_VIA_API` in `lightrag/api/config.py`
- Add `allow_prompt_overrides_via_api` to the `/health` `configuration` block
- Reject request-level `prompt_overrides` before merge/validation when the capability is disabled
- Keep SDK usage unaffected

- [ ] **Step 4: Re-run the API tests**

Run: `./scripts/test.sh tests/test_query_prompt_overrides_api.py -q`

Expected: PASS

- [ ] **Step 5: Commit the API slice**

```bash
git add lightrag/api/config.py lightrag/api/lightrag_server.py lightrag/api/routers/query_routes.py tests/test_query_prompt_overrides_api.py
git commit -m "feat: add API prompt override safety gate"
```

### Task 6: Add WebUI Prompt Override Types, State, Gating, and Editor UI

**Files:**
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/stores/state.ts`
- Modify: `lightrag_webui/src/stores/settings.ts`
- Modify: `lightrag_webui/src/features/RetrievalTesting.tsx`
- Modify: `lightrag_webui/src/components/retrieval/QuerySettings.tsx`
- Create: `lightrag_webui/src/components/retrieval/PromptOverridesEditor.tsx`
- Create: `lightrag_webui/src/utils/promptOverrides.ts`
- Create: `lightrag_webui/src/utils/promptOverrides.test.ts`
- Modify: `lightrag_webui/src/locales/en.json`
- Modify: `lightrag_webui/src/locales/zh.json`

- [ ] **Step 1: Write failing Bun tests for prompt override utilities and persisted state**

```ts
import { describe, expect, test } from 'bun:test'
import { pruneEmptyPromptOverrides, setExampleItem } from './promptOverrides'

describe('promptOverrides', () => {
  test('drops empty nested prompt override objects', () => {
    expect(pruneEmptyPromptOverrides({ query: { rag_response: '' } })).toBeUndefined()
  })

  test('preserves list-style example fields as string arrays', () => {
    expect(setExampleItem(['A'], 1, 'B')).toEqual(['A', 'B'])
  })
})
```

- [ ] **Step 2: Run the Bun tests to verify they fail**

Run: `cd lightrag_webui && bun test src/utils/promptOverrides.test.ts`

Expected: FAIL because the helper file and prompt override types do not exist yet.

- [ ] **Step 3: Implement frontend typing, capability gating, and request plumbing**

```ts
export type QueryPromptOverrides = {
  query?: {
    rag_response?: string
    naive_rag_response?: string
    kg_query_context?: string
    naive_query_context?: string
  }
  keywords?: {
    keywords_extraction?: string
    keywords_extraction_examples?: string[]
  }
}

type LightragStatus = {
  configuration: {
    ...
    allow_prompt_overrides_via_api?: boolean
  }
}
```

Implementation notes:

- Persist `querySettings.prompt_overrides`
- Bump the Zustand storage version and migrate old state to include `prompt_overrides: undefined`
- Preserve locally stored prompt override drafts even if the backend later disables the feature
- In `RetrievalTesting.tsx`, only include `prompt_overrides` in the request payload when backend status says the capability is enabled

- [ ] **Step 4: Implement the prompt override editor and mount it in query settings**

```tsx
<PromptOverridesEditor
  enabled={backendStatus?.configuration?.allow_prompt_overrides_via_api === true}
  value={querySettings.prompt_overrides}
  onChange={(value) => handleChange('prompt_overrides', value)}
/>
```

Implementation notes:

- Keep `user_prompt` as the lightweight top-level field
- Split the new editor into its own component instead of expanding `QuerySettings.tsx` further
- Provide basic and advanced sections
- Use multi-entry list editing for `keywords_extraction_examples`
- Show capability-disabled text when the backend flag is false

- [ ] **Step 5: Re-run frontend tests and build validation**

Run: `cd lightrag_webui && bun test src/utils/promptOverrides.test.ts`

Expected: PASS

Run: `cd lightrag_webui && bun run build`

Expected: PASS

- [ ] **Step 6: Commit the WebUI slice**

```bash
git add lightrag_webui/src/api/lightrag.ts \
        lightrag_webui/src/stores/state.ts \
        lightrag_webui/src/stores/settings.ts \
        lightrag_webui/src/features/RetrievalTesting.tsx \
        lightrag_webui/src/components/retrieval/QuerySettings.tsx \
        lightrag_webui/src/components/retrieval/PromptOverridesEditor.tsx \
        lightrag_webui/src/utils/promptOverrides.ts \
        lightrag_webui/src/utils/promptOverrides.test.ts \
        lightrag_webui/src/locales/en.json \
        lightrag_webui/src/locales/zh.json
git commit -m "feat: add WebUI prompt override editor"
```

### Task 7: Update Documentation and Run Final Verification

**Files:**
- Modify: `README.md`
- Modify: `README-zh.md`
- Modify: `lightrag/api/README.md`
- Modify: `lightrag/api/README-zh.md`

- [ ] **Step 1: Write the missing documentation snippets**

```md
## Prompt Customization

- `LightRAG(prompt_config=...)` sets instance defaults
- `QueryParam(prompt_overrides=...)` sets query-time overrides
- API request overrides require `ALLOW_PROMPT_OVERRIDES_VIA_API=true`
- WebUI only exposes query-time prompt overrides
```

- [ ] **Step 2: Run the focused backend verification suite**

Run:

```bash
./scripts/test.sh \
  tests/test_prompt_config.py \
  tests/test_query_prompt_customization.py \
  tests/test_query_prompt_overrides_api.py \
  tests/test_extract_entities.py -q
```

Expected: PASS

- [ ] **Step 3: Run the frontend verification suite**

Run:

```bash
cd lightrag_webui && bun test src/utils/promptOverrides.test.ts && bun run build
```

Expected: PASS

- [ ] **Step 4: Run lint / smoke verification if time permits**

Run:

```bash
ruff check lightrag tests
```

Expected: PASS or only pre-existing unrelated findings

- [ ] **Step 5: Commit the docs and verification slice**

```bash
git add README.md README-zh.md lightrag/api/README.md lightrag/api/README-zh.md
git commit -m "docs: document prompt customization"
```
