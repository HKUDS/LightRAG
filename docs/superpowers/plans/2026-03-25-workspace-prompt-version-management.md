# Workspace Prompt Version Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement workspace-scoped prompt version management with separate indexing and retrieval version chains, localized seed versions, retrieval-side temporary version testing, and safe fallback to existing LightRAG behavior when no version is active.

**Architecture:** Add a core prompt-version domain and atomic file-backed registry under `lightrag/`, then thread active indexing/retrieval payload resolution into `LightRAG` and `operate.py` without breaking existing `prompt_overrides` semantics. Expose CRUD/activate/diff/initialize APIs through a dedicated FastAPI router, then build a new Prompt Management tab in WebUI plus Retrieval-page temporary version and `Custom / Draft` controls that project only `query` and `keywords` into per-request overrides.

**Tech Stack:** Python dataclasses, FastAPI, Pydantic, pytest via `./scripts/test.sh`, Bun, React 19, Zustand, Vite, JSON file persistence with atomic rename

---

## File Structure

- Modify: `lightrag/prompt.py`
  - Add prompt group constants, localized seed-pack builders, group projection helpers, and validation wrappers that keep `PROMPTS` as the base source of truth.
- Create: `lightrag/prompt_versions.py`
  - Own version record helpers, group payload typing, `ENTITY_TYPES` / `SUMMARY_LANGUAGE` validation, copy helpers, and diff helpers.
- Create: `lightrag/prompt_version_store.py`
  - Implement workspace-scoped JSON registry persistence with atomic write/replace behavior.
- Modify: `lightrag/lightrag.py`
  - Attach prompt version store/runtime helpers to the workspace instance and carry active group payloads into `global_config`.
- Modify: `lightrag/operate.py`
  - Resolve indexing and retrieval config from built-ins, env-backed extras, active versions, and request-scoped retrieval overrides.
- Create: `lightrag/api/routers/prompt_config_routes.py`
  - Provide initialize/list/read/create/activate/delete/diff APIs for indexing and retrieval groups.
- Modify: `lightrag/api/lightrag_server.py`
  - Register prompt-config routes and expose active version summary plus indexing warning metadata in `/health`.
- Modify: `lightrag/api/README.md`
- Modify: `lightrag/api/README-zh.md`
  - Document prompt-config APIs, version semantics, fallback behavior, and retrieval draft scope.
- Modify: `README.md`
- Modify: `README-zh.md`
  - Document prompt management concepts at the project level.
- Modify: `lightrag_webui/src/api/lightrag.ts`
  - Add prompt version API types, retrieval draft helpers, and active-version health typing.
- Modify: `lightrag_webui/src/stores/state.ts`
  - Surface active prompt version metadata from `/health`.
- Modify: `lightrag_webui/src/stores/settings.ts`
  - Persist Prompt Management UI state and Retrieval temporary selection/draft state.
- Modify: `lightrag_webui/src/App.tsx`
- Modify: `lightrag_webui/src/features/SiteHeader.tsx`
  - Add the Prompt Management tab between Knowledge Graph and Retrieval.
- Create: `lightrag_webui/src/features/PromptManagement.tsx`
  - Host the new version-management page.
- Create: `lightrag_webui/src/components/prompt-management/PromptGroupSwitcher.tsx`
  - Toggle between indexing and retrieval version groups.
- Create: `lightrag_webui/src/components/prompt-management/PromptVersionList.tsx`
  - Render versions, active badge, copy entry, and empty states.
- Create: `lightrag_webui/src/components/prompt-management/PromptVersionEditor.tsx`
  - Render metadata fields, family sections, list editors, and activation/save actions.
- Create: `lightrag_webui/src/components/prompt-management/PromptVersionDiffDialog.tsx`
  - Show grouped field diffs against the active version or selected base.
- Create: `lightrag_webui/src/components/prompt-management/PromptListFieldEditor.tsx`
  - Reusable list editor for `ENTITY_TYPES` and prompt example arrays.
- Modify: `lightrag_webui/src/components/retrieval/QuerySettings.tsx`
  - Replace direct prompt-template editing with retrieval version selection plus `Custom / Draft` workflow.
- Modify: `lightrag_webui/src/components/retrieval/PromptOverridesEditor.tsx`
  - Narrow it to request-scoped retrieval draft editing for `query` and `keywords`.
- Create: `lightrag_webui/src/components/retrieval/RetrievalPromptVersionSelector.tsx`
  - Retrieve saved retrieval versions and project selected payloads into request overrides.
- Create: `lightrag_webui/src/utils/promptVersioning.ts`
  - Pure helpers for payload projection, seed labels, diff formatting, and draft pruning.
- Create: `lightrag_webui/src/utils/promptVersioning.test.ts`
  - Bun tests for projection/draft helpers.
- Modify: `lightrag_webui/src/locales/en.json`
- Modify: `lightrag_webui/src/locales/zh.json`
  - Add Prompt Management and retrieval draft copy, warnings, and fallback messaging.
- Create: `tests/test_prompt_versioning.py`
  - Validate group schemas, localized seeds, and extra-field rules.
- Create: `tests/test_prompt_version_store.py`
  - Verify atomic persistence and workspace/group registry behavior.
- Create: `tests/test_prompt_version_runtime.py`
  - Cover fallback, active version resolution, and retrieval temporary projection behavior.
- Create: `tests/test_prompt_config_routes.py`
  - Exercise prompt-config CRUD/activate/diff APIs and indexing warning metadata.
- Modify: `tests/test_query_prompt_overrides_api.py`
  - Keep coverage for retrieval query-time override boundaries.

### Task 1: Build Prompt Version Domain, Group Rules, and Localized Seeds

**Files:**
- Modify: `lightrag/prompt.py`
- Create: `lightrag/prompt_versions.py`
- Test: `tests/test_prompt_versioning.py`

- [ ] **Step 1: Write the failing prompt-version domain tests**

```python
def test_build_seed_versions_returns_indexing_and_retrieval_groups():
    seeds = build_localized_seed_versions("zh")
    assert "indexing" in seeds
    assert "retrieval" in seeds
    assert seeds["indexing"]["version_name"].startswith("indexing")


def test_validate_indexing_payload_accepts_entity_types_and_summary_language():
    validate_prompt_group_payload(
        "indexing",
        {
            "entity_types": ["Person", "Organization"],
            "summary_language": "Chinese",
            "shared": {"tuple_delimiter": "<|#|>"},
        },
    )


def test_validate_retrieval_payload_rejects_entity_extraction_family():
    with pytest.raises(ValueError):
        validate_prompt_group_payload(
            "retrieval",
            {"entity_extraction": {"system_prompt": "bad"}},
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./scripts/test.sh tests/test_prompt_versioning.py -q`

Expected: FAIL with import or attribute errors because prompt-version helpers do not exist yet.

- [ ] **Step 3: Implement group constants, seed builders, and payload validation**

```python
PROMPT_VERSION_GROUPS = {
    "indexing": {"families": {"shared", "entity_extraction", "summary"}},
    "retrieval": {"families": {"query", "keywords"}},
}


def validate_prompt_group_payload(group_type: str, payload: dict[str, Any]) -> None:
    group = PROMPT_VERSION_GROUPS[group_type]
    prompt_subset = {
        key: value for key, value in payload.items() if key in group["families"]
    }
    validate_prompt_config(prompt_subset, allowed_families=group["families"])
    validate_extra_group_fields(group_type, payload)


def build_localized_seed_versions(locale: str) -> dict[str, dict[str, Any]]:
    seed_pack = LOCALIZED_PROMPT_SEED_PACKS[locale]
    return {
        "indexing": seed_pack["indexing"],
        "retrieval": seed_pack["retrieval"],
    }


def project_group_prompt_config(group_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    allowed_families = PROMPT_VERSION_GROUPS[group_type]["families"]
    return {
        family: family_payload
        for family, family_payload in payload.items()
        if family in allowed_families
    }
```

Implementation notes:

- Keep seed packs localized but schema-compatible with current placeholders.
- Validate `entity_types` as non-empty `list[str]`.
- Validate `summary_language` as non-empty `str`.
- Reuse `validate_prompt_config()` for prompt families already in `PROMPT_SCHEMA`.

- [ ] **Step 4: Re-run the domain tests**

Run: `./scripts/test.sh tests/test_prompt_versioning.py -q`

Expected: PASS

- [ ] **Step 5: Commit the prompt-version domain slice**

```bash
git add lightrag/prompt.py lightrag/prompt_versions.py tests/test_prompt_versioning.py
git commit -m "feat: add prompt version domain helpers"
```

### Task 2: Add Atomic Workspace Prompt Version Persistence

**Files:**
- Create: `lightrag/prompt_version_store.py`
- Test: `tests/test_prompt_version_store.py`

- [ ] **Step 1: Write the failing store tests**

```python
def test_initialize_registry_creates_localized_seed_versions(tmp_path):
    store = PromptVersionStore(tmp_path, workspace="demo")
    registry = store.initialize(locale="zh")
    assert registry["indexing"]["versions"]
    assert registry["retrieval"]["versions"]


def test_delete_inactive_version_keeps_lineage_readable(tmp_path):
    store = PromptVersionStore(tmp_path, workspace="demo")
    created = store.create_version(
        "retrieval",
        {
            "query": {"rag_response": "A {context_data}"},
            "keywords": {"keywords_extraction": "Q={query};E={examples}"},
        },
        "v1",
        "first",
        None,
    )
    copied = store.copy_version("retrieval", created["version_id"], "v2", "")
    store.delete_version("retrieval", created["version_id"])
    fetched = store.get_version("retrieval", copied["version_id"])
    assert fetched["source_version_id"] == created["version_id"]


def test_store_writes_registry_atomically(tmp_path):
    store = PromptVersionStore(tmp_path, workspace="demo")
    store.initialize(locale="en")
    assert not list(tmp_path.rglob("*.tmp"))
```

- [ ] **Step 2: Run the store tests to verify they fail**

Run: `./scripts/test.sh tests/test_prompt_version_store.py -q`

Expected: FAIL because the store module and registry operations do not exist yet.

- [ ] **Step 3: Implement the JSON-backed store with atomic replace**

```python
class PromptVersionStore:
    def initialize(self, locale: str) -> dict[str, Any]:
        registry = self._read_or_default()
        if registry["indexing"]["versions"] or registry["retrieval"]["versions"]:
            return registry
        seeds = build_localized_seed_versions(locale)
        registry["indexing"]["versions"] = [seeds["indexing"]]
        registry["retrieval"]["versions"] = [seeds["retrieval"]]
        self._atomic_write(registry)
        return registry

    def create_version(
        self,
        group_type: str,
        payload: dict[str, Any],
        version_name: str,
        comment: str,
        source_version_id: str | None,
    ) -> dict[str, Any]:
        registry = self._read_or_default()
        record = build_version_record(
            group_type, payload, version_name, comment, source_version_id
        )
        registry[group_type]["versions"].append(record)
        self._atomic_write(registry)
        return record

    def activate_version(self, group_type: str, version_id: str) -> dict[str, Any]:
        registry = self._read_or_default()
        registry[group_type]["active_version_id"] = version_id
        self._atomic_write(registry)
        return self.get_version(group_type, version_id)

    def delete_version(self, group_type: str, version_id: str) -> None:
        registry = self._read_or_default()
        if registry[group_type]["active_version_id"] == version_id:
            raise ValueError("Cannot delete the active prompt version")
        registry[group_type]["versions"] = [
            item for item in registry[group_type]["versions"] if item["version_id"] != version_id
        ]
        self._atomic_write(registry)

    def diff_versions(
        self, group_type: str, version_id: str, base_version_id: str | None
    ) -> dict[str, Any]:
        target = self.get_version(group_type, version_id)
        base = self.get_version(group_type, base_version_id) if base_version_id else None
        return build_group_diff(group_type, base["payload"] if base else {}, target["payload"])

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        tmp_path = self.registry_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(self.registry_path)
```

Implementation notes:

- Use one registry file per workspace or one per group, but keep the public store API unchanged.
- Never leave partial writes behind.
- Make inactive-version deletion reject the active version.
- Let lineage display degrade to `Deleted`/`Unknown` in callers rather than forcing soft delete.

- [ ] **Step 4: Re-run the store tests**

Run: `./scripts/test.sh tests/test_prompt_version_store.py -q`

Expected: PASS

- [ ] **Step 5: Commit the persistence slice**

```bash
git add lightrag/prompt_version_store.py tests/test_prompt_version_store.py
git commit -m "feat: add prompt version registry store"
```

### Task 3: Wire Active Version Resolution and Fallback Into Runtime

**Files:**
- Modify: `lightrag/lightrag.py`
- Modify: `lightrag/operate.py`
- Test: `tests/test_prompt_version_runtime.py`
- Test: `tests/test_query_prompt_overrides_api.py`

- [ ] **Step 1: Write the failing runtime-resolution tests**

```python
def test_indexing_resolution_falls_back_to_env_when_no_active_version(tmp_path):
    rag = build_rag(tmp_path)
    config = rag._resolve_active_prompt_groups()
    assert config["indexing"] is None


@pytest.mark.asyncio
async def test_retrieval_runtime_uses_active_retrieval_version_before_request_override(tmp_path):
    rag = build_rag(tmp_path)
    registry = rag.prompt_version_store.initialize(locale="zh")
    active_version_id = registry["retrieval"]["versions"][0]["version_id"]
    rag.prompt_version_store.activate_version("retrieval", active_version_id)
    result = await rag.aquery("hello", QueryParam(only_need_prompt=True))
    assert "expected localized retrieval prompt" in result


@pytest.mark.asyncio
async def test_request_prompt_overrides_still_win_over_active_retrieval_version(tmp_path):
    rag = build_rag(tmp_path)
    registry = rag.prompt_version_store.initialize(locale="zh")
    active_version_id = registry["retrieval"]["versions"][0]["version_id"]
    rag.prompt_version_store.activate_version("retrieval", active_version_id)
    result = await rag.aquery(
        "hello",
        QueryParam(
            only_need_prompt=True,
            prompt_overrides={"query": {"rag_response": "REQUEST={context_data}"}},
        ),
    )
    assert "REQUEST=" in result
```

- [ ] **Step 2: Run the focused runtime tests to verify they fail**

Run: `./scripts/test.sh tests/test_prompt_version_runtime.py tests/test_query_prompt_overrides_api.py -q`

Expected: FAIL because `LightRAG` does not yet expose active version payloads and runtime resolution still only knows built-ins plus request overrides.

- [ ] **Step 3: Add unified runtime resolvers**

```python
class LightRAG:
    def _resolve_active_prompt_groups(self) -> dict[str, Any]:
        return self.prompt_version_store.get_active_group_payloads()


def _resolve_indexing_runtime_config(global_config: dict[str, Any]) -> dict[str, Any]:
    base = build_indexing_base_config(global_config)
    active = global_config.get("active_prompt_groups", {}).get("indexing")
    return merge_indexing_group_payload(base, active)


def _resolve_retrieval_runtime_config(
    global_config: dict[str, Any], query_param: QueryParam | None = None
) -> dict[str, Any]:
    base = merge_prompt_config(get_default_prompt_config(), None)
    active = global_config.get("active_prompt_groups", {}).get("retrieval")
    active_config = merge_prompt_config(base, active, allowed_families={"query", "keywords"})
    return merge_prompt_config(
        active_config,
        query_param.prompt_overrides if query_param else None,
        allowed_families={"query", "keywords"},
    )
```

Implementation notes:

- Indexing resolution order must be: built-ins -> env-backed `ENTITY_TYPES`/`SUMMARY_LANGUAGE` -> active indexing payload.
- Retrieval resolution order must be: built-ins -> active retrieval payload -> request-scoped `prompt_overrides`.
- Preserve the existing runtime path when no active version is set.
- Keep request overrides constrained to `query` and `keywords`.

- [ ] **Step 4: Re-run the focused runtime tests**

Run: `./scripts/test.sh tests/test_prompt_version_runtime.py tests/test_query_prompt_overrides_api.py -q`

Expected: PASS

- [ ] **Step 5: Commit the runtime-resolution slice**

```bash
git add lightrag/lightrag.py lightrag/operate.py tests/test_prompt_version_runtime.py tests/test_query_prompt_overrides_api.py
git commit -m "feat: resolve active prompt versions at runtime"
```

### Task 4: Add Prompt Version Management APIs and Health Metadata

**Files:**
- Create: `lightrag/api/routers/prompt_config_routes.py`
- Modify: `lightrag/api/lightrag_server.py`
- Test: `tests/test_prompt_config_routes.py`

- [ ] **Step 1: Write the failing API tests**

```python
def test_initialize_prompt_config_creates_seed_versions(test_client):
    response = test_client.post("/prompt-config/initialize")
    assert response.status_code == 200
    body = response.json()
    assert body["indexing"]["versions"]


def test_activate_indexing_version_returns_warning_metadata(test_client):
    seeded = test_client.post("/prompt-config/initialize").json()
    version_id = seeded["indexing"]["versions"][0]["version_id"]
    response = test_client.post(f"/prompt-config/indexing/versions/{version_id}/activate")
    assert response.status_code == 200
    assert "warning" in response.json()


def test_delete_active_version_is_rejected(test_client):
    seeded = test_client.post("/prompt-config/initialize").json()
    active_id = seeded["retrieval"]["versions"][0]["version_id"]
    test_client.post(f"/prompt-config/retrieval/versions/{active_id}/activate")
    response = test_client.delete(f"/prompt-config/retrieval/versions/{active_id}")
    assert response.status_code == 400
```

- [ ] **Step 2: Run the API tests to verify they fail**

Run: `./scripts/test.sh tests/test_prompt_config_routes.py -q`

Expected: FAIL because the router and endpoints are not registered yet.

- [ ] **Step 3: Implement the router and register it**

```python
router = APIRouter(prefix="/prompt-config", tags=["prompt-config"])


@router.post("/initialize")
async def initialize_prompt_config(locale: str = "zh") -> dict[str, Any]:
    return prompt_version_store.initialize(locale=locale)


@router.get("/{group_type}/versions")
async def list_versions(group_type: str) -> dict[str, Any]:
    return prompt_version_store.list_versions(group_type)


@router.post("/{group_type}/versions/{version_id}/activate")
async def activate_version(group_type: str, version_id: str) -> dict[str, Any]:
    indexing_warning = INDEXING_ACTIVATION_WARNING
    return {
        "active_version_id": version_id,
        "warning": indexing_warning if group_type == "indexing" else None,
    }
```

Implementation notes:

- Reuse the `LightRAG` workspace and store instance instead of creating separate prompt-config state.
- Add active version metadata to `/health`.
- Keep route payloads explicit and `extra="forbid"` to avoid malformed version objects.

- [ ] **Step 4: Re-run the API tests**

Run: `./scripts/test.sh tests/test_prompt_config_routes.py -q`

Expected: PASS

- [ ] **Step 5: Commit the API slice**

```bash
git add lightrag/api/routers/prompt_config_routes.py lightrag/api/lightrag_server.py tests/test_prompt_config_routes.py
git commit -m "feat: add prompt version management api"
```

### Task 5: Add WebUI API Types, Stores, and Navigation Shell

**Files:**
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/stores/state.ts`
- Modify: `lightrag_webui/src/stores/settings.ts`
- Modify: `lightrag_webui/src/App.tsx`
- Modify: `lightrag_webui/src/features/SiteHeader.tsx`
- Modify: `lightrag_webui/src/locales/en.json`
- Modify: `lightrag_webui/src/locales/zh.json`
- Test: `lightrag_webui/src/utils/promptVersioning.test.ts`

- [ ] **Step 1: Write the failing frontend utility tests**

```ts
import { describe, expect, test } from 'vitest'
import { projectRetrievalVersionToOverrides } from './promptVersioning'

test('projects retrieval payload to query-time overrides only', () => {
  expect(projectRetrievalVersionToOverrides({
    query: { rag_response: '{context_data}' },
    keywords: { keywords_extraction: '{query}' }
  })).toEqual({
    query: { rag_response: '{context_data}' },
    keywords: { keywords_extraction: '{query}' }
  })
})
```

- [ ] **Step 2: Run the frontend utility test to verify it fails**

Run: `cd lightrag_webui && bun test src/utils/promptVersioning.test.ts`

Expected: FAIL because the utility and prompt-version API types do not exist yet.

- [ ] **Step 3: Implement client types, store state, and tab plumbing**

```ts
export type PromptConfigGroup = 'indexing' | 'retrieval'
export type PromptVersionRecord = {
  version_id: string
  group_type: PromptConfigGroup
  version_name: string
  version_number: number
  comment: string
  source_version_id?: string | null
  created_at: string
  payload: Record<string, unknown>
}
export type PromptVersionRegistrySummary = {
  active_version_id: string | null
  versions: PromptVersionRecord[]
}

type SettingsState = {
  currentTab: 'documents' | 'knowledge-graph' | 'prompt-management' | 'retrieval' | 'api'
  retrievalPromptDraft: QueryPromptOverrides | undefined
  retrievalPromptVersionSelection: string | 'active' | 'custom'
}
```

Implementation notes:

- Insert `prompt-management` between knowledge graph and retrieval in both tab list and tab content.
- Store active prompt version summaries from `/health`.
- Add locale copy for warnings, seed labels, and fallback messages.

- [ ] **Step 4: Re-run the frontend utility test**

Run: `cd lightrag_webui && bun test src/utils/promptVersioning.test.ts`

Expected: PASS

- [ ] **Step 5: Commit the navigation/state slice**

```bash
git add lightrag_webui/src/api/lightrag.ts lightrag_webui/src/stores/state.ts lightrag_webui/src/stores/settings.ts lightrag_webui/src/App.tsx lightrag_webui/src/features/SiteHeader.tsx lightrag_webui/src/utils/promptVersioning.ts lightrag_webui/src/utils/promptVersioning.test.ts lightrag_webui/src/locales/en.json lightrag_webui/src/locales/zh.json
git commit -m "feat: add prompt management ui shell"
```

### Task 6: Build the Prompt Management Page

**Files:**
- Create: `lightrag_webui/src/features/PromptManagement.tsx`
- Create: `lightrag_webui/src/components/prompt-management/PromptGroupSwitcher.tsx`
- Create: `lightrag_webui/src/components/prompt-management/PromptVersionList.tsx`
- Create: `lightrag_webui/src/components/prompt-management/PromptVersionEditor.tsx`
- Create: `lightrag_webui/src/components/prompt-management/PromptVersionDiffDialog.tsx`
- Create: `lightrag_webui/src/components/prompt-management/PromptListFieldEditor.tsx`
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Test: `lightrag_webui/src/utils/promptVersioning.test.ts`

- [ ] **Step 1: Write failing helper tests for diff/pruning helpers**

```ts
test('formatVersionLineageLabel falls back to Deleted when source metadata is missing', () => {
  expect(formatVersionLineageLabel({ source_version_id: 'missing' }, {})).toBe('Deleted')
})

test('buildPromptEditorSections returns indexing-only fields for indexing group', () => {
  expect(buildPromptEditorSections('indexing')[0].key).toBe('entity_types')
})
```

- [ ] **Step 2: Run the helper tests to verify they fail**

Run: `cd lightrag_webui && bun test src/utils/promptVersioning.test.ts`

Expected: FAIL because prompt-management helpers and field-section definitions are incomplete.

- [ ] **Step 3: Implement the Prompt Management UI**

```tsx
export default function PromptManagement() {
  return (
    <div className="grid grid-cols-[320px_1fr] gap-4">
      <PromptVersionList
        groupType={groupType}
        versions={versions}
        activeVersionId={activeVersionId}
        onSelectVersion={setSelectedVersionId}
      />
      <PromptVersionEditor
        groupType={groupType}
        version={selectedVersion}
        onSaveVersion={handleSaveVersion}
        onActivateVersion={handleActivateVersion}
      />
    </div>
  )
}
```

Implementation notes:

- Left column: workspace, group switcher, version list, active badge, copy action.
- Right column: version metadata, optional comment, family/list editors, save, diff, activate.
- Render `ENTITY_TYPES` and prompt example arrays with the shared list-field component.
- Show indexing activation warning modal before calling activate.

- [ ] **Step 4: Re-run the helper tests and smoke-check the build**

Run: `cd lightrag_webui && bun test src/utils/promptVersioning.test.ts && bun run build`

Expected: PASS and successful production build

- [ ] **Step 5: Commit the Prompt Management page slice**

```bash
git add lightrag_webui/src/features/PromptManagement.tsx lightrag_webui/src/components/prompt-management lightrag_webui/src/api/lightrag.ts lightrag_webui/src/utils/promptVersioning.ts lightrag_webui/src/utils/promptVersioning.test.ts
git commit -m "feat: add prompt management workspace page"
```

### Task 7: Add Retrieval Temporary Version Selection and `Custom / Draft`

**Files:**
- Modify: `lightrag_webui/src/components/retrieval/QuerySettings.tsx`
- Modify: `lightrag_webui/src/components/retrieval/PromptOverridesEditor.tsx`
- Create: `lightrag_webui/src/components/retrieval/RetrievalPromptVersionSelector.tsx`
- Modify: `lightrag_webui/src/features/RetrievalTesting.tsx`
- Modify: `lightrag_webui/src/stores/settings.ts`
- Test: `lightrag_webui/src/utils/promptVersioning.test.ts`
- Test: `tests/test_query_prompt_overrides_api.py`

- [ ] **Step 1: Add failing tests for retrieval projection boundaries**

```python
def test_retrieval_version_projection_only_emits_query_and_keywords():
    payload = {
        "query": {"rag_response": "{context_data}"},
        "keywords": {"keywords_extraction": "{query}"},
        "entity_extraction": {"system_prompt": "bad"},
    }
    assert project_retrieval_payload_to_overrides(payload) == {
        "query": {"rag_response": "{context_data}"},
        "keywords": {"keywords_extraction": "{query}"},
    }
```

```ts
test('custom draft edits stay request-scoped', () => {
  expect(buildRequestPromptOverrides('custom', activePayload, draft)).toEqual(draft)
})
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `./scripts/test.sh tests/test_query_prompt_overrides_api.py -q`

Run: `cd lightrag_webui && bun test src/utils/promptVersioning.test.ts`

Expected: FAIL because retrieval-version projection helpers and draft UI state are incomplete.

- [ ] **Step 3: Implement retrieval temporary selector and draft editing**

```tsx
<RetrievalPromptVersionSelector
  value={selection}
  versions={retrievalVersions}
  onChange={setSelection}
/>

{selection === 'custom' ? (
  <PromptOverridesEditor
    value={draftOverrides}
    onChange={setDraftOverrides}
  />
) : null}
```

Implementation notes:

- Selecting a saved retrieval version must not activate it.
- Project only `query` and `keywords` into request `prompt_overrides`.
- Keep `Custom / Draft` request-scoped and offer `Save as new version`.
- Continue honoring `ALLOW_PROMPT_OVERRIDES_VIA_API`; if disabled, hide or disable temporary testing affordances.

- [ ] **Step 4: Re-run the focused tests and build**

Run: `./scripts/test.sh tests/test_query_prompt_overrides_api.py -q`

Run: `cd lightrag_webui && bun test src/utils/promptVersioning.test.ts && bun run build`

Expected: PASS

- [ ] **Step 5: Commit the retrieval-testing slice**

```bash
git add lightrag_webui/src/components/retrieval/QuerySettings.tsx lightrag_webui/src/components/retrieval/PromptOverridesEditor.tsx lightrag_webui/src/components/retrieval/RetrievalPromptVersionSelector.tsx lightrag_webui/src/features/RetrievalTesting.tsx lightrag_webui/src/stores/settings.ts lightrag_webui/src/utils/promptVersioning.ts lightrag_webui/src/utils/promptVersioning.test.ts tests/test_query_prompt_overrides_api.py
git commit -m "feat: add retrieval prompt version testing"
```

### Task 8: Update Docs and Run Final Verification

**Files:**
- Modify: `README.md`
- Modify: `README-zh.md`
- Modify: `lightrag/api/README.md`
- Modify: `lightrag/api/README-zh.md`
- Review: `docs/superpowers/specs/2026-03-25-workspace-prompt-version-management-design.md`

- [ ] **Step 1: Write the failing documentation checklist**

```text
- Project README explains indexing vs retrieval config groups
- API README documents prompt-config endpoints and fallback semantics
- Retrieval docs explain temporary version selection and Custom / Draft scope
- Docs warn that indexing activation does not rewrite existing graph data
```

- [ ] **Step 2: Run full targeted verification before doc edits**

Run: `./scripts/test.sh tests/test_prompt_versioning.py tests/test_prompt_version_store.py tests/test_prompt_version_runtime.py tests/test_prompt_config_routes.py tests/test_query_prompt_overrides_api.py -q`

Run: `cd lightrag_webui && bun test src/utils/promptVersioning.test.ts && bun run build`

Expected: PASS for the targeted backend suite, PASS for frontend tests/build

- [ ] **Step 3: Update end-user and API documentation**

```md
### Prompt Management

- Indexing configuration versions control `ENTITY_TYPES`, `SUMMARY_LANGUAGE`, and indexing prompt families
- Retrieval configuration versions control query and keyword prompt families
- If no version is active, LightRAG keeps using the existing built-in/default behavior
```

- [ ] **Step 4: Re-run the same verification after docs and final polish**

Run: `./scripts/test.sh tests/test_prompt_versioning.py tests/test_prompt_version_store.py tests/test_prompt_version_runtime.py tests/test_prompt_config_routes.py tests/test_query_prompt_overrides_api.py -q`

Run: `cd lightrag_webui && bun test src/utils/promptVersioning.test.ts && bun run build`

Expected: PASS

- [ ] **Step 5: Commit the docs and verification slice**

```bash
git add README.md README-zh.md lightrag/api/README.md lightrag/api/README-zh.md
git commit -m "docs: document prompt version management"
```
