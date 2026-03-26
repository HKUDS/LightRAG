# Prompt Inline Save And Index Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users save edits back into the current prompt version, rebuild the workspace from a selected indexing version without deleting source files, and interrupt all document pipelines directly from the document page toolbar.

**Architecture:** Extend the prompt version store and prompt-config API with an update path for in-place saves, then add a document-domain rebuild endpoint that activates a selected indexing version, drops workspace data stores, preserves `input_dir` files, and schedules a fresh scan. In the WebUI, update prompt-management actions to prefer saving the current version while keeping save-as available, and surface a direct cancel-pipeline button on the document toolbar by reusing the existing cancellation API.

**Tech Stack:** Python, FastAPI, Pydantic, pytest via `./scripts/test.sh`, React 19, TypeScript, Vitest, Bun

---

## File Structure

- Modify: `lightrag/prompt_version_store.py`
  - Add in-place version update support with payload validation and metadata preservation.
- Modify: `lightrag/api/routers/prompt_config_routes.py`
  - Add explicit update request model and update route.
- Modify: `lightrag/api/routers/document_routes.py`
  - Add rebuild request/response models and a rebuild endpoint that preserves source files.
- Modify: `tests/test_prompt_version_store.py`
  - Cover in-place update semantics.
- Modify: `tests/test_prompt_config_routes.py`
  - Cover prompt-config update route and rebuild route contract.
- Modify: `lightrag_webui/src/api/lightrag.ts`
  - Add update-version and rebuild-indexing API types/functions.
- Modify: `lightrag_webui/src/features/PromptManagement.tsx`
  - Use update for normal save and wire rebuild action.
- Modify: `lightrag_webui/src/components/prompt-management/PromptVersionEditor.tsx`
  - Default to current-version editing, keep optional save-as, and expose rebuild action only for indexing.
- Modify: `lightrag_webui/src/components/prompt-management/PromptVersionEditor.test.tsx`
  - Cover updated button labels and indexing-only rebuild affordance.
- Modify: `lightrag_webui/src/features/DocumentManager.tsx`
  - Add direct cancel-all-pipelines toolbar button.
- Modify: `lightrag_webui/src/locales/zh.json`
- Modify: `lightrag_webui/src/locales/en.json`
  - Add/adjust strings for in-place save, save-as, rebuild, and direct cancel button.

### Task 1: Add store-level in-place update support

**Files:**
- Modify: `lightrag/prompt_version_store.py`
- Test: `tests/test_prompt_version_store.py`

- [ ] **Step 1: Write the failing test**

```python
def test_update_version_reuses_version_identity_and_updates_payload(tmp_path):
    store = PromptVersionStore(tmp_path, workspace="demo")
    registry = store.initialize(locale="en")
    version = registry["retrieval"]["versions"][0]

    updated = store.update_version(
        "retrieval",
        version["version_id"],
        {"query": {"rag_response": "UPDATED {context_data}"}},
        "renamed",
        "edited",
    )

    assert updated["version_id"] == version["version_id"]
    assert updated["version_number"] == version["version_number"]
    assert updated["version_name"] == "renamed"
    assert updated["comment"] == "edited"
    assert updated["payload"]["query"]["rag_response"] == "UPDATED {context_data}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./scripts/test.sh tests/test_prompt_version_store.py -q`
Expected: FAIL because `PromptVersionStore.update_version` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def update_version(self, group_type, version_id, payload, version_name, comment):
    normalized_payload = normalize_prompt_group_payload(group_type, payload)
    validate_prompt_group_payload(group_type, normalized_payload)
    registry = self._read_or_default()
    for version in registry[group_type]["versions"]:
        if version["version_id"] == version_id:
            version["version_name"] = version_name
            version["comment"] = comment
            version["payload"] = deepcopy(normalized_payload)
            self._atomic_write(registry)
            return deepcopy(version)
    raise ValueError(...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./scripts/test.sh tests/test_prompt_version_store.py -q`
Expected: PASS

### Task 2: Expose prompt-config update API and indexing rebuild API

**Files:**
- Modify: `lightrag/api/routers/prompt_config_routes.py`
- Modify: `lightrag/api/routers/document_routes.py`
- Modify: `tests/test_prompt_config_routes.py`

- [ ] **Step 1: Write the failing API tests**

```python
def test_update_prompt_version_updates_selected_record(test_client):
    seeded = test_client.post("/prompt-config/initialize").json()
    version_id = seeded["retrieval"]["versions"][0]["version_id"]

    response = test_client.patch(
        f"/prompt-config/retrieval/versions/{version_id}",
        json={
            "version_name": "retrieval-inline",
            "comment": "edited",
            "payload": {"query": {"rag_response": "INLINE {context_data}"}},
        },
    )

    assert response.status_code == 200
    assert response.json()["version_id"] == version_id


def test_rebuild_endpoint_preserves_source_files_and_starts_scan(test_client):
    seeded = test_client.post("/prompt-config/initialize").json()
    version_id = seeded["indexing"]["versions"][0]["version_id"]

    response = test_client.post(
        "/documents/rebuild_from_indexing_version",
        json={"version_id": version_id},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "rebuild_started"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./scripts/test.sh tests/test_prompt_config_routes.py -q`
Expected: FAIL because the update/rebuild endpoints do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
@router.patch("/{group_type}/versions/{version_id}")
async def update_version(...):
    return _store().update_version(...)


@router.post("/rebuild_from_indexing_version", response_model=RebuildDocumentsResponse)
async def rebuild_from_indexing_version(...):
    # reject when pipeline busy
    # activate selected indexing version
    # drop storages only, do not delete input_dir files
    # background_tasks.add_task(run_scanning_process, ...)
    return RebuildDocumentsResponse(status="rebuild_started", ...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./scripts/test.sh tests/test_prompt_config_routes.py -q`
Expected: PASS

### Task 3: Update WebUI prompt-management flows

**Files:**
- Modify: `lightrag_webui/src/api/lightrag.ts`
- Modify: `lightrag_webui/src/features/PromptManagement.tsx`
- Modify: `lightrag_webui/src/components/prompt-management/PromptVersionEditor.tsx`
- Modify: `lightrag_webui/src/components/prompt-management/PromptVersionEditor.test.tsx`
- Modify: `lightrag_webui/src/locales/zh.json`
- Modify: `lightrag_webui/src/locales/en.json`

- [ ] **Step 1: Write the failing UI test**

```tsx
test('shows save-current and indexing-only rebuild actions', async () => {
  const html = renderToString(
    <PromptVersionEditor
      groupType="indexing"
      ...
    />
  )

  expect(html).toContain('promptManagement.saveCurrentVersion')
  expect(html).toContain('promptManagement.rebuildFromSelectedVersion')
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd lightrag_webui && bun test src/components/prompt-management/PromptVersionEditor.test.tsx`
Expected: FAIL because the new actions and labels are not rendered yet.

- [ ] **Step 3: Write minimal implementation**

```tsx
useEffect(() => {
  setVersionName(version?.version_name ?? '')
  setComment(version?.comment ?? '')
  setPayload(version ? structuredClone(version.payload) : {})
}, [version])

<Button onClick={saveCurrentVersion}>{t('promptManagement.saveCurrentVersion')}</Button>
<Button onClick={saveAsNewVersion}>{t('promptManagement.saveAsNewVersion')}</Button>
{groupType === 'indexing' ? (
  <Button onClick={rebuildFromVersion}>{t('promptManagement.rebuildFromSelectedVersion')}</Button>
) : null}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd lightrag_webui && bun test src/components/prompt-management/PromptVersionEditor.test.tsx`
Expected: PASS

### Task 4: Add direct cancel-all-pipelines toolbar action

**Files:**
- Modify: `lightrag_webui/src/features/DocumentManager.tsx`
- Modify: `lightrag_webui/src/locales/zh.json`
- Modify: `lightrag_webui/src/locales/en.json`

- [ ] **Step 1: Add a focused render test or extend an existing component test**

```tsx
test('document toolbar exposes direct cancel action copy', async () => {
  // render the toolbar-bearing component or focused child and assert new label
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd lightrag_webui && bun test`
Expected: FAIL because the toolbar action is not rendered yet.

- [ ] **Step 3: Write minimal implementation**

```tsx
<Button
  variant="destructive"
  disabled={!pipelineBusy}
  onClick={handleCancelPipeline}
>
  {t('documentPanel.documentManager.cancelPipelineButton')}
</Button>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd lightrag_webui && bun test`
Expected: PASS

### Task 5: Run focused verification

**Files:**
- Verify only

- [ ] **Step 1: Run focused backend tests**

Run: `./scripts/test.sh tests/test_prompt_version_store.py tests/test_prompt_config_routes.py -q`
Expected: PASS

- [ ] **Step 2: Run focused frontend tests**

Run: `cd lightrag_webui && bun test src/components/prompt-management/PromptVersionEditor.test.tsx`
Expected: PASS

- [ ] **Step 3: Run frontend build**

Run: `cd lightrag_webui && bun run build`
Expected: PASS
