# KG Agent Self-Repair Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each KG iteration LLM Agent stage feed backend rejection reasons back to the model and retry several times, so invalid stage output can self-correct into valid artifacts or proposals before the run stops.

**Architecture:** Keep the deterministic artifact and human-approval safety boundary unchanged. Extend `agent_pipeline.py` with structured per-attempt rejection logs and richer retry prompts, raise the default retry budget to five attempts through backend/API/WebUI defaults, and surface the attempt history in `llm_review_trace.json`, `llm_review_report.md`, and the WebUI LLM review panel.

**Tech Stack:** Python dataclasses/JSON/Markdown, FastAPI route tests, existing `lightrag.kb_iteration` modules, React 19 + TypeScript + Bun tests, Tailwind UI components.

**Execution status (2026-06-19):** Implemented with Subagent-Driven workflow through commit `b569d8dd`. The checklist below is the original execution plan; final verification recorded `142 passed` for the focused backend suite, `41 pass` for the focused frontend suite, ruff/ESLint/build passing, browser verification of attempt display, and a real Agent run that safely stopped at `invalid_llm_output` with six rejected `propose` attempts and no generated proposals.

---

## Design Source

Read before implementation:

- `D:\LightRAG\docs\superpowers\plans\2026-06-19-kg-iteration-multistage-agent-pipeline-implementation.md`
- `D:\LightRAG\docs\superpowers\plans\2026-06-19-kg-iteration-multistage-agent-pipeline-implementation-zh.md`
- `D:\LightRAG\task_plan.md`
- `D:\LightRAG\findings.md`
- `D:\LightRAG\AGENTS.md`

Important constraints:

- Do not expose or log API keys.
- Do not write raw LLM outputs into durable artifacts.
- Do not auto-apply KG mutations, patches, rule changes, prompt changes, workspace rebuilds, or WebUI behavior changes.
- Mutation proposals must remain `requires_approval=true`.
- LLM output is not medical evidence. Proposal evidence must reference deterministic artifacts such as existing `source_id`, `file_path`, entity ids, relation ids, item ids, quality metrics, or quality finding evidence.
- Preserve unrelated dirty changes. `env.example` is currently modified and must not be staged, reverted, or edited unless the user explicitly asks.

## File Structure

Modify:

- `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`
  - Add per-attempt rejection logging.
  - Feed rejection history into retry prompts.
  - Raise default stage retry budget to 5.
  - Include attempt logs in the failure Markdown report.
- `D:\LightRAG\lightrag\api\routers\kb_iteration_routes.py`
  - Raise API default `max_stage_retries` to 5 and allow up to 8.
- `D:\LightRAG\tests\kg\test_kb_iteration_agent_pipeline.py`
  - Add TDD coverage for multi-rejection self-repair, attempt logs, failure reports, and judge fallback.
- `D:\LightRAG\tests\api\routes\test_kb_iteration_routes.py`
  - Add route default coverage for `max_stage_retries=5`.
- `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`
  - Run LLM review with `max_stage_retries: 5`.
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`
  - Parse and render stage attempt logs.
- `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx`
  - Add WebUI rendering coverage for self-repair attempts.

Do not modify:

- `D:\LightRAG\env.example`
- Runtime artifacts under `D:\LightRAG\work\`
- Generated WebUI build output under `D:\LightRAG\lightrag\api\webui\`

## Trace Schema

Each stage trace keeps current fields and adds `attempt_logs`:

```json
{
  "stage": "propose",
  "state": "completed",
  "attempts": 3,
  "attempt_logs": [
    {
      "attempt": 1,
      "state": "invalid_llm_output",
      "error": "proposal expected_metric_change values must be numbers",
      "output_token_estimate": 312
    },
    {
      "attempt": 2,
      "state": "invalid_llm_output",
      "error": "proposal evidence is not grounded in deterministic artifacts",
      "output_token_estimate": 280
    }
  ]
}
```

Notes:

- `attempt_logs` only records structured summaries.
- Do not include raw LLM JSON output.
- A successful final attempt is represented by the stage's `state: "completed"` and `attempts`; it does not need a success log entry.
- Client exceptions may also add `state: "client_error"` attempt logs, but retry prompts for client errors should not claim the model's JSON was rejected.

## Task 1: Backend Attempt Logs And Retry Prompt History

**Files:**

- Modify: `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`
- Modify: `D:\LightRAG\tests\kg\test_kb_iteration_agent_pipeline.py`

- [ ] **Step 1: Add failing tests for multi-rejection self-repair**

Append this client class after `RetryProposeMetricAgentClient` in `D:\LightRAG\tests\kg\test_kb_iteration_agent_pipeline.py`:

```python
class RetryProposeTwiceAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        invalid_metric_proposal = dict(self.outputs[3]["proposals"][0])
        invalid_metric_proposal["expected_metric_change"] = {
            "hierarchy_completeness": "about 6"
        }
        invalid_evidence_proposal = dict(self.outputs[3]["proposals"][0])
        invalid_evidence_proposal["evidence"] = ["not-grounded-source"]
        self.outputs.insert(3, {"proposals": [invalid_metric_proposal]})
        self.outputs.insert(4, {"proposals": [invalid_evidence_proposal]})
```

Append this test near `test_pipeline_feeds_validation_errors_into_retry_prompt`:

```python
def test_pipeline_retries_with_accumulated_rejection_history(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = RetryProposeTwiceAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=2),
    )

    assert result.stop_reason == "pending_human_review"
    assert result.proposal_ids == ["proposal-hierarchy-symptom-001"]
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][3]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["attempts"] == 3
    attempt_logs = propose_trace["attempt_logs"]
    assert len(attempt_logs) == 2
    assert attempt_logs[0]["attempt"] == 1
    assert attempt_logs[0]["state"] == "invalid_llm_output"
    assert (
        attempt_logs[0]["error"]
        == "proposal expected_metric_change values must be numbers"
    )
    assert isinstance(attempt_logs[0]["output_token_estimate"], int)
    assert attempt_logs[0]["output_token_estimate"] > 0
    assert attempt_logs[1]["attempt"] == 2
    assert attempt_logs[1]["state"] == "invalid_llm_output"
    assert (
        attempt_logs[1]["error"]
        == "proposal proposal-hierarchy-symptom-001 evidence is not grounded in deterministic artifacts"
    )
    assert isinstance(attempt_logs[1]["output_token_estimate"], int)
    assert attempt_logs[1]["output_token_estimate"] > 0
    third_propose_prompt = client.calls[5]["user_prompt"]
    assert "Previous rejected attempts:" in third_propose_prompt
    assert "Attempt 1: proposal expected_metric_change values must be numbers" in third_propose_prompt
    assert (
        "Attempt 2: proposal proposal-hierarchy-symptom-001 evidence is not grounded in deterministic artifacts"
        in third_propose_prompt
    )
    assert "Return corrected JSON only." in third_propose_prompt
    assert "Do not change evidence/human-approval requirements." in third_propose_prompt
```

- [ ] **Step 2: Run the failing backend test**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_pipeline.py::test_pipeline_retries_with_accumulated_rejection_history -q
```

Expected:

- FAIL because `attempt_logs` is absent and retry prompts include only the latest error.

- [ ] **Step 3: Add attempt log helpers**

In `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`, add these helpers near `_retry_user_prompt`:

```python
def _append_attempt_log(
    stage_trace: dict[str, Any],
    *,
    attempt: int,
    state: str,
    error: str,
    raw_output: str = "",
) -> None:
    attempt_logs = stage_trace.setdefault("attempt_logs", [])
    if not isinstance(attempt_logs, list):
        attempt_logs = []
        stage_trace["attempt_logs"] = attempt_logs
    attempt_logs.append(
        {
            "attempt": attempt,
            "state": state,
            "error": error,
            "output_token_estimate": _estimate_tokens(raw_output) if raw_output else 0,
        }
    )


def _attempt_error_lines(stage_trace: dict[str, Any]) -> list[str]:
    lines = []
    attempt_logs = stage_trace.get("attempt_logs")
    if not isinstance(attempt_logs, list):
        return lines
    for item in attempt_logs:
        if not isinstance(item, dict):
            continue
        attempt = item.get("attempt")
        error = item.get("error")
        if isinstance(attempt, int) and isinstance(error, str) and error:
            lines.append(f"Attempt {attempt}: {error}")
    return lines
```

- [ ] **Step 4: Initialize `attempt_logs` in stage trace**

In `run_llm_agent_pipeline()`, update the `stage_trace` literal to include:

```python
            "attempt_logs": [],
```

The updated block should be:

```python
        stage_trace: dict[str, Any] = {
            "stage": stage,
            "started_at": _utc_timestamp(),
            "completed_at": "",
            "state": "running",
            "attempts": 0,
            "attempt_logs": [],
            "context_files": [_relative_artifact_path(output_dir, context_path)],
            "model": _client_model(client),
            "input_token_estimate": input_token_estimate,
            "output_token_estimate": 0,
            "proposal_ids": [],
            "artifact_keys": [],
            "error": "",
        }
```

- [ ] **Step 5: Log validation rejections before retrying**

In the `except ValueError as exc:` block inside the stage attempt loop, replace:

```python
                stage_trace["error"] = str(exc)
                stage_trace["output_token_estimate"] = _estimate_tokens(raw_output)
                if attempt < max_attempts:
                    user_prompt = _retry_user_prompt(
                        base_user_prompt,
                        stage=stage,
                        error=str(exc),
                    )
                    continue
```

with:

```python
                error = str(exc)
                stage_trace["error"] = error
                stage_trace["output_token_estimate"] = _estimate_tokens(raw_output)
                _append_attempt_log(
                    stage_trace,
                    attempt=attempt,
                    state="invalid_llm_output",
                    error=error,
                    raw_output=raw_output,
                )
                if attempt < max_attempts:
                    user_prompt = _retry_user_prompt(
                        base_user_prompt,
                        stage=stage,
                        error=error,
                        previous_errors=_attempt_error_lines(stage_trace),
                    )
                    continue
```

Also replace downstream `str(exc)` uses in that block with `error`:

```python
                    return _finish_judge_unavailable(
                        output_dir=output_dir,
                        error=error,
                        proposals=proposals,
                        previous_outputs=previous_outputs,
                        artifact_paths=artifact_paths,
                        trace=trace,
                        stage_trace=stage_trace,
                    )
```

- [ ] **Step 6: Update `_retry_user_prompt` signature and body**

Replace `_retry_user_prompt()` with:

```python
def _retry_user_prompt(
    base_user_prompt: str,
    *,
    stage: str,
    error: str,
    previous_errors: list[str] | None = None,
) -> str:
    guidance = [
        f"Previous output was rejected: {error}.",
        "Return corrected JSON only.",
    ]
    if previous_errors:
        guidance.extend(["Previous rejected attempts:", *previous_errors])
    guidance.extend(
        [
            "Preserve deterministic evidence requirements.",
            "Do not invent source_id, file_path, entity_id, relation_id, item_id, or metric references.",
            "Do not claim LLM inference as medical evidence.",
        ]
    )
    if stage == "propose":
        guidance.append(
            'For "expected_metric_change", use finite JSON numbers only; '
            "use {} if no numeric estimate is available."
        )
        guidance.append(
            "Every non-context proposal must keep requires_approval=true when it can mutate KG, rules, prompts, workspace, or WebUI behavior."
        )
        guidance.append("Do not change evidence/human-approval requirements.")
    return "\n\n".join([base_user_prompt, *guidance])
```

- [ ] **Step 7: Run the new test and existing retry tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_pipeline.py::test_pipeline_retries_with_accumulated_rejection_history tests/kg/test_kb_iteration_agent_pipeline.py::test_pipeline_retries_invalid_stage_output tests/kg/test_kb_iteration_agent_pipeline.py::test_pipeline_feeds_validation_errors_into_retry_prompt -q
```

Expected:

- PASS.

- [ ] **Step 8: Commit Task 1**

Run:

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration/agent_pipeline.py tests/kg/test_kb_iteration_agent_pipeline.py
git commit -m "feat: record kg agent self-repair attempts"
```

## Task 2: Retry Budget Defaults And API Boundaries

**Files:**

- Modify: `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`
- Modify: `D:\LightRAG\lightrag\api\routers\kb_iteration_routes.py`
- Modify: `D:\LightRAG\tests\api\routes\test_kb_iteration_routes.py`

- [ ] **Step 1: Add failing API default test**

Append this test after `test_llm_review_run_defaults_to_agent_pipeline` in `D:\LightRAG\tests\api\routes\test_kb_iteration_routes.py`:

```python
def test_llm_review_run_defaults_to_five_agent_stage_retries(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    agent_calls = []

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "{}"

    def fake_run_llm_agent_pipeline(**kwargs):
        agent_calls.append(kwargs)
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="agent_done",
            proposal_ids=[],
            artifact_paths={},
        )

    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_agent_pipeline",
        fake_run_llm_agent_pipeline,
    )
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda rag: FakeClient(),
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json={"profile": "clinical_guideline_zh"},
    )

    assert response.status_code == 200
    assert len(agent_calls) == 1
    assert agent_calls[0]["config"].max_stage_retries == 5
```

Append this test near other request validation tests:

```python
def test_llm_review_run_accepts_agent_stage_retries_up_to_eight(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.routers import kb_iteration_routes

    client, fixture = _client(tmp_path, monkeypatch)
    agent_calls = []

    class FakeClient:
        def complete(self, *, system_prompt: str, user_prompt: str) -> str:
            return "{}"

    def fake_run_llm_agent_pipeline(**kwargs):
        agent_calls.append(kwargs)
        return SimpleNamespace(
            output_dir=fixture.package,
            stop_reason="agent_done",
            proposal_ids=[],
            artifact_paths={},
        )

    monkeypatch.setattr(
        kb_iteration_routes,
        "run_llm_agent_pipeline",
        fake_run_llm_agent_pipeline,
    )
    monkeypatch.setattr(
        kb_iteration_routes,
        "_default_llm_review_client",
        lambda rag: FakeClient(),
    )

    response = client.post(
        "/kb-iteration/influenza_medical_v1/llm-review/runs",
        headers=HEADERS,
        json={"max_stage_retries": 8},
    )

    assert response.status_code == 200
    assert agent_calls[0]["config"].max_stage_retries == 8
```

- [ ] **Step 2: Run the failing API tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/api/routes/test_kb_iteration_routes.py::test_llm_review_run_defaults_to_five_agent_stage_retries tests/api/routes/test_kb_iteration_routes.py::test_llm_review_run_accepts_agent_stage_retries_up_to_eight -q
```

Expected:

- The first test FAILS because route default is currently 1.
- The second test FAILS with validation error because route max is currently 3.

- [ ] **Step 3: Raise backend config default**

In `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`, change:

```python
    max_stage_retries: int = 1
```

to:

```python
    max_stage_retries: int = 5
```

- [ ] **Step 4: Raise API default and max**

In `D:\LightRAG\lightrag\api\routers\kb_iteration_routes.py`, change:

```python
    max_stage_retries: int = Field(default=1, ge=0, le=3)
```

to:

```python
    max_stage_retries: int = Field(default=5, ge=0, le=8)
```

- [ ] **Step 5: Run API tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/api/routes/test_kb_iteration_routes.py::test_llm_review_run_defaults_to_five_agent_stage_retries tests/api/routes/test_kb_iteration_routes.py::test_llm_review_run_accepts_agent_stage_retries_up_to_eight tests/api/routes/test_kb_iteration_routes.py::test_llm_review_run_defaults_to_agent_pipeline -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration/agent_pipeline.py lightrag/api/routers/kb_iteration_routes.py tests/api/routes/test_kb_iteration_routes.py
git commit -m "feat: increase kg agent repair retry budget"
```

## Task 3: Failure Report Includes Rejection Attempts

**Files:**

- Modify: `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`
- Modify: `D:\LightRAG\tests\kg\test_kb_iteration_agent_pipeline.py`

- [ ] **Step 1: Add failing test for final failure report**

Add this client after `RetryProposeTwiceAgentClient`:

```python
class AlwaysInvalidProposeAgentClient(SequencedAgentClient):
    def __init__(self) -> None:
        super().__init__()
        invalid_proposal = dict(self.outputs[3]["proposals"][0])
        invalid_proposal["expected_metric_change"] = {
            "hierarchy_completeness": "about 6"
        }
        self.outputs[3] = {"proposals": [invalid_proposal]}

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        index = len(self.calls) - 1
        if index >= len(self.outputs):
            index = 3
        output = self.outputs[index]
        if isinstance(output, str):
            return output
        return json.dumps(output, ensure_ascii=False)
```

Add this test near failure artifact tests:

```python
def test_pipeline_failure_report_lists_rejected_attempts(tmp_path: Path):
    package = tmp_path / "package"
    _write_agent_package(package)
    client = AlwaysInvalidProposeAgentClient()

    result = run_llm_agent_pipeline(
        workspace="demo",
        package_dir=package,
        client=client,
        config=LLMAgentPipelineConfig(max_stage_retries=2),
    )

    assert result.stop_reason == "invalid_llm_output"
    assert result.proposal_ids == []
    report = (package / "llm_review_report.md").read_text(encoding="utf-8")
    assert "## Rejected Attempts" in report
    assert "- Attempt 1: proposal expected_metric_change values must be numbers" in report
    assert "- Attempt 2: proposal expected_metric_change values must be numbers" in report
    assert "- Attempt 3: proposal expected_metric_change values must be numbers" in report
    trace = json.loads((package / "llm_review_trace.json").read_text(encoding="utf-8"))
    propose_trace = trace["stages"][-1]
    assert propose_trace["stage"] == "propose"
    assert propose_trace["attempts"] == 3
    assert len(propose_trace["attempt_logs"]) == 3
```

- [ ] **Step 2: Run the failing report test**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_pipeline.py::test_pipeline_failure_report_lists_rejected_attempts -q
```

Expected:

- FAIL because `llm_review_report.md` does not include `## Rejected Attempts`.

- [ ] **Step 3: Add report summary helper**

In `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`, add this helper near `_finish_run`:

```python
def _failure_attempt_report(trace: dict[str, Any]) -> str:
    stages = trace.get("stages")
    if not isinstance(stages, list) or not stages:
        return ""
    stage_trace = stages[-1]
    if not isinstance(stage_trace, dict):
        return ""
    attempt_logs = stage_trace.get("attempt_logs")
    if not isinstance(attempt_logs, list) or not attempt_logs:
        return ""

    lines = ["", "## Rejected Attempts"]
    for item in attempt_logs:
        if not isinstance(item, dict):
            continue
        attempt = item.get("attempt")
        error = item.get("error")
        if isinstance(attempt, int) and isinstance(error, str) and error:
            lines.append(f"- Attempt {attempt}: {error}")
    return "\n".join(lines)
```

- [ ] **Step 4: Append attempts in `_write_failure_artifacts()`**

In `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`, inside `_write_failure_artifacts()`, replace:

```python
    report_lines.extend(["", "## Generated Proposals", "", "- none", ""])
```

with:

```python
    attempt_logs = latest_stage.get("attempt_logs")
    if isinstance(attempt_logs, list) and attempt_logs:
        report_lines.extend(["", "## Rejected Attempts", ""])
        for item in attempt_logs:
            if not isinstance(item, dict):
                continue
            attempt = item.get("attempt")
            attempt_error = item.get("error")
            if isinstance(attempt, int) and isinstance(attempt_error, str):
                clean_error = _single_line(attempt_error)
                if clean_error:
                    report_lines.append(f"- Attempt {attempt}: {clean_error}")
    report_lines.extend(["", "## Generated Proposals", "", "- none", ""])
```

- [ ] **Step 5: Run report test**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_agent_pipeline.py::test_pipeline_failure_report_lists_rejected_attempts -q
```

Expected:

- PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration/agent_pipeline.py tests/kg/test_kb_iteration_agent_pipeline.py
git commit -m "feat: report kg agent rejected attempts"
```

## Task 4: WebUI Shows Self-Repair Attempts

**Files:**

- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx`
- Modify: `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`

- [ ] **Step 1: Add failing panel test**

Append this test in `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.test.tsx` after the multistage stage test:

```tsx
  test('renders multistage self-repair attempt history', () => {
    const markup = renderToStaticMarkup(
      <LLMReviewPanel
        trace={{
          stop_reason: 'invalid_llm_output',
          stages: [
            {
              stage: 'propose',
              state: 'invalid_llm_output',
              attempts: 3,
              attempt_logs: [
                {
                  attempt: 1,
                  state: 'invalid_llm_output',
                  error: 'proposal expected_metric_change values must be numbers'
                },
                {
                  attempt: 2,
                  state: 'invalid_llm_output',
                  error: 'proposal evidence is not grounded in deterministic artifacts'
                }
              ]
            }
          ]
        }}
        report="# LLM Review Report"
        proposals=""
        issueAnalysis=""
        missingBranchInference=""
        evidenceMap=""
        repairPlan=""
        running={false}
        onRun={() => undefined}
      />
    )

    expect(markup).toContain('attempt')
    expect(markup).toContain('3')
    expect(markup).toContain('proposal expected_metric_change values must be numbers')
    expect(markup).toContain('proposal evidence is not grounded in deterministic artifacts')
    expect(markup).not.toContain('[object Object]')
  })
```

- [ ] **Step 2: Run the failing panel test**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx
```

Expected:

- FAIL because `TraceStage` does not normalize/render `attempts` or `attempt_logs`.

- [ ] **Step 3: Extend trace types**

In `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`, add:

```ts
export type TraceAttemptLog = {
  attempt?: number
  state?: string
  error?: string
}
```

Change `TraceStage` to:

```ts
export type TraceStage = {
  stage?: string
  state?: string
  attempts?: number
  artifact_keys?: string[]
  proposal_ids?: string[]
  attempt_logs?: TraceAttemptLog[]
}
```

- [ ] **Step 4: Normalize attempt fields**

Update `normalizeTraceStageEntry()` return value:

```ts
  return {
    stage: typeof stage.stage === 'string' ? stage.stage : undefined,
    state: typeof stage.state === 'string' ? stage.state : undefined,
    attempts: typeof stage.attempts === 'number' && Number.isFinite(stage.attempts)
      ? stage.attempts
      : undefined,
    artifact_keys: stringArray(stage.artifact_keys),
    proposal_ids: stringArray(stage.proposal_ids),
    attempt_logs: attemptLogArray(stage.attempt_logs)
  }
```

Add this helper near `stringArray()`:

```ts
function attemptLogArray(value: unknown) {
  if (!Array.isArray(value)) {
    return undefined
  }

  const logs = value
    .map((item): TraceAttemptLog | null => {
      if (!isRecord(item)) {
        return null
      }
      const attempt =
        typeof item.attempt === 'number' && Number.isFinite(item.attempt)
          ? item.attempt
          : undefined
      const state = typeof item.state === 'string' ? item.state : undefined
      const error = typeof item.error === 'string' ? item.error : undefined
      if (attempt === undefined && !state && !error) {
        return null
      }
      return { attempt, state, error }
    })
    .filter((item): item is TraceAttemptLog => item !== null)

  return logs.length ? logs : undefined
}
```

- [ ] **Step 5: Render attempts inside each stage card**

Inside the stage `<article>` in `LLMReviewPanel`, after the state badge block and before `RoundField`, add:

```tsx
                {typeof stage?.attempts === 'number' ? (
                  <div className="text-muted-foreground mt-2 text-xs">
                    attempt: <span className="font-medium">{stage.attempts}</span>
                  </div>
                ) : null}
                <AttemptLogList logs={stage?.attempt_logs} />
```

Add this component near `RoundField()`:

```tsx
function AttemptLogList({ logs }: { logs?: TraceAttemptLog[] }) {
  if (!Array.isArray(logs) || logs.length === 0) {
    return null
  }

  return (
    <div className="mt-3 space-y-1">
      <div className="text-muted-foreground text-xs">rejected attempts</div>
      {logs.map((log, index) => (
        <div key={`${log.attempt || index}-${log.error || ''}`} className="bg-muted rounded-md px-2 py-1 text-xs">
          <span className="font-medium">Attempt {log.attempt || index + 1}</span>
          {log.state ? <span className="text-muted-foreground"> · {log.state}</span> : null}
          {log.error ? <div className="mt-1 break-words">{log.error}</div> : null}
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 6: Raise WebUI default retry budget**

In `D:\LightRAG\lightrag_webui\src\features\KGMaintenanceConsole.tsx`, change:

```ts
          max_stage_retries: 1,
```

to:

```ts
          max_stage_retries: 5,
```

- [ ] **Step 7: Run frontend panel test**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx
```

Expected:

- PASS.

- [ ] **Step 8: Commit Task 4**

Run:

```powershell
cd D:\LightRAG
git add lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx lightrag_webui/src/features/KGMaintenanceConsole.tsx
git commit -m "feat: show kg agent self-repair attempts"
```

## Task 5: End-To-End Verification And Real Agent Run

**Files:**

- Modify only if verification reveals concrete issues:
  - `D:\LightRAG\lightrag\kb_iteration\agent_pipeline.py`
  - `D:\LightRAG\lightrag\api\routers\kb_iteration_routes.py`
  - `D:\LightRAG\lightrag_webui\src\components\kg-maintenance\LLMReviewPanels.tsx`
  - matching tests

- [ ] **Step 1: Run focused backend tests**

Run:

```powershell
cd D:\LightRAG
uv run pytest tests/kg/test_kb_iteration_quality.py tests/kg/test_kb_iteration_review_context.py tests/kg/test_kb_iteration_agent_context.py tests/kg/test_kb_iteration_agent_outputs.py tests/kg/test_kb_iteration_agent_pipeline.py tests/kg/test_kb_iteration_review_loop.py tests/api/routes/test_kb_iteration_routes.py -q
```

Expected:

- PASS.

- [ ] **Step 2: Run focused frontend tests**

Run:

```powershell
cd D:\LightRAG\lightrag_webui
npx --yes bun test src/components/kg-maintenance/LLMReviewPanels.test.tsx src/components/kg-maintenance/kgIterationLoadUtils.test.ts src/components/kg-maintenance/KGMaintenanceShell.test.tsx
```

Expected:

- PASS.

- [ ] **Step 3: Run lint and build**

Run:

```powershell
cd D:\LightRAG
uv run ruff check lightrag/kb_iteration/agent_pipeline.py lightrag/api/routers/kb_iteration_routes.py tests/kg/test_kb_iteration_agent_pipeline.py tests/api/routes/test_kb_iteration_routes.py
cd D:\LightRAG\lightrag_webui
npx --yes bun run lint
npx --yes bun run build
```

Expected:

- PASS.

- [ ] **Step 4: Run one real Agent review**

Use the running server if available. If the server is not running, start it with the user's existing batch script, not a new startup method:

```powershell
C:\Users\89897\Desktop\Restart LightRAG Server.bat 9621 influenza_medical_v1
```

Then run:

```powershell
$body = @{
  profile = "clinical_guideline_zh"
  mode = "agent_pipeline"
  max_stage_retries = 5
  allow_llm_judge = $true
  require_human_for_mutation = $true
  generate_patch_candidates = $false
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:9621/kb-iteration/influenza_medical_v1/llm-review/runs" -ContentType "application/json" -Body $body
```

Expected:

- Preferably `stopReason` is `pending_human_review`.
- If it still fails, `llm_review_trace.json` must include `attempt_logs` for the failed stage and `llm_review_report.md` must include `## Rejected Attempts`.
- `approval_queue.md` must contain proposals only when `stopReason` is `pending_human_review`; otherwise it may contain `proposals: []`.
- No raw LLM output or API key is written to artifacts.

- [ ] **Step 5: Browser review**

Open the current WebUI URL, for example:

```text
http://127.0.0.1:9621/webui/
```

Verify:

- `LLM 审阅材料` still shows `多阶段 LLM Agent`.
- Stage cards show `attempt` when trace contains attempts.
- Rejected attempt errors are readable.
- Proposal approval remains empty when no proposals are generated.
- Proposal approval shows proposals only after `pending_human_review`.
- Safety notice still says LLM does not automatically modify KG.

- [ ] **Step 6: Commit verification polish if needed**

If verification required edits, commit them:

```powershell
cd D:\LightRAG
git add lightrag/kb_iteration/agent_pipeline.py lightrag/api/routers/kb_iteration_routes.py tests/kg/test_kb_iteration_agent_pipeline.py tests/api/routes/test_kb_iteration_routes.py lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.tsx lightrag_webui/src/components/kg-maintenance/LLMReviewPanels.test.tsx lightrag_webui/src/features/KGMaintenanceConsole.tsx
git commit -m "fix: polish kg agent self-repair loop"
```

If no files changed, do not create an empty commit.

## Completion Checklist

- [ ] Every Agent stage records structured `attempt_logs` for rejected outputs.
- [ ] Retry prompts include current rejection reason and accumulated rejection history.
- [ ] Retry prompts preserve deterministic evidence and human-approval safety requirements.
- [ ] Default stage retry budget is 5 in backend config, API request model, and WebUI run request.
- [ ] API accepts `max_stage_retries` up to 8.
- [ ] Failure reports include `## Rejected Attempts`.
- [ ] WebUI displays stage attempt counts and rejection reasons without rendering object garbage.
- [ ] No raw LLM output or API key is persisted.
- [ ] Backend focused tests pass.
- [ ] Frontend focused tests pass.
- [ ] Frontend lint and build pass.
- [ ] Real Agent run either reaches `pending_human_review` or produces clear attempt logs explaining why it could not.
