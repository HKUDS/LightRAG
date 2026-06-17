# Knowledge Base Iteration Agent Design

## Goal

Design a semi-automatic agent workflow for maintaining and improving a LightRAG knowledge base across repeated document-ingestion and rebuild cycles.

The agent must help an LLM quickly understand the current knowledge base, evaluate entity and relationship quality, propose structured improvements, and preserve a durable audit trail. It may generate reports and proposals automatically, but prompt changes, ontology/rule changes, factual KG changes, workspace deletion, and workspace rebuilds must remain confirmation-gated.

The first concrete acceptance scenario is the current influenza medical workspace:

- Workspace: `influenza_medical_v1`
- Storage root: `data/rag_storage/influenza_medical_v1`
- Input root: `data/inputs/influenza_medical_v1`
- Medical profile: `clinical_guideline_zh`
- Important data files:
  - `graph_chunk_entity_relation.graphml`
  - `kv_store_full_entities.json`
  - `kv_store_full_relations.json`
  - `vdb_entities.json`
  - `vdb_relationships.json`

The design must also support future workspaces and non-influenza medical documents without turning the agent into a flu-only maintenance script.

## Problem

LightRAG can extract entities and relationships into a graph, but an evolving knowledge base needs more than one-time extraction. Over time the project needs to answer:

- What entities, relationships, categories, and source documents exist now?
- Which parts of the graph are noisy, duplicated, weakly sourced, or hard to understand?
- Did the latest prompt/rule/workspace iteration improve the KG or make it worse?
- Which proposed fixes are safe to apply automatically, and which require human confirmation?
- Which failure cases have already been accepted or rejected in earlier review cycles?

Without a persistent KB iteration layer, each LLM session must rediscover the graph from raw files, quality checks remain subjective, and rejected suggestions can reappear repeatedly.

## Confirmed Direction

Use option B: a semi-automatic KB iteration agent.

The agent may automatically:

- Read the current KG and generate machine snapshots.
- Generate compact Markdown memory for LLMs.
- Run quality checks and score the KG.
- Produce structured improvement proposals.
- Compare two workspace snapshots.
- Update reports and logs.

The agent may not automatically:

- Add unsupported medical facts.
- Edit extraction prompts.
- Edit ontology, alias, relation, or hierarchy rules.
- Delete, clear, or rebuild a LightRAG workspace.
- Apply proposed fact-level changes to KG data.

Those actions must enter an approval queue and wait for explicit user confirmation.

## Non-Goals

This first design does not build a fully autonomous medical ontology curator. It also does not replace LightRAG extraction, rewrite the medical KG profile, or create a new WebUI review console.

The first implementation should generate file-based artifacts and quality reports. A later UI can read these artifacts, but the file workflow is the source of truth for the initial version.

## Current Project Hooks

The agent should integrate with existing LightRAG data and source boundaries:

| Need | Existing hook |
| --- | --- |
| Current workspace data | `.env`, `WORKING_DIR`, `INPUT_DIR`, `WORKSPACE` |
| Graph topology | `graph_chunk_entity_relation.graphml` |
| Per-document entity index | `kv_store_full_entities.json` |
| Per-document relation index | `kv_store_full_relations.json` |
| Entity and relation metadata | graph node/edge attributes and vector metadata files |
| Medical extraction prompt | `prompts/entity_type/医学实体类型提示词.yml` |
| Medical ontology/aliases | `lightrag/medical_kg/ontology.py` |
| Medical hierarchy completion | `lightrag/medical_kg/hierarchy.py` |
| Extraction normalization hook | `lightrag/operate.py::extract_entities` |
| Graph medical grouping | `lightrag/medical_kg/graph_projection.py` and WebUI medical relation grouping code |

The agent should read these sources through stable parsers and structured file APIs where possible. It should not scrape large JSON as free text when a parser can preserve fields.

## Agent Roles

### 1. KG Snapshotter

Reads the current workspace and writes machine-readable snapshots. It is responsible for data collection, normalization of field names, and computing simple graph statistics.

Inputs:

- Workspace name and storage/input roots.
- GraphML graph.
- `kv_store_full_entities.json`.
- `kv_store_full_relations.json`.
- Optional API graph payloads for specific root labels.

Outputs:

- `snapshots/kg_snapshot.json`
- `snapshots/entity_stats.json`
- `snapshots/relation_stats.json`
- `snapshots/hierarchy_paths.json`
- `snapshots/source_coverage.json`

### 2. KB Memory Writer

Converts snapshots into layered Markdown memory. Its purpose is to let an LLM understand the current KB quickly without loading every raw graph row.

Outputs:

- `kb_context.md`
- `entity_catalog.md`
- `relation_catalog.md`
- `kg_structure.md`
- Optional split catalogs for large workspaces.

### 3. KG Quality Reviewer

Evaluates entity hygiene, relation semantics, evidence grounding, hierarchy completeness, graph readability, and iteration readiness. It combines deterministic checks with LLM review.

Outputs:

- `quality_report.md`
- `quality_score.json`
- Proposed issues that feed `improvement_backlog.md` and `approval_queue.md`.

### 4. Improvement Planner

Turns quality findings into actionable proposals. It does not mutate prompts, rules, or KG data. It prepares structured change suggestions for human review.

Outputs:

- `improvement_backlog.md`
- `proposals/*.yml` or `proposals/*.json`
- Updated `approval_queue.md`

### 5. Diff Engine

Compares two workspace snapshots or two iteration runs and reports improvements and regressions.

Outputs:

- `diff_report.md`
- `snapshots/diff_summary.json`

### 6. Rule Memory Manager

Maintains long-lived quality rules, known issues, accepted decisions, and rejected decisions so future review cycles do not start from zero.

Outputs:

- `quality_rules.md`
- `known_issues.md`
- `accepted_changes.md`
- `rejected_changes.md`
- Updates to `iteration_log.md`

## Artifact Layout

Artifacts live under `work/kb-iteration/<workspace>/`.

For the flu workspace:

```text
work/kb-iteration/influenza_medical_v1/
  kb_context.md
  entity_catalog.md
  relation_catalog.md
  kg_structure.md
  quality_report.md
  improvement_backlog.md
  approval_queue.md
  quality_rules.md
  known_issues.md
  accepted_changes.md
  rejected_changes.md
  diff_report.md
  iteration_log.md
  snapshots/
    kg_snapshot.json
    entity_stats.json
    relation_stats.json
    hierarchy_paths.json
    source_coverage.json
    quality_score.json
    diff_summary.json
  proposals/
    <proposal-id>.yml
```

## Markdown Artifacts

### `kb_context.md`

Compact LLM entrypoint. It should stay short enough to include in prompts.

Required sections:

- Workspace and run metadata.
- Active profile and document set.
- KG size: nodes, edges, document count, chunks if available.
- Core topics and root disease nodes.
- Entity-type distribution.
- Relation-type distribution.
- Current hierarchy summary.
- Latest quality score summary.
- Known critical issues.
- Links to deeper files.

This file is the first file a KB maintenance agent reads.

### `entity_catalog.md`

Typed entity index.

Required sections:

- Entity counts by type.
- Representative entities per type.
- Suspicious entities per type.
- Alias and canonicalization notes.
- Parent/category placement where known.
- Source coverage summaries.

It must avoid dumping embeddings, vector payload internals, and every low-value row. For large workspaces, split details into `entities/<type>.md`.

### `relation_catalog.md`

Relation/triple index.

Required sections:

- Relation counts by keyword.
- Representative triples per relation type.
- Illegal or generic relation labels.
- Direction-sensitive triples.
- Source/target type consistency.
- Missing description or missing evidence cases.

Main triple format:

```text
source - relation -> target
```

Direction must be preserved even if the Web graph renders visually undirected.

### `kg_structure.md`

Human-readable hierarchy and graph structure.

Required sections:

- Disease-centered navigation paths.
- Top-level categories and subgroups.
- Grouped relation categories.
- Missing branches.
- Overloaded branches.
- Factual edges versus navigation/category edges.

Example path:

```text
流行性感冒 -> 临床表现 -> 全身症状 -> 发热
```

### `quality_report.md`

Latest review output.

Required sections:

- Overall score and sub-scores.
- Critical blockers.
- Entity issues.
- Relation issues.
- Hierarchy issues.
- Evidence/source issues.
- Web readability issues.
- Regression risk.
- Recommended next actions.
- Items sent to approval queue.

Every actionable issue must include severity, evidence when available, expected impact, and suggested fix type.

### `improvement_backlog.md`

Action queue for non-immediate improvements.

Required fields per item:

- Backlog id.
- Issue summary.
- Fix category.
- Target file/data area.
- Evidence.
- Priority.
- Approval requirement.
- Expected metric change.
- Status.

### `approval_queue.md`

Review queue for mutation actions.

Actions requiring approval include:

- Prompt changes.
- Alias merges.
- Hierarchy additions or removals.
- Relation keyword remapping.
- Controlled category additions.
- Workspace deletion or rebuild.
- Any source-derived correction that changes the KG.

Each item must include:

- Proposal id.
- Proposed change.
- Reason.
- Evidence.
- Risk.
- Expected result.
- Rollback or rebuild notes.
- Required confirmation.

### `quality_rules.md`

Living project rulebook.

Rules must be separated into:

- Universal medical KG rules.
- LightRAG/project-specific rules.
- Current workspace/domain rules.
- Flu-specific rules.

Accepted rules should be injected into future reviewer prompts.

### `known_issues.md`

Failure-case library.

Initial issue classes:

- Value-like nodes such as `75 mg`, `每日2次`, `发病48小时内`, threshold values, page numbers, and table fragments.
- Synonym splits such as `流感` and `流行性感冒`.
- Generic relation labels such as `相关` or `邻接`.
- Disease hub overload where many leaf nodes connect directly to a disease center.
- Missing symptom, pathogen, diagnosis, treatment, prevention, or population category paths.
- Relation rows that lack source evidence or clear direction.

Rejected suggestions should be linked so the agent avoids repeating them.

### `accepted_changes.md` and `rejected_changes.md`

Decision history.

Accepted changes record what was approved, why, when, by whom, and what follow-up verification is needed.

Rejected changes record why a suggestion was declined, what evidence was insufficient, and whether the idea can be reconsidered later.

### `diff_report.md`

Workspace or run comparison report.

Required sections:

- Compared snapshots.
- Entity additions, removals, merges, and type changes.
- Relation additions, removals, keyword changes, and description changes.
- Hierarchy path changes.
- Quality score before and after.
- Regressions.
- Improvements.
- Release/rebuild recommendation.

## Machine Snapshots

### `kg_snapshot.json`

Canonical graph snapshot. Suggested top-level shape:

```json
{
  "workspace": "influenza_medical_v1",
  "generated_at": "2026-06-17T00:00:00+08:00",
  "source_files": [],
  "nodes": [],
  "edges": [],
  "metadata": {
    "profile": "clinical_guideline_zh",
    "graph_node_count": 0,
    "graph_edge_count": 0
  }
}
```

The implementation may add fields, but it should keep stable ids, labels, entity types, descriptions, source ids, file paths, relation keywords, and edge direction where available.

### `quality_score.json`

Machine-readable score output. Suggested shape:

```json
{
  "overall": 0,
  "subscores": {
    "entity_hygiene": 0,
    "relation_semantics": 0,
    "hierarchy_completeness": 0,
    "evidence_grounding": 0,
    "web_readability": 0,
    "iteration_readiness": 0
  },
  "metrics": {},
  "critical_blockers": []
}
```

## Quality Model

The quality reviewer should produce a score from 0 to 100. The score is not a medical truth score. It is a maintenance readiness score for this project's KG.

Recommended weights:

| Sub-score | Weight | Purpose |
| --- | ---: | --- |
| Entity hygiene | 20 | Stable concepts, alias merging, value-like noise removal |
| Relation semantics | 20 | Legal relation labels, direction, description support |
| Hierarchy completeness | 20 | Disease/category/subgroup/leaf organization |
| Evidence grounding | 20 | Source ids, file paths, chunk linkage, factual versus generated edge clarity |
| Web readability | 10 | Raw KG readability, medical grouping quality, relation-aware display |
| Iteration readiness | 10 | Diffability, approval queue, actionable backlog |

### Entity Hygiene Metrics

Track:

- Value-like node count.
- Known synonym duplicate count.
- Near-duplicate candidate count.
- Unknown or disallowed entity type count.
- Suspicious short fragment count.
- Entity types with unexpected distribution spikes.

Examples of value-like nodes:

- `75 mg`
- `每日2次`
- `5日疗程`
- `发病48小时内`
- `PaO2/FiO2≤300`
- page/table/footnote fragments.

### Relation Semantics Metrics

Track:

- Illegal relation keyword count.
- Generic relation count.
- Missing description count.
- Missing direction count.
- Source/target type mismatch count.
- Relations whose description does not support the triple.
- Duplicate relation pairs with conflicting keywords.

Generic labels include `相关`, `邻接`, and other labels that do not describe the medical relationship.

### Hierarchy Completeness Metrics

Track:

- Required top-level category coverage.
- Missing subgroup count.
- Leaf facts directly attached to disease when category paths exist.
- Isolated node count.
- Disease hub overload ratio.
- Category variant count outside controlled tables.

For the flu workspace, required first-level branches include:

- `病原体`
- `传播/流行病学`
- `临床表现`
- `并发症/重症`
- `诊断/检查`
- `治疗`
- `预防`
- `高危人群`
- `指南/证据来源`

### Evidence Grounding Metrics

Track:

- Missing `source_id` count.
- Missing `file_path` count.
- Missing chunk/source linkage count.
- Generated navigation edges without internal provenance marker.
- Factual claims that only appear in generated descriptions.
- Relations needing manual source review.

The reviewer must not recommend adding unstated medical facts. It may recommend organizing existing facts into navigation categories.

### Web Readability Metrics

Track:

- Whether the raw KG clearly presents factual nodes and relationships.
- Whether medical grouping reduces noise in the property panel.
- Whether property rows show relation labels instead of `邻接`.
- Whether relation details preserve full triple direction.

### Severity Levels

| Severity | Meaning | Required action |
| --- | --- | --- |
| Critical | Blocks acceptance of a rebuilt KB | Must resolve or explicitly waive before release |
| High | Likely harms retrieval, factuality, or Web understanding | Review before release |
| Medium | Quality issue suitable for backlog | Add to improvement backlog |
| Low | Informational or cleanup | Log only |

## Reviewer Prompt

The first reviewer prompt should be stored as a versioned artifact, for example `quality_reviewer_prompt.md`.

Draft prompt:

```text
你是医学知识图谱质量审查专家。你的任务不是补充医学常识，而是审查当前知识库是否忠实、清晰、可检索、适合人类浏览。

必须遵守：
1. 不得新增原文未支持的医学事实。
2. 可以建议新增导航/分类节点，但必须说明它们是为了组织已有事实。
3. 优先发现结构问题、实体类型问题、关系语义问题、证据缺失问题和 Web 可读性问题。
4. 对每个问题给出严重程度、证据、影响、建议修复方式和是否需要人工确认。
5. 建议分为：提示词优化、同义词归一、层级规则、关系规则、Web 展示、需要人工确认。

检查维度：
- 实体是否是稳定医学概念。
- 是否存在剂量、阈值、时间窗口、页码、表格碎片等值型噪声节点。
- 同义词、简称、全称是否被合并。
- 关系是否有明确医学语义，而不是“相关/邻接”。
- 关系方向和描述是否支持三元组。
- 每条事实关系是否有来源元数据。
- 疾病、症状、诊断、治疗、预防、人群、指南等层级是否适合人类浏览。
- 医学分组是否让原始 KG 更容易理解。
- 本轮建议是否适合自动报告、进入待办、进入审批队列或需要人工复核。

输出格式：
- 总体评分
- 子评分
- 关键问题列表
- 实体问题
- 关系问题
- 层级问题
- 证据/来源问题
- Web 可视化问题
- 结构化改进建议
- 下一轮重建建议
```

## Structured Proposal Format

Every proposal that may change behavior or data must have a machine-readable form.

Suggested YAML:

```yaml
id: proposal-20260617-001
type: alias_merge
target: lightrag/medical_kg/ontology.py
proposed_change:
  from: 流感
  to: 流行性感冒
reason: Known synonym remains split in the current workspace.
evidence:
  triples:
    - source: 流感
      relation: 临床表现
      target: 发热
  files:
    - data/rag_storage/influenza_medical_v1/graph_chunk_entity_relation.graphml
confidence: high
risk: low
requires_approval: true
expected_metric_change:
  synonym_duplicate_count: -1
status: pending_review
```

Proposal types:

- `alias_merge`
- `hierarchy_edge_add`
- `hierarchy_edge_remove`
- `relation_keyword_mapping`
- `prompt_change`
- `controlled_category_add`
- `web_display_change`
- `workspace_rebuild`
- `manual_review`

## Agent Control Loop

The agent's core behavior is a closed loop:

```text
Observe -> Think -> Propose -> Approve -> Execute -> Evaluate -> Remember -> Repeat
```

This loop is the primary contract for every KB iteration run. The detailed workflows below are concrete entry points into this loop, but the loop itself is the agent's operating model.

### Where the Loop Lives

The loop is not only a diagram. Each phase must be anchored in explicit files or modules so a future LLM, developer, or reviewer can inspect the run without guessing what happened.

- Observe lives in `snapshot.py`, `markdown.py`, `snapshots/kg_snapshot.json`, `kb_context.md`, `entity_catalog.md`, `relation_catalog.md`, and `kg_structure.md`.
- Think/diagnose lives in deterministic `quality.py`, the future `quality_reviewer_prompt.md`, `quality_score.json`, and `quality_report.md`. The agent should not persist private chain-of-thought; it should persist audit-grade diagnosis, evidence, assumptions, uncertainty, and recommended next actions.
- Propose lives in `proposals.py`, `approval_queue.md`, and `improvement_backlog.md`.
- Approve lives in human-reviewed `accepted_changes.md`, `rejected_changes.md`, and the approval entries in `iteration_log.md`.
- Execute is intentionally outside the automatic first version unless approval exists. Future execution adapters may edit prompts, rules, ontology files, Web display code, or rebuild workspaces, but each action must record changed files, commands, workspace names, and rollback notes in `iteration_log.md`.
- Evaluate lives in `diff.py`, refreshed snapshots, refreshed `quality_score.json`, `snapshots/diff_summary.json`, and `diff_report.md`.
- Remember/repeat lives in `quality_rules.md`, `known_issues.md`, accepted/rejected history, and the next refreshed `kb_context.md`.

The first deterministic runner should therefore implement `observe -> think -> propose -> pending approval` as its default path. If no approved mutation is present, the loop stops at `pending_user_review`. After the user approves a rebuild or rule change, the agent can run again with a `previous_snapshot` and enter the evaluate/remember portion of the loop.

### 1. Observe

The agent reads the current workspace and builds a reliable picture of the KB state.

Inputs:

- `.env` workspace configuration.
- Graph, entity, relation, and source metadata files.
- Existing `kb_context.md`, quality rules, known issues, accepted decisions, and rejected decisions.
- Optional previous workspace snapshots for comparison.

Outputs:

- Fresh machine snapshots.
- Refreshed Markdown memory.
- A run record in `iteration_log.md`.

### 2. Think

The agent analyzes the current state before proposing changes.

Thinking must combine deterministic checks and LLM review:

- Deterministic checks compute measurable issues such as value-like nodes, missing source metadata, illegal relation labels, and required branch coverage.
- LLM review judges higher-level structure, readability, ambiguity, and likely causes.
- The agent must state its diagnosis, evidence, assumptions, and uncertainty.

The output of this step is not a mutation. It is an analysis package that feeds quality scoring and proposal generation.

### 3. Propose

The agent converts diagnosed issues into structured improvement proposals.

Each proposal must identify:

- The problem.
- The proposed change.
- The target file, rule, prompt, workspace, or UI behavior.
- Supporting evidence.
- Expected metric improvement.
- Risk and confidence.
- Whether approval is required.

All behavior-changing proposals enter `approval_queue.md`. Lower-risk observations enter `improvement_backlog.md`.

### 4. Approve

Approval is the human gate between analysis and mutation.

The agent may continue reporting without approval, but it must not apply prompt edits, ontology edits, hierarchy changes, relation-rule changes, factual KG corrections, or workspace rebuilds until approval is recorded.

Approval results are written to:

- `accepted_changes.md`
- `rejected_changes.md`
- `iteration_log.md`

Rejected proposals become future negative memory so the agent does not keep re-asking for the same change.

### 5. Execute

Execution applies only approved changes.

Execution may include:

- Editing extraction prompts.
- Updating alias, relation, category, or hierarchy rules.
- Creating a new workspace.
- Rebuilding a workspace.
- Updating Web display behavior.

Execution must record the exact files changed, commands run, workspace names, and rebuild parameters. If the action fails, the failed attempt is recorded and the next attempt must use a different approach.

### 6. Evaluate

After execution, the agent immediately measures the result.

Evaluation includes:

- Regenerating snapshots.
- Recomputing quality scores.
- Comparing before/after metrics.
- Writing `diff_report.md`.
- Detecting regressions.
- Deciding whether the iteration should be accepted, revised, or rolled back.

No iteration is considered successful only because changes were applied. It is successful only if quality metrics and regression checks support that conclusion or the user explicitly accepts the trade-off.

### 7. Remember

The agent updates long-term memory.

Memory updates include:

- New accepted rules in `quality_rules.md`.
- New failure cases in `known_issues.md`.
- Accepted and rejected proposals.
- Run summaries in `iteration_log.md`.
- Lessons that should affect future reviewer prompts.

This makes each loop more informed than the previous one.

### Loop State Record

Each loop should produce a compact state record:

```yaml
loop_id: kb-loop-20260617-001
workspace: influenza_medical_v1
phase: evaluate
input_snapshot: snapshots/kg_snapshot.json
previous_snapshot: null
analysis: quality_report.md
proposals:
  - proposals/proposal-20260617-001.yml
approved_changes: []
executed_changes: []
quality_before: null
quality_after: snapshots/quality_score.json
decision: pending_user_review
```

This state record can live inside `iteration_log.md` or as `snapshots/loop_state.json`.

### Loop Stop Conditions

The loop stops when one of these conditions is met:

- Quality score meets the accepted threshold and no critical blockers remain.
- The agent has produced approval-required proposals and is waiting for user review.
- A rebuild or mutation requires explicit user confirmation.
- A critical error prevents reliable snapshot or score generation.
- The user intentionally pauses the iteration.

### Loop Invariants

- Observation and evaluation use structured data, not only free-text summaries.
- Thinking produces evidence-backed diagnosis, not silent assumptions.
- Proposals are structured and reviewable.
- Execution is confirmation-gated.
- Evaluation happens after every execution.
- Memory is updated after every accepted or rejected decision.

## Workflow

### Run After Ingestion

1. Read `.env` and resolve workspace paths.
2. Read graph/entity/relation storage files.
3. Generate snapshots.
4. Generate or refresh Markdown memory.
5. Run deterministic quality checks.
6. Run LLM quality review with compact context and selected evidence.
7. Write `quality_report.md`.
8. Write `improvement_backlog.md`.
9. Write approval-required proposals to `approval_queue.md`.
10. Append run metadata to `iteration_log.md`.

### Run Before Rebuild

1. Review `approval_queue.md`.
2. Apply only approved prompt/rule changes.
3. Record approvals in `accepted_changes.md`.
4. Record rejected items in `rejected_changes.md`.
5. Create a new workspace name if rebuilding.
6. Rebuild only after explicit user confirmation.

### Run After Rebuild

1. Generate a new snapshot.
2. Compare previous and new snapshots.
3. Write `diff_report.md`.
4. Update `quality_report.md`.
5. Flag regressions.
6. Recommend accept, revise, or rollback.

## Safety and Medical Boundaries

Medical facts must remain source-grounded. The agent should treat LLM output as analysis and proposal, not as authority.

Required safeguards:

- Separate factual edges from navigation/category edges.
- Preserve source metadata for factual findings.
- Do not use common medical knowledge to add facts absent from imported documents.
- Do not silently mutate prompts, rules, or workspaces.
- Keep accepted and rejected decisions persistent.
- Make every high-impact recommendation reviewable.

If evidence is missing, the agent should recommend source-link repair, re-extraction, or manual review rather than accepting the triple as reliable.

## Domain Extension Strategy

The design must remain useful when new medical documents are added.

Rules:

- General rules belong in `quality_rules.md` under universal medical KG rules.
- LightRAG-specific behavior belongs under project rules.
- Current document-set rules belong under workspace/domain rules.
- Flu-only aliases and hierarchy seeds must be labeled flu-specific.
- New categories must map into controlled extension categories or enter approval queue.

Controlled extension categories should include:

| Key | Label | Examples |
| --- | --- | --- |
| `differential_diagnosis` | `鉴别诊断` | Similar diseases, differential criteria |
| `nursing_care` | `护理` | Care, home care, observation |
| `follow_up` | `随访` | Revisit, follow-up observation |
| `rehabilitation` | `康复` | Recovery-stage management |
| `contraindication` | `禁忌证` | Drug contraindications, unsuitable groups |
| `adverse_reaction` | `不良反应` | Side effects, adverse events |
| `public_health` | `公共卫生处置` | Reporting, isolation management, school controls |

## Error Handling

The agent must log and degrade gracefully:

- Missing graph file: write an error in `quality_report.md` and stop snapshot-dependent checks.
- Empty workspace: generate an empty snapshot and mark iteration readiness as low.
- Malformed JSON or GraphML: record parser error and do not overwrite the previous good snapshot.
- Missing source metadata: continue analysis but flag evidence coverage issues.
- LLM review failure: keep deterministic checks and mark LLM review incomplete.
- Oversized Markdown catalogs: split by entity type or relation group.

## Testing Expectations

Implementation planning should include tests for:

- Snapshot generation from a small fixture graph.
- Entity catalog grouping by type.
- Relation catalog grouping by keyword and direction.
- Value-like node detection.
- Generic relation detection.
- Missing source metadata detection.
- Quality score calculation.
- Approval queue item formatting.
- Structured proposal schema validation.
- Diff report generation between two fixture snapshots.
- Markdown size controls and split-catalog behavior.

Medical-flu fixture tests should include:

- `流感` and `流行性感冒` synonym duplication.
- `75 mg` and `发病48小时内` as value-like node examples.
- A generic `邻接` relation example.
- A missing `临床表现 -> 全身症状 -> 发热` path example.
- A disease hub overload example.

## Acceptance Criteria

The design is complete when:

- The file layout and artifact responsibilities are defined.
- The semi-automatic safety boundary is explicit.
- The quality scoring model is defined with metrics and severity levels.
- Evidence binding and source-grounding rules are defined.
- The approval workflow is first-class and mutation-gated.
- The diff workflow can compare pre-rebuild and post-rebuild workspaces.
- Persistent rule and failure memory are included.
- Domain extension beyond influenza is addressed.
- Structured proposal format is defined.
- Testing expectations are specific enough to drive implementation planning.

## Implementation Phases

Recommended implementation order:

1. Snapshot generation and deterministic statistics.
2. Markdown memory generation.
3. Deterministic quality checks and score JSON.
4. LLM reviewer prompt and report generation.
5. Improvement backlog and approval queue.
6. Structured proposal files and schema validation.
7. Snapshot diff and regression report.
8. Documentation and tests.

This sequence keeps the first implementation useful even before LLM review is wired in.

## Implemented Deterministic Workflow

Task 7 documents the deterministic modules now available under `lightrag/kb_iteration/` and the operator workflow in `docs/KBIterationAgent.md`.

Current implemented modules:

- `snapshot.py` builds `KGSnapshot` objects from GraphML and writes snapshot artifacts with `write_snapshot_artifacts`.
- `markdown.py` writes LLM-readable memory files with `write_markdown_memory`.
- `quality.py` computes deterministic quality metrics with `evaluate_snapshot_quality` and writes quality artifacts with `write_quality_artifacts`.
- `proposals.py` validates approval-gated `ImprovementProposal` objects and writes the approval queue plus backlog.
- `diff.py` compares snapshots with `compare_snapshots` and writes `diff_report.md` plus `snapshots/diff_summary.json`.
- `runner.py` exposes `run_iteration`, the first deterministic end-to-end runner.

The first runner is intentionally non-mutating. It validates workspace names before path joins, reads the existing workspace graph, writes snapshot, Markdown, quality, and iteration-log artifacts, then records `pending_user_review`. It does not apply prompt changes, rule changes, ontology edits, fact corrections, WebUI changes, or workspace rebuilds.
