import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { LLMJudgePanel, LLMReviewPanel, PatchCandidatesPanel } from './LLMReviewPanels'

describe('LLM review panels', () => {
  test('renders LLM review trace focus and generated artifacts', () => {
    const markup = renderToStaticMarkup(
      <LLMReviewPanel
        trace={{
          stop_reason: 'pending_human_review',
          rounds: [
            {
              round_id: 'r1',
              focus: ['generic_relation'],
              proposal_ids: ['p1'],
              state: 'reviewing'
            }
          ]
        }}
        report="# LLM Review Report"
        proposals="id: p1"
        issueAnalysis=""
        missingBranchInference=""
        evidenceMap=""
        repairPlan=""
        running={false}
        onRun={() => undefined}
      />
    )

    expect(markup).toContain('LLM 审阅材料')
    expect(markup).toContain('不会自动修改 KG')
    expect(markup).toContain('停止原因')
    expect(markup).toContain('pending_human_review')
    expect(markup).toContain('generic_relation')
    expect(markup).toContain('生成的提案')
    expect(markup).toContain('# LLM Review Report')
  })

  test('renders multistage agent stages, artifacts, and safety notice', () => {
    const markup = renderToStaticMarkup(
      <LLMReviewPanel
        trace={{
          stages: [
            { stage: 'explain', state: 'done', artifact_keys: ['llm_issue_analysis'] },
            {
              stage: 'infer_branches',
              state: 'done',
              artifact_keys: ['llm_missing_branch_inference']
            },
            {
              stage: 'locate_evidence',
              state: 'done',
              artifact_keys: ['llm_evidence_map']
            },
            { stage: 'propose', state: 'done', proposal_ids: ['p1', 'p2'] },
            {
              stage: 'rank_repairs',
              state: 'done',
              artifact_keys: ['llm_repair_plan'],
              proposal_ids: ['p2']
            },
            { stage: 'judge', state: 'needs_human' }
          ]
        }}
        report=""
        proposals=""
        issueAnalysis="## Issue\nGeneric relation hides clinical specificity."
        missingBranchInference="## Missing\nAdd influenza treatment branch."
        evidenceMap="## Evidence\nsource: guideline.md"
        repairPlan="## Ranking\n1. p2"
        running={false}
        onRun={() => undefined}
      />
    )

    expect(markup).toContain('多阶段 LLM Agent')
    for (const label of ['问题解释', '缺失分支', '证据定位', '生成提案', '修复排序', '评判复核']) {
      expect(markup).toContain(label)
    }
    expect(markup).toContain('llm_missing_branch_inference')
    expect(markup).toContain('llm_evidence_map')
    expect(markup).toContain('llm_repair_plan')
    expect(markup).toContain('p1')
    expect(markup).toContain('p2')
    expect(markup).toContain('## Issue')
    expect(markup).toContain('## Missing')
    expect(markup).toContain('## Evidence')
    expect(markup).toContain('## Ranking')
    expect(markup).toContain('问题解释')
    expect(markup).toContain('缺失分支推断')
    expect(markup).toContain('证据定位')
    expect(markup).toContain('修复方案排序')
    expect(markup).toContain('人工批准')
    expect(markup).toContain('不会自动修改 KG')
  })

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

    expect(markup).toContain('尝试次数')
    expect(markup).toContain('3')
    expect(markup).toContain('proposal expected_metric_change values must be numbers')
    expect(markup).toContain('proposal evidence is not grounded in deterministic artifacts')
    expect(markup).not.toContain('[object Object]')
  })

  test('renders malformed multistage traces without object garbage', () => {
    const markup = renderToStaticMarkup(
      <LLMReviewPanel
        trace={{
          stages: [
            null,
            { stage: 42 },
            {
              stage: 'explain',
              state: { bad: true },
              artifact_keys: [1, 'llm_issue_analysis'],
              proposal_ids: ['p1', { bad: true }]
            }
          ]
        }}
        report=""
        proposals=""
        issueAnalysis=""
        missingBranchInference=""
        evidenceMap=""
        repairPlan=""
        running={false}
        onRun={() => undefined}
      />
    )

    expect(markup).toContain('llm_issue_analysis')
    expect(markup).toContain('p1')
    expect(markup).not.toContain('[object Object]')
  })

  test('renders deterministic proposal funnel from the source JSON report', () => {
    const markup = renderToStaticMarkup(
      <LLMReviewPanel
        trace={null}
        report=""
        proposals=""
        issueAnalysis=""
        missingBranchInference=""
        evidenceMap=""
        repairPlan=""
        deterministicProposalReport={{
          families: {
            diagnosis: {
              raw_issue_count: 12,
              issue_with_candidate_count: 8,
              action_candidate_count: 9,
              deterministic_covered_count: 5,
              llm_residual_count: 3,
              blocked_safety_count: 1,
              blocked_apply_count: 2,
              blocked_evidence_count: 4,
              deferred_budget_count: 2,
              selected_proposal_count: 5,
              reason_code_counts: {
                SAME_EDGE_CONFLICT: 2,
                FAMILY_CAP_REACHED: 1
              }
            },
            entity_cleanup: {
              raw_issue_count: 1,
              issue_with_candidate_count: 0,
              action_candidate_count: 0,
              deterministic_covered_count: 0,
              llm_residual_count: 1,
              blocked_safety_count: 0,
              blocked_apply_count: 0,
              blocked_evidence_count: 0,
              deferred_budget_count: 0,
              selected_proposal_count: 0,
              reason_code_counts: {}
            }
          }
        }}
        running={false}
        onRun={() => undefined}
      />
    )

    for (const heading of [
      '确定性提案漏斗',
      '家族',
      '原始问题',
      '有候选',
      '候选动作',
      '确定性覆盖',
      'LLM 剩余',
      '阻塞',
      '延后',
      '已选提案',
      '主要原因'
    ]) {
      expect(markup).toContain(heading)
    }
    expect(markup).toContain('诊断')
    expect(markup).toContain('实体清理')
    expect(markup).toContain('SAME_EDGE_CONFLICT')
    expect(markup).toContain('暂无')
    expect(markup).toContain('<td class="px-3 py-2 text-right">7</td>')
  })

  test('renders keyed proposal funnel visibility with Chinese readonly sections', () => {
    const markup = renderToStaticMarkup(
      <LLMReviewPanel
        trace={null}
        report=""
        proposals=""
        issueAnalysis=""
        missingBranchInference=""
        evidenceMap=""
        repairPlan=""
        deterministicProposalReport={{
          summary: {
            issue_accounting_rate: 1,
            candidate_validation_rate: 0.75,
            candidate_to_proposal_rate: 0.5,
            queue_apply_support_rate: 1,
            hard_rejection_recurrence_count: 3,
            exact_duplicate_recurrence_count: 2,
            known_bad_pattern_count: 1
          },
          families: {
            treatment: {
              raw_issue_count: 4,
              deterministic_candidate_issue_count: 1,
              action_candidate_count: 3,
              schema_blocked_count: 1,
              safety_blocked_count: 2,
              evidence_blocked_count: 3,
              apply_blocked_count: 4,
              decision_memory_blocked_count: 5,
              conflict_count: 6,
              deferred_by_family_cap_count: 7,
              deterministic_proposal_count: 8,
              llm_residual_eligible_count: 9,
              llm_residual_selected_count: 10,
              valid_llm_proposal_count: 11,
              conversion_failure_count: 12,
              merge_drop_count: 13,
              selected_approval_proposal_count: 14
            }
          },
          conflict_groups: [
            {
              target: 'edge-treatment',
              proposal_ids: ['p1', 'p2'],
              reason: 'same edge has conflicting predicates'
            }
          ]
        }}
        running={false}
        onRun={() => undefined}
      />
    )

    for (const label of [
      '按家族漏斗统计',
      '阻断原因',
      'LLM 剩余已选/延后',
      '冲突组',
      '复现的拒绝记忆命中数',
      '候选校验率',
      '队列应用支持率'
    ]) {
      expect(markup).toContain(label)
    }
    expect(markup).toContain('治疗')
    expect(markup).toContain('edge-treatment')
    expect(markup).toContain('p1')
    expect(markup).toContain('3')
    expect(markup).not.toContain('raw_issue_count')
    expect(markup).not.toContain('schema_blocked_count')
  })

  test('renders proposal funnel when families are an array', () => {
    const markup = renderToStaticMarkup(
      <LLMReviewPanel
        trace={null}
        report=""
        proposals=""
        issueAnalysis=""
        missingBranchInference=""
        evidenceMap=""
        repairPlan=""
        deterministicProposalReport={{
          summary: {
            hard_rejection_recurrence_count: 1,
            exact_duplicate_recurrence_count: 0,
            known_bad_pattern_count: 0
          },
          families: [
            {
              family: 'diagnosis',
              raw_issue_count: 2,
              llm_residual_eligible_count: 5,
              llm_residual_selected_count: 3,
              deferred_by_family_cap_count: 2,
              selected_approval_proposal_count: 1
            }
          ],
          conflicts: [
            {
              target: 'edge-diagnosis',
              proposal_ids: ['p-diagnosis'],
              reason: 'conflicting repairs'
            }
          ]
        }}
        running={false}
        onRun={() => undefined}
      />
    )

    expect(markup).toContain('诊断')
    expect(markup).toContain('LLM 剩余已选/延后')
    expect(markup).toContain('edge-diagnosis')
    expect(markup).toContain('p-diagnosis')
    expect(markup).toContain('复现的拒绝记忆命中数')
  })

  test('renders patch candidates with proposal ids and selected patch', () => {
    const markup = renderToStaticMarkup(
      <PatchCandidatesPanel
        proposals={'proposals:\n  - id: p1\n    title: candidate one'}
        patchText={'--- a/file\n+++ b/file\n@@\n-old\n+new'}
        onLoadPatch={() => undefined}
      />
    )

    expect(markup).toContain('候选 Patch')
    expect(markup).toContain('已选择 Patch')
    expect(markup).toContain('提案来源')
    expect(markup).toContain('--- a/file')
    expect(markup).toContain('p1')
  })

  test('renders judge report content', () => {
    const markup = renderToStaticMarkup(
      <LLMJudgePanel report={'# Judge Report\n\nstatus: needs_human'} />
    )

    expect(markup).toContain('Judge 评判')
    expect(markup).toContain('人工复核状态')
    expect(markup).toContain('needs_human')
  })
})
