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
    expect(markup).toContain('生成的 proposal')
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
    for (const label of ['Explain', 'Infer', 'Evidence', 'Propose', 'Rank', 'Judge']) {
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
    expect(markup).toContain('proposal 来源')
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
