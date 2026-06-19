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
            { stage: 'explain', state: 'done' },
            { stage: 'infer', state: 'done' },
            { stage: 'evidence', state: 'done', artifact_keys: ['evidence_map'] },
            { stage: 'propose', state: 'done', proposal_ids: ['p1', 'p2'] },
            { stage: 'rank', state: 'done', proposal_ids: ['p2'] },
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
    expect(markup).toContain('p1')
    expect(markup).toContain('p2')
    expect(markup).toContain('## Issue')
    expect(markup).toContain('## Missing')
    expect(markup).toContain('## Evidence')
    expect(markup).toContain('## Ranking')
    expect(markup).toContain('人工批准')
    expect(markup).toContain('不会自动修改 KG')
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
