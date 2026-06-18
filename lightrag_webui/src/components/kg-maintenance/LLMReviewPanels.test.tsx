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
        running={false}
        onRun={() => undefined}
      />
    )

    expect(markup).toContain('LLM 审阅')
    expect(markup).toContain('pending_human_review')
    expect(markup).toContain('generic_relation')
    expect(markup).toContain('# LLM Review Report')
  })

  test('renders patch candidates with proposal ids and selected patch', () => {
    const markup = renderToStaticMarkup(
      <PatchCandidatesPanel
        proposals={`proposals:\n  - id: p1\n    title: candidate one`}
        patchText={`--- a/file\n+++ b/file\n@@\n-old\n+new`}
        onLoadPatch={() => undefined}
      />
    )

    expect(markup).toContain('候选 Patch')
    expect(markup).toContain('--- a/file')
    expect(markup).toContain('p1')
  })

  test('renders judge report content', () => {
    const markup = renderToStaticMarkup(
      <LLMJudgePanel report={`# Judge Report\n\nstatus: needs_human`} />
    )

    expect(markup).toContain('Judge 评判')
    expect(markup).toContain('needs_human')
  })
})
