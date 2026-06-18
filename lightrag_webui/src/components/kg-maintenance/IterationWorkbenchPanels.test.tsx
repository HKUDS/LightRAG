import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { KBIterationSummaryResponse } from '@/api/lightrag'
import {
  DecisionMemoryPanel,
  IterationOverviewPanel,
  JsonArtifactPanel
} from './IterationWorkbenchPanels'

describe('iteration workbench panels', () => {
  test('overview renders all required artifact labels and file names', () => {
    const summary: KBIterationSummaryResponse = {
      workspace: 'medical-kb',
      latestRunId: 'run-1',
      phase: 'pending_user_review',
      counts: {
        nodes: 12,
        edges: 18,
        sources: 4
      },
      quality: {
        overall: 82,
        findings: [],
        critical_blockers: []
      },
      pendingApprovalCount: 3,
      highRiskFindingCount: 1,
      artifacts: [
        { key: 'kb_context', contentType: 'text/markdown', exists: true },
        { key: 'quality_report', contentType: 'text/markdown', exists: true },
        { key: 'kg_snapshot', contentType: 'application/json', exists: true },
        { key: 'quality_score', contentType: 'application/json', exists: false },
        { key: 'approval_queue', contentType: 'text/markdown', exists: true },
        { key: 'improvement_backlog', contentType: 'text/markdown', exists: true },
        { key: 'accepted_changes', contentType: 'text/markdown', exists: true },
        { key: 'rejected_changes', contentType: 'text/markdown', exists: true },
        { key: 'iteration_log', contentType: 'text/markdown', exists: true }
      ]
    }

    const markup = renderToStaticMarkup(
      <IterationOverviewPanel
        summary={summary}
        loading={false}
        onOpenSection={() => undefined}
      />
    )

    expect(markup).toContain('当前 KB 摘要')
    expect(markup).toContain('kb_context.md')
    expect(markup).toContain('质量报告')
    expect(markup).toContain('quality_report.md')
    expect(markup).toContain('图谱快照')
    expect(markup).toContain('snapshots/kg_snapshot.json')
    expect(markup).toContain('质量分数')
    expect(markup).toContain('snapshots/quality_score.json')
    expect(markup).toContain('待审批 proposal')
    expect(markup).toContain('approval_queue.md')
    expect(markup).toContain('改进 backlog')
    expect(markup).toContain('improvement_backlog.md')
    expect(markup).toContain('已接受变更记忆')
    expect(markup).toContain('accepted_changes.md')
    expect(markup).toContain('已拒绝变更记忆')
    expect(markup).toContain('rejected_changes.md')
    expect(markup).toContain('当前阶段')
    expect(markup).toContain('iteration_log.md')
  })

  test('json artifact renders title file summary and raw json without graph markup', () => {
    const payload = {
      workspace: 'medical-kb',
      snapshotId: 'snap-1',
      nodes: [{ id: 'n1' }],
      edges: [{ id: 'e1' }]
    }

    const markup = renderToStaticMarkup(
      <JsonArtifactPanel
        title="图谱快照"
        fileName="snapshots/kg_snapshot.json"
        payload={payload}
        summaryRows={[
          ['节点数', '1'],
          ['关系数', '1'],
          ['Workspace', 'medical-kb']
        ]}
        emptyText="暂无图谱快照。"
      />
    )

    expect(markup).toContain('图谱快照')
    expect(markup).toContain('snapshots/kg_snapshot.json')
    expect(markup).toContain('节点数')
    expect(markup).toContain('medical-kb')
    expect(markup).toContain('&quot;snapshotId&quot;: &quot;snap-1&quot;')
    expect(markup).not.toContain('<svg')
    expect(markup).not.toContain('Medical knowledge graph hierarchy')
  })

  test('decision memory renders accepted and rejected artifacts', () => {
    const markup = renderToStaticMarkup(
      <DecisionMemoryPanel
        acceptedChanges="- 保留稳定的实体命名规则"
        rejectedChanges="- 拒绝未溯源的关系改写"
      />
    )

    expect(markup).toContain('已接受变更记忆')
    expect(markup).toContain('accepted_changes.md')
    expect(markup).toContain('保留稳定的实体命名规则')
    expect(markup).toContain('已拒绝变更记忆')
    expect(markup).toContain('rejected_changes.md')
    expect(markup).toContain('拒绝未溯源的关系改写')
  })
})
