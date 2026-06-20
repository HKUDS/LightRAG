import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type { KBIterationSummaryResponse } from '@/api/lightrag'
import {
  DecisionExecutionPanel,
  DecisionMemoryPanel,
  IterationOverviewPanel,
  JsonArtifactPanel,
  QualityScoreJsonPanel,
  SnapshotReviewPanel
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
        { key: 'agent_memory_summary', contentType: 'text/markdown', exists: true },
        { key: 'accepted_changes_apply_result', contentType: 'text/markdown', exists: true },
        { key: 'accepted_changes_execution', contentType: 'text/markdown', exists: true },
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
    expect(markup).toContain('待审批 Proposal')
    expect(markup).toContain('approval_queue.md')
    expect(markup).toContain('改进 backlog')
    expect(markup).toContain('improvement_backlog.md')
    expect(markup).toContain('已接受变更记忆')
    expect(markup).toContain('accepted_changes.md')
    expect(markup).toContain('已拒绝变更记忆')
    expect(markup).toContain('rejected_changes.md')
    expect(markup).toContain('Agent 压缩记忆')
    expect(markup).toContain('agent_memory_summary.md')
    expect(markup).toContain('真实应用结果')
    expect(markup).toContain('accepted_changes_apply_result.md')
    expect(markup).toContain('accepted_changes_execution.md')
    expect(markup).toContain('当前阶段')
    expect(markup).toContain('iteration_log.md')
    expect(markup).toContain('已生成')
    expect(markup).toContain('缺失')
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

  test('json artifact treats an empty string payload as empty', () => {
    const markup = renderToStaticMarkup(
      <JsonArtifactPanel
        title="质量分数"
        fileName="snapshots/quality_score.json"
        payload=""
        summaryRows={[['Overall', '—']]}
        emptyText="暂无质量分数。"
      />
    )

    expect(markup).toContain('暂无质量分数。')
    expect(markup).not.toContain('&quot;&quot;')
  })

  test('snapshot review summarizes snapshot counts and renders searchable tables without graph markup', () => {
    const markup = renderToStaticMarkup(
      <SnapshotReviewPanel
        snapshot={{
          workspace: 'medical-kb',
          snapshot_id: 'snap-2',
          nodes: [
            { id: 'n1', label: '高血压', entity_type: '疾病', source_id: '', file_path: 'a.md' },
            { id: 'n2', label: '头痛', entity_type: '症状', source_id: 'src-1', file_path: 'b.md' }
          ],
          edges: [{ id: 'e1', source: 'n1', target: 'n2', keywords: '症状', file_path: '' }]
        }}
      />
    )

    expect(markup).toContain('图谱快照')
    expect(markup).toContain('节点数')
    expect(markup).toContain('关系数')
    expect(markup).toContain('medical-kb')
    expect(markup).toContain('snap-2')
    expect(markup).toContain('节点')
    expect(markup).toContain('关系')
    expect(markup).toContain('证据问题')
    expect(markup).toContain('placeholder="搜索"')
    expect(markup).toContain('高血压')
    expect(markup).not.toContain('&quot;snapshot_id&quot;: &quot;snap-2&quot;')
    expect(markup).not.toContain('<svg')
    expect(markup).not.toContain('Medical knowledge graph hierarchy')
  })

  test('quality score json summarizes overall findings and blockers', () => {
    const markup = renderToStaticMarkup(
      <QualityScoreJsonPanel
        qualityScore={{
          overall: 76,
          findings: [
            { severity: 'high', message: '缺少溯源' },
            { severity: 'medium', message: '命名不一致' }
          ],
          critical_blockers: ['规则冲突']
        }}
      />
    )

    expect(markup).toContain('质量分数')
    expect(markup).toContain('Overall')
    expect(markup).toContain('76')
    expect(markup).toContain('Findings')
    expect(markup).toContain('2')
    expect(markup).toContain('Critical blockers')
    expect(markup).toContain('1')
    expect(markup).toContain('&quot;overall&quot;: 76')
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

  test('decision execution renders real apply result before historical execution report', () => {
    const markup = renderToStaticMarkup(
      <DecisionExecutionPanel
        improvementBacklog="backlog content marker"
        acceptedChanges="## proposal-1"
        rejectedChanges="rejected content marker"
        acceptedApplyResult="apply result content marker"
        acceptedExecution="execution report content marker"
        executing={false}
        onExecuteAcceptedChanges={() => undefined}
      />
    )

    expect(markup).toContain('真实应用结果')
    expect(markup).toContain('accepted_changes_apply_result.md')
    expect(markup).toContain('apply result content marker')
    expect(markup).toContain('执行报告')
    expect(markup).toContain('accepted_changes_execution.md')
    expect(markup).toContain('execution report content marker')
    expect(markup.indexOf('accepted_changes_apply_result.md')).toBeLessThan(
      markup.indexOf('accepted_changes_execution.md')
    )
  })
})
