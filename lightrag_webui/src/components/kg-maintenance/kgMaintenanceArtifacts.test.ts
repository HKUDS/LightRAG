import { describe, expect, test } from 'bun:test'
import {
  KG_MAINTENANCE_ARTIFACTS,
  WORKFLOW_STEPS,
  artifactsForStep,
  findArtifactDefinition
} from './kgMaintenanceArtifacts'

describe('kg maintenance artifact catalog', () => {
  test('defines exactly five workflow step ids in display order', () => {
    expect(WORKFLOW_STEPS).toEqual(['check', 'llm-review', 'approval', 'execute', 'validate'])
  })

  test('keeps core source and zh filenames with Chinese titles', () => {
    expect(findArtifactDefinition('kb_context')).toMatchObject({
      key: 'kb_context',
      title: '当前 KB 摘要',
      sourceFile: 'kb_context.md',
      zhFile: 'kb_context.zh.md',
      step: 'check'
    })
    expect(findArtifactDefinition('quality_score')).toMatchObject({
      key: 'quality_score',
      title: '质量分数',
      sourceFile: 'snapshots/quality_score.json',
      zhFile: 'snapshots/quality_score.zh.json',
      step: 'check'
    })
    expect(findArtifactDefinition('approval_queue')).toMatchObject({
      key: 'approval_queue',
      title: '待审批 Proposal',
      sourceFile: 'approval_queue.md',
      zhFile: 'approval_queue.zh.md',
      step: 'approval'
    })
    expect(findArtifactDefinition('agent_memory_summary')).toMatchObject({
      key: 'agent_memory_summary',
      title: 'Agent 压缩记忆',
      sourceFile: 'agent_memory_summary.md',
      zhFile: 'agent_memory_summary.zh.md',
      step: 'approval'
    })
  })

  test('uses Chinese labels for schema migration workflow artifacts', () => {
    expect(findArtifactDefinition('quality_score')?.title).toBe('质量分数')
    expect(findArtifactDefinition('approval_queue')?.title).toBe('待审批 Proposal')
    expect(findArtifactDefinition('agent_memory_summary')?.title).toBe('Agent 压缩记忆')
    expect(findArtifactDefinition('accepted_changes_apply_result')?.title).toBe('真实应用结果')
  })

  test('maps every visible artifact to one workflow step', () => {
    const stepIds = new Set(WORKFLOW_STEPS)
    const keys = new Set<string>()

    for (const artifact of KG_MAINTENANCE_ARTIFACTS) {
      expect(stepIds.has(artifact.step)).toBe(true)
      expect(keys.has(artifact.key)).toBe(false)
      keys.add(artifact.key)
      expect(artifact.sourceFile.length).toBeGreaterThan(0)
      expect(artifact.zhFile.length).toBeGreaterThan(0)
      expect(artifact.title.length).toBeGreaterThan(0)
    }
  })

  test('places proposal revision requests in the approval step', () => {
    expect(findArtifactDefinition('proposal_revision_requests')).toMatchObject({
      key: 'proposal_revision_requests',
      title: 'Proposal 修订请求',
      sourceFile: 'proposal_revision_requests.md',
      zhFile: 'proposal_revision_requests.zh.md',
      step: 'approval'
    })
    expect(artifactsForStep('approval').map((artifact) => artifact.key)).toContain(
      'proposal_revision_requests'
    )
    expect(artifactsForStep('approval').map((artifact) => artifact.key)).toContain(
      'agent_memory_summary'
    )
  })

  test('includes backend-supported catalog, structure, rules, and coverage artifacts', () => {
    expect(findArtifactDefinition('entity_catalog')).toMatchObject({
      key: 'entity_catalog',
      title: '实体目录',
      sourceFile: 'entity_catalog.md',
      zhFile: 'entity_catalog.zh.md',
      step: 'check'
    })
    expect(findArtifactDefinition('relation_catalog')).toMatchObject({
      key: 'relation_catalog',
      title: '关系目录',
      sourceFile: 'relation_catalog.md',
      zhFile: 'relation_catalog.zh.md',
      step: 'check'
    })
    expect(findArtifactDefinition('kg_structure')).toMatchObject({
      key: 'kg_structure',
      title: '图谱结构',
      sourceFile: 'kg_structure.md',
      zhFile: 'kg_structure.zh.md',
      step: 'check'
    })
    expect(findArtifactDefinition('quality_rules')).toMatchObject({
      key: 'quality_rules',
      title: '质量规则',
      sourceFile: 'quality_rules.md',
      zhFile: 'quality_rules.zh.md',
      step: 'validate'
    })
    expect(findArtifactDefinition('known_issues')).toMatchObject({
      key: 'known_issues',
      title: '已知问题',
      sourceFile: 'known_issues.md',
      zhFile: 'known_issues.zh.md',
      step: 'validate'
    })
    expect(findArtifactDefinition('source_coverage')).toMatchObject({
      key: 'source_coverage',
      title: '来源覆盖',
      sourceFile: 'snapshots/source_coverage.json',
      zhFile: 'snapshots/source_coverage.zh.json',
      step: 'check'
    })
  })

  test('returns LLM artifacts for the llm-review step', () => {
    expect(artifactsForStep('llm-review').map((artifact) => artifact.key)).toEqual(
      expect.arrayContaining([
        'llm_issue_analysis',
        'llm_review_trace',
        'llm_missing_branch_inference',
        'llm_evidence_map',
        'llm_repair_plan',
        'llm_judge_report'
      ])
    )
  })

  test('includes orchestrated proposal artifacts in the llm-review step', () => {
    expect(findArtifactDefinition('proposal_task_packs')).toMatchObject({
      key: 'proposal_task_packs',
      title: 'Proposal 任务包',
      sourceFile: 'proposal_task_packs.json',
      zhFile: 'proposal_task_packs.zh.json',
      step: 'llm-review',
      contentType: 'application/json'
    })
    expect(findArtifactDefinition('proposal_merge_report')).toMatchObject({
      key: 'proposal_merge_report',
      title: 'Proposal 合并报告',
      sourceFile: 'proposal_merge_report.md',
      zhFile: 'proposal_merge_report.zh.md',
      step: 'llm-review',
      contentType: 'text/markdown'
    })
    expect(findArtifactDefinition('subagent_output_index')).toMatchObject({
      key: 'subagent_output_index',
      title: '子 Agent 输出索引',
      sourceFile: 'subagent_outputs/index.json',
      zhFile: 'subagent_outputs/index.zh.json',
      step: 'llm-review',
      contentType: 'application/json'
    })
    expect(findArtifactDefinition('issue_ledger')).toMatchObject({
      key: 'issue_ledger',
      title: '问题路由台账',
      sourceFile: 'issue_ledger.json',
      zhFile: 'issue_ledger.zh.json',
      step: 'llm-review',
      contentType: 'application/json'
    })
    expect(findArtifactDefinition('deterministic_proposal_report')).toMatchObject({
      key: 'deterministic_proposal_report',
      title: '确定性 Proposal 漏斗',
      sourceFile: 'deterministic_proposal_report.json',
      zhFile: 'deterministic_proposal_report.zh.json',
      step: 'llm-review',
      contentType: 'application/json'
    })
    expect(findArtifactDefinition('deterministic_proposal_report_md')).toMatchObject({
      key: 'deterministic_proposal_report_md',
      title: '确定性 Proposal 漏斗报告',
      sourceFile: 'deterministic_proposal_report.md',
      zhFile: 'deterministic_proposal_report.zh.md',
      step: 'llm-review',
      contentType: 'text/markdown'
    })
    expect(artifactsForStep('llm-review').map((artifact) => artifact.key)).toEqual(
      expect.arrayContaining([
        'proposal_task_packs',
        'proposal_merge_report',
        'subagent_output_index',
        'issue_ledger',
        'deterministic_proposal_report',
        'deterministic_proposal_report_md'
      ])
    )
  })
})
