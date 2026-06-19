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
})
