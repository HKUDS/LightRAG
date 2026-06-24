import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type {
  KBIterationDisplayArtifactResponse,
  KBIterationProposalDecision,
  KBIterationProposalDecisionResponse,
  KBIterationSummaryResponse
} from '@/api/lightrag'
import type { KGMaintenanceSection } from '@/stores/kgMaintenance'

if (!('localStorage' in globalThis)) {
  const storage = new Map<string, string>()
  Object.defineProperty(globalThis, 'localStorage', {
    value: {
      getItem: (key: string) => storage.get(key) ?? null,
      setItem: (key: string, value: string) => storage.set(key, value),
      removeItem: (key: string) => storage.delete(key),
      clear: () => storage.clear()
    }
  })
}

const { default: KGMaintenanceShell } = await import('./KGMaintenanceShell')
const kgMaintenanceConsoleModule = await import('@/features/KGMaintenanceConsole')
const { ArtifactsDrawerPlaceholder, MainPanel } = kgMaintenanceConsoleModule
const { buildDisplayArtifactItems } = await import('./kgMaintenanceArtifactItems')
const { buildDefaultLLMReviewRequest } = await import('./kgMaintenanceLLMReviewRequest')
const { useKGMaintenanceStore } = await import('@/stores/kgMaintenance')
const {
  applyWorkspaceResponse,
  loadKGMaintenanceWorkspaceBundle,
  normalizeWorkspaceList,
  normalizeOptionalMarkdown,
  optionalMissingResponse,
  runWorkspaceAction,
  isGeneratedDisplayArtifact,
  normalizeTraceArtifactForLogic,
  requestProposalRevisionForWorkspace,
  submitAllProposalDecisionsForWorkspace,
  submitProposalDecisionForWorkspace,
  shouldApplyWorkspaceResponse
} = await import('./kgIterationLoadUtils')

const summary: KBIterationSummaryResponse = {
  workspace: 'influenza_medical_v1',
  latestRunId: 'snapshot-1',
  phase: 'pending_human_review',
  counts: {
    nodes: 1,
    edges: 1,
    sources: 1
  },
  quality: {
    overall: 82
  },
  pendingApprovalCount: 1,
  highRiskFindingCount: 0,
  artifacts: [
    'kb_context',
    'quality_report',
    'kg_snapshot',
    'quality_score',
    'approval_queue',
    'improvement_backlog',
    'accepted_changes',
    'rejected_changes',
    'accepted_changes_apply_result',
    'accepted_changes_execution',
    'iteration_log'
  ].map((key) => ({
    key,
    contentType:
      key === 'kg_snapshot' || key === 'quality_score' ? 'application/json' : 'text/markdown',
    exists: true
  }))
}

const displayArtifacts: Record<string, KBIterationDisplayArtifactResponse> = {
  kb_context: {
    artifactKey: 'kb_context',
    contentType: 'text/markdown',
    content: '# 中文 KB 摘要',
    display: {
      language: 'zh',
      zhFile: 'kb_context.zh.md',
      generated: true,
      fallbackToSource: false
    }
  },
  llm_issue_analysis: {
    artifactKey: 'llm_issue_analysis',
    contentType: 'text/markdown',
    content: '# 中文 LLM 问题分析',
    display: {
      language: 'zh',
      zhFile: 'llm_issue_analysis.zh.md',
      generated: true,
      fallbackToSource: false
    }
  },
  approval_queue: {
    artifactKey: 'approval_queue',
    contentType: 'text/markdown',
    content: '# 中文审批队列',
    display: {
      language: 'zh',
      zhFile: 'approval_queue.zh.md',
      generated: true,
      fallbackToSource: false
    }
  },
  accepted_changes_apply_result: {
    artifactKey: 'accepted_changes_apply_result',
    contentType: 'text/markdown',
    content: '# 中文执行结果',
    display: {
      language: 'zh',
      zhFile: 'accepted_changes_apply_result.zh.md',
      generated: true,
      fallbackToSource: false
    }
  },
  improvement_backlog: {
    artifactKey: 'improvement_backlog',
    contentType: 'text/markdown',
    content: '# 中文改进 Backlog',
    display: {
      language: 'zh',
      zhFile: 'improvement_backlog.zh.md',
      generated: true,
      fallbackToSource: false
    }
  }
}

function renderMainPanel(
  activeSection: KGMaintenanceSection,
  options: {
    acceptedChanges?: string
    rejectedChanges?: string
    deferredChanges?: string
    deferredChangesSource?: string
    acceptedApplyResult?: string
    acceptedApplyResultSource?: string
    qualityScoreSource?: Record<string, any> | null
    omitQualityScoreSource?: boolean
    llmTrace?: Record<string, any> | null
    llmProposals?: string
    llmProposalsSource?: string
    deterministicProposalReport?: Record<string, any> | null
    approvalQueue?: string
    approvalQueueSource?: string
    acceptedExecuting?: boolean
    llmRunning?: boolean
    running?: boolean
    loading?: boolean
  } = {}
) {
  const defaultQualityScore = {
    overall: 97,
    metrics: {
      hierarchy_missing_branch_count: 0
    },
    findings: [{ severity: 'medium' }]
  }
  const defaultApprovalQueue = `# 寰呭鎵?proposal

- id: proposal-1
  type: prompt_edit
  target: workspace_profile.json
  proposed_change: 璋冩暣 workspace 瀹￠槄 profile
  reason: 闇€瑕佹洿绮剧‘鐨?proposal 瀹℃壒绛栫暐
  evidence:
  - rule:r1
  confidence: 0.80
  risk: high
  requires_approval: true
      expected_metric_change:
    approval_latency: -1`
  const approvalQueueForPanel = options.approvalQueue ?? defaultApprovalQueue
  const qualityScoreSourceForPanel = options.omitQualityScoreSource
    ? undefined
    : (options.qualityScoreSource ?? defaultQualityScore)

  return renderToStaticMarkup(
    <MainPanel
      activeSection={activeSection}
      summary={summary}
      quality={{
        workspace: 'influenza_medical_v1',
        runId: 'snapshot-1',
        quality: {
          overall: 82,
          findings: [
            {
              severity: 'medium',
              category: 'coverage',
              message: 'Need more evidence',
              evidence: [],
              suggested_fix_type: 'manual_review',
              requires_approval: true
            }
          ]
        },
        report: '# 质量报告'
      }}
      rules={{
        workspace: 'influenza_medical_v1',
        qualityRules: '',
        knownIssues: '',
        acceptedChanges:
          options.acceptedChanges ??
          `# Accepted Changes

## proposal-1

accepted content marker`,
        rejectedChanges: options.rejectedChanges ?? 'rejected content marker'
      }}
      qualityScoreSource={qualityScoreSourceForPanel}
      approvalQueue={
        options.approvalQueue ??
        `# 待审批 proposal

- id: proposal-1
  type: prompt_edit
  target: workspace_profile.json
  proposed_change: 调整 workspace 审阅 profile
  reason: 需要更精确的 proposal 审批策略
  evidence:
  - rule:r1
  confidence: 0.80
  risk: high
  requires_approval: true
      expected_metric_change:
    approval_latency: -1`
      }
      approvalQueueSource={options.approvalQueueSource ?? approvalQueueForPanel}
      improvementBacklog="backlog content marker"
      deferredChanges={options.deferredChanges ?? 'deferred content marker'}
      deferredChangesSource={options.deferredChangesSource ?? options.deferredChanges ?? ''}
      acceptedApplyResult={
        options.acceptedApplyResult ??
        `Applied: 2
overall: 88 -> 97
hierarchy_missing_branch_count: 4 -> 0`
      }
      acceptedApplyResultSource={options.acceptedApplyResultSource}
      llmTrace={
        options.llmTrace ?? {
          stop_reason: 'pending_human_review',
          rounds: [{ round_id: 'round-1', state: 'pending_human_review' }]
        }
      }
      llmReport="# LLM Review Report"
      llmProposals={options.llmProposals ?? '- id: proposal-1'}
      llmProposalsSource={options.llmProposalsSource}
      deterministicProposalReport={options.deterministicProposalReport ?? null}
      llmJudgeReport="# Judge Report"
      llmIssueAnalysis="# Issue Analysis"
      llmMissingBranchInference="# Missing Branch Inference"
      llmEvidenceMap="# Evidence Map"
      llmRepairPlan="# Repair Plan"
      patchText="patch content marker"
      displayArtifacts={displayArtifacts}
      acceptedExecuting={options.acceptedExecuting ?? false}
      llmRunning={options.llmRunning ?? false}
      running={options.running ?? false}
      loading={options.loading ?? false}
      onOpenSection={() => undefined}
      onRunReview={() => undefined}
      onProposalDecision={() => undefined}
      onAcceptAllProposals={() => undefined}
      onExecuteAcceptedChanges={() => undefined}
      onRunLLMReview={() => undefined}
      onLoadPatch={() => undefined}
    />
  )
}

function renderEmptyMainPanel(activeSection: KGMaintenanceSection) {
  return renderToStaticMarkup(
    <MainPanel
      activeSection={activeSection}
      summary={null}
      quality={null}
      rules={null}
      qualityScoreSource={null}
      approvalQueue=""
      approvalQueueSource=""
      improvementBacklog=""
      deferredChanges=""
      deferredChangesSource=""
      acceptedApplyResult=""
      llmTrace={null}
      llmReport=""
      llmProposals=""
      deterministicProposalReport={null}
      llmJudgeReport=""
      llmIssueAnalysis=""
      llmMissingBranchInference=""
      llmEvidenceMap=""
      llmRepairPlan=""
      patchText=""
      displayArtifacts={{}}
      acceptedExecuting={false}
      llmRunning={false}
      running={false}
      loading={false}
      onOpenSection={() => undefined}
      onRunReview={() => undefined}
      onProposalDecision={() => undefined}
      onExecuteAcceptedChanges={() => undefined}
      onRunLLMReview={() => undefined}
      onLoadPatch={() => undefined}
    />
  )
}
void renderEmptyMainPanel

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  return { promise, resolve, reject }
}

function proposalDecisionResponse(
  decision: KBIterationProposalDecision = 'accept'
): KBIterationProposalDecisionResponse {
  return {
    workspace: 'workspace-a',
    proposalId: 'proposal-1',
    decision,
    record: {
      proposal_id: 'proposal-1',
      decision,
      reason: 'test decision',
      impact_scope: 'test scope',
      verification: 'test verification'
    }
  }
}

function createWorkspaceBundleLoaders(overrides: Record<string, any> = {}) {
  return {
    getSummary: async () => ({
      workspace: 'workspace-a',
      latestRunId: 'run-1',
      phase: 'pending_human_review',
      counts: { nodes: 1, edges: 1, sources: 1 },
      quality: { overall: 80, findings: [] },
      pendingApprovalCount: 0,
      highRiskFindingCount: 0,
      artifacts: []
    }),
    getQuality: async () => ({
      workspace: 'workspace-a',
      runId: 'run-1',
      quality: { overall: 80, findings: [] },
      report: '# quality'
    }),
    getRules: async () => ({
      workspace: 'workspace-a',
      qualityRules: '',
      knownIssues: '',
      acceptedChanges: '',
      rejectedChanges: ''
    }),
    getArtifact: async (_workspace: string, key: string) => ({
      artifactKey: key,
      contentType:
        key === 'kg_snapshot' || key === 'quality_score' ? 'application/json' : 'text/markdown',
      ...(key === 'kg_snapshot' || key === 'quality_score'
        ? { payload: { source: key } }
        : { content: `source:${key}` })
    }),
    getTrace: async () => ({
      artifactKey: 'trace',
      contentType: 'application/json',
      payload: {}
    }),
    getReport: async () => ({
      artifactKey: 'report',
      contentType: 'text/markdown',
      content: 'report'
    }),
    getProposals: async () => ({
      artifactKey: 'proposals',
      contentType: 'text/markdown',
      content: 'proposals'
    }),
    getJudgeReport: async () => ({
      artifactKey: 'judge',
      contentType: 'text/markdown',
      content: 'judge'
    }),
    getDisplayArtifact: async (_workspace: string, key: string) => ({
      artifactKey: key,
      contentType: 'text/markdown',
      content: `zh:${key}`,
      display: {
        language: 'zh',
        zhFile: `${key}.zh.md`,
        exists: true,
        generated: true
      }
    }),
    ...overrides
  }
}

describe('KGMaintenanceShell responsive layout', () => {
  test('store starts at the check workflow section', () => {
    expect(useKGMaintenanceStore.getState().activeSection).toBe('check')
  })

  test('console default LLM review request sends bounded subagent controls', () => {
    expect(buildDefaultLLMReviewRequest('clinical_guideline_zh')).toMatchObject({
      profile: 'clinical_guideline_zh',
      mode: 'agent_pipeline',
      max_stage_retries: 1,
      max_review_rounds: 4,
      max_focus_items_per_round: 3,
      max_subagent_tasks: 8,
      max_parallel_subagents: 4,
      max_subagent_issues_per_task: 4,
      max_subagent_proposals_per_task: 2,
      max_proposals_per_run: 200,
      strict_subagent_role_contracts: true,
      prevalidate_action_candidates: true,
      require_candidate_evidence_allowlist: true,
      skip_deterministic_subagent_calls: true,
      allow_llm_judge: true,
      allow_llm_auto_accept: false,
      allow_low_risk_auto_reject: true,
      generate_patch_candidates: false,
      require_human_for_mutation: true
    })
  })

  test('renders the fallback workspace option when workspaces is empty', () => {
    const markup = renderToStaticMarkup(
      <KGMaintenanceShell
        activeSection="check"
        onSectionChange={() => undefined}
        workspaces={[]}
        selectedWorkspace={null}
        onWorkspaceChange={() => undefined}
        onRefresh={() => undefined}
        onOpenArtifacts={() => undefined}
        loading={false}
        running={false}
        error={null}
      >
        <div>Console body</div>
      </KGMaintenanceShell>
    )

    expect(markup).toContain('value=""')
    expect(markup).toContain('未选择 workspace')
  })

  test('renders only the workflow toolbar actions', () => {
    const markup = renderToStaticMarkup(
      <KGMaintenanceShell
        activeSection="check"
        onSectionChange={() => undefined}
        workspaces={['influenza_medical_v1']}
        selectedWorkspace="influenza_medical_v1"
        onWorkspaceChange={() => undefined}
        onRefresh={() => undefined}
        onOpenArtifacts={() => undefined}
        loading={false}
        running={false}
        error={null}
      >
        <div>Console body</div>
      </KGMaintenanceShell>
    )

    expect(markup).toContain('flex-wrap')
    expect(markup).toContain('min-w-0 rounded-md border px-3 text-sm sm:min-w-52')
    expect(markup).toContain('刷新')
    expect(markup).toContain('全部产物')
    expect(markup).not.toContain('运行审阅包')
  })

  test('renders exactly five Chinese workflow navigation steps', () => {
    const markup = renderToStaticMarkup(
      <KGMaintenanceShell
        activeSection="check"
        onSectionChange={() => undefined}
        workspaces={['influenza_medical_v1']}
        selectedWorkspace="influenza_medical_v1"
        onWorkspaceChange={() => undefined}
        onRefresh={() => undefined}
        onOpenArtifacts={() => undefined}
        loading={false}
        running={false}
        error={null}
      >
        <div>Console body</div>
      </KGMaintenanceShell>
    )

    expect(markup).toContain('检查知识库')
    expect(markup).toContain('LLM 审阅')
    expect(markup).toContain('Proposal 审批')
    expect(markup).toContain('执行变更')
    expect(markup).toContain('验证结果')
    expect(markup).not.toContain('审阅包概览')
    expect(markup).not.toContain('当前阶段')
    expect(markup).not.toContain('快照审阅')
    expect(markup).not.toContain('决策与执行')
    expect(markup).not.toContain('Medical Graph')

    const renderedStepIds = Array.from(markup.matchAll(/data-workflow-step="([^"]+)"/g)).map(
      ([, step]) => step
    )
    expect(renderedStepIds).toEqual(['check', 'llm-review', 'approval', 'execute', 'validate'])
    expect(renderedStepIds).toHaveLength(5)
    expect(markup).toContain('data-workflow-step="check"')
    expect(markup).toContain('aria-current="step"')
  })

  test('uses left navigation and main content without a right inspector column', () => {
    const markup = renderToStaticMarkup(
      <KGMaintenanceShell
        activeSection="llm-review"
        onSectionChange={() => undefined}
        workspaces={['influenza_medical_v1']}
        selectedWorkspace="influenza_medical_v1"
        onWorkspaceChange={() => undefined}
        onRefresh={() => undefined}
        onOpenArtifacts={() => undefined}
        loading={false}
        running={false}
        error={null}
      >
        <div>LLM body</div>
      </KGMaintenanceShell>
    )

    expect(markup).toContain('LLM body')
    expect(markup).toContain('grid-cols-1')
    expect(markup).toContain('lg:grid-cols-[240px_minmax(0,1fr)]')
    expect(markup).not.toContain('Inspector')
    expect(markup).not.toContain('lg:grid-cols-[220px_minmax(0,1fr)_320px]')
  })

  test('renders the temporary artifacts dialog with a close action', () => {
    const markup = renderToStaticMarkup(
      <ArtifactsDrawerPlaceholder open={true} onOpenChange={() => undefined} />
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-label="全部产物"')
    expect(markup).toContain('关闭')
  })

  test('drawer status treats real generated display metadata as generated', () => {
    expect(
      isGeneratedDisplayArtifact({
        artifactKey: 'kb_context',
        contentType: 'text/markdown',
        content: 'zh context',
        display: {
          language: 'zh',
          zhFile: 'kb_context.zh.md',
          generated: true,
          fallbackToSource: false
        }
      })
    ).toBe(true)

    expect(
      isGeneratedDisplayArtifact({
        artifactKey: 'kb_context',
        contentType: 'text/markdown',
        content: 'source context',
        display: {
          language: 'zh',
          zhFile: 'kb_context.zh.md',
          generated: true,
          fallbackToSource: true
        }
      })
    ).toBe(false)
  })
})

describe('MainPanel workflow routing', () => {
  test('main panel maps each workflow section to the correct Chinese work surface', () => {
    const expectations: Array<{
      section: KGMaintenanceSection
      heading: string
      action: string
      artifactFile: string
    }> = [
      {
        section: 'check',
        heading: '检查知识库',
        action: '运行检查',
        artifactFile: 'kb_context.md'
      },
      {
        section: 'llm-review',
        heading: 'LLM 审阅',
        action: '运行 LLM 审阅',
        artifactFile: 'llm_issue_analysis.md'
      },
      {
        section: 'approval',
        heading: 'Proposal 审批',
        action: '查看待审批',
        artifactFile: 'approval_queue.md'
      },
      {
        section: 'execute',
        heading: '执行变更',
        action: '执行已接受变更',
        artifactFile: 'accepted_changes_apply_result.md'
      },
      {
        section: 'validate',
        heading: '验证结果',
        action: '开始下一轮复核',
        artifactFile: 'improvement_backlog.md'
      }
    ]

    for (const expectation of expectations) {
      const markup = renderMainPanel(expectation.section)

      expect(markup).toContain(expectation.heading)
      expect(markup).toContain(expectation.action)
      expect(markup).toContain(expectation.artifactFile)
      expect(markup).toContain('相关产物')
    }
  })

  test('main panel disables step header action while workflow is busy', () => {
    const markup = renderMainPanel('execute', { acceptedExecuting: true })

    expect(markup).toContain('disabled=""')
    expect(markup).toContain('处理中')
    expect(markup).toContain('执行变更')
  })

  test('main panel keeps generated display artifacts out of source content', () => {
    const items = buildDisplayArtifactItems({
      step: 'check',
      displayArtifacts: {
        kb_context: {
          artifactKey: 'kb_context',
          contentType: 'text/markdown',
          content: 'generated zh display content',
          display: {
            language: 'zh',
            zhFile: 'kb_context.zh.md',
            generated: true,
            fallbackToSource: false
          }
        }
      },
      sourceArtifacts: {
        kb_context: 'display-state kbContext content'
      },
      artifactExists: new Map([['kb_context', true]])
    })

    const kbContext = items.find((artifact) => artifact.key === 'kb_context')

    expect(kbContext?.content).toBe('generated zh display content')
    expect(kbContext?.originalContent).toBeUndefined()

    const fallbackItems = buildDisplayArtifactItems({
      step: 'check',
      displayArtifacts: {
        kb_context: {
          artifactKey: 'kb_context',
          contentType: 'text/markdown',
          content: 'source fallback content',
          display: {
            language: 'zh',
            zhFile: 'kb_context.zh.md',
            generated: true,
            fallbackToSource: true
          }
        }
      },
      sourceArtifacts: {},
      artifactExists: new Map([['kb_context', true]])
    })
    const fallbackKbContext = fallbackItems.find((artifact) => artifact.key === 'kb_context')

    expect(fallbackKbContext?.originalContent).toBe('source fallback content')
  })

  test('llm review source artifacts include deterministic proposal report JSON', () => {
    const report = {
      families: {
        diagnosis: {
          raw_issue_count: 2,
          deterministic_covered_count: 1
        }
      }
    }
    const items = buildDisplayArtifactItems({
      step: 'llm-review',
      displayArtifacts: {},
      sourceArtifacts: {
        deterministic_proposal_report: JSON.stringify(report, null, 2)
      },
      artifactExists: new Map([['deterministic_proposal_report', true]])
    })

    const item = items.find((artifact) => artifact.key === 'deterministic_proposal_report')

    expect(item?.originalContent).toBe(JSON.stringify(report, null, 2))
    expect(item?.displayStatus).toBe('原始文件')
  })

  test('main panel production routing excludes transitional workflow section labels', () => {
    const markup = (['check', 'llm-review', 'approval', 'execute', 'validate'] as const)
      .map((section) => renderMainPanel(section))
      .join('\n')

    expect(markup).not.toContain('审阅包概览')
    expect(markup).not.toContain('快照审阅')
    expect(markup).not.toContain('决策与执行')
  })

  test('workspace normalization preserves valid workspace arrays', () => {
    expect(normalizeWorkspaceList(['influenza_medical_v1'])).toEqual(['influenza_medical_v1'])
  })

  test('workspace normalization collapses malformed HTML fallback responses', () => {
    expect(normalizeWorkspaceList('<!doctype html><html><body>dev fallback</body></html>')).toEqual(
      []
    )
  })

  test('optional response helper returns fallback for missing resources only', async () => {
    const result = await optionalMissingResponse(async () => {
      throw Object.assign(new Error('404 Not Found'), { response: { status: 404 } })
    }, null)

    expect(result).toBeNull()
  })

  test('optional response helper returns fallback for wrapper-style 404 messages', async () => {
    const result = await optionalMissingResponse(async () => {
      throw new Error('404 Not Found\n{"detail":"missing"}\n/kb-iteration/workspace/summary')
    }, null)

    expect(result).toBeNull()
  })

  test('optional response helper rethrows unexpected core failures', async () => {
    await expect(
      optionalMissingResponse(async () => {
        throw Object.assign(new Error('500 Internal Server Error'), { response: { status: 500 } })
      }, null)
    ).rejects.toThrow('500 Internal Server Error')
  })

  test('optional response helper rethrows wrapper-style 500 with not-found body text', async () => {
    await expect(
      optionalMissingResponse(async () => {
        throw new Error(
          '500 Internal Server Error\n{"detail":"not found"}\n/kb-iteration/workspace/summary'
        )
      }, null)
    ).rejects.toThrow('500 Internal Server Error')
  })

  test('optional response helper rethrows plain missing artifact messages without explicit 404', async () => {
    await expect(
      optionalMissingResponse(async () => {
        throw new Error('artifact not found')
      }, null)
    ).rejects.toThrow('artifact not found')

    await expect(
      optionalMissingResponse(async () => {
        throw new Error('missing artifact')
      }, null)
    ).rejects.toThrow('missing artifact')
  })

  test('workspace bundle rethrows missing core summary responses', async () => {
    await expect(
      loadKGMaintenanceWorkspaceBundle('workspace-a', {
        getSummary: async () => {
          throw Object.assign(new Error('404 Not Found'), { response: { status: 404 } })
        },
        getQuality: async () => ({
          workspace: 'workspace-a',
          runId: 'run-1',
          quality: { overall: 80, findings: [] },
          report: '# quality'
        }),
        getRules: async () => ({
          workspace: 'workspace-a',
          qualityRules: '',
          knownIssues: '',
          acceptedChanges: '',
          rejectedChanges: ''
        }),
        getArtifact: async () => ({
          artifactKey: 'artifact',
          contentType: 'text/markdown',
          content: 'artifact'
        }),
        getDisplayArtifact: async () => {
          throw Object.assign(new Error('404 Not Found'), { response: { status: 404 } })
        },
        getTrace: async () => ({
          artifactKey: 'trace',
          contentType: 'application/json',
          payload: {}
        }),
        getReport: async () => ({
          artifactKey: 'report',
          contentType: 'text/markdown',
          content: 'report'
        }),
        getProposals: async () => ({
          artifactKey: 'proposals',
          contentType: 'text/markdown',
          content: 'proposals'
        }),
        getJudgeReport: async () => ({
          artifactKey: 'judge',
          contentType: 'text/markdown',
          content: 'judge'
        })
      })
    ).rejects.toThrow('404 Not Found')
  })

  test('workspace bundle keeps optional artifact fallbacks for missing files', async () => {
    const bundle = await loadKGMaintenanceWorkspaceBundle('workspace-a', {
      getSummary: async () => ({
        workspace: 'workspace-a',
        latestRunId: 'run-1',
        phase: 'pending_human_review',
        counts: { nodes: 1, edges: 1, sources: 1 },
        quality: { overall: 80, findings: [] },
        pendingApprovalCount: 0,
        highRiskFindingCount: 0,
        artifacts: []
      }),
      getQuality: async () => ({
        workspace: 'workspace-a',
        runId: 'run-1',
        quality: { overall: 80, findings: [] },
        report: '# quality'
      }),
      getRules: async () => ({
        workspace: 'workspace-a',
        qualityRules: '',
        knownIssues: '',
        acceptedChanges: '',
        rejectedChanges: ''
      }),
      getArtifact: async (_workspace, key) => {
        if (key === 'kb_context') {
          throw Object.assign(new Error('404 Not Found'), { response: { status: 404 } })
        }
        return {
          artifactKey: key,
          contentType: 'text/markdown',
          content: `${key} content`
        }
      },
      getDisplayArtifact: async () => {
        throw Object.assign(new Error('404 Not Found'), { response: { status: 404 } })
      },
      getTrace: async () => {
        throw Object.assign(new Error('404 Not Found'), { response: { status: 404 } })
      },
      getReport: async () => ({
        artifactKey: 'report',
        contentType: 'text/markdown',
        content: 'report'
      }),
      getProposals: async () => ({
        artifactKey: 'proposals',
        contentType: 'text/markdown',
        content: 'proposals'
      }),
      getJudgeReport: async () => ({
        artifactKey: 'judge',
        contentType: 'text/markdown',
        content: 'judge'
      })
    })

    expect(bundle.kbContextArtifact).toBe('')
    expect(bundle.llmTraceArtifact).toBeNull()
    expect(bundle.approvalArtifact).toBe('approval_queue content')
    expect(bundle.deferredChangesArtifact).toBe('deferred_changes content')
    expect(bundle.acceptedApplyResultArtifact).toBe('accepted_changes_apply_result content')
  })

  test('workspace bundle loads zh display artifacts when available', async () => {
    const bundle = await loadKGMaintenanceWorkspaceBundle(
      'workspace-a',
      createWorkspaceBundleLoaders()
    )

    expect(bundle.kbContextArtifact).toBe('zh:kb_context')
    expect(bundle.displayArtifacts.kb_context.content).toBe('zh:kb_context')
    expect(bundle.acceptedApplyResultArtifact).toBe('zh:accepted_changes_apply_result')
    expect(bundle.acceptedApplyResultSourceArtifact).toBe('source:accepted_changes_apply_result')
  })

  test('workspace bundle preserves source approval and quality artifacts for logic', async () => {
    const bundle = await loadKGMaintenanceWorkspaceBundle(
      'workspace-a',
      createWorkspaceBundleLoaders({
        getArtifact: async (_workspace: string, key: string) => ({
          artifactKey: key,
          contentType:
            key === 'quality_score' || key === 'kg_snapshot'
              ? 'application/json'
              : 'text/markdown',
          ...(key === 'quality_score'
            ? { payload: { overall: 97, metrics: { hierarchy_missing_branch_count: 0 } } }
            : { content: `source:${key}` })
        }),
        getDisplayArtifact: async (_workspace: string, key: string) => {
          let content = `zh:${key}`
          if (key === 'approval_queue') {
            content = 'translated approval queue without parseable ids'
          }
          if (key === 'deferred_changes') {
            content = 'translated deferred decisions with display-only ids'
          }

          if (key === 'quality_score') {
            return {
              artifactKey: key,
              contentType: 'application/json',
              payload: {
                overall: 'display-score',
                metrics: { hierarchy_missing_branch_count: 99 }
              },
              display: {
                language: 'zh',
                zhFile: 'quality_score.zh.json',
                generated: true,
                fallbackToSource: false
              }
            }
          }

          return {
            artifactKey: key,
            contentType: 'text/markdown',
            content,
            display: {
              language: 'zh',
              zhFile: `${key}.zh.md`,
              generated: true,
              fallbackToSource: false
            }
          }
        }
      })
    )

    expect(bundle.approvalArtifact).toBe('translated approval queue without parseable ids')
    expect(bundle.approvalArtifactSource).toBe('source:approval_queue')
    expect(bundle.deferredChangesArtifact).toBe(
      'translated deferred decisions with display-only ids'
    )
    expect(bundle.deferredChangesSourceArtifact).toBe('source:deferred_changes')
    expect(bundle.qualityScoreArtifact).toEqual({
      overall: 'display-score',
      metrics: { hierarchy_missing_branch_count: 99 }
    })
    expect(bundle.qualityScoreSourceArtifact).toEqual({
      overall: 97,
      metrics: { hierarchy_missing_branch_count: 0 }
    })
  })

  test('workspace bundle prefers display artifacts for LLM review artifacts', async () => {
    const bundle = await loadKGMaintenanceWorkspaceBundle(
      'workspace-a',
      createWorkspaceBundleLoaders({
        getDisplayArtifact: async (_workspace: string, key: string) => {
          if (key === 'llm_review_trace') {
            return {
              artifactKey: key,
              contentType: 'application/json',
              payload: { stop_reason: 'zh-trace', rounds: [{ round_id: 'zh-round' }] },
              display: {
                language: 'zh',
                zhFile: 'llm_review_trace.zh.json',
                exists: true,
                generated: true
              }
            }
          }

          return {
            artifactKey: key,
            contentType: 'text/markdown',
            content: `zh:${key}`,
            display: {
              language: 'zh',
              zhFile: `${key}.zh.md`,
              exists: true,
              generated: true
            }
          }
        },
        getTrace: async () => ({
          artifactKey: 'trace',
          contentType: 'application/json',
          payload: { stop_reason: 'source-trace' }
        }),
        getReport: async () => ({
          artifactKey: 'report',
          contentType: 'text/markdown',
          content: 'source report'
        }),
        getProposals: async () => ({
          artifactKey: 'proposals',
          contentType: 'text/markdown',
          content: 'source proposals'
        }),
        getJudgeReport: async () => ({
          artifactKey: 'judge',
          contentType: 'text/markdown',
          content: 'source judge'
        })
      })
    )

    expect(bundle.llmTraceArtifact).toEqual({
      stop_reason: 'zh-trace',
      rounds: [{ round_id: 'zh-round' }]
    })
    expect(bundle.llmReportArtifact).toBe('zh:llm_review_report')
    expect(bundle.llmProposalsArtifact).toBe('zh:proposals_generated')
    expect(bundle.llmProposalsSourceArtifact).toBe('source proposals')
    expect(bundle.llmJudgeReportArtifact).toBe('zh:llm_judge_report')
    expect(bundle.displayArtifacts.llm_review_trace.payload).toEqual({
      stop_reason: 'zh-trace',
      rounds: [{ round_id: 'zh-round' }]
    })
    expect(bundle.displayArtifacts.llm_review_report.content).toBe('zh:llm_review_report')
    expect(bundle.displayArtifacts.proposals_generated.content).toBe('zh:proposals_generated')
    expect(bundle.displayArtifacts.llm_judge_report.content).toBe('zh:llm_judge_report')
  })

  test('llm review renders source trace while keeping display trace artifact available', async () => {
    const displayTrace = {
      stop_reason: 'zh_stop_reason_label',
      rounds: [{ round_id: 'zh-round-label' }]
    }
    const sourceTrace = {
      stop_reason: 'pending_human_review',
      rounds: [{ round_id: 'source-round-1', state: 'pending_human_review' }]
    }

    const bundle = await loadKGMaintenanceWorkspaceBundle(
      'workspace-a',
      createWorkspaceBundleLoaders({
        getDisplayArtifact: async (_workspace: string, key: string) => {
          if (key === 'llm_review_trace') {
            return {
              artifactKey: key,
              contentType: 'application/json',
              payload: displayTrace,
              display: {
                language: 'zh',
                zhFile: 'llm_review_trace.zh.json',
                generated: true,
                fallbackToSource: false
              }
            }
          }

          return {
            artifactKey: key,
            contentType: 'text/markdown',
            content: `zh:${key}`,
            display: {
              language: 'zh',
              zhFile: `${key}.zh.md`,
              generated: true,
              fallbackToSource: false
            }
          }
        },
        getTrace: async () => ({
          artifactKey: 'llm_review_trace',
          contentType: 'application/json',
          payload: sourceTrace
        })
      })
    )

    const markup = renderMainPanel('llm-review', {
      llmTrace: normalizeTraceArtifactForLogic(
        bundle.llmTraceSourceArtifact,
        bundle.llmTraceArtifact
      )
    })

    expect(bundle.displayArtifacts.llm_review_trace.payload).toEqual(displayTrace)
    expect(bundle.llmTraceArtifact).toEqual(displayTrace)
    expect(bundle.llmTraceSourceArtifact).toEqual(sourceTrace)
    expect(markup).toContain('pending_human_review')
    expect(markup).toContain('source-round-1')
    expect(markup).not.toContain('zh_stop_reason_label')
    expect(markup).not.toContain('zh-round-label')
  })

  test('workspace bundle prefers proposal funnel report JSON for LLM review logic', async () => {
    const funnelReport = {
      families: {
        treatment: {
          raw_issue_count: 4,
          llm_residual_count: 2
        }
      }
    }
    const legacyReport = {
      families: {
        diagnosis: {
          raw_issue_count: 2,
          llm_residual_count: 1
        }
      }
    }
    const requestedKeys: string[] = []

    const bundle = await loadKGMaintenanceWorkspaceBundle(
      'workspace-a',
      createWorkspaceBundleLoaders({
        getArtifact: async (_workspace: string, key: string) => {
          requestedKeys.push(key)
          if (key === 'proposal_funnel_report') {
            return {
              artifactKey: key,
              contentType: 'application/json',
              payload: funnelReport
            }
          }
          if (key === 'deterministic_proposal_report') {
            return {
              artifactKey: key,
              contentType: 'application/json',
              payload: legacyReport
            }
          }
          return {
            artifactKey: key,
            contentType:
              key === 'kg_snapshot' || key === 'quality_score'
                ? 'application/json'
                : 'text/markdown',
            ...(key === 'kg_snapshot' || key === 'quality_score'
              ? { payload: { source: key } }
              : { content: `source:${key}` })
          }
        }
      })
    )

    expect(requestedKeys).toContain('proposal_funnel_report')
    expect(bundle.deterministicProposalReportArtifact).toEqual(funnelReport)
  })

  test('workspace bundle falls back to deterministic proposal report JSON when funnel report is missing', async () => {
    const report = {
      families: {
        diagnosis: {
          raw_issue_count: 2,
          llm_residual_count: 1
        }
      }
    }
    const requestedKeys: string[] = []

    const bundle = await loadKGMaintenanceWorkspaceBundle(
      'workspace-a',
      createWorkspaceBundleLoaders({
        getArtifact: async (_workspace: string, key: string) => {
          requestedKeys.push(key)
          if (key === 'proposal_funnel_report') {
            throw Object.assign(new Error('404 Not Found'), { response: { status: 404 } })
          }
          if (key === 'deterministic_proposal_report') {
            return {
              artifactKey: key,
              contentType: 'application/json',
              payload: report
            }
          }
          return {
            artifactKey: key,
            contentType:
              key === 'kg_snapshot' || key === 'quality_score'
                ? 'application/json'
                : 'text/markdown',
            ...(key === 'kg_snapshot' || key === 'quality_score'
              ? { payload: { source: key } }
              : { content: `source:${key}` })
          }
        }
      })
    )

    expect(requestedKeys).toContain('proposal_funnel_report')
    expect(requestedKeys).toContain('deterministic_proposal_report')
    expect(bundle.deterministicProposalReportArtifact).toEqual(report)
  })

  test('trace normalization returns null when source artifact is missing', () => {
    expect(
      normalizeTraceArtifactForLogic(null, {
        stop_reason: 'display-only-trace',
        rounds: [{ round_id: 'display-round' }]
      })
    ).toBeNull()
  })

  test('workspace bundle falls back to original artifact content when display is missing', async () => {
    const bundle = await loadKGMaintenanceWorkspaceBundle(
      'workspace-a',
      createWorkspaceBundleLoaders({
        getDisplayArtifact: async (_workspace: string, key: string) => {
          if (key === 'kb_context') {
            throw Object.assign(new Error('404 Not Found'), { response: { status: 404 } })
          }
          return {
            artifactKey: key,
            contentType: 'text/markdown',
            content: `zh:${key}`,
            display: {
              language: 'zh',
              zhFile: `${key}.zh.md`,
              exists: true,
              generated: true
            }
          }
        }
      })
    )

    expect(bundle.kbContextArtifact).toBe('source:kb_context')
    expect(bundle.displayArtifacts.kb_context.content).toBe('source:kb_context')
    expect(bundle.displayArtifacts.kb_context.display.fallbackToSource).toBe(true)
  })

  test('workspace bundle rethrows non-missing display loading failures', async () => {
    await expect(
      loadKGMaintenanceWorkspaceBundle(
        'workspace-a',
        createWorkspaceBundleLoaders({
          getDisplayArtifact: async () => {
            throw Object.assign(new Error('500 Internal Server Error'), {
              response: { status: 500 }
            })
          },
          getTrace: async () => ({
            artifactKey: 'llm_review_trace',
            contentType: 'application/json',
            payload: { stop_reason: 'source-trace' }
          }),
          getReport: async () => ({
            artifactKey: 'llm_review_report',
            contentType: 'text/markdown',
            content: 'source report'
          }),
          getProposals: async () => ({
            artifactKey: 'proposals_generated',
            contentType: 'text/markdown',
            content: 'source proposals'
          }),
          getJudgeReport: async () => ({
            artifactKey: 'llm_judge_report',
            contentType: 'text/markdown',
            content: 'source judge'
          })
        })
      )
    ).rejects.toThrow('500 Internal Server Error')
  })

  test('workspace bundle display loading skips stale removed artifacts', async () => {
    const requestedDisplayKeys: string[] = []

    await loadKGMaintenanceWorkspaceBundle(
      'workspace-a',
      createWorkspaceBundleLoaders({
        getDisplayArtifact: async (_workspace: string, key: string) => {
          requestedDisplayKeys.push(key)
          return {
            artifactKey: key,
            contentType: 'text/markdown',
            content: `zh:${key}`,
            display: {
              language: 'zh',
              zhFile: `${key}.zh.md`,
              exists: true,
              generated: true
            }
          }
        }
      })
    )

    expect(requestedDisplayKeys).toContain('accepted_changes_apply_result')
    expect(requestedDisplayKeys).toContain('deferred_changes')
    expect(requestedDisplayKeys).not.toContain('accepted_changes_execution')
    expect(requestedDisplayKeys).not.toContain('iteration_log')
  })

  test('workspace bundle loads llm agent markdown artifacts', async () => {
    const requestedArtifactKeys: string[] = []

    const bundle = await loadKGMaintenanceWorkspaceBundle('workspace-a', {
      getSummary: async () => ({
        workspace: 'workspace-a',
        latestRunId: 'run-1',
        phase: 'pending_human_review',
        counts: { nodes: 1, edges: 1, sources: 1 },
        quality: { overall: 80, findings: [] },
        pendingApprovalCount: 0,
        highRiskFindingCount: 0,
        artifacts: []
      }),
      getQuality: async () => ({
        workspace: 'workspace-a',
        runId: 'run-1',
        quality: { overall: 80, findings: [] },
        report: '# quality'
      }),
      getRules: async () => ({
        workspace: 'workspace-a',
        qualityRules: '',
        knownIssues: '',
        acceptedChanges: '',
        rejectedChanges: ''
      }),
      getArtifact: async (_workspace, key) => {
        requestedArtifactKeys.push(key)
        return {
          artifactKey: key,
          contentType: 'text/markdown',
          content: `# ${key}`
        }
      },
      getDisplayArtifact: async () => {
        throw Object.assign(new Error('404 Not Found'), { response: { status: 404 } })
      },
      getTrace: async () => ({
        artifactKey: 'trace',
        contentType: 'application/json',
        payload: {}
      }),
      getReport: async () => ({
        artifactKey: 'report',
        contentType: 'text/markdown',
        content: 'report'
      }),
      getProposals: async () => ({
        artifactKey: 'proposals',
        contentType: 'text/markdown',
        content: 'proposals'
      }),
      getJudgeReport: async () => ({
        artifactKey: 'judge',
        contentType: 'text/markdown',
        content: 'judge'
      })
    })

    expect(requestedArtifactKeys).toContain('llm_issue_analysis')
    expect(requestedArtifactKeys).toContain('llm_missing_branch_inference')
    expect(requestedArtifactKeys).toContain('llm_evidence_map')
    expect(requestedArtifactKeys).toContain('llm_repair_plan')
    expect(requestedArtifactKeys).toContain('accepted_changes_apply_result')
    expect(requestedArtifactKeys).toContain('deferred_changes')
    expect(requestedArtifactKeys).not.toContain('accepted_changes_execution')
    expect(requestedArtifactKeys).not.toContain('iteration_log')
    expect(bundle.llmIssueAnalysisArtifact).toBe('# llm_issue_analysis')
    expect(bundle.llmMissingBranchInferenceArtifact).toBe('# llm_missing_branch_inference')
    expect(bundle.llmEvidenceMapArtifact).toBe('# llm_evidence_map')
    expect(bundle.llmRepairPlanArtifact).toBe('# llm_repair_plan')
    expect(bundle.acceptedApplyResultArtifact).toBe('# accepted_changes_apply_result')
    expect(bundle.deferredChangesArtifact).toBe('# deferred_changes')
  })

  test('workspace response guard rejects stale workspace payloads', () => {
    expect(shouldApplyWorkspaceResponse('workspace-a', () => 'workspace-a')).toBe(true)
    expect(shouldApplyWorkspaceResponse('workspace-a', () => 'workspace-b')).toBe(false)
  })

  test('workspace response applier skips stale action results', () => {
    let applied = ''

    expect(
      applyWorkspaceResponse(
        'workspace-a',
        () => 'workspace-b',
        () => {
          applied = 'stale summary'
        }
      )
    ).toBe(false)
    expect(applied).toBe('')

    expect(
      applyWorkspaceResponse(
        'workspace-a',
        () => 'workspace-a',
        () => {
          applied = 'fresh summary'
        }
      )
    ).toBe(true)
    expect(applied).toBe('fresh summary')
  })

  test('workspace action skips stale success but still completes', async () => {
    const action = deferred<string>()
    let currentWorkspace = 'workspace-a'
    let applied = ''
    let completed = false

    const run = runWorkspaceAction({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => currentWorkspace,
      action: () => action.promise,
      onSuccess: (value) => {
        applied = value
      },
      onComplete: () => {
        completed = true
      }
    })

    currentWorkspace = 'workspace-b'
    action.resolve('stale summary')
    await run

    expect(applied).toBe('')
    expect(completed).toBe(true)
  })

  test('workspace action skips stale errors but still completes', async () => {
    const action = deferred<string>()
    let currentWorkspace = 'workspace-a'
    let error = ''
    let completed = false

    const run = runWorkspaceAction({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => currentWorkspace,
      action: () => action.promise,
      onError: (err) => {
        error = err instanceof Error ? err.message : String(err)
      },
      onComplete: () => {
        completed = true
      }
    })

    currentWorkspace = 'workspace-b'
    action.reject(new Error('stale failure'))
    await run

    expect(error).toBe('')
    expect(completed).toBe(true)
  })

  test('workspace action applies current success and error callbacks', async () => {
    const success = deferred<string>()
    let applied = ''
    let completedSuccess = false

    const runSuccess = runWorkspaceAction({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => 'workspace-a',
      action: () => success.promise,
      onSuccess: (value) => {
        applied = value
      },
      onComplete: () => {
        completedSuccess = true
      }
    })
    success.resolve('fresh summary')
    await runSuccess

    expect(applied).toBe('fresh summary')
    expect(completedSuccess).toBe(true)

    const failure = deferred<string>()
    let error = ''
    let completedFailure = false
    const runFailure = runWorkspaceAction({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => 'workspace-a',
      action: () => failure.promise,
      onError: (err) => {
        error = err instanceof Error ? err.message : String(err)
      },
      onComplete: () => {
        completedFailure = true
      }
    })
    failure.reject(new Error('fresh failure'))
    await runFailure

    expect(error).toBe('fresh failure')
    expect(completedFailure).toBe(true)
  })

  test('workspace action can recheck freshness after nested refresh awaits', async () => {
    const action = deferred<string>()
    const refresh = deferred<void>()
    let currentWorkspace = 'workspace-a'
    let applied = ''
    let completed = false
    let sawRecheckFunction = false

    const run = runWorkspaceAction({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => currentWorkspace,
      action: () => action.promise,
      onSuccess: async (value, shouldApply) => {
        sawRecheckFunction = typeof shouldApply === 'function'
        await refresh.promise
        if (shouldApply()) {
          applied = value
        }
      },
      onComplete: () => {
        completed = true
      }
    })

    action.resolve('post-refresh patch clear')
    await Promise.resolve()
    currentWorkspace = 'workspace-b'
    refresh.resolve()
    await run

    expect(sawRecheckFunction).toBe(true)
    expect(applied).toBe('')
    expect(completed).toBe(true)
  })

  test('proposal decision skips stale workspace refreshes', async () => {
    const recordDecision = deferred<KBIterationProposalDecisionResponse>()
    let currentWorkspace = 'workspace-a'
    let refreshCalls = 0
    let bannerError = ''

    const run = submitProposalDecisionForWorkspace({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => currentWorkspace,
      proposal: {
        id: 'proposal-1',
        type: 'prompt_edit',
        target: 'workspace_profile.json',
        proposedChange: 'Tighten approval policy',
        reason: 'More evidence needed',
        evidence: [],
        confidence: '0.8',
        risk: 'high',
        requiresApproval: true,
        expectedMetricChange: 'approval_latency: -1'
      },
      decision: 'accept',
      review: {
        reason: 'Looks good',
        impactScope: 'workspace profile only',
        verification: 'rerun review package',
        confirmation: ''
      },
      reloadWorkspaceData: async () => {
        refreshCalls += 1
      },
      recordDecision: () => recordDecision.promise,
      onError: (error) => {
        bannerError = error instanceof Error ? error.message : String(error)
      }
    })

    currentWorkspace = 'workspace-b'
    recordDecision.resolve(proposalDecisionResponse('accept'))
    await run

    expect(refreshCalls).toBe(0)
    expect(bannerError).toBe('')
  })

  test('proposal decision skips stale workspace errors', async () => {
    const recordDecision = deferred<KBIterationProposalDecisionResponse>()
    let currentWorkspace = 'workspace-a'
    let bannerError = ''

    const run = submitProposalDecisionForWorkspace({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => currentWorkspace,
      proposal: {
        id: 'proposal-1',
        type: 'prompt_edit',
        target: 'workspace_profile.json',
        proposedChange: 'Tighten approval policy',
        reason: 'More evidence needed',
        evidence: [],
        confidence: '0.8',
        risk: 'high',
        requiresApproval: true,
        expectedMetricChange: 'approval_latency: -1'
      },
      decision: 'reject',
      review: {
        reason: 'Need better evidence',
        impactScope: 'none',
        verification: 'keep current profile',
        confirmation: ''
      },
      reloadWorkspaceData: async () => undefined,
      recordDecision: () => recordDecision.promise,
      onError: (error) => {
        bannerError = error instanceof Error ? error.message : String(error)
      }
    })

    currentWorkspace = 'workspace-b'
    recordDecision.reject(new Error('stale failure'))
    await run

    expect(bannerError).toBe('')
  })

  test('proposal decision refreshes the current workspace after success', async () => {
    let refreshCalls = 0

    await submitProposalDecisionForWorkspace({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => 'workspace-a',
      proposal: {
        id: 'proposal-1',
        type: 'prompt_edit',
        target: 'workspace_profile.json',
        proposedChange: 'Tighten approval policy',
        reason: 'More evidence needed',
        evidence: [],
        confidence: '0.8',
        risk: 'high',
        requiresApproval: true,
        expectedMetricChange: 'approval_latency: -1'
      },
      decision: 'defer',
      review: {
        reason: 'Need another pass',
        impactScope: 'review queue only',
        verification: 'revisit next run',
        confirmation: ''
      },
      reloadWorkspaceData: async () => {
        refreshCalls += 1
      },
      recordDecision: async () => proposalDecisionResponse('defer')
    })

    expect(refreshCalls).toBe(1)
  })

  test('bulk proposal accept records each proposal and refreshes once', async () => {
    const calls: string[] = []
    let refreshCalls = 0

    await submitAllProposalDecisionsForWorkspace({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => 'workspace-a',
      proposals: [
        {
          id: 'proposal-1',
          type: 'prompt_edit',
          target: 'workspace_profile.json',
          proposedChange: 'Tighten approval policy',
          reason: 'More evidence needed',
          evidence: [],
          confidence: '0.8',
          risk: 'high',
          requiresApproval: true,
          expectedMetricChange: 'approval_latency: -1'
        },
        {
          id: 'proposal-2',
          type: 'relation_rule_change',
          target: 'relation_rules.json',
          proposedChange: 'Normalize relation labels',
          reason: 'Improve KG quality',
          evidence: [],
          confidence: '0.7',
          risk: 'medium',
          requiresApproval: true,
          expectedMetricChange: 'relation_quality: 1'
        }
      ],
      decision: 'accept',
      reloadWorkspaceData: async () => {
        refreshCalls += 1
      },
      recordDecision: async (workspace, proposalId, decision) => {
        calls.push(`${workspace}:${proposalId}:${decision}`)
        return proposalDecisionResponse(decision)
      }
    })

    expect(calls).toEqual(['workspace-a:proposal-1:accept', 'workspace-a:proposal-2:accept'])
    expect(refreshCalls).toBe(1)
  })

  test('proposal revision request calls backend helper and reloads current workspace', async () => {
    const calls: string[] = []
    let refreshCalls = 0

    await requestProposalRevisionForWorkspace({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => 'workspace-a',
      proposal: {
        id: 'proposal-1',
        type: 'prompt_edit',
        target: 'workspace_profile.json',
        proposedChange: 'Tighten approval policy',
        reason: 'More evidence needed',
        evidence: [],
        confidence: '0.8',
        risk: 'high',
        requiresApproval: true,
        expectedMetricChange: 'approval_latency: -1'
      },
      reloadWorkspaceData: async () => {
        refreshCalls += 1
      },
      requestRevision: async (workspace, proposalId) => {
        calls.push(`${workspace}:${proposalId}`)
        return {
          workspace,
          proposalId,
          artifactKey: 'proposal_revision_requests',
          record: {}
        }
      }
    })

    expect(calls).toEqual(['workspace-a:proposal-1'])
    expect(refreshCalls).toBe(1)
  })

  test('proposal revision request reports current workspace failures', async () => {
    let bannerError = ''

    await requestProposalRevisionForWorkspace({
      requestWorkspace: 'workspace-a',
      getCurrentWorkspace: () => 'workspace-a',
      proposal: {
        id: 'proposal-1',
        type: 'prompt_edit',
        target: 'workspace_profile.json',
        proposedChange: 'Tighten approval policy',
        reason: 'More evidence needed',
        evidence: [],
        confidence: '0.8',
        risk: 'high',
        requiresApproval: true,
        expectedMetricChange: 'approval_latency: -1'
      },
      reloadWorkspaceData: async () => undefined,
      requestRevision: async () => {
        throw new Error('revision failed')
      },
      onError: (error) => {
        bannerError = error instanceof Error ? error.message : String(error)
      }
    })

    expect(bannerError).toBe('revision failed')
  })

  test('markdown normalization accepts pre-normalized optional strings', () => {
    expect(normalizeOptionalMarkdown('loaded markdown')).toBe('loaded markdown')
    expect(normalizeOptionalMarkdown(null)).toBe('')
  })

  test('check renders focused summary metrics and only check-step related files', () => {
    const markup = renderMainPanel('check')
    const relatedFilesMarkup = markup.slice(markup.indexOf('相关产物'))

    expect(markup).toContain('运行检查')
    expect(markup).toContain('influenza_medical_v1 / snapshot-1')
    expect(markup).toContain('pending_human_review')
    expect(markup).toContain('82')
    expect(markup).toContain('待审批 Proposal')
    expect(markup).toContain('1 / 1 / 1')
    expect(relatedFilesMarkup).toContain('kb_context.md')
    expect(relatedFilesMarkup).toContain('quality_report.md')
    expect(relatedFilesMarkup).toContain('snapshots/kg_snapshot.json')
    expect(relatedFilesMarkup).toContain('snapshots/quality_score.json')
    expect(relatedFilesMarkup).toContain('entity_catalog.md')
    expect(relatedFilesMarkup).toContain('relation_catalog.md')
    expect(relatedFilesMarkup).toContain('kg_structure.md')
    expect(relatedFilesMarkup).toContain('snapshots/source_coverage.json')
    expect(markup).not.toContain('approval_queue.md')
    expect(markup).not.toContain('accepted_changes.md')
    expect(markup).not.toContain('rejected_changes.md')
    expect(markup).not.toContain('accepted_changes_apply_result.md')
    expect(markup).not.toContain('iteration_log.md')
    expect(markup).not.toContain('improvement_backlog.md')
  })

  test('check renders production schema issue table from quality score source without graph canvas', () => {
    const markup = renderMainPanel('check', {
      qualityScoreSource: {
        details: {
          medical_schema_issues: [
            {
              issue_kind: 'disease_symptom_taxonomy_misuse',
              edge_id: 'edge-dry-cough-flu',
              keywords: '属于',
              candidate_predicates: ['has_manifestation'],
              source_id: 'chunk-1',
              file_path: 'guide.md'
            }
          ]
        }
      }
    })

    expect(markup).toContain('Schema 问题')
    expect(markup).toContain('disease_symptom_taxonomy_misuse')
    expect(markup).toContain('edge-dry-cough-flu')
    expect(markup).toContain('has_manifestation')
    expect(markup).not.toContain('<canvas')
    expect(markup).not.toContain('Medical knowledge graph hierarchy')
  })

  test('does not export the legacy main panel', () => {
    expect('LegacyMainPanel' in kgMaintenanceConsoleModule).toBe(false)
  })

  test('approval renders proposal review content', () => {
    const markup = renderMainPanel('approval')

    expect(markup).toContain('proposal-1')
    expect(markup).toContain('aria-expanded="true"')
  })

  test('approval marks proposals that already have accepted decisions', () => {
    const markup = renderMainPanel('approval', {
      acceptedChanges: `# Accepted Changes

## proposal-1

\`\`\`json
{"proposal_id":"proposal-1","decision":"accept"}
\`\`\`
`
    })

    expect(markup).toContain('proposal-1')
    expect(markup).not.toContain('reject</button>')
  })

  test('approval marks proposals that already have deferred decisions', () => {
    const markup = renderMainPanel('approval', {
      deferredChanges: `# Deferred Changes

## proposal-1

\`\`\`json
{"proposal_id":"proposal-1","decision":"defer"}
\`\`\`
`,
      deferredChangesSource: `# Deferred Changes

## proposal-1

\`\`\`json
{"proposal_id":"proposal-1","decision":"defer"}
\`\`\`
`
    })

    expect(markup).toContain('proposal-1')
    expect(markup).toContain('已延后')
    expect(markup).not.toContain('bg-amber-100 text-amber-800')
    expect(markup).not.toContain('>延后</button>')
  })

  test('approval ignores display deferred decisions when source memory is empty', () => {
    const markup = renderMainPanel('approval', {
      acceptedChanges: '',
      rejectedChanges: '',
      deferredChanges: `# Deferred Changes

## proposal-1

\`\`\`json
{"proposal_id":"proposal-1","decision":"defer"}
\`\`\`
`,
      deferredChangesSource: ''
    })

    expect(markup).toContain('proposal-1')
    expect(markup).toContain('bg-amber-100 text-amber-800')
    expect(markup).not.toContain('bg-sky-100 text-sky-800')
    expect(markup).not.toContain('disabled=""')
  })

  test('execute renders the focused accepted-change execution surface', () => {
    const markup = renderMainPanel('execute')

    expect(markup).toContain('执行已接受变更')
    expect(markup).toContain('执行变更')
    expect(markup).toContain('Applied: 2')
    expect(markup).toContain('accepted content marker')
    expect(markup).not.toContain('improvement_backlog.md')
    expect(markup).not.toContain('backlog content marker')
    expect(markup).not.toContain('rejected content marker')
  })

  test('execute parses source apply result while rendering display apply result', () => {
    const markup = renderMainPanel('execute', {
      acceptedApplyResult: 'translated apply block without applied count',
      acceptedApplyResultSource: `Applied: 2
overall: 88 -> 97
hierarchy_missing_branch_count: 4 -> 0`
    })

    expect(markup).toContain('translated apply block without applied count')
    expect(markup).toContain('Applied: 2')
  })

  test('llm-review renders auxiliary review materials without memory fallthrough', () => {
    const markup = renderMainPanel('llm-review')

    expect(markup).toContain('pending_human_review')
    expect(markup).toContain('# LLM Review Report')
    expect(markup).not.toContain('accepted content marker')
  })

  test('llm-review parses patch proposal ids from source proposals while rendering display proposals', () => {
    const markup = renderMainPanel('llm-review', {
      llmProposals: 'translated proposal display without parseable ids',
      llmProposalsSource: `proposals:
  - id: proposal-source-1
    title: source proposal`
    })

    expect(markup).toContain('translated proposal display without parseable ids')
    expect(markup).toContain('proposal-source-1')
  })

  test('llm-review passes deterministic proposal report into the review panel', () => {
    const markup = renderMainPanel('llm-review', {
      deterministicProposalReport: {
        families: [
          {
            family: 'treatment',
            raw_issue_count: 4,
            issue_with_candidate_count: 3,
            action_candidate_count: 3,
            deterministic_covered_count: 2,
            llm_residual_count: 1,
            blocked_safety_count: 0,
            blocked_apply_count: 1,
            blocked_evidence_count: 0,
            deferred_budget_count: 0,
            selected_proposal_count: 2,
            reason_code_counts: { APPLY_NOT_SUPPORTED: 1 }
          }
        ]
      }
    })

    expect(markup).toContain('确定性 Proposal 漏斗')
    expect(markup).toContain('治疗')
    expect(markup).toContain('APPLY_NOT_SUPPORTED')
  })

  test('validate renders quality deltas and apply result', () => {
    const markup = renderMainPanel('validate')

    expect(markup).toContain('验证结果')
    expect(markup).toContain('88 → 97')
    expect(markup).toContain('4 → 0')
    expect(markup).toContain('Applied: 2')
    expect(markup).not.toContain('iteration_log.md')
  })

  test('validate parses source apply result while rendering display apply result', () => {
    const markup = renderMainPanel('validate', {
      acceptedApplyResult: 'translated validation block without backend metrics',
      acceptedApplyResultSource: `Applied: 0
overall: 88 -> 97
hierarchy_missing_branch_count: 4 -> 0`
    })

    expect(markup).toContain('translated validation block without backend metrics')
    expect(markup).toContain('88')
    expect(markup).toContain('97')
    expect(markup).toContain('4')
    expect(markup).toContain('bg-emerald-50/70')
  })

  test('validate parses source quality score while display score remains display-only', () => {
    const markup = renderMainPanel('validate', {
      acceptedApplyResult: 'translated validation block without backend metrics',
      acceptedApplyResultSource: `Applied: 0
overall: 88 -> 97
hierarchy_missing_branch_count: 4 -> 0`,
      qualityScoreSource: {
        overall: 97,
        metrics: {
          hierarchy_missing_branch_count: 0
        }
      }
    })

    expect(markup).toContain('88')
    expect(markup).toContain('97')
    expect(markup).toContain('4')
    expect(markup).toContain('0')
    expect(markup).toContain('bg-emerald-50/70')
    expect(markup).not.toContain('display-score')
    expect(markup).not.toContain('99')
  })

  test('validate does not parse display quality score when source is omitted', () => {
    const markup = renderMainPanel('validate', {
      acceptedApplyResult: 'translated validation block without backend metrics',
      acceptedApplyResultSource: `Applied: 0
overall: 88 -> 97
hierarchy_missing_branch_count: 4 -> 0`,
      omitQualityScoreSource: true
    })

    expect(markup).toContain('88')
    expect(markup).toContain('4')
    expect(markup).not.toContain('display-score')
    expect(markup).not.toContain('99')
  })
})
