import { describe, expect, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import type {
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
const { ArtifactsDrawerPlaceholder, MainPanel } = await import('@/features/KGMaintenanceConsole')
const { useKGMaintenanceStore } = await import('@/stores/kgMaintenance')
const {
  applyWorkspaceResponse,
  loadKGMaintenanceWorkspaceBundle,
  normalizeWorkspaceList,
  normalizeOptionalMarkdown,
  optionalMissingResponse,
  runWorkspaceAction,
  requestProposalRevisionForWorkspace,
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

function renderMainPanel(
  activeSection: KGMaintenanceSection,
  options: {
    acceptedChanges?: string
    rejectedChanges?: string
    deferredChanges?: string
    acceptedApplyResult?: string
  } = {}
) {
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
        acceptedChanges: options.acceptedChanges ?? `# Accepted Changes

## proposal-1

accepted content marker`,
        rejectedChanges: options.rejectedChanges ?? 'rejected content marker'
      }}
      kbContext="# 当前 KB 摘要"
      kgSnapshot={{
        workspace: 'influenza_medical_v1',
        snapshot_id: 'snapshot-1',
        nodes: [{ id: 'flu' }],
        edges: [{ source: 'flu', target: 'fever' }]
      }}
      qualityScore={{
        overall: 97,
        metrics: {
          hierarchy_missing_branch_count: 0
        },
        findings: [{ severity: 'medium' }]
      }}
      approvalQueue={`# 待审批 proposal

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
    approval_latency: -1`}
      improvementBacklog="backlog content marker"
      deferredChanges={options.deferredChanges ?? 'deferred content marker'}
      acceptedApplyResult={
        options.acceptedApplyResult ??
        `Applied: 2
overall: 88 -> 97
hierarchy_missing_branch_count: 4 -> 0`
      }
      llmTrace={{
        stop_reason: 'pending_human_review',
        rounds: [{ round_id: 'round-1', state: 'pending_human_review' }]
      }}
      llmReport="# LLM Review Report"
      llmProposals="- id: proposal-1"
      llmJudgeReport="# Judge Report"
      llmIssueAnalysis="# Issue Analysis"
      llmMissingBranchInference="# Missing Branch Inference"
      llmEvidenceMap="# Evidence Map"
      llmRepairPlan="# Repair Plan"
      patchText="patch content marker"
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

function renderEmptyMainPanel(activeSection: KGMaintenanceSection) {
  return renderToStaticMarkup(
    <MainPanel
      activeSection={activeSection}
      summary={null}
      quality={null}
      rules={null}
      kbContext=""
      kgSnapshot={null}
      qualityScore={null}
      approvalQueue=""
      improvementBacklog=""
      deferredChanges=""
      acceptedApplyResult=""
      llmTrace={null}
      llmReport=""
      llmProposals=""
      llmJudgeReport=""
      llmIssueAnalysis=""
      llmMissingBranchInference=""
      llmEvidenceMap=""
      llmRepairPlan=""
      patchText=""
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

describe('KGMaintenanceShell responsive layout', () => {
  test('store starts at the check workflow section', () => {
    expect(useKGMaintenanceStore.getState().activeSection).toBe('check')
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
})

describe('MainPanel workflow routing', () => {
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

  test('check renders the review package artifact list', () => {
    const markup = renderMainPanel('check')

    expect(markup).toContain('运行检查')
    expect(markup).toContain('kb_context.md')
    expect(markup).toContain('quality_report.md')
    expect(markup).toContain('snapshots/kg_snapshot.json')
    expect(markup).toContain('snapshots/quality_score.json')
    expect(markup).toContain('approval_queue.md')
    expect(markup).toContain('improvement_backlog.md')
    expect(markup).toContain('accepted_changes.md')
    expect(markup).toContain('rejected_changes.md')
    expect(markup).toContain('accepted_changes_apply_result.md')
    expect(markup).toContain('iteration_log.md')
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
`
    })

    expect(markup).toContain('proposal-1')
    expect(markup).toContain('已延后')
    expect(markup).not.toContain('bg-amber-100 text-amber-800')
    expect(markup).not.toContain('>延后</button>')
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

  test('llm-review renders auxiliary review materials without memory fallthrough', () => {
    const markup = renderMainPanel('llm-review')

    expect(markup).toContain('pending_human_review')
    expect(markup).toContain('# LLM Review Report')
    expect(markup).not.toContain('accepted content marker')
  })

  test('validate renders quality deltas and apply result', () => {
    const markup = renderMainPanel('validate')

    expect(markup).toContain('验证结果')
    expect(markup).toContain('88 → 97')
    expect(markup).toContain('4 → 0')
    expect(markup).toContain('Applied: 2')
    expect(markup).not.toContain('iteration_log.md')
  })
})
