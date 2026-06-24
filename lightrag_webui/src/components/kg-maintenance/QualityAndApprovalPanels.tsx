import Button from '@/components/ui/Button'
import type {
  KBIterationDiffResponse,
  KBIterationQualityResponse,
  KBIterationRulesResponse,
  KBIterationSummaryResponse
} from '@/api/lightrag'
import type { KBIterationProposalDecision } from '@/api/lightrag'
import {
  parseProposalDecisionStates,
  parseProposalSummaries,
  type ProposalDecisionReview,
  type ProposalSummary
} from './kgMaintenanceData'
import {
  acceptAllPendingProposalsFromPanel,
  pendingApprovalProposals,
  requestProposalRevisionFromPanel
} from './proposalApprovalActions'
import {
  CheckCheckIcon,
  ClipboardCheckIcon,
  ExternalLinkIcon,
  MaximizeIcon,
  MinimizeIcon
} from 'lucide-react'
import { useMemo, useState } from 'react'

interface QualityPanelProps {
  quality: KBIterationQualityResponse | null
}

interface ApprovalPanelProps {
  approvalQueue: string
  approvalQueueSource: string
  improvementBacklog: string
  acceptedChanges?: string
  rejectedChanges?: string
  deferredChanges?: string
  deferredChangesSource?: string
  onOpenEvidence?: (evidenceId: string) => void
  onDecision?: (
    proposal: ProposalSummary,
    decision: KBIterationProposalDecision,
    review: ProposalDecisionReview
  ) => void | Promise<void>
  onAcceptAll?: (proposals: ProposalSummary[]) => void | Promise<void>
  onRequestRevision?: (proposal: ProposalSummary) => void | Promise<void>
}

interface DiffPanelProps {
  diff: KBIterationDiffResponse | null
}

interface RuleMemoryPanelProps {
  rules: KBIterationRulesResponse | null
}

interface RunLogPanelProps {
  runsText: string
  summary: KBIterationSummaryResponse | null
}

const DECISION_LABELS: Record<KBIterationProposalDecision, string> = {
  accept: '接受',
  reject: '拒绝',
  defer: '延后'
}

const RECORDED_DECISION_LABELS: Record<KBIterationProposalDecision, string> = {
  accept: '已接受',
  reject: '已拒绝',
  defer: '已延后'
}

const PENDING_DECISION_LABEL = '待审批'

const RECORDED_DECISION_CLASSES: Record<KBIterationProposalDecision, string> = {
  accept: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-200',
  reject: 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-200',
  defer: 'bg-sky-100 text-sky-800 dark:bg-sky-950/60 dark:text-sky-200'
}

const PENDING_DECISION_CLASS =
  'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-200'

export function QualityPanel({ quality }: QualityPanelProps) {
  const [severityFilter, setSeverityFilter] = useState('all')
  const [query, setQuery] = useState('')
  const findings = useMemo(() => {
    const source = quality?.quality.findings || []
    return source.filter((finding) => {
      const matchesSeverity = severityFilter === 'all' || finding.severity === severityFilter
      const haystack =
        `${finding.severity} ${finding.category} ${finding.message} ${finding.suggested_fix_type}`.toLowerCase()
      return matchesSeverity && haystack.includes(query.trim().toLowerCase())
    })
  }, [quality, query, severityFilter])

  if (!quality) return <EmptyPanel title="质量报告" />

  return (
    <section className="space-y-4">
      <PanelHeader title="质量报告" subtitle={`总分 ${quality.quality.overall ?? 0}/100`} />
      <div className="flex flex-wrap gap-2">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          className="border-input bg-background h-9 min-w-56 rounded-md border px-3 text-sm"
          placeholder="搜索质量发现"
        />
        <select
          value={severityFilter}
          onChange={(event) => setSeverityFilter(event.target.value)}
          className="border-input bg-background h-9 rounded-md border px-3 text-sm"
        >
          <option value="all">全部严重级别</option>
          <option value="critical">严重</option>
          <option value="high">高</option>
          <option value="medium">中</option>
          <option value="low">低</option>
        </select>
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {Object.entries(quality.quality.subscores || {}).map(([name, score]) => (
          <div key={name} className="border-border/70 rounded-lg border p-3">
            <div className="text-muted-foreground text-xs">{name}</div>
            <div className="mt-1 text-xl font-semibold">{score}</div>
          </div>
        ))}
      </div>
      {quality.quality.critical_blockers?.length ? (
        <div className="border-destructive/40 bg-destructive/10 rounded-lg border p-3">
          <div className="text-sm font-semibold">阻塞问题</div>
          <ul className="mt-2 space-y-1 text-sm">
            {quality.quality.critical_blockers.map((blocker) => (
              <li key={blocker}>{blocker}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <div className="space-y-2">
        {findings.map((finding, index) => (
          <article
            key={`${finding.category}-${index}`}
            className="border-border/70 rounded-lg border p-3"
          >
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <div className="text-sm font-medium">{finding.message}</div>
                <div className="text-muted-foreground mt-1 text-xs">
                  {finding.category} / {finding.suggested_fix_type}
                </div>
              </div>
              <span className="rounded-md bg-amber-100 px-2 py-1 text-xs font-medium text-amber-800 dark:bg-amber-900/50 dark:text-amber-200">
                {finding.severity}
              </span>
            </div>
            <div className="text-muted-foreground mt-2 text-xs">
              证据: {finding.evidence.join(', ') || '缺失'}
            </div>
          </article>
        ))}
      </div>
      <MarkdownArtifact title="quality_report.md" content={quality.report} />
    </section>
  )
}

export function ApprovalPanel({
  approvalQueue,
  approvalQueueSource,
  improvementBacklog,
  acceptedChanges = '',
  rejectedChanges = '',
  deferredChanges = '',
  deferredChangesSource = '',
  onOpenEvidence,
  onDecision,
  onAcceptAll,
  onRequestRevision
}: ApprovalPanelProps) {
  const proposals = parseProposalSummaries(approvalQueueSource)
  const decisionStates = useMemo(
    () =>
      parseProposalDecisionStates({
        acceptedChanges,
        rejectedChanges,
        deferredChanges: deferredChangesSource
      }),
    [acceptedChanges, deferredChangesSource, rejectedChanges]
  )
  const pendingProposals = pendingApprovalProposals(proposals, decisionStates)
  const pendingApprovalCount = pendingProposals.length
  const [collapsedProposals, setCollapsedProposals] = useState<Record<string, boolean>>({})
  const [acceptingAll, setAcceptingAll] = useState(false)
  const acceptAllDisabled = pendingApprovalCount === 0 || !onAcceptAll || acceptingAll
  const handleAcceptAll = async () => {
    if (acceptAllDisabled) return
    setAcceptingAll(true)
    try {
      await acceptAllPendingProposalsFromPanel(proposals, decisionStates, onAcceptAll)
    } finally {
      setAcceptingAll(false)
    }
  }

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <PanelHeader title="待审批 proposal" subtitle={`${pendingApprovalCount} 个需要人工审批`} />
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={acceptAllDisabled}
          tooltip={
            pendingApprovalCount > 0
              ? `接受全部 ${pendingApprovalCount} 条待审批 proposal`
              : '没有可接受的待审批 proposal'
          }
          onClick={() => void handleAcceptAll()}
        >
          <CheckCheckIcon className="size-4" />
          {acceptingAll ? '接受中' : '全部接受'}
          <span className="bg-muted text-muted-foreground rounded px-1.5 py-0.5 text-xs">
            {pendingApprovalCount}
          </span>
        </Button>
      </div>
      <div className="space-y-2">
        {proposals.length === 0 ? (
          <div className="border-border/70 bg-muted/20 rounded-lg border p-4 text-sm">
            暂无待审批 proposal。
          </div>
        ) : (
          proposals.map((proposal) => {
            const recordedDecision = decisionStates[proposal.id]
            const collapsed = Boolean(collapsedProposals[proposal.id])
            const detailsId = proposalDetailsId(proposal.id)
            const ToggleIcon = collapsed ? MaximizeIcon : MinimizeIcon
            const decisionDisabled = !onDecision || Boolean(recordedDecision)
            return (
              <article
                key={proposal.id}
                className="border-border/70 rounded-lg border p-3 transition-colors"
              >
                <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`rounded-md px-2 py-1 text-xs font-medium ${
                          recordedDecision
                            ? RECORDED_DECISION_CLASSES[recordedDecision]
                            : PENDING_DECISION_CLASS
                        }`}
                      >
                        {recordedDecision
                          ? RECORDED_DECISION_LABELS[recordedDecision]
                          : PENDING_DECISION_LABEL}
                      </span>
                      <div className="min-w-0 text-sm font-semibold">
                        <span className="mr-2">{proposal.id}</span>
                        <span className="text-muted-foreground font-normal">
                          {proposal.proposedChange || '未说明'}
                        </span>
                      </div>
                    </div>
                    <div className="text-muted-foreground mt-1 truncate text-xs">
                      {proposal.type || '未知类型'} / {proposal.target || '未指定目标'}
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 lg:justify-end">
                    <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                      {proposal.risk || '风险未知'}
                    </span>
                    <Button
                      type="button"
                      size="sm"
                      disabled={decisionDisabled}
                      onClick={() => void onDecision?.(proposal, 'accept', emptyProposalReview())}
                    >
                      {DECISION_LABELS.accept}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={decisionDisabled}
                      onClick={() => void onDecision?.(proposal, 'reject', emptyProposalReview())}
                    >
                      {DECISION_LABELS.reject}
                    </Button>
                    {recordedDecision === 'reject' && (
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        disabled={!onRequestRevision}
                        onClick={() =>
                          void requestProposalRevisionFromPanel(proposal, onRequestRevision)
                        }
                      >
                        让 Agent 修改
                      </Button>
                    )}
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      aria-label={`展开/收起 ${proposal.id}`}
                      aria-controls={detailsId}
                      aria-expanded={!collapsed}
                      tooltip={collapsed ? '展开审批详情' : '收起审批详情'}
                      onClick={() =>
                        setCollapsedProposals((draft) => ({
                          ...draft,
                          [proposal.id]: !draft[proposal.id]
                        }))
                      }
                    >
                      <ToggleIcon className="size-4" />
                      {collapsed ? '展开' : '收起'}
                    </Button>
                  </div>
                </div>
                {!collapsed && (
                  <div id={detailsId} className="mt-3 space-y-3">
                    <div className="grid gap-2 text-sm">
                      <ProposalField label="建议变更" value={proposal.proposedChange} />
                      <ProposalField label="原因" value={proposal.reason} />
                      <div className="grid gap-2 sm:grid-cols-2">
                        <ProposalField label="置信度" value={proposal.confidence || '未说明'} />
                        <ProposalField
                          label="预期指标变化"
                          value={proposal.expectedMetricChange || '未说明'}
                        />
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {proposal.evidence.map((evidence) => (
                        <Button
                          key={evidence}
                          variant="outline"
                          size="sm"
                          onClick={() => onOpenEvidence?.(evidence)}
                        >
                          <ExternalLinkIcon className="size-4" />
                          {evidence}
                        </Button>
                      ))}
                    </div>
                    {recordedDecision ? (
                      <RecordedDecisionNotice decision={recordedDecision} />
                    ) : null}
                  </div>
                )}
              </article>
            )
          })
        )}
      </div>
      <MarkdownArtifact title="approval_queue.md" content={approvalQueue} />
      <MarkdownArtifact title="improvement_backlog.md" content={improvementBacklog} />
      <MarkdownArtifact title="deferred_changes.md" content={deferredChanges} />
    </section>
  )
}

function RecordedDecisionNotice({ decision }: { decision: KBIterationProposalDecision }) {
  return (
    <div className="border-border/70 bg-muted/20 rounded-md border px-3 py-2 text-sm">
      Proposal {RECORDED_DECISION_LABELS[decision]}，不会重复提交审批决定。
    </div>
  )
}

function proposalDetailsId(proposalId: string) {
  return `proposal-details-${proposalId.replace(/[^a-zA-Z0-9_-]/g, '-')}`
}

function emptyProposalReview(): ProposalDecisionReview {
  return {
    reason: '',
    impactScope: '',
    verification: '',
    confirmation: ''
  }
}

function ProposalField({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-muted/30 rounded-md p-2">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 text-sm break-words whitespace-pre-wrap">{value || '未说明'}</div>
    </div>
  )
}

export function DiffPanel({ diff }: DiffPanelProps) {
  if (!diff) return <EmptyPanel title="Diff 审阅" />
  const flags = diff.summary?.dangerous_regression_flags || []
  return (
    <section className="space-y-4">
      <PanelHeader title="Diff 审阅" subtitle="快照回归对比" />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <DiffMetric label="新增节点" value={diff.summary?.added_nodes?.length || 0} />
        <DiffMetric label="移除节点" value={diff.summary?.removed_nodes?.length || 0} />
        <DiffMetric label="新增关系" value={diff.summary?.added_edge_pairs?.length || 0} />
        <DiffMetric label="移除关系" value={diff.summary?.removed_edge_pairs?.length || 0} />
      </div>
      {flags.length ? (
        <div className="border-destructive/40 bg-destructive/10 rounded-lg border p-3 text-sm">
          {flags.join(', ')}
        </div>
      ) : null}
      <MarkdownArtifact title="diff_report.md" content={diff.report} />
    </section>
  )
}

export function RuleMemoryPanel({ rules }: RuleMemoryPanelProps) {
  if (!rules) return <EmptyPanel title="规则记忆" />
  return (
    <section className="space-y-4">
      <PanelHeader title="规则记忆" subtitle="长期审阅规则与决策" />
      <MarkdownArtifact title="质量规则" content={rules.qualityRules} />
      <MarkdownArtifact title="已知问题" content={rules.knownIssues} />
      <MarkdownArtifact title="已接受变更" content={rules.acceptedChanges} />
      <MarkdownArtifact title="已拒绝变更" content={rules.rejectedChanges} />
    </section>
  )
}

export function RunLogPanel({ runsText, summary }: RunLogPanelProps) {
  return (
    <section className="space-y-4">
      <PanelHeader
        title="运行日志"
        subtitle={summary?.phase === 'pending_user_review' ? '审阅包已生成' : '运行历史'}
      />
      <div className="border-border/70 rounded-lg border p-3 text-sm">
        <ClipboardCheckIcon className="mr-2 inline size-4 text-emerald-500" />
        {summary?.phase || '暂无运行阶段'}
      </div>
      <MarkdownArtifact title="iteration_log.md" content={runsText} />
    </section>
  )
}

function DiffMetric({ label, value }: { label: string; value: number }) {
  return (
    <div className="border-border/70 rounded-lg border p-3">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 text-xl font-semibold">{value}</div>
    </div>
  )
}

function PanelHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div>
      <h2 className="text-sm font-semibold">{title}</h2>
      <p className="text-muted-foreground mt-1 text-sm">{subtitle}</p>
    </div>
  )
}

function EmptyPanel({ title }: { title: string }) {
  return (
    <section className="border-border/70 bg-muted/20 rounded-lg border p-6">
      <h2 className="text-sm font-semibold">{title}</h2>
      <p className="text-muted-foreground mt-2 text-sm">请先运行知识库审阅包。</p>
    </section>
  )
}

function MarkdownArtifact({ title, content }: { title: string; content: string }) {
  return (
    <details className="border-border/70 rounded-lg border p-3" open={false}>
      <summary className="cursor-pointer text-sm font-medium">{title}</summary>
      <pre className="text-muted-foreground mt-3 max-h-80 overflow-auto text-xs whitespace-pre-wrap">
        {content || '暂无产物内容'}
      </pre>
    </details>
  )
}
