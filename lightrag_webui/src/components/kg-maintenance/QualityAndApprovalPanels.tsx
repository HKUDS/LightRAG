import Button from '@/components/ui/Button'
import type {
  KBIterationDiffResponse,
  KBIterationQualityResponse,
  KBIterationRulesResponse,
  KBIterationSummaryResponse
} from '@/api/lightrag'
import type { KBIterationProposalDecision } from '@/api/lightrag'
import {
  canSubmitProposalDecision,
  countApprovalRequired,
  MEDICAL_REVIEW_CONFIRMATION,
  parseProposalDecisionStates,
  parseProposalSummaries,
  proposalNeedsConfirmation,
  type ProposalDecisionReview,
  type ProposalSummary
} from './kgMaintenanceData'
import { ClipboardCheckIcon, ExternalLinkIcon } from 'lucide-react'
import { useMemo, useState } from 'react'

interface QualityPanelProps {
  quality: KBIterationQualityResponse | null
}

interface ApprovalPanelProps {
  approvalQueue: string
  improvementBacklog: string
  acceptedChanges?: string
  rejectedChanges?: string
  onOpenEvidence?: (evidenceId: string) => void
  onDecision?: (
    proposal: ProposalSummary,
    decision: KBIterationProposalDecision,
    review: ProposalDecisionReview
  ) => void | Promise<void>
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

const RECORDED_DECISION_CLASSES: Record<KBIterationProposalDecision, string> = {
  accept:
    'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-200',
  reject: 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-200',
  defer: 'bg-sky-100 text-sky-800 dark:bg-sky-950/60 dark:text-sky-200'
}

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
          <article key={`${finding.category}-${index}`} className="border-border/70 rounded-lg border p-3">
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
  improvementBacklog,
  acceptedChanges = '',
  rejectedChanges = '',
  onOpenEvidence,
  onDecision
}: ApprovalPanelProps) {
  const proposals = parseProposalSummaries(approvalQueue)
  const decisionStates = useMemo(
    () => parseProposalDecisionStates({ acceptedChanges, rejectedChanges }),
    [acceptedChanges, rejectedChanges]
  )
  const [reasons, setReasons] = useState<Record<string, string>>({})
  const [impactScopes, setImpactScopes] = useState<Record<string, string>>({})
  const [verifications, setVerifications] = useState<Record<string, string>>({})
  const [confirmations, setConfirmations] = useState<Record<string, string>>({})

  return (
    <section className="space-y-4">
      <PanelHeader
        title="待审批 proposal"
        subtitle={`${countApprovalRequired(proposals)} 个需要人工审批`}
      />
      <div className="space-y-2">
        {proposals.length === 0 ? (
          <div className="border-border/70 bg-muted/20 rounded-lg border p-4 text-sm">
            暂无待审批 proposal。
          </div>
        ) : (
          proposals.map((proposal) => {
            const recordedDecision = decisionStates[proposal.id]
            return (
              <article key={proposal.id} className="border-border/70 rounded-lg border p-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold">{proposal.id}</div>
                    <div className="text-muted-foreground mt-1 text-xs">
                      {proposal.type || '未知类型'} / {proposal.target || '未指定目标'}
                    </div>
                  </div>
                  <div className="flex flex-wrap justify-end gap-2">
                    {recordedDecision && (
                      <span
                        className={`rounded-md px-2 py-1 text-xs font-medium ${RECORDED_DECISION_CLASSES[recordedDecision]}`}
                      >
                        {RECORDED_DECISION_LABELS[recordedDecision]}
                      </span>
                    )}
                    <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                      {proposal.risk || '风险未知'}
                    </span>
                  </div>
                </div>
                <div className="mt-3 grid gap-2 text-sm">
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
                <div className="mt-3 flex flex-wrap gap-2">
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
                ) : (
                  <div className="mt-3 grid gap-2">
                    <textarea
                      value={reasons[proposal.id] || ''}
                      onChange={(event) =>
                        setReasons((draft) => ({ ...draft, [proposal.id]: event.target.value }))
                      }
                      className="border-input bg-background min-h-20 rounded-md border px-3 py-2 text-sm"
                      placeholder="审批理由"
                    />
                    <textarea
                      value={impactScopes[proposal.id] || ''}
                      onChange={(event) =>
                        setImpactScopes((draft) => ({
                          ...draft,
                          [proposal.id]: event.target.value
                        }))
                      }
                      className="border-input bg-background min-h-16 rounded-md border px-3 py-2 text-sm"
                      placeholder="影响范围"
                    />
                    <textarea
                      value={verifications[proposal.id] || ''}
                      onChange={(event) =>
                        setVerifications((draft) => ({
                          ...draft,
                          [proposal.id]: event.target.value
                        }))
                      }
                      className="border-input bg-background min-h-16 rounded-md border px-3 py-2 text-sm"
                      placeholder="验证 / 回滚说明"
                    />
                    {proposalNeedsConfirmation(proposal) && (
                      <div className="border-amber-300 bg-amber-50 text-amber-950 rounded-md border p-3 text-sm dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-100">
                        <p>{MEDICAL_REVIEW_CONFIRMATION}</p>
                        <input
                          value={confirmations[proposal.id] || ''}
                          onChange={(event) =>
                            setConfirmations((draft) => ({
                              ...draft,
                              [proposal.id]: event.target.value
                            }))
                          }
                          className="border-input bg-background mt-2 h-9 w-full rounded-md border px-3 text-sm"
                          placeholder={MEDICAL_REVIEW_CONFIRMATION}
                        />
                      </div>
                    )}
                    <div className="flex flex-wrap gap-2">
                      {(['accept', 'reject', 'defer'] as KBIterationProposalDecision[]).map(
                        (decision) => (
                          <DecisionButton
                            key={decision}
                            proposal={proposal}
                            decision={decision}
                            reason={reasons[proposal.id] || ''}
                            impactScope={impactScopes[proposal.id] || ''}
                            verification={verifications[proposal.id] || ''}
                            confirmation={confirmations[proposal.id] || ''}
                            onDecision={onDecision}
                          />
                        )
                      )}
                    </div>
                  </div>
                )}
              </article>
            )
          })
        )}
      </div>
      <MarkdownArtifact title="approval_queue.md" content={approvalQueue} />
      <MarkdownArtifact title="improvement_backlog.md" content={improvementBacklog} />
    </section>
  )
}

function RecordedDecisionNotice({ decision }: { decision: KBIterationProposalDecision }) {
  return (
    <div className="border-border/70 bg-muted/20 mt-3 rounded-md border px-3 py-2 text-sm">
      Proposal {RECORDED_DECISION_LABELS[decision]}，不会重复提交审批决定。
    </div>
  )
}

function DecisionButton({
  proposal,
  decision,
  reason,
  impactScope,
  verification,
  confirmation,
  onDecision
}: {
  proposal: ProposalSummary
  decision: KBIterationProposalDecision
  reason: string
  impactScope: string
  verification: string
  confirmation: string
  onDecision?: (
    proposal: ProposalSummary,
    decision: KBIterationProposalDecision,
    review: ProposalDecisionReview
  ) => void | Promise<void>
}) {
  const review = { reason, impactScope, verification, confirmation }
  return (
    <Button
      variant={decision === 'accept' ? 'default' : 'outline'}
      size="sm"
      disabled={!onDecision || !canSubmitProposalDecision(proposal, review)}
      onClick={() => void onDecision?.(proposal, decision, review)}
    >
      {DECISION_LABELS[decision]}
    </Button>
  )
}

function ProposalField({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-muted/30 p-2">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 whitespace-pre-wrap break-words text-sm">{value || '未说明'}</div>
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
      <pre className="text-muted-foreground mt-3 max-h-80 overflow-auto whitespace-pre-wrap text-xs">
        {content || '暂无产物内容'}
      </pre>
    </details>
  )
}
