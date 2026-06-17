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

export function QualityPanel({ quality }: QualityPanelProps) {
  const [severityFilter, setSeverityFilter] = useState('all')
  const [query, setQuery] = useState('')
  const findings = useMemo(() => {
    const source = quality?.quality.findings || []
    return source.filter((finding) => {
      const matchesSeverity =
        severityFilter === 'all' || finding.severity === severityFilter
      const haystack =
        `${finding.severity} ${finding.category} ${finding.message} ${finding.suggested_fix_type}`.toLowerCase()
      return matchesSeverity && haystack.includes(query.trim().toLowerCase())
    })
  }, [quality, query, severityFilter])

  if (!quality) return <EmptyPanel title="Quality Report" />

  return (
    <section className="space-y-4">
      <PanelHeader
        title="Quality Report"
        subtitle={`Overall score ${quality.quality.overall ?? 0}/100`}
      />
      <div className="flex flex-wrap gap-2">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          className="border-input bg-background h-9 min-w-56 rounded-md border px-3 text-sm"
          placeholder="Search findings"
        />
        <select
          value={severityFilter}
          onChange={(event) => setSeverityFilter(event.target.value)}
          className="border-input bg-background h-9 rounded-md border px-3 text-sm"
        >
          <option value="all">All severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
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
          <div className="text-sm font-semibold">Critical Blockers</div>
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
              Evidence: {finding.evidence.join(', ') || 'Missing'}
            </div>
          </article>
        ))}
      </div>
      <MarkdownArtifact title="Quality Markdown" content={quality.report} />
    </section>
  )
}

export function ApprovalPanel({
  approvalQueue,
  improvementBacklog,
  onOpenEvidence,
  onDecision
}: ApprovalPanelProps) {
  const proposals = parseProposalSummaries(approvalQueue)
  const [reasons, setReasons] = useState<Record<string, string>>({})
  const [impactScopes, setImpactScopes] = useState<Record<string, string>>({})
  const [verifications, setVerifications] = useState<Record<string, string>>({})
  const [confirmations, setConfirmations] = useState<Record<string, string>>({})
  return (
    <section className="space-y-4">
      <PanelHeader
        title="Approval Queue"
        subtitle={`${countApprovalRequired(proposals)} approval-gated proposals`}
      />
      <div className="space-y-2">
        {proposals.length === 0 ? (
          <div className="border-border/70 bg-muted/20 rounded-lg border p-4 text-sm">
            No proposal is waiting for review.
          </div>
        ) : (
          proposals.map((proposal) => (
            <article key={proposal.id} className="border-border/70 rounded-lg border p-3">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold">{proposal.id}</div>
                  <div className="text-muted-foreground mt-1 text-xs">
                    {proposal.type || 'unknown'} / {proposal.target || 'No target'}
                  </div>
                </div>
                <span className="rounded-md bg-slate-100 px-2 py-1 text-xs font-medium text-slate-700 dark:bg-slate-800 dark:text-slate-200">
                  {proposal.risk || 'risk unknown'}
                </span>
              </div>
              <div className="mt-3 grid gap-2 text-sm">
                <ProposalField label="Proposed change" value={proposal.proposedChange} />
                <ProposalField label="Reason" value={proposal.reason} />
                <div className="grid gap-2 sm:grid-cols-2">
                  <ProposalField label="Confidence" value={proposal.confidence || 'Not stated'} />
                  <ProposalField
                    label="Expected metric change"
                    value={proposal.expectedMetricChange || 'Not stated'}
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
              <div className="mt-3 grid gap-2">
                <textarea
                  value={reasons[proposal.id] || ''}
                  onChange={(event) =>
                    setReasons((draft) => ({ ...draft, [proposal.id]: event.target.value }))
                  }
                  className="border-input bg-background min-h-20 rounded-md border px-3 py-2 text-sm"
                  placeholder="Review reason"
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
                  placeholder="Impact scope"
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
                  placeholder="Verification / rollback notes"
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
                      placeholder={`Type: ${MEDICAL_REVIEW_CONFIRMATION}`}
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
            </article>
          ))
        )}
      </div>
      <MarkdownArtifact title="Approval Queue Markdown" content={approvalQueue} />
      <MarkdownArtifact title="Improvement Backlog Markdown" content={improvementBacklog} />
    </section>
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
      {decision}
    </Button>
  )
}

function ProposalField({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-muted/30 p-2">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 whitespace-pre-wrap break-words text-sm">{value || 'Not stated'}</div>
    </div>
  )
}

export function DiffPanel({ diff }: DiffPanelProps) {
  if (!diff) return <EmptyPanel title="Diff Review" />
  const flags = diff.summary?.dangerous_regression_flags || []
  return (
    <section className="space-y-4">
      <PanelHeader title="Diff Review" subtitle="Snapshot regression comparison" />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <DiffMetric label="Added Nodes" value={diff.summary?.added_nodes?.length || 0} />
        <DiffMetric label="Removed Nodes" value={diff.summary?.removed_nodes?.length || 0} />
        <DiffMetric label="Added Relations" value={diff.summary?.added_edge_pairs?.length || 0} />
        <DiffMetric label="Removed Relations" value={diff.summary?.removed_edge_pairs?.length || 0} />
      </div>
      {flags.length ? (
        <div className="border-destructive/40 bg-destructive/10 rounded-lg border p-3 text-sm">
          {flags.join(', ')}
        </div>
      ) : null}
      <MarkdownArtifact title="Diff Markdown" content={diff.report} />
    </section>
  )
}

export function RuleMemoryPanel({ rules }: RuleMemoryPanelProps) {
  if (!rules) return <EmptyPanel title="Rule Memory" />
  return (
    <section className="space-y-4">
      <PanelHeader title="Rule Memory" subtitle="Long-lived review rules and decisions" />
      <MarkdownArtifact title="Quality Rules" content={rules.qualityRules} />
      <MarkdownArtifact title="Known Issues" content={rules.knownIssues} />
      <MarkdownArtifact title="Accepted Changes" content={rules.acceptedChanges} />
      <MarkdownArtifact title="Rejected Changes" content={rules.rejectedChanges} />
    </section>
  )
}

export function RunLogPanel({ runsText, summary }: RunLogPanelProps) {
  return (
    <section className="space-y-4">
      <PanelHeader
        title="Run Log"
        subtitle={summary?.phase === 'pending_user_review' ? 'Review package generated' : 'Run history'}
      />
      <div className="border-border/70 rounded-lg border p-3 text-sm">
        <ClipboardCheckIcon className="mr-2 inline size-4 text-emerald-500" />
        {summary?.phase || 'No run phase'}
      </div>
      <MarkdownArtifact title="Iteration Log Markdown" content={runsText} />
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
      <p className="text-muted-foreground mt-2 text-sm">Run KB iteration review first.</p>
    </section>
  )
}

function MarkdownArtifact({ title, content }: { title: string; content: string }) {
  return (
    <details className="border-border/70 rounded-lg border p-3" open={false}>
      <summary className="cursor-pointer text-sm font-medium">{title}</summary>
      <pre className="text-muted-foreground mt-3 max-h-80 overflow-auto whitespace-pre-wrap text-xs">
        {content || 'No artifact content'}
      </pre>
    </details>
  )
}
