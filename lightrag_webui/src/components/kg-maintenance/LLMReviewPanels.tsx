import Button from '@/components/ui/Button'
import { BrainCircuitIcon, FileDiffIcon, ShieldCheckIcon } from 'lucide-react'
import type { ReactNode } from 'react'

export type TraceRound = {
  round_id?: string
  focus?: string[]
  proposal_ids?: string[]
  state?: string
}

export type LLMReviewPanelProps = {
  trace: Record<string, any> | null
  report: string
  proposals: string
  running: boolean
  onRun: () => void
}

type PatchCandidatesPanelProps = {
  proposals: string
  patchText: string
  onLoadPatch: (proposalId: string) => void
}

type LLMJudgePanelProps = {
  report: string
}

export function LLMReviewPanel({
  trace,
  report,
  proposals,
  running,
  onRun
}: LLMReviewPanelProps) {
  const stopReason = typeof trace?.stop_reason === 'string' ? trace.stop_reason : ''
  const rounds = Array.isArray(trace?.rounds) ? (trace.rounds as TraceRound[]) : []

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <PanelHeader
          icon={<BrainCircuitIcon className="size-4 text-sky-500" />}
          title="LLM 审阅"
          subtitle={stopReason ? `停止原因：${stopReason}` : '尚未生成 LLM 审阅 trace'}
        />
        <Button variant="outline" size="sm" disabled={running} onClick={onRun}>
          <BrainCircuitIcon className="size-4" />
          {running ? '运行中' : '运行审阅'}
        </Button>
      </div>

      <div className="border-border/70 bg-muted/20 rounded-lg border p-3 text-sm">
        <span className="text-muted-foreground mr-2">Trace stop reason</span>
        <span className="font-medium">{stopReason || '暂无 stop_reason'}</span>
      </div>

      <div className="space-y-2">
        <h3 className="text-sm font-medium">审阅轮次</h3>
        {rounds.length ? (
          rounds.map((round, index) => (
            <article
              key={round.round_id || `round-${index}`}
              className="border-border/70 rounded-lg border p-3 text-sm"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">{round.round_id || `round-${index + 1}`}</span>
                {round.state ? (
                  <span className="bg-muted rounded-md px-2 py-1 text-xs">{round.state}</span>
                ) : null}
              </div>
              <RoundField label="Focus" values={round.focus} emptyText="暂无 focus 项" />
              <RoundField
                label="Proposal IDs"
                values={round.proposal_ids}
                emptyText="暂无 proposal id"
              />
            </article>
          ))
        ) : (
          <EmptyBlock>暂无审阅轮次。</EmptyBlock>
        )}
      </div>

      <ArtifactBlock title="LLM Review Report" content={report} emptyText="暂无审阅报告。" />
      <ArtifactBlock title="Generated Proposals" content={proposals} emptyText="暂无候选 proposal。" />
    </section>
  )
}

export function PatchCandidatesPanel({
  proposals,
  patchText,
  onLoadPatch
}: PatchCandidatesPanelProps) {
  const proposalIds = parseProposalIds(proposals)

  return (
    <section className="space-y-4">
      <PanelHeader
        icon={<FileDiffIcon className="size-4 text-emerald-500" />}
        title="候选 Patch"
        subtitle="从候选 proposal 加载并检查 patch"
      />

      <div className="space-y-2">
        <h3 className="text-sm font-medium">Proposal IDs</h3>
        {proposalIds.length ? (
          <div className="flex flex-wrap gap-2">
            {proposalIds.map((proposalId) => (
              <Button
                key={proposalId}
                variant="outline"
                size="sm"
                onClick={() => onLoadPatch(proposalId)}
              >
                {proposalId}
              </Button>
            ))}
          </div>
        ) : (
          <EmptyBlock>暂无可加载的 proposal id。</EmptyBlock>
        )}
      </div>

      <ArtifactBlock title="Selected Patch" content={patchText} emptyText="尚未选择候选 patch。" />
      <ArtifactBlock title="Proposal Source" content={proposals} emptyText="暂无候选 proposal。" />
    </section>
  )
}

export function LLMJudgePanel({ report }: LLMJudgePanelProps) {
  return (
    <section className="space-y-4">
      <PanelHeader
        icon={<ShieldCheckIcon className="size-4 text-indigo-500" />}
        title="Judge 评判"
        subtitle="LLM judge 报告与人工复核状态"
      />
      <ArtifactBlock title="Judge Report" content={report} emptyText="暂无 judge 评判报告。" />
    </section>
  )
}

function PanelHeader({
  icon,
  title,
  subtitle
}: {
  icon: ReactNode
  title: string
  subtitle: string
}) {
  return (
    <div>
      <div className="flex items-center gap-2">
        {icon}
        <h2 className="text-sm font-semibold">{title}</h2>
      </div>
      <p className="text-muted-foreground mt-1 text-sm">{subtitle}</p>
    </div>
  )
}

function RoundField({
  label,
  values,
  emptyText
}: {
  label: string
  values?: string[]
  emptyText: string
}) {
  const hasValues = Array.isArray(values) && values.length > 0
  return (
    <div className="mt-3">
      <div className="text-muted-foreground text-xs">{label}</div>
      <div className="mt-1 flex flex-wrap gap-2">
        {hasValues ? (
          values.map((value) => (
            <span key={value} className="bg-muted rounded-md px-2 py-1 text-xs">
              {value}
            </span>
          ))
        ) : (
          <span className="text-muted-foreground text-sm">{emptyText}</span>
        )}
      </div>
    </div>
  )
}

function ArtifactBlock({
  title,
  content,
  emptyText
}: {
  title: string
  content: string
  emptyText: string
}) {
  const hasContent = content.trim().length > 0
  return (
    <div className="border-border/70 rounded-lg border p-3">
      <h3 className="text-sm font-medium">{title}</h3>
      <pre className="text-muted-foreground mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words text-xs">
        {hasContent ? content : emptyText}
      </pre>
    </div>
  )
}

function EmptyBlock({ children }: { children: ReactNode }) {
  return (
    <div className="border-border/70 bg-muted/20 text-muted-foreground rounded-lg border p-3 text-sm">
      {children}
    </div>
  )
}

function parseProposalIds(proposals: string) {
  const ids = new Set<string>()
  const idPattern = /^\s*-?\s*id:\s*['"]?([^'"\s#]+)['"]?/gm
  let match: RegExpExecArray | null

  while ((match = idPattern.exec(proposals)) !== null) {
    ids.add(match[1])
  }

  return Array.from(ids)
}
